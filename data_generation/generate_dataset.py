"""Generador del dataset sintetico de retail de gran consumo.

Produce las 7 tablas descritas en DATA_SPEC.md en formato CSV. Todo el dato es
sintetico: no procede de ningun cliente, producto ni retailer real.

Uso:
    python -m data_generation.generate_dataset                    # volumen completo
    python -m data_generation.generate_dataset --scale 0.02       # muestra rapida
    python -m data_generation.generate_dataset --out /tmp/prueba

Reproducibilidad
----------------
Con la misma semilla y la misma escala, dos ejecuciones producen ficheros identicos
byte a byte. Para conseguirlo:

* Un unico `SeedSequence` raiz se reparte en sub-streams independientes y estables por
  etapa (`_rng_streams`), de forma que anadir aleatoriedad a una etapa no desplaza la
  de las demas.
* Las cestas se recorren en orden cronologico estable (`argsort` con `kind="stable"`),
  porque el ciclo de reposicion depende de la compra anterior.
* Los CSV se escriben con `lineterminator="\\n"` y `float_format="%.2f"` fijos, para que
  el hash no dependa del sistema operativo ni de la representacion de los flotantes.

Ver `data_generation/verify_dataset.py` para el chequeo de patrones y volumenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from data_generation import catalog as cat

SEED = 42

# Reparto de la semilla raiz en sub-streams: el orden de esta tupla es parte del
# contrato de reproducibilidad, asi que solo se anade al final, nunca se reordena.
_STREAM_NAMES = (
    "products",
    "customers",
    "promotions",
    "basket_dates",
    "basket_items",
    "sessions",
    "quality",
)


@dataclass
class GeneratorConfig:
    """Volumenes y parametros de una ejecucion del generador.

    Los volumenes por defecto son los de referencia de DATA_SPEC.md. `scale` los
    multiplica en bloque para poder generar muestras pequenas en los tests sin tocar
    ninguna otra logica.
    """

    out_dir: Path = Path("data/raw")
    seed: int = SEED
    scale: float = 1.0

    n_customers: int = 20_000
    # Surtido curado: `products_per_category` referencias en cada una de las categorias
    # del catalogo (ver `catalog.PRODUCTS_PER_CATEGORY`). `n_products` es derivado y no
    # lo toca `scale`: el surtido es el mismo lineal en una muestra de prueba que a
    # volumen completo, lo que cambia es cuanta gente compra en el.
    products_per_category: int = cat.PRODUCTS_PER_CATEGORY
    n_products: int = field(init=False, default=0)
    n_promotions: int = 300
    # DATA_SPEC.md da 300.000 como volumen de referencia, pero con 20.000 clientes eso
    # son ~15 compras por cliente en dos anos (una visita cada ~73 dias): a esa cadencia
    # los ciclos de reposicion no son observables y `churn_label` marca al 45% de la base.
    # Se sube a 600.000 (~30 compras por cliente, una visita cada ~24 dias).
    n_baskets: int = 600_000
    n_sessions: int = 150_000
    n_stores: int = 60

    # Proporcion de cestas sin `customer_id` (compras anonimas legitimas).
    anonymous_basket_share: float = 0.03
    # Proporcion de sesiones online que acaban en cesta.
    session_conversion_rate: float = 0.35
    # Proporcion de clientes que abandonan dentro del periodo simulado.
    churn_share: float = 0.22

    # --- Embudo online (ver `_generate_sessions`) ---
    # Que parte del ticket pasa ademas por la navegacion online. Por debajo de 1 para que
    # "estar en la cesta" no implique "haber dejado rastro en la sesion".
    session_item_browse_rate: float = 0.85
    # Media de productos que se anaden al carrito y NO acaban en el ticket (abandono a
    # nivel de linea). Es lo que rompe la equivalencia add_to_cart == ticket.
    session_abandoned_adds: float = 0.60
    # Media de productos que se ven y nunca se anaden.
    session_view_only: float = 2.50
    # Segundos entre ver un producto y anadirlo al carrito. El retardo es lo que hace
    # utilizable la senal: en un corte temporal hay productos ya vistos y aun no anadidos.
    session_add_lag_s: tuple[int, int] = (20, 240)

    # --- Problemas de calidad deliberados (DATA_SPEC.md, "Calidad del dato") ---
    dup_item_share: float = 0.015
    negative_qty_share: float = 0.004
    messy_category_share: float = 0.04
    amount_outlier_share: float = 0.002
    null_brand_share: float = 0.010
    null_city_share: float = 0.008
    inconsistent_date_share: float = 0.003

    scaled: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        if self.scale <= 0:
            raise ValueError("scale debe ser > 0")
        # Con escalas pequenas se mantienen minimos para que el dataset siga siendo
        # representativo (al menos 1 tienda).
        self.n_customers = max(50, int(round(self.n_customers * self.scale)))
        if self.products_per_category < 2:
            raise ValueError("products_per_category debe ser >= 2")
        self.n_products = self.products_per_category * len(cat.CATEGORIES)
        self.n_promotions = max(10, int(round(self.n_promotions * self.scale)))
        self.n_baskets = max(200, int(round(self.n_baskets * self.scale)))
        self.n_sessions = max(100, int(round(self.n_sessions * self.scale)))
        self.n_stores = max(1, int(round(self.n_stores * self.scale)))


def _rng_streams(seed: int) -> dict[str, np.random.Generator]:
    """Crea un generador independiente por etapa a partir de una unica semilla."""
    children = np.random.SeedSequence(seed).spawn(len(_STREAM_NAMES))
    return {name: np.random.default_rng(ss) for name, ss in zip(_STREAM_NAMES, children)}


# --------------------------------------------------------------------------------------
# Calendario
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Calendar:
    """Eje temporal precomputado del periodo simulado."""

    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    month: np.ndarray  # mes (1-12) de cada dia
    dow: np.ndarray  # dia de la semana (0=lunes) de cada dia
    day_str: list[str]  # "YYYY-MM-DD" de cada dia
    tod_str: list[str]  # "HH:MM:SS" de cada segundo del dia


def _build_calendar() -> Calendar:
    start = pd.Timestamp(cat.PERIOD_START)
    end = pd.Timestamp(cat.PERIOD_END)
    days = pd.date_range(start, end, freq="D")
    tod = [f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}" for s in range(86_400)]
    return Calendar(
        start=start,
        end=end,
        n_days=len(days),
        month=days.month.to_numpy(),
        dow=days.dayofweek.to_numpy(),
        day_str=[d.strftime("%Y-%m-%d") for d in days],
        tod_str=tod,
    )


def _stamp(cal: Calendar, day: np.ndarray, second: np.ndarray) -> list[str]:
    """Formatea pares (indice de dia, segundo del dia) como datetime ISO."""
    ds, ts = cal.day_str, cal.tod_str
    return [f"{ds[d]} {ts[s]}" for d, s in zip(day.tolist(), second.tolist())]


# --------------------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------------------
def _generate_products(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Reparte los productos entre categorias y les asigna marca, precio y ciclo."""
    fake = Faker("es_ES")
    Faker.seed(cfg.seed)
    # Marcas de fabricante: Faker aporta la variedad, la semilla la reproducibilidad.
    brands: list[str] = []
    seen: set[str] = set()
    while len(brands) < 140:
        name = fake.company()
        if name not in seen:
            seen.add(name)
            brands.append(name)

    # Mismo numero de referencias en todas las categorias. Antes el reparto era
    # proporcional a sqrt(popularidad) sobre 1.500 productos (~24 por categoria) y la
    # eleccion dentro de la categoria salia practicamente al azar; con un surtido curado
    # e igual de granular en todas, la senal de SKU la pone la fidelidad de marca
    # (`Category.loyalty`) y no el tamano del surtido.
    counts = np.full(len(cat.CATEGORIES), cfg.products_per_category, dtype=int)

    rows = []
    for ci, (c, n) in enumerate(zip(cat.CATEGORIES, counts)):
        is_pl = rng.random(n) < 0.22
        prices = rng.uniform(c.price_lo, c.price_hi, n)
        # La marca blanca es sistematicamente mas barata que la de fabricante.
        prices = np.where(is_pl, prices * 0.78, prices)
        pack = rng.choice([1, 1, 1, 2, 4, 6, 12], size=n)
        brand_pick = rng.integers(0, len(brands), n)
        pl_pick = rng.integers(0, len(cat.PRIVATE_LABEL_BRANDS), n)
        # Popularidad relativa dentro de la categoria (cola larga tipica de retail).
        pop = rng.lognormal(0.0, 0.75, n)
        for k in range(n):
            rows.append(
                (
                    c.department,
                    c.name,
                    cat.PRIVATE_LABEL_BRANDS[pl_pick[k]] if is_pl[k] else brands[brand_pick[k]],
                    bool(is_pl[k]),
                    c.perishable,
                    round(float(max(prices[k], 0.30)), 2),
                    int(pack[k]),
                    c.repurchase_days,
                    ci,
                    float(pop[k]),
                )
            )

    products = pd.DataFrame(
        rows,
        columns=[
            "department",
            "category",
            "brand",
            "is_private_label",
            "is_perishable",
            "unit_price",
            "pack_size",
            "typical_repurchase_days",
            "category_idx",
            "popularity",
        ],
    )
    products.insert(0, "product_id", [f"P{i:05d}" for i in range(1, len(products) + 1)])
    return products


# --------------------------------------------------------------------------------------
# customers
# --------------------------------------------------------------------------------------
def _generate_customers(
    cfg: GeneratorConfig, cal: Calendar, rng: np.random.Generator
) -> pd.DataFrame:
    """Genera la base de clientes con sus atributos de hogar, canal y fidelidad."""
    n = cfg.n_customers

    household = rng.choice(
        [1, 2, 3, 4, 5, 6], size=n, p=[0.22, 0.30, 0.22, 0.16, 0.07, 0.03]
    )
    tier = rng.choice(cat.LOYALTY_TIERS, size=n, p=cat.LOYALTY_TIER_PROBS)
    channel = rng.choice(cat.CHANNELS, size=n, p=cat.CHANNEL_PROBS)

    city_names = [c[0] for c in cat.CITIES]
    city_w = np.array([c[1] for c in cat.CITIES], dtype=float)
    city = rng.choice(city_names, size=n, p=city_w / city_w.sum())

    # El 60% ya era cliente antes del periodo simulado; el 40% se da de alta dentro.
    legacy = rng.random(n) < 0.60
    signup_day = np.where(
        legacy,
        rng.integers(-1095, 0, n),  # hasta 3 anos antes del inicio del periodo
        rng.integers(0, cal.n_days - 30, n),
    )
    signup_date = (cal.start + pd.to_timedelta(signup_day, unit="D")).strftime("%Y-%m-%d")

    # Gates de hogar: quien tiene bebe, mascota o consume alcohol. Los hogares grandes
    # tienen mas probabilidad de tener bebe.
    baby_p = np.clip(cat.GATE_PREVALENCE["baby"] * (0.4 + 0.25 * household), 0.02, 0.55)
    has_baby = rng.random(n) < baby_p
    has_pet = rng.random(n) < cat.GATE_PREVALENCE["pet"]
    buys_alcohol = rng.random(n) < cat.GATE_PREVALENCE["alcohol"]

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
            "signup_date": signup_date,
            "country": cat.COUNTRY,
            "city": city,
            "household_size_est": household,
            "loyalty_tier": tier,
            "preferred_channel": channel,
            # churn_label se deriva al final, a partir de las cestas realmente generadas.
            "churn_label": False,
        }
    )
    customers["_signup_day"] = signup_day
    customers["_has_baby"] = has_baby
    customers["_has_pet"] = has_pet
    customers["_buys_alcohol"] = buys_alcohol
    return customers


# --------------------------------------------------------------------------------------
# promotions
# --------------------------------------------------------------------------------------
def _generate_promotions(
    cfg: GeneratorConfig, cal: Calendar, products: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Genera campanas promocionales sobre productos concretos y ventanas acotadas."""
    n = cfg.n_promotions
    # Se promociona mas lo que ya se vende: sesgo hacia productos populares.
    w = products["popularity"].to_numpy()
    prod_idx = rng.choice(len(products), size=n, p=w / w.sum())

    promo_type = rng.choice(cat.PROMO_TYPES, size=n, p=cat.PROMO_TYPE_PROBS)
    duration = rng.integers(7, 31, n)
    start_day = rng.integers(0, np.maximum(cal.n_days - duration, 1))
    end_day = np.minimum(start_day + duration - 1, cal.n_days - 1)

    discount = np.where(
        promo_type == "discount_pct",
        np.round(rng.uniform(0.10, 0.50, n), 2),
        np.where(
            promo_type == "2x1",
            0.50,
            rng.choice([1.0, 2.0, 3.0, 5.0], size=n),  # cupon: importe fijo en euros
        ),
    )

    promotions = pd.DataFrame(
        {
            "promotion_id": [f"PR{i:04d}" for i in range(1, n + 1)],
            "product_id": products["product_id"].to_numpy()[prod_idx],
            "promo_type": promo_type,
            "discount_value": np.round(discount, 2),
            "start_date": [cal.day_str[d] for d in start_day],
            "end_date": [cal.day_str[d] for d in end_day],
        }
    )
    promotions["_product_idx"] = prod_idx
    promotions["_start_day"] = start_day
    promotions["_end_day"] = end_day
    return promotions


# --------------------------------------------------------------------------------------
# baskets: cuando compra cada cliente
# --------------------------------------------------------------------------------------
def _generate_basket_dates(
    cfg: GeneratorConfig, cal: Calendar, customers: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decide en que dias compra cada cliente e inyecta la senal de churn progresiva.

    Cada cliente tiene una intensidad diaria propia (lognormal, muy sesgada: unos pocos
    hogares compran cada semana y la mayoria mucho menos). Sobre esa intensidad se
    aplica un perfil diario que combina dia de la semana, estacionalidad global y, para
    los clientes que abandonan, una rampa decreciente en las 6-8 semanas previas a su
    ultima compra: la frecuencia no se corta en seco, decae (DATA_SPEC.md, "Churn
    progresivo").

    Returns:
        Tupla `(customer_idx, day, spend_factor, last_active_day)`. Las tres primeras
        tienen una entrada por cesta identificada; `last_active_day` tiene una entrada
        por cliente (`-1` si el cliente no abandona).
    """
    n = len(customers)
    signup_day = customers["_signup_day"].to_numpy()

    # Perfil de dia de la semana: el fin de semana concentra la compra grande y el
    # domingo cae por el horario comercial espanol.
    dow_w = np.array([0.95, 0.90, 0.95, 1.05, 1.35, 1.45, 0.35])
    month_w = np.ones(13)
    month_w[12] = 1.25  # campana de Navidad
    month_w[8] = 0.90  # vacaciones de agosto
    base_day_w = dow_w[cal.dow] * month_w[cal.month]

    # Los clientes menos fidelizados abandonan mas.
    tier = customers["loyalty_tier"].to_numpy()
    churn_p = np.where(tier == "bronze", 1.35, np.where(tier == "silver", 0.85, 0.40))
    churn_p = np.clip(churn_p * cfg.churn_share, 0.0, 0.95)
    is_churner = rng.random(n) < churn_p

    # Ultima compra del que abandona: en cualquier punto del periodo, pero dejando
    # margen para que la rampa de decaimiento quepa dentro de la ventana observada.
    last_active = np.where(
        is_churner,
        rng.integers(90, cal.n_days - cat.CHURN_WINDOW_DAYS, n),
        cal.n_days - 1,
    )
    last_active = np.maximum(last_active, np.maximum(signup_day, 0) + 21)
    last_active = np.minimum(last_active, cal.n_days - 1)
    decay_weeks = rng.integers(6, 9, n)  # rampa de 6 a 8 semanas

    # Intensidad relativa por cliente. Se normaliza despues para cuadrar el volumen
    # total con `n_baskets`, asi que aqui solo importa la forma de la distribucion.
    # La sigma controla cuanto se parece la base a un Pareto: con valores altos la mitad
    # de los clientes queda practicamente dormida y `churn_label` (derivado a 60 dias sin
    # compra) se dispara, dejando un target de propension casi puro ruido.
    rate = rng.lognormal(0.0, 0.75, n)
    rate *= np.where(tier == "gold", 1.6, np.where(tier == "silver", 1.15, 0.85))
    rate *= 0.75 + 0.10 * customers["household_size_est"].to_numpy()

    start_day = np.maximum(signup_day, 0)

    # Peso diario por cliente y suma del perfil en su ventana activa. Se recorre cliente
    # a cliente porque la rampa de churn depende de su propia fecha de abandono.
    profiles: list[np.ndarray] = []
    totals = np.zeros(n)
    for i in range(n):
        lo, hi = int(start_day[i]), int(last_active[i])
        if hi < lo:
            profiles.append(np.empty(0))
            continue
        w = base_day_w[lo : hi + 1].copy()
        if is_churner[i]:
            ramp_days = int(decay_weeks[i]) * 7
            k = min(ramp_days, w.size)
            # De 1.0 a 0.15 de forma lineal en los ultimos `k` dias activos.
            w[w.size - k :] *= np.linspace(1.0, 0.15, k)
        profiles.append(w)
        totals[i] = w.sum()

    # Escalado global para aterrizar en el volumen objetivo de DATA_SPEC.md.
    n_identified = int(round(cfg.n_baskets * (1.0 - cfg.anonymous_basket_share)))
    expected = rate * totals
    if expected.sum() <= 0:
        raise RuntimeError("Ningun cliente tiene ventana activa; revisa el calendario")
    expected *= n_identified / expected.sum()
    n_per_customer = rng.poisson(expected)

    cust_idx = np.repeat(np.arange(n), n_per_customer)
    day = np.empty(cust_idx.size, dtype=np.int32)
    spend = np.ones(cust_idx.size, dtype=np.float64)

    pos = 0
    for i in range(n):
        k = int(n_per_customer[i])
        if k == 0:
            continue
        w = profiles[i]
        picked = rng.choice(w.size, size=k, p=w / w.sum()) + int(start_day[i])
        day[pos : pos + k] = picked
        if is_churner[i]:
            # El ticket medio tambien se encoge al acercarse el abandono.
            ramp_days = int(decay_weeks[i]) * 7
            closeness = np.clip((last_active[i] - picked) / ramp_days, 0.0, 1.0)
            spend[pos : pos + k] = 0.30 + 0.70 * closeness
        pos += k

    return cust_idx, day, spend, np.where(is_churner, last_active, -1)


# --------------------------------------------------------------------------------------
# basket_items: que entra en cada cesta
# --------------------------------------------------------------------------------------
def _build_customer_preferences(
    customers: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Construye la matriz de afinidad cliente x categoria y sus ciclos de reposicion.

    La afinidad parte de la popularidad base de la categoria, se perturba por cliente
    (unos compran mas droguería, otros mas frescos) y se pone a cero en las categorias
    con `gate` que ese hogar no cumple: sin bebe no hay panales, y eso es lo que hace
    que el par panales-toallitas sea una senal real y no ruido.

    El ciclo de reposicion propio de cada cliente escala `typical_repurchase_days` por
    el tamano del hogar (mas miembros, ciclo mas corto) mas ruido gaussiano.
    """
    n = len(customers)
    n_cat = len(cat.CATEGORIES)
    base = np.array([c.weight for c in cat.CATEGORIES])

    aff = base[None, :] * rng.lognormal(0.0, 0.55, (n, n_cat))

    gates = np.array([c.gate for c in cat.CATEGORIES], dtype=object)
    allowed = {
        "baby": customers["_has_baby"].to_numpy(),
        "pet": customers["_has_pet"].to_numpy(),
        "alcohol": customers["_buys_alcohol"].to_numpy(),
    }
    for gate, mask in allowed.items():
        cols = np.where(gates == gate)[0]
        if cols.size:
            aff[np.ix_(~mask, cols)] = 0.0

    household = customers["household_size_est"].to_numpy()
    hh_factor = 1.45 - 0.125 * household  # hogar de 1 -> 1.325, de 6 -> 0.70
    cycles = np.array([c.repurchase_days for c in cat.CATEGORIES], dtype=float)
    cycle = cycles[None, :] * hh_factor[:, None] * rng.lognormal(0.0, 0.20, (n, n_cat))
    np.clip(cycle, 2.0, None, out=cycle)

    return aff.astype(np.float32), cycle.astype(np.float32)


def _seasonal_matrix() -> np.ndarray:
    """Multiplicador estacional por (mes, categoria), con los valores de DATA_SPEC.md."""
    idx = cat.category_index()
    m = np.ones((13, len(cat.CATEGORIES)))
    for name, months, mult in cat.SEASONALITY:
        for month in months:
            m[month, idx[name]] = mult
    return m



def _apply_promo_uplift(
    base: np.ndarray, idxs: np.ndarray, active: list[tuple[int, int]]
) -> tuple[np.ndarray, dict[int, int]]:
    """Reparte la probabilidad dentro de una categoria aplicando el uplift de promocion.

    DATA_SPEC.md pide multiplicar la *probabilidad* de compra del producto promocionado
    por `PROMO_UPLIFT`, no su peso: si el producto ya concentra buena parte del surtido
    de su categoria, multiplicar el peso por 3 sube la probabilidad mucho menos de 3x
    porque el propio producto domina la normalizacion. Aqui se fija directamente la
    cuota objetivo `PROMO_UPLIFT * cuota_base` y se reparte el resto proporcionalmente
    entre los productos no promocionados, de modo que el uplift observado sea el pedido.

    Returns:
        `(pesos, promo_of)`, con `promo_of` mapeando indice global de producto a fila de
        `promotions` para poder informar `promotion_id` en la linea de ticket.
    """
    total = base.sum()
    promo_of: dict[int, int] = {}
    local: list[int] = []
    for prod_i, promo_i in active:
        hit = np.where(idxs == prod_i)[0]
        if hit.size:
            local.append(int(hit[0]))
            promo_of[int(prod_i)] = promo_i
    if not local or total <= 0:
        return base.copy(), promo_of

    local_idx = np.array(sorted(set(local)))
    share = np.minimum(cat.PROMO_UPLIFT * base[local_idx] / total, 0.90 / local_idx.size)
    rest_mask = np.ones(base.size, dtype=bool)
    rest_mask[local_idx] = False
    rest_total = base[rest_mask].sum()
    if rest_total <= 0:
        return base.copy(), promo_of

    pw = np.empty_like(base)
    pw[local_idx] = share
    pw[rest_mask] = base[rest_mask] / rest_total * (1.0 - share.sum())
    return pw, promo_of


def _product_choice_weights(
    pop: np.ndarray,
    idxs: np.ndarray,
    pref_i: int,
    loyalty_j: float,
    active: list[tuple[int, int]] | None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Pesos de cada referencia de una categoria: fidelidad de marca y luego promocion.

    Es el camino "lento" de la eleccion de producto de `_generate_baskets_and_items` (el
    rapido, sin habito ni promocion, usa directamente la popularidad). Vive aparte para
    que `OracleRecorder` calcule la probabilidad real de cada SKU con la misma cuenta
    que usa el sorteo.
    """
    pw = pop[idxs].astype(np.float64)
    if pref_i >= 0:
        local = int(np.searchsorted(idxs, pref_i))
        rest = pw.sum() - pw[local]
        if rest > 0:
            pw = pw / rest * (1.0 - loyalty_j)
            pw[local] = loyalty_j
    if active:
        return _apply_promo_uplift(pw, idxs, active)
    return pw, {}


@dataclass
class OracleRecorder:
    """Registra, sin consumir aleatoriedad, la probabilidad real de cada cesta (punto A6).

    Para cada cesta con `day >= from_day` guarda el vector de pesos de categoria **justo
    antes del primer sorteo** (afinidad x estacionalidad x ciclo de reposicion, con los
    gates ya aplicados) y, por categoria, la referencia mas probable y su probabilidad si
    esa categoria sale. Es la informacion privilegiada del generador: con ella se calcula
    el oraculo bayesiano, el techo teorico contra el que se mide el recomendador
    (`src/recommender/oracle.py`).

    No toca ningun generador aleatorio ni el estado del bucle, asi que el dataset sale
    identico byte a byte con o sin registrador (lo fija un test). Nunca se escribe en
    `data/raw`: lo vuelca `data_generation/export_oracle.py`.
    """

    from_day: int
    basket_index: list[int] = field(default_factory=list)
    weights: list[np.ndarray] = field(default_factory=list)
    best_product: list[np.ndarray] = field(default_factory=list)
    best_prob: list[np.ndarray] = field(default_factory=list)

    def record(
        self,
        b: int,
        w: np.ndarray,
        ci: int,
        preferred: np.ndarray,
        loyalty: np.ndarray,
        pop: np.ndarray,
        cat_products: list[np.ndarray],
        day_promos: dict[int, list[tuple[int, int]]],
    ) -> None:
        n_cat = w.size
        best = np.full(n_cat, -1, dtype=np.int32)
        prob = np.zeros(n_cat, dtype=np.float32)
        for j in np.flatnonzero(w > 0):
            idxs = cat_products[j]
            pref_i = int(preferred[ci, j]) if ci >= 0 else -1
            pw, _ = _product_choice_weights(pop, idxs, pref_i, loyalty[j], day_promos.get(j))
            top = int(np.argmax(pw))
            best[j] = idxs[top]
            prob[j] = pw[top] / pw.sum()
        self.basket_index.append(b)
        self.weights.append(w.astype(np.float64))
        self.best_product.append(best)
        self.best_prob.append(prob)


def _generate_baskets_and_items(
    cfg: GeneratorConfig,
    cal: Calendar,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    promotions: pd.DataFrame,
    rng_dates: np.random.Generator,
    rng_items: np.random.Generator,
    oracle: OracleRecorder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Construye `baskets` y `basket_items` aplicando todas las reglas de negocio.

    Para cada cesta se parte del peso base de cada categoria y se le aplican, en este
    orden, los cuatro patrones de DATA_SPEC.md:

    1. **Estacionalidad**: multiplicador del mes en curso (turron en diciembre, helados
       en verano...).
    2. **Afinidad del cliente**: su preferencia persistente por la categoria, con los
       gates de hogar ya aplicados.
    3. **Ciclo de reposicion**: la categoria pierde peso justo despues de comprarla y lo
       recupera segun se acerca a `typical_repurchase_days`, ajustado al hogar.
    4. **Afinidad de cesta**: al entrar un producto de una categoria disparadora, la
       categoria asociada ve su peso multiplicado por el lift de la tabla.

    Elegida la categoria, el producto concreto sale de dos efectos que se componen: la
    **fidelidad de marca** (la primera compra del cliente en la categoria le fija una
    referencia preferida, que las siguientes repiten con probabilidad `Category.loyalty`)
    y el **uplift de promocion** (un producto con promocion activa ese dia se lleva
    `PROMO_UPLIFT` veces su cuota). El orden importa: la promocion se aplica sobre la
    cuota ya repartida por habito, de modo que solo puede llevarse la parte no fiel.

    Las cestas se procesan en orden cronologico porque el punto 3 depende de la compra
    anterior del mismo cliente.

    Con `oracle`, ademas, se registran los pesos de cada cesta a partir de
    `oracle.from_day` (ver `OracleRecorder`); el dataset no cambia.
    """
    n_cat = len(cat.CATEGORIES)
    cust_idx, day, spend, last_active = _generate_basket_dates(cfg, cal, customers, rng_dates)

    # --- Cestas anonimas: sin customer_id, tipicas de tienda fisica ---
    n_anon = int(round(cfg.n_baskets * cfg.anonymous_basket_share))
    anon_day = rng_items.integers(0, cal.n_days, n_anon).astype(np.int32)
    cust_idx = np.concatenate([cust_idx, np.full(n_anon, -1, dtype=np.int64)])
    day = np.concatenate([day, anon_day])
    spend = np.concatenate([spend, np.ones(n_anon)])

    # Orden cronologico estable: imprescindible para que el ciclo de reposicion vea la
    # historia real del cliente y para que el resultado no dependa del orden de llegada.
    order = np.lexsort((cust_idx, day))
    cust_idx, day, spend = cust_idx[order], day[order], spend[order]
    n_baskets = cust_idx.size

    # --- Preferencias, ciclos y estado de reposicion ---
    aff, cycle = _build_customer_preferences(customers, rng_items)
    season = _seasonal_matrix()
    base_w = np.array([c.weight for c in cat.CATEGORIES], dtype=np.float64)
    # Perfil "medio" para las cestas anonimas: sin historial ni gates conocidos, se usa
    # la popularidad base atenuada por la prevalencia del gate.
    anon_aff = base_w.copy()
    for i, c in enumerate(cat.CATEGORIES):
        if c.gate:
            anon_aff[i] *= cat.GATE_PREVALENCE[c.gate]

    last_day = np.full((len(customers), n_cat), -1, dtype=np.int32)

    # --- Afinidad de cesta: disparadora -> (asociada, lift) ---
    cidx = cat.category_index()
    affinity_map = {
        cidx[t]: (cidx[a], cat.applied_affinity_lift(lift))
        for t, a, lift in cat.AFFINITY_PAIRS
    }

    # --- Promociones activas por dia y categoria ---
    promo_prod = promotions["_product_idx"].to_numpy()
    promo_start = promotions["_start_day"].to_numpy()
    promo_end = promotions["_end_day"].to_numpy()
    prod_cat = products["category_idx"].to_numpy()
    promos_by_day: list[dict[int, list[tuple[int, int]]]] = [{} for _ in range(cal.n_days)]
    for p in range(len(promotions)):
        for d in range(int(promo_start[p]), int(promo_end[p]) + 1):
            promos_by_day[d].setdefault(int(prod_cat[promo_prod[p]]), []).append(
                (int(promo_prod[p]), p)
            )

    # --- Surtido por categoria, con su cumsum de popularidad precomputado ---
    pop = products["popularity"].to_numpy()
    cat_products: list[np.ndarray] = []
    cat_cumw: list[np.ndarray] = []
    for ci in range(n_cat):
        idxs = np.where(prod_cat == ci)[0]
        cat_products.append(idxs)
        cat_cumw.append(np.cumsum(pop[idxs]))

    # --- Fidelidad de marca: referencia preferida por cliente y categoria ---
    # -1 = el cliente aun no ha comprado nunca en esa categoria. Su primera compra fija
    # la referencia preferida y las siguientes la repiten con probabilidad
    # `Category.loyalty` (alta en categorias de habito, baja en las exploratorias).
    preferred = np.full((len(customers), n_cat), -1, dtype=np.int32)
    loyalty = np.array([c.loyalty for c in cat.CATEGORIES], dtype=np.float64)

    # --- Numero de lineas por cesta ---
    household = customers["household_size_est"].to_numpy()
    hh_size_factor = np.where(cust_idx >= 0, 0.70 + 0.12 * household[cust_idx], 0.60)
    lam = 4.0 * hh_size_factor * spend
    n_items = 1 + rng_items.poisson(np.maximum(lam, 0.2))
    np.clip(n_items, 1, 20, out=n_items)
    total_slots = int(n_items.sum())

    # Aleatoriedad en bloque: mas rapido y, sobre todo, de consumo determinista.
    u_cat = rng_items.random(total_slots)
    u_prod = rng_items.random(total_slots)
    qty_draw = rng_items.choice([1, 2, 3, 4], size=total_slots, p=[0.62, 0.24, 0.10, 0.04])

    out_basket = np.empty(total_slots, dtype=np.int64)
    out_prod = np.empty(total_slots, dtype=np.int64)
    out_promo = np.full(total_slots, -1, dtype=np.int64)
    slot = 0

    picked_buf = np.empty(20, dtype=np.int64)
    for b in range(n_baskets):
        ci = int(cust_idx[b])
        d = int(day[b])
        w = anon_aff.copy() if ci < 0 else aff[ci].astype(np.float64)
        w *= season[cal.month[d]]

        if ci >= 0:
            ld = last_day[ci]
            ratio = (d - ld) / cycle[ci]
            mult = np.clip(np.power(np.maximum(ratio, 0.0), 1.8), 0.03, 4.0)
            # Una categoria nunca comprada no tiene ciclo observable todavia.
            w *= np.where(ld < 0, 1.0, mult)

        day_promos = promos_by_day[d]
        if oracle is not None and d >= oracle.from_day:
            oracle.record(b, w, ci, preferred, loyalty, pop, cat_products, day_promos)
        k = int(n_items[b])
        n_picked = 0
        for _ in range(k):
            cum = np.cumsum(w)
            total = cum[-1]
            if total <= 0:
                break
            j = int(np.searchsorted(cum, u_cat[slot] * total))
            j = min(j, n_cat - 1)

            # Producto dentro de la categoria. Tres efectos, en este orden:
            #   1. fidelidad de marca: la referencia preferida del cliente se lleva una
            #      cuota fija `loyalty[j]` de la eleccion;
            #   2. popularidad: el resto del surtido se reparte por su cola larga;
            #   3. uplift de promocion: se aplica encima, asi una promocion de la
            #      competencia solo puede llevarse la parte no fiel de la categoria --
            #      que es exactamente lo que hace una promocion en gran consumo.
            idxs = cat_products[j]
            active = day_promos.get(j)
            pref_i = int(preferred[ci, j]) if ci >= 0 else -1

            if pref_i < 0 and active is None:
                # Camino rapido: ni habito que respetar ni promocion que aplicar.
                pcum = cat_cumw[j]
                promo_of = {}
            else:
                pw, promo_of = _product_choice_weights(pop, idxs, pref_i, loyalty[j], active)
                pcum = np.cumsum(pw)
            pi = int(np.searchsorted(pcum, u_prod[slot] * pcum[-1]))
            prod_i = int(idxs[min(pi, idxs.size - 1)])
            if ci >= 0 and pref_i < 0:
                preferred[ci, j] = prod_i

            out_basket[slot] = b
            out_prod[slot] = prod_i
            if prod_i in promo_of:
                out_promo[slot] = promo_of[prod_i]
            slot += 1

            picked_buf[n_picked] = j
            n_picked += 1
            w[j] = 0.0  # una sola linea por categoria y cesta
            assoc = affinity_map.get(j)
            if assoc is not None:
                w[assoc[0]] *= assoc[1]

        if ci >= 0 and n_picked:
            last_day[ci, picked_buf[:n_picked]] = d

    out_basket = out_basket[:slot]
    out_prod = out_prod[:slot]
    out_promo = out_promo[:slot]
    qty = qty_draw[:slot].astype(np.int64)

    # --- Precio pagado segun el tipo de promocion ---
    unit_price = products["unit_price"].to_numpy()
    price_paid = unit_price[out_prod].copy()
    has_promo = out_promo >= 0
    if has_promo.any():
        ptype = promotions["promo_type"].to_numpy()[out_promo[has_promo]]
        pval = promotions["discount_value"].to_numpy()[out_promo[has_promo]]
        base = price_paid[has_promo]
        paid = np.where(
            ptype == "discount_pct",
            base * (1.0 - pval),
            np.where(ptype == "2x1", base * 0.5, np.maximum(base - pval, 0.20)),
        )
        price_paid[has_promo] = paid
        # En un 2x1 el cliente se lleva un numero par de unidades.
        two_for_one = np.zeros(slot, dtype=bool)
        two_for_one[np.where(has_promo)[0][ptype == "2x1"]] = True
        qty = np.where(two_for_one, np.maximum(2, (qty // 2) * 2), qty)

    promo_ids = promotions["promotion_id"].to_numpy()
    basket_items = pd.DataFrame(
        {
            "basket_id": out_basket,  # indice temporal, se sustituye por el id abajo
            "product_id": products["product_id"].to_numpy()[out_prod],
            "quantity": qty,
            "unit_price_paid": np.round(price_paid, 2),
            "promotion_id": np.where(has_promo, promo_ids[out_promo], None),
        }
    )

    # --- Cabecera de ticket ---
    line_amount = basket_items["quantity"].to_numpy() * basket_items["unit_price_paid"].to_numpy()
    total_amount = np.bincount(out_basket, weights=line_amount, minlength=n_baskets)

    pref = customers["preferred_channel"].to_numpy()
    channel = np.empty(n_baskets, dtype=object)
    identified = cust_idx >= 0
    # El cliente compra por su canal preferido 3 de cada 4 veces.
    loyal_to_channel = rng_items.random(n_baskets) < 0.75
    fallback = rng_items.choice(cat.CHANNELS, size=n_baskets, p=cat.CHANNEL_PROBS)
    channel[identified] = np.where(
        loyal_to_channel[identified], pref[cust_idx[identified]], fallback[identified]
    )
    # Las compras anonimas son casi siempre en tienda fisica.
    channel[~identified] = np.where(
        rng_items.random((~identified).sum()) < 0.90, "store", "web"
    )

    store_pick = rng_items.integers(1, cfg.n_stores + 1, n_baskets)
    store_id = np.where(channel == "store", [f"S{s:03d}" for s in store_pick], None)

    # Hora del ticket: pico de mediodia y de tarde, mas estrecho en tienda.
    hour = np.clip(rng_items.normal(np.where(channel == "store", 12.5, 18.0), 3.2), 7, 23.5)
    second = (hour * 3600).astype(np.int64) + rng_items.integers(0, 3600, n_baskets)
    np.clip(second, 0, 86_399, out=second)

    basket_ids = np.array([f"B{i:07d}" for i in range(1, n_baskets + 1)])
    baskets = pd.DataFrame(
        {
            "basket_id": basket_ids,
            "customer_id": np.where(
                identified, customers["customer_id"].to_numpy()[cust_idx], None
            ),
            "channel": channel,
            "basket_date": _stamp(cal, day, second),
            "store_id": store_id,
            "total_amount": np.round(total_amount, 2),
        }
    )
    baskets["_customer_idx"] = cust_idx
    baskets["_day"] = day
    baskets["_second"] = second

    basket_items["basket_id"] = basket_ids[out_basket]
    return baskets, basket_items, last_active


# --------------------------------------------------------------------------------------
# sessions / session_events
# --------------------------------------------------------------------------------------
def _generate_sessions(
    cfg: GeneratorConfig,
    cal: Calendar,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    baskets: pd.DataFrame,
    basket_items: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera la navegacion online y sus eventos.

    Las sesiones que convierten se cuelgan de una cesta online real, pero la sesion **no
    es** el ticket escrito de otra forma: eso convertiria `session_events` en una copia
    del target y cualquier modelo entrenado con esa senal daria metricas falsas. El
    embudo se genera con las tres fugas que tiene un embudo real:

    - solo `session_item_browse_rate` de las lineas del ticket pasa por la web (el resto
      entra por lista de la compra, recompra rapida o directamente en tienda), asi que
      `P(visto | en la cesta) < 1`;
    - hay `session_abandoned_adds` productos de media que se anaden al carrito y se
      quedan ahi, asi que `P(en la cesta | add_to_cart) < 1`;
    - hay `session_view_only` productos de media que se miran y no se anaden, asi que
      `P(en la cesta | view)` baja todavia mas.

    Ademas, cada `add_to_cart` va `session_add_lag_s` segundos por detras de su `view`.
    Ese retardo es lo que hace la senal *utilizable* en vez de tautologica: al cortar la
    sesion en un instante t hay productos ya vistos y todavia no anadidos, que son
    justamente los que el recomendador de la Fase 3 tiene que adivinar.

    Las sesiones que no convierten solo dejan vistas y algun `add_to_cart` suelto.
    """
    channel = baskets["channel"].to_numpy()
    online = np.where((channel == "app") | (channel == "web"))[0]

    n_converted = min(int(round(cfg.n_sessions * cfg.session_conversion_rate)), online.size)
    n_open = max(cfg.n_sessions - n_converted, 0)
    converted_baskets = np.sort(rng.choice(online, size=n_converted, replace=False))

    # Offsets de `basket_items` por cesta: las lineas ya vienen agrupadas por cesta.
    n_baskets = len(baskets)
    counts = np.bincount(
        pd.factorize(basket_items["basket_id"], sort=True)[0], minlength=n_baskets
    )
    offsets = np.concatenate([[0], np.cumsum(counts)])
    item_prod = basket_items["product_id"].to_numpy()

    prod_ids = products["product_id"].to_numpy()
    pop = products["popularity"].to_numpy()
    pop_p = pop / pop.sum()

    b_day = baskets["_day"].to_numpy()
    b_second = baskets["_second"].to_numpy()
    b_customer = baskets["customer_id"].to_numpy()
    b_id = baskets["basket_id"].to_numpy()

    n_total = n_converted + n_open
    session_ids = [f"S{i:07d}" for i in range(1, n_total + 1)]
    device = rng.choice(cat.DEVICE_TYPES, size=n_total, p=cat.DEVICE_TYPE_PROBS)

    s_customer: list[str | None] = []
    s_day = np.empty(n_total, dtype=np.int64)
    s_second = np.empty(n_total, dtype=np.int64)
    s_basket: list[str | None] = []

    ev_session: list[str] = []
    ev_product: list[str] = []
    ev_type: list[str] = []
    ev_day: list[int] = []
    ev_second: list[int] = []

    # Duracion de la sesion antes de cerrar la compra (2-40 min).
    conv_len = rng.integers(120, 2400, n_converted)
    n_abandoned = rng.poisson(cfg.session_abandoned_adds, n_converted)
    n_view_only = rng.poisson(cfg.session_view_only, n_converted)
    lag_lo, lag_hi = cfg.session_add_lag_s
    open_views = 1 + rng.poisson(1.4, n_open)
    open_adds = rng.random(n_open) < 0.15
    open_day = rng.integers(0, cal.n_days, n_open)
    open_second = np.clip(rng.normal(19 * 3600, 4 * 3600, n_open), 0, 86_399).astype(np.int64)
    open_anon = rng.random(n_open) < 0.35
    open_cust = rng.integers(0, len(customers), n_open)
    cust_ids = customers["customer_id"].to_numpy()

    # --- Sesiones que convierten ---
    for k, b in enumerate(converted_baskets):
        sid = session_ids[k]
        start = int(b_second[b]) - int(conv_len[k])
        d = int(b_day[b])
        if start < 0:  # la sesion no cruza la medianoche: se pega al inicio del dia
            start = 0
        s_customer.append(b_customer[b])
        s_day[k] = d
        s_second[k] = start
        s_basket.append(b_id[b])

        lo, hi = int(offsets[b]), int(offsets[b + 1])
        in_basket = pd.unique(item_prod[lo:hi])

        # 1. Que parte del ticket deja rastro online. El resto se compra sin navegar.
        browsed = in_basket[rng.random(in_basket.size) < cfg.session_item_browse_rate]

        # 2. Productos ajenos al ticket: unos se abandonan en el carrito, otros solo se
        #    miran. Se sortean por popularidad; si sale uno que ya esta en la cesta o
        #    repetido, se descarta (se queda con su primer papel).
        n_ab, n_vo = int(n_abandoned[k]), int(n_view_only[k])
        n_extra = n_ab + n_vo
        extra = prod_ids[rng.choice(len(prod_ids), size=n_extra, p=pop_p)] if n_extra else ()
        seen = set(in_basket.tolist())
        abandoned: list[str] = []
        view_only: list[str] = []
        for i, prod in enumerate(extra):
            if prod in seen:
                continue
            seen.add(prod)
            (abandoned if i < n_ab else view_only).append(prod)

        added = list(browsed) + abandoned
        browse = added + view_only
        n_browse = len(browse)
        if not n_browse:
            continue

        # 3. Cronologia: la navegacion ocupa el primer 80 % de la sesion y cada anadido
        #    va por detras de su vista. Todo se recorta al instante de la compra.
        span = max(int(b_second[b]) - start, 1)
        order = rng.permutation(n_browse)
        view_at = np.sort(rng.integers(0, max(int(span * 0.8), 1), n_browse))
        lag = rng.integers(lag_lo, lag_hi, n_browse)
        added_set = set(added)

        events: list[tuple[int, str, str]] = []
        for i, pos in enumerate(order):
            prod = browse[pos]
            events.append((int(view_at[i]), prod, "view"))
            if prod in added_set:
                events.append((min(int(view_at[i]) + int(lag[i]), span), prod, "add_to_cart"))
        events.sort(key=lambda e: e[0])
        for second, prod, kind in events:
            ev_session.append(sid)
            ev_product.append(prod)
            ev_type.append(kind)
            ev_day.append(d)
            ev_second.append(start + second)

    # --- Sesiones que no convierten ---
    for j in range(n_open):
        k = n_converted + j
        sid = session_ids[k]
        d = int(open_day[j])
        start = int(open_second[j])
        s_customer.append(None if open_anon[j] else cust_ids[open_cust[j]])
        s_day[k] = d
        s_second[k] = start
        s_basket.append(None)

        n_v = int(open_views[j])
        viewed = prod_ids[rng.choice(len(prod_ids), size=n_v, p=pop_p)]
        for i, p in enumerate(viewed):
            ev_session.append(sid)
            ev_product.append(p)
            ev_type.append("view")
            ev_day.append(d)
            ev_second.append(min(start + i * 45, 86_399))
        if open_adds[j] and n_v:
            ev_session.append(sid)
            ev_product.append(viewed[-1])
            ev_type.append("add_to_cart")
            ev_day.append(d)
            ev_second.append(min(start + n_v * 45, 86_399))

    sessions = pd.DataFrame(
        {
            "session_id": session_ids,
            "customer_id": s_customer,
            "session_date": _stamp(cal, s_day, np.clip(s_second, 0, 86_399)),
            "device_type": device,
            "converted": np.concatenate([np.ones(n_converted, bool), np.zeros(n_open, bool)]),
            "basket_id": s_basket,
        }
    )
    session_events = pd.DataFrame(
        {
            "session_id": ev_session,
            "product_id": ev_product,
            "event_type": ev_type,
            "event_timestamp": _stamp(
                cal, np.array(ev_day, dtype=np.int64), np.clip(np.array(ev_second), 0, 86_399)
            ),
        }
    )
    return sessions, session_events


# --------------------------------------------------------------------------------------
# Problemas de calidad deliberados
# --------------------------------------------------------------------------------------
def _mess_up_category(name: str, style: int) -> str:
    """Ensucia el nombre de una categoria imitando errores tipicos de captura."""
    if style == 0:
        return name.upper()
    if style == 1:
        return name.lower()
    if style == 2:
        return f"  {name}"
    if style == 3:
        return f"{name}  "
    if " " in name:
        return name.replace(" ", "  ")
    return name.upper()


def _inject_quality_issues(
    cfg: GeneratorConfig,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    baskets: pd.DataFrame,
    basket_items: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inyecta los problemas de calidad de DATA_SPEC.md, seccion "Calidad del dato".

    Son deliberados: el objetivo del ETL de la Fase 2 es detectarlos y limpiarlos, y el
    Data Trust Score de la Tarea 1 medirlos. Los nulos de `baskets.customer_id` NO se
    inyectan aqui: son compras anonimas legitimas creadas por el generador y no deben
    descartarse en la limpieza.
    """
    # 1. Categorias mal escritas (mayusculas, minusculas, espacios sobrantes).
    n_p = len(products)
    messy = np.where(rng.random(n_p) < cfg.messy_category_share)[0]
    styles = rng.integers(0, 5, messy.size)
    cats = products["category"].to_numpy().copy()
    for pos, style in zip(messy, styles):
        cats[pos] = _mess_up_category(cats[pos], int(style))
    products = products.assign(category=cats)

    # 2. Marcas sin informar.
    brands = products["brand"].to_numpy().copy().astype(object)
    brands[rng.random(n_p) < cfg.null_brand_share] = None
    products = products.assign(brand=brands)

    # 3. Ciudades sin informar.
    n_c = len(customers)
    cities = customers["city"].to_numpy().copy().astype(object)
    cities[rng.random(n_c) < cfg.null_city_share] = None
    customers = customers.assign(city=cities)

    # 4. Fechas inconsistentes: alta posterior a la primera compra del cliente.
    first_purchase = (
        baskets.loc[baskets["_customer_idx"] >= 0]
        .groupby("_customer_idx")["basket_date"]
        .min()
    )
    candidates = first_purchase.index.to_numpy()
    if candidates.size:
        n_bad = int(round(candidates.size * cfg.inconsistent_date_share))
        if n_bad:
            picked = rng.choice(candidates, size=n_bad, replace=False)
            signup = customers["signup_date"].to_numpy().copy().astype(object)
            for pos in picked:
                # Se mueve el alta unos dias despues de la primera compra registrada.
                base = pd.Timestamp(first_purchase.loc[pos][:10])
                signup[pos] = (base + pd.Timedelta(days=int(rng.integers(1, 45)))).strftime(
                    "%Y-%m-%d"
                )
            customers = customers.assign(signup_date=signup)

    # 5. Lineas de ticket duplicadas.
    n_i = len(basket_items)
    dup_idx = np.where(rng.random(n_i) < cfg.dup_item_share)[0]
    if dup_idx.size:
        basket_items = pd.concat(
            [basket_items, basket_items.iloc[dup_idx]], ignore_index=True
        )
        # Ordenacion estable por cesta: el duplicado queda pegado a la linea original,
        # como pasaria en un TPV que emite la misma linea dos veces.
        basket_items = (
            basket_items.sort_values("basket_id", kind="stable")
            .reset_index(drop=True)
        )

    # 6. Cantidades negativas (devoluciones mal codificadas).
    n_i = len(basket_items)
    neg = rng.random(n_i) < cfg.negative_qty_share
    qty = basket_items["quantity"].to_numpy().copy()
    qty[neg] = -qty[neg]
    basket_items = basket_items.assign(quantity=qty)

    # 7. Outliers de importe en la cabecera de ticket.
    n_b = len(baskets)
    out = rng.random(n_b) < cfg.amount_outlier_share
    amounts = baskets["total_amount"].to_numpy().copy()
    amounts[out] = np.round(amounts[out] * rng.uniform(20, 60, out.sum()), 2)
    baskets = baskets.assign(total_amount=amounts)

    return customers, products, baskets, basket_items


# --------------------------------------------------------------------------------------
# Esquema de salida (orden de columnas segun DATA_SPEC.md) y diccionario de datos
# --------------------------------------------------------------------------------------
TABLE_COLUMNS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "customers": (
        ("customer_id", "string", "PK del cliente."),
        ("signup_date", "date", "Fecha de alta. Puede ser anterior al periodo simulado."),
        ("country", "string", "Pais. Esta version se queda en Espana."),
        ("city", "string", "Ciudad simulada. Admite nulos (problema de calidad)."),
        ("household_size_est", "int", "Tamano estimado del hogar (1-6)."),
        ("loyalty_tier", "string", "bronze / silver / gold."),
        ("preferred_channel", "string", "app / web / store."),
        (
            "churn_label",
            "boolean",
            f"Derivado: True si no compra en los ultimos {cat.CHURN_WINDOW_DAYS} dias "
            "del periodo simulado.",
        ),
    ),
    "products": (
        ("product_id", "string", "PK del producto."),
        ("department", "string", "Departamento (8 valores)."),
        (
            "category",
            "string",
            "Subcategoria. Contiene inconsistencias de mayusculas y espacios "
            "(problema de calidad).",
        ),
        ("brand", "string", "Marca. Admite nulos (problema de calidad)."),
        ("is_private_label", "boolean", "True si es marca de distribuidor."),
        ("is_perishable", "boolean", "True si es perecedero."),
        ("unit_price", "float", "Precio de tarifa por unidad, en euros."),
        ("pack_size", "int", "Unidades por envase."),
        (
            "typical_repurchase_days",
            "int",
            "Intervalo tipico de recompra de la categoria, en dias. Lo usa el "
            "generador y la Tarea 2 (due_for_repurchase).",
        ),
    ),
    "promotions": (
        ("promotion_id", "string", "PK de la promocion."),
        ("product_id", "string", "FK a products."),
        ("promo_type", "string", "2x1 / discount_pct / coupon."),
        (
            "discount_value",
            "float",
            "Fraccion de descuento en discount_pct y 2x1; importe fijo en euros en coupon.",
        ),
        ("start_date", "date", "Primer dia de vigencia (incluido)."),
        ("end_date", "date", "Ultimo dia de vigencia (incluido)."),
    ),
    "baskets": (
        ("basket_id", "string", "PK del ticket."),
        (
            "customer_id",
            "string",
            "FK a customers. Nulo en compras anonimas: son legitimas, no descartar.",
        ),
        ("channel", "string", "app / web / store."),
        ("basket_date", "datetime", "Fecha y hora del ticket."),
        ("store_id", "string", "Tienda fisica. Nulo si el canal no es store."),
        (
            "total_amount",
            "float",
            "Importe del ticket. Contiene outliers y no cuadra con basket_items cuando "
            "hay duplicados o devoluciones: se recalcula en el ETL.",
        ),
    ),
    "basket_items": (
        ("basket_id", "string", "FK a baskets."),
        ("product_id", "string", "FK a products."),
        (
            "quantity",
            "int",
            "Unidades. Hay valores negativos (devoluciones mal codificadas).",
        ),
        ("unit_price_paid", "float", "Precio realmente pagado; difiere de unit_price con promo."),
        ("promotion_id", "string", "FK a promotions. Nulo si la linea no lleva promocion."),
    ),
    "sessions": (
        ("session_id", "string", "PK de la sesion online."),
        ("customer_id", "string", "FK a customers. Nulo si el visitante no se identifica."),
        ("session_date", "datetime", "Inicio de la sesion."),
        ("device_type", "string", "mobile / desktop / tablet."),
        ("converted", "boolean", "True si la sesion termino en una cesta."),
        ("basket_id", "string", "FK a baskets. Nulo si no convirtio."),
    ),
    "session_events": (
        ("session_id", "string", "FK a sessions."),
        ("product_id", "string", "FK a products."),
        (
            "event_type",
            "string",
            "view / add_to_cart. Un add_to_cart NO implica compra: hay abandono de "
            "carrito a nivel de linea (ver _generate_sessions).",
        ),
        (
            "event_timestamp",
            "datetime",
            "Momento del evento dentro de la sesion. Cada add_to_cart va por detras de "
            "su view, de modo que un corte temporal deja productos vistos y aun no "
            "anadidos.",
        ),
    ),
}

# Orden de escritura: fijo, para que el manifiesto y el diccionario sean estables.
TABLE_ORDER: tuple[str, ...] = (
    "customers",
    "products",
    "promotions",
    "baskets",
    "basket_items",
    "sessions",
    "session_events",
)


def _derive_churn_label(
    cal: Calendar, customers: pd.DataFrame, baskets: pd.DataFrame
) -> pd.DataFrame:
    """Calcula `churn_label` a partir de las cestas realmente generadas.

    La etiqueta es la definicion literal de DATA_SPEC.md, no la intencion interna del
    generador: un cliente de baja frecuencia puede acabar etiquetado como churn sin que
    se le marcara como tal, igual que pasaria con datos reales.
    """
    cutoff = cal.n_days - cat.CHURN_WINDOW_DAYS
    last = (
        baskets.loc[baskets["_customer_idx"] >= 0]
        .groupby("_customer_idx")["_day"]
        .max()
    )
    last_day = np.full(len(customers), -1, dtype=np.int64)
    last_day[last.index.to_numpy()] = last.to_numpy()
    return customers.assign(churn_label=last_day < cutoff)


def _write_tables(cfg: GeneratorConfig, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Escribe las tablas en CSV y devuelve el sha256 de cada fichero.

    El formato de escritura esta fijado a proposito (`lineterminator`, `float_format`,
    `na_rep`) para que el hash no dependa del sistema operativo.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name in TABLE_ORDER:
        cols = [c for c, _, _ in TABLE_COLUMNS[name]]
        path = cfg.out_dir / f"{name}.csv"
        tables[name][cols].to_csv(
            path,
            index=False,
            lineterminator="\n",
            float_format="%.2f",
            na_rep="",
            encoding="utf-8",
        )
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _write_data_dictionary(
    cfg: GeneratorConfig, tables: dict[str, pd.DataFrame], hashes: dict[str, str]
) -> None:
    """Genera el diccionario de datos y el manifiesto de la ejecucion."""
    lines = [
        "# Diccionario de datos — dataset sintetico de retail",
        "",
        "**Fichero autogenerado por `data_generation/generate_dataset.py`. No editar a mano.**",
        "",
        f"- Periodo simulado: `{cat.PERIOD_START}` → `{cat.PERIOD_END}`",
        f"- Semilla: `{cfg.seed}` · escala: `{cfg.scale}`",
        f"- Ventana de churn: {cat.CHURN_WINDOW_DAYS} dias",
        "",
        "Todo el dato es **100 % sintetico**: no procede de ningun cliente, producto ni",
        "retailer real.",
        "",
        "## Volumenes",
        "",
        "| Tabla | Filas |",
        "| --- | ---: |",
    ]
    for name in TABLE_ORDER:
        lines.append(f"| `{name}` | {len(tables[name]):,} |".replace(",", "."))
    lines.append("")

    for name in TABLE_ORDER:
        lines += [f"## `{name}`", "", "| Columna | Tipo | Notas |", "| --- | --- | --- |"]
        for col, dtype, note in TABLE_COLUMNS[name]:
            lines.append(f"| `{col}` | {dtype} | {note} |")
        lines += ["", f"sha256: `{hashes[name]}`", ""]

    lines += [
        "## Patrones inyectados",
        "",
        "Ver `DATA_SPEC.md` para los valores exactos. En resumen:",
        "",
        "- **Ciclos de reposicion** por cliente y categoria, escalados por el tamano del hogar.",
        f"- **{len(cat.AFFINITY_PAIRS)} pares de afinidad de cesta** (cerveza→snacks, "
        "panales→toallitas...).",
        f"- **Estacionalidad** en {len(cat.SEASONALITY)} categorias (Navidad, verano, Cuaresma).",
        f"- **Uplift de promocion** de {cat.PROMO_UPLIFT}x sobre el producto promocionado.",
        "- **Churn progresivo**: caida gradual de frecuencia y ticket en las 6-8 semanas previas.",
        "- **Problemas de calidad deliberados**: duplicados, nulos, categorias mal escritas,",
        "  cantidades negativas, fechas inconsistentes y outliers de importe.",
        "",
    ]
    (cfg.out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "seed": cfg.seed,
        "scale": cfg.scale,
        "period_start": cat.PERIOD_START,
        "period_end": cat.PERIOD_END,
        "row_counts": {name: int(len(tables[name])) for name in TABLE_ORDER},
        "sha256": hashes,
    }
    (cfg.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------------------
def generate(
    cfg: GeneratorConfig | None = None, oracle: OracleRecorder | None = None
) -> dict[str, pd.DataFrame]:
    """Genera el dataset completo y lo escribe en `cfg.out_dir`.

    Args:
        cfg: Volumenes y semilla.
        oracle: Registrador opcional de las probabilidades reales de cada cesta (ver
            `OracleRecorder`). No cambia ni un byte del dataset.

    Returns:
        Las 7 tablas ya con el orden de columnas de DATA_SPEC.md.
    """
    cfg = cfg or GeneratorConfig()
    rngs = _rng_streams(cfg.seed)
    cal = _build_calendar()

    products = _generate_products(cfg, rngs["products"])
    customers = _generate_customers(cfg, cal, rngs["customers"])
    promotions = _generate_promotions(cfg, cal, products, rngs["promotions"])

    baskets, basket_items, _last_active = _generate_baskets_and_items(
        cfg,
        cal,
        customers,
        products,
        promotions,
        rngs["basket_dates"],
        rngs["basket_items"],
        oracle=oracle,
    )
    sessions, session_events = _generate_sessions(
        cfg, cal, customers, products, baskets, basket_items, rngs["sessions"]
    )

    customers = _derive_churn_label(cal, customers, baskets)
    customers, products, baskets, basket_items = _inject_quality_issues(
        cfg, customers, products, baskets, basket_items, rngs["quality"]
    )

    tables = {
        "customers": customers,
        "products": products,
        "promotions": promotions,
        "baskets": baskets,
        "basket_items": basket_items,
        "sessions": sessions,
        "session_events": session_events,
    }
    hashes = _write_tables(cfg, tables)
    _write_data_dictionary(cfg, tables, hashes)
    return {name: tables[name][[c for c, _, _ in TABLE_COLUMNS[name]]] for name in TABLE_ORDER}


def _parse_args(argv: list[str] | None = None) -> GeneratorConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/raw", help="Directorio de salida.")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla (por defecto 42).")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Factor sobre los volumenes de referencia (0.02 = muestra rapida).",
    )
    args = parser.parse_args(argv)
    return GeneratorConfig(out_dir=Path(args.out), seed=args.seed, scale=args.scale)


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    tables = generate(cfg)
    print(f"Dataset generado en {cfg.out_dir.resolve()} (seed={cfg.seed}, scale={cfg.scale})")
    for name in TABLE_ORDER:
        print(f"  {name:<16} {len(tables[name]):>10,} filas")


if __name__ == "__main__":
    main()

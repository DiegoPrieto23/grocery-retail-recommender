"""Catalogo de negocio del supermercado sintetico.

Toda la parametrizacion de dominio (departamentos, categorias, ciclos de reposicion,
pares de afinidad de cesta y estacionalidad) vive aqui, separada de la mecanica de
generacion. Los valores de afinidad y estacionalidad son EXACTAMENTE los de
DATA_SPEC.md; si se cambian, hay que actualizar ese documento en el mismo commit.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Periodo simulado
# --------------------------------------------------------------------------------------
# Dos anos naturales completos: cada pico estacional (Navidad, verano, Cuaresma) aparece
# dos veces, que es lo minimo para que un modelo pueda aprender el patron.
PERIOD_START = "2024-01-01"
PERIOD_END = "2025-12-31"

# churn_label = 1 si el cliente no ha comprado en los ultimos CHURN_WINDOW_DAYS del
# periodo simulado (definicion de DATA_SPEC.md).
CHURN_WINDOW_DAYS = 60


# --------------------------------------------------------------------------------------
# Categorias
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Category:
    """Una subcategoria del surtido.

    Attributes:
        name: Nombre de la subcategoria (valor de products.category).
        department: Departamento al que cuelga (valor de products.department).
        repurchase_days: Intervalo tipico de recompra en dias (typical_repurchase_days).
        perishable: Si el producto es perecedero (is_perishable).
        price_lo: Extremo inferior del rango de unit_price.
        price_hi: Extremo superior del rango de unit_price.
        weight: Popularidad base de la categoria antes de estacionalidad y afinidad.
        loyalty: Probabilidad de repetir la referencia preferida del cliente en esa
            categoria (ver "Fidelidad de marca" mas abajo). Tiene que caer dentro de
            HABIT_LOYALTY_BAND o de EXPLORATORY_LOYALTY_BAND.
        gate: Restriccion de hogar; solo los clientes con ese atributo la compran.
            "baby" (hogar con bebe), "pet" (hogar con mascota), "alcohol" (consume
            alcohol) o None (sin restriccion).
    """

    name: str
    department: str
    repurchase_days: int
    perishable: bool
    price_lo: float
    price_hi: float
    weight: float
    loyalty: float
    gate: str | None = None


# --------------------------------------------------------------------------------------
# Fidelidad de marca y tamano del surtido
# --------------------------------------------------------------------------------------
# Surtido por categoria. El catalogo original tenia ~24 referencias por categoria y la
# eleccion dentro de la categoria era casi aleatoria: el hallazgo de la Fase 3 (ver
# ROADMAP.md) es que eso pone un techo artificial al recomendador, porque nadie puede
# predecir algo que se genero al azar. Un surtido curado de 8 referencias es lo que ve un
# cliente real delante del lineal de una categoria concreta, y deja sitio a que la
# fidelidad de marca sea la senal dominante en vez del ruido.
PRODUCTS_PER_CATEGORY = 8

# Fidelidad de marca: la primera compra de un cliente en una categoria le fija una
# referencia preferida, y las siguientes la repiten con probabilidad `Category.loyalty`.
# Las dos bandas separan el comportamiento real en gran consumo:
#
# * Habito: el cliente compra "su" marca y cambiarla tiene coste (mismo cafe, mismo
#   detergente, mismo champu, misma comida del perro). Droguerias, higiene, bebe,
#   mascotas, bebidas de marca y despensa envasada caen aqui.
# * Exploracion: la eleccion la manda el producto concreto del dia, no la marca --
#   frescos que se eligen por aspecto o corte (fruta, verdura, carne, pescado, queso),
#   caprichos que rotan por sabor (snacks, chocolate, helados, galletas) y compras de
#   ocasion (vino, cava, turron, congelados).
HABIT_LOYALTY_BAND = (0.75, 0.85)
EXPLORATORY_LOYALTY_BAND = (0.25, 0.40)


CATEGORIES: tuple[Category, ...] = (
    # --- Frescos ---
    Category("Leche", "Frescos", 6, True, 0.75, 1.60, 9.0, 0.80),
    Category("Yogur", "Frescos", 8, True, 1.20, 3.50, 5.0, 0.75),
    Category("Queso", "Frescos", 12, True, 2.00, 9.50, 4.5, 0.35),
    Category("Huevos", "Frescos", 10, True, 1.80, 4.20, 5.0, 0.78),
    Category("Pan", "Frescos", 4, True, 0.70, 2.80, 8.0, 0.40),
    Category("Embutido y fiambre", "Frescos", 9, True, 1.80, 7.50, 4.5, 0.35),
    Category("Fruta", "Frescos", 5, True, 0.90, 4.50, 7.5, 0.25),
    Category("Verdura", "Frescos", 5, True, 0.80, 3.90, 7.0, 0.25),
    Category("Carne de pollo", "Frescos", 8, True, 3.00, 8.50, 4.5, 0.35),
    Category("Carne de ternera", "Frescos", 12, True, 5.50, 16.00, 3.0, 0.30),
    Category("Pescado blanco", "Frescos", 11, True, 4.50, 14.00, 2.8, 0.28),
    Category("Torrijas y bolleria de Cuaresma", "Frescos", 300, True, 2.50, 6.50, 0.35, 0.30),
    Category("Bacalao", "Frescos", 90, True, 6.00, 18.00, 0.60, 0.30),
    # --- Despensa ---
    Category("Pasta", "Despensa", 21, False, 0.75, 2.60, 5.0, 0.75),
    Category("Salsa de tomate", "Despensa", 24, False, 0.80, 2.40, 3.5, 0.78),
    Category("Arroz", "Despensa", 30, False, 1.00, 3.20, 3.5, 0.80),
    Category("Legumbres", "Despensa", 30, False, 0.85, 2.90, 3.0, 0.75),
    Category("Conservas de pescado", "Despensa", 25, False, 1.30, 5.50, 3.2, 0.75),
    Category("Aceite de oliva", "Despensa", 40, False, 4.50, 12.50, 3.0, 0.82),
    Category("Cafe", "Despensa", 25, False, 2.50, 8.50, 4.5, 0.85),
    Category("Azucar y edulcorante", "Despensa", 45, False, 0.90, 3.10, 2.5, 0.82),
    Category("Cereales", "Despensa", 22, False, 1.90, 4.80, 3.5, 0.75),
    Category("Galletas", "Despensa", 15, False, 1.20, 3.80, 4.5, 0.40),
    Category("Snacks y aperitivos", "Despensa", 12, False, 0.95, 3.40, 5.5, 0.25),
    Category("Palomitas de microondas", "Despensa", 30, False, 1.10, 3.00, 1.8, 0.35),
    Category("Harina", "Despensa", 60, False, 0.60, 1.90, 1.6, 0.80),
    Category("Sal y especias", "Despensa", 90, False, 0.70, 3.50, 1.8, 0.80),
    Category("Chocolate y huevos de Pascua", "Despensa", 14, False, 1.10, 4.50, 4.5, 0.30),
    Category("Sopas y caldos", "Despensa", 20, False, 0.85, 3.20, 2.2, 0.75),
    Category("Turron y mazapan", "Despensa", 330, False, 3.50, 14.00, 0.30, 0.30),
    # --- Bebidas ---
    Category("Agua", "Bebidas", 10, False, 0.45, 3.50, 6.0, 0.85),
    Category("Refrescos", "Bebidas", 12, False, 0.90, 4.20, 5.5, 0.85),
    Category("Zumos", "Bebidas", 12, False, 1.10, 3.60, 3.5, 0.75),
    Category("Cerveza", "Bebidas", 12, False, 1.20, 6.50, 5.0, 0.80, "alcohol"),
    Category("Vino", "Bebidas", 18, False, 2.80, 15.00, 3.5, 0.30, "alcohol"),
    Category("Cava y espumosos", "Bebidas", 120, False, 4.00, 18.00, 0.60, 0.30, "alcohol"),
    # --- Drogueria ---
    Category("Detergente", "Drogueria", 45, False, 3.50, 12.00, 3.2, 0.85),
    Category("Suavizante", "Drogueria", 50, False, 2.20, 7.50, 2.6, 0.82),
    Category("Lavavajillas", "Drogueria", 40, False, 2.00, 8.00, 2.6, 0.80),
    Category("Lejia y limpiadores", "Drogueria", 50, False, 1.10, 5.50, 2.6, 0.78),
    Category("Limpiacristales", "Drogueria", 90, False, 1.60, 4.50, 1.3, 0.78),
    Category("Bolsas de basura", "Drogueria", 45, False, 1.20, 4.00, 2.0, 0.80),
    Category("Protector solar", "Drogueria", 200, False, 6.00, 18.00, 0.55, 0.75),
    # --- Higiene ---
    Category("Papel higienico", "Higiene", 30, False, 2.50, 9.50, 4.5, 0.82),
    Category("Champu", "Higiene", 55, False, 2.20, 8.50, 2.8, 0.85),
    Category("Acondicionador", "Higiene", 60, False, 2.40, 8.00, 2.0, 0.85),
    Category("Gel de ducha", "Higiene", 45, False, 1.80, 6.50, 3.0, 0.82),
    Category("Pasta de dientes", "Higiene", 60, False, 1.50, 5.50, 2.6, 0.85),
    Category("Desodorante", "Higiene", 70, False, 2.00, 6.50, 2.2, 0.85),
    Category("Higiene femenina", "Higiene", 35, False, 1.80, 6.00, 2.0, 0.85),
    # --- Bebe ---
    Category("Panales", "Bebe", 20, False, 6.50, 22.00, 6.0, 0.85, "baby"),
    Category("Toallitas humedas", "Bebe", 18, False, 1.50, 5.50, 5.0, 0.82, "baby"),
    Category("Leche infantil", "Bebe", 25, False, 9.00, 26.00, 3.0, 0.85, "baby"),
    Category("Potitos", "Bebe", 12, False, 0.90, 2.60, 3.5, 0.75, "baby"),
    # --- Mascotas ---
    Category("Comida para perro", "Mascotas", 25, False, 4.50, 26.00, 4.0, 0.85, "pet"),
    Category("Comida para gato", "Mascotas", 22, False, 4.00, 22.00, 3.8, 0.85, "pet"),
    Category("Arena para gato", "Mascotas", 35, False, 3.50, 12.00, 2.2, 0.82, "pet"),
    # --- Congelados ---
    Category("Verduras congeladas", "Congelados", 25, False, 1.20, 4.20, 2.8, 0.40),
    Category("Pizza congelada", "Congelados", 18, False, 2.20, 6.50, 3.2, 0.35),
    Category("Precocinados congelados", "Congelados", 20, False, 2.50, 8.00, 2.8, 0.30),
    Category("Helados", "Congelados", 14, False, 1.80, 7.50, 2.2, 0.25),
    Category("Marisco", "Congelados", 60, False, 5.50, 24.00, 1.0, 0.30),
)

DEPARTMENTS: tuple[str, ...] = (
    "Frescos",
    "Despensa",
    "Bebidas",
    "Drogueria",
    "Higiene",
    "Bebe",
    "Mascotas",
    "Congelados",
)


# --------------------------------------------------------------------------------------
# Afinidad de cesta (tabla de DATA_SPEC.md, seccion "Afinidad de cesta (detalle)")
# --------------------------------------------------------------------------------------
# Si ya hay un producto de la categoria disparadora en la cesta, la probabilidad de anadir
# un producto de la categoria asociada se multiplica por el lift indicado.
#
# Los 10 pares originales de la Fase 1. Se conservan con este nombre porque son la tabla
# historica que verifican el ETL y los tests; la tabla completa desde la Fase 8 es
# COMPLEMENT_PAIRS, que los incluye.
AFFINITY_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("Cerveza", "Snacks y aperitivos", 3.0),
    ("Pasta", "Salsa de tomate", 2.5),
    ("Panales", "Toallitas humedas", 4.0),
    ("Cafe", "Azucar y edulcorante", 2.0),
    ("Pan", "Embutido y fiambre", 2.2),
    ("Cereales", "Leche", 2.8),
    ("Vino", "Queso", 2.5),
    ("Detergente", "Suavizante", 3.5),
    ("Champu", "Acondicionador", 3.0),
    ("Palomitas de microondas", "Refrescos", 2.0),
)

_AFFINITY_MOTIVES = (
    "consumo social",
    "receta directa",
    "cesta de bebe",
    "habito de desayuno",
    "bocadillo",
    "desayuno",
    "maridaje",
    "rutina de lavado",
    "rutina de higiene",
    "noche de peli",
)

# Tabla completa de complementos (Fase 8, punto A5): (disparadora, asociada, lift
# objetivo, motivo). Los grupos nuevos cubren desayuno, higiene, mascota y recetas. El
# lift objetivo es el lift crudo que se quiere medir en `affinity_category`, igual que en
# la tabla original. Una categoria puede disparar varias asociadas.
COMPLEMENT_PAIRS: tuple[tuple[str, str, float, str], ...] = (
    *((t, a, lift, m) for (t, a, lift), m in zip(AFFINITY_PAIRS, _AFFINITY_MOTIVES)),
    # Desayuno
    ("Cafe", "Leche", 2.0, "cafe con leche"),
    ("Galletas", "Leche", 2.0, "desayuno y merienda"),
    ("Cereales", "Yogur", 2.0, "desayuno"),
    ("Pan", "Aceite de oliva", 2.0, "tostada"),
    # Higiene
    ("Gel de ducha", "Champu", 2.2, "rutina de ducha"),
    ("Pasta de dientes", "Desodorante", 2.0, "neceser"),
    ("Papel higienico", "Lejia y limpiadores", 2.0, "limpieza del bano"),
    # Mascota
    ("Comida para gato", "Arena para gato", 3.0, "mismo animal"),
    ("Comida para perro", "Bolsas de basura", 2.0, "paseo del perro"),
    # Recetas
    ("Pasta", "Queso", 2.0, "pasta gratinada"),
    ("Arroz", "Marisco", 2.5, "paella"),
    ("Legumbres", "Embutido y fiambre", 2.0, "cocido y fabada"),
    ("Carne de pollo", "Verdura", 2.0, "plato de diario"),
    ("Harina", "Huevos", 2.5, "reposteria"),
    ("Harina", "Azucar y edulcorante", 2.5, "reposteria"),
    ("Aceite de oliva", "Sal y especias", 2.2, "despensa de cocina"),
    ("Pescado blanco", "Verdura", 2.0, "plato de diario"),
    # Aperitivo y fiesta
    ("Pizza congelada", "Refrescos", 2.0, "cena rapida"),
    ("Cava y espumosos", "Turron y mazapan", 2.5, "Navidad"),
    ("Carne de ternera", "Cerveza", 2.0, "barbacoa"),
)


# --------------------------------------------------------------------------------------
# Sustitucion entre categorias (DATA_SPEC.md, "Sustitucion entre categorias (detalle)")
# --------------------------------------------------------------------------------------
# Al entrar una categoria del grupo, el peso del resto de categorias del grupo se
# multiplica por el factor. (nombre, categorias, factor).
SUBSTITUTION_GROUPS: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("Bebida fria", ("Agua", "Refrescos"), 0.35),
    ("Proteina principal", ("Carne de pollo", "Carne de ternera", "Pescado blanco"), 0.30),
    ("Limpieza de cocina", ("Lavavajillas", "Lejia y limpiadores"), 0.25),
    ("Capricho dulce", ("Galletas", "Chocolate y huevos de Pascua"), 0.35),
    ("Cena congelada", ("Pizza congelada", "Precocinados congelados"), 0.30),
    ("Picoteo", ("Snacks y aperitivos", "Palomitas de microondas"), 0.35),
    ("Alcohol de mesa", ("Cerveza", "Vino"), 0.35),
)


# --------------------------------------------------------------------------------------
# Misiones de compra (DATA_SPEC.md, "Misiones de compra (detalle)")
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Mission:
    """Motivo latente de una visita: que categorias se llevan y cuantas.

    Attributes:
        name: Identificador de la mision.
        base_share: Peso de partida de la mision antes de ajustarlo por cliente, dia y mes.
        size_mean: Media de categorias distintas *ademas de la primera* (la cesta tiene
            `1 + BinomialNegativa(size_mean x factor, size_dispersion)`).
        size_dispersion: Parametro `r` de la binomial negativa: cuanto mas bajo, mas cola.
        core: Multiplicador del peso de las categorias nucleo de la mision.
        leak: Multiplicador del resto de categorias. Nunca 0: en una visita a por pan
            tambien cae alguna otra cosa.
        month_factor: Multiplicador de `base_share` por mes (los que no aparecen, 1).
    """

    name: str
    base_share: float
    size_mean: float
    size_dispersion: float
    core: tuple[tuple[str, float], ...]
    leak: float
    month_factor: tuple[tuple[int, float], ...] = ()


_STOCK_UP_DEPARTMENTS = ("Despensa", "Drogueria", "Higiene")
_TREATS = (
    "Snacks y aperitivos",
    "Chocolate y huevos de Pascua",
    "Helados",
    "Palomitas de microondas",
    "Galletas",
)

MISSIONS: tuple[Mission, ...] = (
    Mission(
        "compra_semanal",
        base_share=0.12,
        size_mean=15.0,
        size_dispersion=4.0,
        core=(),  # se rellena abajo: todas las categorias, con acopio y caprichos
        leak=1.0,
    ),
    Mission(
        "reposicion",
        base_share=0.36,
        size_mean=2.5,
        size_dispersion=2.0,
        core=(
            ("Pan", 3.0), ("Leche", 3.0), ("Fruta", 2.5), ("Verdura", 2.5),
            ("Huevos", 2.0), ("Yogur", 2.0), ("Agua", 1.5), ("Embutido y fiambre", 1.5),
        ),
        leak=0.2,
    ),
    Mission(
        "desayuno",
        base_share=0.10,
        size_mean=3.2,
        size_dispersion=3.0,
        core=(
            ("Leche", 3.0), ("Cafe", 3.0), ("Cereales", 3.0), ("Galletas", 2.5),
            ("Zumos", 2.5), ("Yogur", 2.0), ("Pan", 2.0), ("Azucar y edulcorante", 2.0),
            ("Fruta", 1.2), ("Huevos", 1.2),
        ),
        leak=0.15,
    ),
    Mission(
        "limpieza_hogar",
        base_share=0.09,
        size_mean=3.5,
        size_dispersion=3.0,
        core=(
            ("Detergente", 3.0), ("Suavizante", 3.0), ("Lavavajillas", 3.0),
            ("Lejia y limpiadores", 3.0), ("Limpiacristales", 3.0), ("Bolsas de basura", 3.0),
            ("Papel higienico", 2.5), ("Gel de ducha", 2.0), ("Champu", 2.0),
            ("Acondicionador", 2.0), ("Pasta de dientes", 2.0), ("Desodorante", 2.0),
            ("Higiene femenina", 1.5),
        ),
        leak=0.15,
    ),
    Mission(
        "cena_aperitivo",
        base_share=0.15,
        size_mean=4.0,
        size_dispersion=3.0,
        core=(
            ("Pizza congelada", 3.0), ("Precocinados congelados", 3.0),
            ("Snacks y aperitivos", 3.0), ("Cerveza", 3.0), ("Vino", 2.5), ("Queso", 2.5),
            ("Embutido y fiambre", 2.5), ("Refrescos", 2.5),
            ("Palomitas de microondas", 2.5), ("Pan", 2.0), ("Helados", 2.0),
            ("Conservas de pescado", 1.5),
        ),
        leak=0.2,
    ),
    Mission(
        "bebe",
        base_share=0.22,  # solo en hogares con bebe
        size_mean=3.0,
        size_dispersion=3.0,
        core=(
            ("Panales", 4.0), ("Toallitas humedas", 4.0), ("Leche infantil", 4.0),
            ("Potitos", 4.0), ("Fruta", 1.5), ("Yogur", 1.5), ("Leche", 1.2),
            ("Gel de ducha", 1.2),
        ),
        leak=0.2,
    ),
    Mission(
        "fiesta",
        base_share=0.02,
        size_mean=7.0,
        size_dispersion=3.0,
        core=(
            ("Cerveza", 3.5), ("Snacks y aperitivos", 3.5), ("Carne de ternera", 3.0),
            ("Carne de pollo", 3.0), ("Refrescos", 3.0), ("Cava y espumosos", 3.0),
            ("Turron y mazapan", 3.0), ("Marisco", 3.0), ("Vino", 2.5), ("Agua", 2.0),
            ("Pan", 2.0), ("Embutido y fiambre", 2.0), ("Queso", 2.0), ("Helados", 2.0),
        ),
        leak=0.2,
        month_factor=((6, 3.0), (7, 3.0), (8, 3.0), (9, 3.0), (12, 4.0)),
    ),
)

MISSION_NAMES: tuple[str, ...] = tuple(m.name for m in MISSIONS)

# Ajustes de la probabilidad de mision (ver `mission_affinity` en el generador).
MISSION_DIRICHLET_CONCENTRATION = 20.0
WEEKLY_SHOP_WEEKEND_FACTOR = 1.6  # viernes y sabado
WEEKLY_SHOP_ONLINE_FACTOR = 1.5  # app y web

# Tope de categorias distintas por cesta (antes: 20 lineas, con una por categoria).
MAX_CATEGORIES_PER_BASKET = 50


def mission_profiles() -> "list[list[float]]":
    """Matriz mision x categoria con el multiplicador de cada categoria en cada mision."""
    idx = category_index()
    rows = []
    for mission in MISSIONS:
        row = [mission.leak] * len(CATEGORIES)
        if mission.name == "compra_semanal":
            for i, c in enumerate(CATEGORIES):
                if c.department in _STOCK_UP_DEPARTMENTS:
                    row[i] = 1.3
                if c.name in _TREATS:
                    row[i] = 0.8
        for name, mult in mission.core:
            row[idx[name]] = mult
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# Varias referencias por categoria y marca blanca (Fase 8, punto M2)
# --------------------------------------------------------------------------------------
# Probabilidad de que, al salir una categoria de la banda de exploracion, entre ademas una
# segunda referencia distinta de la misma categoria. En las de habito es 0.
SECOND_REFERENCE_PROB = 0.12

# Dispersion del multiplicador de marca blanca por cliente: rho ~ LogNormal(0, sigma).
PRIVATE_LABEL_PROPENSITY_SIGMA = 0.8


def second_reference_prob(category: "Category") -> float:
    """Probabilidad de segunda referencia en la cesta para una categoria."""
    return SECOND_REFERENCE_PROB if loyalty_band(category) == "exploracion" else 0.0


# --------------------------------------------------------------------------------------
# Estacionalidad (tabla de DATA_SPEC.md, seccion "Estacionalidad (detalle)")
# --------------------------------------------------------------------------------------
# Multiplicador aplicado a la probabilidad base de la categoria durante los meses pico.
# Fuera de esos meses el multiplicador es 1.0.
SEASONALITY: tuple[tuple[str, tuple[int, ...], float], ...] = (
    ("Turron y mazapan", (12,), 8.0),
    ("Marisco", (12,), 3.0),
    ("Cava y espumosos", (12,), 5.0),
    ("Helados", (6, 7, 8), 4.0),
    ("Protector solar", (6, 7, 8), 6.0),
    ("Torrijas y bolleria de Cuaresma", (3, 4), 5.0),
    ("Bacalao", (3, 4), 3.0),
    ("Chocolate y huevos de Pascua", (3, 4), 3.0),
    ("Sopas y caldos", (11, 12, 1, 2), 1.8),
)


# La columna "Lift objetivo" de DATA_SPEC.md es el lift OBSERVADO que se quiere medir a
# nivel de cesta: P(asociada | disparadora) / P(asociada). El multiplicador que aplica el
# generador tiene que ser mayor que ese objetivo, porque solo afecta a los articulos que
# se anaden DESPUES de la categoria disparadora (de media, media cesta) y porque compite
# con el resto de pesos al normalizar. AFFINITY_CALIBRATION cubre esa dilucion:
#
#     multiplicador aplicado = 1 + (lift objetivo - 1) * AFFINITY_CALIBRATION
#
# El valor se ajusto empiricamente en la Fase 1 para que el lift medido por
# `data_generation/verify_dataset.py` aterrizara en el objetivo de la tabla. Desde la
# Fase 8 no se toca: con misiones y cola larga de tamano, el lift crudo ya no depende solo
# del par, y el objetivo pasa a ser la fuerza nominal del par (DATA_SPEC.md, "Afinidad
# de cesta").
AFFINITY_CALIBRATION = 3.8


def applied_affinity_lift(target_lift: float) -> float:
    """Multiplicador interno que hay que aplicar para observar `target_lift` en la cesta."""
    return 1.0 + (target_lift - 1.0) * AFFINITY_CALIBRATION


def interaction_matrix() -> "list[list[float]]":
    """Matriz categoria x categoria que se aplica al peso cuando sale una categoria.

    Fila = categoria que acaba de entrar en la cesta; columna = categoria afectada. Recoge
    los complementos (`applied_affinity_lift`, mayor que 1) y los sustitutos (factor del
    grupo, menor que 1). Es lo unico que el generador hace entre sorteo y sorteo, y lo que
    reproduce el oraculo del recomendador.
    """
    idx = category_index()
    n = len(CATEGORIES)
    m = [[1.0] * n for _ in range(n)]
    for trigger, associated, lift, _motive in COMPLEMENT_PAIRS:
        m[idx[trigger]][idx[associated]] *= applied_affinity_lift(lift)
    for _name, members, factor in SUBSTITUTION_GROUPS:
        for a in members:
            for b in members:
                if a != b:
                    m[idx[a]][idx[b]] *= factor
    return m


# --------------------------------------------------------------------------------------
# Otros parametros de dominio
# --------------------------------------------------------------------------------------
# Uplift de promocion: factor que multiplica la probabilidad de que un producto entre en
# la cesta mientras tiene una promocion activa (DATA_SPEC.md, "Uplift de promocion").
PROMO_UPLIFT = 3.0

# Prevalencia de los gates de hogar.
GATE_PREVALENCE = {"baby": 0.18, "pet": 0.30, "alcohol": 0.70}

LOYALTY_TIERS = ("bronze", "silver", "gold")
LOYALTY_TIER_PROBS = (0.55, 0.32, 0.13)

CHANNELS = ("app", "web", "store")
CHANNEL_PROBS = (0.30, 0.25, 0.45)

DEVICE_TYPES = ("mobile", "desktop", "tablet")
DEVICE_TYPE_PROBS = (0.62, 0.30, 0.08)

PROMO_TYPES = ("2x1", "discount_pct", "coupon")
PROMO_TYPE_PROBS = (0.25, 0.55, 0.20)

# Ciudades espanolas simuladas (peso ~ tamano relativo). No corresponden a ningun
# retailer real: solo dan variedad geografica al dataset.
CITIES: tuple[tuple[str, float], ...] = (
    ("Madrid", 18.0),
    ("Barcelona", 14.0),
    ("Valencia", 8.0),
    ("Sevilla", 7.0),
    ("Zaragoza", 4.5),
    ("Malaga", 4.5),
    ("Murcia", 3.5),
    ("Palma", 3.0),
    ("Las Palmas de Gran Canaria", 3.0),
    ("Bilbao", 3.0),
    ("Alicante", 2.8),
    ("Cordoba", 2.5),
    ("Valladolid", 2.4),
    ("Vigo", 2.4),
    ("Gijon", 2.0),
    ("A Coruna", 2.0),
    ("Granada", 2.0),
    ("Vitoria-Gasteiz", 1.9),
    ("Santa Cruz de Tenerife", 1.9),
    ("Pamplona", 1.8),
    ("Almeria", 1.7),
    ("Donostia-San Sebastian", 1.6),
    ("Santander", 1.5),
    ("Toledo", 1.5),
    ("Salamanca", 1.5),
    ("Badajoz", 1.5),
    ("Logrono", 1.3),
)

COUNTRY = "Espana"

# Marcas de distribuidor (marca blanca). El resto de marcas las genera Faker.
PRIVATE_LABEL_BRANDS: tuple[str, ...] = (
    "Marca Blanca Basico",
    "Marca Blanca Seleccion",
    "Marca Blanca Bio",
)


def category_index() -> dict[str, int]:
    """Devuelve el mapa nombre de categoria -> posicion en CATEGORIES."""
    return {c.name: i for i, c in enumerate(CATEGORIES)}


def loyalty_band(category: Category) -> str:
    """Devuelve "habito" o "exploracion" segun donde caiga la lealtad de la categoria."""
    lo, hi = HABIT_LOYALTY_BAND
    return "habito" if lo <= category.loyalty <= hi else "exploracion"


def validate_catalog() -> None:
    """Comprueba la coherencia interna del catalogo.

    Falla en import si una categoria referenciada por la tabla de afinidad o de
    estacionalidad de DATA_SPEC.md no existe en CATEGORIES, que es el error mas facil
    de cometer al tocar este fichero.
    """
    names = {c.name for c in CATEGORIES}
    if len(names) != len(CATEGORIES):
        raise ValueError("Hay nombres de categoria duplicados en CATEGORIES")

    departments = {c.department for c in CATEGORIES}
    missing_dep = departments - set(DEPARTMENTS)
    if missing_dep:
        raise ValueError(f"Departamentos fuera de DEPARTMENTS: {sorted(missing_dep)}")

    for trigger, associated, _lift in AFFINITY_PAIRS:
        for name in (trigger, associated):
            if name not in names:
                raise ValueError(f"AFFINITY_PAIRS referencia categoria inexistente: {name!r}")

    for c in CATEGORIES:
        lo_h, hi_h = HABIT_LOYALTY_BAND
        lo_e, hi_e = EXPLORATORY_LOYALTY_BAND
        if not (lo_h <= c.loyalty <= hi_h or lo_e <= c.loyalty <= hi_e):
            raise ValueError(
                f"{c.name!r}: loyalty={c.loyalty} fuera de las bandas declaradas "
                f"{EXPLORATORY_LOYALTY_BAND} (exploracion) y {HABIT_LOYALTY_BAND} (habito)"
            )

    for trigger, associated, lift, _motive in COMPLEMENT_PAIRS:
        for name in (trigger, associated):
            if name not in names:
                raise ValueError(f"COMPLEMENT_PAIRS referencia categoria inexistente: {name!r}")
        if trigger == associated or lift <= 1.0:
            raise ValueError(f"Complemento invalido: {trigger!r} -> {associated!r} ({lift})")
    pairs = [(t, a) for t, a, _, _ in COMPLEMENT_PAIRS]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Hay pares repetidos en COMPLEMENT_PAIRS")

    for group, members, factor in SUBSTITUTION_GROUPS:
        if len(members) < 2 or not 0.0 < factor < 1.0:
            raise ValueError(f"Grupo de sustitucion invalido: {group!r}")
        for name in members:
            if name not in names:
                raise ValueError(f"SUBSTITUTION_GROUPS referencia categoria inexistente: {name!r}")

    for mission in MISSIONS:
        if mission.leak <= 0 or mission.size_mean <= 0 or mission.size_dispersion <= 0:
            raise ValueError(f"Mision {mission.name!r}: leak, media y dispersion deben ser > 0")
        for name, mult in mission.core:
            if name not in names or mult <= 0:
                raise ValueError(f"Mision {mission.name!r}: categoria o peso invalido {name!r}")
    if len(set(MISSION_NAMES)) != len(MISSIONS):
        raise ValueError("Hay misiones con el mismo nombre")

    for name, months, _mult in SEASONALITY:
        if name not in names:
            raise ValueError(f"SEASONALITY referencia categoria inexistente: {name!r}")
        for month in months:
            if not 1 <= month <= 12:
                raise ValueError(f"Mes invalido {month} en SEASONALITY para {name!r}")


validate_catalog()

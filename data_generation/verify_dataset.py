"""Verificacion del dataset generado.

Recalcula, leyendo los CSV de `data/raw`, los patrones que el generador dice haber
inyectado: afinidad de cesta, estacionalidad, uplift de promocion, ciclo de reposicion,
senal de churn y problemas de calidad. Cualquier cifra que aparezca en el README del
proyecto debe salir de aqui, no de un notebook (ver CLAUDE.md, "Splits y evaluacion").

Uso:
    python -m data_generation.verify_dataset [--data data/raw] [--json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_generation import catalog as cat
from data_generation.generate_dataset import TABLE_ORDER


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Carga las 7 tablas desde CSV con los tipos utiles ya resueltos."""
    # low_memory=False: las columnas con nulos (store_id, basket_id de sesion) llegan
    # con tipos mixtos por chunk si se deja el lector por defecto.
    tables = {
        name: pd.read_csv(data_dir / f"{name}.csv", low_memory=False)
        for name in TABLE_ORDER
    }
    for name, col in (
        ("baskets", "basket_date"),
        ("sessions", "session_date"),
        ("session_events", "event_timestamp"),
    ):
        tables[name][col] = pd.to_datetime(tables[name][col])
    tables["customers"]["signup_date"] = pd.to_datetime(tables["customers"]["signup_date"])
    for col in ("start_date", "end_date"):
        tables["promotions"][col] = pd.to_datetime(tables["promotions"][col])
    # `category` llega sucia a proposito; para medir patrones se normaliza igual que
    # hara el ETL de la Fase 2.
    tables["products"]["category_clean"] = _clean_category(tables["products"]["category"])
    return tables


def _clean_category(series: pd.Series) -> pd.Series:
    """Normaliza los nombres de categoria ensuciados por el generador."""
    canonical = {c.name.strip().lower().replace("  ", " "): c.name for c in cat.CATEGORIES}
    key = series.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return key.map(canonical)


def volumes(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Filas por tabla."""
    return {name: int(len(df)) for name, df in tables.items()}


def basket_affinity_lift(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Lift observado de cada par de afinidad de `DATA_SPEC.md`.

    lift = P(categoria asociada en la cesta | categoria disparadora en la cesta)
           / P(categoria asociada en la cesta)
    """
    items = tables["basket_items"].merge(
        tables["products"][["product_id", "category_clean"]], on="product_id", how="left"
    )
    # Una cesta "contiene" una categoria si tiene al menos una linea positiva de ella.
    items = items[items["quantity"] > 0]
    pairs = items[["basket_id", "category_clean"]].drop_duplicates()
    n_baskets = pairs["basket_id"].nunique()

    by_cat = {c: set(g) for c, g in pairs.groupby("category_clean")["basket_id"]}
    out: dict[str, float] = {}
    for trigger, associated, target in cat.AFFINITY_PAIRS:
        trig = by_cat.get(trigger, set())
        assoc = by_cat.get(associated, set())
        if not trig or not assoc:
            out[f"{trigger} -> {associated}"] = float("nan")
            continue
        p_assoc = len(assoc) / n_baskets
        p_assoc_given_trig = len(trig & assoc) / len(trig)
        out[f"{trigger} -> {associated}"] = round(p_assoc_given_trig / p_assoc, 2)
        out[f"{trigger} -> {associated} (objetivo)"] = target
    return out


def seasonality_ratio(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Ratio de intensidad mes pico / mes valle para cada categoria estacional.

    Se compara la cuota de lineas de la categoria sobre el total de lineas del mes, para
    neutralizar que en diciembre simplemente se compra mas de todo.
    """
    items = tables["basket_items"].merge(
        tables["products"][["product_id", "category_clean"]], on="product_id", how="left"
    )
    items = items[items["quantity"] > 0].merge(
        tables["baskets"][["basket_id", "basket_date"]], on="basket_id", how="left"
    )
    items["month"] = items["basket_date"].dt.month
    lines_per_month = items.groupby("month").size()

    out: dict[str, float] = {}
    for name, months, target in cat.SEASONALITY:
        sub = items[items["category_clean"] == name].groupby("month").size()
        share = (sub / lines_per_month).fillna(0.0)
        peak = share.reindex(list(months)).mean()
        off = share.drop(index=list(months), errors="ignore").mean()
        out[name] = round(float(peak / off), 2) if off > 0 else float("nan")
        out[f"{name} (objetivo)"] = target
    return out


def promotion_uplift(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Uplift observado: ritmo diario de venta del producto con promo vs sin promo.

    Se mide solo sobre los productos que han tenido alguna promocion, comparando las
    unidades por dia dentro de su ventana promocional con las de fuera.
    """
    items = tables["basket_items"].merge(
        tables["baskets"][["basket_id", "basket_date"]], on="basket_id", how="left"
    )
    items = items[items["quantity"] > 0]
    items["day"] = items["basket_date"].dt.normalize()

    period_days = (
        pd.Timestamp(cat.PERIOD_END) - pd.Timestamp(cat.PERIOD_START)
    ).days + 1
    lines_per_product_day = items.groupby(["product_id", "day"]).size()

    on_rate, off_rate = [], []
    for _, promo in tables["promotions"].iterrows():
        pid = promo["product_id"]
        try:
            sub = lines_per_product_day.loc[pid]
        except KeyError:
            continue
        window = (sub.index >= promo["start_date"]) & (sub.index <= promo["end_date"])
        n_on = int((promo["end_date"] - promo["start_date"]).days) + 1
        n_off = period_days - n_on
        if n_off <= 0:
            continue
        on_rate.append(sub[window].sum() / n_on)
        off_rate.append(sub[~window].sum() / n_off)

    if not on_rate:
        return {"promo_uplift_observado": float("nan"), "promo_uplift_objetivo": cat.PROMO_UPLIFT}
    # Ratio de totales: evita que los productos de cola larga con ventas casi nulas
    # dominen la media de ratios individuales.
    return {
        "promo_uplift_observado": round(float(np.sum(on_rate) / np.sum(off_rate)), 2),
        "promo_uplift_objetivo": cat.PROMO_UPLIFT,
        "promociones_medidas": len(on_rate),
    }


def repurchase_cycles(tables: dict[str, pd.DataFrame], top_n: int = 8) -> dict[str, float]:
    """Intervalo mediano observado entre compras de una misma categoria, por cliente.

    Se compara con `typical_repurchase_days` de la categoria. No tienen que coincidir
    exactamente: el intervalo observado esta escalado por el tamano del hogar y truncado
    por la frecuencia de visita del cliente, pero el orden debe conservarse (la leche
    mucho mas corto que el detergente).
    """
    items = tables["basket_items"][tables["basket_items"]["quantity"] > 0]
    items = items.merge(
        tables["products"][["product_id", "category_clean", "typical_repurchase_days"]],
        on="product_id",
        how="left",
    ).merge(
        tables["baskets"][["basket_id", "customer_id", "basket_date"]],
        on="basket_id",
        how="left",
    )
    items = items.dropna(subset=["customer_id"])
    items["day"] = items["basket_date"].dt.normalize()

    visits = (
        items[["customer_id", "category_clean", "day"]]
        .drop_duplicates()
        .sort_values(["customer_id", "category_clean", "day"], kind="stable")
    )
    visits["gap"] = visits.groupby(["customer_id", "category_clean"])["day"].diff().dt.days
    gaps = visits.dropna(subset=["gap"])

    # El intervalo observado esta acotado por abajo por la frecuencia de visita del
    # cliente: quien pisa la tienda cada 7 semanas no puede comprar leche cada 6 dias.
    # Por eso se mide tambien sobre el decil de clientes mas frecuentes, donde el ciclo
    # real de la categoria si es observable.
    visits_per_customer = (
        items.groupby("customer_id")["day"].nunique().sort_values(ascending=False)
    )
    heavy = set(visits_per_customer.head(max(1, len(visits_per_customer) // 10)).index)
    gaps_heavy = gaps[gaps["customer_id"].isin(heavy)]

    spec = {c.name: c.repurchase_days for c in cat.CATEGORIES}
    counts = gaps.groupby("category_clean").size().sort_values(ascending=False)
    med_all = gaps.groupby("category_clean")["gap"].median()
    med_heavy = gaps_heavy.groupby("category_clean")["gap"].median()

    out: dict[str, float] = {}
    for name in counts.head(top_n).index:
        out[f"{name} (spec)"] = spec[name]
        out[f"{name} (obs. decil frecuente)"] = round(float(med_heavy.get(name, float("nan"))), 1)
        out[f"{name} (obs. global)"] = round(float(med_all[name]), 1)

    # Correlacion de rangos entre el ciclo de la spec y el observado: la senal global.
    ref = pd.Series({k: v for k, v in spec.items() if k in med_all.index})
    out["correlacion_rangos_spec_vs_observado"] = round(
        float(med_all.reindex(ref.index).corr(ref, method="spearman")), 3
    )
    out["visitas_mediana_por_cliente"] = round(float(visits_per_customer.median()), 1)
    out["visitas_mediana_decil_frecuente"] = round(
        float(visits_per_customer.head(max(1, len(visits_per_customer) // 10)).median()), 1
    )
    return out


def sku_loyalty(
    tables: dict[str, pd.DataFrame], min_purchases: int = 3
) -> dict[str, float]:
    """Mide si el cliente repite referencia dentro de una categoria (fidelidad de marca).

    Es la contrapartida de `repurchase_cycles`: aquel mide *cuando* vuelve el cliente a
    una categoria, este mide *que* se lleva cuando vuelve. Era el techo del recomendador
    de SKU antes de la Fase 7a (ver el hallazgo de la Fase 3 en `ROADMAP.md`): con ~24
    referencias por categoria elegidas casi al azar, un cliente con tres o mas compras en
    una categoria se llevaba 0,850 referencias distintas por compra --es decir, casi nunca
    repetia-- frente a 0,440 con el catalogo curado y la fidelidad de marca de la Fase 7a.
    (El 0,86 que cita la Fase 3 es esta misma cifra medida a ojo en su momento.)

    Se cuenta una compra por cesta (no por linea: los duplicados inyectados como problema
    de calidad no pueden contar dos veces) y solo entran los pares cliente-categoria con
    al menos `min_purchases` compras, que son los unicos donde "repetir" significa algo.

    Returns:
        `referencias_distintas_por_compra` (1.0 = nunca repite; cuanto mas bajo, mas
        fiel), `cuota_de_la_referencia_favorita` (que parte de sus compras de la
        categoria se lleva su referencia mas comprada) y el desglose por banda de
        lealtad declarada en el catalogo.
    """
    items = tables["basket_items"][["basket_id", "product_id"]]
    prod = tables["products"][["product_id", "category_clean"]]
    bask = tables["baskets"][["basket_id", "customer_id"]].dropna(subset=["customer_id"])

    df = (
        items.merge(prod, on="product_id")
        .merge(bask, on="basket_id")
        .drop_duplicates(["customer_id", "category_clean", "basket_id", "product_id"])
    )

    # Tras el `drop_duplicates`, el recuento por producto es "en cuantas cestas se lo
    # llevo". Las compras se cuentan por cesta aparte: desde la Fase 8 una cesta puede
    # llevar dos referencias de una categoria de exploracion, y es una sola compra.
    per_product = (
        df.groupby(["customer_id", "category_clean", "product_id"], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    stats = (
        per_product.groupby(["customer_id", "category_clean"], sort=False)["n"]
        .agg(n_referencias="size", favorita="max")
        .join(
            df.groupby(["customer_id", "category_clean"], sort=False)["basket_id"]
            .nunique()
            .rename("n_compras")
        )
        .reset_index()
    )
    stats = stats[stats["n_compras"] >= min_purchases]
    if stats.empty:
        return {"pares_cliente_categoria": 0}

    stats["distintas_por_compra"] = stats["n_referencias"] / stats["n_compras"]
    stats["cuota_favorita"] = stats["favorita"] / stats["n_compras"]

    bands = {c.name: cat.loyalty_band(c) for c in cat.CATEGORIES}
    stats["banda"] = stats["category_clean"].map(bands)
    out = {
        "pares_cliente_categoria": int(len(stats)),
        "referencias_distintas_por_compra": float(stats["distintas_por_compra"].mean()),
        "cuota_de_la_referencia_favorita": float(stats["cuota_favorita"].mean()),
    }
    for banda, sub in stats.groupby("banda"):
        out[f"referencias_distintas_por_compra__{banda}"] = float(
            sub["distintas_por_compra"].mean()
        )
        out[f"cuota_de_la_referencia_favorita__{banda}"] = float(sub["cuota_favorita"].mean())
    return out


def _basket_lines(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Lineas positivas, una por `(cesta, producto)`, con su categoria canonica.

    Los duplicados y las cantidades negativas son problemas de calidad inyectados: aqui
    se mide la forma de la cesta que decidio el generador, no la suciedad.
    """
    items = tables["basket_items"]
    items = items.loc[items["quantity"] > 0, ["basket_id", "product_id"]].drop_duplicates()
    return items.merge(
        tables["products"][["product_id", "category_clean", "is_private_label"]],
        on="product_id",
        how="left",
    )


# Umbrales del tamano de cesta que se reportan. 20 era el tope duro antes de la Fase 8.
SIZE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def basket_size(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Forma de la distribucion del tamano de cesta (punto M1 del diagnostico).

    Cuenta lineas distintas por cesta y, aparte, categorias distintas por cesta. Una
    Poisson tiene coeficiente de variacion `1/sqrt(media)`; una cesta real de gran
    consumo mezcla reposiciones de 1-3 articulos con compras semanales de 30-60 y queda
    muy por encima.
    """
    lines = _basket_lines(tables)
    n_lines = lines.groupby("basket_id").size()
    n_cats = lines.groupby("basket_id")["category_clean"].nunique()
    mean = float(n_lines.mean())
    out: dict[str, float] = {
        "cestas": int(len(n_lines)),
        "lineas_media": round(mean, 3),
        "lineas_coef_variacion": round(float(n_lines.std(ddof=0) / mean), 3),
        "coef_variacion_de_una_poisson_con_esa_media": round(float(1 / np.sqrt(mean)), 3),
    }
    for q in SIZE_QUANTILES:
        out[f"lineas_p{int(round(q * 100)):02d}"] = float(n_lines.quantile(q))
    out |= {
        "lineas_max": int(n_lines.max()),
        "cestas_de_1_a_3_lineas_pct": round(float((n_lines <= 3).mean() * 100), 2),
        "cestas_de_15_o_mas_lineas_pct": round(float((n_lines >= 15).mean() * 100), 2),
        "cestas_de_mas_de_20_lineas_pct": round(float((n_lines > 20).mean() * 100), 2),
        "cestas_de_mas_de_40_lineas_pct": round(float((n_lines > 40).mean() * 100), 3),
        "categorias_media": round(float(n_cats.mean()), 3),
        "categorias_p50": float(n_cats.median()),
        "categorias_p99": float(n_cats.quantile(0.99)),
        "categorias_max": int(n_cats.max()),
    }
    return out


def lines_per_category(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Cuantas veces una cesta lleva dos referencias de la misma categoria (punto M2).

    Hasta la Fase 8 el generador ponia exactamente una linea por categoria y cesta; ahora
    permite, con baja probabilidad, una segunda referencia en las categorias de
    exploracion. Se reporta por banda de lealtad: en las de habito tiene que seguir en 0.
    """
    lines = _basket_lines(tables)
    per_cat = lines.groupby(["basket_id", "category_clean"]).size().rename("n").reset_index()
    bands = {c.name: cat.loyalty_band(c) for c in cat.CATEGORIES}
    per_cat["banda"] = per_cat["category_clean"].map(bands)
    out: dict[str, float] = {
        "cestas_con_alguna_categoria_repetida_pct": round(
            float(per_cat.groupby("basket_id")["n"].max().gt(1).mean() * 100), 3
        ),
        "categorias_de_cesta_con_2_o_mas_referencias_pct": round(
            float(per_cat["n"].gt(1).mean() * 100), 3
        ),
        "lineas_por_categoria_de_cesta_max": int(per_cat["n"].max()),
    }
    for banda, sub in per_cat.groupby("banda"):
        out[f"categorias_de_cesta_con_2_o_mas_referencias_pct__{banda}"] = round(
            float(sub["n"].gt(1).mean() * 100), 3
        )
    return out


# Bandas de tamano (categorias distintas) para el lift controlado por tamano.
SIZE_BANDS = (0, 2, 4, 7, 12, 20, 35, 10_000)


def category_cooccurrence(
    tables: dict[str, pd.DataFrame],
    *,
    min_pair_baskets: int = 50,
    min_band_expected: float = 5.0,
    top: int = 15,
) -> dict[str, object]:
    """Estructura global de co-ocurrencia entre categorias (punto A5 del diagnostico).

    `lift` es la misma cuenta que `affinity_category` del ETL (pares ordenados con al
    menos `min_pair_baskets` cestas en comun). El lift crudo mezcla dos cosas: la
    afinidad real y el **tamano** de la cesta (en una compra grande estan casi todas las
    categorias, asi que cualquier par co-ocurre mas de lo que dicta el azar). Por eso se
    reporta tambien el lift **controlado por tamano**: el lift del par dentro de cada banda
    de tamano (`SIZE_BANDS`, en categorias distintas), promediado con el peso de cestas
    de cada banda. Pesa igual una visita pequena que una compra semanal; un estimador
    ponderado por co-ocurrencias (tipo Mantel-Haenszel) quedaria dominado por las compras
    grandes, que es justo donde menos estructura hay. Solo cuentan las bandas donde el
    par tendria al menos `min_band_expected` cestas en comun por azar.

    Dentro de una banda las categorias no son independientes ni siquiera sin afinidad: se
    sortean sin reemplazo, asi que llevar una deja menos hueco para las demas y el lift de
    un par cualquiera cae algo por debajo de 1. Por eso el controlado se divide por la
    mediana de todos los pares (`controlado_*`): 1 es "lo normal para una cesta de ese
    tamano". Las cestas de una sola categoria no entran en el controlado.
    """
    lines = _basket_lines(tables)
    sets = lines[["basket_id", "category_clean"]].drop_duplicates()
    names = [c.name for c in cat.CATEGORIES]
    b_codes, b_ids = pd.factorize(sets["basket_id"])
    c_codes = sets["category_clean"].map({n: i for i, n in enumerate(names)}).to_numpy()
    x = np.zeros((len(b_ids), len(names)), dtype=np.float32)
    x[b_codes, c_codes] = 1.0

    n_total = x.shape[0]
    single = x.sum(axis=0)
    both = x.T @ x
    size = x.sum(axis=1)
    band = np.digitize(size, SIZE_BANDS[1:-1], right=True)
    weighted = np.zeros_like(both, dtype=np.float64)
    weight = np.zeros_like(both, dtype=np.float64)
    # Una cesta de una sola categoria no dice nada de pares: queda fuera del controlado.
    for b in np.unique(band):
        xb = x[(band == b) & (size >= 2)]
        n_b = xb.shape[0]
        if n_b == 0:
            continue
        p = xb.mean(axis=0).astype(np.float64)
        expected = np.outer(p, p)
        ok = expected * n_b >= min_band_expected
        with np.errstate(divide="ignore", invalid="ignore"):
            lift_b = (xb.T @ xb) / n_b / expected
        weighted += np.where(ok, lift_b * n_b, 0.0)
        weight += np.where(ok, n_b, 0.0)

    off_diag = ~np.eye(len(names), dtype=bool)
    valid = off_diag & (both >= min_pair_baskets) & (weight > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = both * n_total / np.outer(single, single)
        lift_ctrl = weighted / weight
    ctrl_median = float(np.median(lift_ctrl[valid]))
    lift_ctrl = lift_ctrl / ctrl_median
    raw = lift[valid]
    ctrl = lift_ctrl[valid]

    def counts(values: np.ndarray, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}pares_con_lift_mayor_1_5": int((values > 1.5).sum()),
            f"{prefix}pares_con_lift_mayor_1_2": int((values > 1.2).sum()),
            f"{prefix}pares_con_lift_mayor_3": int((values > 3.0).sum()),
            f"{prefix}pares_con_lift_menor_0_8": int((values < 0.8).sum()),
            f"{prefix}lift_mediana": round(float(np.median(values)), 3),
            f"{prefix}lift_p90": round(float(np.percentile(values, 90)), 3),
        }

    order = np.argsort(-np.where(valid, lift_ctrl, -np.inf), axis=None)[:top]
    top_pairs = [
        f"{names[i]} -> {names[j]}: lift {lift[i, j]:.2f}, controlado {lift_ctrl[i, j]:.2f}"
        for i, j in zip(*np.unravel_index(order, lift.shape))
    ]
    order_low = np.argsort(np.where(valid, lift_ctrl, np.inf), axis=None)[:top]
    low_pairs = [
        f"{names[i]} -> {names[j]}: lift {lift[i, j]:.2f}, controlado {lift_ctrl[i, j]:.2f}"
        for i, j in zip(*np.unravel_index(order_low, lift.shape))
    ]
    out: dict[str, object] = {
        "cestas": int(n_total),
        "pares_ordenados_posibles": int(off_diag.sum()),
        "pares_con_soporte_minimo": int(valid.sum()),
        **counts(raw, ""),
        "mediana_del_lift_controlado_sin_normalizar": round(ctrl_median, 3),
        **counts(ctrl, "controlado_"),
        "top_por_lift_controlado": top_pairs,
        "bottom_por_lift_controlado": low_pairs,
    }

    idx = {n: i for i, n in enumerate(names)}

    def pair_values(pairs) -> dict[str, float]:
        res = {}
        for a, b in pairs:
            i, j = idx[a], idx[b]
            res[f"{a} -> {b}"] = (
                f"lift {lift[i, j]:.2f}, controlado {lift_ctrl[i, j]:.2f}"
                if both[i, j] >= min_pair_baskets
                else "sin soporte"
            )
        return res

    complements = getattr(cat, "COMPLEMENT_PAIRS", None)
    if complements is None:
        complements = tuple((t, a) for t, a, _ in cat.AFFINITY_PAIRS)
    else:
        complements = tuple((t, a) for t, a, _, _ in complements)
    out["complementarios_declarados"] = pair_values(complements)
    substitutes = [
        (a, b)
        for _, members, _ in getattr(cat, "SUBSTITUTION_GROUPS", ())
        for a in members
        for b in members
        if a < b
    ]
    if substitutes:
        values = np.array([lift_ctrl[idx[a], idx[b]] for a, b in substitutes])
        raw_values = np.array([lift[idx[a], idx[b]] for a, b in substitutes])
        out["sustitutos_declarados_lift_mediana"] = round(float(np.median(raw_values)), 3)
        out["sustitutos_declarados_lift_controlado_mediana"] = round(float(np.median(values)), 3)
        out["sustitutos_declarados_con_lift_controlado_menor_1"] = (
            f"{int((values < 1).sum())} de {len(values)}"
        )
        out["sustitutos_declarados"] = pair_values(substitutes)
    else:
        # Antes de la Fase 8 no habia grupos: se miden los mismos pares que despues, para
        # poder comparar el antes y el despues.
        out["sustitutos_referencia"] = pair_values(
            (("Agua", "Refrescos"), ("Carne de pollo", "Carne de ternera"),
             ("Lavavajillas", "Lejia y limpiadores"))
        )
    return out


def private_label_propensity(
    tables: dict[str, pd.DataFrame], min_lines: int = 40
) -> dict[str, float]:
    """Heterogeneidad entre clientes en la cuota de marca blanca (punto M2).

    Si todos los clientes tuvieran la misma propension, la cuota de cada uno solo
    variaria por azar (binomial) y por su mezcla de categorias. `ratio_varianza` compara
    la varianza observada entre clientes con la que daria ese azar, contando la mezcla
    de categorias de cada cliente: 1 es "sin propension propia", y cuanto mas alto, mas
    se diferencia el cliente que busca marca blanca del que no.
    """
    lines = _basket_lines(tables).merge(
        tables["baskets"][["basket_id", "customer_id"]].dropna(), on="basket_id"
    )
    lines["pl"] = lines["is_private_label"].astype(float)
    cat_share = lines.groupby("category_clean")["pl"].mean()
    lines["p_cat"] = lines["category_clean"].map(cat_share)
    lines["var_cat"] = lines["p_cat"] * (1 - lines["p_cat"])
    per = lines.groupby("customer_id").agg(
        n=("pl", "size"), pl=("pl", "mean"), p_exp=("p_cat", "mean"), var=("var_cat", "sum")
    )
    per = per[per["n"] >= min_lines]
    resid = per["pl"] - per["p_exp"]
    expected_var = float((per["var"] / per["n"] ** 2).mean())
    return {
        "clientes_evaluados": int(len(per)),
        "cuota_marca_blanca_global": round(float(lines["pl"].mean()), 4),
        "cuota_cliente_p10": round(float(per["pl"].quantile(0.10)), 4),
        "cuota_cliente_p50": round(float(per["pl"].quantile(0.50)), 4),
        "cuota_cliente_p90": round(float(per["pl"].quantile(0.90)), 4),
        "ratio_varianza_observada_vs_azar": round(float(resid.var() / expected_var), 2),
    }


def churn_signal(tables: dict[str, pd.DataFrame], window_days: int = 56) -> dict[str, float]:
    """Comprueba que la frecuencia y el ticket decaen antes del abandono.

    Para cada cliente se compara su ritmo de compra en la ultima ventana de
    `window_days` dias antes de su ultima compra con su ritmo en todo lo anterior. La
    ultima cesta se excluye del numerador: por construccion siempre cae en la ventana
    reciente, y contarla haria que hasta un cliente activo pareciese acelerar.

    En los clientes con `churn_label = 1` el ratio debe quedar claramente por debajo del
    de los activos: eso es la rampa de 6-8 semanas que inyecta el generador.
    """
    b = tables["baskets"].dropna(subset=["customer_id"]).copy()
    b = b.merge(tables["customers"][["customer_id", "churn_label"]], on="customer_id")
    b["day"] = b["basket_date"].dt.normalize()

    grp = b.groupby("customer_id")["day"]
    b["last"] = grp.transform("max")
    b["first"] = grp.transform("min")
    b["n_baskets"] = b.groupby("customer_id")["basket_id"].transform("size")
    b["span"] = (b["last"] - b["first"]).dt.days

    # Se exige historia suficiente para que la comparacion tenga sentido.
    b = b[(b["span"] >= 180) & (b["n_baskets"] >= 6)]
    if b.empty:
        return {"clientes_evaluados": 0}

    days_to_last = (b["last"] - b["day"]).dt.days
    recent = b[(days_to_last > 0) & (days_to_last <= window_days)]
    base = b[days_to_last > window_days]

    keys = ["customer_id", "churn_label"]
    n_recent = recent.groupby(keys).size().rename("n_recent")
    n_base = base.groupby(keys).size().rename("n_base")
    span = base.groupby(keys)["span"].max().rename("span")
    amt_recent = recent.groupby(keys)["total_amount"].median().rename("amt_recent")
    amt_base = base.groupby(keys)["total_amount"].median().rename("amt_base")

    df = pd.concat([n_recent, n_base, span, amt_recent, amt_base], axis=1)
    df["n_recent"] = df["n_recent"].fillna(0.0)
    df = df[(df["n_base"] >= 3) & (df["span"] > window_days)]

    df["rate_recent"] = df["n_recent"] / window_days
    df["rate_base"] = df["n_base"] / (df["span"] - window_days)

    out: dict[str, float] = {}
    for label, tag in ((True, "churn"), (False, "activo")):
        sub = df.xs(label, level="churn_label")
        out[f"ratio_frecuencia_{tag}"] = round(
            float((sub["rate_recent"] / sub["rate_base"]).median()), 3
        )
        tick = sub.dropna(subset=["amt_recent", "amt_base"])
        out[f"ratio_ticket_{tag}"] = round(
            float((tick["amt_recent"] / tick["amt_base"]).median()), 3
        )
        out[f"clientes_evaluados_{tag}"] = int(len(sub))
    out["clientes_churn_pct"] = round(
        float(tables["customers"]["churn_label"].mean() * 100), 2
    )
    return out


def quality_issues(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Cuenta los problemas de calidad deliberados que el ETL tendra que limpiar."""
    items = tables["basket_items"]
    products = tables["products"]
    baskets = tables["baskets"]
    canonical = {c.name for c in cat.CATEGORIES}

    amounts = baskets["total_amount"]
    q1, q3 = amounts.quantile([0.25, 0.75])
    outlier_cut = q3 + 3 * (q3 - q1)

    return {
        "lineas_duplicadas_pct": round(float(items.duplicated().mean() * 100), 3),
        "cantidad_negativa_pct": round(float((items["quantity"] < 0).mean() * 100), 3),
        "categorias_mal_escritas_pct": round(
            float((~products["category"].isin(canonical)).mean() * 100), 3
        ),
        "marca_nula_pct": round(float(products["brand"].isna().mean() * 100), 3),
        "ciudad_nula_pct": round(float(tables["customers"]["city"].isna().mean() * 100), 3),
        "cestas_anonimas_pct": round(float(baskets["customer_id"].isna().mean() * 100), 3),
        "outliers_importe_pct": round(float((amounts > outlier_cut).mean() * 100), 3),
        "alta_posterior_a_primera_compra": int(
            baskets.dropna(subset=["customer_id"])
            .merge(tables["customers"][["customer_id", "signup_date"]], on="customer_id")
            .assign(bad=lambda d: d["basket_date"] < d["signup_date"])
            .groupby("customer_id")["bad"]
            .any()
            .sum()
        ),
    }


def referential_integrity(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Cuenta claves ajenas rotas. Todas deben ser 0: la suciedad es de valores, no de FKs."""
    prod = set(tables["products"]["product_id"])
    cust = set(tables["customers"]["customer_id"])
    bask = set(tables["baskets"]["basket_id"])
    sess = set(tables["sessions"]["session_id"])
    promo = set(tables["promotions"]["promotion_id"])
    items = tables["basket_items"]
    return {
        "basket_items.basket_id_huerfano": int((~items["basket_id"].isin(bask)).sum()),
        "basket_items.product_id_huerfano": int((~items["product_id"].isin(prod)).sum()),
        "basket_items.promotion_id_huerfano": int(
            (~items["promotion_id"].dropna().isin(promo)).sum()
        ),
        "baskets.customer_id_huerfano": int(
            (~tables["baskets"]["customer_id"].dropna().isin(cust)).sum()
        ),
        "sessions.basket_id_huerfano": int(
            (~tables["sessions"]["basket_id"].dropna().isin(bask)).sum()
        ),
        "session_events.session_id_huerfano": int(
            (~tables["session_events"]["session_id"].isin(sess)).sum()
        ),
        "promotions.product_id_huerfano": int(
            (~tables["promotions"]["product_id"].isin(prod)).sum()
        ),
    }


def run_all(data_dir: Path) -> dict[str, dict]:
    """Ejecuta todas las verificaciones sobre el dataset de `data_dir`."""
    tables = load_tables(data_dir)
    return {
        "volumenes": volumes(tables),
        "afinidad_de_cesta": basket_affinity_lift(tables),
        "estacionalidad": seasonality_ratio(tables),
        "uplift_promocion": promotion_uplift(tables),
        "ciclos_reposicion": repurchase_cycles(tables),
        "fidelidad_de_sku": sku_loyalty(tables),
        "tamano_de_cesta": basket_size(tables),
        "lineas_por_categoria": lines_per_category(tables),
        "coocurrencia_de_categorias": category_cooccurrence(tables),
        "propension_marca_blanca": private_label_propensity(tables),
        "senal_churn": churn_signal(tables),
        "calidad_del_dato": quality_issues(tables),
        "integridad_referencial": referential_integrity(tables),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/raw", help="Directorio con los CSV.")
    parser.add_argument("--json", action="store_true", help="Salida en JSON.")
    args = parser.parse_args(argv)

    report = run_all(Path(args.data))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=float))
        return
    for section, values in report.items():
        print(f"\n== {section} ==")
        for k, v in values.items():
            print(f"  {k:<55} {v}")


if __name__ == "__main__":
    main()

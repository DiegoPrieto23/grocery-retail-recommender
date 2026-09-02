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

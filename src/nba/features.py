"""Features de propension en un corte, sin mirar ni un dia mas alla.

Dos granos, porque hay dos preguntas distintas:

- **Cliente** (`customer_features`): alimenta el modelo de churn. Lo que importa aqui no
  es el nivel sino la **derivada**: `DATA_SPEC.md` inyecta el abandono como una caida
  progresiva de frecuencia y ticket en las 6-8 semanas previas, asi que un cliente que
  compraba 4 veces al mes y ahora compra 2 es la senal, y un cliente que siempre compro 2
  no lo es. De ahi los cocientes `trend_*`: comparan la ventana reciente con la anterior
  de la misma longitud, que es justo la forma que tiene la rampa del generador.
- **Cliente x categoria** (`category_features`): alimenta el modelo de compra en categoria.
  El nucleo lo aporta `repurchase_features` de la Fase 2 -- la funcion de la Tarea 2, que
  `CHALLENGE.md` dice explicitamente que debe alimentar tambien al NBA.

## Lo que no entra

`churn_label` no es una feature y no puede serlo: se calcula con compras posteriores al
corte (ver `targets`). Tampoco entra `city`, que tiene demasiados niveles para lo que
aporta, ni nada derivado de la ventana de etiquetado.
"""

from __future__ import annotations

import datetime as dt

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from src.etl.repurchase import repurchase_features

# --------------------------------------------------------------------------------------
# Nombres de las features, que son el contrato con el modelo
# --------------------------------------------------------------------------------------
CUSTOMER_FEATURES: tuple[str, ...] = (
    "recency_days",
    "frequency",
    "monetary",
    "avg_ticket",
    "tenure_days",
    "avg_days_between_baskets",
    "n_baskets_28d",
    "n_baskets_prev_28d",
    "n_baskets_90d",
    "spend_28d",
    "spend_prev_28d",
    "spend_90d",
    "trend_baskets",
    "trend_spend",
    "trend_ticket",
    "n_categories",
    "n_products",
    "promo_line_share",
    "household_size_est",
    "signup_tenure_days",
    "loyalty_tier_idx",
    "preferred_channel_idx",
)

CATEGORY_FEATURES: tuple[str, ...] = (
    "cat_days_since_last_purchase",
    "cat_expected_repurchase_days",
    "cat_observed_repurchase_days",
    "cat_stddev_gap_days",
    "cat_typical_repurchase_days",
    "cat_overdue_ratio",
    "cat_due_for_repurchase",
    "cat_n_purchase_days",
    "cat_spend",
    "cat_spend_share",
    "cat_units",
    "cat_popularity_90d",
    "department_idx",
)

# Las que LightGBM debe tratar como categoricas y no como numeros ordenados.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "loyalty_tier_idx",
    "preferred_channel_idx",
    "department_idx",
)

CHURN_FEATURES: tuple[str, ...] = CUSTOMER_FEATURES
PURCHASE_FEATURES: tuple[str, ...] = CUSTOMER_FEATURES + CATEGORY_FEATURES

# Orden fijo de los niveles categoricos: se indexan a mano en vez de con un StringIndexer
# para que el indice no dependa de las frecuencias del corte que toque y sea comparable
# entre entrenamiento y test.
LOYALTY_TIERS: tuple[str, ...] = ("bronze", "silver", "gold")
CHANNELS: tuple[str, ...] = ("app", "web", "store")
DEPARTMENTS: tuple[str, ...] = (
    "Bebe",
    "Bebidas",
    "Congelados",
    "Despensa",
    "Drogueria",
    "Frescos",
    "Higiene",
    "Mascotas",
)


def _day(value: dt.date) -> Column:
    return F.lit(value.isoformat()).cast("date")


def _index(column: str, levels: tuple[str, ...]) -> Column:
    """Indice fijo de un valor categorico; -1 para lo desconocido o nulo."""
    out = F.lit(-1)
    for i, level in enumerate(levels):
        out = F.when(F.col(column) == F.lit(level), F.lit(i)).otherwise(out)
    return out.cast("int")


def _ratio(recent: Column, previous: Column) -> Column:
    """Cociente reciente/anterior acotado, con el caso "antes tampoco compraba" resuelto.

    Si la ventana anterior es cero el cociente no existe; se devuelve 1.0 (sin cambio)
    en vez de nulo o infinito, porque un cliente que no compraba antes y no compra ahora
    no esta cayendo, simplemente es de baja frecuencia -- y esa informacion ya la lleva el
    nivel en `n_baskets_28d`.
    """
    return F.when(previous > 0, recent / previous).otherwise(F.lit(1.0)).cast("double")


def customer_features(
    baskets: DataFrame,
    lines: DataFrame,
    customers: DataFrame,
    cutoff: dt.date,
) -> DataFrame:
    """Una fila por cliente con actividad antes del corte.

    Args:
        baskets: Cestas **ya filtradas** a `basket_day < cutoff` e identificadas.
        lines: Lineas con categoria, tambien filtradas (ver `targets.category_lines`).
        customers: Maestro, para los atributos estables (hogar, tier, canal, alta).
        cutoff: Dia del corte. Es el origen desde el que se miden todas las recencias.
    """
    cut = _day(cutoff)
    w28 = _day(cutoff - dt.timedelta(days=28))
    w56 = _day(cutoff - dt.timedelta(days=56))
    w90 = _day(cutoff - dt.timedelta(days=90))

    def _count_between(lo: Column, hi: Column) -> Column:
        return F.countDistinct(
            F.when((F.col("basket_day") >= lo) & (F.col("basket_day") < hi), F.col("basket_id"))
        ).cast("double")

    def _sum_between(lo: Column, hi: Column) -> Column:
        return F.coalesce(
            F.sum(
                F.when(
                    (F.col("basket_day") >= lo) & (F.col("basket_day") < hi),
                    F.col("total_amount"),
                )
            ),
            F.lit(0.0),
        )

    agg = baskets.groupBy("customer_id").agg(
        F.max("basket_day").alias("last_purchase_date"),
        F.min("basket_day").alias("first_purchase_date"),
        F.countDistinct("basket_id").cast("double").alias("frequency"),
        F.sum("total_amount").alias("monetary"),
        F.avg("total_amount").alias("avg_ticket"),
        _count_between(w28, cut).alias("n_baskets_28d"),
        _count_between(w56, w28).alias("n_baskets_prev_28d"),
        _count_between(w90, cut).alias("n_baskets_90d"),
        _sum_between(w28, cut).alias("spend_28d"),
        _sum_between(w56, w28).alias("spend_prev_28d"),
        _sum_between(w90, cut).alias("spend_90d"),
    )

    agg = (
        agg.withColumn("recency_days", F.datediff(cut, F.col("last_purchase_date")).cast("double"))
        .withColumn("tenure_days", F.datediff(cut, F.col("first_purchase_date")).cast("double"))
        .withColumn(
            "avg_days_between_baskets",
            F.when(
                F.col("frequency") > 1,
                F.col("tenure_days") / (F.col("frequency") - F.lit(1.0)),
            ).cast("double"),
        )
        .withColumn("trend_baskets", _ratio(F.col("n_baskets_28d"), F.col("n_baskets_prev_28d")))
        .withColumn("trend_spend", _ratio(F.col("spend_28d"), F.col("spend_prev_28d")))
    )
    # El ticket medio reciente contra el anterior: separa "compra menos veces" de "compra
    # lo mismo pero llena menos el carro". El generador encoge las dos cosas.
    agg = agg.withColumn(
        "trend_ticket",
        _ratio(
            F.when(F.col("n_baskets_28d") > 0, F.col("spend_28d") / F.col("n_baskets_28d"))
            .otherwise(F.lit(0.0)),
            F.when(
                F.col("n_baskets_prev_28d") > 0,
                F.col("spend_prev_28d") / F.col("n_baskets_prev_28d"),
            ).otherwise(F.lit(0.0)),
        ),
    )

    variety = lines.groupBy("customer_id").agg(
        F.countDistinct("category").cast("double").alias("n_categories"),
        F.countDistinct("product_id").cast("double").alias("n_products"),
        F.avg(F.col("is_promo").cast("double")).alias("promo_line_share"),
    )

    master = customers.select(
        "customer_id",
        F.col("household_size_est").cast("double").alias("household_size_est"),
        F.datediff(cut, F.col("signup_date")).cast("double").alias("signup_tenure_days"),
        _index("loyalty_tier", LOYALTY_TIERS).alias("loyalty_tier_idx"),
        _index("preferred_channel", CHANNELS).alias("preferred_channel_idx"),
    )

    return (
        agg.join(variety, "customer_id", "left")
        .join(master, "customer_id", "inner")
        .select("customer_id", *CUSTOMER_FEATURES)
    )


def category_popularity(lines: DataFrame, cutoff: dt.date) -> DataFrame:
    """Cuota de cada categoria sobre el gasto total de los ultimos 90 dias antes del corte.

    Es la referencia externa al cliente: sirve para que el modelo distinga "esta categoria
    la compra todo el mundo" de "esta la compra solo este cliente".
    """
    w90 = _day(cutoff - dt.timedelta(days=90))
    recent = lines.filter(F.col("basket_day") >= w90)
    total = recent.agg(F.sum("line_amount").alias("t")).collect()[0]["t"] or 1.0
    return recent.groupBy("category").agg(
        (F.sum("line_amount") / F.lit(float(total))).cast("double").alias("cat_popularity_90d")
    )


def category_features(
    baskets: DataFrame,
    basket_items: DataFrame,
    lines: DataFrame,
    products: DataFrame,
    customers: DataFrame,
    cutoff: dt.date,
) -> DataFrame:
    """Una fila por `(customer_id, category)` con el estado del ciclo de recompra.

    El grueso lo aporta `repurchase_features` (Tarea 2), invocada con el corte como
    `reference_date` y con el historico ya recortado, de modo que `days_since_last_purchase`
    y `due_for_repurchase` se midan desde el corte y no desde el final del dataset.
    """
    repurchase = repurchase_features(
        basket_items,
        baskets,
        products,
        customers,
        reference_date=cutoff,
    ).select(
        "customer_id",
        "category",
        F.col("days_since_last_purchase").cast("double").alias("cat_days_since_last_purchase"),
        F.col("expected_repurchase_days").cast("double").alias("cat_expected_repurchase_days"),
        F.col("observed_repurchase_days").cast("double").alias("cat_observed_repurchase_days"),
        F.col("stddev_gap_days").cast("double").alias("cat_stddev_gap_days"),
        F.col("typical_repurchase_days").cast("double").alias("cat_typical_repurchase_days"),
        F.col("overdue_ratio").cast("double").alias("cat_overdue_ratio"),
        F.col("due_for_repurchase").cast("double").alias("cat_due_for_repurchase"),
        F.col("n_purchase_days").cast("double").alias("cat_n_purchase_days"),
    )

    spend = lines.groupBy("customer_id", "category").agg(
        F.sum("line_amount").cast("double").alias("cat_spend"),
        F.sum("quantity").cast("double").alias("cat_units"),
    )
    totals = spend.groupBy("customer_id").agg(F.sum("cat_spend").alias("_total_spend"))
    spend = (
        spend.join(totals, "customer_id", "inner")
        .withColumn(
            "cat_spend_share",
            F.when(F.col("_total_spend") > 0, F.col("cat_spend") / F.col("_total_spend"))
            .otherwise(F.lit(0.0))
            .cast("double"),
        )
        .drop("_total_spend")
    )

    popularity = category_popularity(lines, cutoff)
    department = products.select("category", "department").distinct()

    return (
        repurchase.join(spend, ["customer_id", "category"], "outer")
        .join(popularity, "category", "left")
        .join(department, "category", "left")
        .withColumn("department_idx", _index("department", DEPARTMENTS))
        .select("customer_id", "category", "department", *CATEGORY_FEATURES)
    )

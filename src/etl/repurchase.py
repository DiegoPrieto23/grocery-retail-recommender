"""Ciclo de recompra por cliente y categoria (Tarea 2 de `CHALLENGE.md`).

Dada una tabla con el formato de `basket_items` mas `customers`, devuelve por cliente y
categoria la ultima fecha de compra, la frecuencia media de recompra observada y el flag
`due_for_repurchase`, que marca si ya toca reponer. Alimenta al recomendador (repescar
productos que "tocan") y al NBA.

## Como se decide que toca

El intervalo esperado sale de dos fuentes, por orden de preferencia:

1. **Lo observado en ese cliente y esa categoria.** Media de dias entre compras. Es la
   senal buena, pero exige al menos dos compras previas.
2. **El intervalo tipico de la categoria** (`products.typical_repurchase_days`), ajustado
   por el tamano del hogar. Es el respaldo para clientes nuevos en la categoria.

El ajuste por hogar es lineal y decreciente (`household_base - household_slope * tamano`):
un hogar de 5 personas vacia el pack de leche antes que uno de 1. Los coeficientes por
defecto reproducen el rango de `DATA_SPEC.md` -- factor 1.325 para un hogar de 1 y 0.70
para uno de 6 -- pero son parametros, no constantes: es una hipotesis de dominio, y el ETL
no importa nada del generador de la Fase 1.

## Grano de la fecha

Se trabaja con **dias distintos de compra**, no con lineas de ticket. Dos cestas del mismo
dia con leche son una sola ocasion de compra; contarlas dos veces meteria un intervalo de
0 dias que hundiria la media.
"""

from __future__ import annotations

import datetime as dt

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# Las formulas del ciclo (y sus coeficientes) se comparten con el recomendador y con la
# ruta de serving en pandas: viven en `src/recommender/formulas.py`. Se reexportan aqui
# para no romper a quien las importaba de este modulo.
from src.recommender import formulas as fx
from src.recommender.formulas import HOUSEHOLD_BASE, HOUSEHOLD_SLOPE, MIN_EXPECTED_DAYS  # noqa: F401
from src.recommender.formulas_spark import SPARK_OPS


def _reference_date_column(
    baskets: DataFrame, reference_date: str | dt.date | None
) -> Column:
    """Fecha desde la que se mide si toca reponer. Por defecto, el ultimo dia con compras."""
    if reference_date is not None:
        return F.lit(str(reference_date)).cast("date")
    last_day = baskets.agg(F.max(F.to_date("basket_date"))).collect()[0][0]
    return F.lit(last_day).cast("date")


def category_repurchase_days(products: DataFrame) -> DataFrame:
    """Intervalo tipico de recompra de cada categoria, a partir de sus productos.

    En el catalogo todos los productos de una categoria comparten
    `typical_repurchase_days`, pero se promedia en lugar de tomar uno cualquiera para que
    la funcion siga siendo correcta si algun dia dejan de coincidir.
    """
    return products.groupBy("category").agg(
        F.round(F.avg("typical_repurchase_days")).cast("int").alias("typical_repurchase_days")
    )


def customer_category_purchases(
    basket_items: DataFrame, baskets: DataFrame, products: DataFrame
) -> DataFrame:
    """Dias distintos en los que cada cliente compro cada categoria.

    Descarta las cestas anonimas: sin `customer_id` no hay ciclo de reposicion que medir.
    """
    return (
        basket_items.select("basket_id", "product_id")
        .join(F.broadcast(products.select("product_id", "category")), "product_id")
        .join(
            baskets.filter(F.col("customer_id").isNotNull()).select(
                "basket_id", "customer_id", F.to_date("basket_date").alias("purchase_date")
            ),
            "basket_id",
        )
        .select("customer_id", "category", "purchase_date")
        .distinct()
    )


def repurchase_features(
    basket_items: DataFrame,
    baskets: DataFrame,
    products: DataFrame,
    customers: DataFrame | None = None,
    *,
    reference_date: str | dt.date | None = None,
    tolerance: float = 1.0,
    min_observations: int = 2,
    household_adjustment: bool = True,
    household_base: float = HOUSEHOLD_BASE,
    household_slope: float = HOUSEHOLD_SLOPE,
) -> DataFrame:
    """Devuelve, por cliente y categoria, el estado del ciclo de recompra.

    Args:
        basket_items: Lineas de ticket limpias (`basket_id`, `product_id`, ...).
        baskets: Cabeceras limpias (`basket_id`, `customer_id`, `basket_date`).
        products: Catalogo con `product_id`, `category` y `typical_repurchase_days`.
        customers: Maestro de clientes. Solo hace falta si `household_adjustment` es True,
            para leer `household_size_est`.
        reference_date: Dia desde el que se mide. Por defecto, el ultimo con compras.
        tolerance: Holgura sobre el intervalo esperado antes de dar por vencida la
            recompra. 1.0 = en cuanto se cumple el ciclo; 1.2 = un 20 % de margen.
        min_observations: Compras minimas en la categoria para fiarse del intervalo
            observado. Con menos, se usa el tipico de la categoria.
        household_adjustment: Escalar el intervalo tipico por el tamano del hogar.
        household_base: Termino independiente del ajuste por hogar.
        household_slope: Pendiente del ajuste por hogar.

    Returns:
        Un DataFrame con una fila por `(customer_id, category)` y las columnas:

        - `last_purchase_date`: ultima vez que compro la categoria.
        - `n_purchase_days`: dias distintos con compra de la categoria.
        - `observed_repurchase_days`: media de dias entre compras (nula si no hay
          suficientes observaciones).
        - `typical_repurchase_days`: intervalo tipico de la categoria.
        - `expected_repurchase_days`: el que se usa para decidir (observado si es fiable,
          si no el tipico ajustado por hogar).
        - `days_since_last_purchase`: dias transcurridos hasta `reference_date`.
        - `overdue_ratio`: `days_since_last_purchase / expected_repurchase_days`. Por
          encima de 1 va con retraso; muy por encima, la categoria esta abandonada.
        - `due_for_repurchase`: el flag de la Tarea 2.

    Raises:
        ValueError: Si `tolerance` no es positiva, `min_observations` es menor que 2, o
            se pide ajuste por hogar sin pasar `customers`.
    """
    if tolerance <= 0:
        raise ValueError("tolerance debe ser > 0")
    if min_observations < 2:
        raise ValueError("min_observations debe ser al menos 2: un intervalo necesita 2 compras")
    if household_adjustment and customers is None:
        raise ValueError("household_adjustment=True requiere pasar `customers`")

    ref = _reference_date_column(baskets, reference_date)
    purchases = customer_category_purchases(basket_items, baskets, products)

    # Dias entre compras consecutivas de la misma categoria por el mismo cliente.
    ordered = Window.partitionBy("customer_id", "category").orderBy("purchase_date")
    gaps = purchases.withColumn(
        "gap_days",
        F.datediff("purchase_date", F.lag("purchase_date").over(ordered)),
    )

    summary = gaps.groupBy("customer_id", "category").agg(
        F.max("purchase_date").alias("last_purchase_date"),
        F.min("purchase_date").alias("first_purchase_date"),
        F.count(F.lit(1)).cast("int").alias("n_purchase_days"),
        F.round(F.avg("gap_days"), 2).alias("mean_gap_days"),
        F.round(F.stddev_samp("gap_days"), 2).alias("stddev_gap_days"),
    )

    typical = category_repurchase_days(products)
    enriched = summary.join(F.broadcast(typical), "category", "left")

    if household_adjustment:
        assert customers is not None  # garantizado por la validacion de arriba
        enriched = enriched.join(
            F.broadcast(customers.select("customer_id", "household_size_est")),
            "customer_id",
            "left",
        )
        # Hogar mas grande, ciclo mas corto. Sin dato se asume un hogar de 1.
        household_factor = fx.household_factor(
            F.col("household_size_est"), SPARK_OPS, base=household_base, slope=household_slope
        )
        enriched = enriched.withColumn("household_factor", F.round(household_factor, 4))
    else:
        enriched = enriched.withColumn("household_factor", F.lit(1.0))

    # El intervalo observado manda en cuanto hay evidencia suficiente; si no, el tipico.
    observed = F.when(F.col("n_purchase_days") >= min_observations, F.col("mean_gap_days"))
    expected = fx.expected_repurchase_days(
        F.col("n_purchase_days"),
        F.col("mean_gap_days"),
        F.col("typical_repurchase_days"),
        F.col("household_factor"),
        SPARK_OPS,
        min_observations=min_observations,
    )

    out = (
        enriched.withColumn("reference_date", ref)
        .withColumn("observed_repurchase_days", observed)
        .withColumn("expected_repurchase_days", F.round(expected, 2))
        .withColumn(
            "days_since_last_purchase", F.datediff(ref, F.col("last_purchase_date"))
        )
    )

    return (
        out.withColumn(
            "overdue_ratio",
            F.round(
                F.col("days_since_last_purchase") / F.col("expected_repurchase_days"), 3
            ),
        )
        .withColumn(
            "due_for_repurchase",
            F.col("days_since_last_purchase")
            >= F.col("expected_repurchase_days") * F.lit(tolerance),
        )
        .select(
            "customer_id",
            "category",
            "last_purchase_date",
            "first_purchase_date",
            "n_purchase_days",
            "observed_repurchase_days",
            "stddev_gap_days",
            "typical_repurchase_days",
            "household_factor",
            "expected_repurchase_days",
            "reference_date",
            "days_since_last_purchase",
            "overdue_ratio",
            "due_for_repurchase",
        )
    )

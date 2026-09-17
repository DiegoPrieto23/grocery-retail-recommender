"""Historial del cliente *as-of* el dia de cada cesta (punto A1 del diagnostico).

Hasta ahora `customer_products`, `customer_stats` y `repurchase` se calculaban una sola
vez por ventana, con las cestas anteriores a su inicio (1-sep o 1-nov). Una query del
20-dic no veia lo que el mismo cliente habia comprado el 5, el 12 o el 18 de diciembre, y
el modelo recomendaba como "vencida" una categoria que el cliente acababa de reponer.

Aqui cada query ve **todas las cestas de su cliente con `basket_day` estrictamente
anterior al suyo**, esten dentro o fuera de la ventana:

    query (cliente C, dia D)  x  cestas de C con dia < D   ->  agregados por query

Es un *as-of join* contra la tabla de eventos del cliente, resuelto con un join por
`customer_id` y un filtro de fecha. A esta escala (~30 cestas por cliente) el join son
unos pocos millones de filas, asi que no hace falta una ventana acumulada.

## Por que el corte es el dia y no `cut_ts`

- Es el grano del ciclo de reposicion: se cuentan **dias** de compra distintos.
- Solo las cestas online tienen `cut_ts`; el dia existe para todas.
- La demo recibe un dia, no un instante.

Lo que se pierde es una segunda cesta del mismo cliente el mismo dia, que es rara y cuyo
orden dentro del dia no siempre se conoce. Estrictamente anterior garantiza que la cesta
que se predice nunca entra en su propio historial.

## Lo que sigue congelado por ventana

ALS, popularidad, afinidades y `known_customers` (que define el perfil). Es el patron de
reentrenamiento periodico habitual, y mantener el perfil congelado deja las cifras por
perfil comparables con las anteriores. Un cliente "nuevo" (perfil 1-2) que ya ha comprado
dentro de la ventana si recibe aqui su historial reciente.

La replica en pandas esta en `src/serving/recommend.py` (`asof_history`); las formulas
que comparten viven en `formulas.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.etl.repurchase import category_repurchase_days
from src.recommender import formulas as fx
from src.recommender.formulas_spark import SPARK_OPS


@dataclass
class AsOfHistory:
    """Historial de cada query, todas las tablas con clave `basket_id` (la query).

    Attributes:
        products: `basket_id`, `product_id`, `hist_n_baskets`, `hist_units`,
            `hist_last_day`.
        categories: `basket_id`, `category`, `cat_n_purchase_days`, `cat_last_day`,
            `cat_expected_days` y el ranking personal (`cat_freq_rank`,
            `cat_freq_share`, `cat_due_rank`).
        customer: `basket_id`, `cust_frequency`, `cust_avg_ticket`, `cust_last_day`,
            `cust_n_products`.
    """

    products: DataFrame
    categories: DataFrame
    customer: DataFrame

    def cache(self) -> "AsOfHistory":
        for frame in (self.products, self.categories, self.customer):
            frame.cache().count()
        return self

    def unpersist(self) -> None:
        for frame in (self.products, self.categories, self.customer):
            frame.unpersist()


def asof_history(
    queries: DataFrame,
    baskets: DataFrame,
    basket_items: DataFrame,
    products: DataFrame,
    customers: DataFrame,
) -> AsOfHistory:
    """Agrega, para cada query con cliente, sus cestas anteriores al dia de la query.

    Args:
        queries: `basket_id`, `customer_id` y `basket_day` de cada query.
        baskets: Cabeceras limpias de **todo** el periodo; el filtro de fecha lo pone esta
            funcion, query a query.
        basket_items: Lineas limpias (`basket_id`, `product_id`, `quantity`).
        products: Catalogo con `category` y `typical_repurchase_days`.
        customers: Maestro con `household_size_est`.
    """
    q = queries.filter(F.col("customer_id").isNotNull()).select(
        "basket_id", "customer_id", F.col("basket_day").alias("_query_day")
    )
    past = baskets.filter(F.col("customer_id").isNotNull()).select(
        "customer_id",
        F.col("basket_id").alias("_past_basket"),
        F.col("basket_day").alias("_past_day"),
        "total_amount",
    )
    visits = q.join(past, "customer_id").filter(F.col("_past_day") < F.col("_query_day"))
    lines = visits.select("basket_id", "customer_id", "_past_basket", "_past_day").join(
        basket_items.select(
            F.col("basket_id").alias("_past_basket"), "product_id", "quantity"
        ),
        "_past_basket",
    )

    # --- Cliente: RFM ligero (lo que antes era `features.customer_profile`) ---
    customer = visits.groupBy("basket_id").agg(
        F.countDistinct("_past_basket").cast("double").alias("cust_frequency"),
        F.avg("total_amount").alias("cust_avg_ticket"),
        F.max("_past_day").alias("cust_last_day"),
    ).join(
        lines.groupBy("basket_id").agg(
            F.countDistinct("product_id").cast("double").alias("cust_n_products")
        ),
        "basket_id",
        "left",
    )

    # --- Cliente x producto (lo que antes era `candidates.fit_customer_products`) ---
    per_product = lines.groupBy("basket_id", "product_id").agg(
        F.countDistinct("_past_basket").cast("double").alias("hist_n_baskets"),
        F.sum("quantity").cast("double").alias("hist_units"),
        F.max("_past_day").alias("hist_last_day"),
    )

    # --- Cliente x categoria: el ciclo de reposicion, con las formulas compartidas ---
    purchase_days = (
        lines.join(F.broadcast(products.select("product_id", "category")), "product_id")
        .select("basket_id", "customer_id", "category", "_past_day")
        .distinct()
    )
    per_category = (
        purchase_days.groupBy("basket_id", "customer_id", "category")
        .agg(
            F.count(F.lit(1)).alias("_n"),
            F.max("_past_day").alias("cat_last_day"),
            F.min("_past_day").alias("_first_day"),
        )
        .join(F.broadcast(category_repurchase_days(products)), "category", "left")
        .join(
            F.broadcast(customers.select("customer_id", "household_size_est")),
            "customer_id",
            "left",
        )
    )
    n = F.col("_n")
    mean_gap = fx.mean_gap_days(n, F.datediff("cat_last_day", "_first_day"), SPARK_OPS)
    expected = fx.expected_repurchase_days(
        n,
        mean_gap,
        F.col("typical_repurchase_days"),
        fx.household_factor(F.col("household_size_est"), SPARK_OPS),
        SPARK_OPS,
    )
    per_category = per_category.select(
        "basket_id",
        "category",
        n.cast("double").alias("cat_n_purchase_days"),
        "cat_last_day",
        expected.cast("double").alias("cat_expected_days"),
    )

    # --- Ranking personal de categorias (punto A4) ---
    customer_days = purchase_days.groupBy("basket_id").agg(
        F.countDistinct("_past_day").alias("_customer_days")
    )
    per_category = with_category_ranks(
        per_category.join(q.select("basket_id", "_query_day"), "basket_id").join(
            customer_days, "basket_id"
        )
    ).drop("_query_day", "_customer_days")

    return AsOfHistory(products=per_product, categories=per_category, customer=customer)


def with_category_ranks(df: DataFrame) -> DataFrame:
    """`cat_freq_rank`, `cat_freq_share` y `cat_due_rank` de cada categoria de la query.

    `df` es la tabla cliente x categoria de una query (`basket_id`, `category`,
    `cat_n_purchase_days`, `cat_last_day`, `cat_expected_days`) con `_query_day` y
    `_customer_days` (dias de compra distintos del cliente). El `cat_due` se evalua el dia
    de la query con las mismas formulas que `with_repurchase_state`. Los puestos son
    `dense_rank`: dos categorias con la misma cifra comparten puesto, asi que el
    resultado no depende de ningun desempate.
    """
    n = F.col("cat_n_purchase_days")
    ratio = fx.overdue_ratio(
        F.datediff("_query_day", "cat_last_day").cast("double"), F.col("cat_expected_days")
    )
    need = fx.category_need_score(n, fx.is_due(ratio, SPARK_OPS))
    by_query = Window.partitionBy("basket_id")
    return (
        df.withColumn("cat_freq_share", n / F.col("_customer_days").cast("double"))
        .withColumn("_need", need)
        .withColumn("cat_freq_rank", F.dense_rank().over(by_query.orderBy(n.desc())).cast("double"))
        .withColumn(
            "cat_due_rank",
            F.dense_rank().over(by_query.orderBy(F.col("_need").desc())).cast("double"),
        )
        .drop("_need")
    )


def with_repurchase_state(df: DataFrame) -> DataFrame:
    """Estado del ciclo el dia de la cesta: `cat_days_since`, `cat_overdue_ratio`, `cat_due`.

    `df` necesita `basket_day`, `cat_last_day` y `cat_expected_days` (nulos si el cliente
    no ha comprado la categoria: entonces el ratio queda nulo y `cat_due` a 0).
    """
    return (
        df.withColumn("cat_days_since", F.datediff("basket_day", "cat_last_day").cast("double"))
        .withColumn(
            "cat_overdue_ratio",
            fx.overdue_ratio(F.col("cat_days_since"), F.col("cat_expected_days")),
        )
        .withColumn("cat_due", fx.is_due(F.col("cat_overdue_ratio"), SPARK_OPS))
    )

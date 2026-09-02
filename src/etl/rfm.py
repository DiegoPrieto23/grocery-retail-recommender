"""RFM por cliente (Recency, Frequency, Monetary).

Se calcula sobre las cestas ya limpias, con `total_amount` recalculado: hacerlo sobre el
importe crudo mezclaria los outliers de la Fase 1 en el componente monetario y desplazaria
a esos clientes al quintil alto sin motivo.

Las cestas anonimas (`customer_id` nulo) quedan fuera por definicion: no hay cliente al que
atribuirlas. Los clientes sin ninguna compra si aparecen, con `frequency = 0` y segmento
`Sin compras`, porque el NBA de la Fase 4 tiene que poder actuar sobre ellos.
"""

from __future__ import annotations

import datetime as dt

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# Etiquetas de segmento. El orden de evaluacion importa: se aplica la primera que encaja.
SEGMENT_NO_PURCHASES = "Sin compras"
SEGMENT_CHAMPIONS = "Campeones"
SEGMENT_LOYAL = "Fieles"
SEGMENT_PROMISING = "Prometedores"
SEGMENT_NEEDS_ATTENTION = "Necesitan atencion"
SEGMENT_AT_RISK = "En riesgo"
SEGMENT_HIBERNATING = "Hibernando"


def _reference_date_column(baskets: DataFrame, reference_date: str | dt.date | None) -> Column:
    """Fecha desde la que se mide la recencia.

    Por defecto, el ultimo dia con actividad en el dataset. Usar `current_date()` daria
    recencias enormes y dependientes del dia en que se ejecute el ETL, y los tests
    dejarian de ser deterministas.
    """
    if reference_date is not None:
        return F.lit(str(reference_date)).cast("date")
    last_day = baskets.agg(F.max(F.to_date("basket_date"))).collect()[0][0]
    return F.lit(last_day).cast("date")


def rfm(
    baskets: DataFrame,
    customers: DataFrame | None = None,
    *,
    reference_date: str | dt.date | None = None,
    n_bins: int = 5,
) -> DataFrame:
    """Calcula RFM y el segmento comercial de cada cliente.

    Args:
        baskets: Cestas limpias. Necesita `customer_id`, `basket_id`, `basket_date` y
            `total_amount`.
        customers: Maestro de clientes. Si se pasa, el resultado incluye tambien a los
            clientes sin compras y anade `loyalty_tier`, `signup_date` y `churn_label`.
        reference_date: Fecha de corte para la recencia. Por defecto, el ultimo dia con
            compras del dataset.
        n_bins: Numero de tramos de los scores R, F y M. 5 es el estandar (quintiles).

    Returns:
        Un DataFrame por `customer_id` con las metricas RFM, los scores 1-`n_bins` y el
        segmento. Score alto siempre significa "mejor": mas reciente, mas frecuente,
        mas gasto.
    """
    if n_bins < 2:
        raise ValueError("n_bins debe ser al menos 2")

    ref = _reference_date_column(baskets, reference_date)
    identified = baskets.filter(F.col("customer_id").isNotNull())

    agg = identified.groupBy("customer_id").agg(
        F.max(F.to_date("basket_date")).alias("last_purchase_date"),
        F.min(F.to_date("basket_date")).alias("first_purchase_date"),
        F.countDistinct("basket_id").cast("int").alias("frequency"),
        F.round(F.sum("total_amount"), 2).alias("monetary"),
        F.round(F.avg("total_amount"), 2).alias("avg_ticket"),
        F.sum("n_units").cast("int").alias("total_units")
        if "n_units" in baskets.columns
        else F.lit(None).cast("int").alias("total_units"),
    )

    metrics = (
        agg.withColumn("reference_date", ref)
        .withColumn("recency_days", F.datediff(ref, F.col("last_purchase_date")))
        .withColumn("tenure_days", F.datediff(ref, F.col("first_purchase_date")))
        .withColumn(
            # Cadencia media entre visitas: util como feature y para leer la recencia.
            "avg_days_between_baskets",
            F.when(
                F.col("frequency") > 1,
                F.round(
                    F.datediff(F.col("last_purchase_date"), F.col("first_purchase_date"))
                    / (F.col("frequency") - 1),
                    1,
                ),
            ),
        )
    )

    if customers is not None:
        extra = [c for c in ("loyalty_tier", "signup_date", "churn_label") if c in customers.columns]
        metrics = customers.select("customer_id", *extra).join(metrics, "customer_id", "left")
        metrics = metrics.withColumn(
            "frequency", F.coalesce("frequency", F.lit(0))
        ).withColumn("monetary", F.coalesce("monetary", F.lit(0.0)))

    return _add_scores(metrics, n_bins=n_bins)


def _add_scores(metrics: DataFrame, *, n_bins: int) -> DataFrame:
    """Anade los scores R, F y M por tramos y el segmento resultante.

    Los tramos se calculan **solo sobre los clientes con compras**, y despues se reincorpora
    al resto con score 1. Calcular el `ntile` sobre todos y enmascarar despues no valdria:
    los clientes sin compras seguirian ocupando sitio en los tramos y desplazarian los
    cortes de los que si compran.
    """
    has_purchases = F.col("frequency") > 0

    # `ntile` reparte en n grupos del mismo tamano. La recencia se ordena descendente
    # para que el tramo alto sea el mas reciente, igual que en F y M.
    buyers = metrics.filter(has_purchases)
    non_buyers = metrics.filter(~has_purchases)

    scored = (
        buyers.withColumn(
            "r_score",
            F.ntile(n_bins).over(Window.orderBy(F.col("recency_days").desc())).cast("int"),
        )
        .withColumn(
            "f_score",
            F.ntile(n_bins).over(Window.orderBy(F.col("frequency").asc())).cast("int"),
        )
        .withColumn(
            "m_score",
            F.ntile(n_bins).over(Window.orderBy(F.col("monetary").asc())).cast("int"),
        )
    )
    if not non_buyers.isEmpty():
        scored = scored.unionByName(
            non_buyers.withColumn("r_score", F.lit(1))
            .withColumn("f_score", F.lit(1))
            .withColumn("m_score", F.lit(1))
        )

    # Reglas clasicas de segmentacion RFM. `fm` combina frecuencia y gasto porque en
    # gran consumo van muy de la mano y separarlos genera segmentos casi vacios.
    fm = F.round((F.col("f_score") + F.col("m_score")) / 2.0)
    high, mid = n_bins - 1, (n_bins + 1) // 2
    segment = (
        F.when(~has_purchases, F.lit(SEGMENT_NO_PURCHASES))
        .when((F.col("r_score") >= high) & (fm >= high), F.lit(SEGMENT_CHAMPIONS))
        .when((F.col("r_score") >= mid) & (fm >= mid), F.lit(SEGMENT_LOYAL))
        .when((F.col("r_score") >= high) & (fm < mid), F.lit(SEGMENT_PROMISING))
        .when((F.col("r_score") == mid) & (fm < mid), F.lit(SEGMENT_NEEDS_ATTENTION))
        .when(fm >= mid, F.lit(SEGMENT_AT_RISK))
        .otherwise(F.lit(SEGMENT_HIBERNATING))
    )

    return (
        scored.withColumn(
            "rfm_score",
            F.concat(F.col("r_score"), F.col("f_score"), F.col("m_score")).cast("string"),
        )
        .withColumn("rfm_segment", segment)
        .withColumn("has_purchases", has_purchases)
    )

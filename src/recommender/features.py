"""Features del ranker: todo lo que se sabe de un candidato en el instante de la compra.

Cada fila es un par `(query, candidato)` y todas sus columnas se calculan con informacion
**anterior** a la cesta que se quiere predecir. Las familias son cinco:

1. **Senal de cada fuente de candidatos** (`candidates.SOURCE_COLUMNS`): score y puesto de
   quien lo propuso, y cuantas fuentes coincidieron.
2. **Cliente x producto y cliente x categoria**: cuantas veces lo ha comprado, cuando fue
   la ultima, y el estado del ciclo de reposicion de su categoria (Tarea 2).
3. **Producto**: popularidad global y reciente, indice estacional del mes de la cesta,
   precio, marca blanca, perecedero.
4. **Contexto y promocion**: canal, mes, dia de la semana, tamano del carrito, RFM del
   cliente y si el producto esta en promocion ese dia.
5. **Sesion**: lo que la navegacion sabia **antes** del corte (`cut_ts`).

## Por que la sesion no filtra el target

`session_events` describe la misma cesta que hay que adivinar, asi que solo es usable con
un corte temporal estricto: entran los eventos con `event_timestamp <= cut_ts` y ni uno
mas. En ese instante, un producto que el cliente ya ha **visto y todavia no ha anadido** es
una pista legitima y fuerte; lo que ya esta anadido se excluye del pool en
`candidates.in_cart`, porque recomendar lo que ya esta en el carrito no es una prediccion.

Medido sobre este dataset, en un corte a media cesta solo ~24 % de lo que falta por anadir
ha sido visto ya, compitiendo con ~1,6 productos vistos que nunca se anadiran: la senal es
util y claramente imperfecta, que es lo que debe ser. Ver `DATA_SPEC.md`, "Embudo online".
"""

from __future__ import annotations

import datetime as dt

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.recommender.candidates import SOURCE_COLUMNS, SOURCE_NAMES
from src.recommender.schema import (
    CATEGORICAL_FEATURES,
    CHANNELS,
    CONTEXT_FEATURES,
    CUSTOMER_CATEGORY_FEATURES,
    CUSTOMER_PRODUCT_FEATURES,
    FEATURE_COLUMNS,
    LOYALTY,
    PRODUCT_FEATURES,
    SESSION_FEATURES,
    SOURCE_FEATURES,
)

# El esquema vive en `schema` (sin PySpark) para que la ruta de serving de la Fase 6b
# use exactamente las mismas columnas y en el mismo orden.
_CHANNELS = CHANNELS
_LOYALTY = LOYALTY


def _code(column: str, values: tuple[str, ...]):
    """Codifica un texto de dominio cerrado como entero.

    Lo desconocido (y lo nulo: una cesta anonima no tiene `loyalty_tier`) recibe el codigo
    `len(values)`, no -1. LightGBM convierte los negativos de una feature categorica en
    NaN y avisa por ello; con un codigo propio, "sin dato" es una categoria mas y el arbol
    puede separarla.
    """
    expr = F.lit(len(values))
    for i, value in enumerate(values):
        expr = F.when(F.col(column) == value, F.lit(i)).otherwise(expr)
    return expr.cast("int")


def index_products(products: DataFrame) -> DataFrame:
    """Catalogo con `department` y `category` codificados por orden alfabetico.

    El orden alfabetico (y no la frecuencia) mantiene el codigo estable entre ejecuciones,
    que es lo que permite que el modelo guardado siga siendo valido.
    """
    dept = (
        products.select("department")
        .distinct()
        .withColumn(
            "department_idx",
            (F.row_number().over(Window.orderBy(F.col("department").asc())) - 1).cast("int"),
        )
    )
    catg = (
        products.select("category")
        .distinct()
        .withColumn(
            "category_idx",
            (F.row_number().over(Window.orderBy(F.col("category").asc())) - 1).cast("int"),
        )
    )
    return (
        products.join(F.broadcast(dept), "department")
        .join(F.broadcast(catg), "category")
        .select(
            "product_id",
            "category",
            "department_idx",
            "category_idx",
            "unit_price",
            F.col("pack_size").cast("double").alias("pack_size"),
            F.col("typical_repurchase_days").cast("double").alias("typical_repurchase_days"),
            F.col("is_private_label").cast("double").alias("is_private_label"),
            F.col("is_perishable").cast("double").alias("is_perishable"),
        )
    )


# --------------------------------------------------------------------------------------
# Sesion
# --------------------------------------------------------------------------------------
def session_events_before_cut(
    queries: DataFrame, sessions: DataFrame, session_events: DataFrame
) -> DataFrame:
    """Eventos de la sesion de cada query anteriores o iguales a `cut_ts`.

    Es el unico punto por el que `session_events` entra en el modelo, y el filtro es
    estricto: todo lo posterior al corte es futuro.
    """
    scoped = queries.filter(F.col("session_id").isNotNull()).select(
        "basket_id", "session_id", "cut_ts"
    )
    return (
        scoped.join(session_events, "session_id")
        .filter(F.col("event_timestamp") <= F.col("cut_ts"))
        .select("basket_id", "product_id", "event_type", "event_timestamp", "cut_ts")
    )


def session_cart(events_before_cut: DataFrame) -> DataFrame:
    """Productos ya anadidos al carrito en el instante del corte.

    Incluye los que el cliente acabara abandonando: en `cut_ts` estan en el carrito, y por
    tanto no son recomendables.
    """
    return (
        events_before_cut.filter(F.col("event_type") == "add_to_cart")
        .select("basket_id", "product_id")
        .distinct()
    )


def session_features(events_before_cut: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Vistas previas al corte, por producto, mas el tamano de la sesion hasta ese punto.

    Returns:
        `(por_producto, por_query)`.
    """
    views = events_before_cut.filter(F.col("event_type") == "view")

    per_product = views.groupBy("basket_id", "product_id").agg(
        F.count(F.lit(1)).cast("double").alias("sess_n_views"),
        F.max("event_timestamp").alias("_last_view"),
        F.first("cut_ts").alias("_cut_ts"),
    ).withColumn(
        "sess_secs_since_view",
        (F.unix_timestamp("_cut_ts") - F.unix_timestamp("_last_view")).cast("double"),
    ).withColumn("sess_viewed", F.lit(1.0)).drop("_last_view", "_cut_ts")

    per_query = events_before_cut.groupBy("basket_id").agg(
        F.count(F.lit(1)).cast("double").alias("sess_n_events_before")
    )
    return per_product, per_query


# --------------------------------------------------------------------------------------
# Cliente
# --------------------------------------------------------------------------------------
def customer_profile(history_baskets: DataFrame, history_items: DataFrame) -> DataFrame:
    """RFM ligero calculado **solo con el historial de la ventana**.

    No se reutiliza `data/processed/rfm`: aquella tabla mide la recencia contra el ultimo
    dia del dataset, es decir contra el futuro de cualquier cesta de test. Recalcularla
    aqui cuesta un `groupBy` y evita una fuga silenciosa.
    """
    identified = history_baskets.filter(F.col("customer_id").isNotNull())
    per_customer = identified.groupBy("customer_id").agg(
        F.countDistinct("basket_id").cast("double").alias("cust_frequency"),
        F.avg("total_amount").alias("cust_avg_ticket"),
        F.max("basket_day").alias("cust_last_day"),
    )
    distinct_products = (
        history_items.select("basket_id", "product_id")
        .join(identified.select("basket_id", "customer_id"), "basket_id")
        .groupBy("customer_id")
        .agg(F.countDistinct("product_id").cast("double").alias("cust_n_products"))
    )
    return per_customer.join(distinct_products, "customer_id", "left")


# --------------------------------------------------------------------------------------
# Ensamblado
# --------------------------------------------------------------------------------------
def build_feature_matrix(
    candidates: DataFrame,
    queries: DataFrame,
    *,
    products: DataFrame,
    popularity: DataFrame,
    customer_products: DataFrame,
    customer_stats: DataFrame,
    repurchase: DataFrame,
    customers: DataFrame,
    promotions: DataFrame,
    session_product: DataFrame,
    session_query: DataFrame,
    target: DataFrame | None = None,
) -> DataFrame:
    """Cruza el pool de candidatos con todas las familias de features.

    Args:
        candidates: Salida de `candidates.union_candidates`.
        queries: Cabeceras de query (`splits.build_queries`).
        products: Catalogo ya indexado por `index_products`.
        popularity: Tabla completa producto x mes de `candidates.fit_popularity`.
        customer_products: Historial cliente x producto.
        customer_stats: Salida de `customer_profile`.
        repurchase: `repurchase_features` recalculada sobre el historial de la ventana.
        customers: Maestro de clientes (hogar y tier).
        promotions: Promociones limpias.
        session_product: Vistas por producto antes del corte.
        session_query: Tamano de la sesion antes del corte.
        target: Pares `(basket_id, product_id)` que hay que adivinar. Si se pasa, se anade
            la columna `label`; si no, la matriz es de inferencia.

    Returns:
        Un DataFrame con `basket_id`, `product_id`, `profile`, `FEATURE_COLUMNS` y, si
        procede, `label`.
    """
    q = queries.select(
        "basket_id",
        "customer_id",
        "basket_day",
        "channel",
        "profile",
        F.col("prefix_size").cast("double").alias("prefix_size"),
        F.col("is_known_customer").cast("double").alias("is_known_customer"),
        F.col("session_id"),
    )

    df = candidates.join(q, "basket_id")

    # --- Producto: catalogo + popularidad e indice estacional del mes de la cesta ---
    df = df.withColumn("basket_month", F.month("basket_day").cast("int")).withColumn(
        "basket_dow", F.dayofweek("basket_day").cast("int")
    )
    pop = popularity.select(
        "product_id",
        "month",
        F.col("pop_seasonal_index").alias("prod_seasonal_index"),
        F.col("pop_rank").cast("double").alias("prod_month_rank"),
        "prod_pop_all",
        "prod_pop_recent",
    )
    df = (
        df.withColumn("month", F.col("basket_month"))
        .join(F.broadcast(pop), ["product_id", "month"], "left")
        .drop("month")
        .join(F.broadcast(products), "product_id", "left")
    )

    # --- Cliente x producto: lo compre quien lo compre, no solo si lo propuso `hist` ---
    hist = customer_products.select(
        "customer_id",
        "product_id",
        "hist_n_baskets",
        "hist_units",
        "hist_last_day",
    )
    df = (
        df.join(hist, ["customer_id", "product_id"], "left")
        .withColumn("hist_days_since", F.datediff("basket_day", "hist_last_day").cast("double"))
        .withColumn("hist_ever_bought", F.col("hist_n_baskets").isNotNull().cast("double"))
        .drop("hist_last_day")
    )

    # --- Cliente x categoria: el ciclo de reposicion de la Tarea 2 ---
    rep = repurchase.select(
        "customer_id",
        "category",
        F.col("n_purchase_days").cast("double").alias("cat_n_purchase_days"),
        F.col("last_purchase_date").alias("cat_last_day"),
        F.col("expected_repurchase_days").alias("cat_expected_days"),
    )
    df = (
        df.join(rep, ["customer_id", "category"], "left")
        .withColumn("cat_days_since", F.datediff("basket_day", "cat_last_day").cast("double"))
        # El `overdue_ratio` se recalcula contra el dia de la cesta, no contra la fecha de
        # corte del ETL: entre una y otra pueden pasar semanas.
        .withColumn("cat_overdue_ratio", F.col("cat_days_since") / F.col("cat_expected_days"))
        .withColumn("cat_due", (F.col("cat_overdue_ratio") >= 1.0).cast("double"))
        .drop("cat_last_day")
    )

    # --- Cliente: RFM de la ventana + maestro ---
    master = customers.select(
        "customer_id",
        F.col("household_size_est").cast("double").alias("household_size_est"),
        "loyalty_tier",
    )
    df = (
        df.join(F.broadcast(customer_stats), "customer_id", "left")
        .withColumn("cust_recency_days", F.datediff("basket_day", "cust_last_day").cast("double"))
        .drop("cust_last_day")
        .join(F.broadcast(master), "customer_id", "left")
        .withColumn("loyalty_idx", _code("loyalty_tier", _LOYALTY))
        .withColumn("channel_idx", _code("channel", _CHANNELS))
        .drop("loyalty_tier", "channel")
    )

    # --- Promocion vigente el dia de la cesta ---
    promo = promotions.filter(~F.col("date_range_invalid")).select(
        "product_id", "start_date", "end_date", "discount_value"
    )
    active = (
        df.select("basket_id", "product_id", "basket_day")
        .join(F.broadcast(promo), "product_id")
        .filter(
            (F.col("basket_day") >= F.col("start_date"))
            & (F.col("basket_day") <= F.col("end_date"))
        )
        .groupBy("basket_id", "product_id")
        .agg(F.max("discount_value").alias("promo_discount"))
        .withColumn("is_on_promo", F.lit(1.0))
    )
    df = df.join(active, ["basket_id", "product_id"], "left")

    # --- Sesion, siempre por detras del corte ---
    df = (
        df.join(session_product, ["basket_id", "product_id"], "left")
        .join(session_query, "basket_id", "left")
        .withColumn("has_session", F.col("session_id").isNotNull().cast("double"))
        .drop("session_id")
    )

    if target is not None:
        df = df.join(
            target.select("basket_id", "product_id").distinct().withColumn("label", F.lit(1)),
            ["basket_id", "product_id"],
            "left",
        ).withColumn("label", F.coalesce(F.col("label"), F.lit(0)))

    # Los ceros son informativos aqui: "no lo ha comprado nunca", "no estaba de oferta",
    # "no lo habia visto". Dejarlos nulos obligaria a LightGBM a inventarse una direccion.
    zero_filled = {
        "hist_n_baskets": 0.0,
        "hist_units": 0.0,
        "hist_ever_bought": 0.0,
        "cat_n_purchase_days": 0.0,
        "cat_due": 0.0,
        "is_on_promo": 0.0,
        "promo_discount": 0.0,
        "sess_viewed": 0.0,
        "sess_n_views": 0.0,
        "sess_n_events_before": 0.0,
        "n_sources": 0.0,
        "cust_frequency": 0.0,
        "cust_n_products": 0.0,
    }
    for column, value in zero_filled.items():
        df = df.withColumn(column, F.coalesce(F.col(column), F.lit(value)))

    keep = ["basket_id", "product_id", "profile", *FEATURE_COLUMNS]
    if target is not None:
        keep.append("label")
    return df.select(*[F.col(c).alias(c) for c in keep])


def default_reference_date(window_start: dt.date | str) -> str:
    """Fecha de corte del `repurchase_features` de una ventana: el dia anterior a su inicio."""
    day = window_start if isinstance(window_start, dt.date) else dt.date.fromisoformat(window_start)
    return str(day - dt.timedelta(days=1))

"""Primera etapa del recomendador: generacion de candidatos.

Reducir 1.500 productos a un centenar antes de puntuarlos es lo que hace viable el patron
de dos etapas de `CHALLENGE.md`: el ranker de la segunda etapa puede permitirse features
caras porque solo ve el pool, no el catalogo.

Cada fuente es **independiente** y aporta su propia senal. Ninguna sabe de las otras, y el
ranker de la Fase 3b es quien decide como pesarlas:

| Fuente | De donde sale | Que perfiles cubre |
| --- | --- | --- |
| `pop` | Popularidad reciente x indice estacional del mes | los cuatro (es el respaldo) |
| `aff` | `affinity_product`: co-compra a nivel de SKU | 2 y 4 (hace falta carrito) |
| `cataff` | `affinity_category` + los mas vendidos de la categoria | 2 y 4, y llega donde `aff` no llega |
| `hist` | Historial del cliente + `due_for_repurchase` | 3 y 4 (hace falta historial) |
| `als` | ALS implicito de Spark MLlib sobre cliente x producto | 3 y 4 |

Un cliente nuevo con el carrito vacio (perfil 1) solo activa `pop`; uno recurrente a media
compra (perfil 4) activa las cinco. Es exactamente lo que dice `CHALLENGE.md`: **lo que
cambia entre perfiles es que fuentes tienen senal, no el ranker**.

Todas las fuentes se ajustan sobre el historial **anterior** a la ventana que se va a
predecir (ver `config.py`), asi que ninguna ve la cesta que tiene que adivinar.
"""

from __future__ import annotations

import datetime as dt

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.etl.affinity import cooccurrence_affinity
from src.recommender.config import ALSConfig, CandidateConfig
# Definidos en `schema` para que la demo (pandas, sin Spark) use las mismas columnas.
from src.recommender.schema import SOURCE_COLUMNS, SOURCE_NAMES

# Columnas que aporta cada fuente al pool unificado. El orden es el del informe.


# --------------------------------------------------------------------------------------
# Fuente 1 - popularidad y estacionalidad
# --------------------------------------------------------------------------------------
def fit_popularity(
    history_baskets: DataFrame,
    history_items: DataFrame,
    *,
    window_start: dt.date | str,
    cfg: CandidateConfig,
) -> DataFrame:
    """Popularidad reciente e indice estacional por producto y mes del ano.

    El **indice estacional** es cuanto se desvia un producto de su propio ritmo en cada
    mes, corregido por el peso del mes en el total:

        indice(p, m) = [cestas de p en el mes m / cestas de p] / [cestas en m / cestas]

    Un turron tiene indice ~8 en diciembre y ~0 el resto del ano, que es justo lo que
    `DATA_SPEC.md` inyecta. La **popularidad reciente** son los ultimos `recent_days` de
    historial, para que la lista de respaldo no la dominen productos ya descatalogados.

    Returns:
        Una fila por `(product_id, month)` -- la tabla completa, sin podar -- con
        `pop_score`, `pop_rank` (puesto dentro del mes), `pop_seasonal_index`,
        `prod_pop_recent` y `prod_pop_all`. La poda a `n_popularity` la hace
        `candidates_popularity`; el resto de la tabla se sigue usando como *feature* de
        producto y para `fit_category_leaders`.
    """
    dated = history_items.select("basket_id", "product_id").join(
        history_baskets.select("basket_id", "basket_day"), "basket_id"
    )

    total_baskets = history_baskets.count()
    by_month_total = (
        history_baskets.groupBy(F.month("basket_day").alias("month"))
        .agg(F.countDistinct("basket_id").cast("double").alias("month_baskets"))
        .withColumn("month_share", F.col("month_baskets") / F.lit(float(total_baskets)))
        .select("month", "month_share")
    )

    per_product = dated.groupBy("product_id").agg(
        F.countDistinct("basket_id").cast("double").alias("prod_pop_all")
    )
    per_product_month = dated.groupBy("product_id", F.month("basket_day").alias("month")).agg(
        F.countDistinct("basket_id").cast("double").alias("prod_month_baskets")
    )

    recent_from = F.lit(str(window_start)).cast("date") - F.expr(f"INTERVAL {cfg.recent_days} DAYS")
    recent = (
        dated.filter(F.col("basket_day") >= recent_from)
        .groupBy("product_id")
        .agg(F.countDistinct("basket_id").cast("double").alias("prod_pop_recent"))
    )

    scored = (
        per_product_month.join(F.broadcast(per_product), "product_id")
        .join(F.broadcast(by_month_total), "month")
        .join(F.broadcast(recent), "product_id", "left")
        .withColumn("prod_pop_recent", F.coalesce(F.col("prod_pop_recent"), F.lit(0.0)))
        .withColumn(
            "pop_seasonal_index",
            F.round(
                (F.col("prod_month_baskets") / F.col("prod_pop_all")) / F.col("month_share"), 4
            ),
        )
        # La popularidad reciente pone la escala; la estacionalidad la corrige. Un
        # producto con poca venta media pero muy de temporada sube en su mes.
        .withColumn("pop_score", F.col("prod_pop_recent") * F.col("pop_seasonal_index"))
    )

    ranked = Window.partitionBy("month").orderBy(
        F.col("pop_score").desc(), F.col("product_id").asc()
    )
    return scored.withColumn("pop_rank", F.row_number().over(ranked)).select(
        "product_id",
        "month",
        "pop_score",
        "pop_rank",
        "pop_seasonal_index",
        "prod_pop_recent",
        "prod_pop_all",
    )


def candidates_popularity(
    queries: DataFrame, popularity: DataFrame, *, cfg: CandidateConfig
) -> DataFrame:
    """Los productos mas vendidos del mes de la cesta. Es la fuente de respaldo."""
    top = popularity.filter(F.col("pop_rank") <= cfg.n_popularity).select(
        "product_id", "month", *SOURCE_COLUMNS["pop"]
    )
    return (
        queries.select("basket_id", F.month("basket_day").alias("month"))
        .join(F.broadcast(top), "month")
        .select("basket_id", "product_id", *SOURCE_COLUMNS["pop"])
    )


# --------------------------------------------------------------------------------------
# Fuente 2 - co-compra a nivel de producto
# --------------------------------------------------------------------------------------
def fit_affinity_product(history_items: DataFrame, *, top_n: int = 20) -> DataFrame:
    """Afinidad SKU x SKU sobre el historial de la ventana (reutiliza el ETL de la Fase 2)."""
    return cooccurrence_affinity(history_items, level="product", top_n=top_n)


def candidates_affinity_product(
    prefix: DataFrame, affinity_product: DataFrame, *, cfg: CandidateConfig
) -> DataFrame:
    """Lo que suele acompanar a lo que el cliente ya lleva en el carrito.

    Un candidato apoyado por varios productos del prefijo (pasta *y* vino sugieren queso)
    acumula `aff_conf_sum` y `aff_n_support`; `aff_lift_max` se queda con el vinculo mas
    fuerte. Los tres van al ranker, que decide cual pesa mas.
    """
    joined = prefix.join(
        affinity_product.select("antecedent", "consequent", "lift", "confidence"),
        prefix["product_id"] == F.col("antecedent"),
    ).select(
        "basket_id",
        F.col("consequent").alias("product_id"),
        "lift",
        "confidence",
    )

    agg = joined.groupBy("basket_id", "product_id").agg(
        F.max("lift").alias("aff_lift_max"),
        F.sum("confidence").alias("aff_conf_sum"),
        F.count(F.lit(1)).cast("int").alias("aff_n_support"),
    )
    ranked = Window.partitionBy("basket_id").orderBy(
        F.col("aff_lift_max").desc(), F.col("aff_conf_sum").desc(), F.col("product_id").asc()
    )
    return (
        agg.withColumn("_rank", F.row_number().over(ranked))
        .filter(F.col("_rank") <= cfg.n_affinity_product)
        .drop("_rank")
    )


# --------------------------------------------------------------------------------------
# Fuente 3 - co-compra a nivel de categoria
# --------------------------------------------------------------------------------------
def fit_affinity_category(history_items: DataFrame, products: DataFrame) -> DataFrame:
    """Afinidad categoria x categoria: las reglas de negocio de `DATA_SPEC.md`."""
    return cooccurrence_affinity(history_items, products, level="category")


def fit_category_leaders(
    popularity: DataFrame, products: DataFrame, *, cfg: CandidateConfig
) -> DataFrame:
    """Los productos mas vendidos de cada categoria, para bajar del par de categorias al SKU."""
    per_product = popularity.groupBy("product_id").agg(
        F.max("prod_pop_recent").alias("prod_pop_recent")
    )
    with_cat = per_product.join(F.broadcast(products.select("product_id", "category")), "product_id")
    ranked = Window.partitionBy("category").orderBy(
        F.col("prod_pop_recent").desc(), F.col("product_id").asc()
    )
    return (
        with_cat.withColumn("_rank", F.row_number().over(ranked))
        .filter(F.col("_rank") <= cfg.n_products_per_category)
        .select("category", "product_id")
    )


def candidates_affinity_category(
    prefix: DataFrame,
    affinity_category: DataFrame,
    category_leaders: DataFrame,
    products: DataFrame,
    *,
    cfg: CandidateConfig,
) -> DataFrame:
    """Co-compra por categoria, para los productos que la afinidad de SKU no alcanza.

    `affinity_product` solo cubre los productos con al menos 50 cestas en comun con otro
    (712 de 1.500 en este dataset): la cola larga del surtido se queda fuera. Bajando por
    categoria se llega igualmente a un SKU concreto, con menos precision pero mas
    cobertura.
    """
    prefix_cats = (
        prefix.join(F.broadcast(products.select("product_id", "category")), "product_id")
        .select("basket_id", F.col("category").alias("antecedent"))
        .distinct()
    )

    linked = prefix_cats.join(
        affinity_category.select("antecedent", "consequent", "lift", "confidence"), "antecedent"
    )
    ranked_cat = Window.partitionBy("basket_id").orderBy(
        F.col("lift").desc(), F.col("consequent").asc()
    )
    top_cats = (
        linked.groupBy("basket_id", "consequent")
        .agg(F.max("lift").alias("lift"), F.max("confidence").alias("confidence"))
        .withColumn("_rank", F.row_number().over(ranked_cat))
        .filter(F.col("_rank") <= cfg.n_affinity_category)
        .drop("_rank")
    )

    return (
        top_cats.join(
            F.broadcast(category_leaders), top_cats["consequent"] == category_leaders["category"]
        )
        .groupBy("basket_id", "product_id")
        .agg(
            F.max("lift").alias("cataff_lift_max"),
            F.max("confidence").alias("cataff_conf_max"),
        )
    )


# --------------------------------------------------------------------------------------
# Fuente 4 - historial personal y recompra
# --------------------------------------------------------------------------------------
def fit_customer_products(history_baskets: DataFrame, history_items: DataFrame) -> DataFrame:
    """Que ha comprado cada cliente, cuantas veces y cuando por ultima vez."""
    return (
        history_items.select("basket_id", "product_id", "quantity")
        .join(
            history_baskets.filter(F.col("customer_id").isNotNull()).select(
                "basket_id", "customer_id", "basket_day"
            ),
            "basket_id",
        )
        .groupBy("customer_id", "product_id")
        .agg(
            F.countDistinct("basket_id").cast("double").alias("hist_n_baskets"),
            F.sum("quantity").cast("double").alias("hist_units"),
            F.max("basket_day").alias("hist_last_day"),
        )
    )


def candidates_personal(
    queries: DataFrame,
    customer_products: DataFrame,
    repurchase: DataFrame,
    products: DataFrame,
    *,
    cfg: CandidateConfig,
) -> DataFrame:
    """Lo que el cliente ya compra, priorizando lo que ademas "toca" reponer.

    El orden mezcla frecuencia y ciclo de reposicion (`due_for_repurchase` de la Tarea 2):
    un producto que se compra mucho pero se acaba de comprar cede su sitio a otro que se
    compra menos pero lleva dos ciclos sin caer. Es la fuente que da sentido al perfil 3
    (cliente recurrente con el carrito vacio), donde no hay nada mas de lo que tirar.
    """
    due = repurchase.select(
        "customer_id",
        "category",
        F.col("due_for_repurchase").cast("double").alias("cat_due"),
        F.col("overdue_ratio").alias("cat_overdue_ratio"),
    )

    per_customer = (
        customer_products.join(F.broadcast(products.select("product_id", "category")), "product_id")
        .join(due, ["customer_id", "category"], "left")
        .withColumn("cat_due", F.coalesce(F.col("cat_due"), F.lit(0.0)))
    )

    scoped = queries.select("basket_id", "customer_id", "basket_day").filter(
        F.col("customer_id").isNotNull()
    ).join(per_customer, "customer_id")

    scored = scoped.withColumn(
        "hist_days_since", F.datediff(F.col("basket_day"), F.col("hist_last_day"))
    ).withColumn(
        # Frecuencia como base, y un empujon a lo que ya cumplio su ciclo.
        "_score",
        F.col("hist_n_baskets") * (F.lit(1.0) + F.col("cat_due"))
        + F.coalesce(F.col("cat_overdue_ratio"), F.lit(0.0)),
    )
    ranked = Window.partitionBy("basket_id").orderBy(
        F.col("_score").desc(), F.col("hist_days_since").asc(), F.col("product_id").asc()
    )
    return (
        scored.withColumn("hist_rank", F.row_number().over(ranked).cast("double"))
        .filter(F.col("hist_rank") <= cfg.n_personal)
        .select("basket_id", "product_id", *SOURCE_COLUMNS["hist"])
    )


# --------------------------------------------------------------------------------------
# Fuente 5 - ALS (Spark MLlib)
# --------------------------------------------------------------------------------------
def _index(df: DataFrame, key: str) -> DataFrame:
    """Numera una clave de texto de forma estable (ALS solo admite indices enteros).

    Se hace con `row_number` sobre el orden alfabetico y no con `StringIndexer` para que
    el indice no dependa de las frecuencias ni del particionado: el mismo historial da
    siempre los mismos indices, y el modelo guardado se puede releer.
    """
    return df.select(key).distinct().withColumn(
        f"{key}_idx", (F.row_number().over(Window.orderBy(F.col(key).asc())) - 1).cast("int")
    )


def fit_als(customer_products: DataFrame, *, cfg: ALSConfig):
    """Entrena un ALS implicito sobre `cliente x producto`.

    Feedback implicito (`implicitPrefs=True`): no hay valoraciones, hay compras repetidas.
    El numero de cestas en las que aparece el producto es la "confianza", no una nota, que
    es exactamente el caso de uso para el que Hu-Koren-Volinsky disenaron esta variante.

    Returns:
        `(modelo, indice_de_clientes, indice_de_productos)`.
    """
    from pyspark.ml.recommendation import ALS

    customer_index = _index(customer_products, "customer_id").cache()
    product_index = _index(customer_products, "product_id").cache()

    ratings = (
        customer_products.join(customer_index, "customer_id")
        .join(product_index, "product_id")
        .select(
            "customer_id_idx",
            "product_id_idx",
            F.col("hist_n_baskets").cast("float").alias("rating"),
        )
    )

    als = ALS(
        userCol="customer_id_idx",
        itemCol="product_id_idx",
        ratingCol="rating",
        rank=cfg.rank,
        maxIter=cfg.max_iter,
        regParam=cfg.reg_param,
        alpha=cfg.alpha,
        implicitPrefs=True,
        coldStartStrategy="drop",
        seed=cfg.seed,
        nonnegative=True,
    )
    return als.fit(ratings), customer_index, product_index


def candidates_als(
    queries: DataFrame,
    model,
    customer_index: DataFrame,
    product_index: DataFrame,
    *,
    cfg: CandidateConfig,
) -> DataFrame:
    """Recomendaciones colaborativas para los clientes con historial.

    Un cliente que no estaba en el historial no tiene factores latentes y simplemente no
    recibe candidatos de esta fuente: es el cold-start de los perfiles 1 y 2, y se cubre
    con popularidad y co-compra.
    """
    users = (
        queries.select("customer_id")
        .filter(F.col("customer_id").isNotNull())
        .distinct()
        .join(customer_index, "customer_id")
        .select("customer_id_idx")
    )
    if users.isEmpty():
        return queries.sparkSession.createDataFrame(
            [], "basket_id string, product_id string, als_score double, als_rank int"
        )

    recs = (
        model.recommendForUserSubset(users, cfg.n_als)
        .select("customer_id_idx", F.posexplode("recommendations").alias("pos", "rec"))
        .select(
            "customer_id_idx",
            F.col("rec.product_id_idx").alias("product_id_idx"),
            F.col("rec.rating").cast("double").alias("als_score"),
            (F.col("pos") + 1).cast("int").alias("als_rank"),
        )
        .join(customer_index, "customer_id_idx")
        .join(product_index, "product_id_idx")
        .select("customer_id", "product_id", "als_score", "als_rank")
    )
    return (
        queries.select("basket_id", "customer_id")
        .filter(F.col("customer_id").isNotNull())
        .join(recs, "customer_id")
        .select("basket_id", "product_id", "als_score", "als_rank")
    )


# --------------------------------------------------------------------------------------
# Union del pool
# --------------------------------------------------------------------------------------
def in_cart(prefix: DataFrame, session_cart: DataFrame | None) -> DataFrame:
    """Lo que ya esta en el carrito en el instante del corte, y por tanto no se recomienda.

    No es solo el prefijo del ticket: si la sesion registro un `add_to_cart` antes del
    corte, ese producto esta en el carrito aunque acabe abandonado y no llegue al ticket.
    Recomendar algo que el cliente ya tiene delante seria un fallo de producto, no de
    modelo, asi que se excluye del pool y no cuenta como negativo.
    """
    cart = prefix.select("basket_id", "product_id")
    if session_cart is not None:
        cart = cart.unionByName(session_cart.select("basket_id", "product_id"))
    return cart.distinct()


def union_candidates(
    sources: dict[str, DataFrame], queries: DataFrame, cart: DataFrame
) -> DataFrame:
    """Funde los pools de todas las fuentes en un candidato por `(basket_id, product_id)`.

    Cada fuente deja sus columnas de score; las de las fuentes que no propusieron ese
    producto quedan nulas, y `src_<fuente>` marca quien lo propuso. `n_sources` resume
    cuantas coincidieron: que dos senales independientes apunten al mismo producto es en
    si mismo informacion, y el ranker la usa.
    """
    all_columns: list[str] = [c for cols in SOURCE_COLUMNS.values() for c in cols]

    frames: list[DataFrame] = []
    for name, df in sources.items():
        own = SOURCE_COLUMNS[name]
        frame = df.select("basket_id", "product_id", *own)
        for column in all_columns:
            if column not in own:
                frame = frame.withColumn(column, F.lit(None).cast("double"))
        for other in SOURCE_NAMES:
            frame = frame.withColumn(f"src_{other}", F.lit(1.0 if other == name else 0.0))
        frames.append(frame.select("basket_id", "product_id", *all_columns, *[f"src_{s}" for s in SOURCE_NAMES]))

    pool = frames[0]
    for frame in frames[1:]:
        pool = pool.unionByName(frame)

    merged = pool.groupBy("basket_id", "product_id").agg(
        *[F.max(c).alias(c) for c in all_columns],
        *[F.max(f"src_{s}").alias(f"src_{s}") for s in SOURCE_NAMES],
    )

    return (
        merged.join(cart, ["basket_id", "product_id"], "left_anti")
        .join(queries.select("basket_id"), "basket_id", "left_semi")
        .withColumn(
            "n_sources", sum(F.col(f"src_{s}") for s in SOURCE_NAMES).cast("double")
        )
    )

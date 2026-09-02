"""Split temporal por cesta y construccion de las *queries* del recomendador.

Una **query** es un instante simulado de la compra: un cliente (o nadie, si la cesta es
anonima), lo que ya lleva en el carrito, y lo que todavia le falta por anadir. Es la
unidad sobre la que se evalua: el sistema propone 5 productos y se compara con lo que
realmente acabo en el ticket.

## Como se parte una cesta

Los productos de la cesta se ordenan **por el momento en que se anadieron al carrito**
cuando la cesta tiene una sesion online detras, y por un hash estable cuando no la tiene
(compra en tienda, o cesta online sin sesion registrada). Sobre ese orden se corta:

    posiciones 1..prefix_size  ->  `prefix`, lo que el cliente ya lleva
    posiciones prefix_size+1.. ->  `target`, lo que hay que adivinar

`prefix_size` es 0 en la mitad de las cestas (perfiles de "carrito vacio") y la mitad de
las lineas en la otra mitad. El reparto lo decide el hash del `basket_id`, no un sorteo:
dos ejecuciones producen exactamente las mismas queries.

## Los cuatro perfiles de `CHALLENGE.md`

|   | Carrito vacio | Con articulos |
| --- | --- | --- |
| **Cliente nuevo** (anonimo, o sin compras antes de la ventana) | perfil 1 | perfil 2 |
| **Cliente recurrente** | perfil 3 | perfil 4 |

El perfil no cambia el modelo: cambia que fuentes de candidatos tienen algo que decir. Se
calcula aqui, se arrastra hasta la evaluacion y sirve para desglosar la metrica.

## El instante del corte (`cut_ts`)

Es la marca de tiempo del ultimo `add_to_cart` del prefijo. Todo lo que la sesion sabe
**despues** de ese instante es futuro y no puede entrar como feature. Es lo que permite
usar `session_events` sin filtrar el target: en `cut_ts` hay productos ya vistos y aun no
anadidos, y son justamente los que el ranker tiene que colocar arriba.
"""

from __future__ import annotations

import datetime as dt

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Etiquetas de los cuatro perfiles de CHALLENGE.md, indexadas por `profile`.
PROFILE_LABELS: dict[int, str] = {
    1: "1 - nuevo, carrito vacio",
    2: "2 - nuevo, con articulos",
    3: "3 - recurrente, carrito vacio",
    4: "4 - recurrente, con articulos",
}


def _as_date_literal(value: dt.date | str):
    return F.lit(str(value)).cast("date")


def baskets_before(baskets: DataFrame, until: dt.date | str) -> DataFrame:
    """Cestas anteriores a `until` (exclusivo). Es el historial visible de una ventana."""
    return baskets.filter(F.col("basket_day") < _as_date_literal(until))


def baskets_between(
    baskets: DataFrame, start: dt.date | str, end: dt.date | str | None = None
) -> DataFrame:
    """Cestas de la ventana `[start, end)`. Sin `end`, hasta el final del dataset."""
    out = baskets.filter(F.col("basket_day") >= _as_date_literal(start))
    if end is not None:
        out = out.filter(F.col("basket_day") < _as_date_literal(end))
    return out


def restrict_items(basket_items: DataFrame, baskets: DataFrame) -> DataFrame:
    """Deja solo las lineas de las cestas dadas. El semi-join no anade columnas."""
    return basket_items.join(baskets.select("basket_id"), "basket_id", "left_semi")


def basket_add_to_cart(sessions: DataFrame, session_events: DataFrame) -> DataFrame:
    """Primer `add_to_cart` de cada `(basket_id, product_id)`, via la sesion que convirtio.

    Solo existe para las cestas online que dejaron sesion. Es lo que da orden temporal al
    contenido de la cesta; el resto se ordena por hash.
    """
    converted = sessions.filter(
        F.col("converted") & F.col("basket_id").isNotNull()
    ).select("session_id", "basket_id")
    return (
        session_events.filter(F.col("event_type") == "add_to_cart")
        .join(converted, "session_id")
        .groupBy("basket_id", "product_id")
        .agg(F.min("event_timestamp").alias("add_ts"))
    )


def known_customers(history_baskets: DataFrame) -> DataFrame:
    """Clientes con al menos una compra en el historial: los "recurrentes" del perfil."""
    return (
        history_baskets.filter(F.col("customer_id").isNotNull())
        .select("customer_id")
        .distinct()
    )


def build_query_items(
    window_baskets: DataFrame,
    basket_items: DataFrame,
    add_to_cart: DataFrame | None = None,
) -> DataFrame:
    """Ordena y corta el contenido de cada cesta de la ventana.

    Returns:
        Una fila por `(basket_id, product_id)` con `position` (1..n), `n_items`,
        `prefix_size` e `is_prefix`.
    """
    items = restrict_items(basket_items, window_baskets).select("basket_id", "product_id").distinct()

    if add_to_cart is not None:
        items = items.join(add_to_cart, ["basket_id", "product_id"], "left")
    else:
        items = items.withColumn("add_ts", F.lit(None).cast("timestamp"))

    # Desempate estable: el hash del par depende solo de los identificadores, asi que el
    # orden es el mismo en cualquier ejecucion y en cualquier numero de particiones.
    items = items.withColumn(
        "ord_hash", F.hash(F.concat_ws("|", F.col("basket_id"), F.col("product_id")))
    )
    ordered = Window.partitionBy("basket_id").orderBy(
        F.col("add_ts").asc_nulls_last(), F.col("ord_hash").asc()
    )
    whole = Window.partitionBy("basket_id")

    # La mitad de las cestas se evalua con el carrito vacio y la otra mitad a media
    # compra. El reparto lo fija el hash del `basket_id`, no un sorteo.
    empty_cart = F.pmod(F.hash(F.concat_ws("#", F.col("basket_id"), F.lit("cart"))), F.lit(2)) == 0

    return (
        items.withColumn("position", F.row_number().over(ordered))
        .withColumn("n_items", F.count(F.lit(1)).over(whole).cast("int"))
        .withColumn(
            "prefix_size",
            F.when(F.col("n_items") < 2, F.lit(0))
            .when(empty_cart, F.lit(0))
            # Con la mitad del ticket dentro queda siempre al menos un producto que
            # adivinar, incluso en cestas de dos lineas.
            .otherwise(F.greatest(F.lit(1), F.floor(F.col("n_items") / 2)))
            .cast("int"),
        )
        .withColumn("is_prefix", F.col("position") <= F.col("prefix_size"))
        .select(
            "basket_id", "product_id", "add_ts", "position", "n_items", "prefix_size", "is_prefix"
        )
    )


def build_queries(
    window_baskets: DataFrame,
    query_items: DataFrame,
    history_customers: DataFrame,
    sessions: DataFrame,
    *,
    n_queries: int | None = None,
    salt: str = "query",
) -> DataFrame:
    """Cabecera de cada query: contexto, perfil e instante del corte.

    Args:
        window_baskets: Cestas de la ventana que se va a evaluar o con la que se entrena.
        query_items: Salida de `build_query_items` para esas cestas.
        history_customers: Clientes con compras anteriores a la ventana.
        sessions: Sesiones limpias, para localizar la sesion de cada cesta.
        n_queries: Si se indica, se queda con esa cantidad de cestas. La muestra es
            determinista (por hash del `basket_id`) y uniforme, asi que no sesga la
            metrica; solo reduce el coste.
        salt: Sal del hash de muestreo, para que train y test no elijan "las mismas"
            cestas dentro de sus ventanas.

    Returns:
        Una fila por query con `basket_id`, `customer_id`, `basket_day`, `channel`,
        `n_items`, `prefix_size`, `is_known_customer`, `profile`, `session_id` y `cut_ts`.
    """
    per_basket = query_items.groupBy("basket_id").agg(
        F.first("n_items").alias("n_items"),
        F.first("prefix_size").alias("prefix_size"),
        # Ultimo momento en que el cliente toco el carrito. Nulo si el prefijo esta vacio
        # o si ninguno de sus productos paso por la web.
        F.max(F.when(F.col("is_prefix"), F.col("add_ts"))).alias("prefix_add_ts"),
    )

    session_of_basket = (
        sessions.filter(F.col("converted") & F.col("basket_id").isNotNull())
        .groupBy("basket_id")
        .agg(
            F.min("session_id").alias("session_id"),
            F.min("session_date").alias("session_start"),
        )
    )

    queries = (
        window_baskets.select("basket_id", "customer_id", "basket_day", "basket_date", "channel")
        .join(per_basket, "basket_id")
        .join(F.broadcast(session_of_basket), "basket_id", "left")
        .join(
            history_customers.withColumn("is_known_customer", F.lit(True)),
            "customer_id",
            "left",
        )
        .withColumn("is_known_customer", F.coalesce(F.col("is_known_customer"), F.lit(False)))
        .withColumn("has_cart", F.col("prefix_size") > 0)
        # Cuantos productos hay que adivinar. Es el denominador de Recall@5, y tiene que
        # salir de la cesta real: si se contaran solo los que llegaron al pool de
        # candidatos, el recall ignoraria lo que la primera etapa ni siquiera propuso.
        .withColumn("n_target", (F.col("n_items") - F.col("prefix_size")).cast("int"))
        .withColumn(
            "profile",
            F.when(~F.col("is_known_customer") & ~F.col("has_cart"), F.lit(1))
            .when(~F.col("is_known_customer") & F.col("has_cart"), F.lit(2))
            .when(F.col("is_known_customer") & ~F.col("has_cart"), F.lit(3))
            .otherwise(F.lit(4)),
        )
        # Sin sesion no hay corte que valer: las features de sesion quedaran nulas.
        .withColumn(
            "cut_ts",
            F.when(
                F.col("session_id").isNotNull(),
                F.coalesce(F.col("prefix_add_ts"), F.col("session_start")),
            ),
        )
        .drop("prefix_add_ts", "session_start")
    )

    if n_queries is not None:
        sampled = Window.orderBy(
            F.hash(F.concat_ws("#", F.col("basket_id"), F.lit(salt))).asc(),
            F.col("basket_id").asc(),
        )
        queries = (
            queries.withColumn("_pick", F.row_number().over(sampled))
            .filter(F.col("_pick") <= n_queries)
            .drop("_pick")
        )

    return queries


def prefix_and_target(query_items: DataFrame, queries: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Separa el contenido de cada query en prefijo (lo que ya lleva) y target.

    Returns:
        `(prefix, target)`, ambos con `basket_id` y `product_id`.
    """
    scoped = query_items.join(queries.select("basket_id"), "basket_id", "left_semi")
    prefix = scoped.filter(F.col("is_prefix")).select("basket_id", "product_id", "position")
    target = scoped.filter(~F.col("is_prefix")).select("basket_id", "product_id")
    return prefix, target

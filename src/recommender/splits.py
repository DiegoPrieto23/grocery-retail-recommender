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

**El orden de las lineas no aporta informacion.** En las cestas con sesion, el generador
asigna los `add_to_cart` con una permutacion aleatoria de las lineas del ticket
(`rng.permutation`), y en las demas el orden lo pone un hash. No hay "disparadora antes
que asociada" ni recorrido por secciones: "lo que viene despues del corte" equivale a "el
resto de la cesta", y cualquier `k` lineas de la cesta son un prefijo tan valido como las
`k` primeras. Por eso el problema se evalua como completar la cesta, no como predecir el
siguiente articulo (punto M3 de `docs/diagnostico-fase7.md`).

## Cuantos cortes por cesta (`CutPlan`)

- **Cabecera** (`headline`, el de siempre): un corte por cesta. `prefix_size` es 0 en la
  mitad de las cestas (perfiles de "carrito vacio") y la mitad de las lineas en la otra
  mitad. El reparto lo decide el hash del `basket_id`, no un sorteo: dos ejecuciones
  producen exactamente las mismas queries. Es la cifra de cabecera de `metrics.md` y no
  cambia, para no romper la serie historica.
- **Varios cortes** (`all_prefixes`, `random_fractions`, `empty_and_half`): una query por
  corte. Sirven para ver como evoluciona el acierto a medida que se llena el carrito
  (`reports/recommender/cuts.md`) y para sobremuestrear el cold-start. La clave de la
  query deja de ser el `basket_id` de la cesta y pasa a ser `<basket_id>#k<k>`. La cesta
  real queda en `source_basket_id`. El resto del pipeline trata `basket_id` como la clave
  opaca de la query, asi que no necesita saber cuantos cortes hay.

Con un corte por cesta, `source_basket_id` es igual a `basket_id`.

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

from src.recommender.config import (
    CUT_ALL_PREFIXES,
    CUT_EMPTY_AND_HALF,
    CUT_HEADLINE,
    CUT_RANDOM_FRACTIONS,
    CutPlan,
)
from src.recommender.schema import PROFILE_LABELS  # noqa: F401  (reexportado)

# Separador de la clave de query cuando una cesta da varios cortes: `<basket_id>#k<k>`.
CUT_KEY_SEPARATOR = "#k"


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


def _sample_order(key: str, salt: str) -> tuple:
    """Orden determinista y uniforme: hash de (`key`, `salt`) y, para desempatar, la clave."""
    return (
        F.hash(F.concat_ws("#", F.col(key), F.lit(salt))).asc(),
        F.col(key).asc(),
    )


def sample_baskets(window_baskets: DataFrame, n: int | None, *, salt: str) -> DataFrame:
    """Las `n` primeras cestas de la ventana en el orden de muestreo de `build_queries`.

    Con varios cortes por cesta, la muestra tiene que hacerse antes de explotar las
    cestas en cortes. Si se muestrearan queries, una cesta entraria con unos cortes y sin
    otros. Con la misma `salt` que la muestra de cabecera, las cestas elegidas son las
    primeras de esa misma lista.
    """
    if n is None:
        return window_baskets
    ordered = Window.orderBy(*_sample_order("basket_id", salt))
    return (
        window_baskets.withColumn("_pick", F.row_number().over(ordered))
        .filter(F.col("_pick") <= n)
        .drop("_pick")
    )


def new_customer_baskets(window_baskets: DataFrame, history_customers: DataFrame) -> DataFrame:
    """Cestas de la ventana sin historial previo: anonimas o de clientes nunca vistos antes.

    Son las que caen en los perfiles 1 y 2. Sirve para sobremuestrear el cold-start
    (punto M4) sin tocar la muestra de cabecera.
    """
    known = history_customers.select("customer_id").withColumn("_known", F.lit(True))
    return (
        window_baskets.join(known, "customer_id", "left")
        .filter(F.col("_known").isNull())
        .drop("_known")
    )


def known_customers(history_baskets: DataFrame) -> DataFrame:
    """Clientes con al menos una compra en el historial: los "recurrentes" del perfil."""
    return (
        history_baskets.filter(F.col("customer_id").isNotNull())
        .select("customer_id")
        .distinct()
    )


def _uniform(basket_id, seed: int, draw: int):
    """Uniforme en [0, 1) a partir del hash de (`basket_id`, `seed`, `draw`).

    Se usa un hash y no `F.rand`: el resultado no depende del particionado ni del orden
    de ejecucion, asi que dos ejecuciones sortean los mismos cortes.
    """
    bits = 1 << 31
    return F.pmod(F.xxhash64(basket_id, F.lit(seed), F.lit(draw)), F.lit(bits)) / F.lit(
        float(bits)
    )


def _cut_sizes(baskets: DataFrame, plan: CutPlan) -> DataFrame:
    """`prefix_size` de cada corte: una fila por (`basket_id`, `prefix_size`).

    `baskets` tiene una fila por cesta con `basket_id` y `n_items`. Las cestas de una sola
    linea solo admiten el carrito vacio, porque con cualquier otro corte no quedaria nada
    que adivinar. Por eso no aparecen en `all_prefixes` ni en `random_fractions`.
    """
    n = F.col("n_items")
    half = F.greatest(F.lit(1), F.floor(n / 2))

    if plan.mode == CUT_HEADLINE:
        # La mitad de las cestas se evalua con el carrito vacio y la otra mitad a media
        # compra. El reparto lo fija el hash del `basket_id`, no un sorteo.
        empty_cart = (
            F.pmod(F.hash(F.concat_ws("#", F.col("basket_id"), F.lit("cart"))), F.lit(2)) == 0
        )
        size = (
            F.when(n < 2, F.lit(0))
            .when(empty_cart, F.lit(0))
            # Con la mitad del ticket dentro queda siempre al menos un producto que
            # adivinar, incluso en cestas de dos lineas.
            .otherwise(half)
        )
        return baskets.select("basket_id", size.cast("int").alias("prefix_size"))

    if plan.mode == CUT_EMPTY_AND_HALF:
        sizes = F.when(n < 2, F.array(F.lit(0))).otherwise(F.array(F.lit(0), half))
    elif plan.mode == CUT_ALL_PREFIXES:
        sizes = F.when(n >= 2, F.sequence(F.lit(1), n - 1))
    elif plan.mode == CUT_RANDOM_FRACTIONS:
        # k = 1 + floor(u * (n - 1)) es uniforme en 1..n-1.
        draws = [
            F.lit(1) + F.floor(_uniform(F.col("basket_id"), plan.seed, j) * (n - 1))
            for j in range(plan.n_fractions)
        ]
        sizes = F.when(n >= 2, F.array_distinct(F.array(*draws)))
    else:  # pragma: no cover - CutPlan ya valida el modo
        raise ValueError(plan.mode)

    # `explode` descarta las filas con el array nulo: las cestas de una linea.
    return baskets.select(
        "basket_id", F.explode(sizes.cast("array<int>")).alias("prefix_size")
    )


def build_query_items(
    window_baskets: DataFrame,
    basket_items: DataFrame,
    add_to_cart: DataFrame | None = None,
    *,
    cuts: CutPlan | None = None,
) -> DataFrame:
    """Ordena y corta el contenido de cada cesta de la ventana.

    Args:
        window_baskets: Cestas de la ventana.
        basket_items: Lineas limpias.
        add_to_cart: Primer `add_to_cart` de cada linea (`basket_add_to_cart`), si lo hay.
        cuts: Cortes por cesta. Por defecto, el de cabecera (uno por cesta).

    Returns:
        Una fila por (query, producto) con `basket_id` (la clave de la query),
        `source_basket_id` (la cesta real), `product_id`, `add_ts`, `position` (1..n),
        `n_items`, `prefix_size` e `is_prefix`.
    """
    plan = cuts or CutPlan()
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
    ranked = items.withColumn("position", F.row_number().over(ordered)).withColumn(
        "n_items", F.count(F.lit(1)).over(whole).cast("int")
    )

    sizes = _cut_sizes(ranked.select("basket_id", "n_items").distinct(), plan)
    cut = ranked.join(sizes, "basket_id").withColumnRenamed("basket_id", "source_basket_id")
    key = (
        F.col("source_basket_id")
        if plan.one_per_basket
        else F.concat(
            F.col("source_basket_id"),
            F.lit(CUT_KEY_SEPARATOR),
            F.col("prefix_size").cast("string"),
        )
    )

    return (
        cut.withColumn("basket_id", key)
        .withColumn("is_prefix", F.col("position") <= F.col("prefix_size"))
        .select(
            "basket_id",
            "source_basket_id",
            "product_id",
            "add_ts",
            "position",
            "n_items",
            "prefix_size",
            "is_prefix",
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
        query_items: Salida de `build_query_items` para esas cestas (con uno o varios
            cortes por cesta).
        history_customers: Clientes con compras anteriores a la ventana.
        sessions: Sesiones limpias, para localizar la sesion de cada cesta.
        n_queries: Si se indica, se queda con esa cantidad de queries. La muestra es
            determinista (por hash de la clave de query) y uniforme, asi que no sesga la
            metrica; solo reduce el coste. Con varios cortes por cesta conviene muestrear
            cestas antes (`sample_baskets`) y dejar este argumento en `None`.
        salt: Sal del hash de muestreo, para que train y test no elijan "las mismas"
            cestas dentro de sus ventanas.

    Returns:
        Una fila por query con `basket_id` (clave de la query), `source_basket_id`,
        `customer_id`, `basket_day`, `channel`, `n_items`, `prefix_size`,
        `is_known_customer`, `profile`, `session_id` y `cut_ts`.
    """
    per_query = query_items.groupBy("basket_id").agg(
        F.first("source_basket_id").alias("source_basket_id"),
        F.first("n_items").alias("n_items"),
        F.first("prefix_size").alias("prefix_size"),
        # Ultimo momento en que el cliente toco el carrito. Nulo si el prefijo esta vacio
        # o si ninguno de sus productos paso por la web.
        F.max(F.when(F.col("is_prefix"), F.col("add_ts"))).alias("prefix_add_ts"),
    )

    session_of_basket = (
        sessions.filter(F.col("converted") & F.col("basket_id").isNotNull())
        .groupBy(F.col("basket_id").alias("source_basket_id"))
        .agg(
            F.min("session_id").alias("session_id"),
            F.min("session_date").alias("session_start"),
        )
    )

    # La cabecera de la cesta y su sesion se cruzan por la cesta real. Todo lo demas va
    # por la clave de la query.
    queries = (
        window_baskets.select(
            F.col("basket_id").alias("source_basket_id"),
            "customer_id",
            "basket_day",
            "basket_date",
            "channel",
        )
        .join(per_query, "source_basket_id")
        .join(F.broadcast(session_of_basket), "source_basket_id", "left")
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
        sampled = Window.orderBy(*_sample_order("basket_id", salt))
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

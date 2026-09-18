"""Inferencia del recomendador en pandas, sin Spark (Fase 6b).

Lee las fuentes que dejo `src.serving.export_bundle` en `data/serving/` y sirve el top-5
con el mismo booster LightGBM que entreno la Fase 3. No entrena nada.

## Que garantiza que esto da lo mismo que Spark

Nada, salvo un test. Este modulo es una reimplementacion, y una reimplementacion de 60
features es justo el sitio donde se cuela una diferencia silenciosa: un `left join` que
descarta filas, un nulo que se rellena con 0 en un lado y se queda NaN en el otro, un
desempate distinto al ordenar. Por eso existe `tests/test_serving_parity.py`, que pasa
las queries de test reales por esta ruta y las compara contra
`predictions/recommendations_test.parquet`, que produjo la ruta de Spark.

El orden de las operaciones sigue deliberadamente el de `src/recommender/candidates.py` y
`src/recommender/features.py`, funcion por funcion, para que las dos versiones se puedan
leer en paralelo cuando algo no cuadre.

## Historial al dia de la cesta (punto A1)

El historial del cliente no es una foto del inicio de la ventana: `asof_history` agrega,
para cada query, todas las cestas de su cliente anteriores al dia de la query, igual que
`src/recommender/history.py` en Spark. Por eso el bundle lleva las lineas de todas las
cestas identificadas (`customer_lines`) y no un historial ya agregado. Las formulas del
ciclo de reposicion y del orden de la fuente `hist` no se reescriben aqui: son las de
`src/recommender/formulas.py`, las mismas que usa Spark.

## Diferencia deliberada con el entrenamiento

En la demo no hay sesion: quien juega con la cesta no ha dejado un rastro de `view` /
`add_to_cart` en `session_events`. Las cinco features de sesion se sirven entonces como
"sin sesion" (`has_session=0`), que es exactamente lo que veia el ranker en las cestas de
tienda. `recommend()` acepta igualmente tablas de sesion para que el test de paridad
pueda reproducir las queries que si la tenian.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommender import formulas as fx
from src.recommender import rerank as rr
from src.recommender.config import CandidateConfig, RerankConfig
from src.recommender.formulas import PANDAS_OPS

# Del modulo `schema`, no de `candidates` / `features`: esos importan PySpark, que es justo
# lo que esta ruta evita. Las constantes son las mismas, no una copia.
from src.recommender.schema import (
    CART_FEATURES,
    CATEGORICAL_FEATURES,
    CHANNELS,
    CUSTOMER_CATEGORY_RANK_FEATURES,
    FEATURE_COLUMNS,
    LOYALTY,
    SOURCE_COLUMNS,
    SOURCE_NAMES,
)

SERVING_DIR = Path("data/serving")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

# Todas las columnas de score que aportan las fuentes, en el orden de `SOURCE_COLUMNS`.
ALL_SOURCE_COLUMNS: tuple[str, ...] = tuple(
    column for columns in SOURCE_COLUMNS.values() for column in columns
)

# Columnas de `AsOfHistory.categories`, en el orden de `history.asof_history`.
CATEGORY_HISTORY_COLUMNS: tuple[str, ...] = (
    "basket_id",
    "category",
    "cat_n_purchase_days",
    "cat_last_day",
    "cat_expected_days",
    *CUSTOMER_CATEGORY_RANK_FEATURES,
)

# Mismos rellenos que `build_feature_matrix`: aqui un 0 significa algo ("nunca lo ha
# comprado", "no estaba de oferta"), no un dato que falte.
ZERO_FILLED: dict[str, float] = {
    "hist_n_baskets": 0.0,
    "hist_units": 0.0,
    "hist_ever_bought": 0.0,
    "cat_n_purchase_days": 0.0,
    "cat_due": 0.0,
    "cat_freq_share": 0.0,
    "is_on_promo": 0.0,
    "promo_discount": 0.0,
    "sess_viewed": 0.0,
    "sess_n_views": 0.0,
    "sess_n_events_before": 0.0,
    "n_sources": 0.0,
    "cust_frequency": 0.0,
    "cust_n_products": 0.0,
    **{column: 0.0 for column in CART_FEATURES},
}


@dataclass
class ServingBundle:
    """Las tablas del recomendador, ya en memoria y listas para servir."""

    popularity: pd.DataFrame
    # Popularidad por categoria y mes (punto M6): la rama por categoria de `pop`.
    category_popularity: pd.DataFrame
    affinity_product: pd.DataFrame
    affinity_category: pd.DataFrame
    category_leaders: pd.DataFrame
    # Ficha del cliente al inicio de la ventana servida. Solo la usa la interfaz de la
    # demo (selector y ficha); el modelo recibe el historial as-of de `asof_history`.
    customer_stats: pd.DataFrame
    category_repurchase_days: pd.DataFrame
    known_customers: pd.DataFrame
    als_topn: pd.DataFrame
    products: pd.DataFrame
    customers: pd.DataFrame
    promotions: pd.DataFrame
    booster: object
    window_start: dt.date
    cfg: CandidateConfig
    # Mismas reglas por defecto que `RecommenderConfig.rerank`, que es con lo que se
    # generaron las predicciones contra las que compara el test de paridad.
    rerank: RerankConfig = RerankConfig()

    # Eventos del cliente (lineas y cabeceras de todas sus cestas) y top-N del ALS,
    # indexados por `customer_id`. Sin el indice, cada recomendacion recorreria los ~3 M
    # de lineas; con el, la busqueda del cliente es directa y la demo responde en decimas.
    lines_by_customer: pd.DataFrame = None  # type: ignore[assignment]
    baskets_by_customer: pd.DataFrame = None  # type: ignore[assignment]
    als_by_customer: pd.DataFrame = None  # type: ignore[assignment]

    @property
    def known_customer_ids(self) -> set[str]:
        return set(self.known_customers["customer_id"])


def _by_customer(frame: pd.DataFrame) -> pd.DataFrame:
    """Indexa por `customer_id` y ordena, para que `.loc` sea una busqueda y no un barrido."""
    return frame.set_index("customer_id").sort_index()


def _for_customers(indexed: pd.DataFrame, customer_ids) -> pd.DataFrame:
    """Filas de los clientes pedidos, tolerando los que no estan en la tabla.

    Las columnas categoricas vuelven como texto: el recorte es pequeno y asi los joins
    posteriores contra columnas de texto no cambian de tipo por el camino.
    """
    present = indexed.index.intersection(pd.Index(customer_ids).dropna().unique())
    out = indexed.iloc[:0] if len(present) == 0 else indexed.loc[present]
    out = out.reset_index()
    categorical = [c for c in out.columns if isinstance(out[c].dtype, pd.CategoricalDtype)]
    return out.astype({c: object for c in categorical})


def index_customer_events(
    lines: pd.DataFrame, baskets: pd.DataFrame, products: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara `customer_lines` y `customer_baskets` para buscar por cliente.

    La categoria de cada linea se pega una sola vez aqui, no en cada recomendacion: es un
    join estatico de millones de filas que no depende de la cesta. Los textos van como
    categoricos para que la tabla quepa holgada en memoria; `_for_customers` los devuelve
    como texto al recortar.
    """
    lines = lines.merge(products[["product_id", "category"]], on="product_id")
    lines["basket_day"] = pd.to_datetime(lines["basket_day"])
    lines = lines.astype(
        {"basket_id": "category", "product_id": "category", "category": "category"}
    )
    baskets = baskets.copy()
    baskets["basket_day"] = pd.to_datetime(baskets["basket_day"])
    return _by_customer(lines), _by_customer(baskets)


def load_bundle(
    serving_dir: Path = SERVING_DIR,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
) -> ServingBundle:
    """Carga el bundle exportado. Explica que falta si no esta."""
    import lightgbm as lgb

    if not (serving_dir / "metadata.json").is_file():
        raise FileNotFoundError(
            f"No encuentro {serving_dir}/metadata.json. Genera el bundle con "
            "`python -m src.serving.export_bundle` (necesita PySpark)."
        )
    metadata = json.loads((serving_dir / "metadata.json").read_text(encoding="utf-8"))

    def table(name: str, directory: Path = serving_dir) -> pd.DataFrame:
        return pd.read_parquet(directory / f"{name}.parquet")

    products = table("products_indexed")
    als_topn = table("als_topn")
    lines_by_customer, baskets_by_customer = index_customer_events(
        table("customer_lines"), table("customer_baskets"), products
    )

    return ServingBundle(
        popularity=table("popularity"),
        category_popularity=table("category_popularity"),
        affinity_product=table("affinity_product"),
        affinity_category=table("affinity_category"),
        category_leaders=table("category_leaders"),
        customer_stats=table("customer_stats"),
        category_repurchase_days=table("category_repurchase_days"),
        known_customers=table("known_customers"),
        als_topn=als_topn,
        products=products,
        customers=table("customers", processed_dir),
        promotions=table("promotions", processed_dir),
        booster=lgb.Booster(model_file=str(models_dir / "recommender_ranker_lgbm.txt")),
        window_start=dt.date.fromisoformat(metadata["window_start"]),
        cfg=CandidateConfig(),
        lines_by_customer=lines_by_customer,
        baskets_by_customer=baskets_by_customer,
        als_by_customer=_by_customer(als_topn),
    )


# --------------------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------------------
def build_query(
    basket_id: str,
    customer_id: str | None,
    cart: list[str],
    *,
    basket_day: dt.date,
    channel: str,
    bundle: ServingBundle,
) -> pd.DataFrame:
    """Una query de una sola cesta, con el mismo esquema que `splits.build_queries`.

    El `profile` es el del reto: 1 y 2 para cliente sin historial (carrito vacio o no),
    3 y 4 para cliente conocido.
    """
    is_known = bool(customer_id) and customer_id in bundle.known_customer_ids
    has_prefix = len(cart) > 0
    profile = (3 if is_known else 1) + (1 if has_prefix else 0)
    return pd.DataFrame(
        [
            {
                "basket_id": basket_id,
                "customer_id": customer_id,
                "basket_day": pd.Timestamp(basket_day),
                "channel": channel,
                "profile": profile,
                "prefix_size": float(len(cart)),
                "is_known_customer": float(is_known),
                "session_id": None,
            }
        ]
    )


# --------------------------------------------------------------------------------------
# Fuentes de candidatos (espejo de src/recommender/candidates.py)
# --------------------------------------------------------------------------------------
def candidates_popularity(
    queries: pd.DataFrame, bundle: ServingBundle
) -> pd.DataFrame:
    columns = ["product_id", "month", "pop_score", "pop_rank"]
    top = bundle.popularity.loc[
        bundle.popularity["pop_rank"] <= bundle.cfg.n_popularity, columns
    ]
    catpop = bundle.category_popularity
    by_category = catpop.loc[
        (catpop["cat_pop_rank"] <= bundle.cfg.n_pop_categories)
        & (catpop["cat_prod_rank"] <= bundle.cfg.n_pop_products_per_category),
        columns,
    ]
    top = pd.concat([top, by_category], ignore_index=True).drop_duplicates(
        ["product_id", "month"]
    )
    q = queries[["basket_id", "basket_day"]].copy()
    q["month"] = q["basket_day"].dt.month
    return q.merge(top, on="month")[["basket_id", "product_id", "pop_score", "pop_rank"]]


def candidates_affinity_product(
    prefix: pd.DataFrame, bundle: ServingBundle
) -> pd.DataFrame:
    affinity = bundle.affinity_product[["antecedent", "consequent", "lift", "confidence"]]
    joined = prefix.merge(affinity, left_on="product_id", right_on="antecedent")
    if joined.empty:
        return _empty(["basket_id", "product_id", *SOURCE_COLUMNS["aff"]])

    agg = (
        joined.groupby(["basket_id", "consequent"], as_index=False)
        .agg(
            aff_lift_max=("lift", "max"),
            aff_conf_sum=("confidence", "sum"),
            aff_n_support=("lift", "size"),
        )
        .rename(columns={"consequent": "product_id"})
    )
    agg["aff_n_support"] = agg["aff_n_support"].astype(float)
    ranked = agg.sort_values(
        ["basket_id", "aff_lift_max", "aff_conf_sum", "product_id"],
        ascending=[True, False, False, True],
    )
    keep = ranked.groupby("basket_id").head(bundle.cfg.n_affinity_product)
    return keep[["basket_id", "product_id", *SOURCE_COLUMNS["aff"]]]


def candidates_affinity_category(
    prefix: pd.DataFrame, bundle: ServingBundle
) -> pd.DataFrame:
    cats = (
        prefix.merge(bundle.products[["product_id", "category"]], on="product_id")[
            ["basket_id", "category"]
        ]
        .rename(columns={"category": "antecedent"})
        .drop_duplicates()
    )
    linked = cats.merge(
        bundle.affinity_category[["antecedent", "consequent", "lift", "confidence"]],
        on="antecedent",
    )
    if linked.empty:
        return _empty(["basket_id", "product_id", *SOURCE_COLUMNS["cataff"]])

    top_cats = linked.groupby(["basket_id", "consequent"], as_index=False).agg(
        lift=("lift", "max"), confidence=("confidence", "max")
    )
    top_cats = top_cats.sort_values(
        ["basket_id", "lift", "consequent"], ascending=[True, False, True]
    ).groupby("basket_id").head(bundle.cfg.n_affinity_category)

    leaders = bundle.category_leaders.rename(columns={"category": "consequent"})
    return (
        top_cats.merge(leaders, on="consequent")
        .groupby(["basket_id", "product_id"], as_index=False)
        .agg(cataff_lift_max=("lift", "max"), cataff_conf_max=("confidence", "max"))
    )


@dataclass
class AsOfHistory:
    """Espejo de `src.recommender.history.AsOfHistory`: todo con clave `basket_id`."""

    products: pd.DataFrame
    categories: pd.DataFrame
    customer: pd.DataFrame


def asof_history(queries: pd.DataFrame, bundle: ServingBundle) -> AsOfHistory:
    """Historial de cada query con las cestas de su cliente anteriores al dia de la query.

    Espejo de `src.recommender.history.asof_history`: mismo corte (dia estrictamente
    anterior), mismas agregaciones y las mismas formulas de `formulas.py`.
    """
    q = queries.loc[
        queries["customer_id"].notna(), ["basket_id", "customer_id", "basket_day"]
    ].rename(columns={"basket_day": "_query_day"})
    wanted = q["customer_id"]
    past = {"basket_id": "_past_basket", "basket_day": "_past_day"}

    visits = q.merge(
        _for_customers(bundle.baskets_by_customer, wanted).rename(columns=past),
        on="customer_id",
    )
    visits = visits.loc[visits["_past_day"] < visits["_query_day"]]
    lines = q.merge(
        _for_customers(bundle.lines_by_customer, wanted).rename(columns=past),
        on="customer_id",
    )
    lines = lines.loc[lines["_past_day"] < lines["_query_day"]]

    # --- Cliente ---
    customer = visits.groupby("basket_id", as_index=False).agg(
        cust_frequency=("_past_basket", "nunique"),
        cust_avg_ticket=("total_amount", "mean"),
        cust_last_day=("_past_day", "max"),
    )
    n_products = lines.groupby("basket_id", as_index=False).agg(
        cust_n_products=("product_id", "nunique")
    )
    customer = customer.merge(n_products, on="basket_id", how="left").astype(
        {"cust_frequency": float, "cust_n_products": float}
    )

    # --- Cliente x producto ---
    products = lines.groupby(["basket_id", "product_id"], as_index=False).agg(
        hist_n_baskets=("_past_basket", "nunique"),
        hist_units=("quantity", "sum"),
        hist_last_day=("_past_day", "max"),
    ).astype({"hist_n_baskets": float, "hist_units": float})

    # --- Cliente x categoria ---
    days = lines[["basket_id", "customer_id", "category", "_past_day"]].drop_duplicates()
    if days.empty:
        # Nadie con historial (clientes anonimos o nuevos): tablas vacias con su esquema.
        empty_categories = _empty(CATEGORY_HISTORY_COLUMNS).astype(
            {"cat_last_day": "datetime64[ns]", "cat_expected_days": float}
        )
        return AsOfHistory(products=products, categories=empty_categories, customer=customer)
    categories = (
        days.groupby(["basket_id", "customer_id", "category"], as_index=False)
        .agg(
            _n=("_past_day", "size"),
            cat_last_day=("_past_day", "max"),
            _first_day=("_past_day", "min"),
        )
        .merge(bundle.category_repurchase_days, on="category", how="left")
        .merge(
            bundle.customers[["customer_id", "household_size_est"]],
            on="customer_id",
            how="left",
        )
    )
    n = categories["_n"]
    span = (categories["cat_last_day"] - categories["_first_day"]).dt.days
    categories["cat_n_purchase_days"] = n.astype(float)
    categories["cat_expected_days"] = fx.expected_repurchase_days(
        n,
        fx.mean_gap_days(n, span, PANDAS_OPS),
        categories["typical_repurchase_days"].astype(float),
        fx.household_factor(categories["household_size_est"], PANDAS_OPS),
        PANDAS_OPS,
    )

    # --- Ranking personal de categorias (punto A4) ---
    customer_days = days.groupby("basket_id")["_past_day"].nunique().rename("_customer_days")
    categories = categories.merge(q[["basket_id", "_query_day"]], on="basket_id").merge(
        customer_days, left_on="basket_id", right_index=True
    )
    categories = _with_category_ranks(categories)[list(CATEGORY_HISTORY_COLUMNS)]
    return AsOfHistory(products=products, categories=categories, customer=customer)


def _with_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Espejo de `history.with_category_ranks` (mismo `dense_rank`, mismas formulas)."""
    n = df["cat_n_purchase_days"]
    days_since = (df["_query_day"] - pd.to_datetime(df["cat_last_day"])).dt.days.astype(float)
    ratio = fx.overdue_ratio(days_since, df["cat_expected_days"])
    need = fx.category_need_score(n, fx.is_due(ratio, PANDAS_OPS))
    df["cat_freq_share"] = n / df["_customer_days"].astype(float)
    by_query = df.assign(_need=need).groupby("basket_id")
    df["cat_freq_rank"] = by_query["cat_n_purchase_days"].rank(method="dense", ascending=False)
    df["cat_due_rank"] = by_query["_need"].rank(method="dense", ascending=False)
    return df


def _with_repurchase_state(df: pd.DataFrame) -> pd.DataFrame:
    """Espejo de `history.with_repurchase_state`."""
    df["cat_days_since"] = (
        df["basket_day"] - pd.to_datetime(df["cat_last_day"])
    ).dt.days.astype("float64")
    df["cat_overdue_ratio"] = fx.overdue_ratio(df["cat_days_since"], df["cat_expected_days"])
    df["cat_due"] = fx.is_due(df["cat_overdue_ratio"], PANDAS_OPS)
    return df


def candidates_personal(
    queries: pd.DataFrame, bundle: ServingBundle, history: AsOfHistory
) -> pd.DataFrame:
    scoped = (
        queries[["basket_id", "basket_day"]]
        .merge(history.products, on="basket_id")
        .merge(bundle.products[["product_id", "category"]], on="product_id")
        .merge(history.categories, on=["basket_id", "category"], how="left")
    )
    if scoped.empty:
        return _empty(["basket_id", "product_id", "hist_rank"])

    scoped = _with_repurchase_state(scoped)
    scoped["hist_days_since"] = (
        scoped["basket_day"] - pd.to_datetime(scoped["hist_last_day"])
    ).dt.days
    scoped["_score"] = fx.personal_score(
        scoped["hist_n_baskets"], scoped["cat_due"], scoped["cat_overdue_ratio"], PANDAS_OPS
    )

    ranked = scoped.sort_values(
        ["basket_id", "_score", "hist_days_since", "product_id"],
        ascending=[True, False, True, True],
    )
    ranked["hist_rank"] = ranked.groupby("basket_id").cumcount() + 1.0
    keep = ranked.loc[ranked["hist_rank"] <= bundle.cfg.n_personal]
    return keep[["basket_id", "product_id", "hist_rank"]]


def candidates_als(queries: pd.DataFrame, bundle: ServingBundle) -> pd.DataFrame:
    identified = queries.loc[queries["customer_id"].notna(), ["basket_id", "customer_id"]]
    if identified.empty:
        return _empty(["basket_id", "product_id", *SOURCE_COLUMNS["als"]])
    als = _for_customers(bundle.als_by_customer, identified["customer_id"])
    return identified.merge(als, on="customer_id")[
        ["basket_id", "product_id", *SOURCE_COLUMNS["als"]]
    ]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def union_candidates(
    sources: dict[str, pd.DataFrame], cart: pd.DataFrame
) -> pd.DataFrame:
    """Funde los pools en un candidato por `(basket_id, product_id)`.

    Igual que en Spark: cada fuente deja sus scores, `src_<fuente>` marca quien lo
    propuso y `n_sources` cuenta las coincidencias.
    """
    frames = []
    for name, frame in sources.items():
        if frame.empty:
            continue
        block = frame.copy()
        for column in ALL_SOURCE_COLUMNS:
            if column not in block.columns:
                block[column] = np.nan
        for other in SOURCE_NAMES:
            block[f"src_{other}"] = 1.0 if other == name else 0.0
        frames.append(
            block[
                ["basket_id", "product_id", *ALL_SOURCE_COLUMNS]
                + [f"src_{s}" for s in SOURCE_NAMES]
            ]
        )

    if not frames:
        return _empty(
            ["basket_id", "product_id", *ALL_SOURCE_COLUMNS]
            + [f"src_{s}" for s in SOURCE_NAMES]
            + ["n_sources"]
        )

    pool = pd.concat(frames, ignore_index=True)
    value_columns = [*ALL_SOURCE_COLUMNS] + [f"src_{s}" for s in SOURCE_NAMES]
    merged = pool.groupby(["basket_id", "product_id"], as_index=False)[value_columns].max()

    if not cart.empty:
        merged = merged.merge(
            cart[["basket_id", "product_id"]].drop_duplicates().assign(_in_cart=1),
            on=["basket_id", "product_id"],
            how="left",
        )
        merged = merged.loc[merged["_in_cart"].isna()].drop(columns="_in_cart")

    merged["n_sources"] = merged[[f"src_{s}" for s in SOURCE_NAMES]].sum(axis=1)
    return merged


# --------------------------------------------------------------------------------------
# Features (espejo de features.build_feature_matrix)
# --------------------------------------------------------------------------------------
def _code(series: pd.Series, values: tuple[str, ...]) -> pd.Series:
    """Codifica un dominio cerrado; lo desconocido y lo nulo reciben `len(values)`."""
    mapping = {value: index for index, value in enumerate(values)}
    return series.map(mapping).fillna(len(values)).astype("int32")


def build_feature_matrix(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    bundle: ServingBundle,
    *,
    cart: pd.DataFrame,
    history: AsOfHistory | None = None,
    session_product: pd.DataFrame | None = None,
    session_query: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cruza el pool de candidatos con las familias de features.

    `cart` es lo mismo que se excluye del pool: lo que el cliente tiene en el carrito.
    `history` es el de `asof_history`; si no se pasa, se calcula aqui.
    """
    if history is None:
        history = asof_history(queries, bundle)
    q = queries[
        [
            "basket_id",
            "customer_id",
            "basket_day",
            "channel",
            "profile",
            "prefix_size",
            "is_known_customer",
            "session_id",
        ]
    ]
    df = candidates.merge(q, on="basket_id")
    df["basket_month"] = df["basket_day"].dt.month.astype("int32")
    # Spark cuenta el domingo como 1; pandas cuenta el lunes como 0.
    df["basket_dow"] = (df["basket_day"].dt.dayofweek + 1) % 7 + 1
    df["basket_dow"] = df["basket_dow"].astype("int32")

    # --- Producto: catalogo + popularidad del mes ---
    pop = bundle.popularity.rename(
        columns={"pop_seasonal_index": "prod_seasonal_index", "pop_rank": "prod_month_rank"}
    )[
        [
            "product_id",
            "month",
            "prod_seasonal_index",
            "prod_month_rank",
            "prod_pop_all",
            "prod_pop_recent",
        ]
    ].copy()
    pop["prod_month_rank"] = pop["prod_month_rank"].astype(float)
    df = df.merge(
        pop, left_on=["product_id", "basket_month"], right_on=["product_id", "month"], how="left"
    ).drop(columns="month")
    df = df.merge(bundle.products, on="product_id", how="left")

    # --- Carrito (espejo de features.cart_features) ---
    df = _add_cart(df, cart, bundle.products)

    # --- Cliente x producto, al dia de la cesta ---
    df = df.merge(history.products, on=["basket_id", "product_id"], how="left")
    df["hist_days_since"] = (
        df["basket_day"] - pd.to_datetime(df["hist_last_day"])
    ).dt.days.astype("float64")
    df["hist_ever_bought"] = df["hist_n_baskets"].notna().astype(float)
    df = df.drop(columns="hist_last_day")

    # --- Cliente x categoria: ciclo de reposicion al dia de la cesta ---
    df = df.merge(history.categories, on=["basket_id", "category"], how="left")
    df = _with_repurchase_state(df).drop(columns="cat_last_day")

    # --- Cliente: RFM al dia de la cesta + maestro ---
    df = df.merge(history.customer, on="basket_id", how="left")
    df["cust_recency_days"] = (
        df["basket_day"] - pd.to_datetime(df["cust_last_day"])
    ).dt.days.astype("float64")
    df = df.drop(columns="cust_last_day")
    master = bundle.customers[["customer_id", "household_size_est", "loyalty_tier"]].copy()
    master["household_size_est"] = master["household_size_est"].astype(float)
    df = df.merge(master, on="customer_id", how="left")
    df["loyalty_idx"] = _code(df["loyalty_tier"], LOYALTY)
    df["channel_idx"] = _code(df["channel"], CHANNELS)
    df = df.drop(columns=["loyalty_tier", "channel"])

    # --- Promocion vigente el dia de la cesta ---
    df = df.merge(_active_promotions(df, bundle), on=["basket_id", "product_id"], how="left")

    # --- Sesion ---
    df = _add_session(df, session_product, session_query)

    for column, value in ZERO_FILLED.items():
        # `to_numeric` antes del relleno: algunas columnas llegan como `object` desde los
        # left join vacios, y `fillna` sobre `object` avisa de un downcast que en una
        # version futura de pandas dejaria de ocurrir.
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(value)

    return df[["basket_id", "product_id", "profile", *FEATURE_COLUMNS]]


def _add_cart(df: pd.DataFrame, cart: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Anade `cat_in_cart`, `dept_n_in_cart` y `dept_share_in_cart` (nulos si no aplican)."""
    items = (
        cart[["basket_id", "product_id"]]
        .drop_duplicates()
        .merge(products[["product_id", "category", "department_idx"]], on="product_id")
    )
    if items.empty:
        for column in CART_FEATURES:
            df[column] = np.nan
        return df

    per_category = items[["basket_id", "category"]].drop_duplicates().assign(cat_in_cart=1.0)
    per_department = (
        items.groupby(["basket_id", "department_idx"], as_index=False)
        .size()
        .rename(columns={"size": "dept_n_in_cart"})
    )
    per_department["dept_n_in_cart"] = per_department["dept_n_in_cart"].astype(float)
    cart_size = items.groupby("basket_id").size().astype(float)

    df = df.merge(per_category, on=["basket_id", "category"], how="left").merge(
        per_department, on=["basket_id", "department_idx"], how="left"
    )
    df["dept_share_in_cart"] = df["dept_n_in_cart"] / df["basket_id"].map(cart_size)
    return df


def _active_promotions(df: pd.DataFrame, bundle: ServingBundle) -> pd.DataFrame:
    """Promocion vigente el dia de la cesta, por `(basket_id, product_id)`."""
    promo = bundle.promotions.loc[
        ~bundle.promotions["date_range_invalid"],
        ["product_id", "start_date", "end_date", "discount_value"],
    ]
    pairs = df[["basket_id", "product_id", "basket_day"]].drop_duplicates()
    joined = pairs.merge(promo, on="product_id")
    if joined.empty:
        return _empty(["basket_id", "product_id", "promo_discount", "is_on_promo"])

    active = joined.loc[
        (joined["basket_day"] >= pd.to_datetime(joined["start_date"]))
        & (joined["basket_day"] <= pd.to_datetime(joined["end_date"]))
    ]
    if active.empty:
        return _empty(["basket_id", "product_id", "promo_discount", "is_on_promo"])

    return (
        active.groupby(["basket_id", "product_id"], as_index=False)
        .agg(promo_discount=("discount_value", "max"))
        .assign(is_on_promo=1.0)
    )


def _add_session(
    df: pd.DataFrame,
    session_product: pd.DataFrame | None,
    session_query: pd.DataFrame | None,
) -> pd.DataFrame:
    """Anade las cinco features de sesion. Sin sesion, `has_session` queda a 0."""
    if session_product is not None and not session_product.empty:
        df = df.merge(session_product, on=["basket_id", "product_id"], how="left")
    else:
        for column in ("sess_viewed", "sess_n_views", "sess_secs_since_view"):
            df[column] = np.nan

    if session_query is not None and not session_query.empty:
        df = df.merge(session_query, on="basket_id", how="left")
    else:
        df["sess_n_events_before"] = np.nan

    df["has_session"] = df["session_id"].notna().astype(float)
    return df.drop(columns="session_id")


# --------------------------------------------------------------------------------------
# Puntuacion
# --------------------------------------------------------------------------------------
def score_matrix(matrix: pd.DataFrame, bundle: ServingBundle) -> pd.DataFrame:
    """Puntua con el booster y devuelve la matriz con la columna `score`.

    Las features se pasan en **float32**, igual que en el entrenamiento y en la evaluacion
    (`ranker.collect_for_ranking` las castea en Spark). No es cosmetico: un valor que en
    float64 cae un epsilon por encima del umbral de un corte del arbol puede caer por
    debajo al redondear a float32, y la hoja -- y con ella el score -- cambia. Con el
    dataset de la Fase 8 eso pasaba en 1 de cada 10.000 filas y lo detectaba
    `tests/test_serving_parity.py`.
    """
    if matrix.empty:
        return matrix.assign(score=pd.Series(dtype="float64"))
    features = matrix[list(FEATURE_COLUMNS)].astype("float32")
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("int32")
    return matrix.assign(score=bundle.booster.predict(features))


def rank_queries(
    queries: pd.DataFrame,
    prefix: pd.DataFrame,
    bundle: ServingBundle,
    *,
    cart: pd.DataFrame | None = None,
    session_product: pd.DataFrame | None = None,
    session_query: pd.DataFrame | None = None,
    top_k: int = 5,
    rerank: RerankConfig | None = None,
) -> pd.DataFrame:
    """Ruta completa para un lote de queries: candidatos, features, score y top-k.

    `cart` es lo que se excluye del pool y de lo que salen las features de carrito. Por
    defecto es el propio prefijo; el test de paridad pasa el carrito de la Fase 3, que
    ademas incluye los `add_to_cart` de sesion. `rerank` por defecto es el del bundle.
    """
    history = asof_history(queries, bundle)
    sources = {
        "pop": candidates_popularity(queries, bundle),
        "aff": candidates_affinity_product(prefix, bundle),
        "cataff": candidates_affinity_category(prefix, bundle),
        "hist": candidates_personal(queries, bundle, history),
        "als": candidates_als(queries, bundle),
    }
    exclude = cart if cart is not None else prefix[["basket_id", "product_id"]]
    pool = union_candidates(sources, exclude)
    if pool.empty:
        return _empty(["basket_id", "product_id", "rank", "score"])

    matrix = build_feature_matrix(
        pool,
        queries,
        bundle,
        cart=exclude,
        history=history,
        session_product=session_product,
        session_query=session_query,
    )
    scored = score_matrix(matrix, bundle)
    # La misma funcion que `evaluate.top_k_predictions`: desempate y re-ranking identicos.
    top = rr.top_k(scored, k=top_k, rerank=bundle.rerank if rerank is None else rerank)
    keep = ["basket_id", "product_id", "rank", "score", "profile", "n_sources"]
    keep += [f"src_{s}" for s in SOURCE_NAMES]
    keep += ["cat_due", "cat_overdue_ratio", "hist_ever_bought", "is_on_promo", "aff_lift_max"]
    keep += ["cat_in_cart"]
    return top[[c for c in keep if c in top.columns]].reset_index(drop=True)


def recommend(
    bundle: ServingBundle,
    *,
    customer_id: str | None,
    cart: list[str],
    basket_day: dt.date | None = None,
    channel: str = "app",
    top_k: int = 5,
) -> pd.DataFrame:
    """Top-k para una cesta en curso. Es la funcion que consume la demo.

    `basket_day` por defecto es el primer dia de la ventana servida. Pasar otra fecha
    mueve la estacionalidad, las promociones vigentes y el historial del cliente (todo lo
    que compro antes de ese dia), que es justo lo que hace interesante el selector de la
    demo.
    """
    day = basket_day or bundle.window_start
    queries = build_query(
        "demo", customer_id, cart, basket_day=day, channel=channel, bundle=bundle
    )
    prefix = pd.DataFrame({"basket_id": ["demo"] * len(cart), "product_id": cart})
    return rank_queries(queries, prefix, bundle, top_k=top_k)

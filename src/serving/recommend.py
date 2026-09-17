"""Inferencia del recomendador en pandas, sin Spark (Fase 6b).

Lee las fuentes que dejo `src.serving.export_bundle` en `data/serving/` y sirve el top-5
con el mismo booster LightGBM que entreno la Fase 3. No entrena nada.

## Que garantiza que esto da lo mismo que Spark

Nada, salvo un test. Este modulo es una reimplementacion, y una reimplementacion de 57
features es justo el sitio donde se cuela una diferencia silenciosa: un `left join` que
descarta filas, un nulo que se rellena con 0 en un lado y se queda NaN en el otro, un
desempate distinto al ordenar. Por eso existe `tests/test_serving_parity.py`, que pasa
las queries de test reales por esta ruta y las compara contra
`predictions/recommendations_test.parquet`, que produjo la ruta de Spark.

El orden de las operaciones sigue deliberadamente el de `src/recommender/candidates.py` y
`src/recommender/features.py`, funcion por funcion, para que las dos versiones se puedan
leer en paralelo cuando algo no cuadre.

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

from src.recommender import rerank as rr
from src.recommender.config import CandidateConfig, RerankConfig

# Del modulo `schema`, no de `candidates` / `features`: esos importan PySpark, que es justo
# lo que esta ruta evita. Las constantes son las mismas, no una copia.
from src.recommender.schema import (
    CART_FEATURES,
    CATEGORICAL_FEATURES,
    CHANNELS,
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

# Mismos rellenos que `build_feature_matrix`: aqui un 0 significa algo ("nunca lo ha
# comprado", "no estaba de oferta"), no un dato que falte.
ZERO_FILLED: dict[str, float] = {
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
    **{column: 0.0 for column in CART_FEATURES},
}


@dataclass
class ServingBundle:
    """Las tablas del recomendador, ya en memoria y listas para servir."""

    popularity: pd.DataFrame
    affinity_product: pd.DataFrame
    affinity_category: pd.DataFrame
    category_leaders: pd.DataFrame
    customer_products: pd.DataFrame
    customer_stats: pd.DataFrame
    repurchase: pd.DataFrame
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

    # Vistas indexadas por `customer_id` de las tres tablas grandes. Sin ellas, cada
    # recomendacion recorre los 2,2 M de filas del historial y tarda ~3,5 s; con ellas,
    # la busqueda del cliente es directa y la demo responde en decimas.
    customer_products_by_customer: pd.DataFrame = None  # type: ignore[assignment]
    repurchase_by_customer: pd.DataFrame = None  # type: ignore[assignment]
    als_by_customer: pd.DataFrame = None  # type: ignore[assignment]

    @property
    def known_customer_ids(self) -> set[str]:
        return set(self.known_customers["customer_id"])


def _by_customer(frame: pd.DataFrame) -> pd.DataFrame:
    """Indexa por `customer_id` y ordena, para que `.loc` sea una busqueda y no un barrido."""
    return frame.set_index("customer_id").sort_index()


def _for_customers(indexed: pd.DataFrame, customer_ids) -> pd.DataFrame:
    """Filas de los clientes pedidos, tolerando los que no estan en la tabla."""
    present = indexed.index.intersection(pd.Index(customer_ids).dropna().unique())
    if len(present) == 0:
        return indexed.iloc[:0].reset_index()
    return indexed.loc[present].reset_index()


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
    customer_products = table("customer_products")
    repurchase = table("repurchase")
    als_topn = table("als_topn")

    # La categoria de cada producto se pega una sola vez aqui, no en cada recomendacion:
    # es un join estatico de 2,2 M de filas que no depende de la cesta.
    customer_products_with_category = customer_products.merge(
        products[["product_id", "category"]], on="product_id"
    )

    return ServingBundle(
        popularity=table("popularity"),
        affinity_product=table("affinity_product"),
        affinity_category=table("affinity_category"),
        category_leaders=table("category_leaders"),
        customer_products=customer_products,
        customer_stats=table("customer_stats"),
        repurchase=repurchase,
        known_customers=table("known_customers"),
        als_topn=als_topn,
        products=products,
        customers=table("customers", processed_dir),
        promotions=table("promotions", processed_dir),
        booster=lgb.Booster(model_file=str(models_dir / "recommender_ranker_lgbm.txt")),
        window_start=dt.date.fromisoformat(metadata["window_start"]),
        cfg=CandidateConfig(),
        customer_products_by_customer=_by_customer(customer_products_with_category),
        repurchase_by_customer=_by_customer(repurchase),
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
    top = bundle.popularity.loc[
        bundle.popularity["pop_rank"] <= bundle.cfg.n_popularity,
        ["product_id", "month", "pop_score", "pop_rank"],
    ]
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


def candidates_personal(queries: pd.DataFrame, bundle: ServingBundle) -> pd.DataFrame:
    identified = queries.loc[
        queries["customer_id"].notna(), ["basket_id", "customer_id", "basket_day"]
    ]
    if identified.empty:
        return _empty(["basket_id", "product_id", "hist_rank"])

    # Solo el historial de los clientes de estas queries: la tabla completa son 2,2 M de
    # filas y recorrerla entera en cada recomendacion es lo que hacia lenta la demo.
    wanted = identified["customer_id"]
    due = _for_customers(bundle.repurchase_by_customer, wanted)[
        ["customer_id", "category", "due_for_repurchase", "overdue_ratio"]
    ].rename(
        columns={"due_for_repurchase": "cat_due", "overdue_ratio": "cat_overdue_ratio"}
    )
    due = due.assign(cat_due=due["cat_due"].astype(float))

    per_customer = _for_customers(bundle.customer_products_by_customer, wanted).merge(
        due, on=["customer_id", "category"], how="left"
    )
    per_customer["cat_due"] = per_customer["cat_due"].fillna(0.0)

    scoped = identified.merge(per_customer, on="customer_id")
    if scoped.empty:
        return _empty(["basket_id", "product_id", "hist_rank"])

    scoped["hist_days_since"] = (
        scoped["basket_day"] - pd.to_datetime(scoped["hist_last_day"])
    ).dt.days
    scoped["_score"] = scoped["hist_n_baskets"] * (1.0 + scoped["cat_due"]) + scoped[
        "cat_overdue_ratio"
    ].fillna(0.0)

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
    session_product: pd.DataFrame | None = None,
    session_query: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cruza el pool de candidatos con las familias de features.

    `cart` es lo mismo que se excluye del pool: lo que el cliente tiene en el carrito.
    """
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

    # --- Cliente x producto ---
    # Igual que en las fuentes: se recorta a los clientes de estas queries antes de cruzar.
    customers_here = df["customer_id"]
    hist = _for_customers(bundle.customer_products_by_customer, customers_here)[
        ["customer_id", "product_id", "hist_n_baskets", "hist_units", "hist_last_day"]
    ]
    df = df.merge(hist, on=["customer_id", "product_id"], how="left")
    df["hist_days_since"] = (
        df["basket_day"] - pd.to_datetime(df["hist_last_day"])
    ).dt.days.astype("float64")
    df["hist_ever_bought"] = df["hist_n_baskets"].notna().astype(float)
    df = df.drop(columns="hist_last_day")

    # --- Cliente x categoria: ciclo de reposicion ---
    rep = _for_customers(bundle.repurchase_by_customer, customers_here).rename(
        columns={
            "n_purchase_days": "cat_n_purchase_days",
            "last_purchase_date": "cat_last_day",
            "expected_repurchase_days": "cat_expected_days",
        }
    )[["customer_id", "category", "cat_n_purchase_days", "cat_last_day", "cat_expected_days"]].copy()
    rep["cat_n_purchase_days"] = rep["cat_n_purchase_days"].astype(float)
    df = df.merge(rep, on=["customer_id", "category"], how="left")
    df["cat_days_since"] = (
        df["basket_day"] - pd.to_datetime(df["cat_last_day"])
    ).dt.days.astype("float64")
    df["cat_overdue_ratio"] = df["cat_days_since"] / df["cat_expected_days"]
    # `>= 1.0` sobre un NaN da False en pandas y null en Spark; el relleno a 0 posterior
    # deja las dos versiones en 0, que es lo que quiere decir "no le toca".
    df["cat_due"] = (df["cat_overdue_ratio"] >= 1.0).astype(float)
    df = df.drop(columns="cat_last_day")

    # --- Cliente: RFM de la ventana + maestro ---
    df = df.merge(bundle.customer_stats, on="customer_id", how="left")
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
    """Puntua con el booster y devuelve la matriz con la columna `score`."""
    if matrix.empty:
        return matrix.assign(score=pd.Series(dtype="float64"))
    features = matrix[list(FEATURE_COLUMNS)].copy()
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
    sources = {
        "pop": candidates_popularity(queries, bundle),
        "aff": candidates_affinity_product(prefix, bundle),
        "cataff": candidates_affinity_category(prefix, bundle),
        "hist": candidates_personal(queries, bundle),
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

    `basket_day` por defecto es el primer dia de la ventana servida, que es donde el
    bundle tiene toda su informacion; pasar otra fecha mueve la estacionalidad y las
    promociones vigentes, que es justo lo que hace interesante el selector de la demo.
    """
    day = basket_day or bundle.window_start
    queries = build_query(
        "demo", customer_id, cart, basket_day=day, channel=channel, bundle=bundle
    )
    prefix = pd.DataFrame({"basket_id": ["demo"] * len(cart), "product_id": cart})
    return rank_queries(queries, prefix, bundle, top_k=top_k)

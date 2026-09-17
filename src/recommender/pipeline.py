"""Orquestador de la Fase 3: de `data/processed` a un top-5 evaluado.

    python -m src.recommender.pipeline
    python -m src.recommender.pipeline --no-write        # sin guardar modelo ni predicciones
    python -m src.recommender.pipeline --quick           # muestra pequena, para probar

El flujo es el de `CHALLENGE.md`, Tarea 3a:

1. **Split temporal por cesta** en tres ventanas (`config.py`).
2. **Ajuste de las fuentes de candidatos** sobre el historial de cada ventana. Se hace
   **dos veces**: una con el historial hasta `fit_end` (para las queries con las que se
   entrena el ranker) y otra con el historial hasta `test_start` (para las de test). Es lo
   que impide que el ranker aprenda con features que ya contienen la respuesta. El
   historial personal (fuente `hist` y features de cliente) no se congela: se calcula
   as-of el dia de cada cesta (`history.py`, punto A1).
3. **Generacion de candidatos** y union del pool.
4. **Features** del par `(query, candidato)`.
5. **LambdaRank** con relevancia graduada (punto A4), re-ranking final (`rerank.py`) y
   top-5.
6. **NDCG@5 graduada** (metrica principal), NDCG@5 / Recall@5 de SKU y acierto de
   categoria, en total y por perfil, mas cuatro ablaciones (sin senal de sesion, sin
   features de carrito, relevancia binaria de SKU y sin ranking personal de categorias),
   un baseline de popularidad y los antes/despues de los puntos A1, A2 y A4.
7. **Incertidumbre y cortes** (puntos M3 y M4): intervalo de confianza de las cifras de
   cabecera y bootstrap pareado entre sistemas. Ademas, varios cortes por cesta sobre una
   muestra de test (`cuts.md`) y el cold-start sobremuestreado.

    python -m src.recommender.pipeline --cuts random_fractions --cut-fractions 3
    python -m src.recommender.pipeline --cuts off --no-cold-start   # solo la cabecera

Cualquier cifra que aparezca en el README sale de aqui (`CLAUDE.md`, "Splits y evaluacion").
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import tracking
from src.etl.schemas import read_processed
from src.etl.session import get_spark
from src.recommender import candidates as cand
from src.recommender import evaluate as ev
from src.recommender import features as feat
from src.recommender import history as hs
from src.recommender import ranker as rk
from src.recommender import splits
from src.recommender.config import (
    CUT_ALL_PREFIXES,
    CUT_EMPTY_AND_HALF,
    CUT_RANDOM_FRACTIONS,
    RELEVANCE_BINARY_SKU,
    REQUIRED_TABLES,
    BootstrapConfig,
    CutPlan,
    RecommenderConfig,
    RerankConfig,
)


# F1 del primer puesto de "Instacart Market Basket Analysis" (Kaggle, 2017), redondeado.
# Es una referencia de orden de magnitud, no un benchmark equivalente: ver
# `_kaggle_section`.
INSTACART_TOP_F1 = 0.41

# Metricas de la Fase 3 original (1.500 productos, sin fidelidad de marca), congeladas con
# `git show b3c29ba:reports/recommender/metrics.json`. Es lo que permite que la comparacion
# de la Fase 7c se recalcule en cada ejecucion en vez de copiarse a mano.
BASELINE_FILENAME = "baseline_fase3.json"

# Metricas del LambdaRank antes del punto A2 (sin features de carrito ni re-ranking),
# congeladas por `verify_recommender_diagnostics --freeze` en la Sesion 1.
PRE_A2_FILENAME = "baseline_pre_diagnostico.json"

# Metricas del LambdaRank antes del punto A1 (historial congelado al inicio de la
# ventana), congeladas por `verify_recommender_diagnostics --freeze --snapshot ...`.
PRE_A1_FILENAME = "baseline_pre_a1.json"

# Metricas del LambdaRank antes del punto A4 (relevancia binaria de SKU, sin ranking
# personal de categorias), congeladas por `verify_recommender_diagnostics --freeze
# --snapshot baseline_pre_a4.json`.
PRE_A4_FILENAME = "baseline_pre_a4.json"

# Top-5 de la ablacion con relevancia binaria de SKU, para que el verificador de
# diagnostico la compare con los baselines de categoria sobre las mismas queries.
SKU_PREDICTIONS_FILENAME = "recommendations_test_sku.parquet"

# Informe del desglose por corte (punto M3).
CUTS_REPORT = "cuts.md"
CUTS_JSON = "cuts.json"

# Sistemas que se puntuan sobre cada ventana de test: nombre -> columna de score. Todos
# pasan por el mismo re-ranking servido. El primero es el servido.
LAMBDARANK = "lambdarank"
SYSTEM_SCORES: dict[str, str] = {
    LAMBDARANK: "score",
    "popularidad": "score_popularity",
    "sin_sesion": "score_no_session",
    "sin_carrito": "score_no_cart",
    "relevancia_sku": "score_sku",
    "sin_ranking_personal": "score_no_rank",
}
SYSTEM_NAMES: dict[str, str] = {
    LAMBDARANK: "LambdaRank servido",
    "popularidad": "Popularidad reciente x estacionalidad (mismo pool)",
    "sin_sesion": "LambdaRank sin senal de sesion",
    "sin_carrito": "LambdaRank sin features de carrito",
    "relevancia_sku": "LambdaRank con relevancia binaria de SKU",
    "sin_ranking_personal": "LambdaRank sin ranking personal de categorias",
}

# Metricas que se contrastan entre sistemas (columnas de `evaluate.system_per_query`).
COMPARED_METRICS = ("ndcg_graded", "cat_hit", "sku_hit", "ndcg")

# La seccion de baselines y techo teorico de `metrics.md` la escribe
# `verify_recommender_diagnostics.py`, no este orquestador. Va entre estas marcas para
# que reentrenar no la borre (queda visible hasta que se vuelva a verificar).
DIAGNOSTICS_START = "<!-- diagnostics:start -->"
DIAGNOSTICS_END = "<!-- diagnostics:end -->"


def extract_diagnostics(text: str) -> str:
    """El bloque de diagnostico de un `metrics.md`, marcas incluidas; vacio si no hay."""
    start, end = text.find(DIAGNOSTICS_START), text.find(DIAGNOSTICS_END)
    if start < 0 or end < start:
        return ""
    return text[start : end + len(DIAGNOSTICS_END)]


def with_diagnostics(text: str, block: str) -> str:
    """Sustituye (o anade al final) el bloque de diagnostico de un `metrics.md`."""
    current = extract_diagnostics(text)
    if current:
        return text.replace(current, block)
    return text.rstrip("\n") + "\n\n" + block + "\n" if block else text


class _Timer:
    """Cronometro de etapas, igual que en el ETL de la Fase 2."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.last = self.start

    def step(self, label: str) -> None:
        now = time.perf_counter()
        print(f"  [{now - self.start:6.1f}s] {label} (+{now - self.last:.1f}s)", flush=True)
        self.last = now


@dataclass
class SourceBundle:
    """Las fuentes de candidatos ya ajustadas sobre el historial de una ventana."""

    window_start: dt.date
    popularity: DataFrame
    affinity_product: DataFrame
    affinity_category: DataFrame
    category_leaders: DataFrame
    # Foto cliente x producto de la ventana: solo la usa el ALS. Las features personales
    # se calculan as-of en `build_window`.
    customer_products: DataFrame
    known_customers: DataFrame
    als_model: object
    als_customer_index: DataFrame
    als_product_index: DataFrame

    def unpersist(self) -> None:
        """Suelta las caches de la ventana cuando ya no se van a usar."""
        for frame in (
            self.popularity,
            self.affinity_product,
            self.affinity_category,
            self.category_leaders,
            self.customer_products,
            self.known_customers,
            self.als_customer_index,
            self.als_product_index,
        ):
            frame.unpersist()


def fit_sources(
    tables: dict[str, DataFrame], window_start: dt.date, cfg: RecommenderConfig
) -> SourceBundle:
    """Ajusta las cinco fuentes con todo lo anterior a `window_start`, y nada mas."""
    history_baskets = splits.baskets_before(tables["baskets"], window_start).cache()
    history_items = splits.restrict_items(tables["basket_items"], history_baskets).cache()
    history_baskets.count(), history_items.count()

    popularity = cand.fit_popularity(
        history_baskets, history_items, window_start=window_start, cfg=cfg.candidates
    ).cache()
    affinity_product = cand.fit_affinity_product(history_items).cache()
    affinity_category = cand.fit_affinity_category(history_items, tables["products"]).cache()
    category_leaders = cand.fit_category_leaders(
        popularity, tables["products"], cfg=cfg.candidates
    ).cache()
    customer_products = cand.fit_customer_products(history_baskets, history_items).cache()
    known = splits.known_customers(history_baskets).cache()

    # Materializar aqui evita que cada fuente se recalcule en cada accion posterior.
    for df in (
        popularity,
        affinity_product,
        affinity_category,
        category_leaders,
        customer_products,
        known,
    ):
        df.count()

    als_model, als_customers, als_products = cand.fit_als(customer_products, cfg=cfg.als)

    return SourceBundle(
        window_start=window_start,
        popularity=popularity,
        affinity_product=affinity_product,
        affinity_category=affinity_category,
        category_leaders=category_leaders,
        customer_products=customer_products,
        known_customers=known,
        als_model=als_model,
        als_customer_index=als_customers,
        als_product_index=als_products,
    )


@dataclass
class WindowInputs:
    """Todo lo que define una ventana antes de generar candidatos.

    Existe para que el exportador de la Fase 6b (`src/serving/export_bundle.py`) pueda
    volcar exactamente las mismas queries, carrito y tablas de sesion que uso la Fase 3,
    sin duplicar aqui la logica: el test de paridad de la demo no valdria de nada si la
    ruta de pandas se comparase contra una ventana construida de otra forma.
    """

    queries: DataFrame
    prefix: DataFrame
    target: DataFrame
    cart: DataFrame
    session_product: DataFrame
    session_query: DataFrame


def build_window_inputs(
    tables: dict[str, DataFrame],
    bundle: SourceBundle,
    *,
    start: dt.date,
    end: dt.date | None,
    n_queries: int | None,
    salt: str,
    cuts: CutPlan | None = None,
    new_customers_only: bool = False,
) -> WindowInputs:
    """Queries, prefijo, target, carrito y sesion de una ventana.

    Con `cuts` de varios cortes por cesta, `n_queries` cuenta **cestas** y la muestra se
    toma antes de explotarlas (`splits.sample_baskets`). Con `new_customers_only`, solo
    entran las cestas sin historial previo (perfiles 1 y 2).
    """
    plan = cuts or CutPlan()
    window = splits.baskets_between(tables["baskets"], start, end)
    if new_customers_only:
        window = splits.new_customer_baskets(window, bundle.known_customers)
    if not plan.one_per_basket:
        window = splits.sample_baskets(window, n_queries, salt=salt)
        n_queries = None
    add_to_cart = splits.basket_add_to_cart(tables["sessions"], tables["session_events"])
    query_items = splits.build_query_items(
        window, tables["basket_items"], add_to_cart, cuts=plan
    )

    queries = splits.build_queries(
        window,
        query_items,
        bundle.known_customers,
        tables["sessions"],
        n_queries=n_queries,
        salt=salt,
    ).cache()
    queries.count()

    prefix, target = splits.prefix_and_target(query_items, queries)
    prefix = prefix.cache()

    # La sesion, siempre recortada al instante del corte.
    events_before = feat.session_events_before_cut(
        queries, tables["sessions"], tables["session_events"]
    ).cache()
    events_before.count()
    session_product, session_query = feat.session_features(events_before)
    cart = cand.in_cart(prefix, feat.session_cart(events_before))

    return WindowInputs(
        queries=queries,
        prefix=prefix,
        target=target,
        cart=cart,
        session_product=session_product,
        session_query=session_query,
    )


def build_window(
    tables: dict[str, DataFrame],
    bundle: SourceBundle,
    *,
    start: dt.date,
    end: dt.date | None,
    n_queries: int | None,
    salt: str,
    cfg: RecommenderConfig,
    products_indexed: DataFrame,
    cuts: CutPlan | None = None,
    new_customers_only: bool = False,
) -> tuple[DataFrame, DataFrame, DataFrame, hs.AsOfHistory]:
    """Construye las queries de una ventana, su matriz de features y su contexto.

    Returns:
        `(queries, feature_matrix, context, history)`, donde `context` es una fila por
        producto de cada cesta con `role` = `prefix` (lo que el cliente ya llevaba) o
        `target` (lo que habia que adivinar), y `history` el historial as-of cacheado, que
        el llamador suelta (`history.unpersist()`) cuando ya ha recogido la matriz.
    """
    win = build_window_inputs(
        tables,
        bundle,
        start=start,
        end=end,
        n_queries=n_queries,
        salt=salt,
        cuts=cuts,
        new_customers_only=new_customers_only,
    )
    queries, prefix, target = win.queries, win.prefix, win.target
    session_product, session_query, cart = win.session_product, win.session_query, win.cart

    # Historial personal de cada query al dia de su cesta, con todas las cestas del
    # cliente anteriores a ese dia, tambien las de dentro de la ventana (punto A1).
    history = hs.asof_history(
        queries,
        tables["baskets"],
        tables["basket_items"],
        tables["products"],
        tables["customers"],
    ).cache()

    sources = {
        "pop": cand.candidates_popularity(queries, bundle.popularity, cfg=cfg.candidates),
        "aff": cand.candidates_affinity_product(
            prefix, bundle.affinity_product, cfg=cfg.candidates
        ),
        "cataff": cand.candidates_affinity_category(
            prefix,
            bundle.affinity_category,
            bundle.category_leaders,
            tables["products"],
            cfg=cfg.candidates,
        ),
        "hist": cand.candidates_personal(
            queries, history, tables["products"], cfg=cfg.candidates
        ),
        "als": cand.candidates_als(
            queries,
            bundle.als_model,
            bundle.als_customer_index,
            bundle.als_product_index,
            cfg=cfg.candidates,
        ),
    }
    pool = cand.union_candidates(sources, queries, cart)

    matrix = feat.build_feature_matrix(
        pool,
        queries,
        products=products_indexed,
        popularity=bundle.popularity,
        history=history,
        customers=tables["customers"],
        promotions=tables["promotions"],
        session_product=session_product,
        session_query=session_query,
        cart=cart,
        target=target,
    )
    context = prefix.select("basket_id", "product_id", F.lit("prefix").alias("role")).unionByName(
        target.select("basket_id", "product_id", F.lit("target").alias("role"))
    )
    return queries, matrix, context, history


def cart_rerank_variants(served: RerankConfig) -> list[tuple[str, str, str, RerankConfig]]:
    """Variantes del antes/despues del punto A2: (clave, descripcion, score, reglas)."""
    return [
        (
            "antes",
            "Sin features de carrito, sin re-ranking (antes)",
            "score_no_cart",
            RerankConfig.off(),
        ),
        ("solo_rerank", "Sin features de carrito + re-ranking servido", "score_no_cart", served),
        ("solo_features", "Con features de carrito, sin re-ranking", "score", RerankConfig.off()),
        (
            "features_diversidad",
            "Con features de carrito + 1 por categoria",
            "score",
            RerankConfig(max_per_category=1, exclude_cart_categories=False),
        ),
        (
            "features_diversidad_exclusion",
            "Con features de carrito + 1 por categoria + exclusion del carrito",
            "score",
            RerankConfig(max_per_category=1, exclude_cart_categories=True),
        ),
    ]


def cart_rerank_ablation(
    scored: pd.DataFrame,
    queries: pd.DataFrame,
    target: pd.DataFrame,
    prefix: pd.DataFrame,
    product_category: pd.DataFrame,
    *,
    k: int,
    served: RerankConfig,
) -> pd.DataFrame:
    """Antes/despues del punto A2: cada combinacion de features de carrito y re-ranking.

    Una fila por variante con las metricas de SKU y de categoria (total) y los huecos
    regalados, en total y en las queries con carrito.
    """
    rows = []
    for key, label, score_col, rules in cart_rerank_variants(served):
        top = ev.top_k_predictions(scored, k=k, score_col=score_col, rerank=rules)
        sku = ev.summarise(ev.per_query_metrics(top, queries, k=k), k=k).iloc[0]
        cat = ev.category_metrics(top, target, product_category, queries, k=k).iloc[0]
        wasted = ev.wasted_slot_metrics(top, prefix, product_category, queries, k=k)
        wasted = wasted.set_index("grupo")
        recs = top[["basket_id", "product_id"]].merge(product_category, on="product_id")
        per_list = recs.groupby("basket_id")["category"].agg(["size", "nunique"])
        rows.append(
            {
                "variante": key,
                "descripcion": label,
                "servida": score_col == "score" and rules == served,
                f"cat_hit_rate@{k}": float(cat[f"cat_hit_rate@{k}"]),
                f"sku_hit_rate@{k}": float(cat[f"sku_hit_rate@{k}"]),
                f"ndcg@{k}": float(sku[f"ndcg@{k}"]),
                f"recall@{k}": float(sku[f"recall@{k}"]),
                f"cat_precision@{k}": float(cat[f"cat_precision@{k}"]),
                f"huecos_regalados@{k}": float(wasted.loc["total", f"huecos_regalados@{k}"]),
                f"huecos_repetidos@{k}": float(wasted.loc["total", f"huecos_repetidos@{k}"]),
                f"huecos_en_carrito@{k}_con_carrito": float(
                    wasted.loc[ev.WITH_CART_GROUP, f"huecos_en_carrito@{k}"]
                ),
                f"listas_con_carrito_afectadas@{k}": float(
                    wasted.loc[ev.WITH_CART_GROUP, f"listas_con_regalo@{k}"]
                ),
                f"listas_con_repetida@{k}": float((per_list["nunique"] < per_list["size"]).mean()),
                "categorias_distintas_medias": float(per_list["nunique"].mean()),
            }
        )
    return pd.DataFrame(rows)


# Variantes del antes/despues del punto A4: (clave, descripcion, columna de score).
OBJECTIVE_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("sku", "Relevancia binaria de SKU (objetivo anterior)", "score_sku"),
    ("graduada_sin_ranking", "Relevancia graduada, sin ranking personal", "score_no_rank"),
    ("graduada", "Relevancia graduada + ranking personal (servido)", "score"),
)


def objective_ablation(
    scored: pd.DataFrame,
    queries: pd.DataFrame,
    target: pd.DataFrame,
    product_category: pd.DataFrame,
    *,
    k: int,
    rerank: RerankConfig,
) -> pd.DataFrame:
    """Antes/despues del punto A4: una fila por variante y grupo (total y perfiles).

    Todas con el mismo pool y el mismo re-ranking servido: solo cambia el modelo.
    """
    frames = []
    for key, label, score_col in OBJECTIVE_VARIANTS:
        top = ev.top_k_predictions(scored, k=k, score_col=score_col, rerank=rerank)
        sku = ev.summarise(ev.per_query_metrics(top, queries, k=k), k=k)
        cat = ev.category_metrics(top, target, product_category, queries, k=k)
        merged = cat.merge(sku[["grupo", f"ndcg@{k}", f"recall@{k}"]], on="grupo")
        frames.append(merged.assign(variante=key, descripcion=label))
    return pd.concat(frames, ignore_index=True)


def rank_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Ganancia y puesto (sobre todas las features) del ranking personal de categorias."""
    ranked = importance.sort_values("gain", ascending=False).reset_index(drop=True)
    ranked["puesto"] = ranked.index + 1
    ranked["gain_share"] = ranked["gain"] / ranked["gain"].sum()
    return ranked.loc[
        ranked["feature"].isin(feat.CUSTOMER_CATEGORY_RANK_FEATURES),
        ["feature", "puesto", "gain", "gain_share", "split"],
    ].reset_index(drop=True)


def _queries_to_pandas(queries: DataFrame) -> pd.DataFrame:
    return queries.select(
        "basket_id",
        "customer_id",
        "profile",
        "n_target",
        "prefix_size",
        "basket_day",
        "channel",
        "source_basket_id",
        "n_items",
    ).toPandas()


# Columnas de la matriz que sobreviven a la puntuacion: identificadores, etiqueta, las dos
# del re-ranking y las banderas que explican de donde salio cada candidato. Arrastrar
# todas las features en cada `sort_values` de la evaluacion multiplicaria la memoria sin
# aportar nada.
EXPLAIN_COLUMNS = [f"src_{s}" for s in cand.SOURCE_NAMES] + [
    "n_sources",
    "sess_viewed",
    "is_on_promo",
    "cat_in_cart",
]


@dataclass
class Scorers:
    """Los modelos entrenados, con las features que ve cada uno."""

    served: object
    no_session: object
    no_cart: object
    sku: object
    no_rank: object
    no_session_features: tuple[str, ...]
    no_cart_features: tuple[str, ...]
    no_rank_features: tuple[str, ...]

    def score(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Matriz de features -> una columna de score por sistema (`SYSTEM_SCORES`)."""
        keep = ["basket_id", "product_id", "profile", "label", "category_idx", *EXPLAIN_COLUMNS]
        scored = matrix[keep].copy()
        scored["score"] = rk.score(self.served, matrix)
        scored["score_no_session"] = rk.score(
            self.no_session, matrix, feature_columns=self.no_session_features
        )
        scored["score_no_cart"] = rk.score(
            self.no_cart, matrix, feature_columns=self.no_cart_features
        )
        scored["score_sku"] = rk.score(self.sku, matrix)
        scored["score_no_rank"] = rk.score(
            self.no_rank, matrix, feature_columns=self.no_rank_features
        )
        scored["score_popularity"] = ev.popularity_baseline(matrix)
        return scored


def per_system_tables(
    scored: pd.DataFrame,
    queries: pd.DataFrame,
    target: pd.DataFrame,
    product_category: pd.DataFrame,
    *,
    k: int,
    rerank: RerankConfig,
    systems: tuple[str, ...] = tuple(SYSTEM_SCORES),
) -> dict[str, pd.DataFrame]:
    """Metricas por query de cada sistema (`evaluate.system_per_query`), mismo re-ranking."""
    return {
        name: ev.system_per_query(
            ev.top_k_predictions(scored, k=k, score_col=SYSTEM_SCORES[name], rerank=rerank),
            queries,
            target,
            product_category,
            k=k,
        )
        for name in systems
    }


def compare_systems(
    tables: dict[str, pd.DataFrame], bootstrap: BootstrapConfig, *, k: int
) -> pd.DataFrame:
    """Bootstrap pareado del LambdaRank servido frente a cada otro sistema, total y por perfil."""
    names = ev.system_metrics(k)
    metrics = {column: names[column] for column in COMPARED_METRICS}
    return ev.paired_bootstrap(tables, LAMBDARANK, metrics, bootstrap)


@dataclass
class ExtraWindow:
    """Una evaluacion adicional sobre la ventana de test (cortes o cold-start)."""

    plan: CutPlan
    queries: pd.DataFrame
    per_system: dict[str, pd.DataFrame]
    n_baskets: int


def evaluate_extra_window(
    tables: dict[str, DataFrame],
    bundle: SourceBundle,
    cfg: RecommenderConfig,
    *,
    products_indexed: DataFrame,
    product_category: pd.DataFrame,
    scorers: Scorers,
    cuts: CutPlan,
    n_baskets: int | None,
    salt: str,
    new_customers_only: bool = False,
    systems: tuple[str, ...] = tuple(SYSTEM_SCORES),
) -> ExtraWindow:
    """Construye, puntua y evalua otra muestra de la ventana de test con los mismos modelos.

    Las fuentes de candidatos (`bundle`) y los modelos son los de la evaluacion de
    cabecera; solo cambian las queries.
    """
    queries, matrix, context, history = build_window(
        tables,
        bundle,
        start=cfg.test_start,
        end=None,
        n_queries=n_baskets,
        salt=salt,
        cfg=cfg,
        products_indexed=products_indexed,
        cuts=cuts,
        new_customers_only=new_customers_only,
    )
    pdf = rk.collect_for_ranking(matrix)
    history.unpersist()
    q = _queries_to_pandas(queries)
    ctx = context.toPandas()
    queries.unpersist()
    scored = scorers.score(pdf)
    del pdf
    gc.collect()
    target = ctx.loc[ctx["role"] == "target", ["basket_id", "product_id"]]
    per_system = per_system_tables(
        scored, q, target, product_category, k=cfg.top_k, rerank=cfg.rerank, systems=systems
    )
    return ExtraWindow(
        plan=cuts,
        queries=q,
        per_system=per_system,
        n_baskets=int(q["source_basket_id"].nunique()),
    )


def run(
    spark: SparkSession,
    cfg: RecommenderConfig,
    *,
    write: bool = True,
) -> dict[str, object]:
    """Ejecuta la Fase 3 completa y devuelve modelo, metricas y predicciones."""
    timer = _Timer()

    tables = read_processed(spark, REQUIRED_TABLES, cfg.processed_dir)
    products_indexed = feat.index_products(tables["products"]).cache()
    products_indexed.count()
    timer.step("Lectura de data/processed")

    # --- Ventana del ranker: fuentes hasta `fit_end` ---
    train_bundle = fit_sources(tables, cfg.fit_end, cfg)
    timer.step(f"Fuentes de candidatos ajustadas hasta {cfg.fit_end}")

    n_rank = cfg.n_train_queries + cfg.n_valid_queries
    rank_queries, rank_matrix, _, rank_history = build_window(
        tables,
        train_bundle,
        start=cfg.fit_end,
        end=cfg.test_start,
        n_queries=n_rank,
        salt="ranker",
        cfg=cfg,
        products_indexed=products_indexed,
    )
    rank_pdf = rk.collect_for_ranking(rank_matrix)
    timer.step(f"Matriz del ranker: {len(rank_pdf):,} filas / {rank_queries.count():,} queries")
    rank_history.unpersist()

    # Train y validacion se separan tambien por cesta: una cesta entera cae a un lado.
    baskets_sorted = sorted(rank_pdf["basket_id"].unique())
    valid_ids = set(baskets_sorted[: cfg.n_valid_queries])
    is_valid = rank_pdf["basket_id"].isin(valid_ids)
    # Con la relevancia graduada se quedan las cestas con al menos un candidato de una
    # categoria del target, no solo las que tienen el SKU exacto en el pool.
    positive = cfg.ranker.label_column
    train_pdf = rk.drop_groups_without_positives(
        rank_pdf.loc[~is_valid].reset_index(drop=True), positive
    )
    valid_pdf = rk.drop_groups_without_positives(
        rank_pdf.loc[is_valid].reset_index(drop=True), positive
    )
    del rank_pdf, is_valid  # la matriz completa ya no hace falta y ocupa varios cientos de MB

    booster, evals = rk.train_ranker(train_pdf, valid_pdf, cfg=cfg.ranker)
    timer.step(f"LambdaRank entrenado ({booster.best_iteration} arboles)")

    # Ablacion del punto A4: el objetivo de antes, relevancia binaria de SKU exacto, con
    # las mismas features. Descarta las cestas sin el SKU en el pool, como antes.
    sku_ranker = dataclasses.replace(cfg.ranker, relevance=RELEVANCE_BINARY_SKU)
    train_sku = rk.drop_groups_without_positives(train_pdf, "label")
    valid_sku = rk.drop_groups_without_positives(valid_pdf, "label")
    n_train_queries_sku = int(train_sku["basket_id"].nunique())
    booster_sku, _ = rk.train_ranker(train_sku, valid_sku, cfg=sku_ranker, verbose_eval=0)
    del train_sku, valid_sku
    gc.collect()
    timer.step(f"Ablacion con relevancia binaria de SKU ({booster_sku.best_iteration} arboles)")

    # Y el objetivo nuevo sin el ranking personal de categorias: aisla lo que aporta cada
    # una de las dos piezas del punto A4.
    no_rank = tuple(
        c for c in feat.FEATURE_COLUMNS if c not in feat.CUSTOMER_CATEGORY_RANK_FEATURES
    )
    booster_nr, _ = rk.train_ranker(
        train_pdf, valid_pdf, cfg=cfg.ranker, feature_columns=no_rank, verbose_eval=0
    )
    timer.step("Ablacion sin ranking personal de categorias")

    # Ablacion: el mismo modelo sin ninguna feature de sesion, para poder decir cuanto
    # aporta realmente la senal que obligo a arreglar el generador.
    no_session = tuple(c for c in feat.FEATURE_COLUMNS if c not in feat.SESSION_FEATURES)
    booster_ns, _ = rk.train_ranker(
        train_pdf, valid_pdf, cfg=cfg.ranker, feature_columns=no_session, verbose_eval=0
    )
    timer.step("Ablacion sin senal de sesion")

    # Ablacion del punto A2: el mismo modelo sin las features de carrito. Sin re-ranking
    # reproduce el sistema de antes; con el, aisla lo que aporta cada una de las dos piezas.
    no_cart = tuple(c for c in feat.FEATURE_COLUMNS if c not in feat.CART_FEATURES)
    booster_nc, _ = rk.train_ranker(
        train_pdf, valid_pdf, cfg=cfg.ranker, feature_columns=no_cart, verbose_eval=0
    )
    timer.step("Ablacion sin features de carrito")

    # La ventana del ranker ya no se usa: soltar sus matrices y caches antes de construir
    # la de test, que es la que marca el pico de memoria.
    n_train_queries = int(train_pdf["basket_id"].nunique())
    del train_pdf, valid_pdf
    gc.collect()
    rank_queries.unpersist()
    train_bundle.unpersist()

    # --- Ventana de test: fuentes rehechas con todo lo anterior a `test_start` ---
    test_bundle = fit_sources(tables, cfg.test_start, cfg)
    timer.step(f"Fuentes de candidatos ajustadas hasta {cfg.test_start}")

    test_queries, test_matrix, test_context, test_history = build_window(
        tables,
        test_bundle,
        start=cfg.test_start,
        end=None,
        n_queries=cfg.n_test_queries,
        salt="test",
        cfg=cfg,
        products_indexed=products_indexed,
    )
    test_pdf = rk.collect_for_ranking(test_matrix)
    test_history.unpersist()
    test_q = _queries_to_pandas(test_queries)
    timer.step(f"Matriz de test: {len(test_pdf):,} filas / {len(test_q):,} queries")

    scorers = Scorers(
        served=booster,
        no_session=booster_ns,
        no_cart=booster_nc,
        sku=booster_sku,
        no_rank=booster_nr,
        no_session_features=no_session,
        no_cart_features=no_cart,
        no_rank_features=no_rank,
    )
    scored = scorers.score(test_pdf)
    del test_pdf
    explain = EXPLAIN_COLUMNS

    # Todas las variantes con el mismo re-ranking: la comparacion es de orden, no de reglas.
    rerank = cfg.rerank
    summary, per_query = ev.evaluate(
        scored, test_q, k=cfg.top_k, rerank=rerank, bootstrap=cfg.bootstrap
    )
    summary_ns, _ = ev.evaluate(
        scored, test_q, k=cfg.top_k, score_col="score_no_session", rerank=rerank
    )
    summary_pop, _ = ev.evaluate(
        scored, test_q, k=cfg.top_k, score_col="score_popularity", rerank=rerank
    )
    pool = ev.candidate_recall(scored, test_q)
    importance = rk.feature_importance(booster)
    all_importance = rk.feature_importance(booster, top=len(feat.FEATURE_COLUMNS))

    # Se guardan tambien las banderas de fuente: sin ellas no se puede explicar *por que*
    # se recomendo cada producto, que es justo lo que ensena la demo de los perfiles.
    top_k = ev.top_k_predictions(scored, k=cfg.top_k, rerank=rerank)

    context_pdf = test_context.toPandas()
    target_pdf = context_pdf.loc[context_pdf["role"] == "target", ["basket_id", "product_id"]]
    prefix_pdf = context_pdf.loc[context_pdf["role"] == "prefix", ["basket_id", "product_id"]]
    product_category = tables["products"].select("product_id", "category").toPandas()
    by_category = ev.category_metrics(
        top_k, target_pdf, product_category, test_q, k=cfg.top_k, bootstrap=cfg.bootstrap
    )
    wasted = ev.wasted_slot_metrics(top_k, prefix_pdf, product_category, test_q, k=cfg.top_k)
    cart_ablation = cart_rerank_ablation(
        scored, test_q, target_pdf, prefix_pdf, product_category, k=cfg.top_k, served=rerank
    )
    objective = objective_ablation(
        scored, test_q, target_pdf, product_category, k=cfg.top_k, rerank=rerank
    )
    top_k_sku = ev.top_k_predictions(scored, k=cfg.top_k, score_col="score_sku", rerank=rerank)
    timer.step("Evaluacion")

    # --- Incertidumbre (punto M4): el servido frente a cada sistema, mismas queries ---
    headline_tables = per_system_tables(
        scored, test_q, target_pdf, product_category, k=cfg.top_k, rerank=rerank
    )
    comparisons = compare_systems(headline_tables, cfg.bootstrap, k=cfg.top_k)
    timer.step(f"Bootstrap pareado ({cfg.bootstrap.n_resamples} remuestreos)")
    del scored
    gc.collect()

    extra = dict(
        products_indexed=products_indexed,
        product_category=product_category,
        scorers=scorers,
    )
    # --- Varios cortes por cesta (punto M3) ---
    cut_eval = None
    if cfg.n_cut_baskets:
        # Misma sal que la muestra de cabecera: las cestas son las primeras de su lista.
        cut_eval = evaluate_extra_window(
            tables,
            test_bundle,
            cfg,
            cuts=cfg.cuts,
            n_baskets=cfg.n_cut_baskets,
            salt="test",
            systems=(LAMBDARANK,),
            **extra,
        )
        timer.step(
            f"Cortes ({cfg.cuts.mode}): {len(cut_eval.queries):,} queries / "
            f"{cut_eval.n_baskets:,} cestas"
        )

    # --- Cold-start sobremuestreado (punto M4) ---
    cold_eval = None
    if cfg.cold_start_oversample:
        cold_eval = evaluate_extra_window(
            tables,
            test_bundle,
            cfg,
            cuts=CutPlan(mode=CUT_EMPTY_AND_HALF),
            n_baskets=None,
            salt="cold_start",
            new_customers_only=True,
            **extra,
        )
        timer.step(
            f"Cold-start: {len(cold_eval.queries):,} queries / {cold_eval.n_baskets:,} cestas"
        )
    recommendations = top_k[
        ["basket_id", "product_id", "rank", "score", "label", "profile", *explain]
    ]

    result: dict[str, object] = {
        "booster": booster,
        "booster_no_session": booster_ns,
        "summary": summary,
        "summary_no_session": summary_ns,
        "summary_popularity": summary_pop,
        "candidate_recall": pool,
        "by_category": by_category,
        "wasted_slots": wasted,
        "cart_ablation": cart_ablation,
        "objective_ablation": objective,
        "rank_feature_importance": rank_feature_importance(all_importance),
        "rerank": rerank,
        "feature_importance": importance,
        "per_query": per_query,
        "recommendations": recommendations,
        "valid_ndcg": float(evals["valid"][f"ndcg@{cfg.top_k}"][booster.best_iteration - 1]),
        "n_test_queries": int(len(test_q)),
        "n_train_queries": n_train_queries,
        "n_train_queries_sku": n_train_queries_sku,
        "test_queries": test_q,
        "comparisons": comparisons,
        "bootstrap": cfg.bootstrap,
        "cuts": None if cut_eval is None else cut_report(cut_eval, headline_tables[LAMBDARANK], cfg),
        "cold_start": None if cold_eval is None else cold_start_report(cold_eval, cfg),
    }

    if write:
        rk.save(booster, cfg.models_dir)
        Path(cfg.predictions_dir).mkdir(parents=True, exist_ok=True)
        recommendations.to_parquet(
            Path(cfg.predictions_dir) / "recommendations_test.parquet", index=False
        )
        per_query.to_parquet(
            Path(cfg.predictions_dir) / "recommender_per_query.parquet", index=False
        )
        top_k_sku[["basket_id", "product_id", "rank", "score_sku", "label", "profile"]].to_parquet(
            Path(cfg.predictions_dir) / SKU_PREDICTIONS_FILENAME, index=False
        )
        test_q.to_parquet(
            Path(cfg.predictions_dir) / "recommender_test_queries.parquet", index=False
        )
        context_pdf.to_parquet(
            Path(cfg.predictions_dir) / "recommender_test_context.parquet", index=False
        )
        _write_reports(cfg, result)
        if result["cuts"] is not None:
            _write_cut_report(cfg, result)
        timer.step("Modelo, predicciones e informes escritos")
        _track(cfg, result)

    return result


def _track(cfg: RecommenderConfig, result: dict[str, object]) -> None:
    """Registra el run en MLflow si esta instalado; si no, no hace nada (`src/tracking.py`)."""
    with tracking.track("grocery-recommender", f"lambdarank-{cfg.test_start}") as run:
        run.set_tags({"fase": "3", "tarea": "3a"})
        run.log_params(tracking.recommender_params(cfg))
        run.log_metrics(tracking.recommender_metrics(result, k=cfg.top_k))
        for name in ("metrics.json", "metrics.md"):
            run.log_artifact(Path(cfg.reports_dir) / name)
        run.log_artifact(Path(cfg.models_dir) / rk.MODEL_FILENAME)


def _table(df: pd.DataFrame) -> str:
    """DataFrame a tabla Markdown, con las columnas numericas ya redondeadas."""
    rounded = df.copy()
    for column in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda v: f"{v:.4f}")
    header = "| " + " | ".join(rounded.columns) + " |"
    sep = "| " + " | ".join("---" for _ in rounded.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in rounded.to_numpy()]
    return "\n".join([header, sep, *rows])


def _system_row(name: str, frame: pd.DataFrame, k: int, *, bold: bool = False) -> str:
    """Una fila de la tabla de comparacion entre sistemas."""
    row = frame.iloc[0]
    metrics = ("ndcg", "recall", "precision", "f1", "hit_rate")
    fmt = "**{:.4f}**" if bold else "{:.4f}"
    cells = [fmt.format(row[f"{m}@{k}"]) for m in metrics]
    return "| " + " | ".join([name, *cells]) + " |"


def _sku_category_paragraph(row: pd.Series, k: int) -> str:
    """Lectura de la tabla SKU/categoria, con las cifras de esta ejecucion."""
    cat, sku = row[f"cat_hit_rate@{k}"], row[f"sku_hit_rate@{k}"]
    return (
        f"El sistema acierta la categoria en el **{cat:.1%}** de las cestas y el SKU exacto "
        f"en el **{sku:.1%}**; el cociente entre las dos es **{sku / cat:.1%}**. La "
        "distancia entre las dos columnas mide cuanto del error esta en *elegir la "
        "referencia* y no en *saber que categoria toca*. Desde la Fase 7a el surtido es de "
        "8 referencias por categoria y el cliente repite su referencia preferida con la "
        "lealtad de la categoria (`DATA_SPEC.md`, \"Fidelidad de marca\")."
    )


def _kaggle_section(summary: pd.DataFrame, k: int) -> str:
    """F1@k frente al primer puesto de Instacart, con la salvedad al lado del numero."""
    total = summary.iloc[0]
    f1, f1_basket = total[f"f1@{k}"], total[f"f1@{k}_por_cesta"]
    return f"""## F1@{k} frente a Kaggle "Instacart Market Basket Analysis"

| | F1 |
| --- | ---: |
| Este sistema, F1@{k} (media armonica de Precision@{k} y Recall@{k} medios) | {f1:.4f} |
| Este sistema, F1@{k} por cesta (media del F1 de cada cesta) | {f1_basket:.4f} |
| Instacart, 1er puesto (aprox.) | {INSTACART_TOP_F1:.2f} |

**No es el mismo benchmark y las cifras no se deben leer como una comparacion directa.**
Se ponen juntas solo como orden de magnitud, con estas diferencias de planteamiento:

- **Que se predice.** Instacart pide solo *recompras*: que productos que el usuario ya
  compro antes estaran en su siguiente pedido. Aqui el top-{k} mezcla recompra con
  *descubrimiento* (popularidad, co-compra, ALS), y el target incluye productos que el
  cliente no habia comprado nunca.
- **Tamano de la lista.** En Instacart cada pedido recibe un conjunto de **tamano
  variable**, elegido para maximizar el F1 esperado de ese pedido (F1-maximization),
  incluida la opcion de predecir "ninguno". Aqui la lista es **siempre de {k}**: con
  {total['n_target_medio']:.1f} productos por adivinar de media, la Precision@{k} y el
  Recall@{k} estan acotados por el propio formato, acierte lo que acierte el modelo.
- **Que se optimiza.** El ranker se entrena con LambdaRank para NDCG@{k} graduada (SKU y
  categoria), no para F1.
- **Contexto.** Aqui se predice a mitad de cesta: lo que ya esta en el carrito queda fuera
  del target. Instacart predice el pedido entero.
- **Agregacion.** Instacart promediaba el F1 de cada pedido; la variante mas cercana es la
  segunda fila ("por cesta"), no la de cabecera.
"""


def _baseline_section(cfg: RecommenderConfig, result: dict[str, object], k: int) -> str:
    """Comparacion con la Fase 3 original, si la referencia congelada esta disponible."""
    path = Path(cfg.reports_dir) / BASELINE_FILENAME
    if not path.is_file():
        return ""
    base = json.loads(path.read_text(encoding="utf-8"))
    old, old_cat = base["summary"][0], base["by_category"][0]
    new = result["summary"].iloc[0]  # type: ignore[union-attr]
    new_cat = result["by_category"].iloc[0]  # type: ignore[union-attr]

    # La Fase 3 no guardaba Precision@k, pero es exactamente su `sku_precision@k`.
    old_precision = old_cat[f"sku_precision@{k}"]
    rows = [
        (f"NDCG@{k}", old[f"ndcg@{k}"], new[f"ndcg@{k}"]),
        (f"Recall@{k}", old[f"recall@{k}"], new[f"recall@{k}"]),
        (f"Precision@{k}", old_precision, new[f"precision@{k}"]),
        (f"F1@{k}", ev.harmonic_f1(old_precision, old[f"recall@{k}"]), new[f"f1@{k}"]),
        (f"hit_rate@{k} (SKU)", old[f"hit_rate@{k}"], new[f"hit_rate@{k}"]),
        (f"hit_rate@{k} (categoria)", old_cat[f"cat_hit_rate@{k}"], new_cat[f"cat_hit_rate@{k}"]),
        (
            "SKU / categoria (hit_rate)",
            old_cat[f"sku_hit_rate@{k}"] / old_cat[f"cat_hit_rate@{k}"],
            new_cat[f"sku_hit_rate@{k}"] / new_cat[f"cat_hit_rate@{k}"],
        ),
        # Otra lectura del mismo cociente: de las recomendaciones que aciertan la
        # categoria, cuantas son ademas el SKU exacto. Es el "13 %" de la Fase 3.
        (
            "SKU / categoria (precision)",
            old_cat[f"sku_precision@{k}"] / old_cat[f"cat_precision@{k}"],
            new_cat[f"sku_precision@{k}"] / new_cat[f"cat_precision@{k}"],
        ),
    ]
    lines = [f"| {name} | {a:.4f} | {b:.4f} | {b / a:.2f}x |" for name, a, b in rows]
    return "\n".join(
        [
            "## Frente a la Fase 3 original",
            "",
            "La Fase 3 se entreno sobre el dataset anterior a la Fase 7a (1.500 productos, "
            "~24 referencias por categoria y eleccion de SKU casi aleatoria). Sus cifras "
            f"estan congeladas en `{Path(cfg.reports_dir).as_posix()}/{BASELINE_FILENAME}`. "
            "Mismo codigo, mismas ventanas y mismo numero de queries de test; cambia el dato.",
            "",
            "| Metrica | Fase 3 (dataset viejo) | Fase 7c (dataset nuevo) | Cambio |",
            "| --- | ---: | ---: | ---: |",
            *lines,
            "",
            "Un matiz al leerlo: el catalogo pasa de 1.500 a 496 productos, asi que un "
            "top-5 al azar tambien acierta mas que antes. El baseline de popularidad de la "
            "tabla de comparacion, sobre el mismo pool, es lo que aisla lo que aporta el "
            "ranker.",
        ]
    )


def _rerank_label(rules: RerankConfig) -> str:
    parts = []
    if rules.max_per_category is not None:
        parts.append(f"como maximo {rules.max_per_category} referencia(s) por categoria")
    if rules.exclude_cart_categories:
        parts.append("categorias del carrito relegadas")
    return ", ".join(parts) if parts else "sin re-ranking"


def _asof_section(cfg: RecommenderConfig, result: dict[str, object], k: int) -> str:
    """Antes/despues del punto A1: historial personal as-of el dia de cada cesta."""
    path = Path(cfg.reports_dir) / PRE_A1_FILENAME
    if not path.is_file():
        return ""
    snap = json.loads(path.read_text(encoding="utf-8"))
    old_sku = pd.DataFrame(snap["metrics"]["summary"]).set_index("grupo")
    old_cat = pd.DataFrame(snap["metrics"]["by_category"]).set_index("grupo")
    new_sku = result["summary"].set_index("grupo")  # type: ignore[union-attr]
    new_cat = result["by_category"].set_index("grupo")  # type: ignore[union-attr]

    lines = [
        f"| Grupo | cat_hit_rate@{k} antes | despues | cambio | sku_hit_rate@{k} antes "
        f"| despues | cambio | NDCG@{k} antes | despues |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in new_cat.index:
        if group not in old_cat.index:
            continue
        c0, c1 = old_cat.loc[group, f"cat_hit_rate@{k}"], new_cat.loc[group, f"cat_hit_rate@{k}"]
        s0, s1 = old_cat.loc[group, f"sku_hit_rate@{k}"], new_cat.loc[group, f"sku_hit_rate@{k}"]
        n0, n1 = old_sku.loc[group, f"ndcg@{k}"], new_sku.loc[group, f"ndcg@{k}"]
        lines.append(
            f"| {group} | {c0:.4f} | {c1:.4f} | {(c1 - c0) * 100:+.2f} pp | {s0:.4f} "
            f"| {s1:.4f} | {(s1 - s0) * 100:+.2f} pp | {n0:.4f} | {n1:.4f} |"
        )
    return "\n".join(
        [
            "## Historial al dia de la cesta (punto A1)",
            "",
            "Las features de cliente x producto, cliente x categoria y cliente, y la fuente "
            "`hist` con su `due_for_repurchase`, se calculan con todas las cestas del "
            "cliente anteriores al dia de cada query (`src/recommender/history.py`), "
            "tambien las de dentro de la ventana. ALS, popularidad, afinidades y el perfil "
            "siguen congelados al inicio de cada ventana.",
            "",
            f"\"Antes\" es el modelo con el historial congelado al inicio de la ventana "
            f"(`{Path(cfg.reports_dir).as_posix()}/{PRE_A1_FILENAME}`, {snap['congelado']}, "
            f"commit `{(snap['commit'] or '?')[:7]}`), sobre las mismas queries de test.",
            "",
            *lines,
            "",
            "El detalle de los huecos que caian en categorias recien repuestas esta en la "
            "seccion de diagnostico (`verify_recommender_diagnostics`).",
        ]
    )


def _objective_summary(result: dict[str, object], variant: str, k: int) -> pd.DataFrame:
    """Fila total de una variante de `objective_ablation`, con los nombres de `summarise`."""
    table: pd.DataFrame = result["objective_ablation"]  # type: ignore[assignment]
    row = table.loc[(table["variante"] == variant) & (table["grupo"] == "total")].iloc[0]
    precision, recall = row[f"sku_precision@{k}"], row[f"recall@{k}"]
    return pd.DataFrame(
        [
            {
                f"ndcg@{k}": row[f"ndcg@{k}"],
                f"recall@{k}": recall,
                f"precision@{k}": precision,
                f"f1@{k}": ev.harmonic_f1(precision, recall),
                f"hit_rate@{k}": row[f"sku_hit_rate@{k}"],
            }
        ]
    )


def _objective_section(cfg: RecommenderConfig, result: dict[str, object], k: int) -> str:
    """Antes/despues del punto A4: relevancia graduada y ranking personal de categorias."""
    table: pd.DataFrame = result["objective_ablation"]  # type: ignore[assignment]
    metrics = [
        f"ndcg_graded@{k}",
        f"cat_hit_rate@{k}",
        f"sku_hit_rate@{k}",
        f"ndcg@{k}",
        f"cat_precision@{k}",
        f"sku_precision@{k}",
    ]
    lines = [
        f"| Variante | NDCG@{k} graduada | cat_hit_rate@{k} | sku_hit_rate@{k} | NDCG@{k} SKU "
        f"| cat_precision@{k} | sku_precision@{k} |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    path = Path(cfg.reports_dir) / PRE_A4_FILENAME
    snap = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if snap is not None:
        cat = snap["metrics"]["by_category"][0]
        sku = snap["metrics"]["summary"][0]
        frozen = {**cat, f"ndcg@{k}": sku[f"ndcg@{k}"]}
        cells = [frozen.get(m) for m in metrics]
        lines.append(
            "| Modelo en disco antes de A4 (congelado) | "
            + " | ".join("—" if v is None else f"{v:.4f}" for v in cells)
            + " |"
        )
    total = table.loc[table["grupo"] == "total"]
    for row in total.to_dict(orient="records"):
        cells = [row["descripcion"], *(f"{row[m]:.4f}" for m in metrics)]
        if row["variante"] == "graduada":
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")

    by = table.set_index(["variante", "grupo"])
    profile_lines = [
        f"| Grupo | cat_hit_rate@{k} SKU | graduada | cambio | sku_hit_rate@{k} SKU "
        f"| graduada | cambio | NDCG@{k} graduada SKU | graduada |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in table.loc[table["variante"] == "graduada", "grupo"]:
        before, after = by.loc[("sku", group)], by.loc[("graduada", group)]
        c0, c1 = before[f"cat_hit_rate@{k}"], after[f"cat_hit_rate@{k}"]
        s0, s1 = before[f"sku_hit_rate@{k}"], after[f"sku_hit_rate@{k}"]
        g0, g1 = before[f"ndcg_graded@{k}"], after[f"ndcg_graded@{k}"]
        profile_lines.append(
            f"| {group} | {c0:.4f} | {c1:.4f} | {(c1 - c0) * 100:+.2f} pp | {s0:.4f} "
            f"| {s1:.4f} | {(s1 - s0) * 100:+.2f} pp | {g0:.4f} | {g1:.4f} |"
        )

    def delta(a: str, b: str, metric: str) -> float:
        return (by.loc[(b, "total"), metric] - by.loc[(a, "total"), metric]) * 100

    cat_m, sku_m = f"cat_hit_rate@{k}", f"sku_hit_rate@{k}"
    reading = (
        f"Cambiar solo el objetivo (fila de relevancia binaria a graduada sin ranking) "
        f"mueve cat_hit_rate@{k} **{delta('sku', 'graduada_sin_ranking', cat_m):+.2f} pp** "
        f"y sku_hit_rate@{k} **{delta('sku', 'graduada_sin_ranking', sku_m):+.2f} pp**. "
        f"Anadir el ranking personal mueve cat_hit_rate@{k} "
        f"**{delta('graduada_sin_ranking', 'graduada', cat_m):+.2f} pp** y sku_hit_rate@{k} "
        f"**{delta('graduada_sin_ranking', 'graduada', sku_m):+.2f} pp**. En conjunto, "
        f"frente al objetivo anterior: cat_hit_rate@{k} "
        f"**{delta('sku', 'graduada', cat_m):+.2f} pp**, sku_hit_rate@{k} "
        f"**{delta('sku', 'graduada', sku_m):+.2f} pp**."
    )

    ranks: pd.DataFrame = result["rank_feature_importance"]  # type: ignore[assignment]
    n_features = len(feat.FEATURE_COLUMNS)
    rank_lines = [
        "| Feature | Puesto por ganancia | Ganancia | % de la ganancia total | Splits |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| `{r['feature']}` | {int(r['puesto'])} de {n_features} | {r['gain']:.0f} "
            f"| {r['gain_share']:.1%} | {int(r['split'])} |"
            for r in ranks.to_dict(orient="records")
        ),
    ]

    frozen_note = ""
    if snap is not None:
        frozen_note = (
            f"La fila congelada es el modelo que habia en disco antes de este cambio "
            f"(`{Path(cfg.reports_dir).as_posix()}/{PRE_A4_FILENAME}`, {snap['congelado']}, "
            f"commit `{(snap['commit'] or '?')[:7]}`), que no media la NDCG graduada. La fila "
            '"relevancia binaria" lo reentrena con el codigo actual, que ya incluye el '
            "ranking personal entre sus features, asi que puede diferir un poco."
        )

    return "\n".join(
        [
            "## Objetivo del ranker (punto A4)",
            "",
            "Hasta el punto A4 el LambdaRank optimizaba la NDCG@5 con relevancia binaria de "
            "SKU exacto. Ahora optimiza la graduada (`RankerConfig.relevance`) y tiene tres "
            "features nuevas de ranking personal de categorias "
            "(`CUSTOMER_CATEGORY_RANK_FEATURES`: `cat_freq_rank`, `cat_freq_share`, "
            "`cat_due_rank`). Las variantes se entrenan con las mismas queries y se "
            "evaluan sobre el mismo pool, con el mismo re-ranking servido.",
            "",
            *lines,
            "",
            reading,
            "",
            frozen_note,
            "",
            "### Por perfil: relevancia binaria de SKU frente a la graduada servida",
            "",
            *profile_lines,
            "",
            "### Importancia del ranking personal de categorias",
            "",
            "Puesto entre todas las features del modelo servido (ganancia).",
            "",
            *rank_lines,
            "",
            "La comparacion con los baselines de categoria y con el techo teorico esta en "
            "la seccion de diagnostico, al final (`verify_recommender_diagnostics`).",
        ]
    )


def _cart_section(cfg: RecommenderConfig, result: dict[str, object], k: int) -> str:
    """Antes/despues del punto A2: features de carrito y re-ranking final."""
    table: pd.DataFrame = result["cart_ablation"]  # type: ignore[assignment]
    rules: RerankConfig = result["rerank"]  # type: ignore[assignment]

    def pct(value: float) -> str:
        return f"{value:.1%}"

    lines = [
        f"| Variante | cat_hit_rate@{k} | sku_hit_rate@{k} | NDCG@{k} | Huecos regalados "
        "| Huecos en cat. del carrito (perfiles 2 y 4) | Listas con carrito afectadas "
        "| Listas con cat. repetida | Categorias distintas |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.to_dict(orient="records"):
        cells = [
            row["descripcion"],
            f"{row[f'cat_hit_rate@{k}']:.4f}",
            f"{row[f'sku_hit_rate@{k}']:.4f}",
            f"{row[f'ndcg@{k}']:.4f}",
            pct(row[f"huecos_regalados@{k}"]),
            pct(row[f"huecos_en_carrito@{k}_con_carrito"]),
            pct(row[f"listas_con_carrito_afectadas@{k}"]),
            pct(row[f"listas_con_repetida@{k}"]),
            f"{row['categorias_distintas_medias']:.2f}",
        ]
        if row["servida"]:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")

    before = table.set_index("variante").loc["antes"]
    served = table.loc[table["servida"]]
    delta = ""
    if len(served):
        after = served.iloc[0]
        d_cat = (after[f"cat_hit_rate@{k}"] - before[f"cat_hit_rate@{k}"]) * 100
        d_sku = (after[f"sku_hit_rate@{k}"] - before[f"sku_hit_rate@{k}"]) * 100
        delta = (
            'Frente a la fila "antes" (mismo entrenamiento, sin las tres features ni '
            f"reglas), el sistema servido (en negrita) cambia cat_hit_rate@{k} en "
            f"**{d_cat:+.2f} pp** y sku_hit_rate@{k} en **{d_sku:+.2f} pp**, y los huecos "
            f"regalados pasan del {pct(before[f'huecos_regalados@{k}'])} al "
            f"{pct(after[f'huecos_regalados@{k}'])}."
        )

    frozen = ""
    path = Path(cfg.reports_dir) / PRE_A2_FILENAME
    if path.is_file():
        snap = json.loads(path.read_text(encoding="utf-8"))
        old_cat = snap["metrics"]["by_category"][0]
        old = snap["metrics"]["summary"][0]
        frozen = (
            "Referencia congelada del modelo que habia en disco antes de este cambio "
            f"(`{Path(cfg.reports_dir).as_posix()}/{PRE_A2_FILENAME}`, "
            f"{snap['congelado']}): cat_hit_rate@{k} {old_cat[f'cat_hit_rate@{k}']:.4f}, "
            f"sku_hit_rate@{k} {old_cat[f'sku_hit_rate@{k}']:.4f}, NDCG@{k} "
            f'{old[f"ndcg@{k}"]:.4f}. La fila "antes" lo reentrena con el codigo actual y '
            "puede diferir un poco: el muestreo de columnas de LightGBM depende del numero "
            "de features."
        )

    wasted: pd.DataFrame = result["wasted_slots"]  # type: ignore[assignment]
    return "\n".join(
        [
            "## Carrito y diversidad (punto A2)",
            "",
            f"Con una linea por categoria en cada cesta, un hueco del top-{k} se *regala* si "
            "su categoria ya esta en el carrito (no puede acertar) o ya salio mas arriba en "
            "la lista (de las dos, como mucho acierta una). Dos piezas atacan el problema: "
            "tres features de carrito (`cat_in_cart`, `dept_n_in_cart`, "
            "`dept_share_in_cart`) y un re-ranking final (`src/recommender/rerank.py`). "
            f"Reglas servidas: **{_rerank_label(rules)}** (`RecommenderConfig.rerank`), las "
            "mismas en esta evaluacion, en las tablas de arriba y en la demo.",
            "",
            *lines,
            "",
            'Las filas "sin features de carrito" usan un LambdaRank entrenado aparte con las '
            "mismas queries y sin esas tres columnas. Los huecos en categorias del carrito "
            "se miden contra el prefijo del ticket; la regla de exclusion usa el carrito en "
            "el corte, que ademas incluye los `add_to_cart` de sesion.",
            "",
            delta,
            "",
            frozen,
            "",
            "### Huecos regalados del sistema servido, por perfil",
            "",
            _table(wasted),
        ]
    )


# --------------------------------------------------------------------------------------
# Incertidumbre y cortes (puntos M3 y M4)
# --------------------------------------------------------------------------------------
# Metricas de las tablas de cortes y de cold-start, en el orden en que se ensenan.
REPORTED_METRICS = ("ndcg_graded", "cat_hit", "sku_hit", "ndcg", "recall")


def _reported(k: int) -> dict[str, str]:
    names = ev.system_metrics(k)
    return {column: names[column] for column in REPORTED_METRICS}


def cut_report(
    cut_eval: ExtraWindow, headline: pd.DataFrame, cfg: RecommenderConfig
) -> dict[str, object]:
    """Desglose por corte del LambdaRank servido, con IC por cesta (punto M3).

    `headline` es la tabla por query de la evaluacion de cabecera: sirve de referencia,
    con el corte de siempre, sobre las mismas cestas.
    """
    k = cfg.top_k
    per_query = cut_eval.per_system[LAMBDARANK]
    metrics = _reported(k)
    plain = {"n_target": "n_target_medio"}
    by_size, by_fraction = ev.cut_groups(per_query)

    def table(groups: dict[str, object], frame: pd.DataFrame) -> pd.DataFrame:
        return ev.summarise_by(
            frame, groups, metrics, bootstrap=cfg.bootstrap, plain=plain  # type: ignore[arg-type]
        )

    baskets = set(per_query[ev.CLUSTER_COLUMN])
    same = headline.loc[headline[ev.CLUSTER_COLUMN].isin(baskets)]
    empty = (same["prefix_size"] == 0).to_numpy()
    reference = table(
        {
            "cabecera: carrito vacio": empty,
            "cabecera: mitad del ticket": ~empty,
            "cabecera: las dos": np.ones(len(same), dtype=bool),
        },
        same,
    )
    return {
        "plan": dataclasses.asdict(cut_eval.plan),
        "n_baskets": cut_eval.n_baskets,
        "n_queries": int(len(per_query)),
        "n_headline_baskets": int(len(same)),
        "baskets_in_headline_sample": bool(len(same) == cut_eval.n_baskets),
        "overall": table({"todos los cortes": np.ones(len(per_query), dtype=bool)}, per_query),
        "by_prefix_size": table(by_size, per_query),
        "by_fraction": table(by_fraction, per_query),
        "headline_same_baskets": reference,
    }


def cold_start_report(cold_eval: ExtraWindow, cfg: RecommenderConfig) -> dict[str, object]:
    """Perfiles 1 y 2 sobre todas las cestas sin historial de la ventana (punto M4)."""
    k = cfg.top_k
    served = cold_eval.per_system[LAMBDARANK]
    profile = served["profile"].to_numpy()
    groups = {"perfiles 1 y 2": np.ones(len(served), dtype=bool)}
    for value in sorted(np.unique(profile)):
        groups[splits.PROFILE_LABELS[int(value)]] = profile == value
    queries = cold_eval.queries
    return {
        "n_baskets": cold_eval.n_baskets,
        "n_queries": int(len(served)),
        "n_anonymous_baskets": int(
            queries.loc[queries["customer_id"].isna(), "source_basket_id"].nunique()
        ),
        "n_new_customers": int(queries["customer_id"].dropna().nunique()),
        "summary": ev.summarise_by(
            served,
            groups,
            _reported(k),
            bootstrap=cfg.bootstrap,
            plain={"n_target": "n_target_medio"},
        ),
        "comparisons": compare_systems(cold_eval.per_system, cfg.bootstrap, k=k),
    }


def _fold_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Junta cada metrica con su intervalo en una celda: `valor [bajo, alto]`."""
    out = df.copy()
    for column in df.columns:
        low, high = column + ev.CI_LOW, column + ev.CI_HIGH
        if low in df.columns and high in df.columns:
            out[column] = [
                f"{v:.4f} [{lo:.4f}, {hi:.4f}]"
                for v, lo, hi in zip(df[column], df[low], df[high])
            ]
            out = out.drop(columns=[low, high])
    return out


def _ci_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Tabla Markdown con los intervalos plegados (y, si se indica, solo esas columnas)."""
    if columns is not None:
        keep = [
            c
            for c in df.columns
            if c in columns or any(c == m + suffix for m in columns for suffix in (ev.CI_LOW, ev.CI_HIGH))
        ]
        df = df[keep]
    return _table(_fold_ci(df))


def _is_rate(metric: str) -> bool:
    return "hit_rate" in metric or "precision" in metric or "recall" in metric


def _diff_cell(row: dict) -> str:
    """Diferencia con su intervalo; en puntos porcentuales si la metrica es una tasa."""
    d, lo, hi = row["diferencia"], row["ci_low"], row["ci_high"]
    if _is_rate(row["metrica"]):
        return f"{d * 100:+.2f} pp [{lo * 100:+.2f}, {hi * 100:+.2f}]"
    return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _p_cell(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _comparison_table(comparisons: pd.DataFrame, group: str) -> list[str]:
    """Filas `sistema | diferencia [IC] | p` por metrica, para un grupo."""
    sub = comparisons.loc[comparisons["grupo"] == group]
    metrics = list(dict.fromkeys(sub["metrica"]))
    head = "| Frente a | " + " | ".join(f"{m}: diferencia [IC 95 %] | p" for m in metrics) + " |"
    lines = [head, "| --- |" + " ---: | ---: |" * len(metrics)]
    for name in dict.fromkeys(sub["sistema"]):
        cells = []
        for metric in metrics:
            row = sub.loc[(sub["sistema"] == name) & (sub["metrica"] == metric)].iloc[0]
            cells += [_diff_cell(row.to_dict()), _p_cell(row["p_value"])]
        lines.append(f"| {SYSTEM_NAMES.get(name, name)} | " + " | ".join(cells) + " |")
    return lines


def _bootstrap_note(bootstrap: BootstrapConfig) -> str:
    return (
        f"Entre corchetes, el intervalo de confianza al {bootstrap.confidence:.0%}: "
        f"bootstrap percentil con {bootstrap.n_resamples:,} remuestreos de cestas "
        f"(semilla {bootstrap.seed}; `evaluate.bootstrap_means`)."
    )


def _width_note(by_category: pd.DataFrame, k: int) -> str:
    """Anchura del intervalo en el grupo mas pequeno, para leer la tabla con ella delante."""
    metric = f"cat_hit_rate@{k}"
    if metric + ev.CI_LOW not in by_category.columns:
        return ""
    row = by_category.loc[by_category["n_queries"].idxmin()]
    width = (row[metric + ev.CI_HIGH] - row[metric + ev.CI_LOW]) * 100
    return (
        f"En el grupo mas pequeno ({row['grupo']}, n = {int(row['n_queries']):,}) el "
        f"intervalo de {metric} mide {width:.1f} puntos: las diferencias entre perfiles "
        "pequenos se leen con eso delante. Los perfiles 1 y 2, con muchas mas queries, "
        'estan en "Cold-start sobremuestreado".'
    )


def _comparison_section(result: dict[str, object], k: int) -> str:
    """Diferencias del LambdaRank servido con cada sistema, con IC y p-valor."""
    comparisons: pd.DataFrame = result["comparisons"]  # type: ignore[assignment]
    bootstrap: BootstrapConfig = result["bootstrap"]  # type: ignore[assignment]
    return "\n".join(
        [
            "### Diferencias con intervalo de confianza (bootstrap pareado)",
            "",
            "Diferencia = LambdaRank servido - el otro sistema, sobre las mismas queries y "
            "con el mismo re-ranking. Cada remuestreo sortea las mismas cestas para los dos "
            f"sistemas (`evaluate.paired_bootstrap`, {bootstrap.n_resamples:,} remuestreos). "
            "El p-valor es bilateral y no esta corregido por comparaciones multiples. Las "
            "tasas van en puntos porcentuales. El desglose por perfil esta en "
            "`metrics.json` (`comparisons`).",
            "",
            *_comparison_table(comparisons, "total"),
        ]
    )


def _cold_start_section(result: dict[str, object], k: int) -> str:
    """Perfiles 1 y 2 sobremuestreados: todas las cestas sin historial de la ventana."""
    cold = result["cold_start"]
    if cold is None:
        return ""
    summary: pd.DataFrame = result["by_category"]  # type: ignore[assignment]
    head = summary.set_index("grupo")["n_queries"]
    in_headline = [
        f"{int(head[label]):,} en el perfil {p}"
        for p, label in splits.PROFILE_LABELS.items()
        if p in (1, 2) and label in head.index
    ]
    comparisons: pd.DataFrame = cold["comparisons"]  # type: ignore[index]
    profile_tables: list[str] = []
    for p in (1, 2):
        label = splits.PROFILE_LABELS[p]
        if (comparisons["grupo"] == label).any():
            profile_tables += [
                f"**{label}.**",
                "",
                *_comparison_table(comparisons, label),
                "",
            ]
    return "\n".join(
        [
            "## Cold-start sobremuestreado (punto M4)",
            "",
            f"En la muestra de cabecera el cold-start son pocas queries ({' y '.join(in_headline)}). "
            f"Aqui entran **todas** las cestas de la ventana de test sin compras anteriores a "
            f"{result['test_start']}: {cold['n_baskets']:,} cestas "  # type: ignore[index]
            f"({cold['n_anonymous_baskets']:,} anonimas y el resto de "  # type: ignore[index]
            f"{cold['n_new_customers']:,} clientes nuevos), cada una con los dos cortes de "  # type: ignore[index]
            f"cabecera (carrito vacio y mitad del ticket): {cold['n_queries']:,} queries. "  # type: ignore[index]
            "Las queries de una misma cesta estan correladas, asi que el bootstrap remuestrea "
            "cestas.",
            "",
            "Estos clientes no tienen ninguna cesta antes del inicio del test, asi que no han "
            "entrado en ninguna fuente de candidatos (popularidad, afinidades, ALS) ni en el "
            "entrenamiento del ranker. Son clientes nunca vistos por ningun modelo. Lo unico "
            "que se sabe de ellos es su historial as-of dentro de la propia ventana, igual que "
            "en produccion. Mismos modelos y mismas fuentes que la cabecera.",
            "",
            _ci_table(cold["summary"]),  # type: ignore[index]
            "",
            _bootstrap_note(result["bootstrap"]),  # type: ignore[arg-type]
            "",
            "### Frente a los demas sistemas, por perfil",
            "",
            "Diferencia = LambdaRank servido - el otro sistema (bootstrap pareado por cesta).",
            "",
            *profile_tables,
        ]
    )


def _write_cut_report(cfg: RecommenderConfig, result: dict[str, object]) -> None:
    """`cuts.md` y `cuts.json`: el acierto a medida que se llena el carrito (punto M3)."""
    reports = Path(cfg.reports_dir)
    k = cfg.top_k
    cuts: dict = result["cuts"]  # type: ignore[assignment]
    plan = cuts["plan"]
    by_size: pd.DataFrame = cuts["by_prefix_size"]
    graded, cat = f"ndcg_graded@{k}", f"cat_hit_rate@{k}"
    first, last = by_size.iloc[0], by_size.iloc[-1]
    mode_text = {
        CUT_ALL_PREFIXES: "todos los cortes `k` = 1..n-1 de cada cesta",
        CUT_RANDOM_FRACTIONS: (
            f"{plan['n_fractions']} cortes por cesta con `k` uniforme en 1..n-1 (sorteo por "
            f"hash con semilla {plan['seed']}; si dos sorteos coinciden, cuenta uno)"
        ),
    }.get(plan["mode"], plan["mode"])
    subset = (
        "si: son las primeras de la misma lista"
        if cuts["baskets_in_headline_sample"]
        else f"no del todo ({cuts['n_headline_baskets']:,} de {cuts['n_baskets']:,})"
    )

    text = f"""# Evaluacion con varios cortes por cesta (punto M3)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra se copia a mano. La cifra de
cabecera del recomendador sigue siendo la de [`metrics.md`](metrics.md): un corte por cesta
(carrito vacio o mitad del ticket), para no romper la serie historica. Este informe
responde a otra pregunta: **como cambia el acierto a medida que se llena el carrito**.

## Montaje

| | |
| --- | --- |
| Cortes | {mode_text} (`CutPlan(mode="{plan['mode']}")`) |
| Cestas | {cuts['n_baskets']:,} de la ventana de test, desde {cfg.test_start} (cestas de 2 o mas lineas) |
| Queries | {cuts['n_queries']:,} ({cuts['n_queries'] / max(cuts['n_baskets'], 1):.1f} por cesta) |
| Cestas dentro de la muestra de cabecera | {subset} |
| Modelo y fuentes | los de la cabecera (mismo LambdaRank, mismo re-ranking) |
| Intervalos | bootstrap por cesta, {cfg.bootstrap.n_resamples:,} remuestreos, {cfg.bootstrap.confidence:.0%} |

**El orden de las lineas no aporta informacion.** En las cestas con sesion, el generador
asigna los `add_to_cart` con una permutacion aleatoria del ticket, y en las demas el orden
lo pone un hash (`src/recommender/splits.py`). El prefijo de tamano `k` es, a efectos
practicos, un subconjunto aleatorio de `k` lineas. Un corte "tardio" no es "el final de la
compra": es una cesta con mas contexto y menos que adivinar.

Las queries de una misma cesta estan correladas, asi que `n_cestas` (y no `n_queries`) es
el tamano efectivo de cada fila. Dos efectos mecanicos al leer las tablas:

- al crecer `k` quedan menos productos por adivinar (`n_target_medio`), y el
  `hit_rate` y el `recall` se mueven aunque el modelo no cambie;
- las filas altas de `prefix_size` solo tienen cestas largas, que son otra poblacion.

La fraccion del ticket (`prefix_size / n_items`) compara cestas de distinto tamano en el
mismo punto de la compra, y corrige en parte lo segundo.

## Lectura rapida

Con 1 linea en el carrito, NDCG@{k} graduada {first[graded]:.4f} y cat_hit_rate@{k}
{first[cat]:.4f}. Con {last['grupo']} lineas, {last[graded]:.4f} y {last[cat]:.4f}
(n_target medio {first['n_target_medio']:.1f} frente a {last['n_target_medio']:.1f}).

## Todos los cortes

{_ci_table(cuts['overall'])}

## Por `prefix_size` (lineas ya en el carrito)

{_ci_table(by_size)}

## Por fraccion del ticket ya en el carrito

{_ci_table(cuts['by_fraction'])}

## Referencia: el corte de cabecera sobre las mismas cestas

Las mismas cestas, evaluadas con el corte de siempre (una query por cesta: la mitad con el
carrito vacio y la otra mitad con la mitad del ticket).

{_ci_table(cuts['headline_same_baskets'])}
"""
    (reports / CUTS_REPORT).write_text(text, encoding="utf-8")

    payload = {
        "test_start": str(cfg.test_start),
        "top_k": k,
        "bootstrap": dataclasses.asdict(cfg.bootstrap),
        **{
            key: (value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value)
            for key, value in cuts.items()
        },
    }
    (reports / CUTS_JSON).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_reports(cfg: RecommenderConfig, result: dict[str, object]) -> None:
    """Deja el informe de la fase en `reports/recommender/`."""
    reports = Path(cfg.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    k = cfg.top_k

    summary: pd.DataFrame = result["summary"]  # type: ignore[assignment]
    result = {**result, "test_start": cfg.test_start}
    headline_columns = [
        "grupo",
        "n_queries",
        f"ndcg_graded@{k}",
        f"cat_hit_rate@{k}",
        f"sku_hit_rate@{k}",
    ]
    cuts_line = (
        f"\nEl acierto segun cuantas lineas lleva ya el carrito (varios cortes por cesta, "
        f"punto M3) esta en [`{CUTS_REPORT}`]({CUTS_REPORT}).\n"
        if result.get("cuts") is not None
        else ""
    )

    text = f"""# Recomendador de cesta (Fase 3, Tarea 3a)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra de este informe se copia a
mano: se recalcula ejecutando ese comando.

## Montaje

| | |
| --- | --- |
| Fuentes de candidatos (ranker) | cestas anteriores a {cfg.fit_end} |
| Queries de entrenamiento | {cfg.fit_end} a {cfg.test_start} ({result['n_train_queries']:,} cestas con algun candidato relevante en el pool; {result['n_train_queries_sku']:,} con el SKU exacto) |
| Fuentes de candidatos (test) | cestas anteriores a {cfg.test_start} |
| Queries de test | desde {cfg.test_start} ({result['n_test_queries']:,} cestas) |
| Ranker | LightGBM `lambdarank`, {result['booster'].num_trees()} arboles |
| Objetivo | NDCG@{k} con relevancia graduada: 2 SKU exacto, 1 misma categoria (`label_gain` = {list(cfg.ranker.label_gain)}) |
| NDCG@{k} graduada de validacion | {result['valid_ndcg']:.4f} |

El split es temporal **y por cesta**: ninguna cesta se reparte entre train y test, y las
fuentes de candidatos se reajustan para cada ventana con solo el pasado de esa ventana.

## Metrica principal

La metrica principal del recomendador es la **NDCG@{k} con relevancia graduada**
(`CHALLENGE.md`, punto A4 de `docs/diagnostico-fase7.md`): 3 puntos por hueco si es el SKU
exacto, 1 si solo acierta la categoria. Es la que optimiza el LambdaRank. Se lee siempre
junto a `cat_hit_rate@{k}` (lo que ensena la demo) y `sku_hit_rate@{k}`.

{_ci_table(result['by_category'], headline_columns)}

{_bootstrap_note(cfg.bootstrap)} {_width_note(result['by_category'], k)}
{cuts_line}
{_objective_section(cfg, result, k)}

## Resultado a nivel de SKU

Las metricas de SKU exacto de siempre (NDCG@{k} binaria, recall, precision, F1), con su
intervalo de confianza.

{_ci_table(summary)}

## Comparacion

Mismo pool de candidatos, distinta forma de ordenarlo. Es lo que aisla la aportacion del
ranker de la de la primera etapa.

| Sistema | NDCG@{k} | Recall@{k} | Precision@{k} | F1@{k} | hit_rate@{k} |
| --- | ---: | ---: | ---: | ---: | ---: |
{_system_row("Popularidad reciente x estacionalidad (sin aprendizaje)", result["summary_popularity"], k)}
{_system_row("LambdaRank sin senal de sesion", result["summary_no_session"], k)}
{_system_row("LambdaRank con relevancia binaria de SKU (objetivo anterior)", _objective_summary(result, "sku", k), k)}
{_system_row("**LambdaRank completo**", summary, k, bold=True)}

{_comparison_section(result, k)}

### Por perfil, sin senal de sesion

{_table(result['summary_no_session'])}

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

{_ci_table(result['by_category'])}

{_sku_category_paragraph(result['by_category'].iloc[0], k)}

{_asof_section(cfg, result, k)}

{_cart_section(cfg, result, k)}

{_kaggle_section(summary, k)}

{_baseline_section(cfg, result, k)}

{_cold_start_section(result, k)}

## Techo de la primera etapa

Que parte del target llego siquiera al pool de candidatos. Lo que no esta aqui, el ranker
no lo puede recuperar.

{_table(result['candidate_recall'])}

## Que features usa el ranker

Importancia por ganancia, las {len(result['feature_importance'])} primeras.

{_table(result['feature_importance'])}
"""
    metrics_md = reports / "metrics.md"
    previous = metrics_md.read_text(encoding="utf-8") if metrics_md.is_file() else ""
    metrics_md.write_text(with_diagnostics(text, extract_diagnostics(previous)), encoding="utf-8")

    payload = {
        "fit_end": str(cfg.fit_end),
        "test_start": str(cfg.test_start),
        "top_k": k,
        "n_test_queries": result["n_test_queries"],
        "n_train_queries": result["n_train_queries"],
        "valid_ndcg": result["valid_ndcg"],
        "summary": summary.to_dict(orient="records"),
        "summary_no_session": result["summary_no_session"].to_dict(orient="records"),
        "summary_popularity": result["summary_popularity"].to_dict(orient="records"),
        "candidate_recall": result["candidate_recall"].to_dict(orient="records"),
        "by_category": result["by_category"].to_dict(orient="records"),
        "rerank": {
            "max_per_category": result["rerank"].max_per_category,  # type: ignore[union-attr]
            "exclude_cart_categories": result["rerank"].exclude_cart_categories,  # type: ignore[union-attr]
        },
        "wasted_slots": result["wasted_slots"].to_dict(orient="records"),  # type: ignore[union-attr]
        "cart_ablation": result["cart_ablation"].to_dict(orient="records"),  # type: ignore[union-attr]
        "ranker": {
            "relevance": cfg.ranker.relevance,
            "label_gain": list(cfg.ranker.label_gain),
            "n_train_queries_sku": result["n_train_queries_sku"],
        },
        "objective_ablation": result["objective_ablation"].to_dict(orient="records"),  # type: ignore[union-attr]
        "rank_feature_importance": result["rank_feature_importance"].to_dict(orient="records"),  # type: ignore[union-attr]
        "bootstrap": dataclasses.asdict(cfg.bootstrap),
        "comparisons": result["comparisons"].to_dict(orient="records"),  # type: ignore[union-attr]
    }
    cold = result.get("cold_start")
    if cold is not None:
        payload["cold_start"] = {
            key: (value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value)
            for key, value in cold.items()  # type: ignore[union-attr]
        }
    (reports / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument("--reports", type=Path, default=Path("reports/recommender"))
    parser.add_argument("--no-write", action="store_true", help="No guardar nada en disco.")
    parser.add_argument(
        "--driver-memory",
        default="6g",
        help="Memoria del driver de Spark (ver el comentario de `main`).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Muestra reducida de queries: sirve para comprobar que el flujo corre.",
    )
    parser.add_argument(
        "--cuts",
        choices=(CUT_ALL_PREFIXES, CUT_RANDOM_FRACTIONS, "off"),
        default=CUT_ALL_PREFIXES,
        help="Evaluacion con varios cortes por cesta (punto M3); 'off' la desactiva.",
    )
    parser.add_argument(
        "--cut-baskets",
        type=int,
        default=RecommenderConfig.n_cut_baskets,
        help="Cestas de test que se explotan en cortes.",
    )
    parser.add_argument(
        "--cut-fractions",
        type=int,
        default=CutPlan.n_fractions,
        help="Cortes por cesta con --cuts random_fractions.",
    )
    parser.add_argument(
        "--no-cold-start",
        action="store_true",
        help="No evaluar el cold-start sobremuestreado (punto M4).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BootstrapConfig.n_resamples,
        help="Remuestreos del bootstrap de los intervalos de confianza.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    kwargs: dict = {
        "processed_dir": args.processed,
        "models_dir": args.models,
        "predictions_dir": args.predictions,
        "reports_dir": args.reports,
    }
    kwargs |= {
        "n_cut_baskets": None if args.cuts == "off" else args.cut_baskets,
        "cold_start_oversample": not args.no_cold_start,
        "bootstrap": BootstrapConfig(n_resamples=args.bootstrap),
    }
    if args.cuts != "off":
        kwargs["cuts"] = CutPlan(mode=args.cuts, n_fractions=args.cut_fractions)
    if args.quick:
        kwargs |= {"n_train_queries": 2_000, "n_valid_queries": 500, "n_test_queries": 2_000}
        if kwargs["n_cut_baskets"] is not None:
            kwargs["n_cut_baskets"] = min(kwargs["n_cut_baskets"], 300)
    cfg = RecommenderConfig(**kwargs)

    # 6 GB no es capricho. Con 4 GB las caches de `fit_sources` -- 2,5 M de lineas mas las
    # dos tablas de afinidad -- se desalojan, y cada `count()` posterior vuelve a lanzar la
    # auto-union del calculo de afinidad: la etapa pasa de 4 minutos a mas de 15. El pico
    # de memoria del driver de Python ya no compite, porque `collect_for_ranking` recoge en
    # float32 por Arrow en vez de dejar que pandas consolide en float64.
    spark = get_spark("grocery-retail-recommender", driver_memory=args.driver_memory)
    try:
        result = run(spark, cfg, write=not args.no_write)
        k = cfg.top_k
        total = result["summary"].iloc[0]
        print(
            f"\nNDCG@{k} = {total[f'ndcg@{k}']:.4f} | Recall@{k} = {total[f'recall@{k}']:.4f} "
            f"| hit_rate@{k} = {total[f'hit_rate@{k}']:.4f} "
            f"sobre {result['n_test_queries']:,} cestas de test"
        )
        print("\nPor perfil:")
        print(_fold_ci(result["summary"]).to_string(index=False))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

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
5. **LambdaRank**, re-ranking final (`rerank.py`) y top-5.
6. **NDCG@5 / Recall@5**, en total y por perfil, mas dos ablaciones (sin senal de sesion,
   sin features de carrito), un baseline de popularidad y el antes/despues del punto A2
   (features de carrito y re-ranking).

Cualquier cifra que aparezca en el README sale de aqui (`CLAUDE.md`, "Splits y evaluacion").
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

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
from src.recommender.config import REQUIRED_TABLES, RecommenderConfig, RerankConfig


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
) -> WindowInputs:
    """Queries, prefijo, target, carrito y sesion de una ventana."""
    window = splits.baskets_between(tables["baskets"], start, end)
    add_to_cart = splits.basket_add_to_cart(tables["sessions"], tables["session_events"])
    query_items = splits.build_query_items(window, tables["basket_items"], add_to_cart)

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
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Construye las queries de una ventana, su matriz de features y su contexto.

    Returns:
        `(queries, feature_matrix, context, history)`, donde `context` es una fila por
        producto de cada cesta con `role` = `prefix` (lo que el cliente ya llevaba) o
        `target` (lo que habia que adivinar), y `history` el historial as-of cacheado, que
        el llamador suelta (`history.unpersist()`) cuando ya ha recogido la matriz.
    """
    win = build_window_inputs(
        tables, bundle, start=start, end=end, n_queries=n_queries, salt=salt
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


def _queries_to_pandas(queries: DataFrame) -> pd.DataFrame:
    return queries.select(
        "basket_id", "customer_id", "profile", "n_target", "prefix_size", "basket_day", "channel"
    ).toPandas()


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
    train_pdf = rk.drop_groups_without_positives(rank_pdf.loc[~is_valid].reset_index(drop=True))
    valid_pdf = rk.drop_groups_without_positives(rank_pdf.loc[is_valid].reset_index(drop=True))
    del rank_pdf, is_valid  # la matriz completa ya no hace falta y ocupa varios cientos de MB

    booster, evals = rk.train_ranker(train_pdf, valid_pdf, cfg=cfg.ranker)
    timer.step(f"LambdaRank entrenado ({booster.best_iteration} arboles)")

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

    # A partir de aqui solo se necesitan los identificadores, la etiqueta, los scores, las
    # dos columnas del re-ranking y las banderas que explican de donde salio cada
    # candidato. Arrastrar todas las features en cada `sort_values` de la evaluacion
    # multiplicaria la memoria sin aportar nada.
    explain = [f"src_{s}" for s in cand.SOURCE_NAMES] + [
        "n_sources",
        "sess_viewed",
        "is_on_promo",
        "cat_in_cart",
    ]
    keep = ["basket_id", "product_id", "profile", "label", "category_idx", *explain]
    scored = test_pdf[keep].copy()
    scored["score"] = rk.score(booster, test_pdf)
    scored["score_no_session"] = rk.score(booster_ns, test_pdf, feature_columns=no_session)
    scored["score_no_cart"] = rk.score(booster_nc, test_pdf, feature_columns=no_cart)
    scored["score_popularity"] = ev.popularity_baseline(test_pdf)
    del test_pdf

    # Todas las variantes con el mismo re-ranking: la comparacion es de orden, no de reglas.
    rerank = cfg.rerank
    summary, per_query = ev.evaluate(scored, test_q, k=cfg.top_k, rerank=rerank)
    summary_ns, _ = ev.evaluate(
        scored, test_q, k=cfg.top_k, score_col="score_no_session", rerank=rerank
    )
    summary_pop, _ = ev.evaluate(
        scored, test_q, k=cfg.top_k, score_col="score_popularity", rerank=rerank
    )
    pool = ev.candidate_recall(scored, test_q)
    importance = rk.feature_importance(booster)

    # Se guardan tambien las banderas de fuente: sin ellas no se puede explicar *por que*
    # se recomendo cada producto, que es justo lo que ensena la demo de los perfiles.
    top_k = ev.top_k_predictions(scored, k=cfg.top_k, rerank=rerank)

    context_pdf = test_context.toPandas()
    target_pdf = context_pdf.loc[context_pdf["role"] == "target", ["basket_id", "product_id"]]
    prefix_pdf = context_pdf.loc[context_pdf["role"] == "prefix", ["basket_id", "product_id"]]
    product_category = tables["products"].select("product_id", "category").toPandas()
    by_category = ev.category_metrics(top_k, target_pdf, product_category, test_q, k=cfg.top_k)
    wasted = ev.wasted_slot_metrics(top_k, prefix_pdf, product_category, test_q, k=cfg.top_k)
    cart_ablation = cart_rerank_ablation(
        scored, test_q, target_pdf, prefix_pdf, product_category, k=cfg.top_k, served=rerank
    )
    timer.step("Evaluacion")
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
        "rerank": rerank,
        "feature_importance": importance,
        "per_query": per_query,
        "recommendations": recommendations,
        "valid_ndcg": float(evals["valid"][f"ndcg@{cfg.top_k}"][booster.best_iteration - 1]),
        "n_test_queries": int(len(test_q)),
        "n_train_queries": n_train_queries,
        "test_queries": test_q,
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
        test_q.to_parquet(
            Path(cfg.predictions_dir) / "recommender_test_queries.parquet", index=False
        )
        context_pdf.to_parquet(
            Path(cfg.predictions_dir) / "recommender_test_context.parquet", index=False
        )
        _write_reports(cfg, result)
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
- **Que se optimiza.** El ranker se entrena con LambdaRank para NDCG@{k}, no para F1.
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


def _write_reports(cfg: RecommenderConfig, result: dict[str, object]) -> None:
    """Deja el informe de la fase en `reports/recommender/`."""
    reports = Path(cfg.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    k = cfg.top_k

    summary: pd.DataFrame = result["summary"]  # type: ignore[assignment]
    total = summary.iloc[0]

    text = f"""# Recomendador de cesta (Fase 3, Tarea 3a)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra de este informe se copia a
mano: se recalcula ejecutando ese comando.

## Montaje

| | |
| --- | --- |
| Fuentes de candidatos (ranker) | cestas anteriores a {cfg.fit_end} |
| Queries de entrenamiento | {cfg.fit_end} a {cfg.test_start} ({result['n_train_queries']:,} cestas con acierto en el pool) |
| Fuentes de candidatos (test) | cestas anteriores a {cfg.test_start} |
| Queries de test | desde {cfg.test_start} ({result['n_test_queries']:,} cestas) |
| Ranker | LightGBM `lambdarank`, {result['booster'].num_trees()} arboles |
| NDCG@{k} de validacion | {result['valid_ndcg']:.4f} |

El split es temporal **y por cesta**: ninguna cesta se reparte entre train y test, y las
fuentes de candidatos se reajustan para cada ventana con solo el pasado de esa ventana.

## Resultado

{_table(summary)}

## Comparacion

Mismo pool de candidatos, distinta forma de ordenarlo. Es lo que aisla la aportacion del
ranker de la de la primera etapa.

| Sistema | NDCG@{k} | Recall@{k} | Precision@{k} | F1@{k} | hit_rate@{k} |
| --- | ---: | ---: | ---: | ---: | ---: |
{_system_row("Popularidad reciente x estacionalidad (sin aprendizaje)", result["summary_popularity"], k)}
{_system_row("LambdaRank sin senal de sesion", result["summary_no_session"], k)}
{_system_row("**LambdaRank completo**", summary, k, bold=True)}

### Por perfil, sin senal de sesion

{_table(result['summary_no_session'])}

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

{_table(result['by_category'])}

{_sku_category_paragraph(result['by_category'].iloc[0], k)}

{_asof_section(cfg, result, k)}

{_cart_section(cfg, result, k)}

{_kaggle_section(summary, k)}

{_baseline_section(cfg, result, k)}

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    kwargs: dict = {
        "processed_dir": args.processed,
        "models_dir": args.models,
        "predictions_dir": args.predictions,
        "reports_dir": args.reports,
    }
    if args.quick:
        kwargs |= {"n_train_queries": 2_000, "n_valid_queries": 500, "n_test_queries": 2_000}
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
        print(result["summary"].to_string(index=False))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

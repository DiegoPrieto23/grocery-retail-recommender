"""Orquestador de la Fase 3: de `data/processed` a un top-5 evaluado.

    python -m src.recommender.pipeline
    python -m src.recommender.pipeline --no-write        # sin guardar modelo ni predicciones
    python -m src.recommender.pipeline --quick           # muestra pequena, para probar

El flujo es el de `CHALLENGE.md`, Tarea 3a:

1. **Split temporal por cesta** en tres ventanas (`config.py`).
2. **Ajuste de las fuentes de candidatos** sobre el historial de cada ventana. Se hace
   **dos veces**: una con el historial hasta `fit_end` (para las queries con las que se
   entrena el ranker) y otra con el historial hasta `test_start` (para las de test). Es lo
   que impide que el ranker aprenda con features que ya contienen la respuesta.
3. **Generacion de candidatos** y union del pool.
4. **Features** del par `(query, candidato)`.
5. **LambdaRank** y top-5.
6. **NDCG@5 / Recall@5**, en total y por perfil, mas una ablacion sin senal de sesion y
   un baseline de popularidad.

Cualquier cifra que aparezca en el README sale de aqui (`CLAUDE.md`, "Splits y evaluacion").
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import tracking
from src.etl.repurchase import repurchase_features
from src.etl.schemas import read_processed
from src.etl.session import get_spark
from src.recommender import candidates as cand
from src.recommender import evaluate as ev
from src.recommender import features as feat
from src.recommender import ranker as rk
from src.recommender import splits
from src.recommender.config import REQUIRED_TABLES, RecommenderConfig


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
    customer_products: DataFrame
    customer_stats: DataFrame
    repurchase: DataFrame
    known_customers: DataFrame
    als_model: object
    als_customer_index: DataFrame
    als_product_index: DataFrame


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
    customer_stats = feat.customer_profile(history_baskets, history_items).cache()
    repurchase = repurchase_features(
        history_items,
        history_baskets,
        tables["products"],
        tables["customers"],
        reference_date=feat.default_reference_date(window_start),
    ).cache()
    known = splits.known_customers(history_baskets).cache()

    # Materializar aqui evita que cada fuente se recalcule en cada accion posterior.
    for df in (
        popularity,
        affinity_product,
        affinity_category,
        category_leaders,
        customer_products,
        customer_stats,
        repurchase,
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
        customer_stats=customer_stats,
        repurchase=repurchase,
        known_customers=known,
        als_model=als_model,
        als_customer_index=als_customers,
        als_product_index=als_products,
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
        `(queries, feature_matrix, context)`, donde `context` es una fila por producto de
        cada cesta con `role` = `prefix` (lo que el cliente ya llevaba) o `target` (lo que
        habia que adivinar). Es lo que hace legible la demo de los cuatro perfiles.
    """
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
            queries,
            bundle.customer_products,
            bundle.repurchase,
            tables["products"],
            cfg=cfg.candidates,
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
        customer_products=bundle.customer_products,
        customer_stats=bundle.customer_stats,
        repurchase=bundle.repurchase,
        customers=tables["customers"],
        promotions=tables["promotions"],
        session_product=session_product,
        session_query=session_query,
        target=target,
    )
    context = prefix.select("basket_id", "product_id", F.lit("prefix").alias("role")).unionByName(
        target.select("basket_id", "product_id", F.lit("target").alias("role"))
    )
    return queries, matrix, context


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
    rank_queries, rank_matrix, _ = build_window(
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

    # --- Ventana de test: fuentes rehechas con todo lo anterior a `test_start` ---
    test_bundle = fit_sources(tables, cfg.test_start, cfg)
    timer.step(f"Fuentes de candidatos ajustadas hasta {cfg.test_start}")

    test_queries, test_matrix, test_context = build_window(
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
    test_q = _queries_to_pandas(test_queries)
    timer.step(f"Matriz de test: {len(test_pdf):,} filas / {len(test_q):,} queries")

    # A partir de aqui solo se necesitan los identificadores, la etiqueta, los scores y las
    # banderas que explican de donde salio cada candidato. Arrastrar las 54 features en
    # cada `sort_values` de la evaluacion multiplicaria la memoria sin aportar nada.
    explain = [f"src_{s}" for s in cand.SOURCE_NAMES] + ["n_sources", "sess_viewed", "is_on_promo"]
    scored = test_pdf[["basket_id", "product_id", "profile", "label", *explain]].copy()
    scored["score"] = rk.score(booster, test_pdf)
    scored["score_no_session"] = rk.score(booster_ns, test_pdf, feature_columns=no_session)
    scored["score_popularity"] = ev.popularity_baseline(test_pdf)
    del test_pdf

    summary, per_query = ev.evaluate(scored, test_q, k=cfg.top_k)
    summary_ns, _ = ev.evaluate(scored, test_q, k=cfg.top_k, score_col="score_no_session")
    summary_pop, _ = ev.evaluate(scored, test_q, k=cfg.top_k, score_col="score_popularity")
    pool = ev.candidate_recall(scored, test_q)
    importance = rk.feature_importance(booster)

    # Se guardan tambien las banderas de fuente: sin ellas no se puede explicar *por que*
    # se recomendo cada producto, que es justo lo que ensena la demo de los perfiles.
    top_k = ev.top_k_predictions(scored, k=cfg.top_k)

    context_pdf = test_context.toPandas()
    by_category = ev.category_metrics(
        top_k,
        context_pdf.loc[context_pdf["role"] == "target", ["basket_id", "product_id"]],
        tables["products"].select("product_id", "category").toPandas(),
        test_q,
        k=cfg.top_k,
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
        "feature_importance": importance,
        "per_query": per_query,
        "recommendations": recommendations,
        "valid_ndcg": float(evals["valid"][f"ndcg@{cfg.top_k}"][booster.best_iteration - 1]),
        "n_test_queries": int(len(test_q)),
        "n_train_queries": int(train_pdf["basket_id"].nunique()),
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

| Sistema | NDCG@{k} | Recall@{k} | hit_rate@{k} |
| --- | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (sin aprendizaje) | {result['summary_popularity'].iloc[0][f'ndcg@{k}']:.4f} | {result['summary_popularity'].iloc[0][f'recall@{k}']:.4f} | {result['summary_popularity'].iloc[0][f'hit_rate@{k}']:.4f} |
| LambdaRank sin senal de sesion | {result['summary_no_session'].iloc[0][f'ndcg@{k}']:.4f} | {result['summary_no_session'].iloc[0][f'recall@{k}']:.4f} | {result['summary_no_session'].iloc[0][f'hit_rate@{k}']:.4f} |
| **LambdaRank completo** | **{total[f'ndcg@{k}']:.4f}** | **{total[f'recall@{k}']:.4f}** | **{total[f'hit_rate@{k}']:.4f}** |

### Por perfil, sin senal de sesion

{_table(result['summary_no_session'])}

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

{_table(result['by_category'])}

La distancia entre las dos columnas es la respuesta a por que el NDCG@5 de SKU es bajo:
el sistema **si sabe que categoria toca**, y falla al elegir cual de las ~24 referencias de
esa categoria. En este dataset ese segundo paso esta cerca del azar por construccion --
un cliente con tres o mas compras en una categoria compra 0,86 referencias distintas por
compra, es decir casi nunca repite SKU--, asi que el techo de la metrica de SKU lo pone el
generador, no el modelo. Ver la nota de la Fase 3 en `ROADMAP.md`.

## Techo de la primera etapa

Que parte del target llego siquiera al pool de candidatos. Lo que no esta aqui, el ranker
no lo puede recuperar.

{_table(result['candidate_recall'])}

## Que features usa el ranker

Importancia por ganancia, las {len(result['feature_importance'])} primeras.

{_table(result['feature_importance'])}
"""
    (reports / "metrics.md").write_text(text, encoding="utf-8")

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
    spark = get_spark("grocery-retail-recommender", driver_memory="6g")
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

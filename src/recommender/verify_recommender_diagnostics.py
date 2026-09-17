"""Verificacion del diagnostico de la Fase 7: baselines, oraculo y techo teorico (A3, A6, M8).

    python -m data_generation.export_oracle                           # una vez, ~3 min
    python -m src.recommender.verify_recommender_diagnostics
    python -m src.recommender.verify_recommender_diagnostics --freeze  # congelar snapshot
    python -m src.recommender.verify_recommender_diagnostics --freeze --snapshot baseline_pre_a1.json

Convierte en codigo las cifras de `docs/diagnostico-fase7.md` que salian de un script ad
hoc. Mide, **sobre las mismas queries de test** que el LambdaRank
(`predictions/recommender_test_queries.parquet`):

1. el LambdaRank tal y como quedo en disco (`predictions/recommendations_test.parquet`),
   junto a sus cifras congeladas en `reports/recommender/baseline_pre_diagnostico.json`;
2. la bateria de baselines independientes del pool (`evaluate.run_baselines`), con el
   mismo historial que el sistema: popularidad y reglas congeladas en `test_start`, e
   historial personal as-of el dia de cada cesta (punto A1);
3. el oraculo bayesiano (`oracle.py`): el top-5 de categorias por probabilidad real,
   con su valor realizado y su valor esperado por Monte Carlo;
4. los huecos del top-5 que caen en categorias que el cliente acababa de comprar (punto
   A1), para el LambdaRank actual y, si estan en `--reference` (por defecto
   `predictions/pre_a1/`), para las predicciones congeladas antes del cambio;
5. el objetivo del ranker (punto A4): la ablacion con relevancia binaria de SKU que deja
   el pipeline (`predictions/recommendations_test_sku.parquet`) y, si estan en
   `--reference-a4` (por defecto `predictions/pre_a4/`), las predicciones del modelo
   congelado en `baseline_pre_a4.json`. Todos los sistemas llevan la NDCG@5 graduada, la
   metrica principal;
6. el LambdaRank frente a cada baseline con **bootstrap pareado por cesta** (punto M4):
   diferencia, intervalo de confianza al 95 % y p-valor (`evaluate.paired_bootstrap`).

Escribe `reports/recommender/diagnostics.json` y la seccion de diagnostico de
`reports/recommender/metrics.md`, y termina con codigo de salida 1 si falla alguna
comprobacion de coherencia. No entrena ni modifica ningun modelo.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.etl.schemas import read_processed
from src.etl.session import get_spark
from src.recommender import candidates as cand
from src.recommender import evaluate as ev
from src.recommender import history as hs
from src.recommender import oracle as orc
from src.recommender import pipeline as pl
from src.recommender import splits
from src.recommender.config import SEED, BootstrapConfig, RecommenderConfig
from src.recommender.schema import PROFILE_LABELS

SNAPSHOT_FILENAME = "baseline_pre_diagnostico.json"
DIAGNOSTICS_FILENAME = "diagnostics.json"

# Referencia del "antes" del punto A1: las predicciones del modelo con historial
# congelado, copiadas a `predictions/pre_a1/`, y sus cifras en `baseline_pre_a1.json`
# (cuyo sha256 de predicciones permite comprobar que la copia es la buena).
PRE_A1_SNAPSHOT = "baseline_pre_a1.json"
DEFAULT_REFERENCE_DIR = Path("predictions/pre_a1")
REFERENCE = "lambdarank_pre_a1"

# Lo mismo para el punto A4: el modelo con relevancia binaria que habia en disco antes del
# cambio, y la ablacion con relevancia binaria que reentrena el pipeline actual.
PRE_A4_SNAPSHOT = pl.PRE_A4_FILENAME
DEFAULT_REFERENCE_A4_DIR = Path("predictions/pre_a4")
REFERENCE_A4 = "lambdarank_pre_a4"
LAMBDARANK_SKU = "lambdarank_sku"

# Artefactos de los que salen las cifras del LambdaRank. El snapshot guarda su sha256
# para saber, mas adelante, si una cifra se calculo sobre estas predicciones o sobre otras.
PREDICTION_FILES = (
    "recommendations_test.parquet",
    "recommender_per_query.parquet",
    "recommender_test_queries.parquet",
    "recommender_test_context.parquet",
)

# Confianza minima de una regla categoria -> categoria. Con lift > 1 y este umbral quedan
# las reglas que ademas de ir por encima del azar se cumplen en al menos 1 de cada 10
# cestas con el antecedente; por debajo, la regla casi nunca acierta y solo desplaza a la
# popularidad. Es un parametro del baseline, no algo ajustado contra el test.
DEFAULT_MIN_CONFIDENCE = 0.10

# Muestras de Monte Carlo por query. Con 1.000 el error del techo agregado es de
# decimas de punto; el tamano efectivo por query se reporta.
DEFAULT_MC_SAMPLES = 1_000

# Cuantos errores estandar se toleran entre el techo realizado y el esperado. Son 20
# contrastes (4 metricas x total y 4 perfiles): con 3,5 la probabilidad de una falsa
# alarma ronda el 1 %.
MAX_Z_ORACLE = 3.5

LAMBDARANK = "lambdarank"

# Metricas del contraste pareado entre sistemas (columnas de `evaluate.system_per_query`).
COMPARED_METRICS = ("ndcg_graded", "cat_hit", "sku_hit")
ORACLE_CAT = "oracle_category"
ORACLE_SKU = "oracle_sku"
SYSTEM_LABELS: dict[str, str] = {
    ORACLE_CAT: "Oraculo de categoria (techo)",
    ORACLE_SKU: "Oraculo de SKU (techo)",
    LAMBDARANK: "LambdaRank servido: relevancia graduada (predicciones en disco)",
    LAMBDARANK_SKU: "LambdaRank con relevancia binaria de SKU (ablacion A4)",
    REFERENCE_A4: "LambdaRank antes de A4 (relevancia binaria, congelado)",
    REFERENCE: "LambdaRank antes de A1 (historial congelado)",
    **ev.BASELINE_LABELS,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------------------
# 1. Snapshot congelado
# --------------------------------------------------------------------------------------
def freeze_snapshot(
    cfg: RecommenderConfig, *, name: str = SNAPSHOT_FILENAME, force: bool = False
) -> Path:
    """Congela `metrics.json` (y `diagnostics.json`, si existe) con los hashes de las
    predicciones de las que salen.

    `baseline_pre_diagnostico.json` es la referencia "antes" de las Sesiones 2 en
    adelante del plan de mejora, como `baseline_fase3.json` lo fue para la Fase 7;
    `baseline_pre_a1.json`, la del punto A1.
    """
    reports = Path(cfg.reports_dir)
    path = reports / name
    if path.exists() and not force:
        raise FileExistsError(f"{path} ya existe; usar --force para sobrescribirlo")
    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    diagnostics_path = reports / DIAGNOSTICS_FILENAME
    payload = {
        "descripcion": (
            f"Metricas del LambdaRank congeladas como {name} antes de aplicar el siguiente "
            "punto de docs/diagnostico-fase7.md. No regenerar: es la referencia del antes."
        ),
        "congelado": dt.date.today().isoformat(),
        "commit": _git_head(),
        "fuente": (reports / "metrics.json").as_posix(),
        "predicciones_sha256": {
            file: _sha256(Path(cfg.predictions_dir) / file) for file in PREDICTION_FILES
        },
        "modelo_sha256": _sha256(Path(cfg.models_dir) / "recommender_ranker_lgbm.txt"),
        # Con que dataset se midio: los hashes de las 7 tablas de `data/raw`. Desde la
        # Fase 8 el dato cambia, y un "antes/despues" solo tiene sentido con el mismo.
        "dataset_sha256": pl.dataset_fingerprint(),
        "metrics": metrics,
    }
    if name != SNAPSHOT_FILENAME and diagnostics_path.is_file():
        payload["diagnostics"] = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# 2. Entradas: queries de test e historial
# --------------------------------------------------------------------------------------
@dataclass
class TestSet:
    """Las queries de test tal y como las evaluo el pipeline."""

    queries: pd.DataFrame
    prefix: pd.DataFrame
    target: pd.DataFrame
    context: pd.DataFrame
    lambdarank: pd.DataFrame


def load_test_set(predictions_dir: Path) -> TestSet:
    queries = pd.read_parquet(predictions_dir / "recommender_test_queries.parquet")
    context = pd.read_parquet(predictions_dir / "recommender_test_context.parquet")
    recs = pd.read_parquet(predictions_dir / "recommendations_test.parquet")
    return TestSet(
        queries=queries,
        prefix=context.loc[context["role"] == "prefix", ["basket_id", "product_id"]],
        target=context.loc[context["role"] == "target", ["basket_id", "product_id"]],
        context=context,
        lambdarank=recs[["basket_id", "product_id", "rank", "label"]],
    )


def load_reference(
    reference_dir: Path, filename: str = "recommendations_test.parquet"
) -> pd.DataFrame | None:
    """Top-5 de referencia, si existe. `measure` comprueba que es el bueno."""
    path = reference_dir / filename
    if not path.is_file():
        return None
    return pd.read_parquet(path)[["basket_id", "product_id", "rank", "label"]]


def fit_history(
    spark: SparkSession, cfg: RecommenderConfig, queries: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Historial de los baselines, con las mismas funciones que el pipeline.

    Popularidad y reglas, con lo anterior a `test_start` (como `pipeline.fit_sources`);
    historial personal, as-of el dia de cada query (como `pipeline.build_window`).
    """
    tables = read_processed(
        spark, ("baskets", "basket_items", "products", "customers"), cfg.processed_dir
    )
    products = tables["products"]
    hist_b = splits.baskets_before(tables["baskets"], cfg.test_start).cache()
    hist_i = splits.restrict_items(tables["basket_items"], hist_b).cache()

    popularity = cand.fit_popularity(
        hist_b, hist_i, window_start=cfg.test_start, cfg=cfg.candidates
    )
    product_pop = popularity.groupBy("product_id").agg(
        F.max("prod_pop_recent").alias("n_baskets")
    )

    recent_from = F.lit(str(cfg.test_start)).cast("date") - F.expr(
        f"INTERVAL {cfg.candidates.recent_days} DAYS"
    )
    category_pop = (
        hist_i.select("basket_id", "product_id")
        .join(hist_b.filter(F.col("basket_day") >= recent_from).select("basket_id"), "basket_id")
        .join(F.broadcast(products.select("product_id", "category")), "product_id")
        .groupBy("category")
        .agg(F.countDistinct("basket_id").cast("double").alias("n_baskets"))
    )

    spark_queries = spark.createDataFrame(
        queries[["basket_id", "customer_id"]].assign(
            basket_day=pd.to_datetime(queries["basket_day"]).dt.date
        ),
        schema="basket_id string, customer_id string, basket_day date",
    )
    asof = hs.asof_history(
        spark_queries, tables["baskets"], tables["basket_items"], products, tables["customers"]
    )
    customer_products = asof.products.select(
        "basket_id", "product_id", F.col("hist_n_baskets").alias("n_baskets")
    )
    customer_categories = asof.categories.select(
        "basket_id",
        "category",
        F.col("cat_n_purchase_days").alias("n_purchase_days"),
        # Como texto: una fecha viaja bien a pandas, pero asi no hay dudas de zona horaria.
        F.date_format("cat_last_day", "yyyy-MM-dd").alias("last_purchase_date"),
        F.col("cat_expected_days").alias("expected_repurchase_days"),
    )
    rules = cand.fit_affinity_category(hist_i, products).select(
        "antecedent", "consequent", "confidence", "lift"
    )

    return {
        "products": products.select("product_id", "category").toPandas(),
        "product_popularity": product_pop.toPandas(),
        "category_popularity": category_pop.toPandas(),
        "customer_products": customer_products.toPandas(),
        "customer_categories": customer_categories.toPandas(),
        "category_rules": rules.toPandas(),
    }


# --------------------------------------------------------------------------------------
# 3. Medicion
# --------------------------------------------------------------------------------------
@dataclass
class Diagnostics:
    """Todo lo medido, listo para escribir."""

    k: int
    summaries: dict[str, pd.DataFrame]
    oracle_expected: dict[str, float]
    oracle_expected_by_profile: pd.DataFrame
    ess: dict[str, float]
    rules: dict[str, float]
    snapshot: dict | None
    recency: dict[str, pd.DataFrame] = field(default_factory=dict)
    recency_comparison: pd.DataFrame | None = None
    pre_a1: dict | None = None
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    comparisons: pd.DataFrame | None = None
    bootstrap: BootstrapConfig | None = None


def realised_oracle_metrics(
    inputs: orc.OracleInputs, mc: orc.MonteCarloResult, *, k: int
) -> dict[str, np.ndarray]:
    """Lo que las listas del oraculo aciertan de verdad, query a query.

    `cat_*` sale de la lista de categoria y `sku_*` de la de SKU. El acierto de SKU se
    cuenta en probabilidad (la del mejor SKU en cada categoria acertada), no con el SKU
    realmente comprado: asi la comparacion con el valor esperado solo lleva el ruido del
    sorteo de categorias, que es el que se quiere contrastar.
    """

    def hits(lists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        safe = np.maximum(lists, 0)
        hit = np.take_along_axis(inputs.target_mask, safe, axis=1) & (lists >= 0)
        return hit, np.take_along_axis(inputs.best_prob, safe, axis=1) * (lists >= 0)

    cat_hits, _ = hits(mc.cat_lists)
    sku_hits, p = hits(mc.sku_lists)
    return {
        "cat_hit": cat_hits.any(axis=1).astype(float),
        "cat_precision": cat_hits.sum(axis=1) / k,
        "sku_hit": 1.0 - np.prod(np.where(sku_hits, 1.0 - p, 1.0), axis=1),
        "sku_precision": (sku_hits * p).sum(axis=1) / k,
    }


def _profile_table(per_query: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Media de columnas por query, en total y por perfil, con los nombres de `summarise`."""
    rows = [{"grupo": "total", **per_query[columns].mean().to_dict()}]
    for profile, sub in per_query.groupby("profile"):
        rows.append({"grupo": PROFILE_LABELS[int(profile)], **sub[columns].mean().to_dict()})
    return pd.DataFrame(rows)


def measure(
    test: TestSet,
    history: dict[str, pd.DataFrame],
    *,
    oracle_dir: Path,
    k: int,
    seed: int,
    min_confidence: float,
    n_samples: int,
    snapshot: dict | None,
    current_metrics: dict | None,
    reference: pd.DataFrame | None = None,
    pre_a1: dict | None = None,
    reference_sha: str | None = None,
    window_start: dt.date | None = None,
    extra_systems: dict[str, pd.DataFrame] | None = None,
    extra_checks: list[tuple[str, bool, str]] | None = None,
    bootstrap: BootstrapConfig | None = None,
) -> Diagnostics:
    """Mide todos los sistemas sobre las mismas queries.

    `extra_systems` son top-k adicionales (por nombre de `SYSTEM_LABELS`) que solo se
    resumen, sin analisis de recencia: la ablacion y el modelo congelado del punto A4.
    Con `bootstrap`, el LambdaRank se contrasta con cada baseline y con esos sistemas
    adicionales (bootstrap pareado por cesta).
    """
    products = history["products"]
    queries = test.queries
    per_query: dict[str, pd.DataFrame] = {}

    def summary(top_k: pd.DataFrame, name: str | None = None) -> pd.DataFrame:
        if name is not None and bootstrap is not None:
            per_query[name] = ev.system_per_query(top_k, queries, test.target, products, k=k)
        return ev.system_summary(top_k, queries, test.target, products, k=k, prefix=test.prefix)

    summaries: dict[str, pd.DataFrame] = {}
    checks: list[tuple[str, bool, str]] = list(extra_checks or [])

    # --- Oraculo ---
    inputs = orc.load_inputs(oracle_dir, queries, test.context, products)
    # Desde la Fase 8 una categoria de exploracion puede llevar dos lineas, y el corte
    # puede dejar una a cada lado: esa categoria ya esta en el carrito y no cuenta como
    # resto. Lo que no puede pasar es que haya mas categorias que lineas.
    lines = queries["n_target"].to_numpy()
    fits = bool((inputs.n_target <= lines).all())
    split = float((inputs.target_mask & inputs.prefix_mask).any(axis=1).mean())
    repeated = float((inputs.target_mask.sum(axis=1) < lines).mean())
    checks.append(
        (
            "Target en categorias <= target en lineas (segunda referencia de la Fase 8)",
            fits,
            f"{repeated:.1%} de queries con dos lineas de una categoria en el target; "
            f"{split:.1%} con una categoria a los dos lados del corte",
        )
    )
    t0 = time.perf_counter()
    mc = orc.monte_carlo(inputs, k=k, n_samples=n_samples, seed=seed)
    print(
        f"  Monte Carlo: {len(queries):,} queries x {n_samples} muestras "
        f"({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )
    summaries[ORACLE_CAT] = summary(orc.lists_to_top_k(inputs, mc.cat_lists, test.target))
    summaries[ORACLE_SKU] = summary(orc.lists_to_top_k(inputs, mc.sku_lists, test.target))

    oracle_per_query = pd.DataFrame(
        {
            "profile": queries["profile"].to_numpy(),
            f"cat_hit_rate@{k}": mc.cat_hit,
            f"cat_precision@{k}": mc.cat_precision,
            f"sku_hit_rate@{k}": mc.sku_hit,
            f"sku_precision@{k}": mc.sku_precision,
        }
    )
    expected_by_profile = _profile_table(
        oracle_per_query, [c for c in oracle_per_query.columns if "@" in c]
    )
    expected = expected_by_profile.iloc[0].drop("grupo").astype(float).to_dict()

    # --- LambdaRank ---
    summaries[LAMBDARANK] = summary(test.lambdarank, LAMBDARANK)
    if current_metrics is not None:
        on_disk = current_metrics["by_category"][0][f"cat_hit_rate@{k}"]
        now = float(summaries[LAMBDARANK].iloc[0][f"cat_hit_rate@{k}"])
        ok = abs(on_disk - now) < 1e-12
        checks.append(
            (
                "LambdaRank recalculado desde predictions/ = reports/recommender/metrics.json",
                ok,
                f"{now:.4f} frente a {on_disk:.4f}",
            )
        )

    # --- Baselines ---
    data = ev.BaselineInputs(
        queries=queries[["basket_id", "customer_id", "basket_day"]],
        prefix=test.prefix,
        target=test.target,
        products=products,
        product_popularity=history["product_popularity"],
        category_popularity=history["category_popularity"],
        customer_products=history["customer_products"],
        customer_categories=history["customer_categories"],
        category_rules=history["category_rules"],
    )
    baseline_lists = ev.run_baselines(data, k=k, seed=seed, min_confidence=min_confidence)
    for name, top_k in baseline_lists.items():
        summaries[name] = summary(top_k, name)
    for name, top_k in (extra_systems or {}).items():
        summaries[name] = summary(top_k, name)

    # --- Comprobaciones del techo ---
    # Si el oraculo reproduce el generador, lo realizado en cada query es una muestra de
    # lo esperado: la diferencia media tiene que ser ruido. Se contrasta con la desviacion
    # empirica de las diferencias por query (pareado), en total y por perfil.
    realised = realised_oracle_metrics(inputs, mc, k=k)
    z_scores: dict[str, float] = {}
    for metric in ("cat_hit", "cat_precision", "sku_hit", "sku_precision"):
        diff = realised[metric] - getattr(mc, metric)
        z_scores[metric] = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
        for profile in sorted(queries["profile"].unique()):
            sub = diff[queries["profile"].to_numpy() == profile]
            z_scores[f"{metric}_p{profile}"] = float(
                sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub)))
            )
    worst = max(z_scores, key=lambda key: abs(z_scores[key]))
    checks.append(
        (
            f"Techo realizado compatible con el esperado en las 4 metricas, total y por "
            f"perfil (max |z| <= {MAX_Z_ORACLE})",
            abs(z_scores[worst]) <= MAX_Z_ORACLE,
            f"peor: {worst}, z = {z_scores[worst]:+.2f}",
        )
    )
    realised_hit = float(summaries[ORACLE_CAT].iloc[0][f"cat_hit_rate@{k}"])
    se = float(np.sqrt((mc.cat_hit * (1 - mc.cat_hit)).sum()) / len(mc.cat_hit))
    best_other = max(
        (float(s.iloc[0][f"cat_hit_rate@{k}"]), n)
        for n, s in summaries.items()
        if n not in (ORACLE_CAT, ORACLE_SKU, REFERENCE)
    )
    checks.append(
        (
            "Ningun sistema supera al oraculo de categoria en cat_hit_rate",
            best_other[0] <= realised_hit,
            f"mejor sistema {best_other[1]} = {best_other[0]:.4f}",
        )
    )

    # --- Huecos en categorias recien compradas (punto A1) ---
    # La ultima compra real de cada categoria antes del dia de la cesta: es lo que el
    # modelo con historial congelado no veia.
    last_day = history["customer_categories"].rename(columns={"last_purchase_date": "last_day"})

    def slots(top_k: pd.DataFrame) -> pd.DataFrame:
        return ev.recency_slots(top_k, queries, test.target, last_day, products)

    recency_slots = {LAMBDARANK: slots(test.lambdarank)}
    if reference is not None:
        same_queries = set(reference["basket_id"]) == set(test.lambdarank["basket_id"])
        detail = ""
        if pre_a1 is not None:
            frozen = pre_a1["predicciones_sha256"]["recommendations_test.parquet"]
            same_file = frozen == reference_sha
            detail = "sha256 = el de " + PRE_A1_SNAPSHOT if same_file else "sha256 distinto"
            same_queries = same_queries and same_file
        checks.append(
            (
                "Las predicciones de referencia (antes de A1) son las congeladas y cubren "
                "las mismas queries",
                same_queries,
                detail,
            )
        )
        recency_slots[REFERENCE] = slots(reference)
        summaries[REFERENCE] = summary(reference, REFERENCE if same_queries else None)
    recency_slots["personal_due"] = slots(baseline_lists["personal_due"])
    start = window_start or pd.to_datetime(queries["basket_day"]).min()
    recency = {
        name: ev.recency_slot_metrics(frame, window_start=start)
        for name, frame in recency_slots.items()
    }
    comparison = (
        ev.recency_query_comparison(recency_slots, REFERENCE, k=k)
        if REFERENCE in recency_slots
        else None
    )

    # --- Incertidumbre (punto M4): LambdaRank frente a cada sistema, mismas cestas ---
    comparisons = None
    if bootstrap is not None:
        names = ev.system_metrics(k)
        keys = set(queries["basket_id"])
        comparable = {
            name: table
            for name, table in per_query.items()
            if len(table) == len(keys) and set(table["basket_id"]) == keys
        }
        comparisons = ev.paired_bootstrap(
            comparable,
            LAMBDARANK,
            {c: names[c] for c in COMPARED_METRICS},
            bootstrap,
        )

    rules = history["category_rules"]
    kept = rules.loc[(rules["confidence"] >= min_confidence) & (rules["lift"] > 1.0)]
    return Diagnostics(
        k=k,
        summaries=summaries,
        oracle_expected=expected,
        oracle_expected_by_profile=expected_by_profile,
        ess={
            "median": float(np.median(mc.ess)),
            "p05": float(np.percentile(mc.ess, 5)),
            "min": float(mc.ess.min()),
            "n_samples": float(n_samples),
            "n_fallback": float(mc.n_fallback),
            "cat_hit_se": se,
            "z": z_scores,
        },
        rules={
            "min_confidence": min_confidence,
            "n_rules_total": float(len(rules)),
            "n_rules_used": float(len(kept)),
        },
        snapshot=snapshot,
        recency=recency,
        recency_comparison=comparison,
        pre_a1=pre_a1,
        checks=checks,
        comparisons=comparisons,
        bootstrap=bootstrap,
    )


# --------------------------------------------------------------------------------------
# 4. Informe
# --------------------------------------------------------------------------------------
def _pct(value: float) -> str:
    return f"{value:.1%}"


def _num(value: float) -> str:
    return f"{value:.4f}"


def _pp(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def headline(diag: Diagnostics) -> dict[str, object]:
    """Las dos respuestas que pide la sesion: distancia al mejor baseline y % de techo."""
    k = diag.k
    col = f"cat_hit_rate@{k}"
    total = {name: s.iloc[0] for name, s in diag.summaries.items()}
    baselines = {n: float(total[n][col]) for n in ev.BASELINE_LABELS}
    best = max(baselines, key=baselines.get)
    lr = float(total[LAMBDARANK][col])
    ceiling = float(total[ORACLE_CAT][col])
    sku_col = f"sku_hit_rate@{k}"
    sku_ceiling = float(total[ORACLE_SKU][sku_col])
    optional = (LAMBDARANK_SKU, REFERENCE_A4, REFERENCE)
    systems = [LAMBDARANK, *ev.BASELINE_LABELS, *[n for n in optional if n in total]]
    return {
        "best_baseline": best,
        "best_baseline_cat_hit_rate": baselines[best],
        "lambdarank_cat_hit_rate": lr,
        "gap_best_baseline_minus_lambdarank": baselines[best] - lr,
        "ceiling_cat_hit_rate": ceiling,
        "ceiling_sku_hit_rate": sku_ceiling,
        "pct_of_ceiling": {name: float(total[name][col]) / ceiling for name in systems},
        "pct_of_sku_ceiling": {
            name: float(total[name][sku_col]) / sku_ceiling for name in systems
        },
    }


def _system_order(diag: Diagnostics) -> list[str]:
    k = diag.k
    rest = sorted(
        ev.BASELINE_LABELS,
        key=lambda n: -float(diag.summaries[n].iloc[0][f"cat_hit_rate@{k}"]),
    )
    before = [n for n in (LAMBDARANK_SKU, REFERENCE_A4, REFERENCE) if n in diag.summaries]
    return [ORACLE_CAT, ORACLE_SKU, LAMBDARANK, *before, *rest]


def _by_profile_table(diag: Diagnostics, column: str, ceiling: str) -> list[str]:
    """Filas `sistema | total | p1..p4`, con el % del techo del mismo grupo entre parentesis."""
    groups = list(diag.summaries[ceiling]["grupo"])
    ceil = diag.summaries[ceiling].set_index("grupo")[column]
    head = "| Sistema | " + " | ".join(
        "Total" if g == "total" else f"Perfil {g}" for g in groups
    ) + " |"
    lines = [head, "| --- |" + " ---: |" * len(groups)]
    for name in _system_order(diag):
        values = diag.summaries[name].set_index("grupo")[column]
        cells = []
        for g in groups:
            cell = _num(values[g])
            if name not in (ORACLE_CAT, ORACLE_SKU):
                cell += f" ({_pct(values[g] / ceil[g])})"
            cells.append(cell)
        label = SYSTEM_LABELS[name]
        if name in (ORACLE_CAT, ORACLE_SKU, LAMBDARANK):
            label = f"**{label}**"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _recency_markdown(diag: Diagnostics) -> list[str]:
    """Seccion del punto A1: huecos en categorias que el cliente acababa de comprar."""
    k = diag.k
    names = [n for n in (REFERENCE, LAMBDARANK, "personal_due") if n in diag.recency]
    groups = list(diag.recency[LAMBDARANK]["grupo"])
    head = "| Huecos en categorias compradas... | " + " | ".join(
        f"{SYSTEM_LABELS[n]}: % huecos | SKU prec. | cat. prec." for n in names
    ) + " |"
    lines = [
        "### Huecos en categorias recien compradas (punto A1)",
        "",
        "Cada hueco del top-5 se clasifica por los dias desde que el cliente compro por "
        "ultima vez su categoria, con **todo** su historial anterior al dia de la cesta "
        "(lo viera o no el modelo). El generador castiga con fuerza reponer justo despues "
        "de comprar, asi que los huecos de los primeros dias casi nunca aciertan. "
        "\"En la ventana\" es desde el inicio del test. Precision = parte de esos huecos "
        "que acierta el SKU o la categoria.",
        "",
        head,
        "| --- |" + " ---: | ---: | ---: |" * len(names),
    ]
    tables = {n: diag.recency[n].set_index("grupo") for n in names}
    for g in groups:
        cells = []
        for n in names:
            row = tables[n].loc[g]
            cells += [
                _pct(row["proporcion_huecos"]),
                _pct(row["sku_precision"]),
                _pct(row["cat_precision"]),
            ]
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines.append("")

    comp = diag.recency_comparison
    if comp is not None:
        lines += [
            f"**Queries donde el modelo de antes gastaba huecos en categorias recien "
            f"compradas.** Para cada umbral, las queries cuya lista antes de A1 tenia al "
            f"menos un hueco en una categoria comprada hace <= d dias, y como les va a cada "
            f"sistema (lista entera de {k}).",
            "",
            f"| Umbral | Sistema | Queries | Huecos recientes | sku_precision@{k} "
            f"| cat_precision@{k} | sku_hit_rate@{k} | cat_hit_rate@{k} |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in comp.to_dict(orient="records"):
            lines.append(
                f"| <= {row['umbral_dias']} dias | {SYSTEM_LABELS[row['sistema']]} "
                f"| {row['n_queries']:,} | {_pct(row[f'huecos_recientes@{k}'])} "
                f"| {_num(row[f'sku_precision@{k}'])} | {_num(row[f'cat_precision@{k}'])} "
                f"| {_num(row[f'sku_hit_rate@{k}'])} | {_num(row[f'cat_hit_rate@{k}'])} |"
            )
        lines.append("")
    if diag.pre_a1 is not None:
        lines += [
            f"Las predicciones de antes son las congeladas en `{PRE_A1_SNAPSHOT}` "
            f"({diag.pre_a1['congelado']}, commit `{(diag.pre_a1['commit'] or '?')[:7]}`).",
            "",
        ]
    return lines


def _comparison_row(diag: Diagnostics, system: str, metric: str, group: str = "total") -> dict | None:
    if diag.comparisons is None:
        return None
    table = diag.comparisons
    match = table.loc[
        (table["sistema"] == system) & (table["metrica"] == metric) & (table["grupo"] == group)
    ]
    return None if match.empty else match.iloc[0].to_dict()


def _gap_detail(diag: Diagnostics, system: str) -> str:
    """`, IC 95 % [a, b] pp, p = ...` de la diferencia en cat_hit_rate, si se calculo."""
    row = _comparison_row(diag, system, f"cat_hit_rate@{diag.k}")
    if row is None:
        return ""
    return (
        f" (LambdaRank - baseline: {pl._diff_cell(row)}, p {pl._p_cell(row['p_value'])})"
    )


def _comparison_markdown(diag: Diagnostics) -> list[str]:
    """LambdaRank frente a cada sistema, con IC y p-valor (bootstrap pareado)."""
    if diag.comparisons is None or diag.comparisons.empty:
        return []
    order = [n for n in _system_order(diag) if n in set(diag.comparisons["sistema"])]
    table = diag.comparisons.copy()
    table["_order"] = table["sistema"].map({n: i for i, n in enumerate(order)})
    table = table.sort_values("_order", kind="stable").drop(columns="_order")
    lines = pl._comparison_table(
        table.assign(sistema=table["sistema"].map(SYSTEM_LABELS)), "total"
    )
    boot = diag.bootstrap
    return [
        "### LambdaRank frente a cada sistema (bootstrap pareado, punto M4)",
        "",
        "Diferencia = LambdaRank servido - el otro sistema, sobre las mismas queries. Cada "
        f"remuestreo sortea las mismas cestas para los dos ({boot.n_resamples:,} "
        f"remuestreos, intervalo percentil al {boot.confidence:.0%}; "
        "`evaluate.paired_bootstrap`). El p-valor es bilateral y no esta corregido por "
        "comparaciones multiples. Las tasas van en puntos porcentuales. El desglose por "
        "perfil esta en `diagnostics.json` (`comparisons`).",
        "",
        *lines,
        "",
    ]


def _objective_bullets(diag: Diagnostics, h: dict[str, object]) -> list[str]:
    """Lectura del punto A4: que cambia al pasar de relevancia binaria a graduada."""
    if LAMBDARANK_SKU not in diag.summaries:
        return []
    k = diag.k
    now, sku = diag.summaries[LAMBDARANK].iloc[0], diag.summaries[LAMBDARANK_SKU].iloc[0]
    cat, skuc, graded = f"cat_hit_rate@{k}", f"sku_hit_rate@{k}", f"ndcg_graded@{k}"
    best = h["best_baseline_cat_hit_rate"]
    return [
        f"- **Objetivo del ranker (A4):** con relevancia binaria de SKU, cat_hit_rate@{k} "
        f"{_num(sku[cat])} ({_pp(sku[cat] - best)} frente al mejor baseline); con la "
        f"graduada servida, {_num(now[cat])} ({_pp(now[cat] - best)}). sku_hit_rate@{k} "
        f"pasa de {_num(sku[skuc])} a {_num(now[skuc])} y la NDCG@{k} graduada de "
        f"{_num(sku[graded])} a {_num(now[graded])}.",
    ]


def pre_fase8_comparison(diag: Diagnostics, cfg: RecommenderConfig) -> pd.DataFrame | None:
    """Cada sistema frente a si mismo sobre el dataset anterior a la Fase 8.

    Solo tiene sentido cuando el dataset en uso ya no es el congelado: mientras lo sea,
    devuelve None. Las queries no son las mismas (otro dataset), asi que la comparacion es
    de nivel, no pareada.
    """
    path = Path(cfg.reports_dir) / pl.PRE_FASE8_FILENAME
    if not path.is_file():
        return None
    snap = json.loads(path.read_text(encoding="utf-8"))
    if pl.same_dataset(snap) or "diagnostics" not in snap:
        return None
    k = diag.k
    old_systems = snap["diagnostics"]["systems"]
    old_ceiling = old_systems[ORACLE_CAT]["by_profile"][0][f"cat_hit_rate@{k}"]
    old_sku_ceiling = old_systems[ORACLE_SKU]["by_profile"][0][f"sku_hit_rate@{k}"]
    total = {name: s.iloc[0] for name, s in diag.summaries.items()}
    ceiling = float(total[ORACLE_CAT][f"cat_hit_rate@{k}"])
    sku_ceiling = float(total[ORACLE_SKU][f"sku_hit_rate@{k}"])
    rows = []
    for name in _system_order(diag):
        if name not in old_systems:
            continue
        old = old_systems[name]["by_profile"][0]
        now = total[name]
        rows.append(
            {
                "sistema": name,
                "ndcg_graded_antes": old[f"ndcg_graded@{k}"],
                "ndcg_graded_ahora": float(now[f"ndcg_graded@{k}"]),
                "cat_hit_antes": old[f"cat_hit_rate@{k}"],
                "cat_hit_ahora": float(now[f"cat_hit_rate@{k}"]),
                "pct_techo_cat_antes": old[f"cat_hit_rate@{k}"] / old_ceiling,
                "pct_techo_cat_ahora": float(now[f"cat_hit_rate@{k}"]) / ceiling,
                "sku_hit_antes": old[f"sku_hit_rate@{k}"],
                "sku_hit_ahora": float(now[f"sku_hit_rate@{k}"]),
                "pct_techo_sku_antes": old[f"sku_hit_rate@{k}"] / old_sku_ceiling,
                "pct_techo_sku_ahora": float(now[f"sku_hit_rate@{k}"]) / sku_ceiling,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["congelado"] = snap["congelado"]
    frame.attrs["commit"] = snap["commit"]
    return frame


def _pre_fase8_markdown(diag: Diagnostics, cfg: RecommenderConfig) -> list[str]:
    table = pre_fase8_comparison(diag, cfg)
    if table is None or table.empty:
        return []
    k = diag.k
    lines = [
        "### Frente al dataset anterior a la Fase 8",
        "",
        "Los mismos sistemas, medidos con el mismo codigo sobre el dataset de la Fase 7a "
        f"(`{Path(cfg.reports_dir).as_posix()}/{pl.PRE_FASE8_FILENAME}`, "
        f"{table.attrs['congelado']}, commit `{(table.attrs['commit'] or '?')[:7]}`; el "
        "resto de artefactos de ese momento esta en `snapshots/pre-fase-8/`). Son "
        "queries distintas de datasets distintos: la comparacion es de nivel, no pareada. "
        "El techo tambien cambia, asi que la columna que dice cuanto mejora cada sistema "
        "*respecto a lo alcanzable* es el % del techo.",
        "",
        f"| Sistema | NDCG@{k} graduada antes | ahora | cat_hit_rate@{k} antes | ahora "
        f"| % techo antes | ahora | sku_hit_rate@{k} antes | ahora | % techo SKU antes "
        "| ahora |",
        "| --- |" + " ---: |" * 10,
    ]
    for row in table.to_dict(orient="records"):
        name = row["sistema"]
        is_oracle = name in (ORACLE_CAT, ORACLE_SKU)
        label = SYSTEM_LABELS[name]
        if is_oracle or name == LAMBDARANK:
            label = f"**{label}**"
        pct = (lambda v: "—") if is_oracle else _pct
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _num(row["ndcg_graded_antes"]),
                    _num(row["ndcg_graded_ahora"]),
                    _num(row["cat_hit_antes"]),
                    _num(row["cat_hit_ahora"]),
                    pct(row["pct_techo_cat_antes"]),
                    pct(row["pct_techo_cat_ahora"]),
                    _num(row["sku_hit_antes"]),
                    _num(row["sku_hit_ahora"]),
                    pct(row["pct_techo_sku_antes"]),
                    pct(row["pct_techo_sku_ahora"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_markdown(diag: Diagnostics, cfg: RecommenderConfig, predictions_sha: str) -> str:
    k = diag.k
    h = headline(diag)
    col = lambda m: f"{m}@{k}"  # noqa: E731
    total = {name: s.iloc[0] for name, s in diag.summaries.items()}
    cat_ceiling = h["ceiling_cat_hit_rate"]
    sku_ceiling = h["ceiling_sku_hit_rate"]

    main = [
        f"| Sistema | NDCG@{k} graduada | cat_hit_rate@{k} | % techo | cat_precision@{k} "
        f"| sku_hit_rate@{k} | % techo SKU | sku_precision@{k} | NDCG@{k} SKU "
        "| Huecos regalados |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in _system_order(diag):
        row = total[name]
        is_oracle = name in (ORACLE_CAT, ORACLE_SKU)
        label = SYSTEM_LABELS[name]
        if is_oracle or name == LAMBDARANK:
            label = f"**{label}**"
        main.append(
            "| "
            + " | ".join(
                [
                    label,
                    _num(row[col("ndcg_graded")]),
                    _num(row[col("cat_hit_rate")]),
                    "—" if is_oracle else _pct(row[col("cat_hit_rate")] / cat_ceiling),
                    _num(row[col("cat_precision")]),
                    _num(row[col("sku_hit_rate")]),
                    "—" if is_oracle else _pct(row[col("sku_hit_rate")] / sku_ceiling),
                    _num(row[col("sku_precision")]),
                    _num(row[col("ndcg")]),
                    _pct(row[col("huecos_regalados")]),
                ]
            )
            + " |"
        )

    snap_lines: list[str] = []
    if diag.snapshot is not None:
        frozen = diag.snapshot["metrics"]["by_category"][0]
        now = total[LAMBDARANK]
        same = all(
            abs(frozen[col(m)] - now[col(m)]) < 1e-12
            for m in ("cat_hit_rate", "cat_precision", "sku_hit_rate", "sku_precision")
        )
        snap_lines = [
            f"Cifras congeladas del LambdaRank en `{Path(cfg.reports_dir).as_posix()}/"
            f"{SNAPSHOT_FILENAME}` ({diag.snapshot['congelado']}, commit "
            f"`{(diag.snapshot['commit'] or '?')[:7]}`): cat_hit_rate@{k} "
            f"{_num(frozen[col('cat_hit_rate')])}, sku_hit_rate@{k} "
            f"{_num(frozen[col('sku_hit_rate')])}. "
            + (
                "Las predicciones en disco las reproducen exactamente."
                if same
                else "**Las predicciones en disco ya no son las congeladas**: la fila del "
                "LambdaRank de arriba es la version actual."
            ),
            "",
        ]

    exp = diag.oracle_expected
    realised = total[ORACLE_CAT]
    expected_rows = [
        "| Techo | Realizado (estas queries) | Esperado (Monte Carlo) |",
        "| --- | ---: | ---: |",
        f"| cat_hit_rate@{k} (oraculo de categoria) | {_num(realised[col('cat_hit_rate')])} "
        f"| {_num(exp[col('cat_hit_rate')])} ± {_num(1.96 * diag.ess['cat_hit_se'])} |",
        f"| cat_precision@{k} (oraculo de categoria) | {_num(realised[col('cat_precision')])} "
        f"| {_num(exp[col('cat_precision')])} |",
        f"| sku_hit_rate@{k} (oraculo de SKU) | {_num(total[ORACLE_SKU][col('sku_hit_rate')])} "
        f"| {_num(exp[col('sku_hit_rate')])} |",
        f"| sku_precision@{k} (oraculo de SKU) | {_num(total[ORACLE_SKU][col('sku_precision')])} "
        f"| {_num(exp[col('sku_precision')])} |",
    ]

    checks = [
        f"- {'OK' if ok else '**FALLA**'} — {name}" + (f" ({detail})" if detail else "")
        for name, ok, detail in diag.checks
    ]

    best = h["best_baseline"]
    gap = h["gap_best_baseline_minus_lambdarank"]
    pct = h["pct_of_ceiling"]
    lines = [
        pl.DIAGNOSTICS_START,
        "## Diagnostico: baselines independientes del pool y techo teorico",
        "",
        "Generado por `python -m src.recommender.verify_recommender_diagnostics` "
        f"({dt.date.today().isoformat()}), sobre las {int(total[LAMBDARANK]['n_queries']):,} "
        f"queries de test de las predicciones en disco (sha256 de "
        f"`recommendations_test.parquet`: `{predictions_sha[:12]}`). El oraculo necesita "
        "antes `python -m data_generation.export_oracle`. Esta seccion no la reescribe el "
        "pipeline: si se reentrena, hay que volver a lanzar el verificador. Puntos A3, A4, "
        "A6 y M8 de `docs/diagnostico-fase7.md`. La metrica principal es la NDCG@"
        f"{k} graduada (3 por SKU exacto, 1 por categoria; `evaluate.category_metrics`).",
        "",
        "### Lectura rapida",
        "",
        f"- **Mejor baseline en cat_hit_rate@{k}:** {SYSTEM_LABELS[best]}, "
        f"{_num(h['best_baseline_cat_hit_rate'])} frente a {_num(h['lambdarank_cat_hit_rate'])} "
        f"del LambdaRank: {_pp(gap)}"
        + (" a favor del baseline" if gap > 0 else ", el LambdaRank va por delante")
        + _gap_detail(diag, best)
        + ".",
        f"- **Techo teorico de cat_hit_rate@{k}:** {_num(cat_ceiling)}. El LambdaRank "
        f"alcanza el {_pct(pct[LAMBDARANK])} y el mejor baseline el {_pct(pct[best])}.",
        f"- **Techo de sku_hit_rate@{k}:** {_num(sku_ceiling)}. El LambdaRank alcanza el "
        f"{_pct(h['pct_of_sku_ceiling'][LAMBDARANK])}.",
        *_objective_bullets(diag, h),
        "",
        "### Todos los sistemas, total",
        "",
        "Las columnas \"% techo\" dividen por el oraculo correspondiente sobre las mismas "
        "queries. \"Huecos regalados\" es la parte del top-5 en una categoria que ya esta "
        "en el carrito o repetida mas arriba en la lista (`evaluate.wasted_slot_metrics`, "
        "punto A2). Los baselines de categoria no regalan ninguno por construccion; el "
        "LambdaRank los evita con el re-ranking de `RecommenderConfig.rerank`.",
        "",
        *main,
        "",
        *snap_lines,
        *_pre_fase8_markdown(diag, cfg),
        *_comparison_markdown(diag),
        f"### cat_hit_rate@{k} por perfil (entre parentesis, % del techo del perfil)",
        "",
        *_by_profile_table(diag, col("cat_hit_rate"), ORACLE_CAT),
        "",
        f"### sku_hit_rate@{k} por perfil (entre parentesis, % del techo de SKU del perfil)",
        "",
        *_by_profile_table(diag, col("sku_hit_rate"), ORACLE_SKU),
        "",
        *_recency_markdown(diag),
        "### Como se construye el techo",
        "",
        "El generador exporta, para cada cesta de test, el peso de cada categoria antes de "
        "aplicar la mision (afinidad x estacionalidad x ciclo de reposicion), la probabilidad "
        "de cada mision de compra en esa visita, su factor de tamano y la referencia mas "
        "probable de cada categoria (`data/oracle/`, fuera de `data/raw`); el manifiesto "
        "trae los perfiles de mision, el tamano de cada una y la matriz de complementos y "
        "sustitutos. El oraculo conoce el carrito y cuantas categorias distintas tiene la "
        "cesta, pero no la mision. La probabilidad de que cada categoria este en el resto se "
        "calcula por Monte Carlo con muestreo secuencial por importancia: replica el sorteo "
        "del generador paso a paso, forzando que el carrito quede dentro, y pondera cada "
        "muestra por su probabilidad real (exacto salvo ruido de muestreo; detalle en "
        "`src/recommender/oracle.py`). El **oraculo de categoria** recomienda las "
        f"{k} categorias de mas probabilidad fuera del carrito, cada una con su referencia "
        "mas probable; el **oraculo de SKU** ordena por `P(categoria en el resto) x "
        "P(mejor referencia en la cesta)`. Las dos listas se eligen con la mitad de las "
        "muestras y se miden con la otra, asi que el techo es una cota inferior muy "
        "ajustada del maximo. Ningun sistema que solo vea el pasado puede superarlos en "
        "valor esperado: lo que queda entre el techo y el 100 % es entropia del generador.",
        "",
        "El realizado y el esperado deben coincidir salvo ruido (ver comprobaciones):",
        "",
        *expected_rows,
        "",
        f"Muestras por query: {int(diag.ess['n_samples']):,} (mitad de evaluacion: tamano "
        f"efectivo mediano {diag.ess['median']:.0f}, percentil 5 {diag.ess['p05']:.0f}; "
        f"queries sin ninguna muestra valida: {int(diag.ess['n_fallback'])}). Reglas de asociacion: "
        f"{int(diag.rules['n_rules_used'])} de {int(diag.rules['n_rules_total'])} pares con "
        f"confianza >= {diag.rules['min_confidence']:.2f} y lift > 1.",
        "",
        "### Techo esperado por perfil",
        "",
        pl._table(diag.oracle_expected_by_profile),
        "",
        "### Comprobaciones",
        "",
        *checks,
        pl.DIAGNOSTICS_END,
    ]
    return "\n".join(lines)


def to_json(diag: Diagnostics, cfg: RecommenderConfig, predictions_sha: str) -> dict:
    return {
        "generated": dt.date.today().isoformat(),
        "test_start": str(cfg.test_start),
        "top_k": diag.k,
        "recommendations_sha256": predictions_sha,
        "headline": headline(diag),
        "systems": {
            name: {"label": SYSTEM_LABELS[name], "by_profile": s.to_dict(orient="records")}
            for name, s in diag.summaries.items()
        },
        "oracle_expected": diag.oracle_expected,
        "oracle_expected_by_profile": diag.oracle_expected_by_profile.to_dict(orient="records"),
        "monte_carlo": diag.ess,
        "association_rules": diag.rules,
        "recency_slots": {
            name: frame.to_dict(orient="records") for name, frame in diag.recency.items()
        },
        "recency_comparison": (
            None
            if diag.recency_comparison is None
            else diag.recency_comparison.to_dict(orient="records")
        ),
        "checks": [{"check": n, "ok": ok, "detail": d} for n, ok, d in diag.checks],
        "bootstrap": None if diag.bootstrap is None else dataclasses.asdict(diag.bootstrap),
        "comparisons": (
            None if diag.comparisons is None else diag.comparisons.to_dict(orient="records")
        ),
        "pre_fase8": (
            None
            if (table := pre_fase8_comparison(diag, cfg)) is None
            else table.to_dict(orient="records")
        ),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument("--reports", type=Path, default=Path("reports/recommender"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--oracle", type=Path, default=Path("data/oracle"))
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BootstrapConfig.n_resamples,
        help="Remuestreos del bootstrap pareado (0 lo desactiva).",
    )
    parser.add_argument("--no-write", action="store_true", help="No escribir informes.")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help=f"Congelar metrics.json en {SNAPSHOT_FILENAME} y salir.",
    )
    parser.add_argument("--force", action="store_true", help="Con --freeze, sobrescribir.")
    parser.add_argument(
        "--snapshot",
        default=SNAPSHOT_FILENAME,
        help=f"Nombre del snapshot de --freeze (por defecto {SNAPSHOT_FILENAME}).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Predicciones de antes del punto A1, para comparar (se ignora si no existen).",
    )
    parser.add_argument(
        "--reference-a4",
        type=Path,
        default=DEFAULT_REFERENCE_A4_DIR,
        help="Predicciones de antes del punto A4, para comparar (se ignora si no existen).",
    )
    return parser.parse_args(argv)


def _objective_systems(
    cfg: RecommenderConfig, reference_a4: Path, lambdarank: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, bool, str]]]:
    """Los dos sistemas del punto A4 que haya en disco, con su comprobacion de origen."""
    systems: dict[str, pd.DataFrame] = {}
    checks: list[tuple[str, bool, str]] = []
    queries = set(lambdarank["basket_id"])

    sku = load_reference(Path(cfg.predictions_dir), pl.SKU_PREDICTIONS_FILENAME)
    if sku is not None:
        systems[LAMBDARANK_SKU] = sku
        same = set(sku["basket_id"]) == queries
        checks.append(
            (
                "La ablacion con relevancia binaria cubre las mismas queries que el LambdaRank",
                same,
                "",
            )
        )

    frozen = load_reference(reference_a4)
    snapshot = pl._load_same_dataset_snapshot(cfg, PRE_A4_SNAPSHOT)
    if frozen is not None and snapshot is not None:
        sha = _sha256(reference_a4 / "recommendations_test.parquet")
        same_file = sha == snapshot["predicciones_sha256"]["recommendations_test.parquet"]
        same = same_file and set(frozen["basket_id"]) == queries
        checks.append(
            (
                "Las predicciones de referencia (antes de A4) son las congeladas y cubren "
                "las mismas queries",
                same,
                "sha256 = el de " + PRE_A4_SNAPSHOT if same_file else "sha256 distinto",
            )
        )
        systems[REFERENCE_A4] = frozen
    return systems, checks


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = RecommenderConfig(
        processed_dir=args.processed,
        predictions_dir=args.predictions,
        reports_dir=args.reports,
        models_dir=args.models,
    )
    if args.freeze:
        path = freeze_snapshot(cfg, name=args.snapshot, force=args.force)
        print(f"Snapshot congelado en {path}")
        return 0

    reports = Path(cfg.reports_dir)
    # Los snapshots de antes de la Fase 8 se midieron sobre otro dataset: con el actual,
    # sus cifras y sus predicciones no son comparables query a query.
    snapshot = pl._load_same_dataset_snapshot(cfg, SNAPSHOT_FILENAME)
    metrics_path = reports / "metrics.json"
    current = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    )

    pre_a1 = pl._load_same_dataset_snapshot(cfg, PRE_A1_SNAPSHOT)

    start = time.perf_counter()
    test = load_test_set(Path(cfg.predictions_dir))
    reference = load_reference(args.reference) if pre_a1 is not None else None
    reference_sha = (
        _sha256(args.reference / "recommendations_test.parquet") if reference is not None else None
    )
    extras, extra_checks = _objective_systems(cfg, args.reference_a4, test.lambdarank)
    spark = get_spark("grocery-recommender-diagnostics", driver_memory="6g")
    try:
        history = fit_history(spark, cfg, test.queries)
    finally:
        spark.stop()
    print(f"  Historial de los baselines ({time.perf_counter() - start:.0f}s)", flush=True)

    diag = measure(
        test,
        history,
        oracle_dir=args.oracle,
        k=cfg.top_k,
        seed=SEED,
        min_confidence=args.min_confidence,
        n_samples=args.mc_samples,
        snapshot=snapshot,
        current_metrics=current,
        reference=reference,
        pre_a1=pre_a1,
        reference_sha=reference_sha,
        window_start=cfg.test_start,
        extra_systems=extras,
        extra_checks=extra_checks,
        bootstrap=BootstrapConfig(n_resamples=args.bootstrap) if args.bootstrap else None,
    )
    predictions_sha = _sha256(Path(cfg.predictions_dir) / "recommendations_test.parquet")
    block = render_markdown(diag, cfg, predictions_sha)

    if not args.no_write:
        (reports / DIAGNOSTICS_FILENAME).write_text(
            json.dumps(to_json(diag, cfg, predictions_sha), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        md = reports / "metrics.md"
        md.write_text(pl.with_diagnostics(md.read_text(encoding="utf-8"), block), encoding="utf-8")

    h = headline(diag)
    k = cfg.top_k
    print(f"\ncat_hit_rate@{k}:")
    for name in _system_order(diag):
        value = float(diag.summaries[name].iloc[0][f"cat_hit_rate@{k}"])
        share = "" if name in (ORACLE_CAT, ORACLE_SKU) else f"  ({h['pct_of_ceiling'][name]:.1%} del techo)"
        print(f"  {SYSTEM_LABELS[name]:<66} {value:.4f}{share}")
    print(
        f"\nMejor baseline - LambdaRank: {h['gap_best_baseline_minus_lambdarank'] * 100:+.2f} pp "
        f"({SYSTEM_LABELS[h['best_baseline']]})"
    )
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        for name, table in diag.recency.items():
            print(f"\nHuecos por recencia de la categoria - {SYSTEM_LABELS[name]}:")
            print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if diag.recency_comparison is not None:
            print("\nQueries con huecos recientes en la lista de antes:")
            print(diag.recency_comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    failed = [name for name, ok, _ in diag.checks if not ok]
    for name, ok, detail in diag.checks:
        print(f"[{'OK' if ok else 'FALLA'}] {name} {detail}")
    print(f"Total: {time.perf_counter() - start:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

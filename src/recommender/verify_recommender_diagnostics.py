"""Verificacion del diagnostico de la Fase 7: baselines, oraculo y techo teorico (A3, A6, M8).

    python -m data_generation.export_oracle                           # una vez, ~3 min
    python -m src.recommender.verify_recommender_diagnostics
    python -m src.recommender.verify_recommender_diagnostics --freeze  # congelar snapshot

Convierte en codigo las cifras de `docs/diagnostico-fase7.md` que salian de un script ad
hoc. Mide, **sobre las mismas queries de test** que el LambdaRank
(`predictions/recommender_test_queries.parquet`):

1. el LambdaRank tal y como quedo en disco (`predictions/recommendations_test.parquet`),
   junto a sus cifras congeladas en `reports/recommender/baseline_pre_diagnostico.json`;
2. la bateria de baselines independientes del pool (`evaluate.run_baselines`), ajustados
   con el mismo historial que el sistema (todo lo anterior a `test_start`);
3. el oraculo bayesiano (`oracle.py`): el top-5 de categorias por probabilidad real,
   con su valor realizado y su valor esperado por Monte Carlo.

Escribe `reports/recommender/diagnostics.json` y la seccion de diagnostico de
`reports/recommender/metrics.md`, y termina con codigo de salida 1 si falla alguna
comprobacion de coherencia. No entrena ni modifica ningun modelo.
"""

from __future__ import annotations

import argparse
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

from src.etl.repurchase import repurchase_features
from src.etl.schemas import read_processed
from src.etl.session import get_spark
from src.recommender import candidates as cand
from src.recommender import evaluate as ev
from src.recommender import features as feat
from src.recommender import oracle as orc
from src.recommender import pipeline as pl
from src.recommender import splits
from src.recommender.config import SEED, RecommenderConfig
from src.recommender.schema import PROFILE_LABELS

SNAPSHOT_FILENAME = "baseline_pre_diagnostico.json"
DIAGNOSTICS_FILENAME = "diagnostics.json"

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
ORACLE_CAT = "oracle_category"
ORACLE_SKU = "oracle_sku"
SYSTEM_LABELS: dict[str, str] = {
    ORACLE_CAT: "Oraculo de categoria (techo)",
    ORACLE_SKU: "Oraculo de SKU (techo)",
    LAMBDARANK: "LambdaRank (Fase 7c, predicciones en disco)",
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
def freeze_snapshot(cfg: RecommenderConfig, *, force: bool = False) -> Path:
    """Congela `metrics.json` y los hashes de las predicciones de las que sale.

    Es la referencia "antes" de las Sesiones 2 en adelante del plan de mejora, como
    `baseline_fase3.json` lo fue para la Fase 7.
    """
    reports = Path(cfg.reports_dir)
    path = reports / SNAPSHOT_FILENAME
    if path.exists() and not force:
        raise FileExistsError(f"{path} ya existe; usar --force para sobrescribirlo")
    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    payload = {
        "descripcion": (
            "Metricas del LambdaRank de la Fase 7c congeladas antes de aplicar los puntos "
            "de docs/diagnostico-fase7.md. No regenerar: es la referencia del antes."
        ),
        "congelado": dt.date.today().isoformat(),
        "commit": _git_head(),
        "fuente": (reports / "metrics.json").as_posix(),
        "predicciones_sha256": {
            name: _sha256(Path(cfg.predictions_dir) / name) for name in PREDICTION_FILES
        },
        "modelo_sha256": _sha256(Path(cfg.models_dir) / "recommender_ranker_lgbm.txt"),
        "metrics": metrics,
    }
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


def fit_history(spark: SparkSession, cfg: RecommenderConfig) -> dict[str, pd.DataFrame]:
    """Historial anterior a `test_start`, con las mismas funciones que `pipeline.fit_sources`."""
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

    customer_products = cand.fit_customer_products(hist_b, hist_i).select(
        "customer_id", "product_id", F.col("hist_n_baskets").alias("n_baskets")
    )
    customer_categories = repurchase_features(
        hist_i,
        hist_b,
        products,
        tables["customers"],
        reference_date=feat.default_reference_date(cfg.test_start),
    ).select(
        "customer_id",
        "category",
        "n_purchase_days",
        # Como texto: una fecha viaja bien a pandas, pero asi no hay dudas de zona horaria.
        F.date_format("last_purchase_date", "yyyy-MM-dd").alias("last_purchase_date"),
        "expected_repurchase_days",
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
    checks: list[tuple[str, bool, str]] = field(default_factory=list)


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
) -> Diagnostics:
    products = history["products"]
    queries = test.queries

    def summary(top_k: pd.DataFrame) -> pd.DataFrame:
        return ev.system_summary(top_k, queries, test.target, products, k=k)

    summaries: dict[str, pd.DataFrame] = {}
    checks: list[tuple[str, bool, str]] = []

    # --- Oraculo ---
    inputs = orc.load_inputs(oracle_dir, queries, test.context, products)
    same_size = bool((inputs.n_target == queries["n_target"].to_numpy()).all())
    checks.append(
        (
            "Una linea por categoria: el target en categorias coincide con n_target",
            same_size,
            "" if same_size else "el oraculo asume una linea por categoria",
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

    per_query = pd.DataFrame(
        {
            "profile": queries["profile"].to_numpy(),
            f"cat_hit_rate@{k}": mc.cat_hit,
            f"cat_precision@{k}": mc.cat_precision,
            f"sku_hit_rate@{k}": mc.sku_hit,
            f"sku_precision@{k}": mc.sku_precision,
        }
    )
    expected_by_profile = _profile_table(per_query, [c for c in per_query.columns if "@" in c])
    expected = expected_by_profile.iloc[0].drop("grupo").astype(float).to_dict()

    # --- LambdaRank ---
    summaries[LAMBDARANK] = summary(test.lambdarank)
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
    for name, top_k in ev.run_baselines(
        data, k=k, seed=seed, min_confidence=min_confidence
    ).items():
        summaries[name] = summary(top_k)

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
        if n not in (ORACLE_CAT, ORACLE_SKU)
    )
    checks.append(
        (
            "Ningun sistema supera al oraculo de categoria en cat_hit_rate",
            best_other[0] <= realised_hit,
            f"mejor sistema {best_other[1]} = {best_other[0]:.4f}",
        )
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
        checks=checks,
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
    return {
        "best_baseline": best,
        "best_baseline_cat_hit_rate": baselines[best],
        "lambdarank_cat_hit_rate": lr,
        "gap_best_baseline_minus_lambdarank": baselines[best] - lr,
        "ceiling_cat_hit_rate": ceiling,
        "ceiling_sku_hit_rate": sku_ceiling,
        "pct_of_ceiling": {
            name: float(total[name][col]) / ceiling
            for name in [LAMBDARANK, *ev.BASELINE_LABELS]
        },
        "pct_of_sku_ceiling": {
            name: float(total[name][sku_col]) / sku_ceiling
            for name in [LAMBDARANK, *ev.BASELINE_LABELS]
        },
    }


def _system_order(diag: Diagnostics) -> list[str]:
    k = diag.k
    rest = sorted(
        ev.BASELINE_LABELS,
        key=lambda n: -float(diag.summaries[n].iloc[0][f"cat_hit_rate@{k}"]),
    )
    return [ORACLE_CAT, ORACLE_SKU, LAMBDARANK, *rest]


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


def render_markdown(diag: Diagnostics, cfg: RecommenderConfig, predictions_sha: str) -> str:
    k = diag.k
    h = headline(diag)
    col = lambda m: f"{m}@{k}"  # noqa: E731
    total = {name: s.iloc[0] for name, s in diag.summaries.items()}
    cat_ceiling = h["ceiling_cat_hit_rate"]
    sku_ceiling = h["ceiling_sku_hit_rate"]

    main = [
        f"| Sistema | cat_hit_rate@{k} | % techo | cat_precision@{k} | sku_hit_rate@{k} "
        f"| % techo SKU | sku_precision@{k} | NDCG@{k} |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                    _num(row[col("cat_hit_rate")]),
                    "—" if is_oracle else _pct(row[col("cat_hit_rate")] / cat_ceiling),
                    _num(row[col("cat_precision")]),
                    _num(row[col("sku_hit_rate")]),
                    "—" if is_oracle else _pct(row[col("sku_hit_rate")] / sku_ceiling),
                    _num(row[col("sku_precision")]),
                    _num(row[col("ndcg")]),
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
        "pipeline: si se reentrena, hay que volver a lanzar el verificador. Puntos A3, A6 "
        "y M8 de `docs/diagnostico-fase7.md`.",
        "",
        "### Lectura rapida",
        "",
        f"- **Mejor baseline en cat_hit_rate@{k}:** {SYSTEM_LABELS[best]}, "
        f"{_num(h['best_baseline_cat_hit_rate'])} frente a {_num(h['lambdarank_cat_hit_rate'])} "
        f"del LambdaRank: {_pp(gap)}"
        + (" a favor del baseline." if gap > 0 else " (el LambdaRank va por delante)."),
        f"- **Techo teorico de cat_hit_rate@{k}:** {_num(cat_ceiling)}. El LambdaRank "
        f"alcanza el {_pct(pct[LAMBDARANK])} y el mejor baseline el {_pct(pct[best])}.",
        f"- **Techo de sku_hit_rate@{k}:** {_num(sku_ceiling)}. El LambdaRank alcanza el "
        f"{_pct(h['pct_of_sku_ceiling'][LAMBDARANK])}.",
        "",
        "### Todos los sistemas, total",
        "",
        "Las columnas \"% techo\" dividen por el oraculo correspondiente sobre las mismas "
        "queries. Los baselines nunca recomiendan una categoria que ya esta en el carrito; "
        "el LambdaRank si puede hacerlo (punto A2).",
        "",
        *main,
        "",
        *snap_lines,
        f"### cat_hit_rate@{k} por perfil (entre parentesis, % del techo del perfil)",
        "",
        *_by_profile_table(diag, col("cat_hit_rate"), ORACLE_CAT),
        "",
        f"### sku_hit_rate@{k} por perfil (entre parentesis, % del techo de SKU del perfil)",
        "",
        *_by_profile_table(diag, col("sku_hit_rate"), ORACLE_SKU),
        "",
        "### Como se construye el techo",
        "",
        "El generador exporta, para cada cesta de test, el peso de cada categoria antes del "
        "primer sorteo (afinidad x estacionalidad x ciclo de reposicion) y la referencia mas "
        "probable de cada una (`data/oracle/`, fuera de `data/raw`). Como el sorteo es "
        "secuencial y sin reemplazo, la probabilidad de que cada categoria este en el resto "
        "de la cesta, dado el carrito, se calcula por Monte Carlo: una carrera de relojes "
        "exponenciales con los lifts de afinidad, condicionada al carrito por muestreo por "
        "importancia (exacta salvo ruido de muestreo; detalle en "
        "`src/recommender/oracle.py`). El **oraculo de categoria** recomienda las "
        f"{k} categorias de mas peso fuera del carrito (con el lift de las disparadoras del "
        "carrito aplicado), cada una con su referencia mas probable: es la lista optima "
        "salvo en los 10 pares de afinidad, asi que su valor es una cota inferior muy "
        "ajustada del maximo. El **oraculo de SKU** ordena por `P(categoria en el resto) x "
        "P(mejor referencia)`, estimada con la mitad de las muestras y medida con la otra. "
        "Ningun sistema que solo vea el pasado puede superarlos en valor esperado: lo que "
        "queda entre el techo y el 100 % es entropia del generador.",
        "",
        "El realizado y el esperado deben coincidir salvo ruido (ver comprobaciones):",
        "",
        *expected_rows,
        "",
        f"Muestras por query: {int(diag.ess['n_samples']):,} (mitad de evaluacion: tamano "
        f"efectivo mediano {diag.ess['median']:.0f}, percentil 5 {diag.ess['p05']:.0f}; "
        f"queries con peso de respaldo: {int(diag.ess['n_fallback'])}). Reglas de asociacion: "
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
        "checks": [{"check": n, "ok": ok, "detail": d} for n, ok, d in diag.checks],
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
    parser.add_argument("--no-write", action="store_true", help="No escribir informes.")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help=f"Congelar metrics.json en {SNAPSHOT_FILENAME} y salir.",
    )
    parser.add_argument("--force", action="store_true", help="Con --freeze, sobrescribir.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = RecommenderConfig(
        processed_dir=args.processed,
        predictions_dir=args.predictions,
        reports_dir=args.reports,
        models_dir=args.models,
    )
    if args.freeze:
        print(f"Snapshot congelado en {freeze_snapshot(cfg, force=args.force)}")
        return 0

    reports = Path(cfg.reports_dir)
    snapshot_path = reports / SNAPSHOT_FILENAME
    snapshot = (
        json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.is_file() else None
    )
    metrics_path = reports / "metrics.json"
    current = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    )

    start = time.perf_counter()
    test = load_test_set(Path(cfg.predictions_dir))
    spark = get_spark("grocery-recommender-diagnostics", driver_memory="6g")
    try:
        history = fit_history(spark, cfg)
    finally:
        spark.stop()
    print(f"  Historial anterior a {cfg.test_start} ({time.perf_counter() - start:.0f}s)", flush=True)

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
    failed = [name for name, ok, _ in diag.checks if not ok]
    for name, ok, detail in diag.checks:
        print(f"[{'OK' if ok else 'FALLA'}] {name} {detail}")
    print(f"Total: {time.perf_counter() - start:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

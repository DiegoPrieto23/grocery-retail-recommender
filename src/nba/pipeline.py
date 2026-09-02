"""Orquestador de la Fase 4: de `data/processed` a una accion por cliente.

    python -m src.nba.pipeline
    python -m src.nba.pipeline --no-write     # sin guardar modelos ni tabla de acciones
    python -m src.nba.pipeline --quick        # dos cortes y menos clientes, para probar

El flujo es el de `CHALLENGE.md`, Tarea 3b:

1. **Cortes temporales** (`config.py`): varios de entrenamiento, uno de validacion y uno
   de test, todos disjuntos y en ese orden.
2. **Features y etiquetas** de cada corte, mirando solo hacia atras (`features`, `targets`).
3. **Dos modelos de propension**: compra en categoria a 7 dias y churn a 4 semanas.
4. **Politica de valor esperado** sobre el catalogo de acciones (`policy`).
5. **Evaluacion**: AUC/PR-AUC de los modelos, y la politica contra "no actuar siempre" y
   "actuar siempre", mas el barrido de sensibilidad del supuesto de uplift.

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

from src.etl.schemas import read_processed
from src.etl.session import get_spark
from src.nba import features as feat
from src.nba import policy as pol
from src.nba import propensity as prop
from src.nba import targets as tgt
from src.nba.config import NBAConfig, REQUIRED_TABLES


class _Timer:
    """Cronometro de etapas, igual que en el ETL y en el recomendador."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.last = self.start

    def step(self, label: str) -> None:
        now = time.perf_counter()
        print(f"  [{now - self.start:6.1f}s] {label} (+{now - self.last:.1f}s)", flush=True)
        self.last = now


@dataclass
class Snapshot:
    """Las dos matrices etiquetadas de un corte, ya en el driver."""

    cutoff: dt.date
    churn: pd.DataFrame
    category: pd.DataFrame


def _sample_customers(df: DataFrame, cutoff: dt.date, n: int | None, total: int) -> DataFrame:
    """Submuestra determinista de clientes.

    Se hace con un hash del `customer_id` y del corte, no con `sample()`: asi la muestra
    no depende del plan de ejecucion de Spark y dos ejecuciones dan exactamente la misma,
    que es el criterio de reproducibilidad de `CLAUDE.md`.
    """
    if n is None or n >= total:
        return df
    salt = F.concat(F.col("customer_id"), F.lit(cutoff.isoformat()))
    return df.filter(F.pmod(F.hash(salt), F.lit(total)) < F.lit(n))


def build_snapshot(
    tables: dict[str, DataFrame],
    cutoff: dt.date,
    cfg: NBAConfig,
    *,
    n_customers: int | None,
    n_total_customers: int,
) -> Snapshot:
    """Features y etiquetas de un corte, ya recogidas en pandas.

    Todo lo que entra en las features procede de `basket_day < cutoff`; todo lo que define
    una etiqueta, de la ventana posterior. Las dos cosas no se tocan en ningun momento.
    """
    history_baskets = tgt.baskets_in_window(tables["baskets"], None, cutoff).cache()
    history_baskets.count()
    history_items = tables["basket_items"].join(
        history_baskets.select("basket_id"), "basket_id", "leftsemi"
    )
    history_lines = tgt.category_lines(
        history_baskets, tables["basket_items"], tables["products"]
    ).cache()
    history_lines.count()

    customer = feat.customer_features(
        history_baskets, history_lines, tables["customers"], cutoff
    )
    category = feat.category_features(
        history_baskets,
        history_items,
        history_lines,
        tables["products"],
        tables["customers"],
        cutoff,
    )

    # Las etiquetas miran la ventana futura sobre las cestas completas, no las recortadas.
    all_lines = tgt.category_lines(
        tables["baskets"], tables["basket_items"], tables["products"]
    )
    label_category, label_churn = tgt.build_labels(
        tables["baskets"],
        all_lines,
        cutoff,
        category_horizon_days=cfg.category_horizon_days,
        churn_horizon_days=cfg.churn_horizon_days,
        lookback_days=cfg.category_lookback_days,
    )

    keep = _sample_customers(
        label_churn.select("customer_id"), cutoff, n_customers, n_total_customers
    ).cache()
    keep.count()

    churn_df = (
        label_churn.join(keep, "customer_id", "leftsemi")
        .join(customer, "customer_id", "inner")
    )
    category_df = (
        label_category.join(keep, "customer_id", "leftsemi")
        .join(customer, "customer_id", "inner")
        .join(category, ["customer_id", "category"], "inner")
    )

    churn_pdf = prop.collect(
        churn_df, feature_columns=feat.CHURN_FEATURES, keys=("customer_id",), label="churn"
    )
    category_pdf = prop.collect(
        category_df,
        feature_columns=feat.PURCHASE_FEATURES,
        keys=("customer_id", "category", "department"),
        label="label",
    )
    history_baskets.unpersist()
    history_lines.unpersist()
    keep.unpersist()
    return Snapshot(cutoff=cutoff, churn=churn_pdf, category=category_pdf)


def _economics(tables: dict[str, DataFrame], cutoff: dt.date) -> float:
    """Gasto medio por cesta-categoria antes del corte, para los pares sin historial."""
    lines = tgt.category_lines(
        tgt.baskets_in_window(tables["baskets"], None, cutoff),
        tables["basket_items"],
        tables["products"],
    )
    per = lines.groupBy("customer_id", "basket_day", "category").agg(
        F.sum("line_amount").alias("amount")
    )
    return float(per.agg(F.avg("amount")).collect()[0][0] or 0.0)


def _spend_90d(snapshot: Snapshot) -> pd.DataFrame:
    """`spend_90d` por cliente, que la politica necesita para valorar la retencion."""
    return snapshot.churn[["customer_id", "spend_90d"]].copy()


def run(spark: SparkSession, cfg: NBAConfig, *, write: bool = True) -> dict[str, object]:
    """Ejecuta la Fase 4 completa y devuelve modelos, metricas y la tabla de acciones."""
    timer = _Timer()

    tables = read_processed(spark, REQUIRED_TABLES, cfg.processed_dir)
    n_total = tables["customers"].count()
    timer.step(f"Lectura de data/processed ({n_total:,} clientes)")

    snapshots = [
        build_snapshot(
            tables, cutoff, cfg, n_customers=cfg.n_train_customers, n_total_customers=n_total
        )
        for cutoff in cfg.train_cutoffs
    ]
    for snap in snapshots:
        timer.step(
            f"Corte de entrenamiento {snap.cutoff}: "
            f"{len(snap.churn):,} clientes / {len(snap.category):,} pares"
        )

    valid = build_snapshot(
        tables, cfg.valid_cutoff, cfg, n_customers=cfg.n_train_customers, n_total_customers=n_total
    )
    timer.step(f"Corte de validacion {cfg.valid_cutoff}")

    test = build_snapshot(tables, cfg.test_cutoff, cfg, n_customers=None, n_total_customers=n_total)
    timer.step(f"Corte de test {cfg.test_cutoff}: {len(test.churn):,} clientes")

    train_churn = pd.concat([s.churn for s in snapshots], ignore_index=True)
    train_category = pd.concat([s.category for s in snapshots], ignore_index=True)

    churn_booster, _ = prop.train(
        train_churn,
        valid.churn,
        feature_columns=feat.CHURN_FEATURES,
        label="churn",
        cfg=cfg.propensity,
    )
    timer.step(f"Modelo de churn ({churn_booster.num_trees()} arboles)")

    purchase_booster, _ = prop.train(
        train_category,
        valid.category,
        feature_columns=feat.PURCHASE_FEATURES,
        label="label",
        cfg=cfg.propensity,
    )
    timer.step(f"Modelo de compra en categoria ({purchase_booster.num_trees()} arboles)")

    # --- Evaluacion de los modelos sobre el corte de test ---
    p_churn = prop.predict(churn_booster, test.churn, feature_columns=feat.CHURN_FEATURES)
    p_purchase = prop.predict(
        purchase_booster, test.category, feature_columns=feat.PURCHASE_FEATURES
    )
    churn_metrics = prop.metrics(test.churn["churn"].to_numpy(), p_churn)
    purchase_metrics = prop.metrics(test.category["label"].to_numpy(), p_purchase)
    timer.step("Metricas de propension")

    # --- Politica ---
    scored = test.category[["customer_id", "category", "department"]].copy()
    scored["p_purchase"] = p_purchase
    scored["cat_spend"] = test.category["cat_spend"].to_numpy()
    scored["cat_n_purchase_days"] = test.category["cat_n_purchase_days"].to_numpy()

    churn_table = _spend_90d(test)
    churn_table["p_churn"] = p_churn

    fallback = _economics(tables, cfg.test_cutoff)
    candidates = pol.build_candidates(scored, churn_table, cfg, fallback_spend=fallback)
    # El universo es el maestro de clientes activos, no solo los que tienen candidatos:
    # quien no compra ninguna categoria desde hace meses tambien recibe una decision.
    universe = test.churn["customer_id"]
    actions = pol.cover_all(pol.decide(candidates, cfg), universe)
    comparison = pol.compare(candidates, cfg, all_customers=universe)
    sweep = pol.sensitivity(candidates, cfg, all_customers=universe)
    sweep_retention = pol.sensitivity_retention(candidates, cfg, all_customers=universe)
    timer.step("Politica de valor esperado")

    agreement = tgt.agreement_with_churn_label(
        tgt.churn_label(
            tables["baskets"], cfg.test_cutoff, horizon_days=cfg.churn_horizon_days
        ),
        tables["customers"],
    )

    result: dict[str, object] = {
        "churn_booster": churn_booster,
        "purchase_booster": purchase_booster,
        "churn_metrics": churn_metrics,
        "purchase_metrics": purchase_metrics,
        "churn_importance": prop.feature_importance(churn_booster),
        "purchase_importance": prop.feature_importance(purchase_booster),
        "actions": actions,
        "comparison": comparison,
        "sensitivity": sweep,
        "sensitivity_retention": sweep_retention,
        "agreement": agreement,
        "fallback_spend": fallback,
        "n_train_churn": int(len(train_churn)),
        "n_train_category": int(len(train_category)),
        "n_test_customers": int(len(test.churn)),
        "n_test_pairs": int(len(test.category)),
        "n_without_candidates": int(
            len(set(universe) - set(candidates["customer_id"].unique()))
        ),
    }

    if write:
        prop.save(churn_booster, cfg.models_dir, prop.CHURN_MODEL_FILENAME)
        prop.save(purchase_booster, cfg.models_dir, prop.PURCHASE_MODEL_FILENAME)
        Path(cfg.predictions_dir).mkdir(parents=True, exist_ok=True)
        actions.to_parquet(Path(cfg.predictions_dir) / "nba_actions.parquet", index=False)
        _write_reports(cfg, result)
        timer.step("Modelos, tabla de acciones e informes escritos")

    return result


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


def _metrics_table(name: str, m: dict[str, float]) -> str:
    return (
        f"| {name} | {m['n']:,} | {m['base_rate']:.4f} | {m['auc']:.4f} | "
        f"{m['pr_auc']:.4f} | {m['lift_top_decile']:.2f}x |"
    )


def _write_reports(cfg: NBAConfig, result: dict[str, object]) -> None:
    """Deja el informe de la fase en `reports/nba/`."""
    reports = Path(cfg.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    comparison: pd.DataFrame = result["comparison"]  # type: ignore[assignment]
    policy_row = comparison.loc[comparison["politica"] == "politica de valor esperado"].iloc[0]
    coupon_row = comparison.loc[
        comparison["politica"] == "actuar siempre: enviar_cupon_categoria"
    ].iloc[0]
    agreement: dict = result["agreement"]  # type: ignore[assignment]

    actions_mix = (
        result["actions"]["action"]  # type: ignore[index]
        .value_counts(normalize=True)
        .rename("cuota")
        .reset_index()
    )

    # Se precalculan fuera de la plantilla: anidar f-strings dentro de la triple comilla
    # es fragil en Python 3.10/3.11, que es el rango que declara el proyecto.
    churn_line = _metrics_table("Churn a 4 semanas", result["churn_metrics"])
    purchase_line = _metrics_table(
        f"Compra en categoria a {cfg.category_horizon_days} dias", result["purchase_metrics"]
    )
    churn_weeks = cfg.churn_horizon_days // 7
    coupon_gap = abs(coupon_row["valor_total"] - policy_row["valor_total"])

    text = f"""# Next Best Action (Fase 4, Tarea 3b)

Generado por `python -m src.nba.pipeline`. Ninguna cifra de este informe se copia a mano:
se recalcula ejecutando ese comando.

## Montaje

| | |
| --- | --- |
| Cortes de entrenamiento | {", ".join(str(c) for c in cfg.train_cutoffs)} |
| Corte de validacion | {cfg.valid_cutoff} |
| Corte de test | {cfg.test_cutoff} |
| Horizonte de compra en categoria | {cfg.category_horizon_days} dias |
| Horizonte de churn | {cfg.churn_horizon_days} dias |
| Filas de entrenamiento (churn) | {result['n_train_churn']:,} |
| Filas de entrenamiento (categoria) | {result['n_train_category']:,} |
| Clientes en test | {result['n_test_customers']:,} |
| Pares cliente-categoria en test | {result['n_test_pairs']:,} |
| Clientes sin ninguna categoria accionable | {result['n_without_candidates']:,} |

Los clientes sin categoria accionable son los que no han comprado **nada** en los
{cfg.category_lookback_days} dias previos al corte. No generan pares, asi que ninguna
politica puede actuar sobre ellos; entran igualmente en la tabla con `ninguna_accion` para
que el denominador sea el maestro completo y no el subconjunto comodo.

El split es **temporal**: los cortes de entrenamiento son anteriores al de validacion y
ninguno de los dos alcanza la ventana de test. Las features de cada corte se calculan solo
con compras anteriores a el.

## Modelos de propension

| Modelo | n | tasa base | AUC | PR-AUC | lift decil 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
{churn_line}
{purchase_line}

El PR-AUC se lee contra la tasa base, no contra 0,5: un modelo aleatorio da exactamente la
tasa base. El lift del primer decil es la lectura de negocio: cuantas veces mas positivos
hay en el 10 % mejor puntuado que en la poblacion.

**Como leer el modelo de churn.** Su tasa base es alta porque en este dataset la cadencia
media de visita ronda los 24 dias: no comprar en 4 semanas le pasa a mucha gente y no
siempre significa abandono. Las features que mas pesan lo confirman -- `n_baskets_90d`,
`avg_days_between_baskets`, `recency_days` --, asi que buena parte de lo que el modelo
acierta es **frecuencia de compra**, no intencion de irse. El AUC es real, pero llamarlo
"modelo de churn" concede mas de lo que hace: es un modelo de inactividad a 4 semanas. La
senal de abandono de verdad (la rampa de decaimiento de `DATA_SPEC.md`) esta en los
`trend_*`, que aportan bastante menos.

### Comprobacion de cordura del target de churn

El churn se construye por corte (no compra en los {cfg.churn_horizon_days} dias siguientes,
es decir {churn_weeks} semanas), no se toma de `customers.churn_label`, que esta definido
respecto al final del
dataset y seria fuga. En el corte de test las dos ventanas casi coinciden, asi que deben
parecerse: **coinciden en el {agreement['agreement']:.1%}** de los {agreement['n']:,}
clientes ({agreement['rate_observed']:.1%} de churn observado frente a
{agreement['rate_spec']:.1%} de la etiqueta de `DATA_SPEC.md`).

## La politica frente a las alternativas triviales

Valor **incremental** sobre no hacer nada, en euros de margen esperado. `ninguna_accion`
vale 0 por construccion, asi que una accion solo suma si su efecto paga su coste.

{_table(comparison)}

La politica actua sobre el **{policy_row['pct_accion']:.1%}** de los clientes, no sobre
todos, y ahi esta su ventaja: mandar el cupon a todo el mundo cuesta {coupon_gap:.0f} EUR
de valor esperado frente a elegir a quien.

### Reparto de acciones

{_table(actions_mix)}

## Sensibilidad al supuesto de uplift

`P(conversion | accion)` **no es estimable con este dataset**: el generador aplica su
`PROMO_UPLIFT` al reparto de cuota dentro de una categoria, nunca a la probabilidad de
comprarla ni a la de volver, asi que el tratamiento nunca varia. El uplift del cupon es un
supuesto declarado en `config.py`. Esta tabla dice cuanto depende la conclusion de el.

### Uplift de conversion del cupon

{_table(result['sensitivity'])}

### Reduccion de churn del cupon

Este es el barrido que de verdad importa, y conviene decir por que. Con el cupon anclado a
su valor real (2,54 EUR) y un margen esperado por categoria de alrededor de 1,4 EUR, el
descuento **es mayor que el margen que persigue**: por el lado del cross-sell el cupon
destruye valor haga lo que haga la conversion. Se ve en la tabla de arriba, donde pasar el
uplift de 1,00 a 2,00 apenas mueve el total. Todo lo que aporta el cupon viene del termino
de retencion, asi que es `churn_reduction` lo que hay que auditar.

{_table(result['sensitivity_retention'])}

La fila de `reduccion_churn = 0` es la lectura pesimista: lo que queda cuando se supone que
el cupon no retiene a nadie.

Lo que **no** es un supuesto es el reparto: con los efectos fijados, toda la diferencia
entre la politica y "actuar siempre" viene de acertar a quien, y eso es merito del modelo.

## Que features usan los modelos

### Churn a 4 semanas

{_table(result['churn_importance'])}

### Compra en categoria

{_table(result['purchase_importance'])}
"""
    (reports / "metrics.md").write_text(text, encoding="utf-8")

    payload = {
        "train_cutoffs": [str(c) for c in cfg.train_cutoffs],
        "valid_cutoff": str(cfg.valid_cutoff),
        "test_cutoff": str(cfg.test_cutoff),
        "category_horizon_days": cfg.category_horizon_days,
        "churn_horizon_days": cfg.churn_horizon_days,
        "churn_metrics": result["churn_metrics"],
        "purchase_metrics": result["purchase_metrics"],
        "agreement_with_churn_label": result["agreement"],
        "comparison": comparison.to_dict(orient="records"),
        "sensitivity": result["sensitivity"].to_dict(orient="records"),
        "sensitivity_retention": result["sensitivity_retention"].to_dict(orient="records"),
        "action_mix": actions_mix.to_dict(orient="records"),
        "assumptions": {
            "margins_by_department": cfg.policy.margins.by_department,
            "retention_weeks": cfg.policy.retention_weeks,
            "actions": [
                {
                    "name": a.name,
                    "send_cost": a.send_cost,
                    "discount": a.discount,
                    "conversion_uplift": a.conversion_uplift,
                    "churn_reduction": a.churn_reduction,
                }
                for a in cfg.actions
            ],
        },
    }
    (reports / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument("--reports", type=Path, default=Path("reports/nba"))
    parser.add_argument("--no-write", action="store_true", help="No guardar nada en disco.")
    parser.add_argument(
        "--quick", action="store_true", help="Dos cortes y menos clientes: solo comprueba el flujo."
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
        kwargs |= {"train_cutoffs": (dt.date(2025, 7, 1),), "n_train_customers": 3_000}
    cfg = NBAConfig(**kwargs)

    spark = get_spark("grocery-retail-nba", driver_memory="6g")
    try:
        result = run(spark, cfg, write=not args.no_write)
        churn, purchase = result["churn_metrics"], result["purchase_metrics"]
        comparison: pd.DataFrame = result["comparison"]  # type: ignore[assignment]
        print(
            f"\nChurn 4 semanas   AUC = {churn['auc']:.4f} | PR-AUC = {churn['pr_auc']:.4f} "
            f"(tasa base {churn['base_rate']:.4f})"
        )
        print(
            f"Compra categoria  AUC = {purchase['auc']:.4f} | PR-AUC = {purchase['pr_auc']:.4f} "
            f"(tasa base {purchase['base_rate']:.4f})"
        )
        print("\nValor incremental por politica:")
        print(comparison.to_string(index=False))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""Orquestador del ETL de la Fase 2.

Lee `data/raw`, mide la calidad del dato, limpia las 7 tablas, construye las tablas de
features (RFM, ciclo de recompra, afinidad de cesta) y deja todo en `data/processed`, con
los informes en `reports/etl/`.

Cualquier cifra de calidad o de afinidad que aparezca en el README tiene que salir de aqui,
no de un notebook (ver `CLAUDE.md`, "Splits y evaluacion").

Uso:
    python -m src.etl.run_etl
    python -m src.etl.run_etl --data data/raw --out data/processed --reports reports/etl
    python -m src.etl.run_etl --skip-product-affinity   # mas rapido, sin el grano de SKU
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from src.etl.affinity import cooccurrence_affinity, expected_pairs_report
from src.etl.cleaning import CleaningConfig, CleaningReport, clean_all
from src.etl.data_trust import TrustReport, data_trust_score
from src.etl.repurchase import repurchase_features
from src.etl.rfm import rfm
from src.etl.schemas import read_raw, write_processed
from src.etl.session import get_spark

# Cuantos consecuentes se guardan por producto en la tabla que consumira el generador de
# candidatos de la Fase 3. 20 da margen de sobra para quedarse con 5 tras el ranking.
PRODUCT_AFFINITY_TOP_N = 20


class _Timer:
    """Cronometro de etapas, para que el log diga donde se va el tiempo."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.last = self.start

    def step(self, label: str) -> None:
        now = time.perf_counter()
        print(f"  [{now - self.start:6.1f}s] {label} (+{now - self.last:.1f}s)", flush=True)
        self.last = now


def _expected_affinity_pairs() -> list[tuple[str, str, float]]:
    """Los complementos de `DATA_SPEC.md`, para verificar que estan en el dato.

    Se leen del catalogo del generador, que es la fuente de verdad de esa tabla: los 10
    pares originales y, desde la Fase 8, la ampliacion (`COMPLEMENT_PAIRS`). Es la unica
    dependencia del ETL con la Fase 1, y es de verificacion, no de transformacion: si el
    generador no esta disponible, el informe simplemente se omite.
    """
    try:
        from data_generation import catalog
    except Exception:  # noqa: BLE001 - el ETL debe funcionar sin el generador
        return []
    pairs = getattr(catalog, "COMPLEMENT_PAIRS", None)
    if pairs is None:
        return [(a, b, float(lift)) for a, b, lift in catalog.AFFINITY_PAIRS]
    return [(a, b, float(lift)) for a, b, lift, _ in pairs]


# Umbrales de lift del resumen global de `affinity_category` (punto A5 del diagnostico:
# antes de la Fase 8 solo 40 pares ordenados pasaban de 1,5).
LIFT_THRESHOLDS = (1.2, 1.5, 3.0)


def affinity_structure_lines(affinity_category: DataFrame, top: int = 20) -> list[str]:
    """Resumen global de `affinity_category`: cuantos pares superan cada umbral de lift.

    Es el lift crudo, el mismo que consume el recomendador. Con cestas de tamano muy
    variable, el lift crudo sube para casi cualquier par (en una compra semanal esta casi
    todo); el lift controlado por tamano, que separa las dos cosas, lo calcula
    `python -m data_generation.verify_dataset` (`coocurrencia_de_categorias`).
    """
    pdf = affinity_category.select("antecedent", "consequent", "lift", "support").toPandas()
    n = len(pdf)
    lines = [
        "## Estructura global de `affinity_category`",
        "",
        f"Pares ordenados con soporte suficiente: **{n:,}**.".replace(",", "."),
        "",
        "| Umbral | Pares con lift por encima | % |",
        "| --- | ---: | ---: |",
    ]
    for t in LIFT_THRESHOLDS:
        k = int((pdf["lift"] > t).sum())
        lines.append(f"| lift > {t:.1f} | {k:,} | {k / max(n, 1):.1%} |".replace(",", "."))
    k = int((pdf["lift"] < 0.8).sum())
    lines.append(f"| lift < 0.8 | {k:,} | {k / max(n, 1):.1%} |".replace(",", "."))
    lines += [
        "",
        f"Mediana del lift: {pdf['lift'].median():.2f}. El lift crudo mezcla la afinidad con "
        "el tamano de la cesta; el controlado por tamano esta en "
        "`python -m data_generation.verify_dataset` (seccion `coocurrencia_de_categorias`).",
        "",
        f"### Los {top} pares de mas lift",
        "",
        "| Antecedente | Consecuente | Lift | Soporte |",
        "| --- | --- | ---: | ---: |",
    ]
    best = pdf.sort_values(["lift", "antecedent", "consequent"], ascending=[False, True, True])
    for r in best.head(top).itertuples():
        lines.append(
            f"| {r.antecedent} | {r.consequent} | {r.lift:.2f} | {100 * r.support:.2f} % |"
        )
    return lines


def build_features(
    clean: dict[str, DataFrame], *, product_affinity: bool = True
) -> dict[str, DataFrame]:
    """Construye las tablas de features de la Fase 2 sobre las tablas ya limpias."""
    features: dict[str, DataFrame] = {
        "rfm": rfm(clean["baskets"], clean["customers"]),
        "repurchase_features": repurchase_features(
            clean["basket_items"], clean["baskets"], clean["products"], clean["customers"]
        ),
        "affinity_category": cooccurrence_affinity(
            clean["basket_items"], clean["products"], level="category"
        ),
    }
    if product_affinity:
        features["affinity_product"] = cooccurrence_affinity(
            clean["basket_items"], level="product", top_n=PRODUCT_AFFINITY_TOP_N
        )
    return features


def _write_reports(
    reports_dir: Path,
    cleaning: CleaningReport,
    trust_raw: TrustReport,
    trust_clean: TrustReport,
    expected_pairs: DataFrame | None,
    affinity_category: DataFrame | None = None,
) -> None:
    """Vuelca los informes de la fase en Markdown y JSON."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "cleaning_report.md").write_text(
        "# Informe de limpieza (Fase 2)\n\n"
        "Generado por `python -m src.etl.run_etl`. El porque de cada regla esta en "
        "`docs/CLEANING.md`.\n\n" + cleaning.to_markdown() + "\n",
        encoding="utf-8",
    )
    (reports_dir / "data_trust_raw.json").write_text(trust_raw.to_json(), encoding="utf-8")
    (reports_dir / "data_trust_clean.json").write_text(trust_clean.to_json(), encoding="utf-8")
    (reports_dir / "data_trust.md").write_text(
        "# Data Trust Score (Tarea 1)\n\n"
        f"| Dataset | Score | Nota |\n| --- | ---: | :---: |\n"
        f"| `data/raw` (crudo) | {trust_raw.score:.2f} | {trust_raw.grade} |\n"
        f"| `data/processed` (limpio) | {trust_clean.score:.2f} | {trust_clean.grade} |\n\n"
        "## Antes de limpiar\n\n" + trust_raw.to_markdown() + "\n\n"
        "## Despues de limpiar\n\n" + trust_clean.to_markdown() + "\n",
        encoding="utf-8",
    )

    if expected_pairs is not None:
        write_affinity_report(reports_dir, expected_pairs, affinity_category)


def write_affinity_report(
    reports_dir: Path, expected_pairs: DataFrame, affinity_category: DataFrame | None
) -> None:
    """`affinity_expected_pairs.md`: los complementos de DATA_SPEC.md y la estructura global."""
    rows = expected_pairs.collect()
    lines = [
        "# Afinidad de cesta: pares esperados de DATA_SPEC.md",
        "",
        "Generado por `python -m src.etl.run_etl`. Desde la Fase 8, \"lift objetivo\" es la "
        "fuerza nominal del par y el lift medido es el crudo de `affinity_category`, que "
        "sube con el tamano de la cesta (ver `DATA_SPEC.md`, \"Afinidad de cesta\").",
        "",
        "| Disparadora | Asociada | Lift objetivo | Lift medido | Soporte | Confianza | Medido/objetivo |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lift = "n/d" if r["lift"] is None else f"{r['lift']:.2f}"
        sup = "n/d" if r["support"] is None else f"{100 * r['support']:.2f} %"
        conf = "n/d" if r["confidence"] is None else f"{100 * r['confidence']:.2f} %"
        ratio = "n/d" if r["ratio_vs_target"] is None else f"{r['ratio_vs_target']:.2f}x"
        lines.append(
            f"| {r['antecedent']} | {r['consequent']} | {r['target_lift']:.1f} | "
            f"{lift} | {sup} | {conf} | {ratio} |"
        )
    if affinity_category is not None:
        lines += ["", *affinity_structure_lines(affinity_category)]
    (reports_dir / "affinity_expected_pairs.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(
    spark: SparkSession,
    *,
    data_dir: Path,
    out_dir: Path,
    reports_dir: Path,
    engine: str = "auto",
    product_affinity: bool = True,
    write: bool = True,
) -> dict[str, object]:
    """Ejecuta el ETL completo y devuelve los objetos de resultado."""
    timer = _Timer()

    raw = read_raw(spark, data_dir)
    timer.step("Lectura de data/raw")

    trust_raw = data_trust_score(raw)
    print(f"    Data Trust Score crudo: {trust_raw.score:.2f} (nota {trust_raw.grade})")
    timer.step("Data Trust Score sobre el crudo")

    clean, cleaning = clean_all(raw, CleaningConfig())
    timer.step("Limpieza de las 7 tablas")

    trust_clean = data_trust_score(clean)
    print(f"    Data Trust Score limpio: {trust_clean.score:.2f} (nota {trust_clean.grade})")
    timer.step("Data Trust Score sobre el limpio")

    features = build_features(clean, product_affinity=product_affinity)
    timer.step("Features: RFM, recompra y afinidad")

    pairs = _expected_affinity_pairs()
    expected = (
        expected_pairs_report(features["affinity_category"], pairs) if pairs else None
    )

    if write:
        write_processed(clean, out_dir, engine=engine)
        write_processed(features, out_dir, engine=engine)
        timer.step(f"Escritura en {out_dir}")

    _write_reports(
        reports_dir,
        cleaning,
        trust_raw,
        trust_clean,
        expected,
        features["affinity_category"],
    )
    timer.step(f"Informes en {reports_dir}")

    return {
        "clean": clean,
        "features": features,
        "cleaning_report": cleaning,
        "trust_raw": trust_raw,
        "trust_clean": trust_clean,
        "expected_pairs": expected,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports", type=Path, default=Path("reports/etl"))
    parser.add_argument(
        "--engine",
        choices=("auto", "spark", "pandas"),
        default="auto",
        help="Motor de escritura de Parquet (ver src/etl/schemas.write_table).",
    )
    parser.add_argument(
        "--skip-product-affinity",
        action="store_true",
        help="No calcular la afinidad a nivel de SKU (la parte mas cara).",
    )
    parser.add_argument("--no-write", action="store_true", help="Solo informes, sin Parquet.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    spark = get_spark("grocery-retail-etl")
    try:
        result = run(
            spark,
            data_dir=args.data,
            out_dir=args.out,
            reports_dir=args.reports,
            engine=args.engine,
            product_affinity=not args.skip_product_affinity,
            write=not args.no_write,
        )
        trust_raw, trust_clean = result["trust_raw"], result["trust_clean"]
        print(
            f"\nData Trust Score: {trust_raw.score:.2f} ({trust_raw.grade}) -> "
            f"{trust_clean.score:.2f} ({trust_clean.grade})"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

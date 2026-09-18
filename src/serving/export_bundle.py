"""Exporta a parquet las fuentes de candidatos ya ajustadas, para servir sin Spark.

    python -m src.serving.export_bundle

## Por que existe este paso

La Fase 3 construye las cinco fuentes de candidatos sobre DataFrames de Spark. Eso esta
bien para entrenar y evaluar, pero la demo de la Fase 6b necesita recalcular el top-5 cada
vez que alguien anade un producto a la cesta, y levantar una `SparkSession` dentro de
Streamlit costaria ~10 s de arranque y 300 MB de dependencias para hacer, en realidad,
unos cuantos joins sobre tablas pequenas.

Asi que el trabajo se parte en dos:

1. **Aqui (con Spark, una sola vez)**: se ajustan las fuentes exactamente igual que en la
   Fase 3 y se vuelcan a `data/serving/` en parquet.
2. **En `src/serving/recommend.py` (con pandas)**: se leen esas tablas y se sirve la
   inferencia, con el mismo booster LightGBM que entreno la Fase 3.

El modelo de ALS es el unico que no se puede volcar tal cual, porque es un modelo de Spark
MLlib. No hace falta: `candidates_als` solo lo usa para pedir el top-N por cliente, asi que
se precalcula ese top-N y se guarda como tabla. El resultado es identico mientras el
conjunto de clientes no cambie, que es justo el caso de una demo sobre datos fijos.

## Historial del cliente

Desde el punto A1 del diagnostico, las features personales se calculan as-of el dia de
cada cesta. Por eso no se vuelca un historial ya agregado, sino los eventos de **todas**
las cestas identificadas (`customer_lines` y `customer_baskets`): la ruta de pandas los
filtra por fecha en cada query, igual que `src/recommender/history.py`.

## Que ventana se exporta

La de **test** (`cfg.test_start`), no la del ranker. Es la ventana con la que se midieron
el NDCG@5 y el Recall@5 que aparecen en el README, asi que sirviendo con estas mismas
fuentes la demo ensena el sistema que esta medido, y el test de paridad
(`tests/test_serving_parity.py`) puede compararse contra `recommendations_test.parquet`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.etl.repurchase import category_repurchase_days
from src.etl.schemas import read_processed
from src.etl.session import get_spark
from src.recommender import candidates as cand
from src.recommender import features as feat
from src.recommender import splits
from src.recommender.config import REQUIRED_TABLES, RecommenderConfig
from src.recommender.pipeline import build_window_inputs, fit_sources

SERVING_DIR = Path("data/serving")

# Tablas del `SourceBundle` que se vuelcan tal cual.
BUNDLE_TABLES = (
    "popularity",
    "category_popularity",
    "affinity_product",
    "affinity_category",
    "category_leaders",
    "known_customers",
)


def customer_events(tables: dict[str, DataFrame]) -> tuple[DataFrame, DataFrame]:
    """Lineas y cabeceras de todas las cestas con cliente, sin corte de fecha.

    El corte lo pone la ruta de serving en cada query (dia estrictamente anterior), asi
    que aqui no se filtra nada: la demo puede servir cualquier dia del periodo.
    """
    baskets = tables["baskets"].filter(F.col("customer_id").isNotNull())
    lines = tables["basket_items"].select("basket_id", "product_id", "quantity").join(
        baskets.select("basket_id", "customer_id", "basket_day"), "basket_id"
    )
    return (
        lines.select("customer_id", "basket_id", "basket_day", "product_id", "quantity"),
        baskets.select("customer_id", "basket_id", "basket_day", "total_amount"),
    )


def _write(df: DataFrame, destination: Path) -> int:
    """Escribe un DataFrame de Spark como un unico parquet legible por pandas.

    Se pasa por pandas a proposito: Spark escribiria un directorio con particiones y un
    `_SUCCESS`, y lo que consume la demo es un fichero suelto.
    """
    pdf = df.toPandas()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(destination, index=False)
    return len(pdf)


def export_als_topn(bundle, cfg: RecommenderConfig) -> DataFrame:
    """Precalcula el top-N del ALS por cliente y lo devuelve como tabla.

    Es lo mismo que hace `candidates_als`, pero para todos los clientes de golpe en vez
    de para los de una tanda de queries.
    """
    return (
        bundle.als_model.recommendForAllUsers(cfg.candidates.n_als)
        .select("customer_id_idx", F.posexplode("recommendations").alias("pos", "rec"))
        .select(
            "customer_id_idx",
            F.col("rec.product_id_idx").alias("product_id_idx"),
            F.col("rec.rating").cast("double").alias("als_score"),
            (F.col("pos") + 1).cast("int").alias("als_rank"),
        )
        .join(bundle.als_customer_index, "customer_id_idx")
        .join(bundle.als_product_index, "product_id_idx")
        .select("customer_id", "product_id", "als_score", "als_rank")
    )


def export_parity_fixture(
    tables: dict[str, DataFrame], bundle, cfg: RecommenderConfig, serving_dir: Path
) -> dict[str, int]:
    """Vuelca la ventana de test tal y como la construyo la Fase 3.

    Sin esto, el test de paridad no podria ser exacto: la sesion aporta al carrito
    productos que nunca llegan al ticket (un `add_to_cart` abandonado) y que aun asi se
    excluyen del pool de candidatos, y las features de sesion no se pueden reconstruir
    desde `predictions/`. Se usa `build_window_inputs`, la misma funcion que usa el
    pipeline, para que la ventana no pueda divergir.
    """
    win = build_window_inputs(
        tables,
        bundle,
        start=cfg.test_start,
        end=None,
        n_queries=cfg.n_test_queries,
        salt="test",
    )
    parity_dir = serving_dir / "parity"
    written = {
        "queries": _write(win.queries, parity_dir / "queries.parquet"),
        "cart": _write(win.cart, parity_dir / "cart.parquet"),
        "session_product": _write(
            win.session_product, parity_dir / "session_product.parquet"
        ),
        "session_query": _write(win.session_query, parity_dir / "session_query.parquet"),
    }
    for name, rows in written.items():
        print(f"  parity/{name:14} {rows:>9,} filas")
    return written


def run(
    spark: SparkSession,
    cfg: RecommenderConfig,
    serving_dir: Path,
    *,
    with_parity: bool = True,
) -> dict[str, int]:
    """Ajusta las fuentes en la ventana de test y las vuelca a `serving_dir`."""
    tables = read_processed(spark, REQUIRED_TABLES, cfg.processed_dir)

    print(f"Ajustando las fuentes con todo el historial anterior a {cfg.test_start}...")
    bundle = fit_sources(tables, cfg.test_start, cfg)

    lines, baskets = customer_events(tables)
    history_baskets = splits.baskets_before(tables["baskets"], cfg.test_start)
    extra = {
        "customer_lines": lines,
        "customer_baskets": baskets,
        "category_repurchase_days": category_repurchase_days(tables["products"]),
        # Ficha del cliente al inicio de la ventana: solo para la interfaz de la demo.
        "customer_stats": feat.customer_profile(
            history_baskets, splits.restrict_items(tables["basket_items"], history_baskets)
        ),
    }

    written: dict[str, int] = {}
    tables_out = {name: getattr(bundle, name) for name in BUNDLE_TABLES} | extra
    for name, frame in tables_out.items():
        written[name] = _write(frame, serving_dir / f"{name}.parquet")
        print(f"  {name:24} {written[name]:>9,} filas")

    written["als_topn"] = _write(
        export_als_topn(bundle, cfg), serving_dir / "als_topn.parquet"
    )
    print(f"  {'als_topn':20} {written['als_topn']:>9,} filas")

    # El catalogo ya indexado: la demo necesita `department_idx` y `category_idx` con
    # exactamente la misma codificacion con la que se entreno el ranker.
    written["products_indexed"] = _write(
        feat.index_products(tables["products"]), serving_dir / "products_indexed.parquet"
    )
    print(f"  {'products_indexed':20} {written['products_indexed']:>9,} filas")

    if with_parity:
        written.update(
            {
                f"parity_{k}": v
                for k, v in export_parity_fixture(tables, bundle, cfg, serving_dir).items()
            }
        )

    metadata = {
        "window_start": str(cfg.test_start),
        "n_als": cfg.candidates.n_als,
        "n_popularity": cfg.candidates.n_popularity,
        "n_pop_categories": cfg.candidates.n_pop_categories,
        "n_pop_products_per_category": cfg.candidates.n_pop_products_per_category,
        "n_affinity_product": cfg.candidates.n_affinity_product,
        "n_affinity_category": cfg.candidates.n_affinity_category,
        "n_personal": cfg.candidates.n_personal,
        "rows": written,
    }
    (serving_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-parity",
        action="store_true",
        help="no exporta el fixture de la ventana de test (tests/test_serving_parity.py)",
    )
    parser.add_argument(
        "--serving-dir",
        type=Path,
        default=SERVING_DIR,
        help="directorio de salida (por defecto data/serving)",
    )
    args = parser.parse_args(argv)

    cfg = RecommenderConfig()
    spark = get_spark("export-serving-bundle")
    try:
        written = run(spark, cfg, args.serving_dir, with_parity=not args.no_parity)
    finally:
        spark.stop()

    total = sum(written.values())
    print(f"\nListo: {len(written)} tablas, {total:,} filas en {args.serving_dir}/")


if __name__ == "__main__":
    main()

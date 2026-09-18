"""Recorta el bundle de serving a los clientes que la demo puede llegar a ensenar.

## Por que existe

El bundle completo son 826 MB en memoria, y el 99 % es historial de clientes que la demo
**nunca ofrece**: el selector propone 20 por escenario (fiel / ocasional / en riesgo /
todos), asi que como mucho se pueden elegir unas decenas de los 18.729. En local eso da
igual; en la app publicada no, porque Streamlit Community Cloud asigna entre 690 MB y
2,7 GB por app y ahi no hay margen para cargar el historial de 18.669 clientes que nadie va
a mirar.

Este modulo escribe en `data/serving/demo/` las mismas tablas, filtradas a esos clientes.
Es lo unico que se versiona de las tablas grandes; el bundle completo se regenera con
`python -m src.serving.export_bundle`. Ver `docs/DESPLIEGUE.md`.

## La propiedad que hay que conservar

**El recorte no puede cambiar lo que la demo ensena.** Quien entra en la app publicada
tiene que ver exactamente el mismo selector, los mismos escenarios y las mismas
recomendaciones que quien la levanta en local con el bundle entero.

Eso obliga a no tocar tres tablas que parecen recortables y no lo son:

- `customer_stats` y `parity/queries.parquet` deciden **quien** entra en el selector: los
  escenarios se definen por cuantiles de la poblacion elegible (`src/demo/customers.py`),
  asi que filtrarlas cambiaria los umbrales y, con ellos, la lista de clientes ofrecidos.
  Es circular: el recorte se calcula a partir de ellas.
- `known_customers` decide si un cliente es recurrente o cold-start.

Las tres son pequenas (menos de 1,2 MB entre las tres), asi que se versionan enteras. Lo
que se recorta son solo las tablas que se consultan **para el cliente ya elegido**:
`customer_lines`, `customer_baskets`, `als_topn` y las lineas de ticket de sus cestas.

## Como lo consume la app

`src/serving/recommend.load_bundle` y `src/demo/baskets.basket_items` prefieren siempre la
tabla completa y solo caen a la recortada si no esta. Asi el entorno de desarrollo usa el
bundle entero --y los tests de paridad, que recorren clientes de todo el dataset, siguen
midiendo lo que median-- mientras que el despliegue, donde solo esta la recortada, sirve
con ella sin configurar nada.

Un cliente que no estuviera en la tabla recortada no rompe nada: `_for_customers` tolera al
que falta y el recomendador lo trata como cold-start. Pero no deberia pasar, y por eso hay
un test que compara la lista exportada con la que ofrece el selector.

    python -m src.serving.export_demo_bundle
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from src.demo import customers
from src.demo.baskets import demo_baskets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVING_DIR = PROJECT_ROOT / "data" / "serving"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# El subdirectorio del bundle donde vive el recorte. Cuelga de `data/serving/` a proposito:
# es el mismo artefacto, con menos filas, no una fuente distinta.
DEMO_SUBDIR = "demo"

# Las tablas del bundle que crecen con el numero de clientes. Son las tres que se llevan
# 810 de los 826 MB, y las tres se consultan por `customer_id`.
PER_CUSTOMER_TABLES: tuple[str, ...] = (
    "customer_lines",
    "customer_baskets",
    "als_topn",
)


def demo_customer_ids(
    stats: pd.DataFrame,
    queries: pd.DataFrame,
    actions: pd.DataFrame,
) -> list[str]:
    """Los `customer_id` que el selector de la demo puede llegar a ofrecer.

    Es la union de los cuatro escenarios, calculada con las **mismas** funciones que usa la
    app (`customers.build_pool` y `customers.pick`) y no con una regla paralela: si manana
    cambia el criterio del selector, el recorte cambia con el en vez de quedarse sirviendo
    a los clientes de antes.

    Args:
        stats: `customer_stats` del bundle, entero.
        queries: Las cestas de test, enteras (`parity/queries.parquet`).
        actions: `predictions/nba_actions.parquet`, para el `p_churn` de "En riesgo".
    """
    pool = customers.build_pool(
        stats,
        demo_baskets(queries).groupby("customer_id")["basket_id"].nunique(),
        actions,
    )
    ofrecidos = {
        customer_id
        for escenario in customers.NOMBRES
        for customer_id in customers.pick(pool, escenario)
    }
    return sorted(ofrecidos)


def demo_basket_ids(queries: pd.DataFrame, customer_ids: list[str]) -> list[str]:
    """Las cestas de test que esos clientes pueden cargar en la demo.

    Solo las que pasan el filtro de `demo_baskets`: el selector de cestas no ofrece las
    demas, asi que sus lineas no hacen falta para nada.
    """
    ofrecidas = demo_baskets(queries)
    mias = ofrecidas[ofrecidas["customer_id"].isin(customer_ids)]
    return sorted(mias["basket_id"].unique().tolist())


def export(
    serving_dir: Path = SERVING_DIR,
    processed_dir: Path = PROCESSED_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> dict[str, object]:
    """Escribe el recorte en `<serving_dir>/demo/` y devuelve su manifiesto."""
    demo_dir = serving_dir / DEMO_SUBDIR
    demo_dir.mkdir(parents=True, exist_ok=True)

    stats = pd.read_parquet(serving_dir / "customer_stats.parquet")
    queries = pd.read_parquet(serving_dir / "parity" / "queries.parquet")

    actions_path = predictions_dir / "nba_actions.parquet"
    if not actions_path.is_file():
        raise FileNotFoundError(
            f"No encuentro {actions_path}. Genera las acciones del NBA con "
            "`python -m src.pipeline nba`: sin `p_churn` el escenario 'En riesgo' no "
            "existe y el recorte dejaria fuera a los clientes mas interesantes."
        )
    actions = pd.read_parquet(actions_path)

    customer_ids = demo_customer_ids(stats, queries, actions)
    basket_ids = demo_basket_ids(queries, customer_ids)

    manifest: dict[str, object] = {
        "generado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_customers": len(customer_ids),
        "n_baskets": len(basket_ids),
        "customer_ids": customer_ids,
        "tablas": {},
    }

    for name in PER_CUSTOMER_TABLES:
        full = pd.read_parquet(serving_dir / f"{name}.parquet")
        slim = full[full["customer_id"].isin(customer_ids)]
        destination = demo_dir / f"{name}.parquet"
        slim.to_parquet(destination, index=False)
        manifest["tablas"][name] = {  # type: ignore[index]
            "filas_completa": len(full),
            "filas_recorte": len(slim),
            "bytes": destination.stat().st_size,
        }
        print(f"  {name:<20} {len(full):>10,} -> {len(slim):>8,} filas")

    # Las lineas de ticket no vienen del bundle sino de la Fase 2, y se filtran por cesta:
    # la demo solo pinta las de la compra que se ha cargado.
    items = pd.read_parquet(
        processed_dir / "basket_items.parquet",
        filters=[("basket_id", "in", basket_ids)],
    )
    destination = demo_dir / "basket_items.parquet"
    items.to_parquet(destination, index=False)
    manifest["tablas"]["basket_items"] = {  # type: ignore[index]
        "filas_recorte": len(items),
        "bytes": destination.stat().st_size,
    }
    print(f"  {'basket_items':<20} {'':>10} -> {len(items):>8,} filas")

    (demo_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total = sum(
        int(t["bytes"]) for t in manifest["tablas"].values()  # type: ignore[union-attr]
    )
    # `relative_to` revienta si el destino esta fuera del repo, que es lo que pasa en los
    # tests (tmp_path). El mensaje no merece tirar la exportacion entera.
    try:
        donde = demo_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        donde = demo_dir
    print(
        f"\n{len(customer_ids)} clientes y {len(basket_ids)} cestas en "
        f"{donde} ({total / 1e6:.1f} MB)"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--serving-dir", type=Path, default=SERVING_DIR, help="Bundle completo de entrada"
    )
    args = parser.parse_args(argv)
    export(serving_dir=args.serving_dir)


if __name__ == "__main__":
    main()

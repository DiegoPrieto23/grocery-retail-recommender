"""Exporta la probabilidad real de cada cesta de test, para el oraculo del recomendador (A6).

    python -m data_generation.export_oracle
    python -m data_generation.export_oracle --from-date 2025-11-01 --out data/oracle

El generador sabe, para cada cesta, con que peso sale cada categoria antes de aplicar la
mision (afinidad x estacionalidad x ciclo de reposicion, con los gates de hogar), con que
probabilidad tocaba cada mision de compra en esa visita, su factor de tamano y que
referencia es la mas probable dentro de cada categoria. Este script vuelve a ejecutar el
generador con un `OracleRecorder` y vuelca esa informacion para las cestas con fecha
`>= from_date` (la ventana de test del recomendador).

## Por que no va en `data/raw`

Es informacion que ningun sistema real tendria. Si viviera junto al dataset, cualquier
feature podria acabar leyendola sin querer. Va a `data/oracle/` (fuera de git, como
`data/processed/`) y solo la lee `src/recommender/oracle.py`.

## Como se garantiza que corresponde al dataset en uso

El generador se ejecuta completo en un directorio temporal y los sha256 de las 7 tablas
se comparan con `data/raw/manifest.json`. Si no coinciden (otra semilla, otra escala u
otra version del generador), el script falla sin escribir nada: un oraculo de otro
dataset daria un techo falso. De paso, esa comparacion demuestra que el registrador no
cambia ni un byte del dataset.

## Que se escribe

- `category_weights.parquet`: una fila por `(basket_id, category)` con peso > 0:
  `weight` (sin normalizar), `best_product_id` y `best_product_prob` (probabilidad de esa
  referencia si la categoria sale).
- `basket_missions.parquet`: una fila por cesta: `size_factor` y la probabilidad de cada
  mision (una columna `mission_<nombre>` por mision).
- `manifest.json`: semilla, escala, fecha de corte, orden de categorias, el resto del
  proceso generativo (matriz de complementos y sustitutos **aplicada**, perfiles, tamano
  de cada mision y tope de categorias) y los hashes contra los que se valido.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_generation import catalog as cat
from data_generation.generate_dataset import SEED, GeneratorConfig, OracleRecorder, generate

# Inicio de la ventana de test del recomendador (`src/recommender/config.TEST_START`). Se
# repite aqui en vez de importarlo para que el generador no dependa de `src/`; el
# verificador comprueba que el oraculo cubre todas las queries de test.
DEFAULT_FROM_DATE = dt.date(2025, 11, 1)

WEIGHTS_FILENAME = "category_weights.parquet"
MISSIONS_FILENAME = "basket_missions.parquet"
MANIFEST_FILENAME = "manifest.json"


def _day_index(value: dt.date) -> int:
    return (pd.Timestamp(value) - pd.Timestamp(cat.PERIOD_START)).days


def oracle_frame(recorder: OracleRecorder, basket_ids: np.ndarray, product_ids: np.ndarray) -> pd.DataFrame:
    """Pasa lo registrado a formato largo, una fila por cesta y categoria con peso > 0."""
    columns = ["basket_id", "category", "weight", "best_product_id", "best_product_prob"]
    if not recorder.basket_index:
        return pd.DataFrame(columns=columns)
    names = np.array([c.name for c in cat.CATEGORIES], dtype=object)
    weights = np.vstack(recorder.weights)
    best = np.vstack(recorder.best_product)
    prob = np.vstack(recorder.best_prob)
    rows, cols = np.nonzero(weights > 0)
    b_idx = np.asarray(recorder.basket_index)[rows]
    return pd.DataFrame(
        {
            "basket_id": basket_ids[b_idx],
            "category": names[cols],
            "weight": weights[rows, cols],
            "best_product_id": product_ids[best[rows, cols]],
            "best_product_prob": prob[rows, cols].astype(np.float64),
        },
        columns=columns,
    )


def missions_frame(recorder: OracleRecorder, basket_ids: np.ndarray) -> pd.DataFrame:
    """Una fila por cesta registrada: factor de tamano y probabilidad de cada mision."""
    columns = ["basket_id", "size_factor", *(f"mission_{m}" for m in cat.MISSION_NAMES)]
    if not recorder.basket_index:
        return pd.DataFrame(columns=columns)
    probs = np.vstack(recorder.mission_probs)
    frame = pd.DataFrame(
        {
            "basket_id": basket_ids[np.asarray(recorder.basket_index)],
            "size_factor": np.asarray(recorder.size_factor, dtype=np.float64),
        }
    )
    for i, name in enumerate(cat.MISSION_NAMES):
        frame[f"mission_{name}"] = probs[:, i]
    return frame[columns]


def generative_process() -> dict:
    """Lo que el oraculo necesita del proceso, ademas de lo registrado por cesta."""
    return {
        "missions": [
            {
                "name": m.name,
                "size_mean": m.size_mean,
                "size_dispersion": m.size_dispersion,
            }
            for m in cat.MISSIONS
        ],
        "mission_profiles": cat.mission_profiles(),
        # Fila = categoria que entra; columna = categoria afectada. Complementos con el
        # lift **aplicado** (no el nominal de DATA_SPEC.md) y sustitutos con su factor.
        "interaction": cat.interaction_matrix(),
        "max_categories_per_basket": cat.MAX_CATEGORIES_PER_BASKET,
        "second_reference_prob": [cat.second_reference_prob(c) for c in cat.CATEGORIES],
    }


def export(
    *,
    raw_dir: Path,
    out_dir: Path,
    from_date: dt.date,
    seed: int = SEED,
    scale: float = 1.0,
    check_manifest: bool = True,
) -> dict:
    """Regenera el dataset con el registrador y escribe el oraculo en `out_dir`.

    Raises:
        RuntimeError: Si el dataset regenerado no coincide con `raw_dir/manifest.json`.
    """
    recorder = OracleRecorder(from_day=_day_index(from_date))
    with tempfile.TemporaryDirectory(prefix="oracle_gen_") as tmp:
        cfg = GeneratorConfig(out_dir=Path(tmp), seed=seed, scale=scale)
        tables = generate(cfg, oracle=recorder)
        regenerated = json.loads((Path(tmp) / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    if check_manifest:
        expected = json.loads((raw_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        if regenerated["sha256"] != expected["sha256"]:
            differ = sorted(
                k for k in expected["sha256"] if expected["sha256"][k] != regenerated["sha256"].get(k)
            )
            raise RuntimeError(
                f"El dataset regenerado no coincide con {raw_dir / MANIFEST_FILENAME} "
                f"(tablas distintas: {differ}). El oraculo no corresponderia al dato en uso."
            )

    frame = oracle_frame(
        recorder,
        tables["baskets"]["basket_id"].to_numpy(),
        tables["products"]["product_id"].to_numpy(),
    )

    missions = missions_frame(recorder, tables["baskets"]["basket_id"].to_numpy())

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / WEIGHTS_FILENAME, index=False)
    missions.to_parquet(out_dir / MISSIONS_FILENAME, index=False)
    manifest = {
        "seed": seed,
        "scale": scale,
        "from_date": str(from_date),
        "n_baskets": int(frame["basket_id"].nunique()),
        "n_rows": int(len(frame)),
        "categories": [c.name for c in cat.CATEGORIES],
        **generative_process(),
        "dataset_sha256": regenerated["sha256"],
        "validated_against": str((raw_dir / MANIFEST_FILENAME).as_posix()) if check_manifest else None,
        "weights_sha256": hashlib.sha256((out_dir / WEIGHTS_FILENAME).read_bytes()).hexdigest(),
        "missions_sha256": hashlib.sha256((out_dir / MISSIONS_FILENAME).read_bytes()).hexdigest(),
    }
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/oracle"))
    parser.add_argument("--from-date", type=dt.date.fromisoformat, default=DEFAULT_FROM_DATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    start = time.perf_counter()
    manifest = export(
        raw_dir=args.raw,
        out_dir=args.out,
        from_date=args.from_date,
        seed=args.seed,
        scale=args.scale,
    )
    print(
        f"Oraculo escrito en {args.out.resolve()}: {manifest['n_baskets']:,} cestas desde "
        f"{manifest['from_date']} ({time.perf_counter() - start:.0f}s). "
        f"Dataset validado contra {manifest['validated_against']}."
    )


if __name__ == "__main__":
    main()

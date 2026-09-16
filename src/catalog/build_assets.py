"""Fase 6a: construye `assets/` — grupos visuales, fotos de Pexels y CSV de mapeo.

    python -m src.catalog.build_assets              # solo descarga lo que falte
    python -m src.catalog.build_assets --force      # vuelve a descargar todo
    python -m src.catalog.build_assets --offline    # regenera los CSV sin tocar la red

Se ejecuta **una sola vez**: el resultado se comitea y se trata como un fixture. Los
resultados de busqueda de Pexels no son reproducibles por semilla (el catalogo de fotos
cambia con el tiempo), asi que volver a lanzarlo con `--force` puede dar otras fotos. Por
eso el modo por defecto respeta lo ya descargado y no vuelve a llamar a la API.

Aqui se usa pandas y no PySpark a proposito: son 496 filas y un puñado de llamadas HTTP,
no un ETL a escala.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.catalog.image_hash import dhash, is_duplicate
from src.catalog.pexels import (
    fetch_photo_bytes,
    load_api_key,
    save_photo,
    search_candidates,
)
from src.catalog.visual_groups import (
    MERGES,
    MIN_GROUP_SIZE,
    build_group_table,
    build_product_catalog,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_PARQUET = PROJECT_ROOT / "data" / "processed" / "products.parquet"
PRODUCTS_CSV = PROJECT_ROOT / "data" / "raw" / "products.csv"
ASSETS_DIR = PROJECT_ROOT / "assets"

GROUPS_CSV = ASSETS_DIR / "visual_groups.csv"
CATALOG_CSV = ASSETS_DIR / "product_catalog.csv"
CREDITS_CSV = ASSETS_DIR / "image_credits.csv"


def load_products() -> pd.DataFrame:
    """Lee `products` de `data/processed/`; si la Fase 2 no se ha ejecutado, del crudo."""
    if PRODUCTS_PARQUET.exists():
        return pd.read_parquet(PRODUCTS_PARQUET)
    if PRODUCTS_CSV.exists():
        return pd.read_csv(PRODUCTS_CSV)
    raise FileNotFoundError(
        "No encuentro `products`. Lanza antes `python data_generation/generate_dataset.py` "
        "y `python -m src.etl.run_etl`."
    )


def fetch_images(
    groups: pd.DataFrame, *, force: bool, offline: bool
) -> tuple[dict[str, str], list[dict], list[str]]:
    """Descarga una foto por grupo. Devuelve (rutas, creditos, grupos sin foto).

    Un grupo ya descargado se salta sin llamar a la API salvo `--force`: eso hace el
    script idempotente y barato de relanzar cuando solo han fallado unos pocos.
    """
    image_paths: dict[str, str] = {}
    credits: list[dict] = []
    missing: list[str] = []
    api_key: str | None = None
    # Fotos ya asignadas a otro grupo (en esta tanda o en una anterior): no se repiten.
    used_photo_ids: set[int] = set()
    if CREDITS_CSV.exists():
        previous = pd.read_csv(CREDITS_CSV)
        used_photo_ids = set(previous["pexels_photo_id"].dropna().astype(int))

    # Las huellas se calculan de lo que hay en `assets/`, no del CSV: asi una foto que
    # se re-descarga tambien se compara con las que ya estaban de tandas anteriores.
    fingerprints: dict[str, int] = {}
    if not force:
        for existing in sorted(ASSETS_DIR.glob("*.jpg")):
            fingerprints[existing.stem] = dhash(existing.read_bytes())

    for row in groups.itertuples(index=False):
        group, term = row.visual_group, row.search_term
        destination = ASSETS_DIR / f"{group}.jpg"
        relative = f"assets/{group}.jpg"

        if destination.exists() and not force:
            image_paths[group] = relative
            print(f"  = {group}: ya estaba en assets/, no se vuelve a pedir")
            continue

        if offline:
            missing.append(group)
            print(f"  - {group}: sin foto (modo --offline)")
            continue

        if api_key is None:
            api_key = load_api_key()

        print(f"  > {group}: buscando {term!r}")
        chosen = None
        for candidate in search_candidates(term, api_key, exclude_ids=used_photo_ids):
            content = fetch_photo_bytes(candidate)
            if content is None:
                continue
            fingerprint = dhash(content)
            twin = is_duplicate(fingerprint, fingerprints)
            if twin is not None:
                print(f"      descartada: misma imagen que {twin}")
                continue
            save_photo(content, destination)
            fingerprints[group] = fingerprint
            chosen = candidate
            break

        if chosen is None:
            missing.append(group)
            print(f"  ! {group}: ninguna foto valida, queda sin imagen")
            continue

        used_photo_ids.add(chosen.photo_id)
        image_paths[group] = relative
        credits.append(
            {
                "visual_group": group,
                "image_path": relative,
                "pexels_photo_id": chosen.photo_id,
                "photographer": chosen.photographer,
                "photographer_url": chosen.photographer_url,
                "pexels_url": chosen.page_url,
                "alt": chosen.alt,
                "query_used": chosen.query_used,
                "score": chosen.score,
            }
        )
        size_kb = destination.stat().st_size / 1024
        print(f"  + {group}: {relative} ({size_kb:.0f} KB) — {chosen.alt[:60]!r}")

    return image_paths, credits, missing


def merge_credits(new_credits: list[dict]) -> pd.DataFrame:
    """Funde los creditos nuevos con los ya guardados, para no perder los de una tanda previa."""
    frames = []
    if CREDITS_CSV.exists():
        previous = pd.read_csv(CREDITS_CSV)
        if not previous.empty:
            frames.append(previous)
    if new_credits:
        frames.append(pd.DataFrame(new_credits))
    if not frames:
        return pd.DataFrame(
            columns=[
                "visual_group",
                "image_path",
                "pexels_photo_id",
                "photographer",
                "photographer_url",
                "pexels_url",
                "alt",
                "query_used",
                "score",
            ]
        )
    merged = pd.concat(frames, ignore_index=True)
    # La ultima descarga de cada grupo manda.
    merged = merged.drop_duplicates("visual_group", keep="last")
    return merged.sort_values("visual_group", ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="vuelve a descargar incluso los grupos que ya tienen foto en assets/",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="regenera los CSV a partir de lo que ya hay en assets/, sin llamar a Pexels",
    )
    args = parser.parse_args()

    products = load_products()
    groups = build_group_table(products)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"{len(products)} productos, {products['category'].nunique()} categorias "
        f"-> {len(groups)} grupos visuales ({len(MERGES)} fusiones)."
    )
    small = groups.loc[groups["n_products"] < MIN_GROUP_SIZE, "visual_group"].tolist()
    print(f"Grupos con menos de {MIN_GROUP_SIZE} productos: {small}")

    groups[["visual_group", "search_term"]].to_csv(
        GROUPS_CSV, index=False, encoding="utf-8"
    )
    print(f"\nEscrito {GROUPS_CSV.relative_to(PROJECT_ROOT)}")

    print("\nImagenes:")
    image_paths, credits, missing = fetch_images(
        groups, force=args.force, offline=args.offline
    )

    catalog = build_product_catalog(products, image_paths)
    catalog.to_csv(CATALOG_CSV, index=False, encoding="utf-8")
    merge_credits(credits).to_csv(CREDITS_CSV, index=False, encoding="utf-8")

    without_image = int((catalog["image_path"] == "").sum())
    print(f"\nEscrito {CATALOG_CSV.relative_to(PROJECT_ROOT)}: {len(catalog)} productos")
    print(f"Escrito {CREDITS_CSV.relative_to(PROJECT_ROOT)}")
    print(
        f"Grupos con foto: {len(image_paths)}/{len(groups)} | "
        f"productos sin foto: {without_image}"
    )
    if missing:
        print(f"Sin foto valida: {missing}")


if __name__ == "__main__":
    main()

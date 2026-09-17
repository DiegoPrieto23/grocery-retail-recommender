"""Paridad entre la ruta de serving en pandas (Fase 6b) y la de Spark (Fase 3).

`src/serving/recommend.py` es una reimplementacion: replica en pandas las cinco fuentes de
candidatos y las 60 features que la Fase 3 construye con Spark. Una reimplementacion sin
un test que la ate no vale nada — la demo ensenaria un top-5 parecido pero no el del
sistema medido, y el NDCG@5 del README dejaria de describirla.

Este test pasa las **queries de test reales** por la ruta de pandas y compara el resultado
contra `predictions/recommendations_test.parquet`, que produjo la ruta de Spark. No
necesita Spark: el fixture de la ventana lo dejo `python -m src.serving.export_bundle`.

Se salta entero si no estan el bundle ni las predicciones, para que la CI de un repo
recien clonado (donde `data/` y `predictions/` no se versionan) siga en verde.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVING_DIR = PROJECT_ROOT / "data" / "serving"
PARITY_DIR = SERVING_DIR / "parity"
PREDICTIONS = PROJECT_ROOT / "predictions" / "recommendations_test.parquet"

# Cuantas queries se comparan. Las 18.000 tardan varios minutos y no dicen nada que no
# diga una muestra grande; el muestreo es determinista (las primeras por `basket_id`).
N_QUERIES = 2_000

needs_bundle = pytest.mark.skipif(
    not (SERVING_DIR / "metadata.json").is_file()
    or not (PARITY_DIR / "queries.parquet").is_file()
    or not PREDICTIONS.is_file(),
    reason=(
        "falta el bundle de serving o las predicciones de la Fase 3; "
        "genera con `python -m src.serving.export_bundle`"
    ),
)


@pytest.fixture(scope="module")
def parity() -> dict:
    """Corre las dos rutas sobre la misma muestra de queries y devuelve los dos top-5."""
    from src.serving.recommend import load_bundle, rank_queries

    bundle = load_bundle(SERVING_DIR)
    queries = pd.read_parquet(PARITY_DIR / "queries.parquet")
    cart = pd.read_parquet(PARITY_DIR / "cart.parquet")
    session_product = pd.read_parquet(PARITY_DIR / "session_product.parquet")
    session_query = pd.read_parquet(PARITY_DIR / "session_query.parquet")

    sample_ids = sorted(queries["basket_id"].unique())[:N_QUERIES]
    queries = queries.loc[queries["basket_id"].isin(sample_ids)].copy()
    queries["basket_day"] = pd.to_datetime(queries["basket_day"])
    cart = cart.loc[cart["basket_id"].isin(sample_ids)]

    context = pd.read_parquet(PROJECT_ROOT / "predictions" / "recommender_test_context.parquet")
    prefix = context.loc[
        (context["role"] == "prefix") & (context["basket_id"].isin(sample_ids)),
        ["basket_id", "product_id"],
    ]

    got = rank_queries(
        queries,
        prefix,
        bundle,
        cart=cart,
        session_product=session_product.loc[session_product["basket_id"].isin(sample_ids)],
        session_query=session_query.loc[session_query["basket_id"].isin(sample_ids)],
        top_k=5,
    )
    expected = pd.read_parquet(PREDICTIONS)
    expected = expected.loc[expected["basket_id"].isin(sample_ids)]
    return {"got": got, "expected": expected, "sample_ids": sample_ids}


@needs_bundle
def test_se_recomienda_para_las_mismas_cestas(parity: dict) -> None:
    assert set(parity["got"]["basket_id"]) == set(parity["expected"]["basket_id"])


@needs_bundle
def test_el_top5_coincide_producto_a_producto(parity: dict) -> None:
    """El conjunto de 5 productos recomendados es el mismo en las dos rutas."""
    got = parity["got"].groupby("basket_id")["product_id"].apply(frozenset)
    expected = parity["expected"].groupby("basket_id")["product_id"].apply(frozenset)
    joined = pd.concat([got.rename("got"), expected.rename("expected")], axis=1)
    iguales = (joined["got"] == joined["expected"]).mean()
    distintas = joined.loc[joined["got"] != joined["expected"]]
    assert iguales == 1.0, (
        f"solo el {iguales:.2%} de las cestas tiene el mismo top-5. "
        f"Ejemplos: {distintas.head(3).to_dict(orient='index')}"
    )


@needs_bundle
def test_el_orden_dentro_del_top5_coincide(parity: dict) -> None:
    """Mismo producto en el mismo puesto: el ranking, no solo el conjunto."""
    got = parity["got"].set_index(["basket_id", "rank"])["product_id"]
    expected = parity["expected"].set_index(["basket_id", "rank"])["product_id"]
    joined = pd.concat([got.rename("got"), expected.rename("expected")], axis=1).dropna()
    iguales = (joined["got"] == joined["expected"]).mean()
    assert iguales == 1.0, f"solo coincide el puesto en el {iguales:.2%} de las filas"


@needs_bundle
def test_los_scores_del_booster_coinciden(parity: dict) -> None:
    """Las features llegan iguales: si una difiere, el score se mueve."""
    got = parity["got"].set_index(["basket_id", "product_id"])["score"]
    expected = parity["expected"].set_index(["basket_id", "product_id"])["score"]
    joined = pd.concat([got.rename("got"), expected.rename("expected")], axis=1).dropna()
    diferencia = (joined["got"] - joined["expected"]).abs().max()
    assert diferencia < 1e-6, f"el score se desvia hasta {diferencia:.3g}"


@needs_bundle
def test_las_fuentes_atribuidas_coinciden(parity: dict) -> None:
    """`src_*` y `n_sources` son lo que la demo ensena como motivo de cada recomendacion."""
    columnas = ["src_pop", "src_aff", "src_cataff", "src_hist", "src_als", "n_sources"]
    got = parity["got"].set_index(["basket_id", "product_id"])[columnas]
    expected = parity["expected"].set_index(["basket_id", "product_id"])[columnas]
    alineados = got.join(expected, lsuffix="_got", rsuffix="_exp", how="inner")
    for columna in columnas:
        iguales = (alineados[f"{columna}_got"] == alineados[f"{columna}_exp"]).mean()
        assert iguales == 1.0, f"{columna} coincide solo en el {iguales:.2%}"


@needs_bundle
def test_no_se_recomienda_lo_que_ya_esta_en_el_carrito(parity: dict) -> None:
    cart = pd.read_parquet(PARITY_DIR / "cart.parquet")
    cart = cart.loc[cart["basket_id"].isin(parity["sample_ids"])]
    pares_cart = set(map(tuple, cart[["basket_id", "product_id"]].to_numpy()))
    pares_got = set(map(tuple, parity["got"][["basket_id", "product_id"]].to_numpy()))
    assert not (pares_cart & pares_got)

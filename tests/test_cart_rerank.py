"""Tests del punto A2: features de carrito y re-ranking final del top-k.

* **Re-ranking**: la cuota por categoria y la exclusion del carrito relegan, no borran, y
  sin reglas el orden es exactamente el del ranker.
* **Huecos regalados**: la metrica cuenta a mano lo que debe contar.
* **Features de carrito**: Spark (`features.cart_features`) y la replica de pandas de la
  demo (`serving.recommend._add_cart`) dan lo mismo sobre el mismo carrito.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.recommender import evaluate as ev
from src.recommender import rerank as rr
from src.recommender.config import RecommenderConfig, RerankConfig
from src.recommender.schema import CART_FEATURES, FEATURE_COLUMNS
from src.serving import recommend as srv
from tests.helpers import spark_df


def _scored(rows: list[tuple[str, str, float, int, int]]) -> pd.DataFrame:
    """Filas `(basket_id, product_id, score, category_idx, cat_in_cart)`."""
    frame = pd.DataFrame(
        rows, columns=["basket_id", "product_id", "score", "category_idx", "cat_in_cart"]
    )
    return frame.assign(label=0, cat_in_cart=frame["cat_in_cart"].astype(float))


# Una query con dos leches arriba (cat 0), una categoria del carrito (cat 1) y relleno.
QUERY = _scored(
    [
        ("B1", "LECHE_A", 0.9, 0, 0),
        ("B1", "LECHE_B", 0.8, 0, 0),
        ("B1", "PAN", 0.7, 1, 1),
        ("B1", "PASTA", 0.6, 2, 0),
        ("B1", "ARROZ", 0.5, 3, 0),
        ("B1", "HUEVOS", 0.4, 4, 0),
        ("B1", "ACEITE", 0.3, 5, 0),
    ]
)


def _ids(top: pd.DataFrame) -> list[str]:
    return top.sort_values("rank")["product_id"].tolist()


def test_sin_reglas_el_top_k_es_el_orden_del_ranker() -> None:
    for rules in (None, RerankConfig.off()):
        top = rr.top_k(QUERY, k=5, rerank=rules)
        assert _ids(top) == ["LECHE_A", "LECHE_B", "PAN", "PASTA", "ARROZ"]


def test_la_cuota_deja_una_referencia_por_categoria() -> None:
    rules = RerankConfig(max_per_category=1, exclude_cart_categories=False)
    top = rr.top_k(QUERY, k=5, rerank=rules)
    assert _ids(top) == ["LECHE_A", "PAN", "PASTA", "ARROZ", "HUEVOS"]
    assert top["category_idx"].is_unique


def test_la_exclusion_relega_las_categorias_del_carrito() -> None:
    top = rr.top_k(QUERY, k=5, rerank=RerankConfig())
    assert _ids(top) == ["LECHE_A", "PASTA", "ARROZ", "HUEVOS", "ACEITE"]
    assert (top["cat_in_cart"] == 0).all()


def test_los_relegados_rellenan_si_el_pool_no_da_para_k() -> None:
    """Las reglas no dejan huecos vacios: lo relegado vuelve al final, en orden de score."""
    corto = QUERY.loc[QUERY["product_id"].isin(["LECHE_A", "LECHE_B", "PAN", "PASTA"])]
    top = rr.top_k(corto, k=5, rerank=RerankConfig())
    assert _ids(top) == ["LECHE_A", "PASTA", "LECHE_B", "PAN"]


def test_el_re_ranking_es_determinista_y_por_query() -> None:
    otra = QUERY.assign(basket_id="B2")
    both = pd.concat([QUERY, otra], ignore_index=True)
    first = rr.top_k(both, k=5, rerank=RerankConfig())
    second = rr.top_k(both.sample(frac=1, random_state=3), k=5, rerank=RerankConfig())
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("basket_id").size().tolist() == [5, 5]


def test_evaluate_y_serving_usan_el_mismo_re_ranking() -> None:
    """`top_k_predictions` delega en `rerank.top_k`, igual que `serving.rank_queries`."""
    rules = RerankConfig()
    pd.testing.assert_frame_equal(
        ev.top_k_predictions(QUERY, k=5, rerank=rules), rr.top_k(QUERY, k=5, rerank=rules)
    )
    assert srv.ServingBundle.__dataclass_fields__["rerank"].default == RecommenderConfig().rerank


def test_las_features_de_carrito_forman_parte_del_contrato() -> None:
    assert set(CART_FEATURES) <= set(FEATURE_COLUMNS)
    assert set(CART_FEATURES) <= set(srv.ZERO_FILLED)


# --------------------------------------------------------------------------------------
# Huecos regalados
# --------------------------------------------------------------------------------------
def test_huecos_regalados_contra_el_calculo_a_mano() -> None:
    """B1 lleva pan en el carrito y recibe [leche, leche, pan, pasta, arroz].

    - en carrito: el pan (1 de 5)
    - repetida: la segunda leche (1 de 5)
    B2 no tiene carrito y recibe cinco categorias distintas: ningun regalo.
    """
    catalog = pd.DataFrame(
        {
            "product_id": ["LECHE_A", "LECHE_B", "PAN", "PAN_B", "PASTA", "ARROZ", "HUEVOS"],
            "category": ["leche", "leche", "pan", "pan", "pasta", "arroz", "huevos"],
        }
    )
    top = pd.DataFrame(
        {
            "basket_id": ["B1"] * 5 + ["B2"] * 5,
            "product_id": ["LECHE_A", "LECHE_B", "PAN", "PASTA", "ARROZ"]
            + ["LECHE_A", "PAN", "PASTA", "ARROZ", "HUEVOS"],
            "rank": list(range(1, 6)) * 2,
        }
    )
    prefix = pd.DataFrame({"basket_id": ["B1"], "product_id": ["PAN_B"]})
    queries = pd.DataFrame({"basket_id": ["B1", "B2"], "profile": [4, 3]})

    out = ev.wasted_slot_metrics(top, prefix, catalog, queries, k=5).set_index("grupo")
    assert out.loc["total", "huecos_en_carrito@5"] == pytest.approx(1 / 10)
    assert out.loc["total", "huecos_repetidos@5"] == pytest.approx(1 / 10)
    assert out.loc["total", "huecos_regalados@5"] == pytest.approx(2 / 10)
    assert out.loc["total", "listas_con_regalo@5"] == pytest.approx(0.5)
    assert out.loc[ev.WITH_CART_GROUP, "huecos_regalados@5"] == pytest.approx(2 / 5)
    assert out.loc[ev.WITH_CART_GROUP, "listas_con_regalo@5"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Features de carrito: Spark y pandas
# --------------------------------------------------------------------------------------
PRODUCTS = pd.DataFrame(
    {
        "product_id": ["P1", "P2", "P3", "P4", "P5"],
        "category": ["Leche", "Yogur", "Pasta", "Arroz", "Leche"],
        "department_idx": [0, 0, 1, 1, 0],
    }
)
CART = pd.DataFrame(
    {
        # B1: dos lacteos y una pasta. B2: carrito vacio. P1 repetido no cuenta dos veces.
        "basket_id": ["B1", "B1", "B1", "B1"],
        "product_id": ["P1", "P2", "P3", "P1"],
    }
)
CANDIDATES = pd.DataFrame(
    {
        "basket_id": ["B1", "B1", "B1", "B2"],
        "product_id": ["P5", "P4", "P2", "P5"],
    }
)
# (cat_in_cart, dept_n_in_cart, dept_share_in_cart) esperados tras el relleno a 0.
EXPECTED = {
    ("B1", "P5"): (1.0, 2.0, 2 / 3),  # leche en el carrito; 2 de 3 lineas son lacteos
    ("B1", "P4"): (0.0, 1.0, 1 / 3),  # arroz no esta; 1 de 3 lineas es de su departamento
    ("B1", "P2"): (1.0, 2.0, 2 / 3),
    ("B2", "P5"): (0.0, 0.0, 0.0),  # sin carrito
}


def _check(frame: pd.DataFrame) -> None:
    for (basket, product), values in EXPECTED.items():
        row = frame.loc[(frame["basket_id"] == basket) & (frame["product_id"] == product)]
        assert len(row) == 1
        got = tuple(float(row[c].iloc[0]) for c in CART_FEATURES)
        np.testing.assert_allclose(got, values, err_msg=f"{basket}/{product}")


def test_features_de_carrito_en_pandas() -> None:
    joined = CANDIDATES.merge(PRODUCTS, on="product_id")
    out = srv._add_cart(joined, CART, PRODUCTS)
    for column in CART_FEATURES:
        out[column] = out[column].fillna(0.0)
    _check(out)


def test_features_de_carrito_sin_carrito_en_pandas() -> None:
    joined = CANDIDATES.merge(PRODUCTS, on="product_id")
    out = srv._add_cart(joined, CART.iloc[:0], PRODUCTS)
    assert out[list(CART_FEATURES)].isna().all().all()


def test_features_de_carrito_en_spark_igual_que_en_pandas(spark) -> None:
    from pyspark.sql import functions as F

    from src.recommender import features as feat

    products = spark_df(
        spark, PRODUCTS.to_dict("records"), "product_id string, category string, department_idx int"
    )
    cart = spark_df(spark, CART.to_dict("records"), "basket_id string, product_id string")
    candidates = spark_df(
        spark, CANDIDATES.to_dict("records"), "basket_id string, product_id string"
    ).join(products, "product_id")

    per_category, per_department, per_query = feat.cart_features(cart, products)
    out = (
        candidates.join(per_category, ["basket_id", "category"], "left")
        .join(per_department, ["basket_id", "department_idx"], "left")
        .join(per_query, "basket_id", "left")
        .withColumn("dept_share_in_cart", F.col("dept_n_in_cart") / F.col("_cart_size"))
        .fillna(0.0, subset=list(CART_FEATURES))
        .toPandas()
    )
    _check(out)

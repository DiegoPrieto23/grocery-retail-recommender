"""Tests del punto A4: objetivo del ranker alineado con el acierto de categoria.

* **Relevancia graduada**: 2 para el SKU exacto, 1 para otra referencia de una categoria
  del target y 0 para el resto, igual en la matriz de Spark que en la metrica.
* **NDCG@5 graduada** (la metrica principal): cuenta a mano lo que debe contar y nunca
  pasa de 1.
* **Configuracion**: la relevancia elige etiqueta, ganancias y cestas de entrenamiento.
* **Ranking personal de categorias**: puestos y cuota as-of, con la misma formula que el
  baseline `personal_due`. La paridad Spark / pandas de estas columnas la cubre
  `tests/test_asof_features.py`, que compara la tabla de categorias entera.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.recommender import evaluate as ev
from src.recommender import formulas as fx
from src.recommender import ranker as rk
from src.recommender.config import RankerConfig
from src.recommender.schema import (
    CUSTOMER_CATEGORY_RANK_FEATURES,
    FEATURE_COLUMNS,
    RELEVANCE_CATEGORY,
    RELEVANCE_GAIN,
    RELEVANCE_NONE,
    RELEVANCE_SKU,
)
from src.serving import recommend as srv
from tests.helpers import spark_df
from tests.test_asof_features import QUERIES, _bundle, _row

CATALOG = pd.DataFrame(
    {
        "product_id": ["L1", "L2", "P1", "P2", "H1"],
        "category": ["leche", "leche", "pan", "pan", "huevos"],
    }
)
DISCOUNT = 1.0 / np.log2(np.arange(2, 7))


# --------------------------------------------------------------------------------------
# Configuracion y etiquetas
# --------------------------------------------------------------------------------------
def test_la_relevancia_elige_etiqueta_y_ganancias() -> None:
    graded = RankerConfig()
    assert (graded.relevance, graded.label_column, graded.gains) == (
        "graded",
        "relevance",
        [0.0, 1.0, 3.0],
    )
    sku = RankerConfig(relevance="sku")
    assert (sku.label_column, sku.gains) == ("label", [0.0, 1.0])
    with pytest.raises(ValueError):
        RankerConfig(relevance="categoria")
    # La ganancia de cada nivel es 2**relevancia - 1 y el SKU exacto vale mas.
    assert RELEVANCE_GAIN == tuple(2.0**r - 1 for r in range(3))
    assert RELEVANCE_NONE < RELEVANCE_CATEGORY < RELEVANCE_SKU


def test_el_ranking_personal_entra_en_las_features() -> None:
    assert set(CUSTOMER_CATEGORY_RANK_FEATURES) <= set(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 60


def test_con_relevancia_graduada_basta_acertar_la_categoria_para_entrenar() -> None:
    pdf = pd.DataFrame(
        {
            "basket_id": ["A", "A", "B", "B", "C"],
            "label": [0, 1, 0, 0, 0],
            "relevance": [0, 2, 1, 0, 0],
        }
    )
    assert set(rk.drop_groups_without_positives(pdf, "label")["basket_id"]) == {"A"}
    assert set(rk.drop_groups_without_positives(pdf, "relevance")["basket_id"]) == {"A", "B"}


def test_la_matriz_de_spark_gradua_la_relevancia(spark) -> None:
    from src.recommender.features import with_relevance

    candidates = spark_df(
        spark,
        [
            {"basket_id": "Q", "product_id": "L1", "category": "leche"},  # SKU exacto
            {"basket_id": "Q", "product_id": "P2", "category": "pan"},  # otra referencia
            {"basket_id": "Q", "product_id": "H1", "category": "huevos"},  # nada
            {"basket_id": "R", "product_id": "L1", "category": "leche"},  # otra cesta
        ],
        "basket_id string, product_id string, category string",
    )
    target = spark_df(
        spark,
        [{"basket_id": "Q", "product_id": "L1"}, {"basket_id": "Q", "product_id": "P1"}],
        "basket_id string, product_id string",
    )
    products = spark_df(
        spark, CATALOG.to_dict("records"), "product_id string, category string"
    )
    got = (
        with_relevance(candidates, target, products)
        .toPandas()
        .set_index(["basket_id", "product_id"])
    )
    assert got.loc[("Q", "L1"), ["label", "relevance"]].tolist() == [1, RELEVANCE_SKU]
    assert got.loc[("Q", "P2"), ["label", "relevance"]].tolist() == [0, RELEVANCE_CATEGORY]
    assert got.loc[("Q", "H1"), ["label", "relevance"]].tolist() == [0, RELEVANCE_NONE]
    assert got.loc[("R", "L1"), ["label", "relevance"]].tolist() == [0, RELEVANCE_NONE]
    assert len(got) == 4


# --------------------------------------------------------------------------------------
# NDCG@5 graduada
# --------------------------------------------------------------------------------------
def _graded(top: list[tuple[str, int, str, int]], queries: pd.DataFrame, k: int = 5) -> dict:
    """`top` son filas `(basket_id, rank, product_id, label)`."""
    top_k = pd.DataFrame(top, columns=["basket_id", "rank", "product_id", "label"])
    target = pd.DataFrame(
        [("Q", "L1"), ("Q", "P1"), ("R", "H1")], columns=["basket_id", "product_id"]
    )
    out = ev.category_metrics(top_k, target, CATALOG, queries, k=k)
    return out.set_index("grupo")[f"ndcg_graded@{k}"].to_dict()


def test_ndcg_graduada_cuenta_sku_y_categoria() -> None:
    queries = pd.DataFrame({"basket_id": ["Q"], "profile": [3], "n_target": [2]})
    # Puesto 1: SKU exacto (3). Puesto 2: otro pan (1). Puesto 3: huevos (0).
    got = _graded([("Q", 1, "L1", 1), ("Q", 2, "P2", 0), ("Q", 3, "H1", 0)], queries)
    dcg = 3 * DISCOUNT[0] + 1 * DISCOUNT[1]
    idcg = 3 * DISCOUNT[:2].sum() + 1 * DISCOUNT[2:].sum()
    assert got["total"] == pytest.approx(dcg / idcg)


def test_ndcg_graduada_vale_1_en_la_lista_ideal_y_0_sin_recomendaciones() -> None:
    queries = pd.DataFrame(
        {"basket_id": ["Q", "R"], "profile": [3, 1], "n_target": [2, 1]}
    )
    ideal = [
        ("Q", 1, "L1", 1),
        ("Q", 2, "P1", 1),
        ("Q", 3, "L2", 0),
        ("Q", 4, "P2", 0),
        ("Q", 5, "L2", 0),
    ]
    got = _graded(ideal, queries)
    assert got["3 - recurrente, carrito vacio"] == pytest.approx(1.0)
    # R no recibe nada: cuenta con 0, no desaparece de la media.
    assert got["1 - nuevo, carrito vacio"] == 0.0
    assert got["total"] == pytest.approx(0.5)


def test_el_orden_importa_en_la_ndcg_graduada() -> None:
    queries = pd.DataFrame({"basket_id": ["Q"], "profile": [3], "n_target": [1]})
    sku_first = _graded([("Q", 1, "L1", 1), ("Q", 2, "P2", 0)], queries)["total"]
    cat_first = _graded([("Q", 1, "P2", 0), ("Q", 2, "L1", 1)], queries)["total"]
    assert sku_first > cat_first > 0


# --------------------------------------------------------------------------------------
# Ranking personal de categorias
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def history() -> srv.AsOfHistory:
    return srv.asof_history(QUERIES, _bundle())


def test_ranking_personal_por_frecuencia_y_cuota(history) -> None:
    # Q1 (20-dic): leche en 3 dias (1-oct, 15-oct, 5-dic) y pan en 1 (1-oct).
    leche = _row(history.categories, basket_id="Q1", category="leche")
    pan = _row(history.categories, basket_id="Q1", category="pan")
    assert (leche["cat_freq_rank"], pan["cat_freq_rank"]) == (1.0, 2.0)
    assert leche["cat_freq_share"] == pytest.approx(1.0)
    assert pan["cat_freq_share"] == pytest.approx(1 / 3)


def test_el_ranking_por_necesidad_dobla_lo_que_toca_reponer(history) -> None:
    # Q1: leche repuesta hace 15 dias (ciclo 32,5) -> 3 x 1; pan vencido -> 1 x 2.
    leche = _row(history.categories, basket_id="Q1", category="leche")
    pan = _row(history.categories, basket_id="Q1", category="pan")
    assert (leche["cat_due_rank"], pan["cat_due_rank"]) == (1.0, 2.0)
    # Q2 (1-nov): leche 2 dias y vencida (17 dias con ciclo de 14) -> 4; pan 1 x 2 -> 2.
    leche2 = _row(history.categories, basket_id="Q2", category="leche")
    pan2 = _row(history.categories, basket_id="Q2", category="pan")
    assert (leche2["cat_freq_rank"], pan2["cat_freq_rank"]) == (1.0, 2.0)
    assert (leche2["cat_due_rank"], pan2["cat_due_rank"]) == (1.0, 2.0)


def test_empates_comparten_puesto() -> None:
    frame = pd.DataFrame(
        {
            "basket_id": ["Q", "Q", "Q"],
            "category": ["a", "b", "c"],
            "cat_n_purchase_days": [2.0, 2.0, 1.0],
            "cat_last_day": pd.to_datetime(["2025-12-01"] * 3),
            "cat_expected_days": [100.0, 100.0, 100.0],
            "_query_day": pd.to_datetime(["2025-12-02"] * 3),
            "_customer_days": [4, 4, 4],
        }
    )
    got = srv._with_category_ranks(frame)
    assert got["cat_freq_rank"].tolist() == [1.0, 1.0, 2.0]
    assert got["cat_due_rank"].tolist() == [1.0, 1.0, 2.0]
    assert got["cat_freq_share"].tolist() == [0.5, 0.5, 0.25]


def test_la_necesidad_es_la_del_baseline_personal_due() -> None:
    freq = np.array([[3.0, 1.0, 0.0]])
    due = np.array([[0.0, 1.0, 1.0]])
    np.testing.assert_allclose(fx.category_need_score(freq, due), [[3.0, 2.0, 0.0]])
    series = fx.category_need_score(pd.Series([3.0, 1.0]), pd.Series([0.0, 1.0]))
    assert series.tolist() == [3.0, 2.0]

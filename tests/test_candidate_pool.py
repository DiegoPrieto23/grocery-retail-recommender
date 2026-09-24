"""Tests de los puntos M6 y B4 del diagnostico.

* **M6 - rama por categoria de la fuente `pop`**: `fit_category_popularity` ordena las
  categorias por peso esperado del mes y las referencias dentro de cada una, y
  `candidates_popularity` une esa rama con el top global sin duplicar productos ni
  inventar rangos. La paridad Spark / pandas de la fuente entera la cubre
  `tests/test_serving_parity.py`.
* **M6 - techo por categoria**: `evaluate.candidate_recall` separa el techo de SKU del de
  categoria, que es el que estaba bajo en el cold-start y el que manda sobre la metrica
  principal.
* **B4 - de donde sale la validacion**: `validation_baskets` reparte cestas enteras, y en
  modo temporal se queda exactamente con la cola de la ventana.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.recommender import candidates as cand
from src.recommender import evaluate as ev
from src.recommender import splits
from src.recommender.config import (
    VALID_HASH,
    VALID_TEMPORAL,
    CandidateConfig,
    ValidationSplit,
)
from src.serving import recommend as srv
from tests.helpers import spark_df

# Dos categorias, una de mucho peso y otra de poco. La popularidad global se lleva las
# tres referencias de `leche` antes de tocar `pan`: es justo el fallo del punto M6.
POPULARITY_SCHEMA = "product_id string, month int, pop_score double, pop_rank int"
POPULARITY = [
    {"product_id": "L1", "month": 11, "pop_score": 100.0, "pop_rank": 1},
    {"product_id": "L2", "month": 11, "pop_score": 90.0, "pop_rank": 2},
    {"product_id": "L3", "month": 11, "pop_score": 80.0, "pop_rank": 3},
    {"product_id": "P1", "month": 11, "pop_score": 20.0, "pop_rank": 4},
    {"product_id": "P2", "month": 11, "pop_score": 10.0, "pop_rank": 5},
    {"product_id": "H1", "month": 11, "pop_score": 5.0, "pop_rank": 6},
]
PRODUCTS_SCHEMA = "product_id string, category string"
PRODUCTS = [
    {"product_id": p, "category": c}
    for p, c in [
        ("L1", "leche"),
        ("L2", "leche"),
        ("L3", "leche"),
        ("P1", "pan"),
        ("P2", "pan"),
        ("H1", "huevos"),
    ]
]
QUERIES_SCHEMA = "basket_id string, basket_day date"
QUERIES = [{"basket_id": "B1", "basket_day": dt.date(2025, 11, 5)}]

CATPOP_SCHEMA = (
    "product_id string, month int, category string, cat_pop_score double, "
    "cat_pop_rank int, cat_prod_rank int, pop_score double, pop_rank int"
)


@pytest.fixture(scope="module")
def category_popularity(spark) -> pd.DataFrame:
    fitted = cand.fit_category_popularity(
        spark_df(spark, POPULARITY, POPULARITY_SCHEMA),
        spark_df(spark, PRODUCTS, PRODUCTS_SCHEMA),
    )
    return fitted.toPandas().sort_values("product_id").reset_index(drop=True)


def _pool(spark, category_popularity, cfg) -> pd.DataFrame:
    """La fuente `pop` para la unica query de prueba, ya en pandas."""
    return cand.candidates_popularity(
        spark_df(spark, QUERIES, QUERIES_SCHEMA),
        spark_df(spark, POPULARITY, POPULARITY_SCHEMA),
        spark_df(spark, category_popularity.to_dict("records"), CATPOP_SCHEMA),
        cfg=cfg,
    ).toPandas()


def test_categorias_ordenadas_por_peso_esperado_del_mes(category_popularity):
    """`leche` (270) pesa mas que `pan` (30) y `pan` mas que `huevos` (5)."""
    ranks = category_popularity.set_index("product_id")["cat_pop_rank"]
    assert ranks["L1"] == 1 and ranks["L2"] == 1 and ranks["L3"] == 1
    assert ranks["P1"] == 2
    assert ranks["H1"] == 3


def test_referencias_ordenadas_dentro_de_su_categoria(category_popularity):
    """Dentro de cada categoria manda el `pop_score`, con el `product_id` de desempate."""
    within = category_popularity.set_index("product_id")["cat_prod_rank"]
    assert [within[p] for p in ("L1", "L2", "L3")] == [1, 2, 3]
    assert [within[p] for p in ("P1", "P2")] == [1, 2]
    assert within["H1"] == 1


def test_la_rama_por_categoria_alcanza_lo_que_la_global_no(spark, category_popularity):
    """Con `n_popularity = 2` la popularidad global solo ve `leche`; la de categoria, todo."""
    cfg = CandidateConfig(
        n_popularity=2, n_pop_categories=3, n_pop_products_per_category=1
    )
    pool = _pool(spark, category_popularity, cfg)

    # L1 y L2 por la rama global; L1 (repetido), P1 y H1 por la de categoria.
    assert sorted(pool["product_id"]) == ["H1", "L1", "L2", "P1"]
    assert not pool.duplicated(["basket_id", "product_id"]).any()


def test_el_rango_que_ve_el_ranker_sigue_siendo_el_global(spark, category_popularity):
    """Un candidato que entra por la rama de categoria conserva su `pop_rank` global."""
    cfg = CandidateConfig(
        n_popularity=1, n_pop_categories=3, n_pop_products_per_category=1
    )
    pool = _pool(spark, category_popularity, cfg).set_index("product_id")
    assert pool.loc["H1", "pop_rank"] == 6
    assert pool.loc["H1", "pop_score"] == pytest.approx(5.0)


def test_n_pop_categories_cero_deja_la_popularidad_global_de_antes(
    spark, category_popularity
):
    """La rama nueva se puede apagar entera, que es como se mide el antes/despues."""
    cfg = CandidateConfig(n_popularity=2, n_pop_categories=0)
    pool = _pool(spark, category_popularity, cfg)
    assert sorted(pool["product_id"]) == ["L1", "L2"]


def test_paridad_pandas_de_la_fuente_de_popularidad(spark, category_popularity):
    """La ruta de la demo (pandas) propone exactamente los mismos productos que Spark."""
    cfg = CandidateConfig(
        n_popularity=2, n_pop_categories=3, n_pop_products_per_category=2
    )
    from_spark = _pool(spark, category_popularity, cfg)

    bundle = object.__new__(srv.ServingBundle)
    bundle.popularity = pd.DataFrame(POPULARITY)
    bundle.category_popularity = category_popularity
    bundle.cfg = cfg
    queries = pd.DataFrame(QUERIES)
    queries["basket_day"] = pd.to_datetime(queries["basket_day"])
    from_pandas = srv.candidates_popularity(queries, bundle)

    assert sorted(from_spark["product_id"]) == sorted(from_pandas["product_id"])


# --------------------------------------------------------------------------------------
# Punto B4: de donde sale la validacion de la parada temprana
# --------------------------------------------------------------------------------------
BASKET_DAYS = pd.Series(
    {f"B{i:02d}": dt.date(2025, 10, 1) + dt.timedelta(days=i) for i in range(30)},
    name="basket_day",
)


def test_validacion_temporal_es_la_cola_de_la_ventana():
    """Con 7 dias entran los 7 ultimos dias que aparecen en la muestra, no 7 cestas."""
    chosen = splits.validation_baskets(
        BASKET_DAYS, split=ValidationSplit(mode=VALID_TEMPORAL, days=7), n_valid=3
    )
    days = BASKET_DAYS.loc[sorted(chosen)]
    assert days.min() == dt.date(2025, 10, 24)
    assert days.max() == BASKET_DAYS.max()
    assert len(chosen) == 7


def test_validacion_temporal_no_toca_el_pasado_de_la_ventana():
    """Ninguna cesta de entrenamiento es posterior a la primera de validacion."""
    split = ValidationSplit(mode=VALID_TEMPORAL, days=10)
    chosen = splits.validation_baskets(BASKET_DAYS, split=split, n_valid=3)
    train = BASKET_DAYS.loc[[b for b in BASKET_DAYS.index if b not in chosen]]
    assert train.max() < BASKET_DAYS.loc[sorted(chosen)].min()


def test_validacion_por_hash_aparta_el_numero_pedido_de_cestas():
    """El modo anterior sigue disponible como ablacion, y reparte por cesta entera."""
    chosen = splits.validation_baskets(
        BASKET_DAYS, split=ValidationSplit(mode=VALID_HASH), n_valid=5
    )
    assert len(chosen) == 5
    assert chosen <= set(BASKET_DAYS.index)


def test_validacion_por_hash_con_ventana_corta_deja_cestas_para_entrenar():
    """A escala reducida la ventana trae menos cestas de las pedidas: se aparta la misma
    proporcion en vez de mandar la ventana entera a validacion."""
    chosen = splits.validation_baskets(
        BASKET_DAYS, split=ValidationSplit(mode=VALID_HASH), n_valid=50, n_train=250
    )
    assert len(chosen) == 5  # 30 cestas * 50 / 300
    assert len(BASKET_DAYS) - len(chosen) == 25


def test_modo_de_validacion_desconocido_falla_pronto():
    with pytest.raises(ValueError, match="modo de validacion desconocido"):
        ValidationSplit(mode="aleatorio")


# --------------------------------------------------------------------------------------
# Punto M6: el techo de la primera etapa, tambien a nivel de categoria
# --------------------------------------------------------------------------------------
POOL_CATEGORIES = pd.DataFrame(
    {
        "product_id": ["L1", "L2", "L3", "P1", "H1"],
        "category": ["leche", "leche", "leche", "pan", "huevos"],
    }
)


def test_cat_pool_recall_distingue_un_pool_profundo_de_uno_ancho():
    """Tres leches cubren una categoria de tres; el mismo recall de SKU, distinto techo."""
    # El target son tres categorias distintas y el pool solo alcanza `leche`.
    scored = pd.DataFrame(
        [
            {"basket_id": "B1", "product_id": p, "label": int(p == "L1"), "score": 1.0}
            for p in ("L1", "L2", "L3")
        ]
    )
    target = pd.DataFrame(
        [{"basket_id": "B1", "product_id": p} for p in ("L1", "P1", "H1")]
    )
    queries = pd.DataFrame([{"basket_id": "B1", "profile": 1, "n_target": 3}])

    techo = ev.candidate_recall(scored, queries, target, POOL_CATEGORIES)
    # Un SKU de tres, y una categoria de tres: aqui las dos coinciden.
    assert techo.loc[0, "pool_recall"] == pytest.approx(1 / 3)
    assert techo.loc[0, "cat_pool_recall"] == pytest.approx(1 / 3)

    # Mismo recall de SKU, pero el pool ahora toca las tres categorias: el techo de la
    # metrica principal sube aunque el de SKU no se mueva. Es el caso del punto M6.
    ancho = pd.DataFrame(
        [
            {"basket_id": "B1", "product_id": p, "label": int(p == "L1"), "score": 1.0}
            for p in ("L1", "P1", "H1")
        ]
    )
    techo_ancho = ev.candidate_recall(ancho, queries, target, POOL_CATEGORIES)
    assert techo_ancho.loc[0, "pool_recall"] == pytest.approx(1 / 3)
    assert techo_ancho.loc[0, "cat_pool_recall"] == pytest.approx(1.0)


def test_sin_catalogo_no_se_inventa_la_columna_de_categoria():
    """La firma antigua sigue valiendo y no anade una columna a medias."""
    scored = pd.DataFrame([{"basket_id": "B1", "product_id": "L1", "label": 1, "score": 1.0}])
    queries = pd.DataFrame([{"basket_id": "B1", "profile": 3, "n_target": 2}])
    techo = ev.candidate_recall(scored, queries)
    assert "cat_pool_recall" not in techo.columns
    assert techo.loc[0, "pool_recall"] == pytest.approx(0.5)

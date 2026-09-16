"""Tests del recomendador de cesta (Fase 3).

Se agrupan en cuatro bloques, y el primero es el importante:

* **Fuga de datos**: que ninguna cesta se parta entre ventanas, que el target nunca entre
  en el prefijo ni en el pool de candidatos, y que la senal de sesion se corte de verdad
  en `cut_ts`. Es lo que hace creible el NDCG@5 que se publica.
* **Perfiles**: que los cuatro casos de `CHALLENGE.md` se identifiquen bien y que cada uno
  active las fuentes que le tocan.
* **Fuentes de candidatos**: estacionalidad, co-compra e historial.
* **Metricas**: NDCG@5 y Recall@5 contra valores calculados a mano.

Los DataFrames son minusculos y estan escritos a mano: se busca que un cambio de logica
falle aqui de forma legible, no reproducir el dataset real.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from src.recommender import candidates as cand
from src.recommender import evaluate as ev
from src.recommender import features as feat
from src.recommender import splits
from src.recommender.config import CandidateConfig
from tests.helpers import spark_df

# Esquemas de las tablas **ya limpias** (`data/processed`), que es lo que consume la fase.
BASKETS_DDL = (
    "basket_id string, customer_id string, channel string, basket_date timestamp, "
    "basket_day date, total_amount double"
)
ITEMS_DDL = "basket_id string, product_id string, quantity int"
PRODUCTS_DDL = (
    "product_id string, department string, category string, brand string, "
    "is_private_label boolean, is_perishable boolean, unit_price double, "
    "pack_size int, typical_repurchase_days int"
)
SESSIONS_DDL = (
    "session_id string, customer_id string, session_date timestamp, device_type string, "
    "converted boolean, basket_id string"
)
EVENTS_DDL = (
    "session_id string, product_id string, event_type string, event_timestamp timestamp"
)


def _basket(basket_id: str, customer_id: str | None, day: str, channel: str = "web") -> dict:
    return {
        "basket_id": basket_id,
        "customer_id": customer_id,
        "channel": channel,
        "basket_date": f"{day} 18:00:00",
        "basket_day": day,
        "total_amount": 30.0,
    }


def _items(basket_id: str, product_ids: list[str]) -> list[dict]:
    return [{"basket_id": basket_id, "product_id": p, "quantity": 1} for p in product_ids]


def _product(product_id: str, category: str, department: str = "Despensa") -> dict:
    return {
        "product_id": product_id,
        "department": department,
        "category": category,
        "brand": "Marca",
        "is_private_label": False,
        "is_perishable": False,
        "unit_price": 2.0,
        "pack_size": 1,
        "typical_repurchase_days": 20,
    }


@pytest.fixture()
def toy(spark):
    """Un mundo minimo: 4 cestas repartidas en dos ventanas y una sesion online."""
    baskets = spark_df(
        spark,
        [
            # Historial (anterior a la ventana de test)
            _basket("B1", "C1", "2025-01-10"),
            _basket("B2", "C1", "2025-02-10"),
            # Ventana de test
            _basket("B3", "C1", "2025-03-10"),  # cliente recurrente
            _basket("B4", "C9", "2025-03-11"),  # cliente sin historial
            _basket("B5", None, "2025-03-12"),  # compra anonima
        ],
        BASKETS_DDL,
    )
    items = spark_df(
        spark,
        _items("B1", ["P1", "P2"])
        + _items("B2", ["P1", "P3"])
        + _items("B3", ["P1", "P2", "P3", "P4"])
        + _items("B4", ["P1", "P2"])
        + _items("B5", ["P3", "P4"]),
        ITEMS_DDL,
    )
    products = spark_df(
        spark,
        [
            _product("P1", "Leche"),
            _product("P2", "Cereales"),
            _product("P3", "Pasta"),
            _product("P4", "Salsa de tomate"),
        ],
        PRODUCTS_DDL,
    )
    sessions = spark_df(
        spark,
        [
            {
                "session_id": "S1",
                "customer_id": "C1",
                "session_date": "2025-03-10 17:30:00",
                "device_type": "mobile",
                "converted": True,
                "basket_id": "B3",
            }
        ],
        SESSIONS_DDL,
    )
    events = spark_df(
        spark,
        [
            # Orden real de la sesion: se ve P1 y se anade; se ve P3 -- que se anadira
            # mucho despues -- y se ve P2, que se anade acto seguido. El corte cae en ese
            # ultimo anadido, asi que la vista de P3 queda dentro y la de P4 fuera. P9 se
            # mira y no se compra nunca.
            {"session_id": "S1", "product_id": "P1", "event_type": "view",
             "event_timestamp": "2025-03-10 17:31:00"},
            {"session_id": "S1", "product_id": "P1", "event_type": "add_to_cart",
             "event_timestamp": "2025-03-10 17:32:00"},
            {"session_id": "S1", "product_id": "P3", "event_type": "view",
             "event_timestamp": "2025-03-10 17:32:30"},
            {"session_id": "S1", "product_id": "P2", "event_type": "view",
             "event_timestamp": "2025-03-10 17:33:00"},
            {"session_id": "S1", "product_id": "P9", "event_type": "view",
             "event_timestamp": "2025-03-10 17:33:30"},
            {"session_id": "S1", "product_id": "P2", "event_type": "add_to_cart",
             "event_timestamp": "2025-03-10 17:34:00"},
            {"session_id": "S1", "product_id": "P3", "event_type": "add_to_cart",
             "event_timestamp": "2025-03-10 17:40:00"},
            {"session_id": "S1", "product_id": "P4", "event_type": "view",
             "event_timestamp": "2025-03-10 17:45:00"},
            {"session_id": "S1", "product_id": "P4", "event_type": "add_to_cart",
             "event_timestamp": "2025-03-10 17:46:00"},
        ],
        EVENTS_DDL,
    )
    return {
        "baskets": baskets,
        "basket_items": items,
        "products": products,
        "sessions": sessions,
        "session_events": events,
    }


# --------------------------------------------------------------------------------------
# Fuga de datos
# --------------------------------------------------------------------------------------
def test_las_ventanas_no_comparten_ninguna_cesta(toy) -> None:
    """El split es temporal, pero la unidad sigue siendo la cesta entera."""
    history = splits.baskets_before(toy["baskets"], dt.date(2025, 3, 1))
    window = splits.baskets_between(toy["baskets"], dt.date(2025, 3, 1))

    ids_history = {r["basket_id"] for r in history.select("basket_id").collect()}
    ids_window = {r["basket_id"] for r in window.select("basket_id").collect()}
    assert ids_history == {"B1", "B2"}
    assert ids_window == {"B3", "B4", "B5"}
    assert not ids_history & ids_window


def test_prefijo_y_target_parten_la_cesta_sin_solaparse(toy) -> None:
    """Cada producto de la cesta cae en un lado y solo en uno."""
    window = splits.baskets_between(toy["baskets"], dt.date(2025, 3, 1))
    query_items = splits.build_query_items(window, toy["basket_items"])
    rows = query_items.toPandas()

    for basket_id, group in rows.groupby("basket_id"):
        prefix = set(group.loc[group["is_prefix"], "product_id"])
        target = set(group.loc[~group["is_prefix"], "product_id"])
        assert not prefix & target
        assert prefix | target == set(group["product_id"])
        # Siempre queda algo que adivinar, incluso cortando por la mitad.
        assert target


def test_el_target_nunca_entra_en_el_pool_de_candidatos(toy) -> None:
    """`union_candidates` quita del pool lo que ya esta en el carrito, y solo eso."""
    window = splits.baskets_between(toy["baskets"], dt.date(2025, 3, 1))
    query_items = splits.build_query_items(window, toy["basket_items"])
    queries = splits.build_queries(
        window, query_items, splits.known_customers(splits.baskets_before(toy["baskets"], "2025-03-01")),
        toy["sessions"],
    )
    prefix, target = splits.prefix_and_target(query_items, queries)

    source = prefix.select("basket_id").distinct().crossJoin(
        toy["products"].select("product_id")
    ).withColumn("pop_score", F.lit(1.0)).withColumn("pop_rank", F.lit(1.0))
    pool = cand.union_candidates({"pop": source}, queries, cand.in_cart(prefix, None))

    en_pool = {(r["basket_id"], r["product_id"]) for r in pool.collect()}
    en_prefijo = {(r["basket_id"], r["product_id"]) for r in prefix.collect()}
    en_target = {(r["basket_id"], r["product_id"]) for r in target.collect()}

    assert not en_pool & en_prefijo, "el pool propone algo que ya esta en el carrito"
    # El target si tiene que poder estar: es lo que hay que acertar.
    assert en_pool & en_target


def test_la_sesion_se_corta_en_el_instante_del_prefijo(toy) -> None:
    """Ningun evento posterior a `cut_ts` puede llegar a las features.

    Es la condicion que hace utilizable `session_events` sin filtrar el target. En la
    cesta B3 el corte cae tras anadir P2, asi que la vista de P3 (anterior) cuenta y la
    de P4 (posterior) no.
    """
    window = splits.baskets_between(toy["baskets"], dt.date(2025, 3, 1))
    add_to_cart = splits.basket_add_to_cart(toy["sessions"], toy["session_events"])
    query_items = splits.build_query_items(window, toy["basket_items"], add_to_cart)
    queries = splits.build_queries(
        window,
        query_items,
        splits.known_customers(splits.baskets_before(toy["baskets"], "2025-03-01")),
        toy["sessions"],
    )
    before = feat.session_events_before_cut(queries, toy["sessions"], toy["session_events"])
    rows = before.toPandas()

    b3 = queries.filter(F.col("basket_id") == "B3").collect()[0]
    assert b3["prefix_size"] == 2, "B3 tiene 4 lineas: el corte cae a la mitad"

    eventos_b3 = rows.loc[rows["basket_id"] == "B3"]
    assert (eventos_b3["event_timestamp"] <= eventos_b3["cut_ts"]).all()
    vistos = set(eventos_b3.loc[eventos_b3["event_type"] == "view", "product_id"])
    assert "P3" in vistos, "P3 se vio antes del corte y es una pista legitima"
    assert "P4" not in vistos, "P4 se vio despues del corte: seria mirar el futuro"

    anadidos = set(
        feat.session_cart(before).filter(F.col("basket_id") == "B3").toPandas()["product_id"]
    )
    assert anadidos == {"P1", "P2"}


# --------------------------------------------------------------------------------------
# Perfiles
# --------------------------------------------------------------------------------------
def test_los_cuatro_perfiles_se_identifican(toy) -> None:
    """Nuevo/recurrente por historial, y carrito vacio/lleno por `prefix_size`."""
    history = splits.baskets_before(toy["baskets"], dt.date(2025, 3, 1))
    window = splits.baskets_between(toy["baskets"], dt.date(2025, 3, 1))
    query_items = splits.build_query_items(window, toy["basket_items"])
    queries = splits.build_queries(
        window, query_items, splits.known_customers(history), toy["sessions"]
    ).toPandas()

    por_cesta = queries.set_index("basket_id")
    # B3 es de C1, que ya compro en enero y febrero.
    assert bool(por_cesta.loc["B3", "is_known_customer"]) is True
    # B4 es de C9 y B5 es anonima: los dos cuentan como cliente nuevo.
    assert bool(por_cesta.loc["B4", "is_known_customer"]) is False
    assert bool(por_cesta.loc["B5", "is_known_customer"]) is False

    esperado = {
        (False, False): 1,
        (False, True): 2,
        (True, False): 3,
        (True, True): 4,
    }
    for basket_id, row in por_cesta.iterrows():
        clave = (bool(row["is_known_customer"]), row["prefix_size"] > 0)
        assert row["profile"] == esperado[clave], basket_id

    assert (por_cesta["n_target"] == por_cesta["n_items"] - por_cesta["prefix_size"]).all()


def test_sin_carrito_no_hay_candidatos_de_co_compra(toy) -> None:
    """Los perfiles 1 y 3 no pueden activar la co-compra: no hay antecedente."""
    window = splits.baskets_between(toy["baskets"], dt.date(2025, 3, 1))
    query_items = splits.build_query_items(window, toy["basket_items"])
    queries = splits.build_queries(
        window, query_items, splits.known_customers(splits.baskets_before(toy["baskets"], "2025-03-01")),
        toy["sessions"],
    )
    prefix, _ = splits.prefix_and_target(query_items, queries)

    vacias = {
        r["basket_id"]
        for r in queries.filter(F.col("prefix_size") == 0).select("basket_id").collect()
    }
    con_prefijo = {r["basket_id"] for r in prefix.select("basket_id").distinct().collect()}
    assert not vacias & con_prefijo


# --------------------------------------------------------------------------------------
# Fuentes de candidatos
# --------------------------------------------------------------------------------------
def test_el_indice_estacional_detecta_un_producto_de_temporada(spark) -> None:
    """Un producto que solo se vende en diciembre tiene que destacar en diciembre.

    Es la comprobacion de que la fuente de estacionalidad recoge los multiplicadores de
    `DATA_SPEC.md` (turron x8, helados x4) y no solo la popularidad media.
    """
    rows, items = [], []
    # P_ALL se vende todos los meses; P_DIC solo en diciembre.
    for month in range(1, 13):
        for i in range(10):
            basket_id = f"B{month:02d}{i:02d}"
            rows.append(_basket(basket_id, f"C{i}", f"2024-{month:02d}-15"))
            items += _items(basket_id, ["P_ALL"] + (["P_DIC"] if month == 12 else []))
    baskets = spark_df(spark, rows, BASKETS_DDL)
    basket_items = spark_df(spark, items, ITEMS_DDL)

    popularity = cand.fit_popularity(
        baskets, basket_items, window_start="2025-01-01", cfg=CandidateConfig(recent_days=400)
    ).toPandas()

    dic = popularity.loc[popularity["month"] == 12].set_index("product_id")
    assert dic.loc["P_DIC", "pop_seasonal_index"] > 5.0
    assert dic.loc["P_ALL", "pop_seasonal_index"] == pytest.approx(1.0, abs=0.05)
    # Y fuera de temporada no aparece en absoluto.
    assert "P_DIC" not in set(popularity.loc[popularity["month"] == 6, "product_id"])


def test_la_co_compra_propone_lo_que_acompana_al_carrito(spark) -> None:
    """Pasta en el carrito debe traer salsa de tomate, que es la regla de `DATA_SPEC.md`."""
    rows, items = [], []
    for i in range(120):
        basket_id = f"B{i:04d}"
        rows.append(_basket(basket_id, f"C{i % 20}", "2024-05-10"))
        # Pasta y salsa siempre juntas; leche por su cuenta.
        items += _items(basket_id, ["P_PASTA", "P_SALSA"] if i % 2 == 0 else ["P_LECHE"])
    baskets = spark_df(spark, rows, BASKETS_DDL)
    basket_items = spark_df(spark, items, ITEMS_DDL)

    affinity = cand.fit_affinity_product(basket_items)
    prefix = spark_df(
        spark, [{"basket_id": "Q1", "product_id": "P_PASTA"}], "basket_id string, product_id string"
    )
    propuesto = cand.candidates_affinity_product(
        prefix, affinity, cfg=CandidateConfig()
    ).toPandas()

    assert set(propuesto["product_id"]) == {"P_SALSA"}
    assert propuesto["aff_lift_max"].iloc[0] > 1.5


def test_el_pool_marca_que_fuente_propuso_cada_producto(spark) -> None:
    """`src_*` y `n_sources` son features: que dos fuentes coincidan es informacion."""
    queries = spark_df(spark, [{"basket_id": "Q1"}], "basket_id string")
    vacio = spark_df(spark, [], "basket_id string, product_id string")
    pop = spark_df(
        spark,
        [{"basket_id": "Q1", "product_id": "P1", "pop_score": 9.0, "pop_rank": 1.0},
         {"basket_id": "Q1", "product_id": "P2", "pop_score": 8.0, "pop_rank": 2.0}],
        "basket_id string, product_id string, pop_score double, pop_rank double",
    )
    als = spark_df(
        spark,
        [{"basket_id": "Q1", "product_id": "P1", "als_score": 0.7, "als_rank": 1.0}],
        "basket_id string, product_id string, als_score double, als_rank double",
    )
    pool = cand.union_candidates({"pop": pop, "als": als}, queries, vacio).toPandas()
    por_producto = pool.set_index("product_id")

    assert por_producto.loc["P1", "n_sources"] == 2
    assert por_producto.loc["P2", "n_sources"] == 1
    assert por_producto.loc["P1", "src_als"] == 1.0
    assert por_producto.loc["P2", "src_als"] == 0.0
    # Las columnas de las fuentes que no participaron existen igual, a nulo.
    assert pd.isna(por_producto.loc["P2", "als_score"])
    assert set(pool.columns) >= {f"src_{s}" for s in cand.SOURCE_NAMES}


# --------------------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------------------
def _scored(rows: list[tuple[str, str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["basket_id", "product_id", "score", "label"])


def test_ndcg_y_recall_contra_el_calculo_a_mano() -> None:
    """Un acierto en el puesto 2 de un target de 2 productos.

    DCG  = 1/log2(3)                     = 0.6309
    IDCG = 1/log2(2) + 1/log2(3)         = 1.6309
    NDCG = 0.3869 ; Recall = 1/2 = 0.5
    """
    scored = _scored(
        [
            ("B1", "P1", 0.9, 0),
            ("B1", "P2", 0.8, 1),
            ("B1", "P3", 0.7, 0),
            ("B1", "P4", 0.6, 0),
            ("B1", "P5", 0.5, 0),
            ("B1", "P6", 0.4, 1),  # sexto: fuera del top-5, no cuenta
        ]
    )
    queries = pd.DataFrame([{"basket_id": "B1", "profile": 4, "n_target": 2}])
    resumen, por_query = ev.evaluate(scored, queries, k=5)

    esperado = (1 / np.log2(3)) / (1 / np.log2(2) + 1 / np.log2(3))
    assert por_query["ndcg"].iloc[0] == pytest.approx(esperado, abs=1e-6)
    assert por_query["recall"].iloc[0] == pytest.approx(0.5)
    assert resumen.loc[0, "ndcg@5"] == pytest.approx(esperado, abs=1e-6)


def test_precision_y_f1_contra_el_calculo_a_mano() -> None:
    """Dos cestas, para que las dos variantes de F1@5 se separen.

    B1: 1 acierto, T = 2  ->  P = 1/5, R = 1/2, F1 = 2/7
    B2: 0 aciertos, T = 3 ->  P = 0,   R = 0,   F1 = 0
    Media de P = 0,1 ; media de R = 0,25
    F1@5 (armonica de las medias) = 2 x 0,1 x 0,25 / 0,35 = 1/7
    F1@5 por cesta (media de los F1)                       = 1/7 tambien, por casualidad:
    se rompe la igualdad con B2 acertando algo, que es la tercera comprobacion.
    """
    scored = _scored(
        [
            ("B1", "P1", 0.9, 0),
            ("B1", "P2", 0.8, 1),
            ("B2", "P1", 0.9, 0),
        ]
    )
    queries = pd.DataFrame(
        [
            {"basket_id": "B1", "profile": 4, "n_target": 2},
            {"basket_id": "B2", "profile": 4, "n_target": 3},
        ]
    )
    resumen, por_query = ev.evaluate(scored, queries, k=5)
    por_query = por_query.set_index("basket_id")
    assert por_query.loc["B1", "precision"] == pytest.approx(0.2)
    assert por_query.loc["B1", "f1"] == pytest.approx(2 / 7)
    assert por_query.loc["B2", "f1"] == 0.0
    assert resumen.loc[0, "precision@5"] == pytest.approx(0.1)
    assert resumen.loc[0, "f1@5"] == pytest.approx(2 * 0.1 * 0.25 / 0.35)

    # Con B2 acertando uno de tres, las dos variantes dejan de coincidir.
    scored.loc[scored["basket_id"] == "B2", "label"] = 1
    resumen, _ = ev.evaluate(scored, queries, k=5)
    p, r = (0.2 + 0.2) / 2, (0.5 + 1 / 3) / 2
    assert resumen.loc[0, "f1@5"] == pytest.approx(2 * p * r / (p + r))
    assert resumen.loc[0, "f1@5_por_cesta"] == pytest.approx((2 / 7 + 2 / 8) / 2)
    assert resumen.loc[0, "f1@5"] != pytest.approx(resumen.loc[0, "f1@5_por_cesta"])


def test_el_orden_perfecto_da_ndcg_uno() -> None:
    """Con los dos aciertos en los dos primeros huecos, NDCG@5 = 1."""
    scored = _scored(
        [("B1", "P1", 0.9, 1), ("B1", "P2", 0.8, 1), ("B1", "P3", 0.7, 0)]
    )
    queries = pd.DataFrame([{"basket_id": "B1", "profile": 4, "n_target": 2}])
    _, por_query = ev.evaluate(scored, queries, k=5)
    assert por_query["ndcg"].iloc[0] == pytest.approx(1.0)
    assert por_query["recall"].iloc[0] == pytest.approx(1.0)


def test_el_recall_se_mide_contra_la_cesta_real_no_contra_el_pool() -> None:
    """Si la primera etapa no propuso un producto, eso es un fallo, no un caso omitido."""
    scored = _scored([("B1", "P1", 0.9, 1)])
    # La cesta tenia 4 productos por adivinar; al pool solo llego uno.
    queries = pd.DataFrame([{"basket_id": "B1", "profile": 3, "n_target": 4}])
    _, por_query = ev.evaluate(scored, queries, k=5)
    assert por_query["recall"].iloc[0] == pytest.approx(0.25)

    techo = ev.candidate_recall(scored, queries)
    assert techo.loc[0, "pool_recall"] == pytest.approx(0.25)


def test_una_query_sin_ningun_candidato_puntua_cero() -> None:
    """No se puede desaparecer del denominador por no haber propuesto nada."""
    scored = _scored([("B1", "P1", 0.9, 1)])
    queries = pd.DataFrame(
        [
            {"basket_id": "B1", "profile": 4, "n_target": 1},
            {"basket_id": "B2", "profile": 1, "n_target": 2},
        ]
    )
    resumen, por_query = ev.evaluate(scored, queries, k=5)
    assert len(por_query) == 2
    assert por_query.set_index("basket_id").loc["B2", "ndcg"] == 0.0
    assert resumen.loc[0, "n_queries"] == 2
    assert resumen.loc[0, "ndcg@5"] == pytest.approx(0.5)


def test_el_resumen_desglosa_los_perfiles() -> None:
    """El desglose por perfil es lo que responde a "¿aguanta el cold-start?"."""
    scored = _scored([("B1", "P1", 0.9, 1), ("B2", "P1", 0.9, 0)])
    queries = pd.DataFrame(
        [
            {"basket_id": "B1", "profile": 4, "n_target": 1},
            {"basket_id": "B2", "profile": 1, "n_target": 1},
        ]
    )
    resumen, _ = ev.evaluate(scored, queries, k=5)
    por_grupo = resumen.set_index("grupo")
    assert por_grupo.loc[splits.PROFILE_LABELS[4], "ndcg@5"] == pytest.approx(1.0)
    assert por_grupo.loc[splits.PROFILE_LABELS[1], "ndcg@5"] == pytest.approx(0.0)


def test_el_top_k_desempata_de_forma_estable() -> None:
    """Con scores empatados, dos ejecuciones deben devolver la misma lista."""
    scored = _scored([("B1", f"P{i}", 0.5, 0) for i in range(9, 0, -1)])
    primero = ev.top_k_predictions(scored, k=5)["product_id"].tolist()
    segundo = ev.top_k_predictions(scored.sample(frac=1, random_state=7), k=5)[
        "product_id"
    ].tolist()
    assert primero == segundo == ["P1", "P2", "P3", "P4", "P5"]


# --------------------------------------------------------------------------------------
# Contrato del modelo
# --------------------------------------------------------------------------------------
def test_las_columnas_de_features_no_se_repiten() -> None:
    """`FEATURE_COLUMNS` es el contrato del modelo guardado: sin duplicados y sin huecos."""
    assert len(feat.FEATURE_COLUMNS) == len(set(feat.FEATURE_COLUMNS))
    assert set(feat.CATEGORICAL_FEATURES) <= set(feat.FEATURE_COLUMNS)
    for source, columns in cand.SOURCE_COLUMNS.items():
        assert set(columns) <= set(feat.FEATURE_COLUMNS), source
        assert f"src_{source}" in feat.FEATURE_COLUMNS

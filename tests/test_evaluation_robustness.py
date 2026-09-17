"""Tests de la evaluacion robusta del recomendador (puntos M3 y M4 del diagnostico).

Dos bloques:

* **Cortes por cesta (M3)**: que el corte de cabecera no cambia, que los modos de varios
  cortes generan exactamente los `k` esperados con una clave de query por corte, que cada
  corte parte la cesta sin solaparse y que la sesion se corta en el instante de **ese**
  corte. Tambien, que la muestra por cesta y el filtro de cold-start eligen lo que dicen.
* **Incertidumbre (M4)**: el bootstrap por cesta y el pareado, contra casos cuyo
  resultado se conoce sin simular nada.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.recommender import candidates as cand
from src.recommender import evaluate as ev
from src.recommender import features as feat
from src.recommender import splits
from src.recommender.config import (
    CUT_ALL_PREFIXES,
    CUT_EMPTY_AND_HALF,
    CUT_RANDOM_FRACTIONS,
    BootstrapConfig,
    CandidateConfig,
    CutPlan,
)
from tests.helpers import spark_df
from tests.test_recommender import ITEMS_DDL, toy  # noqa: F401  (fixture)

TEST_START = dt.date(2025, 3, 1)


def _window(toy):  # noqa: F811
    return splits.baskets_between(toy["baskets"], TEST_START)


def _history_customers(toy):  # noqa: F811
    return splits.known_customers(splits.baskets_before(toy["baskets"], TEST_START))


def _cuts(query_items) -> dict[str, set[int]]:
    rows = query_items.select("source_basket_id", "prefix_size").distinct().toPandas()
    return rows.groupby("source_basket_id")["prefix_size"].apply(set).to_dict()


# --------------------------------------------------------------------------------------
# Cortes por cesta (M3)
# --------------------------------------------------------------------------------------
def test_el_corte_de_cabecera_no_cambia(toy) -> None:  # noqa: F811
    """Un corte por cesta, con la clave de siempre: la serie historica no se rompe."""
    items = splits.build_query_items(_window(toy), toy["basket_items"]).toPandas()

    assert (items["basket_id"] == items["source_basket_id"]).all()
    assert items.groupby("basket_id")["prefix_size"].nunique().eq(1).all()
    # B3 tiene 4 lineas y su hash la manda a "mitad del ticket" (lo fija tambien
    # `test_la_sesion_se_corta_en_el_instante_del_prefijo`).
    assert items.loc[items["basket_id"] == "B3", "prefix_size"].iloc[0] == 2


def test_todos_los_prefijos_dan_un_corte_por_cada_k(toy) -> None:  # noqa: F811
    plan = CutPlan(mode=CUT_ALL_PREFIXES)
    items = splits.build_query_items(_window(toy), toy["basket_items"], cuts=plan)

    assert _cuts(items) == {"B3": {1, 2, 3}, "B4": {1}, "B5": {1}}
    rows = items.toPandas()
    esperadas = {f"B3#k{k}" for k in (1, 2, 3)} | {"B4#k1", "B5#k1"}
    assert set(rows["basket_id"]) == esperadas

    for key, group in rows.groupby("basket_id"):
        k = int(group["prefix_size"].iloc[0])
        prefix = set(group.loc[group["is_prefix"], "product_id"])
        target = set(group.loc[~group["is_prefix"], "product_id"])
        assert len(prefix) == k, key
        assert not prefix & target, key
        assert target, f"{key}: siempre queda algo que adivinar"
        # El prefijo son las k primeras posiciones del mismo orden en todos los cortes.
        assert set(group.loc[group["position"] <= k, "product_id"]) == prefix


def test_las_cestas_de_una_linea_solo_admiten_el_carrito_vacio(spark) -> None:
    baskets = splits.baskets_between(
        spark_df(
            spark,
            [
                {"basket_id": "U1", "customer_id": None, "channel": "tienda",
                 "basket_date": "2025-03-05 10:00:00", "basket_day": "2025-03-05",
                 "total_amount": 3.0},
            ],
            "basket_id string, customer_id string, channel string, basket_date timestamp, "
            "basket_day date, total_amount double",
        ),
        TEST_START,
    )
    items = spark_df(spark, [{"basket_id": "U1", "product_id": "P1", "quantity": 1}], ITEMS_DDL)

    for mode in (CUT_ALL_PREFIXES, CUT_RANDOM_FRACTIONS):
        assert splits.build_query_items(baskets, items, cuts=CutPlan(mode=mode)).count() == 0
    vacio = splits.build_query_items(baskets, items, cuts=CutPlan(mode=CUT_EMPTY_AND_HALF))
    assert _cuts(vacio) == {"U1": {0}}


def test_carrito_vacio_y_mitad_da_los_dos_cortes_de_cabecera(toy) -> None:  # noqa: F811
    plan = CutPlan(mode=CUT_EMPTY_AND_HALF)
    items = splits.build_query_items(_window(toy), toy["basket_items"], cuts=plan)
    assert _cuts(items) == {"B3": {0, 2}, "B4": {0, 1}, "B5": {0, 1}}


def test_las_fracciones_aleatorias_son_deterministas(toy) -> None:  # noqa: F811
    plan = CutPlan(mode=CUT_RANDOM_FRACTIONS, n_fractions=5, seed=7)
    first = _cuts(splits.build_query_items(_window(toy), toy["basket_items"], cuts=plan))
    second = _cuts(splits.build_query_items(_window(toy), toy["basket_items"], cuts=plan))

    assert first == second
    assert first["B4"] == {1} and first["B5"] == {1}  # con dos lineas solo cabe k = 1
    assert first["B3"] <= {1, 2, 3} and 1 <= len(first["B3"]) <= 5


def test_cada_corte_es_una_query_con_su_perfil_y_su_instante(toy) -> None:  # noqa: F811
    """El perfil y `cut_ts` son del corte, no de la cesta."""
    window = _window(toy)
    add_to_cart = splits.basket_add_to_cart(toy["sessions"], toy["session_events"])
    items = splits.build_query_items(
        window, toy["basket_items"], add_to_cart, cuts=CutPlan(mode=CUT_EMPTY_AND_HALF)
    )
    queries = splits.build_queries(
        window, items, _history_customers(toy), toy["sessions"]
    ).toPandas().set_index("basket_id")

    assert set(queries.index) == {"B3#k0", "B3#k2", "B4#k0", "B4#k1", "B5#k0", "B5#k1"}
    assert (queries["source_basket_id"] == queries.index.str.split("#").str[0]).all()
    assert queries.loc["B3#k0", "profile"] == 3
    assert queries.loc["B3#k2", "profile"] == 4
    assert queries.loc["B4#k0", "profile"] == 1
    assert queries.loc["B5#k1", "profile"] == 2
    assert (queries["n_target"] == queries["n_items"] - queries["prefix_size"]).all()

    # Con el carrito vacio el corte es el inicio de la sesion; con dos lineas, el anadido
    # de P2. La vista de P3 (17:32:30) solo cabe en el segundo.
    assert queries.loc["B3#k0", "cut_ts"] == pd.Timestamp("2025-03-10 17:30:00")
    assert queries.loc["B3#k2", "cut_ts"] == pd.Timestamp("2025-03-10 17:34:00")
    events = feat.session_events_before_cut(
        splits.build_queries(window, items, _history_customers(toy), toy["sessions"]),
        toy["sessions"],
        toy["session_events"],
    ).toPandas()
    vistos = events.loc[events["event_type"] == "view"].groupby("basket_id")["product_id"].apply(set)
    assert "B3#k0" not in vistos.index
    assert vistos["B3#k2"] == {"P1", "P2", "P3", "P9"}


def test_prefijo_y_target_por_corte(toy) -> None:  # noqa: F811
    window = _window(toy)
    items = splits.build_query_items(
        window, toy["basket_items"], cuts=CutPlan(mode=CUT_ALL_PREFIXES)
    )
    queries = splits.build_queries(window, items, _history_customers(toy), toy["sessions"])
    prefix, target = splits.prefix_and_target(items, queries)

    n_prefix = prefix.groupBy("basket_id").count().toPandas().set_index("basket_id")["count"]
    n_target = target.groupBy("basket_id").count().toPandas().set_index("basket_id")["count"]
    assert n_prefix.to_dict() == {"B3#k1": 1, "B3#k2": 2, "B3#k3": 3, "B4#k1": 1, "B5#k1": 1}
    assert n_target.to_dict() == {"B3#k1": 3, "B3#k2": 2, "B3#k3": 1, "B4#k1": 1, "B5#k1": 1}


def test_la_muestra_por_cesta_sigue_el_orden_de_la_cabecera(toy) -> None:  # noqa: F811
    """Las n primeras cestas por hash son las mismas que elige `build_queries`."""
    window = _window(toy)
    items = splits.build_query_items(window, toy["basket_items"])
    for n in (1, 2, 3):
        sampled = {
            r["basket_id"]
            for r in splits.sample_baskets(window, n, salt="test").select("basket_id").collect()
        }
        headline = {
            r["basket_id"]
            for r in splits.build_queries(
                window, items, _history_customers(toy), toy["sessions"], n_queries=n, salt="test"
            )
            .select("basket_id")
            .collect()
        }
        assert sampled == headline
    assert splits.sample_baskets(window, None, salt="test").count() == 3


def test_el_cold_start_son_las_cestas_sin_historial(toy) -> None:  # noqa: F811
    nuevas = splits.new_customer_baskets(_window(toy), _history_customers(toy))
    assert {r["basket_id"] for r in nuevas.collect()} == {"B4", "B5"}


def test_sin_clientes_conocidos_el_als_no_propone_nada(toy) -> None:  # noqa: F811
    """Todo el lote en cold-start: la fuente ALS devuelve un vacio con su esquema.

    Es el caso del cold-start sobremuestreado. El vacio tiene que construirse en la JVM:
    un `createDataFrame([])` levanta un worker de Python que en Windows casca.
    """
    window = splits.new_customer_baskets(_window(toy), _history_customers(toy))
    items = splits.build_query_items(window, toy["basket_items"])
    queries = splits.build_queries(window, items, _history_customers(toy), toy["sessions"])
    index = queries.sparkSession.createDataFrame(
        pd.DataFrame({"customer_id": ["C1"], "customer_id_idx": [0]})
    )

    out = cand.candidates_als(queries, None, index, index, cfg=CandidateConfig())
    assert out.count() == 0
    assert out.columns == ["basket_id", "product_id", "als_score", "als_rank"]
    assert dict(out.dtypes)["als_rank"] == "int"


def test_el_modo_de_corte_se_valida() -> None:
    with pytest.raises(ValueError):
        CutPlan(mode="siguiente_articulo")
    with pytest.raises(ValueError):
        CutPlan(mode=CUT_RANDOM_FRACTIONS, n_fractions=0)


def test_el_desglose_por_corte_agrupa_prefijos_y_fracciones() -> None:
    per_query = pd.DataFrame({"prefix_size": [1, 2, 10, 12, 3], "n_items": [4, 4, 11, 13, 4]})
    by_size, by_fraction = ev.cut_groups(per_query, max_prefix=9)

    assert by_size["1"].tolist() == [True, False, False, False, False]
    assert by_size["10+"].tolist() == [False, False, True, True, False]
    # 1/4 = 25 % cae en el primer tramo (cerrado por la derecha); 3/4 en el tercero.
    assert by_fraction["(0%, 25%]"].tolist() == [True, False, False, False, False]
    assert by_fraction["(50%, 75%]"].tolist() == [False, False, False, False, True]
    assert sum(m.sum() for m in by_fraction.values()) == len(per_query)


# --------------------------------------------------------------------------------------
# Incertidumbre (M4)
# --------------------------------------------------------------------------------------
BOOT = BootstrapConfig(n_resamples=400, seed=11)


def test_el_bootstrap_de_una_constante_no_tiene_anchura() -> None:
    frame = pd.DataFrame({"x": np.full(50, 0.3)})
    ci = ev.bootstrap_means(frame, {"x": "x"}, BOOT)
    assert ci["x_ci_low"] == pytest.approx(0.3)
    assert ci["x_ci_high"] == pytest.approx(0.3)


def test_el_bootstrap_remuestrea_cestas_no_queries() -> None:
    """Duplicar cada query dentro de su cesta no aporta informacion: el IC no cambia.

    Tratar los duplicados como independientes estrecharia el intervalo, que es justo el
    error que evita remuestrear por cesta cuando hay varios cortes.
    """
    rng = np.random.default_rng(3)
    values = (rng.random(300) < 0.4).astype(float)
    single = ev.bootstrap_replicates(values, BOOT)
    doubled = ev.bootstrap_replicates(
        np.repeat(values, 2), BOOT, clusters=np.repeat(np.arange(300), 2)
    )
    np.testing.assert_allclose(single, doubled)

    naive = ev.bootstrap_replicates(np.repeat(values, 2), BOOT)
    assert naive.std() < 0.85 * single.std()


def test_el_intervalo_cubre_la_media_y_tiene_la_anchura_esperada() -> None:
    rng = np.random.default_rng(5)
    hits = (rng.random(2_000) < 0.6).astype(float)
    frame = pd.DataFrame({"hit": hits})
    ci = ev.bootstrap_means(frame, {"hit": "hit_rate"}, BootstrapConfig(n_resamples=2_000))
    se = np.sqrt(hits.mean() * (1 - hits.mean()) / len(hits))
    assert ci["hit_rate_ci_low"] < hits.mean() < ci["hit_rate_ci_high"]
    assert ci["hit_rate_ci_high"] - ci["hit_rate_ci_low"] == pytest.approx(2 * 1.96 * se, rel=0.1)


def _per_query(values: np.ndarray, profiles: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "basket_id": [f"B{i}" for i in range(len(values))],
            "profile": profiles,
            "hit": values,
        }
    )


def test_el_bootstrap_pareado_de_dos_sistemas_iguales_no_ve_diferencia() -> None:
    rng = np.random.default_rng(1)
    a = _per_query((rng.random(500) < 0.5).astype(float), rng.integers(1, 5, 500))
    out = ev.paired_bootstrap({"a": a, "b": a.copy()}, "a", {"hit": "hit_rate"}, BOOT)

    assert set(out["grupo"]) == {"total", *[ev.PROFILE_LABELS[p] for p in (1, 2, 3, 4)]}
    assert (out["diferencia"] == 0).all()
    assert (out["ci_low"] == 0).all() and (out["ci_high"] == 0).all()
    assert (out["p_value"] == 1).all()


def test_el_bootstrap_pareado_cancela_la_dificultad_de_cada_query() -> None:
    """Una mejora de +0,05 en todas las queries es segura, aunque las medias sean ruidosas."""
    rng = np.random.default_rng(2)
    base = rng.random(200)
    a = _per_query(base + 0.05, np.ones(200, dtype=int))
    b = _per_query(base, np.ones(200, dtype=int))
    out = ev.paired_bootstrap({"a": a, "b": b.sample(frac=1, random_state=0)}, "a",
                              {"hit": "m"}, BOOT, by_profile=False)
    row = out.iloc[0]

    assert row["diferencia"] == pytest.approx(0.05)
    assert row["ci_low"] == pytest.approx(0.05) and row["ci_high"] == pytest.approx(0.05)
    assert row["p_value"] == pytest.approx(1 / (BOOT.n_resamples + 1))
    assert row["media_referencia"] - row["media_sistema"] == pytest.approx(0.05)


def test_el_bootstrap_pareado_exige_las_mismas_queries() -> None:
    a = _per_query(np.ones(10), np.ones(10, dtype=int))
    with pytest.raises(ValueError):
        ev.paired_bootstrap({"a": a, "b": a.iloc[:9]}, "a", {"hit": "m"}, BOOT)


def test_los_intervalos_no_mueven_las_cifras_puntuales() -> None:
    rng = np.random.default_rng(4)
    n = 400
    per_query = pd.DataFrame(
        {
            "basket_id": [f"B{i}" for i in range(n)],
            "profile": rng.integers(1, 5, n),
            "n_target": rng.integers(1, 8, n),
            "ndcg": rng.random(n),
            "recall": rng.random(n),
            "precision": rng.random(n) / 2,
            "f1": rng.random(n),
            "hit": (rng.random(n) < 0.5).astype(float),
        }
    )
    plain = ev.summarise(per_query, k=5)
    with_ci = ev.summarise(per_query, k=5, bootstrap=BOOT)

    pd.testing.assert_frame_equal(plain, with_ci[plain.columns])
    for metric in ("ndcg@5", "recall@5", "precision@5", "f1@5", "f1@5_por_cesta", "hit_rate@5"):
        assert (with_ci[metric + ev.CI_LOW] <= with_ci[metric]).all(), metric
        assert (with_ci[metric] <= with_ci[metric + ev.CI_HIGH]).all(), metric

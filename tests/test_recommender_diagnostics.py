"""Tests del diagnostico del recomendador: baselines independientes del pool y oraculo.

No necesitan Spark ni el dataset: los baselines y el oraculo son funciones puras sobre
pandas/numpy, y aqui se ejercitan con tablas de juguete cuyo resultado se sabe a mano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.recommender import evaluate as ev
from src.recommender import oracle as orc
from src.recommender import pipeline as pl


# --------------------------------------------------------------------------------------
# Oraculo
# --------------------------------------------------------------------------------------
WEIGHTS = np.array([5.0, 1.0, 0.3, 2.0, 0.5, 1.5])
# Disparadora poco probable (2) con lift fuerte, y otra muy probable (0).
PAIRS = ((2, 4, 6.0), (0, 1, 3.0))


def _brute_force_inclusion(prefix: tuple[int, ...], n_target: int, n: int, seed: int):
    """El proceso del generador tal cual: sorteo secuencial con lift, y rechazo por prefijo."""
    rng = np.random.default_rng(seed)
    c = WEIGHTS.size
    w = np.tile(WEIGHTS, (n, 1))
    drawn = np.zeros((n, c), dtype=bool)
    rows = np.arange(n)
    for _ in range(len(prefix) + n_target):
        cum = w.cumsum(axis=1)
        pick = np.minimum((cum < rng.random(n)[:, None] * cum[:, -1:]).sum(axis=1), c - 1)
        drawn[rows, pick] = True
        w[rows, pick] = 0.0
        for trigger, associated, lift in PAIRS:
            w[pick == trigger, associated] *= lift
    accepted = drawn[:, list(prefix)].all(axis=1)
    rest = drawn[accepted]
    rest[:, list(prefix)] = False
    p = rest.mean(axis=0)
    return p, np.sqrt(p * (1 - p) / accepted.sum())


@pytest.mark.parametrize(
    "prefix, n_target",
    [((), 2), ((1,), 2), ((2,), 2), ((4,), 2), ((2, 1), 1)],
    ids=["sin-prefijo", "asociada", "disparadora", "asociada-de-poca", "ambas"],
)
def test_la_carrera_reproduce_el_sorteo_secuencial(prefix, n_target) -> None:
    """P(categoria en el resto | prefijo) coincide con simular el generador y rechazar."""
    expected, se_bf = _brute_force_inclusion(prefix, n_target, n=600_000, seed=1)

    mask = np.zeros((1, WEIGHTS.size), dtype=bool)
    mask[0, list(prefix)] = True
    draws = np.random.default_rng(7).standard_exponential((1, 200_000, WEIGHTS.size))
    in_rest, log_w, _ = orc.race(WEIGHTS[None], mask, np.array([n_target]), PAIRS, draws)
    w = orc._normalise(log_w)
    got = (w[:, :, None] * in_rest).sum(axis=1)[0]

    ess = 1.0 / (w**2).sum()
    se = np.sqrt(se_bf**2 + got * (1 - got) / ess)
    assert np.all(np.abs(got - expected) <= 5 * se + 1e-9), (got, expected)
    assert np.all(got[list(prefix)] == 0)


def test_los_pesos_ajustados_quitan_el_prefijo_y_aplican_el_lift() -> None:
    mask = np.array([[False, False, True, False, False, False]])
    adjusted = orc.adjusted_weights(WEIGHTS[None], mask, PAIRS)[0]
    assert adjusted[2] == 0.0
    assert adjusted[4] == pytest.approx(0.5 * 6.0)
    assert adjusted[1] == pytest.approx(1.0)  # su disparadora no esta en el carrito


def test_top_k_respeta_lo_permitido_y_marca_huecos() -> None:
    scores = np.array([[0.1, 0.9, 0.0, 0.5]])
    allowed = np.array([[True, False, True, True]])
    assert orc.top_k_categories(scores, allowed, 3).tolist() == [[3, 0, -1]]


def _toy_inputs(prefix: np.ndarray, target: np.ndarray) -> orc.OracleInputs:
    q = prefix.shape[0]
    return orc.OracleInputs(
        basket_ids=np.array([f"B{i}" for i in range(q)]),
        categories=[f"c{i}" for i in range(WEIGHTS.size)],
        weights=np.tile(WEIGHTS, (q, 1)),
        best_product=np.array([[f"P{i}" for i in range(WEIGHTS.size)]] * q, dtype=object),
        best_prob=np.full((q, WEIGHTS.size), 0.5),
        prefix_mask=prefix,
        target_mask=target,
        pairs=PAIRS,
    )


def test_con_k_mayor_que_el_resto_el_oraculo_acierta_seguro() -> None:
    """Si la lista cubre todas las categorias posibles, el acierto esperado es 1."""
    prefix = np.zeros((1, WEIGHTS.size), dtype=bool)
    target = np.zeros((1, WEIGHTS.size), dtype=bool)
    target[0, [0, 3]] = True
    mc = orc.monte_carlo(_toy_inputs(prefix, target), k=6, n_samples=200)
    assert mc.cat_hit[0] == pytest.approx(1.0)
    # Con 6 huecos y 2 categorias en el resto, la precision esperada es 2/6.
    assert mc.cat_precision[0] == pytest.approx(2 / 6)
    assert mc.n_fallback == 0


def test_la_lista_del_oraculo_se_concreta_en_su_mejor_referencia() -> None:
    prefix = np.zeros((1, WEIGHTS.size), dtype=bool)
    target = np.zeros((1, WEIGHTS.size), dtype=bool)
    inputs = _toy_inputs(prefix, target)
    lists = np.array([[0, 3, -1]])
    truth = pd.DataFrame({"basket_id": ["B0"], "product_id": ["P3"]})
    top = orc.lists_to_top_k(inputs, lists, truth)
    assert top["product_id"].tolist() == ["P0", "P3"]
    assert top["label"].tolist() == [0, 1]


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------
def _baseline_inputs(**overrides) -> ev.BaselineInputs:
    """Cuatro categorias con dos referencias cada una y tres queries.

    - Q1: cliente C1, carrito vacio.
    - Q2: cliente C1, lleva `leche`.
    - Q3: anonimo, lleva `pan`.
    """
    products = pd.DataFrame(
        {
            "product_id": ["L1", "L2", "P1", "P2", "Y1", "Y2", "Z1", "Z2"],
            "category": ["leche", "leche", "pan", "pan", "yogur", "yogur", "zumo", "zumo"],
        }
    )
    data = dict(
        queries=pd.DataFrame(
            {
                "basket_id": ["Q1", "Q2", "Q3"],
                "customer_id": ["C1", "C1", None],
                "basket_day": pd.to_datetime(["2025-11-10", "2025-11-10", "2025-11-10"]),
            }
        ),
        prefix=pd.DataFrame({"basket_id": ["Q2", "Q3"], "product_id": ["L1", "P2"]}),
        target=pd.DataFrame(
            {"basket_id": ["Q1", "Q1", "Q2", "Q3"], "product_id": ["Z2", "Y1", "P1", "Z2"]}
        ),
        products=products,
        product_popularity=pd.DataFrame(
            {
                "product_id": ["L1", "L2", "P1", "P2", "Y1", "Y2", "Z1", "Z2"],
                "n_baskets": [90.0, 10.0, 70.0, 5.0, 20.0, 30.0, 8.0, 1.0],
            }
        ),
        category_popularity=pd.DataFrame(
            {"category": ["leche", "pan", "yogur", "zumo"], "n_baskets": [100.0, 75.0, 50.0, 9.0]}
        ),
        customer_products=pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C1"],
                "product_id": ["Z2", "Y1", "Y2"],
                "n_baskets": [6.0, 2.0, 1.0],
            }
        ),
        customer_categories=pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "category": ["zumo", "yogur"],
                "n_purchase_days": [6, 2],
                "last_purchase_date": ["2025-11-08", "2025-10-01"],
                "expected_repurchase_days": [7.0, 7.0],
            }
        ),
        category_rules=pd.DataFrame(
            {
                "antecedent": ["pan", "pan", "leche"],
                "consequent": ["zumo", "yogur", "pan"],
                "confidence": [0.30, 0.05, 0.60],
                "lift": [2.0, 3.0, 0.9],
            }
        ),
    )
    data.update(overrides)
    return ev.BaselineInputs(**data)


def _lists(top_k: pd.DataFrame) -> dict[str, list[str]]:
    return top_k.sort_values("rank").groupby("basket_id")["product_id"].apply(list).to_dict()


@pytest.fixture(scope="module")
def baselines() -> dict[str, pd.DataFrame]:
    return ev.run_baselines(_baseline_inputs(), k=3, seed=42, min_confidence=0.10)


def test_ningun_baseline_recomienda_una_categoria_del_carrito(baselines) -> None:
    cats = _baseline_inputs().products.set_index("product_id")["category"]
    for name, top in baselines.items():
        recs = _lists(top)
        assert not {cats[p] for p in recs["Q2"]} & {"leche"}, name
        assert not {cats[p] for p in recs["Q3"]} & {"pan"}, name


def test_popularidad_de_categoria_con_su_referencia_lider(baselines) -> None:
    recs = _lists(baselines["category_popularity"])
    assert recs["Q1"] == ["L1", "P1", "Y2"]
    assert recs["Q3"] == ["L1", "Y2", "Z1"]


def test_popularidad_global_puede_repetir_categoria(baselines) -> None:
    assert _lists(baselines["global_popularity"])["Q1"] == ["L1", "P1", "Y2"]
    assert _lists(baselines["global_popularity"])["Q2"] == ["P1", "Y2", "Y1"]


def test_frecuencia_personal_con_referencia_favorita_y_relleno(baselines) -> None:
    recs = _lists(baselines["personal_frequency"])
    # zumo (6 dias) > yogur (2) > relleno por popularidad (leche).
    assert recs["Q1"] == ["Z2", "Y1", "L1"]
    # El anonimo no tiene historial: se queda en popularidad de categoria.
    assert recs["Q3"] == _lists(baselines["category_popularity"])["Q3"]


def test_due_for_repurchase_adelanta_lo_que_toca_reponer(baselines) -> None:
    # zumo se compro hace 2 dias (no toca): 6. yogur hace 40 dias (toca): 2 x 2 = 4.
    assert _lists(baselines["personal_due"])["Q1"] == ["Z2", "Y1", "L1"]
    inputs = _baseline_inputs(
        customer_categories=pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "category": ["zumo", "yogur"],
                "n_purchase_days": [3, 2],
                "last_purchase_date": ["2025-11-08", "2025-10-01"],
                "expected_repurchase_days": [7.0, 7.0],
            }
        )
    )
    # Ahora yogur (2 x 2 = 4) supera a zumo (3).
    recs = _lists(ev.run_baselines(inputs, k=3, seed=42, min_confidence=0.1)["personal_due"])
    assert recs["Q1"][:2] == ["Y1", "Z2"]


def test_repetir_favoritas_ordena_por_veces_compradas(baselines) -> None:
    assert _lists(baselines["repeat_favorite"])["Q1"] == ["Z2", "Y1", "Y2"]
    assert _lists(baselines["repeat_favorite"])["Q3"] == ["L1", "Y2", "Y1"]


def test_reglas_de_asociacion_con_umbral_de_confianza_y_lift(baselines) -> None:
    recs = _lists(baselines["association_rules"])
    # pan -> zumo pasa (0,30 y lift 2); pan -> yogur no llega a la confianza minima.
    assert recs["Q3"][0] == "Z1"
    # leche -> pan tiene lift < 1: no cuenta, asi que Q2 es popularidad pura.
    assert recs["Q2"] == ["P1", "Y2", "Z1"]
    # Sin carrito no hay reglas que disparar.
    assert recs["Q1"] == _lists(baselines["category_popularity"])["Q1"]


def test_aleatorio_es_reproducible_con_la_semilla() -> None:
    a = ev.run_baselines(_baseline_inputs(), k=3, seed=1, min_confidence=0.1)["random"]
    b = ev.run_baselines(_baseline_inputs(), k=3, seed=1, min_confidence=0.1)["random"]
    pd.testing.assert_frame_equal(a, b)


def test_las_etiquetas_marcan_el_sku_exacto_del_target(baselines) -> None:
    top = baselines["personal_frequency"]
    labelled = set(top.loc[top["label"] == 1, ["basket_id", "product_id"]].itertuples(index=False))
    # Q2 acierta con el relleno: pan no esta en su historial, pero es su lider.
    assert labelled == {("Q1", "Z2"), ("Q1", "Y1"), ("Q2", "P1")}


def test_resumen_de_sistema_junta_sku_y_categoria(baselines) -> None:
    inputs = _baseline_inputs()
    queries = inputs.queries.assign(profile=[3, 4, 2], n_target=[2, 1, 1])
    summary = ev.system_summary(
        baselines["personal_frequency"], queries, inputs.target, inputs.products, k=3
    )
    total = summary.iloc[0]
    assert total["grupo"] == "total"
    # Q1 y Q2 aciertan el SKU; Q3 acierta la categoria (zumo) con otra referencia.
    assert total["sku_hit_rate@3"] == pytest.approx(2 / 3)
    assert total["cat_hit_rate@3"] == pytest.approx(1.0)
    assert total["precision@3"] == pytest.approx((2 + 1 + 0) / 3 / 3)


# --------------------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------------------
def test_el_bloque_de_diagnostico_sobrevive_a_reescribir_metrics_md() -> None:
    block = f"{pl.DIAGNOSTICS_START}\n## Diagnostico\nv1\n{pl.DIAGNOSTICS_END}"
    first = pl.with_diagnostics("# Informe\n", block)
    assert first.endswith(block + "\n")

    # El pipeline regenera el informe y conserva el bloque que hubiera.
    regenerated = pl.with_diagnostics("# Informe nuevo\n", pl.extract_diagnostics(first))
    assert regenerated.startswith("# Informe nuevo")
    assert pl.extract_diagnostics(regenerated) == block

    # El verificador sustituye el bloque en su sitio, sin duplicarlo.
    updated = pl.with_diagnostics(regenerated, block.replace("v1", "v2"))
    assert updated.count(pl.DIAGNOSTICS_START) == 1
    assert "v2" in updated and "v1" not in updated

    # Sin bloque previo, el informe queda igual.
    assert pl.with_diagnostics("# Informe\n", "") == "# Informe\n"

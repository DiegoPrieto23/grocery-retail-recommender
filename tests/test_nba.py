"""Tests del Next Best Action (Fase 4).

Cuatro bloques, y los dos primeros son los que sostienen la credibilidad de las cifras:

* **Fuga temporal**: que las features de un corte no vean ni un dia posterior, que las
  etiquetas solo miren la ventana futura, y que `churn_label` de `DATA_SPEC.md` -- que se
  calcula con compras del final del dataset -- no se cuele como feature.
* **Etiquetas**: que el churn sea "no comprar" y no lo contrario, y que el lookback de
  categorias acote de verdad los pares candidatos.
* **Politica**: la aritmetica del valor esperado comprobada a mano, incluida la asimetria
  entre coste de envio (siempre) y descuento (solo si compra).
* **Metricas**: AUC y PR-AUC contra casos triviales.

Los DataFrames son minusculos y estan escritos a mano, igual que en `test_recommender.py`.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.nba import features as feat
from src.nba import policy as pol
from src.nba import propensity as prop
from src.nba import targets as tgt
from src.nba.config import ACTION_BY_NAME, NBAConfig
from tests.helpers import spark_df

# Esquemas de las tablas **ya limpias** (`data/processed`), que es lo que consume la fase.
BASKETS_DDL = (
    "basket_id string, customer_id string, channel string, basket_date timestamp, "
    "basket_day date, total_amount double"
)
ITEMS_DDL = (
    "basket_id string, product_id string, quantity int, unit_price_paid double, "
    "promotion_id string, line_amount double, is_promo boolean"
)
PRODUCTS_DDL = (
    "product_id string, department string, category string, brand string, "
    "is_private_label boolean, is_perishable boolean, unit_price double, "
    "pack_size int, typical_repurchase_days int"
)
CUSTOMERS_DDL = (
    "customer_id string, signup_date date, country string, city string, "
    "household_size_est int, loyalty_tier string, preferred_channel string, "
    "churn_label boolean"
)

CUTOFF = dt.date(2025, 6, 1)


def _basket(basket_id: str, customer_id: str | None, day: str, amount: float = 30.0) -> dict:
    return {
        "basket_id": basket_id,
        "customer_id": customer_id,
        "channel": "web",
        "basket_date": f"{day} 18:00:00",
        "basket_day": day,
        "total_amount": amount,
    }


def _item(basket_id: str, product_id: str, amount: float = 10.0) -> dict:
    return {
        "basket_id": basket_id,
        "product_id": product_id,
        "quantity": 1,
        "unit_price_paid": amount,
        "promotion_id": None,
        "line_amount": amount,
        "is_promo": False,
    }


def _product(product_id: str, category: str, department: str = "Despensa") -> dict:
    return {
        "product_id": product_id,
        "department": department,
        "category": category,
        "brand": "Marca",
        "is_private_label": False,
        "is_perishable": False,
        "unit_price": 10.0,
        "pack_size": 1,
        "typical_repurchase_days": 30,
    }


def _customer(customer_id: str, churn_label: bool = False) -> dict:
    return {
        "customer_id": customer_id,
        "signup_date": "2024-01-01",
        "country": "Espana",
        "city": "Madrid",
        "household_size_est": 2,
        "loyalty_tier": "silver",
        "preferred_channel": "web",
        "churn_label": churn_label,
    }


@pytest.fixture
def toy(spark):
    """Dos clientes: C1 sigue comprando tras el corte, C2 desaparece."""
    baskets = spark_df(
        spark,
        [
            # C1: compra antes y despues del corte.
            _basket("B1", "C1", "2025-05-01"),
            _basket("B2", "C1", "2025-05-20"),
            _basket("B3", "C1", "2025-06-03"),  # dentro de la ventana de 7 dias
            # C2: solo antes. Es el churner.
            _basket("B4", "C2", "2025-04-10"),
            _basket("B5", "C2", "2025-05-15"),
            # Una cesta anonima, que no debe contar para nada.
            _basket("B6", None, "2025-05-11"),
        ],
        BASKETS_DDL,
    )
    items = spark_df(
        spark,
        [
            _item("B1", "P1"),
            _item("B2", "P2"),
            _item("B3", "P1"),
            _item("B4", "P1"),
            _item("B5", "P2"),
            _item("B6", "P1"),
        ],
        ITEMS_DDL,
    )
    products = spark_df(
        spark,
        [_product("P1", "Leche"), _product("P2", "Pasta")],
        PRODUCTS_DDL,
    )
    customers = spark_df(
        spark, [_customer("C1"), _customer("C2", churn_label=True)], CUSTOMERS_DDL
    )
    return {
        "baskets": baskets,
        "basket_items": items,
        "products": products,
        "customers": customers,
    }


# --------------------------------------------------------------------------------------
# Fuga temporal
# --------------------------------------------------------------------------------------
def test_churn_label_de_data_spec_no_es_una_feature() -> None:
    """Se calcula con compras del final del dataset: como feature seria fuga pura."""
    assert "churn_label" not in feat.CHURN_FEATURES
    assert "churn_label" not in feat.PURCHASE_FEATURES


def test_las_features_no_cambian_si_se_anaden_compras_tras_el_corte(toy, spark) -> None:
    """Anadir una compra posterior al corte no puede mover ninguna feature."""
    history = tgt.baskets_in_window(toy["baskets"], None, CUTOFF)
    lines = tgt.category_lines(history, toy["basket_items"], toy["products"])
    before = (
        feat.customer_features(history, lines, toy["customers"], CUTOFF)
        .orderBy("customer_id")
        .toPandas()
    )

    extra = toy["baskets"].unionByName(
        spark_df(spark, [_basket("B99", "C1", "2025-12-31", amount=999.0)], BASKETS_DDL)
    )
    history2 = tgt.baskets_in_window(extra, None, CUTOFF)
    lines2 = tgt.category_lines(history2, toy["basket_items"], toy["products"])
    after = (
        feat.customer_features(history2, lines2, toy["customers"], CUTOFF)
        .orderBy("customer_id")
        .toPandas()
    )

    pd.testing.assert_frame_equal(before, after)


def test_las_cestas_anonimas_no_entran_en_ningun_grano(toy) -> None:
    """`B6` no tiene `customer_id`: no puede generar ni feature ni etiqueta."""
    active = tgt.active_customers(toy["baskets"], CUTOFF).toPandas()
    assert sorted(active["customer_id"]) == ["C1", "C2"]


# --------------------------------------------------------------------------------------
# Etiquetas
# --------------------------------------------------------------------------------------
def test_el_churn_marca_al_que_no_compra_y_no_al_reves(toy) -> None:
    churn = (
        tgt.churn_label(toy["baskets"], CUTOFF, horizon_days=28)
        .orderBy("customer_id")
        .toPandas()
    )
    assert dict(zip(churn["customer_id"], churn["churn"])) == {"C1": 0, "C2": 1}


def test_la_etiqueta_de_categoria_solo_mira_la_ventana_futura(toy) -> None:
    """C1 compra Leche el 3 de junio: entra a 7 dias. Pasta, que compro antes, no."""
    lines = tgt.category_lines(toy["baskets"], toy["basket_items"], toy["products"])
    positives = tgt.category_label(lines, CUTOFF, horizon_days=7).toPandas()
    pairs = {(r.customer_id, r.category) for r in positives.itertuples()}
    assert pairs == {("C1", "Leche")}


def test_una_compra_fuera_del_horizonte_no_cuenta(toy) -> None:
    """Con horizonte de 1 dia, la compra del 3 de junio queda fuera."""
    lines = tgt.category_lines(toy["baskets"], toy["basket_items"], toy["products"])
    positives = tgt.category_label(lines, CUTOFF, horizon_days=1).toPandas()
    assert positives.empty


def test_el_lookback_acota_los_pares_candidatos(toy) -> None:
    """Con 20 dias de lookback solo sobrevive lo comprado justo antes del corte."""
    lines = tgt.category_lines(toy["baskets"], toy["basket_items"], toy["products"])
    wide = tgt.candidate_pairs(lines, CUTOFF, lookback_days=180).toPandas()
    narrow = tgt.candidate_pairs(lines, CUTOFF, lookback_days=20).toPandas()

    assert len(wide) == 4  # C1 x {Leche, Pasta}, C2 x {Leche, Pasta}
    # En los 20 dias previos al 1 de junio solo hay B2 (C1/Pasta) y B5 (C2/Pasta).
    assert {(r.customer_id, r.category) for r in narrow.itertuples()} == {
        ("C1", "Pasta"),
        ("C2", "Pasta"),
    }


def test_los_pares_candidatos_llevan_su_etiqueta_a_cero_si_no_compran(toy) -> None:
    lines = tgt.category_lines(toy["baskets"], toy["basket_items"], toy["products"])
    category, churn = tgt.build_labels(
        toy["baskets"],
        lines,
        CUTOFF,
        category_horizon_days=7,
        churn_horizon_days=28,
        lookback_days=180,
    )
    got = {(r.customer_id, r.category): r.label for r in category.toPandas().itertuples()}
    assert got[("C1", "Leche")] == 1
    assert got[("C1", "Pasta")] == 0
    assert got[("C2", "Leche")] == 0
    assert len(churn.toPandas()) == 2


# --------------------------------------------------------------------------------------
# Politica de valor esperado
# --------------------------------------------------------------------------------------
def _candidates(**overrides) -> pd.DataFrame:
    """Un unico candidato, con los numeros elegidos para poder hacer la cuenta a mano."""
    row = {
        "customer_id": "C1",
        "category": "Pasta",
        "department": "Despensa",  # margen 0,25
        "p_purchase": 0.20,
        "expected_spend": 10.0,
        "gross_margin": 2.50,
        "p_churn": 0.0,
        "retention_value": 0.0,
        # Relevancia plena: la categoria ya le toca. Asi las cuentas a mano de abajo no
        # arrastran el factor, que tiene sus propios tests mas abajo.
        "relevance": 1.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_ninguna_accion_vale_exactamente_cero() -> None:
    value = pol.action_value(_candidates(), ACTION_BY_NAME["ninguna_accion"])
    assert value.iloc[0] == 0.0


def test_el_valor_de_recomendar_se_calcula_a_mano() -> None:
    """p=0,20 -> 0,22 con uplift 1,10; margen 2,50; envio 0,01.

    V(accion)  = 0,22 x 2,50 - 0,01 = 0,54
    V(ninguna) = 0,20 x 2,50        = 0,50
    delta      = 0,04
    """
    value = pol.action_value(_candidates(), ACTION_BY_NAME["recomendar_categoria"])
    assert value.iloc[0] == pytest.approx(0.04, abs=1e-9)


def test_el_descuento_del_cupon_solo_se_paga_si_el_cliente_compra() -> None:
    """El descuento entra dentro del parentesis, multiplicado por la probabilidad.

    Si se cobrara siempre, el valor seria 0,27 x 2,50 - 0,05 - 2,54 = -1,915.
    Al cobrarse solo si compra: 0,27 x (2,50 - 2,54) - 0,05 - 0,50 = -0,5608.
    """
    value = pol.action_value(_candidates(), ACTION_BY_NAME["enviar_cupon_categoria"])
    assert value.iloc[0] == pytest.approx(-0.5608, abs=1e-9)


def test_la_retencion_puede_rescatar_una_accion_que_pierde_en_margen() -> None:
    """Mismo candidato, pero un cliente que se esta yendo y vale mucho."""
    sin_riesgo = pol.action_value(
        _candidates(), ACTION_BY_NAME["enviar_cupon_categoria"]
    ).iloc[0]
    con_riesgo = pol.action_value(
        _candidates(p_churn=0.9, retention_value=20.0),
        ACTION_BY_NAME["enviar_cupon_categoria"],
    ).iloc[0]
    # 0,10 x 0,9 x 20 = 1,80 de retencion sobre los -0,5608 de margen.
    assert con_riesgo == pytest.approx(sin_riesgo + 1.80, abs=1e-9)
    assert sin_riesgo < 0 < con_riesgo


def test_la_probabilidad_no_pasa_de_uno_por_mucho_uplift() -> None:
    value = pol.action_value(
        _candidates(p_purchase=0.95), ACTION_BY_NAME["recomendar_categoria"]
    )
    # Con p=0,95 y uplift 1,10 la probabilidad se corta en 1,0, no sube a 1,045.
    esperado = 1.0 * 2.50 - 0.01 - 0.95 * 2.50
    assert value.iloc[0] == pytest.approx(esperado, abs=1e-9)


def test_decide_devuelve_exactamente_una_fila_por_cliente() -> None:
    cfg = NBAConfig()
    candidates = pd.concat(
        [
            _candidates(customer_id="C1", category="Pasta"),
            _candidates(customer_id="C1", category="Leche", p_purchase=0.40),
            _candidates(customer_id="C2", category="Pasta", p_purchase=0.05),
        ],
        ignore_index=True,
    )
    out = pol.decide(candidates, cfg)
    assert len(out) == 2
    assert sorted(out["customer_id"]) == ["C1", "C2"]


def test_si_ninguna_accion_compensa_la_politica_no_actua() -> None:
    """Un cliente con propension ridicula y sin riesgo de fuga no merece gasto."""
    cfg = NBAConfig()
    out = pol.decide(_candidates(p_purchase=0.0001), cfg)
    assert out["action"].iloc[0] == "ninguna_accion"
    assert out["expected_value"].iloc[0] == 0.0
    assert out["category"].isna().all()


def test_los_clientes_sin_candidatos_reciben_ninguna_accion_explicita() -> None:
    """La Tarea 3b pide una accion por cliente; quedarse fuera no es una respuesta."""
    cfg = NBAConfig()
    actions = pol.decide(_candidates(customer_id="C1"), cfg)
    completa = pol.cover_all(actions, pd.Series(["C1", "C2", "C3"]))

    assert sorted(completa["customer_id"]) == ["C1", "C2", "C3"]
    huerfanos = completa[completa["customer_id"].isin(["C2", "C3"])]
    assert (huerfanos["action"] == "ninguna_accion").all()
    assert (huerfanos["expected_value"] == 0.0).all()


def test_completar_no_duplica_a_quien_ya_estaba() -> None:
    cfg = NBAConfig()
    actions = pol.decide(_candidates(customer_id="C1"), cfg)
    assert len(pol.cover_all(actions, pd.Series(["C1"]))) == 1


def test_la_politica_nunca_pierde_frente_a_no_actuar() -> None:
    """Es una garantia estructural: `ninguna_accion` esta en el argmax y vale 0."""
    cfg = NBAConfig()
    rng = np.random.default_rng(0)
    candidates = pd.DataFrame(
        {
            "customer_id": [f"C{i // 3}" for i in range(60)],
            "category": ["Pasta", "Leche", "Cafe"] * 20,
            "department": ["Despensa"] * 60,
            "p_purchase": rng.uniform(0, 0.6, 60),
            "expected_spend": rng.uniform(1, 40, 60),
            "p_churn": rng.uniform(0, 1, 60),
            "retention_value": rng.uniform(0, 30, 60),
        }
    )
    candidates["gross_margin"] = candidates["expected_spend"] * 0.25
    candidates["relevance"] = 1.0
    comparison = pol.compare(candidates, cfg)

    no_actuar = comparison.loc[comparison["politica"] == "no actuar siempre", "valor_total"]
    politica = comparison.loc[
        comparison["politica"] == "politica de valor esperado", "valor_total"
    ]
    assert no_actuar.iloc[0] == 0.0
    assert politica.iloc[0] >= 0.0


def test_el_barrido_de_sensibilidad_es_monotono_en_el_uplift() -> None:
    """A mas efecto supuesto del cupon, mas valor y mas cupones repartidos."""
    cfg = NBAConfig()
    rng = np.random.default_rng(1)
    candidates = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(40)],
            "category": ["Pasta"] * 40,
            "department": ["Higiene"] * 40,
            "p_purchase": rng.uniform(0.05, 0.8, 40),
            "expected_spend": rng.uniform(5, 50, 40),
            "p_churn": rng.uniform(0, 1, 40),
            "retention_value": rng.uniform(0, 20, 40),
        }
    )
    candidates["gross_margin"] = candidates["expected_spend"] * 0.35
    candidates["relevance"] = 1.0
    sweep = pol.sensitivity(candidates, cfg)

    assert sweep["valor_politica"].is_monotonic_increasing
    assert sweep["pct_cupon"].is_monotonic_increasing


# --------------------------------------------------------------------------------------
# Metricas de propension
# --------------------------------------------------------------------------------------
def test_un_orden_perfecto_da_auc_uno() -> None:
    y = np.array([0, 0, 1, 1])
    m = prop.metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc"] == pytest.approx(1.0)
    assert m["base_rate"] == pytest.approx(0.5)


def test_un_orden_invertido_da_auc_cero() -> None:
    y = np.array([0, 0, 1, 1])
    m = prop.metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert m["auc"] == pytest.approx(0.0)


def test_con_una_sola_clase_las_metricas_son_nan_y_no_revientan() -> None:
    m = prop.metrics(np.zeros(10, dtype=int), np.linspace(0, 1, 10))
    assert np.isnan(m["auc"]) and np.isnan(m["pr_auc"])
    assert m["n"] == 10


def test_el_pr_auc_de_un_modelo_aleatorio_ronda_la_tasa_base() -> None:
    rng = np.random.default_rng(7)
    y = (rng.random(5_000) < 0.10).astype(int)
    m = prop.metrics(y, rng.random(5_000))
    assert m["pr_auc"] == pytest.approx(m["base_rate"], abs=0.03)
    assert m["lift_top_decile"] == pytest.approx(1.0, abs=0.35)


# --------------------------------------------------------------------------------------
# Contrato de features
# --------------------------------------------------------------------------------------
def test_las_columnas_de_features_no_se_repiten() -> None:
    assert len(set(feat.PURCHASE_FEATURES)) == len(feat.PURCHASE_FEATURES)
    assert len(set(feat.CHURN_FEATURES)) == len(feat.CHURN_FEATURES)


def test_las_categoricas_estan_declaradas_en_el_conjunto_de_features() -> None:
    for column in feat.CATEGORICAL_FEATURES:
        assert column in feat.PURCHASE_FEATURES


# --------------------------------------------------------------------------------------
# Capa comun de necesidad de categoria (punto M7)
# --------------------------------------------------------------------------------------
def test_la_relevancia_es_la_misma_formula_que_usa_el_recomendador() -> None:
    """No es una heuristica nueva del NBA: es `category_need_weight` de la Fase 3.

    Si alguien cambia la formula compartida, este test se entera. Es lo unico que ata las
    dos tareas a la misma nocion de "esta categoria le toca".
    """
    from src.recommender.formulas import PANDAS_OPS, category_need_weight

    cfg = NBAConfig().policy
    ratios = pd.Series([0.0, 0.25, 0.5, 1.0, 3.0, np.nan])
    esperado = cfg.min_relevance + (1.0 - cfg.min_relevance) * category_need_weight(
        ratios, PANDAS_OPS
    )
    pd.testing.assert_series_equal(pol.relevance(ratios, cfg), esperado)


def test_la_relevancia_va_del_suelo_a_uno_y_se_satura() -> None:
    cfg = NBAConfig().policy
    got = pol.relevance(pd.Series([0.0, 0.5, 1.0, 9.0, np.nan]), cfg)
    assert got.iloc[0] == pytest.approx(cfg.min_relevance)   # recien repuesta
    assert got.iloc[2] == pytest.approx(1.0)                 # justo cuando toca
    assert got.iloc[3] == pytest.approx(1.0)                 # muy vencida: no sube mas
    assert got.iloc[4] == pytest.approx(cfg.min_relevance)   # sin historial
    assert got.iloc[0] < got.iloc[1] < got.iloc[2]


def test_un_cupon_de_categoria_recien_repuesta_retiene_menos() -> None:
    """El punto M7: la retencion de una oferta irrelevante no puede valer lo mismo.

    Mismo cliente en riesgo y mismo valor de retener; lo unico que cambia es si la
    categoria le toca. El que no le toca tiene que valer estrictamente menos.
    """
    cupon = ACTION_BY_NAME["enviar_cupon_categoria"]
    toca = pol.action_value(_candidates(p_churn=0.9, retention_value=20.0), cupon).iloc[0]
    repuesta = pol.action_value(
        _candidates(p_churn=0.9, retention_value=20.0, relevance=NBAConfig().policy.min_relevance),
        cupon,
    ).iloc[0]
    assert repuesta < toca
    # Toda la diferencia esta en el termino de retencion, no en el de cross-sell.
    perdido = cupon.churn_reduction * (1.0 - NBAConfig().policy.min_relevance) * 0.9 * 20.0
    assert toca - repuesta == pytest.approx(perdido, abs=1e-9)


def test_la_politica_prefiere_la_categoria_que_toca_a_la_recien_repuesta() -> None:
    """Antes de M7 elegia justo la contraria, por minimizar la fuga de descuento.

    Dos categorias del mismo cliente: una que acaba de reponer (y que por eso el modelo de
    propension da por poco probable) y otra que ya le toca. Sin el factor de relevancia el
    `argmax` se iba a la primera, porque el descuento del cupon solo se paga si compra y
    ahi casi no compra. Con el factor, gana la que de verdad le sirve al cliente.
    """
    cfg = NBAConfig()
    candidates = pd.concat(
        [
            _candidates(
                category="Leche",  # recien repuesta: no la va a comprar
                p_purchase=0.01,
                p_churn=0.9,
                retention_value=20.0,
                relevance=cfg.policy.min_relevance,
            ),
            _candidates(
                category="Cafe",  # ya le toca
                p_purchase=0.15,
                p_churn=0.9,
                retention_value=20.0,
                relevance=1.0,
            ),
        ],
        ignore_index=True,
    )
    out = pol.decide(candidates, cfg)
    assert out["category"].iloc[0] == "Cafe"


def test_la_accion_nula_no_depende_de_la_relevancia() -> None:
    """Vale 0 por construccion, le toque al cliente la categoria o no."""
    nula = ACTION_BY_NAME["ninguna_accion"]
    for rel in (0.0, 0.15, 1.0):
        assert pol.action_value(_candidates(relevance=rel), nula).iloc[0] == 0.0


def test_la_accion_de_categoria_se_llama_por_lo_que_hace() -> None:
    """Punto M7: la accion decide una categoria, no una referencia."""
    assert "recomendar_producto" not in ACTION_BY_NAME
    accion = ACTION_BY_NAME["recomendar_categoria"]
    assert accion.needs_category
    assert accion.discount == 0.0

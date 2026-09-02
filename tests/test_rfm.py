"""Tests del RFM por cliente.

Cinco clientes con perfiles deliberadamente separados, para que los quintiles caigan uno
en cada tramo y se pueda comprobar el mapeo score -> segmento sin ambiguedad:

| Cliente | Compras | Ultima     | Gasto | Perfil                                    |
| ------- | ------: | ---------- | ----: | ----------------------------------------- |
| C1      |       5 | 2024-06-28 | 500 € | Compra mucho y hace nada: campeon         |
| C2      |       4 | 2024-06-20 | 400 € | Bueno pero un peldano por debajo          |
| C3      |       3 | 2024-04-01 | 300 € | A medio camino                            |
| C4      |       2 | 2024-02-01 | 200 € | Se esta enfriando                         |
| C5      |       1 | 2024-01-05 | 100 € | Una compra y desaparecio                  |
| C6      |       0 | -          |     - | Alta sin ninguna compra                   |
"""

from __future__ import annotations

import pytest

from src.etl.rfm import SEGMENT_CHAMPIONS, SEGMENT_NO_PURCHASES, rfm
from tests.helpers import collect_dicts, spark_df

BASKETS_DDL = (
    "basket_id string, customer_id string, channel string, basket_date timestamp, "
    "store_id string, total_amount double, n_units int"
)
CUSTOMERS_DDL = (
    "customer_id string, signup_date date, loyalty_tier string, churn_label boolean"
)

# (cliente, dias de compra, importe por cesta)
PROFILES = [
    ("C1", ["2024-05-01", "2024-05-20", "2024-06-05", "2024-06-15", "2024-06-28"], 100.0),
    ("C2", ["2024-04-10", "2024-05-05", "2024-06-01", "2024-06-20"], 100.0),
    ("C3", ["2024-02-10", "2024-03-05", "2024-04-01"], 100.0),
    ("C4", ["2024-01-15", "2024-02-01"], 100.0),
    ("C5", ["2024-01-05"], 100.0),
]

BASKETS = [
    {
        "basket_id": f"B{customer}{i}",
        "customer_id": customer,
        "channel": "app",
        "basket_date": f"{day} 10:00:00",
        "store_id": None,
        "total_amount": amount,
        "n_units": 3,
    }
    for customer, days, amount in PROFILES
    for i, day in enumerate(days)
] + [
    # Compra anonima: no se puede atribuir a nadie y no debe aparecer en el RFM.
    {
        "basket_id": "BANON",
        "customer_id": None,
        "channel": "store",
        "basket_date": "2024-06-30 10:00:00",
        "store_id": "S001",
        "total_amount": 999.0,
        "n_units": 1,
    }
]

CUSTOMERS = [
    {"customer_id": c, "signup_date": "2023-01-01", "loyalty_tier": "gold", "churn_label": False}
    for c in ("C1", "C2", "C3", "C4", "C5", "C6")
]


@pytest.fixture(scope="module")
def baskets(spark):
    return spark_df(spark, BASKETS, BASKETS_DDL)


@pytest.fixture(scope="module")
def customers(spark):
    return spark_df(spark, CUSTOMERS, CUSTOMERS_DDL)


def by_customer(df) -> dict[str, dict]:
    return {r["customer_id"]: r for r in collect_dicts(df)}


# --------------------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------------------


def test_las_metricas_basicas_cuadran(baskets):
    result = by_customer(rfm(baskets, reference_date="2024-06-30"))["C1"]

    assert result["frequency"] == 5
    assert result["monetary"] == pytest.approx(500.0)
    assert result["avg_ticket"] == pytest.approx(100.0)
    assert str(result["last_purchase_date"]) == "2024-06-28"
    assert result["recency_days"] == 2


def test_la_cesta_anonima_no_entra(baskets):
    result = by_customer(rfm(baskets, reference_date="2024-06-30"))

    assert None not in result
    assert sum(r["monetary"] for r in result.values()) == pytest.approx(1500.0)


def test_la_fecha_de_referencia_por_defecto_es_la_ultima_compra(baskets):
    """La cesta mas reciente es la anonima del 30-06, y marca el "hoy" del dataset."""
    result = by_customer(rfm(baskets))["C1"]

    assert str(result["reference_date"]) == "2024-06-30"


def test_la_cadencia_media_entre_cestas(baskets):
    """C1 compra 5 veces entre el 01-05 y el 28-06: 58 dias repartidos en 4 huecos."""
    result = by_customer(rfm(baskets, reference_date="2024-06-30"))["C1"]

    assert result["avg_days_between_baskets"] == pytest.approx(58 / 4, abs=0.1)


def test_una_sola_compra_no_tiene_cadencia(baskets):
    result = by_customer(rfm(baskets, reference_date="2024-06-30"))["C5"]

    assert result["avg_days_between_baskets"] is None


# --------------------------------------------------------------------------------------
# Scores y segmentos
# --------------------------------------------------------------------------------------


def test_el_score_alto_es_siempre_el_mejor(baskets):
    """Mas reciente, mas frecuente y mas gasto tienen que dar 5; lo contrario, 1."""
    result = by_customer(rfm(baskets, reference_date="2024-06-30"))

    assert result["C1"]["r_score"] == 5
    assert result["C1"]["f_score"] == 5
    assert result["C1"]["m_score"] == 5
    assert result["C5"]["r_score"] == 1
    assert result["C5"]["f_score"] == 1
    assert result["C5"]["m_score"] == 1


def test_el_mejor_cliente_es_campeon(baskets):
    result = by_customer(rfm(baskets, reference_date="2024-06-30"))["C1"]

    assert result["rfm_segment"] == SEGMENT_CHAMPIONS
    assert result["rfm_score"] == "555"


def test_el_cliente_sin_compras_aparece_con_su_propio_segmento(baskets, customers):
    result = by_customer(rfm(baskets, customers, reference_date="2024-06-30"))["C6"]

    assert result["frequency"] == 0
    assert result["monetary"] == pytest.approx(0.0)
    assert result["has_purchases"] is False
    assert result["rfm_segment"] == SEGMENT_NO_PURCHASES


def test_los_clientes_sin_compras_no_desplazan_los_quintiles(baskets, customers):
    """Con y sin maestro de clientes, quien si compra tiene que sacar el mismo score."""
    solo_compradores = by_customer(rfm(baskets, reference_date="2024-06-30"))
    con_maestro = by_customer(rfm(baskets, customers, reference_date="2024-06-30"))

    for customer in ("C1", "C2", "C3", "C4", "C5"):
        assert con_maestro[customer]["r_score"] == solo_compradores[customer]["r_score"]
        assert con_maestro[customer]["f_score"] == solo_compradores[customer]["f_score"]


def test_el_maestro_aporta_sus_atributos(baskets, customers):
    result = by_customer(rfm(baskets, customers, reference_date="2024-06-30"))["C1"]

    assert result["loyalty_tier"] == "gold"
    assert result["churn_label"] is False


def test_todos_los_clientes_aparecen_una_sola_vez(baskets, customers):
    result = rfm(baskets, customers, reference_date="2024-06-30")

    assert result.count() == 6
    assert result.select("customer_id").distinct().count() == 6


def test_se_puede_cambiar_el_numero_de_tramos(baskets):
    result = by_customer(rfm(baskets, reference_date="2024-06-30", n_bins=3))

    assert max(r["r_score"] for r in result.values()) == 3


def test_numero_de_tramos_invalido(baskets):
    with pytest.raises(ValueError, match="n_bins"):
        rfm(baskets, n_bins=1)

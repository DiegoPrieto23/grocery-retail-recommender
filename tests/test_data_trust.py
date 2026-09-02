"""Tests del Data Trust Score (Tarea 1 de `CHALLENGE.md`).

La estrategia es la misma en todos: se parte de un dataset diminuto pero **impecable**,
que debe puntuar 100, y cada test introduce **un solo** defecto y comprueba que lo detecta
la comprobacion concreta que le toca, en la dimension que le toca. Asi, cuando uno falla,
senala el problema exacto y no "el score ha bajado".
"""

from __future__ import annotations

import json

import pytest

from src.etl.data_trust import DIMENSIONS, data_trust_score
from tests.helpers import make_tables

# --------------------------------------------------------------------------------------
# Dataset de referencia: pequeno, completo y sin un solo defecto
# --------------------------------------------------------------------------------------

CUSTOMERS = [
    {
        "customer_id": "C1",
        "signup_date": "2023-06-01",
        "country": "Espana",
        "city": "Madrid",
        "household_size_est": 3,
        "loyalty_tier": "gold",
        "preferred_channel": "app",
        "churn_label": False,
    },
    {
        "customer_id": "C2",
        "signup_date": "2023-07-15",
        "country": "Espana",
        "city": "Sevilla",
        "household_size_est": 1,
        "loyalty_tier": "bronze",
        "preferred_channel": "web",
        "churn_label": True,
    },
]

PRODUCTS = [
    {
        "product_id": "P1",
        "department": "Frescos",
        "category": "Leche",
        "brand": "Marca A",
        "is_private_label": False,
        "is_perishable": True,
        "unit_price": 1.50,
        "pack_size": 6,
        "typical_repurchase_days": 6,
    },
    {
        "product_id": "P2",
        "department": "Despensa",
        "category": "Cereales",
        "brand": "Marca B",
        "is_private_label": False,
        "is_perishable": False,
        "unit_price": 3.00,
        "pack_size": 1,
        "typical_repurchase_days": 22,
    },
    {
        "product_id": "P3",
        "department": "Frescos",
        "category": "Pan",
        "brand": "Marca C",
        "is_private_label": True,
        "is_perishable": True,
        "unit_price": 1.00,
        "pack_size": 1,
        "typical_repurchase_days": 4,
    },
]

PROMOTIONS = [
    {
        "promotion_id": "PR1",
        "product_id": "P1",
        "promo_type": "discount_pct",
        "discount_value": 0.20,
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
    }
]

# B3 es una compra anonima: `customer_id` nulo es legitimo y no debe penalizar.
BASKETS = [
    {
        "basket_id": "B1",
        "customer_id": "C1",
        "channel": "app",
        "basket_date": "2024-01-05 10:00:00",
        "store_id": None,
        "total_amount": 5.40,
    },
    {
        "basket_id": "B2",
        "customer_id": "C1",
        "channel": "store",
        "basket_date": "2024-01-12 10:00:00",
        "store_id": "S001",
        "total_amount": 4.50,
    },
    {
        "basket_id": "B3",
        "customer_id": None,
        "channel": "store",
        "basket_date": "2024-01-20 10:00:00",
        "store_id": "S002",
        "total_amount": 1.00,
    },
]

BASKET_ITEMS = [
    {"basket_id": "B1", "product_id": "P1", "quantity": 2, "unit_price_paid": 1.20, "promotion_id": "PR1"},
    {"basket_id": "B1", "product_id": "P2", "quantity": 1, "unit_price_paid": 3.00, "promotion_id": None},
    {"basket_id": "B2", "product_id": "P1", "quantity": 1, "unit_price_paid": 1.50, "promotion_id": None},
    {"basket_id": "B2", "product_id": "P3", "quantity": 3, "unit_price_paid": 1.00, "promotion_id": None},
    {"basket_id": "B3", "product_id": "P3", "quantity": 1, "unit_price_paid": 1.00, "promotion_id": None},
]

SESSIONS = [
    {
        "session_id": "S1",
        "customer_id": "C1",
        "session_date": "2024-01-05 09:50:00",
        "device_type": "mobile",
        "converted": True,
        "basket_id": "B1",
    },
    {
        "session_id": "S2",
        "customer_id": None,
        "session_date": "2024-01-06 12:00:00",
        "device_type": "desktop",
        "converted": False,
        "basket_id": None,
    },
]

SESSION_EVENTS = [
    {"session_id": "S1", "product_id": "P1", "event_type": "view", "event_timestamp": "2024-01-05 09:51:00"},
    {"session_id": "S1", "product_id": "P1", "event_type": "add_to_cart", "event_timestamp": "2024-01-05 09:52:00"},
    {"session_id": "S2", "product_id": "P2", "event_type": "view", "event_timestamp": "2024-01-06 12:01:00"},
]

_BASE = {
    "customers": CUSTOMERS,
    "products": PRODUCTS,
    "promotions": PROMOTIONS,
    "baskets": BASKETS,
    "basket_items": BASKET_ITEMS,
    "sessions": SESSIONS,
    "session_events": SESSION_EVENTS,
}


def build(spark, **overrides: list[dict]) -> dict:
    """Construye las 7 tablas, sustituyendo las que se indiquen."""
    rows = {name: list(value) for name, value in _BASE.items()}
    rows.update({name: list(value) for name, value in overrides.items()})
    return make_tables(spark, **rows)


def replace(rows: list[dict], index: int, **changes) -> list[dict]:
    """Copia una lista de filas cambiando campos de una de ellas."""
    out = [dict(r) for r in rows]
    out[index].update(changes)
    return out


@pytest.fixture(scope="module")
def clean_report(spark):
    """El informe del dataset impecable, que muchos tests usan como referencia."""
    return data_trust_score(build(spark))


# --------------------------------------------------------------------------------------
# Referencia: el dataset limpio puntua 100
# --------------------------------------------------------------------------------------


def test_dataset_impecable_puntua_100(clean_report):
    assert clean_report.score == 100.0
    assert clean_report.grade == "A"
    assert clean_report.failing() == ()


def test_todas_las_dimensiones_tienen_comprobaciones(clean_report):
    for dimension in DIMENSIONS:
        assert any(c.dimension == dimension for c in clean_report.checks), dimension


def test_los_nulos_legitimos_no_penalizan(clean_report):
    """Compra anonima, sesion sin convertir y linea sin promocion no son defectos."""
    assert clean_report.check("baskets", "basket_id_unique").rows_failed == 0
    # La FK de `customer_id` solo evalua las filas informadas: B3 queda fuera.
    assert clean_report.check("baskets", "customer_id_fk").rows_total == 2


# --------------------------------------------------------------------------------------
# Un defecto, una dimension
# --------------------------------------------------------------------------------------


def test_valor_obligatorio_nulo_baja_completeness(spark, clean_report):
    report = data_trust_score(build(spark, customers=replace(CUSTOMERS, 0, city=None)))

    assert report.check("customers", "city_not_null").rows_failed == 1
    assert report.dimension_scores["completeness"] < clean_report.dimension_scores["completeness"]
    assert report.dimension_scores["integrity"] == 100.0


def test_linea_duplicada_baja_uniqueness(spark, clean_report):
    duplicated = BASKET_ITEMS + [dict(BASKET_ITEMS[0])]
    report = data_trust_score(build(spark, basket_items=duplicated))

    assert report.check("basket_items", "no_duplicate_rows").rows_failed == 1
    assert report.check("basket_items", "basket_product_unique").rows_failed == 1
    assert report.dimension_scores["uniqueness"] < clean_report.dimension_scores["uniqueness"]


def test_pk_repetida_baja_uniqueness(spark):
    report = data_trust_score(build(spark, customers=CUSTOMERS + [dict(CUSTOMERS[0])]))

    assert report.check("customers", "customer_id_unique").rows_failed == 1


def test_cantidad_negativa_baja_validity(spark, clean_report):
    report = data_trust_score(build(spark, basket_items=replace(BASKET_ITEMS, 0, quantity=-2)))

    assert report.check("basket_items", "quantity_positive").rows_failed == 1
    assert report.dimension_scores["validity"] < clean_report.dimension_scores["validity"]


def test_categoria_mal_escrita_baja_validity(spark):
    """`LECHE ` frente a `Leche`: misma categoria, grafia distinta."""
    messy = PRODUCTS + [dict(PRODUCTS[0], product_id="P4", category="  LECHE ")]
    report = data_trust_score(build(spark, products=messy))

    assert report.check("products", "category_canonical").rows_failed == 1


def test_valor_fuera_de_dominio_baja_validity(spark):
    report = data_trust_score(build(spark, customers=replace(CUSTOMERS, 1, loyalty_tier="platino")))

    assert report.check("customers", "loyalty_tier_domain").rows_failed == 1


def test_hogar_fuera_de_rango_baja_validity(spark):
    report = data_trust_score(build(spark, customers=replace(CUSTOMERS, 0, household_size_est=99)))

    assert report.check("customers", "household_size_in_range").rows_failed == 1


def test_fk_huerfana_baja_integrity(spark, clean_report):
    orphan = BASKET_ITEMS + [
        {"basket_id": "B1", "product_id": "P404", "quantity": 1, "unit_price_paid": 1.0, "promotion_id": None}
    ]
    report = data_trust_score(build(spark, basket_items=orphan))

    assert report.check("basket_items", "product_id_fk").rows_failed == 1
    assert report.dimension_scores["integrity"] < clean_report.dimension_scores["integrity"]


def test_alta_posterior_a_la_primera_compra_baja_consistency(spark, clean_report):
    report = data_trust_score(build(spark, customers=replace(CUSTOMERS, 0, signup_date="2024-06-01")))

    assert report.check("customers", "signup_before_first_purchase").rows_failed == 1
    assert report.check("customers", "signup_within_period").rows_failed == 1
    assert report.dimension_scores["consistency"] < clean_report.dimension_scores["consistency"]


def test_cabecera_que_no_cuadra_con_las_lineas_baja_consistency(spark):
    report = data_trust_score(build(spark, baskets=replace(BASKETS, 0, total_amount=540.0)))

    assert report.check("baskets", "total_amount_matches_items").rows_failed == 1


def test_store_id_incoherente_con_el_canal_baja_consistency(spark):
    """Una compra por app no puede tener tienda fisica asignada."""
    report = data_trust_score(build(spark, baskets=replace(BASKETS, 0, store_id="S009")))

    assert report.check("baskets", "store_id_matches_channel").rows_failed == 1


def test_converted_incoherente_con_basket_id_baja_consistency(spark):
    report = data_trust_score(build(spark, sessions=replace(SESSIONS, 1, converted=True)))

    assert report.check("sessions", "converted_matches_basket").rows_failed == 1


def test_promocion_fuera_de_vigencia_baja_consistency(spark):
    """La linea de B1 lleva PR1, pero la promocion se aplica a un dia fuera de ventana."""
    moved = replace(PROMOTIONS, 0, start_date="2024-02-01", end_date="2024-02-28")
    report = data_trust_score(build(spark, promotions=moved))

    assert report.check("basket_items", "promo_active_on_basket_date").rows_failed == 1


def test_evento_anterior_al_inicio_de_sesion_baja_consistency(spark):
    early = replace(SESSION_EVENTS, 0, event_timestamp="2024-01-05 08:00:00")
    report = data_trust_score(build(spark, session_events=early))

    assert report.check("session_events", "event_after_session_start").rows_failed == 1


# --------------------------------------------------------------------------------------
# Agregacion: como se combinan las comprobaciones en la nota final
# --------------------------------------------------------------------------------------


def test_un_dataset_sucio_puntua_menos_que_uno_limpio(spark, clean_report):
    dirty = build(
        spark,
        customers=replace(CUSTOMERS, 0, city=None, signup_date="2024-06-01"),
        basket_items=BASKET_ITEMS + [dict(BASKET_ITEMS[0])],
    )
    report = data_trust_score(dirty)

    assert report.score < clean_report.score
    assert len(report.failing()) >= 3


def test_la_nota_mezcla_tasa_de_filas_y_tasa_de_reglas(spark):
    """Un solo fallo hunde mas la tasa de reglas que la de filas, y eso es lo buscado."""
    report = data_trust_score(build(spark, basket_items=replace(BASKET_ITEMS, 0, quantity=-2)))
    validity = report.dimension_detail["validity"]

    assert validity["checks_failing"] == 1
    assert validity["check_score"] < validity["row_score"]
    assert validity["score"] == pytest.approx(
        0.5 * validity["row_score"] + 0.5 * validity["check_score"], abs=0.01
    )


def test_los_pesos_se_respetan(spark):
    """Con todo el peso en integrity, la nota es exactamente la de esa dimension."""
    tables = build(spark, customers=replace(CUSTOMERS, 0, city=None))
    only_integrity = data_trust_score(tables, weights={"integrity": 1.0})

    assert only_integrity.score == only_integrity.dimension_scores["integrity"]
    assert only_integrity.score == 100.0


def test_se_puede_limitar_a_unas_tablas(spark):
    report = data_trust_score(build(spark), only_tables=("products",))

    assert {c.table for c in report.checks} == {"products"}


def test_falta_una_tabla(spark):
    tables = build(spark)
    del tables["sessions"]

    with pytest.raises(KeyError, match="sessions"):
        data_trust_score(tables)


# --------------------------------------------------------------------------------------
# Serializacion del informe
# --------------------------------------------------------------------------------------


def test_el_informe_serializa_a_json(clean_report):
    payload = json.loads(clean_report.to_json())

    assert payload["score"] == 100.0
    assert payload["checks_failing"] == 0
    assert len(payload["checks"]) == len(clean_report.checks)
    assert set(payload["dimension_scores"]) == set(DIMENSIONS)


def test_el_informe_se_lee_en_markdown(clean_report):
    markdown = clean_report.to_markdown()

    assert "Data Trust Score: 100.00" in markdown
    for dimension in DIMENSIONS:
        assert f"`{dimension}`" in markdown


def test_solo_fallos_recorta_el_markdown(spark):
    report = data_trust_score(build(spark, basket_items=replace(BASKET_ITEMS, 0, quantity=-2)))

    assert len(report.to_markdown(only_failing=True)) < len(report.to_markdown())


def test_check_desconocida_falla_claro(clean_report):
    with pytest.raises(KeyError, match="no_existe"):
        clean_report.check("products", "no_existe")

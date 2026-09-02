"""Tests de `due_for_repurchase` y del ciclo de recompra (Tarea 2 de `CHALLENGE.md`).

El dataset de apoyo es minimo y esta construido para que **cada cifra sea comprobable a
mano**: cadencias exactas de 7 y 4 dias, un solo cliente por escenario y fechas redondas.
Si un test falla, el numero esperado se puede recalcular mentalmente.

Escenarios cubiertos:

| Cliente | Categoria  | Compras                          | Que ejercita                        |
| ------- | ---------- | -------------------------------- | ----------------------------------- |
| C1      | Leche      | 01-01, 01-08, 01-15, 01-22       | Cadencia observada de 7 dias        |
| C1      | Pan        | 01-01 (x2 cestas), 01-05         | Dos cestas el mismo dia = 1 ocasion |
| C1      | Detergente | 01-01                            | Sin historia: respaldo por categoria |
| C2      | Detergente | 01-01                            | Mismo caso con hogar de 5           |
| anonimo | Leche      | 01-28                            | Cesta sin cliente: se ignora        |
"""

from __future__ import annotations

import pytest

from src.etl.repurchase import category_repurchase_days, repurchase_features
from tests.helpers import collect_dicts, make_tables, spark_df

CUSTOMERS = [
    {
        "customer_id": "C1",
        "signup_date": "2023-01-01",
        "country": "Espana",
        "city": "Madrid",
        "household_size_est": 1,
        "loyalty_tier": "gold",
        "preferred_channel": "app",
        "churn_label": False,
    },
    {
        "customer_id": "C2",
        "signup_date": "2023-01-01",
        "country": "Espana",
        "city": "Bilbao",
        "household_size_est": 5,
        "loyalty_tier": "silver",
        "preferred_channel": "store",
        "churn_label": False,
    },
]

PRODUCTS = [
    {
        "product_id": "P1",
        "department": "Frescos",
        "category": "Leche",
        "brand": "A",
        "is_private_label": False,
        "is_perishable": True,
        "unit_price": 1.5,
        "pack_size": 6,
        "typical_repurchase_days": 6,
    },
    {
        "product_id": "P2",
        "department": "Drogueria",
        "category": "Detergente",
        "brand": "B",
        "is_private_label": False,
        "is_perishable": False,
        "unit_price": 8.0,
        "pack_size": 1,
        "typical_repurchase_days": 45,
    },
    {
        "product_id": "P3",
        "department": "Frescos",
        "category": "Pan",
        "brand": "C",
        "is_private_label": True,
        "is_perishable": True,
        "unit_price": 1.0,
        "pack_size": 1,
        "typical_repurchase_days": 4,
    },
]


def _basket(basket_id: str, customer_id: str | None, day: str) -> dict:
    return {
        "basket_id": basket_id,
        "customer_id": customer_id,
        "channel": "app",
        "basket_date": f"{day} 10:00:00",
        "store_id": None,
        "total_amount": 10.0,
    }


BASKETS = [
    _basket("B1", "C1", "2024-01-01"),
    _basket("B2", "C1", "2024-01-08"),
    _basket("B3", "C1", "2024-01-15"),
    _basket("B4", "C1", "2024-01-22"),
    _basket("B5", "C1", "2024-01-01"),  # segunda cesta del mismo dia
    _basket("B6", "C1", "2024-01-05"),
    _basket("B7", "C2", "2024-01-01"),
    _basket("B8", None, "2024-01-28"),  # compra anonima, la mas reciente del dataset
]


def _line(basket_id: str, product_id: str) -> dict:
    return {
        "basket_id": basket_id,
        "product_id": product_id,
        "quantity": 1,
        "unit_price_paid": 1.0,
        "promotion_id": None,
    }


BASKET_ITEMS = [
    _line("B1", "P1"), _line("B2", "P1"), _line("B3", "P1"), _line("B4", "P1"),
    _line("B1", "P2"),
    _line("B1", "P3"), _line("B5", "P3"), _line("B6", "P3"),
    _line("B7", "P2"),
    _line("B8", "P1"),
]


@pytest.fixture(scope="module")
def tables(spark):
    return make_tables(
        spark, customers=CUSTOMERS, products=PRODUCTS, baskets=BASKETS, basket_items=BASKET_ITEMS
    )


def features(tables: dict, **kwargs):
    """Atajo: llama a `repurchase_features` con las tablas del modulo."""
    return repurchase_features(
        tables["basket_items"], tables["baskets"], tables["products"], tables["customers"], **kwargs
    )


def by_key(tables: dict, **kwargs) -> dict[tuple[str, str], dict]:
    """Resultado indexado por `(customer_id, category)`, que es como se lee en los tests."""
    rows = collect_dicts(features(tables, **kwargs))
    return {(r["customer_id"], r["category"]): r for r in rows}


# --------------------------------------------------------------------------------------
# Grano y cobertura del resultado
# --------------------------------------------------------------------------------------


def test_una_fila_por_cliente_y_categoria(tables):
    result = by_key(tables, reference_date="2024-02-01")

    assert set(result) == {
        ("C1", "Leche"),
        ("C1", "Pan"),
        ("C1", "Detergente"),
        ("C2", "Detergente"),
    }


def test_las_cestas_anonimas_se_ignoran(tables):
    """B8 es del 28-01 y lleva leche, pero no tiene cliente: no puede mover nada."""
    result = by_key(tables, reference_date="2024-02-01")

    assert all(key[0] is not None for key in result)
    assert str(result[("C1", "Leche")]["last_purchase_date"]) == "2024-01-22"


def test_dos_cestas_del_mismo_dia_son_una_sola_ocasion(tables):
    """B1 y B5 llevan pan el mismo dia: si contasen dos veces habria un hueco de 0 dias."""
    pan = by_key(tables, reference_date="2024-02-01")[("C1", "Pan")]

    assert pan["n_purchase_days"] == 2
    assert pan["observed_repurchase_days"] == pytest.approx(4.0)


# --------------------------------------------------------------------------------------
# Intervalo esperado: observado frente a respaldo por categoria
# --------------------------------------------------------------------------------------


def test_con_historia_manda_la_cadencia_observada(tables):
    """C1 compra leche cada 7 dias, aunque el tipico de la categoria sean 6."""
    leche = by_key(tables, reference_date="2024-02-01")[("C1", "Leche")]

    assert leche["n_purchase_days"] == 4
    assert leche["observed_repurchase_days"] == pytest.approx(7.0)
    assert leche["typical_repurchase_days"] == 6
    assert leche["expected_repurchase_days"] == pytest.approx(7.0)


def test_sin_historia_se_usa_el_tipico_ajustado_por_hogar(tables):
    """Una sola compra no da cadencia: 45 dias de categoria x 1.325 del hogar de 1."""
    det = by_key(tables, reference_date="2024-02-01")[("C1", "Detergente")]

    assert det["n_purchase_days"] == 1
    assert det["observed_repurchase_days"] is None
    assert det["household_factor"] == pytest.approx(1.325)
    assert det["expected_repurchase_days"] == pytest.approx(45 * 1.325, abs=0.01)


def test_un_hogar_grande_acorta_el_ciclo(tables):
    """Mismo producto y misma compra: el hogar de 5 repone antes que el de 1."""
    result = by_key(tables, reference_date="2024-02-01")
    solo, familia = result[("C1", "Detergente")], result[("C2", "Detergente")]

    assert familia["household_factor"] == pytest.approx(0.825)
    assert familia["expected_repurchase_days"] == pytest.approx(45 * 0.825, abs=0.01)
    assert familia["expected_repurchase_days"] < solo["expected_repurchase_days"]


def test_se_puede_desactivar_el_ajuste_por_hogar(tables):
    det = by_key(tables, reference_date="2024-02-01", household_adjustment=False)[
        ("C1", "Detergente")
    ]

    assert det["household_factor"] == pytest.approx(1.0)
    assert det["expected_repurchase_days"] == pytest.approx(45.0)


def test_min_observations_exige_mas_historia_para_fiarse(tables):
    """Con 4 compras y min_observations=5, la cadencia observada se descarta."""
    leche = by_key(tables, reference_date="2024-02-01", min_observations=5)[("C1", "Leche")]

    assert leche["observed_repurchase_days"] is None
    assert leche["expected_repurchase_days"] == pytest.approx(6 * 1.325, abs=0.01)


# --------------------------------------------------------------------------------------
# El flag due_for_repurchase
# --------------------------------------------------------------------------------------


def test_toca_reponer_cuando_se_cumple_el_ciclo(tables):
    """Ultima leche el 22-01, cadencia de 7: el 01-02 lleva 10 dias, toca."""
    leche = by_key(tables, reference_date="2024-02-01")[("C1", "Leche")]

    assert leche["days_since_last_purchase"] == 10
    assert leche["due_for_repurchase"] is True
    assert leche["overdue_ratio"] == pytest.approx(10 / 7, abs=0.001)


def test_no_toca_reponer_dentro_del_ciclo(tables):
    """El 25-01 solo han pasado 3 dias de los 7 de cadencia."""
    leche = by_key(tables, reference_date="2024-01-25")[("C1", "Leche")]

    assert leche["days_since_last_purchase"] == 3
    assert leche["due_for_repurchase"] is False
    assert leche["overdue_ratio"] < 1.0


def test_no_toca_reponer_una_categoria_de_ciclo_largo(tables):
    """31 dias desde el detergente, pero su ciclo esperado ronda los 60."""
    det = by_key(tables, reference_date="2024-02-01")[("C1", "Detergente")]

    assert det["days_since_last_purchase"] == 31
    assert det["due_for_repurchase"] is False


def test_la_tolerancia_retrasa_el_aviso(tables):
    """Con un 100 % de margen, 10 dias ya no bastan para una cadencia de 7."""
    sin_margen = by_key(tables, reference_date="2024-02-01")[("C1", "Leche")]
    con_margen = by_key(tables, reference_date="2024-02-01", tolerance=2.0)[("C1", "Leche")]

    assert sin_margen["due_for_repurchase"] is True
    assert con_margen["due_for_repurchase"] is False


def test_el_hogar_grande_entra_antes_en_due(tables):
    """A los 45 dias, el hogar de 5 ya toca detergente y el de 1 todavia no."""
    result = by_key(tables, reference_date="2024-02-15")

    assert result[("C2", "Detergente")]["due_for_repurchase"] is True
    assert result[("C1", "Detergente")]["due_for_repurchase"] is False


def test_la_fecha_de_referencia_por_defecto_es_el_ultimo_dia_con_compras(tables):
    """Sin `reference_date` se usa el 28-01, que es la cesta mas reciente (la anonima)."""
    leche = by_key(tables)[("C1", "Leche")]

    assert str(leche["reference_date"]) == "2024-01-28"
    assert leche["days_since_last_purchase"] == 6


# --------------------------------------------------------------------------------------
# Intervalo tipico por categoria
# --------------------------------------------------------------------------------------


def test_el_tipico_de_la_categoria_promedia_sus_productos(spark):
    """Si dos productos de la misma categoria discrepan, se promedia en vez de elegir uno."""
    products = spark_df(
        spark,
        [
            dict(PRODUCTS[0], product_id="P1", typical_repurchase_days=6),
            dict(PRODUCTS[0], product_id="P9", typical_repurchase_days=10),
        ],
        "product_id string, department string, category string, brand string, "
        "is_private_label boolean, is_perishable boolean, unit_price double, "
        "pack_size int, typical_repurchase_days int",
    )
    rows = collect_dicts(category_repurchase_days(products))

    assert rows == [{"category": "Leche", "typical_repurchase_days": 8}]


# --------------------------------------------------------------------------------------
# Validacion de argumentos
# --------------------------------------------------------------------------------------


def test_tolerancia_invalida(tables):
    with pytest.raises(ValueError, match="tolerance"):
        features(tables, tolerance=0)


def test_min_observations_invalido(tables):
    with pytest.raises(ValueError, match="min_observations"):
        features(tables, min_observations=1)


def test_ajuste_por_hogar_sin_clientes(tables):
    with pytest.raises(ValueError, match="household_adjustment"):
        repurchase_features(
            tables["basket_items"], tables["baskets"], tables["products"], None
        )


def test_sin_ajuste_por_hogar_no_hacen_falta_clientes(tables):
    result = repurchase_features(
        tables["basket_items"],
        tables["baskets"],
        tables["products"],
        None,
        household_adjustment=False,
        reference_date="2024-02-01",
    )

    assert result.count() == 4

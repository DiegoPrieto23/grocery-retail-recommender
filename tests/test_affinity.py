"""Tests de la tabla de afinidad de cesta.

El dataset esta construido para que soporte, confianza y lift salgan a mano:

    10 cestas en total
    Alfa  en 6 cestas   (B01-B06)
    Beta  en 8 cestas   (B01-B04, B07-B10)
    Gamma en 2 cestas   (B09, B10)
    Alfa + Beta  juntas en 4 cestas   (B01-B04)
    Beta + Gamma juntas en 2 cestas   (B09, B10)
    Alfa + Gamma nunca coinciden

    support(Alfa, Beta)    = 4 / 10           = 0.40
    confidence(Alfa->Beta) = 4 / 6            = 0.6667
    lift(Alfa->Beta)       = 0.6667 / (8/10)  = 0.8333
    confidence(Beta->Alfa) = 4 / 8            = 0.50
    lift(Beta->Alfa)       = 0.50   / (6/10)  = 0.8333   (el lift es simetrico)
    jaccard(Alfa, Beta)    = 4 / (6 + 8 - 4)  = 0.40
"""

from __future__ import annotations

import pytest

from src.etl.affinity import (
    basket_item_sets,
    cooccurrence_affinity,
    expected_pairs_report,
    fpgrowth_rules,
)
from tests.helpers import collect_dicts, make_tables

CATEGORY_OF = {"PA": "Alfa", "PB": "Beta", "PC": "Gamma"}

PRODUCTS = [
    {
        "product_id": product_id,
        "department": "Despensa",
        "category": category,
        "brand": "Marca",
        "is_private_label": False,
        "is_perishable": False,
        "unit_price": 1.0,
        "pack_size": 1,
        "typical_repurchase_days": 10,
    }
    for product_id, category in CATEGORY_OF.items()
]

# Que producto lleva cada cesta, segun el reparto del docstring.
BASKET_CONTENTS = {
    "B01": ["PA", "PB"],
    "B02": ["PA", "PB"],
    "B03": ["PA", "PB"],
    "B04": ["PA", "PB"],
    "B05": ["PA"],
    "B06": ["PA"],
    "B07": ["PB"],
    "B08": ["PB"],
    "B09": ["PB", "PC"],
    "B10": ["PB", "PC"],
}

BASKETS = [
    {
        "basket_id": basket_id,
        "customer_id": "C1",
        "channel": "app",
        "basket_date": "2024-01-01 10:00:00",
        "store_id": None,
        "total_amount": 1.0,
    }
    for basket_id in BASKET_CONTENTS
]

BASKET_ITEMS = [
    {
        "basket_id": basket_id,
        "product_id": product_id,
        "quantity": 1,
        "unit_price_paid": 1.0,
        "promotion_id": None,
    }
    for basket_id, products in BASKET_CONTENTS.items()
    for product_id in products
]


@pytest.fixture(scope="module")
def tables(spark):
    return make_tables(spark, products=PRODUCTS, baskets=BASKETS, basket_items=BASKET_ITEMS)


@pytest.fixture(scope="module")
def affinity(tables):
    """Afinidad de categorias con el umbral de soporte al minimo util."""
    return {
        (r["antecedent"], r["consequent"]): r
        for r in collect_dicts(
            cooccurrence_affinity(
                tables["basket_items"], tables["products"], level="category", min_pair_baskets=2
            )
        )
    }


# --------------------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------------------


def test_las_metricas_salen_como_a_mano(affinity):
    pair = affinity[("Alfa", "Beta")]

    assert pair["total_baskets"] == 10
    assert pair["n_baskets_antecedent"] == 6
    assert pair["n_baskets_consequent"] == 8
    assert pair["n_baskets_both"] == 4
    assert pair["support"] == pytest.approx(0.40)
    assert pair["confidence"] == pytest.approx(4 / 6, abs=1e-5)
    assert pair["lift"] == pytest.approx((4 / 6) / (8 / 10), abs=1e-4)
    assert pair["jaccard"] == pytest.approx(0.40)


def test_el_par_se_guarda_en_las_dos_direcciones(affinity):
    assert ("Alfa", "Beta") in affinity
    assert ("Beta", "Alfa") in affinity


def test_el_lift_es_simetrico_y_la_confianza_no(affinity):
    ida, vuelta = affinity[("Alfa", "Beta")], affinity[("Beta", "Alfa")]

    assert ida["lift"] == pytest.approx(vuelta["lift"])
    assert ida["support"] == pytest.approx(vuelta["support"])
    assert ida["jaccard"] == pytest.approx(vuelta["jaccard"])
    assert ida["confidence"] != pytest.approx(vuelta["confidence"])


def test_un_par_que_nunca_coincide_no_aparece(affinity):
    assert ("Alfa", "Gamma") not in affinity
    assert ("Gamma", "Alfa") not in affinity


def test_una_categoria_rara_puede_tener_lift_alto(affinity):
    """Gamma solo esta en 2 cestas, pero siempre junto a Beta."""
    pair = affinity[("Gamma", "Beta")]

    assert pair["confidence"] == pytest.approx(1.0)
    assert pair["lift"] == pytest.approx(1.0 / 0.8, abs=1e-4)


# --------------------------------------------------------------------------------------
# Filtros y grano
# --------------------------------------------------------------------------------------


def test_el_soporte_minimo_descarta_los_pares_flojos(tables):
    result = cooccurrence_affinity(
        tables["basket_items"], tables["products"], level="category", min_pair_baskets=3
    )
    pairs = {(r["antecedent"], r["consequent"]) for r in collect_dicts(result)}

    assert ("Alfa", "Beta") in pairs
    assert ("Beta", "Gamma") not in pairs  # solo coinciden en 2 cestas


def test_top_n_deja_el_mejor_consecuente_por_antecedente(tables):
    result = cooccurrence_affinity(
        tables["basket_items"],
        tables["products"],
        level="category",
        min_pair_baskets=2,
        top_n=1,
    )
    rows = collect_dicts(result)

    assert len(rows) == len({r["antecedent"] for r in rows})
    assert all(r["rank"] == 1 for r in rows)
    # Beta acompana a Alfa y a Gamma; se queda con el de mas lift, que es Gamma.
    assert next(r["consequent"] for r in rows if r["antecedent"] == "Beta") == "Gamma"


def test_el_grano_de_producto_usa_los_product_id(tables):
    result = cooccurrence_affinity(
        tables["basket_items"], level="product", min_pair_baskets=2
    )
    antecedents = {r["antecedent"] for r in collect_dicts(result)}

    assert antecedents <= set(CATEGORY_OF)


def test_los_conjuntos_de_cesta_no_repiten_articulo(tables):
    sets = basket_item_sets(
        tables["basket_items"], None, tables["products"], level="category"
    )

    assert sets.count() == sets.distinct().count()
    assert sets.count() == sum(len(v) for v in BASKET_CONTENTS.values())


def test_nivel_desconocido(tables):
    with pytest.raises(ValueError, match="level"):
        cooccurrence_affinity(tables["basket_items"], level="departamento")


def test_el_nivel_categoria_exige_el_catalogo(tables):
    with pytest.raises(ValueError, match="products"):
        cooccurrence_affinity(tables["basket_items"], None, level="category")


# --------------------------------------------------------------------------------------
# Contraste con los pares esperados y con FP-Growth
# --------------------------------------------------------------------------------------


def test_el_informe_de_pares_esperados_localiza_lo_que_hay_y_lo_que_falta(tables):
    affinity = cooccurrence_affinity(
        tables["basket_items"], tables["products"], level="category", min_pair_baskets=2
    )
    rows = collect_dicts(
        expected_pairs_report(affinity, [("Alfa", "Beta", 0.8), ("Alfa", "Gamma", 2.0)])
    )

    assert rows[0]["antecedent"] == "Alfa" and rows[0]["consequent"] == "Beta"
    assert rows[0]["lift"] == pytest.approx(0.8333, abs=1e-3)
    assert rows[0]["ratio_vs_target"] == pytest.approx(0.8333 / 0.8, abs=1e-2)
    # El par que no existe en el dato se reporta a nulo, no se pierde por el camino.
    assert rows[1]["consequent"] == "Gamma"
    assert rows[1]["lift"] is None


def test_fpgrowth_encuentra_el_mismo_par_dominante(tables):
    _, rules = fpgrowth_rules(
        tables["basket_items"], tables["products"], level="category", min_support=0.2
    )
    found = {
        (tuple(r["antecedent"]), tuple(r["consequent"])) for r in collect_dicts(rules)
    }

    assert (("Alfa",), ("Beta",)) in found

"""Tests de la limpieza de la Fase 2.

Cubren las decisiones que no son mecanicas y que, si se rompen, arrastran un error silencioso
a todo lo que viene despues: el orden signo-antes-que-duplicado, la eleccion de la grafia
canonica de categoria, el recalculo de `total_amount` y el respeto por los nulos legitimos.
"""

from __future__ import annotations

import pytest

from src.etl.cleaning import (
    CleaningConfig,
    CleaningReport,
    canonical_category_map,
    clean_all,
    clean_basket_items,
    clean_products,
)
from tests.helpers import collect_dicts, make_tables, spark_df

PRODUCTS_DDL = (
    "product_id string, department string, category string, brand string, "
    "is_private_label boolean, is_perishable boolean, unit_price double, "
    "pack_size int, typical_repurchase_days int"
)
ITEMS_DDL = (
    "basket_id string, product_id string, quantity int, unit_price_paid double, "
    "promotion_id string"
)


def product(product_id: str, category: str, brand: str | None = "Marca") -> dict:
    return {
        "product_id": product_id,
        "department": "Frescos",
        "category": category,
        "brand": brand,
        "is_private_label": False,
        "is_perishable": True,
        "unit_price": 1.0,
        "pack_size": 1,
        "typical_repurchase_days": 6,
    }


def line(basket_id: str, product_id: str, quantity: int, price: float = 2.0) -> dict:
    return {
        "basket_id": basket_id,
        "product_id": product_id,
        "quantity": quantity,
        "unit_price_paid": price,
        "promotion_id": None,
    }


# --------------------------------------------------------------------------------------
# Normalizacion de categorias
# --------------------------------------------------------------------------------------


def test_la_grafia_canonica_es_la_mayoritaria(spark):
    """Tres productos escriben `Leche` bien y dos mal: gana la mayoria."""
    products = spark_df(
        spark,
        [
            product("P1", "Leche"),
            product("P2", "Leche"),
            product("P3", "Leche"),
            product("P4", "LECHE"),
            product("P5", "  leche  "),
        ],
        PRODUCTS_DDL,
    )
    rows = collect_dicts(canonical_category_map(products))

    assert rows == [{"category_key": "leche", "category_canonical": "Leche"}]


def test_los_espacios_internos_tambien_se_colapsan(spark):
    products = spark_df(
        spark,
        [product("P1", "Salsa de tomate"), product("P2", "Salsa  de  tomate")],
        PRODUCTS_DDL,
    )
    cleaned = clean_products(products, CleaningConfig(), CleaningReport())

    assert {r["category"] for r in collect_dicts(cleaned)} == {"Salsa de tomate"}


def test_la_categoria_original_se_conserva_para_auditar(spark):
    products = spark_df(spark, [product("P1", "Leche"), product("P2", "LECHE")], PRODUCTS_DDL)
    rows = {r["product_id"]: r for r in collect_dicts(clean_products(products, CleaningConfig(), CleaningReport()))}

    assert rows["P2"]["category"] == "Leche"
    assert rows["P2"]["category_raw"] == "LECHE"


def test_la_normalizacion_no_inventa_categorias(spark):
    """Dos categorias distintas de verdad no pueden acabar fundidas en una."""
    products = spark_df(
        spark, [product("P1", "Leche"), product("P2", "Leche infantil")], PRODUCTS_DDL
    )
    cleaned = clean_products(products, CleaningConfig(), CleaningReport())

    assert cleaned.select("category").distinct().count() == 2


def test_la_marca_nula_se_rellena_y_se_marca(spark):
    products = spark_df(spark, [product("P1", "Leche", brand=None)], PRODUCTS_DDL)
    row = collect_dicts(clean_products(products, CleaningConfig(), CleaningReport()))[0]

    assert row["brand"] == "Sin marca"
    assert row["brand_is_missing"] is True


# --------------------------------------------------------------------------------------
# El orden signo -> duplicado (la decision central de `clean_basket_items`)
# --------------------------------------------------------------------------------------


def test_el_duplicado_con_el_signo_invertido_colapsa(spark):
    """Un TPV emite la linea dos veces y a una copia se le invierte el signo.

    Corrigiendo el signo primero, las dos copias vuelven a ser identicas y quedan en una.
    Al reves sobrevivirian las dos y la cesta contaria el doble.
    """
    items = spark_df(spark, [line("B1", "P1", 2), line("B1", "P1", -2)], ITEMS_DDL)
    rows = collect_dicts(clean_basket_items(items, CleaningConfig(), CleaningReport()))

    assert len(rows) == 1
    assert rows[0]["quantity"] == 2
    assert rows[0]["quantity_sign_corrected"] is True


def test_no_queda_mas_de_una_linea_por_cesta_y_producto(spark):
    items = spark_df(
        spark,
        [line("B1", "P1", 2), line("B1", "P1", -2), line("B1", "P1", 2), line("B1", "P2", 1)],
        ITEMS_DDL,
    )
    cleaned = clean_basket_items(items, CleaningConfig(), CleaningReport())

    assert cleaned.groupBy("basket_id", "product_id").count().filter("count > 1").count() == 0


def test_la_cantidad_negativa_se_corrige_a_positiva(spark):
    items = spark_df(spark, [line("B1", "P1", -3)], ITEMS_DDL)
    row = collect_dicts(clean_basket_items(items, CleaningConfig(), CleaningReport()))[0]

    assert row["quantity"] == 3
    assert row["line_amount"] == pytest.approx(6.0)


def test_la_politica_drop_descarta_las_negativas(spark):
    items = spark_df(spark, [line("B1", "P1", -3), line("B1", "P2", 1)], ITEMS_DDL)
    cfg = CleaningConfig(negative_quantity_policy="drop")
    rows = collect_dicts(clean_basket_items(items, cfg, CleaningReport()))

    assert [r["product_id"] for r in rows] == ["P2"]


def test_politica_desconocida(spark):
    with pytest.raises(ValueError, match="negative_quantity_policy"):
        CleaningConfig(negative_quantity_policy="ni_idea")


def test_las_lineas_sin_unidades_se_descartan(spark):
    items = spark_df(spark, [line("B1", "P1", 0), line("B1", "P2", 1)], ITEMS_DDL)
    rows = collect_dicts(clean_basket_items(items, CleaningConfig(), CleaningReport()))

    assert [r["product_id"] for r in rows] == ["P2"]


def test_el_informe_cuenta_lo_corregido(spark):
    report = CleaningReport()
    items = spark_df(spark, [line("B1", "P1", 2), line("B1", "P1", -2), line("B2", "P1", 1)], ITEMS_DDL)
    clean_basket_items(items, CleaningConfig(), report)

    assert report.rows_affected("basket_items", "quantity negativa") == 1
    assert report.rows_affected("basket_items", "linea de ticket duplicada") == 1
    assert (
        report.rows_affected(
            "basket_items", "duplicados que solo afloran tras corregir el signo"
        )
        == 1
    )


# --------------------------------------------------------------------------------------
# Limpieza completa: cabecera de ticket, clientes y nulos legitimos
# --------------------------------------------------------------------------------------


def customer(customer_id: str, signup: str, city: str | None = "Madrid") -> dict:
    return {
        "customer_id": customer_id,
        "signup_date": signup,
        "country": "Espana",
        "city": city,
        "household_size_est": 2,
        "loyalty_tier": "gold",
        "preferred_channel": "app",
        "churn_label": False,
    }


def basket(basket_id: str, customer_id: str | None, day: str, total: float, channel: str = "app") -> dict:
    return {
        "basket_id": basket_id,
        "customer_id": customer_id,
        "channel": channel,
        "basket_date": f"{day} 10:00:00",
        "store_id": "S001" if channel == "store" else None,
        "total_amount": total,
    }


@pytest.fixture(scope="module")
def cleaned(spark):
    """Limpieza completa de un dataset con un defecto de cada tipo."""
    tables = make_tables(
        spark,
        customers=[
            # El alta es posterior a su primera compra: hay que corregirla.
            customer("C1", "2024-03-01"),
            customer("C2", "2023-01-01", city=None),
        ],
        products=[product("P1", "Leche"), product("P2", "LECHE"), product("P3", "Pan")],
        baskets=[
            # 2 x 2.0 + 1 x 2.0 = 6.0, pero la cabecera declara 300: outlier.
            basket("B1", "C1", "2024-01-10", 300.0),
            basket("B2", "C2", "2024-01-11", 4.0),
            basket("B3", None, "2024-01-12", 2.0, channel="store"),
        ],
        basket_items=[
            line("B1", "P1", 2),
            line("B1", "P1", -2),  # duplicado con signo invertido
            line("B1", "P3", 1),
            line("B2", "P2", 2),
            line("B3", "P3", 1),
        ],
        promotions=[
            {
                "promotion_id": "PR1",
                "product_id": "P1",
                "promo_type": "2x1",
                "discount_value": 0.5,
                "start_date": "2024-02-01",
                "end_date": "2024-01-01",  # ventana invertida
            }
        ],
        session_events=[
            {"session_id": "S1", "product_id": "P1", "event_type": "view", "event_timestamp": "2024-01-10 09:00:00"},
            {"session_id": "S1", "product_id": "P1", "event_type": "view", "event_timestamp": "2024-01-10 09:00:00"},
        ],
    )
    return clean_all(tables, CleaningConfig())


def test_el_importe_se_recalcula_desde_las_lineas(cleaned):
    tables, _ = cleaned
    b1 = {r["basket_id"]: r for r in collect_dicts(tables["baskets"])}["B1"]

    assert b1["total_amount"] == pytest.approx(6.0)
    assert b1["total_amount_raw"] == pytest.approx(300.0)
    assert b1["total_amount_is_outlier"] is True
    assert b1["n_lines"] == 2
    assert b1["n_units"] == 3


def test_una_cesta_normal_no_se_marca_como_outlier(cleaned):
    tables, _ = cleaned
    b2 = {r["basket_id"]: r for r in collect_dicts(tables["baskets"])}["B2"]

    assert b2["total_amount"] == pytest.approx(4.0)
    assert b2["total_amount_is_outlier"] is False


def test_la_compra_anonima_sobrevive(cleaned):
    """`customer_id` nulo es una compra legitima de tienda, no un defecto."""
    tables, _ = cleaned
    b3 = {r["basket_id"]: r for r in collect_dicts(tables["baskets"])}["B3"]

    assert b3["customer_id"] is None
    assert b3["is_anonymous"] is True
    assert tables["baskets"].count() == 3


def test_el_alta_posterior_a_la_compra_se_corrige(cleaned):
    tables, _ = cleaned
    c1 = {r["customer_id"]: r for r in collect_dicts(tables["customers"])}["C1"]

    assert str(c1["signup_date"]) == "2024-01-10"
    assert str(c1["signup_date_raw"]) == "2024-03-01"
    assert c1["signup_date_corrected"] is True


def test_el_alta_correcta_no_se_toca(cleaned):
    tables, _ = cleaned
    c2 = {r["customer_id"]: r for r in collect_dicts(tables["customers"])}["C2"]

    assert str(c2["signup_date"]) == "2023-01-01"
    assert c2["signup_date_corrected"] is False


def test_la_ciudad_nula_se_rellena_y_se_marca(cleaned):
    tables, _ = cleaned
    c2 = {r["customer_id"]: r for r in collect_dicts(tables["customers"])}["C2"]

    assert c2["city"] == "Desconocida"
    assert c2["city_is_missing"] is True


def test_la_promocion_con_ventana_invertida_se_marca_pero_no_se_borra(cleaned):
    """Borrarla dejaria huerfana cualquier linea que la referencie."""
    tables, _ = cleaned
    rows = collect_dicts(tables["promotions"])

    assert len(rows) == 1
    assert rows[0]["date_range_invalid"] is True


def test_los_eventos_duplicados_se_eliminan(cleaned):
    tables, _ = cleaned

    assert tables["session_events"].count() == 1


def test_las_siete_tablas_salen_de_la_limpieza(cleaned):
    tables, _ = cleaned

    assert set(tables) == {
        "customers",
        "products",
        "promotions",
        "baskets",
        "basket_items",
        "sessions",
        "session_events",
    }


def test_el_informe_se_lee_en_markdown(cleaned):
    _, report = cleaned
    markdown = report.to_markdown()

    assert "| `basket_items` |" in markdown
    assert "LECHE" in markdown  # la traduccion de categorias queda documentada


def test_el_informe_falla_claro_ante_un_paso_inexistente(cleaned):
    _, report = cleaned

    with pytest.raises(KeyError):
        report.rows_affected("basket_items", "problema inventado")

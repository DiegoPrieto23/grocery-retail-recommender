"""Tests de las nueve preguntas de negocio de la Tarea 1 (`src/eda/questions.py`).

Estas consultas vivian dentro del notebook de EDA, donde no habia forma de comprobarlas:
una celda que devuelve una tabla plausible parece correcta aunque el `JOIN` este mal. Al
sacarlas a funciones se pueden ejercitar sobre un dataset diminuto donde **cada cifra se
calcula a mano**.

El dataset de apoyo (todas las cantidades elegidas para que las cuentas salgan redondas):

| Cesta | Cliente | Canal  | Dia        | Lineas                       | Total |
| ----- | ------- | ------ | ---------- | ---------------------------- | ----: |
| B1    | C1 gold | online | 2024-01-10 | Leche 2ud 6 EUR, Pan 1ud 4   |    10 |
| B2    | C1 gold | store  | 2024-01-20 | Leche 1ud 3, Detergente 1 17 |    20 |
| B3    | C2 gold | online | 2024-12-15 | Turron 2ud 30, Leche 2ud 10  |    40 |
| B4    | C3 silv | store  | 2024-01-15 | Detergente 1ud 30            |    30 |
| B5    | anonima | store  | 2024-01-15 | Pan 1ud 5                    |     5 |

C4 (bronze) no tiene ni una compra: esta para que se vea si un `LEFT JOIN` lo pierde.
"""

from __future__ import annotations

import math

import pytest

from src.eda import questions as q
from tests.helpers import collect_dicts, register_eda_views

CUSTOMERS = [
    {"customer_id": "C1", "loyalty_tier": "gold", "household_size_est": 1, "churn_label": False},
    {"customer_id": "C2", "loyalty_tier": "gold", "household_size_est": 4, "churn_label": True},
    {"customer_id": "C3", "loyalty_tier": "silver", "household_size_est": 2, "churn_label": True},
    {"customer_id": "C4", "loyalty_tier": "bronze", "household_size_est": 2, "churn_label": False},
]

PRODUCTS = [
    {"product_id": "P1", "department": "Frescos", "category": "Leche",
     "is_private_label": True, "typical_repurchase_days": 6},
    {"product_id": "P2", "department": "Frescos", "category": "Pan",
     "is_private_label": False, "typical_repurchase_days": 4},
    {"product_id": "P3", "department": "Drogueria", "category": "Detergente",
     "is_private_label": False, "typical_repurchase_days": 45},
    {"product_id": "P4", "department": "Despensa", "category": "Turron y mazapan",
     "is_private_label": False, "typical_repurchase_days": 300},
    # Producto del surtido que no se vende nunca: no debe aparecer en Q2.
    {"product_id": "P5", "department": "Bebidas", "category": "Cerveza",
     "is_private_label": True, "typical_repurchase_days": 10},
]

# La promocion cubre 2024-01-01..2024-01-12, asi que solo alcanza al 10 de enero.
PROMOTIONS = [
    {"promotion_id": "PM1", "product_id": "P1", "promo_type": "2x1",
     "start_date": "2024-01-01", "end_date": "2024-01-12"},
]


def _basket(basket_id, customer_id, channel, ts, day, total, n_lines, n_units, anon=False):
    return {
        "basket_id": basket_id, "customer_id": customer_id, "channel": channel,
        "basket_date": ts, "basket_day": day, "total_amount": total,
        "n_lines": n_lines, "n_units": n_units, "is_anonymous": anon,
    }


BASKETS = [
    _basket("B1", "C1", "online", "2024-01-10 20:00:00", "2024-01-10", 10.0, 2, 3),
    _basket("B2", "C1", "store", "2024-01-20 12:00:00", "2024-01-20", 20.0, 2, 2),
    _basket("B3", "C2", "online", "2024-12-15 21:00:00", "2024-12-15", 40.0, 2, 4),
    _basket("B4", "C3", "store", "2024-01-15 12:00:00", "2024-01-15", 30.0, 1, 1),
    _basket("B5", None, "store", "2024-01-15 13:00:00", "2024-01-15", 5.0, 1, 1, anon=True),
]


def _line(basket_id, product_id, quantity, amount, promotion_id=None):
    return {
        "basket_id": basket_id, "product_id": product_id, "quantity": quantity,
        "promotion_id": promotion_id, "line_amount": amount,
        "is_promo": promotion_id is not None,
    }


BASKET_ITEMS = [
    _line("B1", "P1", 2, 6.0, "PM1"),
    _line("B1", "P2", 1, 4.0),
    _line("B2", "P1", 1, 3.0),
    _line("B2", "P3", 1, 17.0),
    _line("B3", "P4", 2, 30.0),
    _line("B3", "P1", 2, 10.0),
    _line("B4", "P3", 1, 30.0),
    _line("B5", "P2", 1, 5.0),
]

RFM = [
    {"customer_id": "C1", "recency_days": 5, "frequency": 20, "monetary": 400.0,
     "avg_ticket": 20.0, "avg_days_between_baskets": 18.0, "rfm_segment": "Campeones",
     "churn_label": False, "has_purchases": True},
    {"customer_id": "C2", "recency_days": 80, "frequency": 5, "monetary": 200.0,
     "avg_ticket": 40.0, "avg_days_between_baskets": 60.0, "rfm_segment": "En riesgo",
     "churn_label": True, "has_purchases": True},
    {"customer_id": "C3", "recency_days": 90, "frequency": 4, "monetary": 100.0,
     "avg_ticket": 25.0, "avg_days_between_baskets": 70.0, "rfm_segment": "En riesgo",
     "churn_label": True, "has_purchases": True},
    {"customer_id": "C4", "recency_days": None, "frequency": 0, "monetary": 0.0,
     "avg_ticket": None, "avg_days_between_baskets": None, "rfm_segment": "Nuevos",
     "churn_label": False, "has_purchases": False},
]

# Ciclo observado decreciente con el tamano del hogar (C1 vive solo, C2 son cuatro).
REPURCHASE = [
    {"customer_id": "C1", "category": "Leche", "typical_repurchase_days": 6,
     "observed_repurchase_days": 9.0, "due_for_repurchase": True},
    {"customer_id": "C3", "category": "Leche", "typical_repurchase_days": 6,
     "observed_repurchase_days": 7.0, "due_for_repurchase": True},
    {"customer_id": "C2", "category": "Leche", "typical_repurchase_days": 6,
     "observed_repurchase_days": 5.0, "due_for_repurchase": False},
    {"customer_id": "C1", "category": "Detergente", "typical_repurchase_days": 45,
     "observed_repurchase_days": 60.0, "due_for_repurchase": False},
    {"customer_id": "C3", "category": "Detergente", "typical_repurchase_days": 45,
     "observed_repurchase_days": 50.0, "due_for_repurchase": False},
    {"customer_id": "C2", "category": "Detergente", "typical_repurchase_days": 45,
     "observed_repurchase_days": 40.0, "due_for_repurchase": False},
]

AFFINITY = [
    {"antecedent": "Cerveza", "consequent": "Snacks y aperitivos", "n_baskets_both": 600,
     "support": 0.06, "confidence": 0.50, "lift": 3.1},
    {"antecedent": "Pasta", "consequent": "Salsa de tomate", "n_baskets_both": 700,
     "support": 0.07, "confidence": 0.60, "lift": 2.4},
    # Lift altisimo con soporte ridiculo: el filtro tiene que dejarlo fuera.
    {"antecedent": "Leche", "consequent": "Pan", "n_baskets_both": 100,
     "support": 0.01, "confidence": 0.20, "lift": 9.9},
]

SESSIONS = [
    {"session_id": "S1", "customer_id": "C1", "device_type": "app",
     "converted": True, "basket_id": "B1"},
    {"session_id": "S2", "customer_id": "C2", "device_type": "web",
     "converted": False, "basket_id": None},
    {"session_id": "S3", "customer_id": "C3", "device_type": "app",
     "converted": True, "basket_id": "B4"},
]


def _event(session_id, product_id, event_type, ts):
    return {"session_id": session_id, "product_id": product_id,
            "event_type": event_type, "event_timestamp": ts}


SESSION_EVENTS = [
    _event("S1", "P1", "view", "2024-01-10 19:00:00"),
    _event("S1", "P3", "view", "2024-01-10 19:05:00"),
    _event("S1", "P1", "add_to_cart", "2024-01-10 19:10:00"),
    _event("S1", "P3", "add_to_cart", "2024-01-10 19:12:00"),  # abandonado: B1 no lo tiene
    _event("S2", "P5", "view", "2024-02-01 10:00:00"),         # sesion no convertida
    _event("S3", "P3", "view", "2024-01-15 11:00:00"),
    _event("S3", "P3", "add_to_cart", "2024-01-15 11:30:00"),
]


@pytest.fixture(scope="module")
def views(spark):
    register_eda_views(
        spark,
        customers=CUSTOMERS,
        products=PRODUCTS,
        promotions=PROMOTIONS,
        baskets=BASKETS,
        basket_items=BASKET_ITEMS,
        rfm=RFM,
        repurchase_features=REPURCHASE,
        affinity_category=AFFINITY,
        sessions=SESSIONS,
        session_events=SESSION_EVENTS,
    )
    return spark


def by(rows: list[dict], key: str) -> dict:
    """Indexa una lista de filas por el valor de una columna."""
    return {row[key]: row for row in rows}


# --------------------------------------------------------------------------------------
# Q1 - Quien sostiene la facturacion
# --------------------------------------------------------------------------------------
def test_q1_agrega_facturacion_y_ticket_por_nivel(views):
    rows = by(collect_dicts(q.q1_value_by_tier(views)), "loyalty_tier")

    assert rows["gold"]["clientes"] == 2
    assert rows["gold"]["cestas"] == 3           # B1, B2 de C1 y B3 de C2
    assert rows["gold"]["facturacion"] == 70.0
    assert rows["gold"]["ticket_medio"] == pytest.approx(23.33)
    assert rows["silver"]["facturacion"] == 30.0


def test_q1_un_cliente_sin_compras_sigue_contando(views):
    rows = by(collect_dicts(q.q1_value_by_tier(views)), "loyalty_tier")

    # El LEFT JOIN lo conserva: cuenta como cliente, con cero cestas y cero facturacion.
    assert rows["bronze"]["clientes"] == 1
    assert rows["bronze"]["cestas"] == 0
    assert rows["bronze"]["facturacion"] == 0.0
    assert rows["bronze"]["ticket_medio"] is None


def test_q1_la_cesta_anonima_no_se_atribuye_a_nadie(views):
    total = sum(r["facturacion"] for r in collect_dicts(q.q1_value_by_tier(views)))

    # 105 EUR vendidos, pero 5 son de la cesta sin cliente: no cuelgan de ningun nivel.
    assert total == 100.0


def test_q1_el_indice_de_valor_compara_peso_en_venta_con_peso_en_clientes(views):
    pdf = q.add_share_columns(q.q1_value_by_tier(views).toPandas())
    rows = by(pdf.to_dict("records"), "loyalty_tier")

    assert rows["gold"]["pct_clientes"] == pytest.approx(50.0)
    assert rows["gold"]["pct_facturacion"] == pytest.approx(70.0)
    assert rows["gold"]["indice_valor"] == pytest.approx(1.40)
    assert rows["bronze"]["indice_valor"] == pytest.approx(0.0)


def test_q1_ticket_y_frecuencia_por_tamano_de_hogar(views):
    rows = by(collect_dicts(q.q1_basket_by_household(views)), "tamano_hogar")

    assert rows[1]["ticket_medio"] == pytest.approx(15.0)   # (10 + 20) / 2
    assert rows[1]["cestas_por_cliente"] == pytest.approx(2.0)
    # Hogar de 2: C3 con una cesta y C4 con ninguna -> media de 0,5 cestas por cliente.
    assert rows[2]["clientes"] == 2
    assert rows[2]["cestas_por_cliente"] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Q2 - Que se vende y cuanta marca blanca hay
# --------------------------------------------------------------------------------------
def test_q2_importe_unidades_y_cestas_por_categoria(views):
    rows = by(collect_dicts(q.q2_sales_by_category(views)), "categoria")

    assert rows["Detergente"] == {
        "departamento": "Drogueria", "categoria": "Detergente",
        "cestas": 2, "unidades": 2, "importe": 47.0,
    }
    assert rows["Leche"]["importe"] == 19.0      # 6 + 3 + 10
    assert rows["Leche"]["unidades"] == 5        # 2 + 1 + 2
    assert rows["Leche"]["cestas"] == 3


def test_q2_una_categoria_sin_ventas_no_aparece(views):
    categorias = {r["categoria"] for r in collect_dicts(q.q2_sales_by_category(views))}

    assert "Cerveza" not in categorias           # P5 esta en el surtido y no se vendio


def test_q2_cuota_de_marca_blanca_por_departamento(views):
    rows = by(collect_dicts(q.q2_private_label_by_department(views)), "departamento")

    # Frescos vende 28 EUR, de los que 19 son Leche (marca propia): 67,86 %.
    assert rows["Frescos"]["pct_marca_blanca"] == pytest.approx(67.86)
    assert rows["Drogueria"]["pct_marca_blanca"] == 0.0


def test_q2_concentracion_del_surtido(views):
    pdf = q.q2_sales_by_category(views).toPandas()

    # Cuotas 44,8 / 28,6 / 18,1 / 8,6: hacen falta tres categorias para pasar del 80 %.
    assert q.categories_covering(pdf) == 3
    assert q.categories_covering(pdf, threshold=40.0) == 1


def test_q2_cuota_global_de_marca_blanca_pondera_por_importe(views):
    pdf = q.q2_private_label_by_department(views).toPandas()

    # 19 EUR de marca propia sobre 105 EUR vendidos.
    assert q.private_label_share(pdf) == pytest.approx(100 * 19 / 105, abs=0.02)


def test_q2_umbral_de_cobertura_invalido(views):
    pdf = q.q2_sales_by_category(views).toPandas()

    with pytest.raises(ValueError):
        q.categories_covering(pdf, threshold=0.0)


# --------------------------------------------------------------------------------------
# Q3 - Estacionalidad
# --------------------------------------------------------------------------------------
def test_q3_venta_mensual(views):
    rows = by(collect_dicts(q.q3_monthly_sales(views)), "mes")

    assert rows["2024-01"] == {"mes": "2024-01", "importe": 65.0, "cestas": 4}
    assert rows["2024-12"] == {"mes": "2024-12", "importe": 40.0, "cestas": 1}


def test_q3_el_indice_mide_cuota_del_mes_no_importe_absoluto(views):
    rows = {
        (r["categoria"], r["mes"]): r["indice_estacional"]
        for r in collect_dicts(q.q3_seasonal_index(views))
    }

    # Leche pesa 9/65 en enero y 10/40 en diciembre. Su cuota media es 0,1942, asi que
    # el indice es 0,713 en enero y 1,287 en diciembre: vende menos en diciembre en
    # importe absoluto y aun asi pesa mas dentro del mes.
    assert rows[("Leche", 1)] == pytest.approx(0.713, abs=0.001)
    assert rows[("Leche", 12)] == pytest.approx(1.287, abs=0.001)


def test_q3_una_categoria_de_un_solo_mes_tiene_indice_uno(views):
    rows = {
        (r["categoria"], r["mes"]): r["indice_estacional"]
        for r in collect_dicts(q.q3_seasonal_index(views))
    }

    assert rows[("Turron y mazapan", 12)] == pytest.approx(1.0)


def test_q3_el_pico_estacional_sale_del_mes_de_mayor_indice(views):
    pdf = q.q3_seasonal_index(views).toPandas()
    matrix = q.seasonal_matrix(pdf, categories=("Leche", "Turron y mazapan"))
    peaks = q.seasonal_peaks(matrix)

    assert list(matrix.index) == ["Leche", "Turron y mazapan"]
    assert peaks.loc["Leche", "mes_pico"] == 12
    assert peaks.loc["Leche", "indice_en_el_pico"] == pytest.approx(1.29, abs=0.01)


# --------------------------------------------------------------------------------------
# Q4 - Anatomia de la cesta
# --------------------------------------------------------------------------------------
def test_q4_tamano_de_cesta_por_canal(views):
    rows = by(collect_dicts(q.q4_basket_by_channel(views)), "canal")

    assert rows["online"]["cestas"] == 2
    assert rows["online"]["lineas_medias"] == pytest.approx(2.0)
    assert rows["online"]["ticket_medio"] == pytest.approx(25.0)
    assert rows["store"]["cestas"] == 3
    assert rows["store"]["unidades_medias"] == pytest.approx(1.33)


def test_q4_las_cestas_anonimas_se_cuentan_como_tales(views):
    rows = by(collect_dicts(q.q4_basket_by_channel(views)), "canal")

    assert rows["store"]["pct_anonimas"] == pytest.approx(33.33)
    assert rows["online"]["pct_anonimas"] == 0.0


def test_q4_reparto_horario_por_canal(views):
    rows = {
        (r["canal"], r["hora"]): r["cestas"]
        for r in collect_dicts(q.q4_hour_profile(views))
    }

    assert rows[("online", 20)] == 1 and rows[("online", 21)] == 1
    assert rows[("store", 12)] == 2 and rows[("store", 13)] == 1


def test_q4_distribucion_de_lineas_recorta_la_cola(views):
    rows = by(collect_dicts(q.q4_lines_distribution(views)), "lineas")
    assert rows[1]["cestas"] == 2 and rows[2]["cestas"] == 3

    # Con el tope en 1, las cestas de 2 lineas se quedan fuera.
    recortado = by(collect_dicts(q.q4_lines_distribution(views, max_lines=1)), "lineas")
    assert set(recortado) == {1}


# --------------------------------------------------------------------------------------
# Q5 - Afinidad de cesta
# --------------------------------------------------------------------------------------
def test_q5_el_soporte_minimo_descarta_los_pares_anecdoticos(views):
    rows = collect_dicts(q.q5_top_affinity_pairs(views, min_baskets=500))

    # Leche -> Pan tiene el lift mas alto (9,9) y solo 100 cestas: no debe salir.
    assert [r["disparadora"] for r in rows] == ["Cerveza", "Pasta"]
    assert rows[0]["pct_confianza"] == pytest.approx(50.0)
    assert rows[0]["pct_soporte"] == pytest.approx(6.0)


def test_q5_sin_soporte_minimo_manda_el_lift(views):
    rows = collect_dicts(q.q5_top_affinity_pairs(views, min_baskets=0))

    assert [r["disparadora"] for r in rows] == ["Leche", "Cerveza", "Pasta"]


def test_q5_compara_cada_par_declarado_con_lo_medido(views):
    pdf = q.q5_expected_pairs(views)
    rows = by(pdf.to_dict("records"), "disparadora")

    assert len(pdf) == len(q.EXPECTED_PAIRS)
    assert rows["Cerveza"]["lift_medido"] == pytest.approx(3.1)
    assert rows["Cerveza"]["ratio"] == pytest.approx(1.03)
    # Un par declarado que no aparece en la tabla medida se queda a nulo, no desaparece.
    assert math.isnan(rows["Panales"]["lift_medido"])


# --------------------------------------------------------------------------------------
# Q6 - Ciclo de recompra (validacion de la Tarea 2)
# --------------------------------------------------------------------------------------
def test_q6_ciclo_teorico_frente_a_observado_por_categoria(views):
    rows = by(collect_dicts(q.q6_cycle_by_category(views, min_customers=3)), "categoria")

    assert rows["Leche"]["ciclo_teorico"] == 6
    assert rows["Leche"]["ciclo_observado"] == pytest.approx(7.0)   # mediana de 5, 7, 9
    assert rows["Leche"]["clientes"] == 3
    assert rows["Leche"]["pct_toca_reponer"] == pytest.approx(66.7)
    assert rows["Detergente"]["ciclo_observado"] == pytest.approx(50.0)


def test_q6_el_minimo_de_clientes_descarta_categorias_sin_evidencia(views):
    rows = collect_dicts(q.q6_cycle_by_category(views, min_customers=4))

    assert rows == []


def test_q6_el_ciclo_se_acorta_en_hogares_grandes(views):
    rows = {
        (r["categoria"], r["tamano_hogar"]): r["ciclo_observado"]
        for r in collect_dicts(q.q6_cycle_by_household(views, ("Leche", "Detergente")))
    }

    assert rows[("Leche", 1)] == pytest.approx(9.0)   # C1 vive solo
    assert rows[("Leche", 2)] == pytest.approx(7.0)
    assert rows[("Leche", 4)] == pytest.approx(5.0)   # C2 son cuatro en casa
    assert ("Pan", 1) not in rows                     # no se pidio esa categoria


def test_q6_resumen_de_recompras_vencidas(views):
    row = collect_dicts(q.q6_due_summary(views))[0]

    assert row["pares_cliente_categoria"] == 6
    assert row["con_recompra_vencida"] == 2
    assert row["pct"] == pytest.approx(33.3)
    assert row["clientes_afectados"] == 2


def test_q6_la_correlacion_de_rangos_mide_el_orden_no_la_magnitud(views):
    pdf = q.q6_cycle_by_category(views, min_customers=3).toPandas()

    # Teoricos 6 y 45, observados 7 y 50: el orden se conserva entero.
    assert q.cycle_rank_correlation(pdf) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Q7 - Quien abandona y cuanto valor esta en riesgo
# --------------------------------------------------------------------------------------
def test_q7_segmentos_rfm_con_su_tasa_de_abandono(views):
    rows = collect_dicts(q.q7_rfm_segments(views))
    indexed = by(rows, "segmento")

    assert rows[0]["segmento"] == "Campeones"        # ordenado por gasto total
    assert indexed["En riesgo"]["clientes"] == 2
    assert indexed["En riesgo"]["gasto_total"] == 300.0
    assert indexed["En riesgo"]["pct_churn"] == pytest.approx(100.0)
    assert indexed["Campeones"]["pct_churn"] == 0.0


def test_q7_perfil_medio_de_activos_frente_a_los_que_abandonan(views):
    rows = by(collect_dicts(q.q7_churn_profile(views)), "estado")

    # C4 no tiene compras: queda fuera de las dos filas.
    assert rows["Activos"]["clientes"] == 1
    assert rows["Abandonan"]["clientes"] == 2
    assert rows["Abandonan"]["recencia_media"] == pytest.approx(85.0)
    assert rows["Abandonan"]["dias_entre_cestas"] == pytest.approx(65.0)


def test_q7_cuanto_gasto_concentran_los_segmentos_de_riesgo(views):
    pdf = q.q7_rfm_segments(views).toPandas()
    share = q.at_risk_share(pdf, segments=("En riesgo",))

    assert share["clientes"] == 2
    assert share["pct_clientes"] == pytest.approx(50.0)
    assert share["gasto_total"] == pytest.approx(300.0)
    assert share["pct_gasto"] == pytest.approx(100 * 300 / 700)


# --------------------------------------------------------------------------------------
# Q8 - Efectividad promocional
# --------------------------------------------------------------------------------------
def test_q8_el_uplift_se_mide_sobre_la_rejilla_completa_de_producto_por_dia(views):
    rows = by(collect_dicts(q.q8_promo_uplift(views)), "estado")

    # P1 esta en promocion solo el 10 de enero (2 ud vendidas): 2,0 ud/dia.
    assert rows["Con promocion"]["dias_producto"] == 1
    assert rows["Con promocion"]["unidades_por_dia"] == pytest.approx(2.0)
    # Fuera de la ventana quedan 3 dias del calendario con 3 unidades en total: 1,0 ud/dia.
    # El 15 de enero, sin ventas de P1, cuenta como dia de cero y no se ignora.
    assert rows["Sin promocion"]["dias_producto"] == 3
    assert rows["Sin promocion"]["unidades_por_dia"] == pytest.approx(1.0)


def test_q8_el_ratio_de_uplift_sale_de_las_dos_filas(views):
    pdf = q.q8_promo_uplift(views).toPandas()

    assert q.promo_uplift_ratio(pdf) == pytest.approx(2.0)


def test_q8_venta_en_promocion_por_departamento(views):
    rows = by(collect_dicts(q.q8_promo_share_by_department(views)), "departamento")

    # De los 28 EUR de Frescos, 6 se vendieron con la promocion PM1.
    assert rows["Frescos"]["pct_venta_en_promo"] == pytest.approx(21.43)
    assert rows["Drogueria"]["pct_venta_en_promo"] == 0.0


def test_q8_desglose_por_mecanica_promocional(views):
    rows = by(collect_dicts(q.q8_promo_by_type(views)), "tipo")

    assert rows["2x1"]["promociones"] == 1
    assert rows["2x1"]["lineas"] == 1
    assert rows["2x1"]["importe"] == 6.0
    assert rows["2x1"]["unidades_por_linea"] == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# Q9 - Embudo online y fiabilidad de la senal de sesion
# --------------------------------------------------------------------------------------
def test_q9_embudo_por_dispositivo(views):
    rows = by(collect_dicts(q.q9_funnel_by_device(views)), "dispositivo")

    assert rows["app"]["sesiones"] == 2
    assert rows["app"]["con_add_to_cart"] == 2
    assert rows["app"]["pct_conversion"] == pytest.approx(100.0)
    # La sesion web ni anadio ni convirtio.
    assert rows["web"]["con_add_to_cart"] == 0
    assert rows["web"]["pct_conversion"] == 0.0


def test_q9_no_todo_lo_anadido_al_carrito_acaba_en_el_ticket(views):
    row = collect_dicts(q.q9_event_to_basket_overlap(views))[0]

    # 3 pares (sesion, producto) anadidos en sesiones convertidas; 2 estan en la cesta.
    assert row["evento"] == "add_to_cart"
    assert row["productos"] == 3
    assert row["acabaron_en_la_cesta"] == 2
    assert row["pct"] == pytest.approx(66.67)


def test_q9_las_vistas_son_mas_ruidosas_que_los_anadidos(views):
    vistas = collect_dicts(q.q9_event_to_basket_overlap(views, event_type="view"))[0]

    # Solo entran las sesiones convertidas: la vista de P5 en S2 no cuenta.
    assert vistas["productos"] == 3
    assert vistas["pct"] == pytest.approx(66.67)


def test_q9_no_toda_la_cesta_deja_rastro_online(views):
    row = collect_dicts(q.q9_basket_seen_online(views))[0]

    # B1 tiene Leche y Pan, pero el Pan no se vio ni se anadio en la sesion.
    assert row["lineas_del_ticket"] == 3
    assert row["con_rastro_online"] == 2
    assert row["pct"] == pytest.approx(66.67)


def test_q9_tipo_de_evento_invalido(views):
    with pytest.raises(ValueError):
        q.q9_event_to_basket_overlap(views, event_type="purchase")

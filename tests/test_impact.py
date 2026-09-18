"""Tests del resumen de impacto de negocio (Fase 5, Tarea 4 de `CHALLENGE.md`).

Traducir una metrica de modelo a euros es donde mas facil resulta colar un factor de mas.
Por eso la aritmetica esta en funciones puras (`src/impact/model.py`) y aqui se comprueba
con numeros redondos que se pueden multiplicar de cabeza:

    1.000 cestas online/mes x (0,20 - 0,10) de delta = 100 cestas con acierto nuevo
    100 x 0,10 de incrementalidad                    =  10 unidades extra
    10 x 2,00 EUR de linea media                     =  20 EUR de venta
    20 x 0,25 de margen                              =   5 EUR de margen al mes
"""

from __future__ import annotations

import json

import pytest

from src.impact.config import ImpactConfig
from src.impact.model import (
    MONTHS_PER_YEAR,
    CrossSellImpact,
    NBAImpact,
    blended_margin_rate,
    sweep_incremental_rate,
)
from src.impact import pipeline


@pytest.fixture
def cross_sell() -> CrossSellImpact:
    return CrossSellImpact(
        online_baskets_per_month=1_000,
        hit_rate_model=0.20,
        hit_rate_baseline=0.10,
        incremental_rate=0.10,
        avg_line_amount=2.0,
        margin_rate=0.25,
    )


@pytest.fixture
def nba() -> NBAImpact:
    return NBAImpact(
        customers=1_000,
        policy_value_per_wave=500.0,
        best_trivial_value_per_wave=200.0,
        floor_value_per_wave=150.0,
        waves_per_year=12,
    )


# --------------------------------------------------------------------------------------
# Cross-sell: la cadena de multiplicaciones, paso a paso
# --------------------------------------------------------------------------------------
def test_la_cadena_de_cross_sell_multiplica_en_el_orden_esperado(cross_sell):
    assert cross_sell.hit_rate_delta == pytest.approx(0.10)
    assert cross_sell.baskets_with_new_hit == pytest.approx(100.0)
    assert cross_sell.extra_units_per_month == pytest.approx(10.0)
    assert cross_sell.revenue_per_month == pytest.approx(20.0)
    assert cross_sell.margin_per_month == pytest.approx(5.0)
    assert cross_sell.margin_per_year == pytest.approx(5.0 * MONTHS_PER_YEAR)


def test_la_mejora_relativa_no_depende_de_ningun_supuesto(cross_sell):
    # 10 puntos sobre un baseline de 10 puntos: el doble de cestas con acierto.
    assert cross_sell.relative_lift == pytest.approx(1.0)
    # Y no cambia al mover la incrementalidad, que es justo lo que la hace publicable.
    assert cross_sell.with_incremental_rate(0.5).relative_lift == pytest.approx(1.0)


def test_un_modelo_que_no_mejora_al_baseline_no_vale_nada(cross_sell):
    plano = CrossSellImpact(
        online_baskets_per_month=1_000,
        hit_rate_model=0.10,
        hit_rate_baseline=0.10,
        incremental_rate=0.99,
        avg_line_amount=2.0,
        margin_rate=0.25,
    )

    # Ni con una incrementalidad del 99 %: el delta es cero y todo lo demas lo multiplica.
    assert plano.margin_per_month == pytest.approx(0.0)


def test_reescalar_es_lineal_en_las_cestas(cross_sell):
    diez_veces = cross_sell.rescaled(10_000)

    assert diez_veces.margin_per_month == pytest.approx(10 * cross_sell.margin_per_month)
    assert diez_veces.relative_lift == pytest.approx(cross_sell.relative_lift)


def test_el_barrido_solo_mueve_la_incrementalidad(cross_sell):
    rows = sweep_incremental_rate(cross_sell, (0.05, 0.10, 0.20))

    assert [r["incrementalidad"] for r in rows] == [0.05, 0.10, 0.20]
    assert rows[0]["margen_extra_mes"] == pytest.approx(2.5)
    assert rows[1]["margen_extra_mes"] == pytest.approx(5.0)
    # El doble de incrementalidad, el doble de margen: la relacion es lineal.
    assert rows[2]["margen_extra_mes"] == pytest.approx(2 * rows[1]["margen_extra_mes"])


def test_un_baseline_a_cero_no_admite_mejora_relativa():
    sin_baseline = CrossSellImpact(
        online_baskets_per_month=1_000,
        hit_rate_model=0.20,
        hit_rate_baseline=0.0,
        incremental_rate=0.10,
        avg_line_amount=2.0,
        margin_rate=0.25,
    )

    with pytest.raises(ValueError):
        _ = sin_baseline.relative_lift


# --------------------------------------------------------------------------------------
# Margen mezclado
# --------------------------------------------------------------------------------------
def test_el_margen_se_pondera_por_lo_que_vende_cada_departamento():
    ventas = {"Frescos": 750.0, "Drogueria": 250.0}
    margenes = {"Frescos": 0.20, "Drogueria": 0.40}

    # 0,75 x 0,20 + 0,25 x 0,40 = 0,25. Un promedio simple daria 0,30.
    assert blended_margin_rate(ventas, margenes, default=0.25) == pytest.approx(0.25)


def test_un_departamento_sin_margen_declarado_usa_el_por_defecto():
    ventas = {"Desconocido": 100.0}

    assert blended_margin_rate(ventas, {}, default=0.33) == pytest.approx(0.33)


def test_sin_venta_no_hay_margen_que_ponderar():
    with pytest.raises(ValueError):
        blended_margin_rate({"Frescos": 0.0}, {"Frescos": 0.2}, default=0.25)


# --------------------------------------------------------------------------------------
# NBA
# --------------------------------------------------------------------------------------
def test_el_valor_del_nba_se_reparte_por_cliente_y_se_anualiza(nba):
    assert nba.value_per_customer_per_wave == pytest.approx(0.5)
    assert nba.value_per_year == pytest.approx(6_000.0)
    assert nba.floor_per_year == pytest.approx(1_800.0)


def test_el_uplift_sobre_lo_trivial_aisla_lo_que_aporta_elegir_a_quien(nba):
    assert nba.uplift_vs_trivial_per_wave == pytest.approx(300.0)


def test_reescalar_el_nba_mantiene_el_valor_por_cliente(nba):
    grande = nba.rescaled(100_000)

    assert grande.value_per_customer_per_wave == pytest.approx(
        nba.value_per_customer_per_wave
    )
    assert grande.policy_value_per_wave == pytest.approx(50_000.0)
    assert grande.floor_value_per_wave == pytest.approx(15_000.0)


def test_una_base_de_clientes_vacia_falla_claro():
    vacio = NBAImpact(
        customers=0,
        policy_value_per_wave=0.0,
        best_trivial_value_per_wave=0.0,
        floor_value_per_wave=0.0,
        waves_per_year=12,
    )

    with pytest.raises(ValueError):
        _ = vacio.value_per_customer_per_wave


# --------------------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------------------
def test_la_incrementalidad_es_una_probabilidad():
    with pytest.raises(ValueError):
        ImpactConfig(incremental_rate=1.5)


def test_hace_falta_al_menos_una_oleada_al_ano():
    with pytest.raises(ValueError):
        ImpactConfig(nba_waves_per_year=0)


# --------------------------------------------------------------------------------------
# Lectura de los informes de las Fases 3 y 4
# --------------------------------------------------------------------------------------
RECOMMENDER_METRICS = {
    "top_k": 5,
    "summary": [
        {"grupo": "total", "hit_rate@5": 0.20},
        {"grupo": "1 - nuevo, carrito vacio", "hit_rate@5": 0.05},
    ],
    "summary_popularity": [{"grupo": "total", "hit_rate@5": 0.10}],
}

NBA_METRICS = {
    "comparison": [
        {"politica": "no actuar siempre", "n_clientes": 1_000, "valor_total": 0.0},
        {"politica": "actuar siempre: recomendar_categoria", "n_clientes": 1_000,
         "valor_total": 200.0},
        {"politica": "actuar siempre: enviar_cupon_categoria", "n_clientes": 1_000,
         "valor_total": -50.0},
        {"politica": "politica de valor esperado", "n_clientes": 1_000, "valor_total": 500.0},
    ],
    "sensitivity_retention": [
        {"reduccion_churn": 0.0, "valor_politica": 150.0},
        {"reduccion_churn": 0.1, "valor_politica": 500.0},
    ],
}


@pytest.fixture
def cfg_con_metricas(tmp_path) -> ImpactConfig:
    rec = tmp_path / "rec.json"
    nba_path = tmp_path / "nba.json"
    rec.write_text(json.dumps(RECOMMENDER_METRICS), encoding="utf-8")
    nba_path.write_text(json.dumps(NBA_METRICS), encoding="utf-8")
    return ImpactConfig(recommender_metrics=rec, nba_metrics=nba_path)


def test_el_cross_sell_toma_el_total_y_no_un_perfil_suelto(cfg_con_metricas):
    baseline = pipeline.Baseline(
        months=1, baskets=1_000, baskets_per_month=1_000.0,
        online_baskets=1_000, online_baskets_per_month=1_000.0,
        lines=5_000, avg_line_amount=2.0, avg_lines_per_basket=5.0,
        customers_with_purchases=500, sales_by_department={"Frescos": 1.0},
        margin_rate=0.25,
    )

    impact = pipeline.build_cross_sell(cfg_con_metricas, baseline)

    assert impact.hit_rate_model == pytest.approx(0.20)   # la fila "total", no la del perfil 1
    assert impact.hit_rate_baseline == pytest.approx(0.10)
    assert impact.margin_per_month == pytest.approx(5.0)


def test_el_nba_compara_contra_la_mejor_alternativa_trivial(cfg_con_metricas):
    impact = pipeline.build_nba(cfg_con_metricas)

    assert impact.policy_value_per_wave == pytest.approx(500.0)
    # La mejor trivial son los 200 EUR de recomendar, no los -50 del cupon ni el 0.
    assert impact.best_trivial_value_per_wave == pytest.approx(200.0)
    # El suelo es la fila mas baja del barrido de retencion.
    assert impact.floor_value_per_wave == pytest.approx(150.0)


def test_una_fila_que_falta_en_el_informe_falla_claro(cfg_con_metricas):
    with pytest.raises(KeyError):
        pipeline._row(RECOMMENDER_METRICS["summary"], "grupo", "no existe")

"""Tests del registro de experimentos con MLflow (`src/tracking.py`).

Dos cosas que comprobar, y solo una necesita MLflow:

1. **Que se registra.** Los constructores de params y metricas son funciones puras sobre
   la configuracion y el resultado del pipeline. Se ejercitan con un resultado de mentira
   pero de la forma exacta que devuelven `recommender.pipeline.run` y `nba.pipeline.run`,
   asi que si alguien cambia una clave del resultado, esto falla.
2. **Que no estorba.** Con MLflow ausente o desactivado, `track` tiene que devolver un run
   inerte y no reventar. Es la garantia de que la CI, que no instala MLflow, sigue verde.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import tracking
from src.nba.config import NBAConfig
from src.recommender.config import RecommenderConfig


# --------------------------------------------------------------------------------------
# Activacion
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["off", "0", "false", "OFF", "No"])
def test_la_variable_de_entorno_apaga_el_registro(monkeypatch, flag):
    monkeypatch.setenv(tracking.ENV_FLAG, flag)

    assert tracking.is_enabled() is False


def test_apagado_el_run_es_inerte_y_no_falla(monkeypatch, tmp_path):
    monkeypatch.setenv(tracking.ENV_FLAG, "off")

    with tracking.track("experimento", "run") as run:
        assert run.active is False
        # Ninguna de estas llamadas debe hacer nada ni levantar nada.
        run.set_tags({"fase": "3"})
        run.log_params({"a": 1})
        run.log_metrics({"b": 2.0})
        run.log_artifact(tmp_path / "no_existe.json")


def test_sin_mlflow_instalado_no_se_registra(monkeypatch):
    monkeypatch.delenv(tracking.ENV_FLAG, raising=False)
    # Simula que el import falla, que es la situacion de la CI.
    monkeypatch.setitem(__import__("sys").modules, "mlflow", None)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert tracking.is_enabled() is False


# --------------------------------------------------------------------------------------
# Que se registra del recomendador
# --------------------------------------------------------------------------------------
def _summary_row(grupo: str, ndcg: float) -> dict:
    return {
        "grupo": grupo,
        "n_queries": 100,
        "ndcg@5": ndcg,
        "recall@5": ndcg / 2,
        "hit_rate@5": ndcg * 3,
        "n_target_medio": 4.0,
    }


RECOMMENDER_RESULT = {
    "summary": pd.DataFrame(
        [
            _summary_row("total", 0.03),
            _summary_row("1 - nuevo, carrito vacio", 0.02),
            _summary_row("4 - recurrente, con articulos", 0.04),
        ]
    ),
    "summary_no_session": pd.DataFrame([_summary_row("total", 0.025)]),
    "summary_popularity": pd.DataFrame([_summary_row("total", 0.02)]),
    "candidate_recall": pd.DataFrame(
        [{"grupo": "total", "n_queries": 100, "pool_recall": 0.31, "pool_size_medio": 155.0}]
    ),
    "by_category": pd.DataFrame(
        [
            {
                "grupo": "total",
                "n_queries": 100,
                "cat_hit_rate@5": 0.51,
                "cat_precision@5": 0.18,
                "sku_hit_rate@5": 0.12,
                "sku_precision@5": 0.025,
            }
        ]
    ),
    "valid_ndcg": 0.083,
    "n_test_queries": 18_000,
    "n_train_queries": 8_549,
}


def test_los_params_del_recomendador_llevan_ventanas_e_hiperparametros():
    params = tracking.recommender_params(RecommenderConfig())

    assert params["modelo"] == "lightgbm_lambdarank"
    assert params["top_k"] == RecommenderConfig().top_k
    # Las dos ventanas del split son lo primero que hay que poder comparar entre runs.
    assert params["fit_end"] == str(RecommenderConfig().fit_end)
    assert params["test_start"] == str(RecommenderConfig().test_start)
    assert params["seed"] == RecommenderConfig().seed


def test_las_metricas_del_recomendador_incluyen_las_dos_ablaciones():
    metrics = tracking.recommender_metrics(RECOMMENDER_RESULT, k=5)

    assert metrics["ndcg@5"] == pytest.approx(0.03)
    assert metrics["ndcg@5_sin_sesion"] == pytest.approx(0.025)
    assert metrics["ndcg@5_baseline_popularidad"] == pytest.approx(0.02)
    # Sin las tres en el mismo run, la comparacion entre ellas habria que hacerla a mano.
    assert metrics["pool_recall"] == pytest.approx(0.31)
    assert metrics["cat_hit_rate@5"] == pytest.approx(0.51)


def test_las_metricas_del_recomendador_se_desglosan_por_perfil():
    metrics = tracking.recommender_metrics(RECOMMENDER_RESULT, k=5)

    assert metrics["ndcg@5_perfil_1"] == pytest.approx(0.02)
    assert metrics["ndcg@5_perfil_4"] == pytest.approx(0.04)
    assert "ndcg@5_perfil_total" not in metrics


# --------------------------------------------------------------------------------------
# Que se registra del NBA
# --------------------------------------------------------------------------------------
NBA_RESULT = {
    "churn_metrics": {"auc": 0.85, "pr_auc": 0.86, "base_rate": 0.48, "lift_top_decile": 2.07},
    "purchase_metrics": {"auc": 0.76, "pr_auc": 0.22, "base_rate": 0.06, "lift_top_decile": 3.51},
    "comparison": pd.DataFrame(
        [
            {"politica": "no actuar siempre", "valor_total": 0.0, "pct_accion": 0.0},
            {"politica": "politica de valor esperado", "valor_total": 4012.0, "pct_accion": 0.76},
        ]
    ),
    "sensitivity_retention": pd.DataFrame(
        [
            {"reduccion_churn": 0.0, "valor_politica": 1209.6},
            {"reduccion_churn": 0.1, "valor_politica": 4012.0},
        ]
    ),
}


def test_los_params_del_nba_registran_los_supuestos_economicos():
    params = tracking.nba_params(NBAConfig())

    # Sin estos tres, comparar dos runs de la politica es adivinar.
    assert params["coupon_discount"] == pytest.approx(2.54)
    assert params["coupon_conversion_uplift"] == pytest.approx(1.35)
    assert params["coupon_churn_reduction"] == pytest.approx(0.10)
    assert params["retention_weeks"] == pytest.approx(4.0)


def test_las_metricas_del_nba_incluyen_el_valor_de_cada_politica():
    metrics = tracking.nba_metrics(NBA_RESULT)

    assert metrics["churn_auc"] == pytest.approx(0.85)
    assert metrics["purchase_pr_auc"] == pytest.approx(0.22)
    assert metrics["valor_politica_de_valor_esperado"] == pytest.approx(4012.0)
    assert metrics["pct_accion_politica_de_valor_esperado"] == pytest.approx(0.76)


def test_las_metricas_del_nba_guardan_el_suelo_del_barrido():
    metrics = tracking.nba_metrics(NBA_RESULT)

    # La fila pesimista es la que sostiene la conclusion de la Fase 4: se registra aparte.
    assert metrics["valor_politica_suelo_retencion"] == pytest.approx(1209.6)


# --------------------------------------------------------------------------------------
# Robustez del Run
# --------------------------------------------------------------------------------------
class _FakeMlflow:
    def __init__(self) -> None:
        self.params: dict = {}
        self.metrics: dict = {}
        self.artifacts: list[str] = []
        self.tags: dict = {}

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_artifact(self, path):
        self.artifacts.append(str(path))

    def set_tags(self, tags):
        self.tags.update(tags)


def test_los_valores_nulos_no_llegan_a_mlflow():
    fake = _FakeMlflow()
    run = tracking.Run(active=True, _mlflow=fake)

    run.log_params({"a": 1, "b": None})
    run.log_metrics({"x": 2.0, "y": None})

    assert fake.params == {"a": 1}
    assert fake.metrics == {"x": 2.0}


def test_un_artefacto_que_no_existe_no_tumba_el_pipeline(tmp_path):
    fake = _FakeMlflow()
    run = tracking.Run(active=True, _mlflow=fake)
    existente = tmp_path / "metrics.json"
    existente.write_text("{}", encoding="utf-8")

    run.log_artifact(existente)
    run.log_artifact(tmp_path / "no_existe.json")

    assert fake.artifacts == [str(existente)]


# --------------------------------------------------------------------------------------
# Nombres que MLflow acepta, y fallos que no se propagan
# --------------------------------------------------------------------------------------
def test_la_arroba_de_ndcg_se_traduce_a_algo_que_mlflow_acepta():
    # MLflow rechaza `@` en un nombre de metrica, y las metricas de ranking del proyecto
    # se llaman `ndcg@5`. Sin esta traduccion el registro tumbaba el pipeline entero.
    assert tracking.sanitize_name("ndcg@5") == "ndcg_at_5"
    assert tracking.sanitize_name("ndcg@5_perfil_1") == "ndcg_at_5_perfil_1"
    assert tracking.sanitize_name("valor_politica") == "valor_politica"


def test_los_nombres_llegan_saneados_a_mlflow():
    fake = _FakeMlflow()
    run = tracking.Run(active=True, _mlflow=fake)

    run.log_metrics({"ndcg@5": 0.03, "recall@5": 0.02})

    assert set(fake.metrics) == {"ndcg_at_5", "recall_at_5"}


class _BrokenMlflow(_FakeMlflow):
    def log_metrics(self, metrics):
        raise RuntimeError("el servidor de tracking no responde")


def test_un_fallo_de_mlflow_no_tumba_el_pipeline(capsys):
    run = tracking.Run(active=True, _mlflow=_BrokenMlflow())

    run.log_metrics({"ndcg@5": 0.03})  # no debe levantar nada

    # Pero tampoco se calla: el fallo se avisa por consola.
    assert "no pudo registrar" in capsys.readouterr().out


def test_si_no_se_puede_abrir_el_run_se_devuelve_uno_inerte(monkeypatch, capsys):
    monkeypatch.setenv(tracking.ENV_FLAG, "on")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise RuntimeError("mlflow roto")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with tracking.track("experimento", "run") as run:
        assert run.active is False
        run.log_metrics({"a": 1.0})

    assert "no pudo abrir el run" in capsys.readouterr().out

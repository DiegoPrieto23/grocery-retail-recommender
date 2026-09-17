"""Registro de experimentos con MLflow, opcional y sin acoplar los pipelines.

    pip install mlflow
    python -m src.recommender.pipeline     # ahora deja un run en ./mlflow.db
    python -m src.nba.pipeline
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000

## Por que asi

MLflow es una dependencia pesada y el proyecto tiene que seguir corriendo sin ella (la CI
instala `requirements.txt`, donde no esta). El diseno separa dos cosas:

- **Que se registra** (`recommender_params`, `recommender_metrics`, `nba_params`,
  `nba_metrics`): funciones puras que reciben la configuracion y el resultado del pipeline
  y devuelven diccionarios planos. Se pueden testear sin MLflow instalado, y de hecho
  `tests/test_tracking.py` lo hace.
- **Donde se registra** (`track`): un gestor de contexto que abre un run si MLflow esta
  disponible y **no hace absolutamente nada** si no lo esta. Los pipelines llaman igual en
  los dos casos, sin `if` ni `try` repartidos por el codigo.

Un run no arranca cuando el pipeline corre con `--no-write`: si no hay artefactos que
guardar, tampoco hay experimento que registrar, solo ruido en el almacen.

## Como se activa y se apaga

| `GROCERY_TRACKING` | MLflow instalado | Resultado |
| --- | --- | --- |
| sin definir | si | se registra |
| sin definir | no | no se registra (silencioso) |
| `off` / `0` / `false` | da igual | no se registra |
| `on` / `1` / `true` | no | error claro al importar mlflow |

## Donde escribe

`MLFLOW_TRACKING_URI` se respeta si esta definida. Si no, se usa `sqlite:///mlflow.db` en
la raiz del repo, con los artefactos en `./mlruns` -- las dos rutas estan en el
`.gitignore`. Ese default no es la eleccion por defecto de MLflow, y tiene dos motivos
concretos:

- **El almacen de ficheros esta en modo mantenimiento desde MLflow 3** y lanza una
  excepcion salvo que se ponga `MLFLOW_ALLOW_FILE_STORE=true`. SQLite es el backend
  recomendado y no necesita levantar nada.
- **La ruta de este repo tiene espacios** (`C:\\Users\\Diego Prieto\\...`). Con el almacen
  de ficheros, MLflow construye un `file://` con la ruta percent-encodeada y luego la
  interpreta sin decodificar, asi que intenta crear `C:\\Users\\Diego%20Prieto` y muere con
  `PermissionError`. Con SQLite el problema no aparece.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ENV_FLAG = "GROCERY_TRACKING"
_OFF = {"0", "off", "false", "no"}
_ON = {"1", "on", "true", "yes"}

# Backend por defecto. Ver "Donde escribe" en el docstring del modulo: ni el almacen de
# ficheros de MLflow ni su URI `file://` funcionan en una ruta con espacios.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def is_enabled() -> bool:
    """True si hay que registrar el experimento en este entorno."""
    flag = os.environ.get(ENV_FLAG, "").strip().lower()
    if flag in _OFF:
        return False
    if flag in _ON:
        return True
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


# MLflow solo admite alfanumericos, `_ - . / ` y espacios en el nombre de una metrica. Las
# metricas de ranking del proyecto se llaman `ndcg@5`, y esa arroba tumbaba el registro
# entero -- despues de que el pipeline ya hubiera escrito modelos e informes.
_NAME_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./ ")


def sanitize_name(name: str) -> str:
    """Convierte un nombre de metrica del proyecto en uno que MLflow acepte.

    La arroba de `ndcg@5` se traduce a `_at_` en vez de a un guion bajo suelto, para que
    `ndcg_at_5` siga leyendose en la UI como lo que es.
    """
    return "".join(
        c if c in _NAME_OK else ("_at_" if c == "@" else "_") for c in name
    )


@dataclass
class Run:
    """Un run abierto. Cuando `active` es False, todos los metodos son no-ops.

    Dos decisiones defensivas, las dos por el mismo motivo -- **el registro es un canal
    lateral y nunca debe tumbar un pipeline de 33 minutos que ya ha escrito sus modelos**:

    - Los valores nulos se descartan y los nombres se sanean antes de enviarlos.
    - Cualquier fallo de MLflow se avisa por consola y se traga. Si el servidor de tracking
      no responde o rechaza un nombre, se pierde el run, no el entrenamiento.
    """

    active: bool = False
    _mlflow: Any = None

    def _safe(self, what: str, action: Any) -> None:
        """Ejecuta una llamada a MLflow avisando, pero sin propagar, si falla."""
        if not self.active:
            return
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - el registro nunca tumba el pipeline
            print(f"  [tracking] MLflow no pudo registrar {what}: {exc}", flush=True)

    def log_params(self, params: dict[str, Any]) -> None:
        clean = {sanitize_name(k): v for k, v in params.items() if v is not None}
        self._safe("los parametros", lambda: self._mlflow.log_params(clean))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        clean = {
            sanitize_name(k): float(v) for k, v in metrics.items() if v is not None
        }
        self._safe("las metricas", lambda: self._mlflow.log_metrics(clean))

    def log_artifact(self, path: str | Path) -> None:
        """Sube un fichero si existe. Que falte no es motivo para tumbar el pipeline."""
        if not Path(path).exists():
            return
        self._safe(f"el artefacto {Path(path).name}", lambda: self._mlflow.log_artifact(str(path)))

    def set_tags(self, tags: dict[str, str]) -> None:
        clean = {sanitize_name(k): v for k, v in tags.items()}
        self._safe("las etiquetas", lambda: self._mlflow.set_tags(clean))


@contextmanager
def track(experiment: str, run_name: str) -> Iterator[Run]:
    """Abre un run de MLflow, o devuelve un `Run` inerte si no hay MLflow o si falla.

    Args:
        experiment: Nombre del experimento (se crea si no existe).
        run_name: Nombre de este run concreto.

    Yields:
        El `Run` sobre el que registrar. Nunca es `None`, asi que el llamante no necesita
        comprobar nada.
    """
    if not is_enabled():
        yield Run(active=False)
        return

    try:
        import mlflow

        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name)
    except Exception as exc:  # noqa: BLE001 - ver el docstring de `Run`
        print(f"  [tracking] MLflow no pudo abrir el run: {exc}", flush=True)
        yield Run(active=False)
        return

    with run:
        yield Run(active=True, _mlflow=mlflow)


# ======================================================================================
# Que registrar de cada fase. Funciones puras: entra config y resultado, sale un dict.
# ======================================================================================
def recommender_params(cfg: Any) -> dict[str, Any]:
    """Hiperparametros y ventanas del recomendador (Fase 3)."""
    ranker = cfg.ranker
    return {
        "fase": 3,
        "modelo": "lightgbm_lambdarank",
        "fit_end": str(cfg.fit_end),
        "test_start": str(cfg.test_start),
        "top_k": cfg.top_k,
        "n_train_queries": cfg.n_train_queries,
        "n_valid_queries": cfg.n_valid_queries,
        "n_test_queries": cfg.n_test_queries,
        "relevance": ranker.relevance,
        "label_gain": list(ranker.label_gain),
        "learning_rate": ranker.learning_rate,
        "num_leaves": ranker.num_leaves,
        "min_data_in_leaf": ranker.min_data_in_leaf,
        "num_boost_round": ranker.num_boost_round,
        "early_stopping_rounds": ranker.early_stopping_rounds,
        "seed": cfg.seed,
    }


def recommender_metrics(result: dict[str, Any], *, k: int) -> dict[str, float]:
    """Metricas del recomendador, incluidas las dos ablaciones y el techo del pool.

    Se registran las tres variantes en el mismo run -- modelo completo, sin senal de
    sesion y baseline de popularidad -- porque la comparacion **entre** ellas es el
    resultado, y separarlas en tres runs obligaria a cruzarlos a mano para leerla.
    """
    total = result["summary"].iloc[0]
    no_session = result["summary_no_session"].iloc[0]
    popularity = result["summary_popularity"].iloc[0]
    pool = result["candidate_recall"].iloc[0]
    by_category = result["by_category"].iloc[0]

    metrics = {
        f"ndcg@{k}": total[f"ndcg@{k}"],
        f"recall@{k}": total[f"recall@{k}"],
        f"hit_rate@{k}": total[f"hit_rate@{k}"],
        f"precision@{k}": total[f"precision@{k}"],
        f"f1@{k}": total[f"f1@{k}"],
        f"f1@{k}_por_cesta": total[f"f1@{k}_por_cesta"],
        f"ndcg@{k}_sin_sesion": no_session[f"ndcg@{k}"],
        f"ndcg@{k}_baseline_popularidad": popularity[f"ndcg@{k}"],
        "valid_ndcg": result["valid_ndcg"],
        "pool_recall": pool["pool_recall"],
        "pool_size_medio": pool["pool_size_medio"],
        f"ndcg_graded@{k}": by_category[f"ndcg_graded@{k}"],
        f"cat_hit_rate@{k}": by_category[f"cat_hit_rate@{k}"],
        f"sku_hit_rate@{k}": by_category[f"sku_hit_rate@{k}"],
        "n_test_queries": result["n_test_queries"],
        "n_train_queries": result["n_train_queries"],
    }
    # Y el desglose por perfil, que es donde se ve si aguanta el cold-start.
    for _, row in result["summary"].iloc[1:].iterrows():
        profile = str(row["grupo"]).split(" - ")[0].strip()
        metrics[f"ndcg@{k}_perfil_{profile}"] = row[f"ndcg@{k}"]
    return metrics


def nba_params(cfg: Any) -> dict[str, Any]:
    """Cortes, horizontes y **supuestos economicos** de la Fase 4.

    Los supuestos van como parametros a proposito: son lo que distingue un run de otro
    cuando se barre la politica, y no registrarlos convertiria la comparacion de runs en
    adivinanza.
    """
    coupon = next(a for a in cfg.actions if a.name == "enviar_cupon_categoria")
    return {
        "fase": 4,
        "modelo": "lightgbm_binary_x2",
        "train_cutoffs": ",".join(str(c) for c in cfg.train_cutoffs),
        "valid_cutoff": str(cfg.valid_cutoff),
        "test_cutoff": str(cfg.test_cutoff),
        "category_horizon_days": cfg.category_horizon_days,
        "churn_horizon_days": cfg.churn_horizon_days,
        "category_lookback_days": cfg.category_lookback_days,
        "n_train_customers": cfg.n_train_customers,
        "learning_rate": cfg.propensity.learning_rate,
        "num_leaves": cfg.propensity.num_leaves,
        "num_boost_round": cfg.propensity.num_boost_round,
        "retention_weeks": cfg.policy.retention_weeks,
        "coupon_discount": coupon.discount,
        "coupon_conversion_uplift": coupon.conversion_uplift,
        "coupon_churn_reduction": coupon.churn_reduction,
        "seed": cfg.seed,
    }


def nba_metrics(result: dict[str, Any]) -> dict[str, float]:
    """AUC/PR-AUC de los dos modelos y el valor de cada politica, en euros."""
    churn = result["churn_metrics"]
    purchase = result["purchase_metrics"]
    comparison = result["comparison"]

    metrics = {
        "churn_auc": churn["auc"],
        "churn_pr_auc": churn["pr_auc"],
        "churn_base_rate": churn["base_rate"],
        "churn_lift_top_decile": churn["lift_top_decile"],
        "purchase_auc": purchase["auc"],
        "purchase_pr_auc": purchase["pr_auc"],
        "purchase_base_rate": purchase["base_rate"],
        "purchase_lift_top_decile": purchase["lift_top_decile"],
    }
    for _, row in comparison.iterrows():
        slug = str(row["politica"]).replace(" ", "_").replace(":", "")
        metrics[f"valor_{slug}"] = row["valor_total"]
        metrics[f"pct_accion_{slug}"] = row["pct_accion"]

    # La fila pesimista del barrido de retencion: es la que sostiene la conclusion.
    floor = min(row["valor_politica"] for _, row in result["sensitivity_retention"].iterrows())
    metrics["valor_politica_suelo_retencion"] = floor
    return metrics

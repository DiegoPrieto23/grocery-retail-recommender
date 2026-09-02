"""Los dos modelos de propension: compra en categoria a 7 dias y churn a 4 semanas.

Son dos clasificadores binarios con la misma maquinaria y distinto grano. Lo unico
particular es que **la validacion tambien es temporal**: el conjunto de validacion no es
una muestra aleatoria del entrenamiento sino un corte posterior a todos los de
entrenamiento, para que la parada temprana no premie a un modelo que solo sabe interpolar
dentro del mismo periodo.

## Por que AUC *y* PR-AUC

Las dos clases positivas son minoritarias y en proporciones muy distintas. El AUC es
comodo de comparar pero es optimista cuando la clase positiva es rara, porque premia
ordenar bien el monton de negativos faciles. El PR-AUC mira solo lo que pasa arriba de la
lista, que es donde se decide a quien se manda un cupon: contra el se ve si el modelo
sirve para gastar dinero. Se reporta ademas el `base_rate` para poder leer el PR-AUC
(un modelo aleatorio da PR-AUC = base_rate, no 0,5) y el lift del primer decil, que es la
lectura de negocio directa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.nba.config import PropensityConfig
from src.nba.features import CATEGORICAL_FEATURES

CHURN_MODEL_FILENAME = "nba_churn_lgbm.txt"
PURCHASE_MODEL_FILENAME = "nba_purchase_lgbm.txt"


def collect(
    df: DataFrame,
    *,
    feature_columns: tuple[str, ...],
    keys: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    """Trae la matriz al driver en `float32`, casteando dentro de Spark.

    Mismo criterio que en el ranker de la Fase 3: convertir despues en pandas no ahorra
    memoria, porque el pico ya se ha producido al materializar en `float64`.
    """
    import pyarrow as pa

    selected = [F.col(k) for k in keys]
    selected += [F.col(c).cast("float").alias(c) for c in feature_columns]
    selected.append(F.col(label).cast("byte").alias(label))

    batches = df.select(*selected)._collect_as_arrow()  # noqa: SLF001
    if not batches:
        return pd.DataFrame(columns=[*keys, *feature_columns, label])
    table = pa.Table.from_batches(batches, schema=batches[0].schema)
    del batches
    return table.to_pandas(split_blocks=True, self_destruct=True)


def train(
    train_pdf: pd.DataFrame,
    valid_pdf: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    label: str,
    cfg: PropensityConfig,
    verbose_eval: int = 0,
):
    """Entrena un binario con parada temprana sobre el AUC del corte de validacion."""
    import lightgbm as lgb

    columns = list(feature_columns)
    categorical = [c for c in CATEGORICAL_FEATURES if c in columns]

    dtrain = lgb.Dataset(
        train_pdf[columns],
        label=train_pdf[label],
        categorical_feature=categorical,
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        valid_pdf[columns],
        label=valid_pdf[label],
        categorical_feature=categorical,
        reference=dtrain,
        free_raw_data=True,
    )

    params = {
        "objective": cfg.objective,
        "metric": cfg.metric,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l2": cfg.lambda_l2,
        "seed": cfg.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        "num_threads": 0,
    }

    evals: dict = {}
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.record_evaluation(evals),
            lgb.log_evaluation(verbose_eval),
        ],
    )
    return booster, evals


def predict(booster, pdf: pd.DataFrame, *, feature_columns: tuple[str, ...]) -> np.ndarray:
    """Probabilidad de la clase positiva. Aqui si es una probabilidad, no solo un orden."""
    return booster.predict(
        pdf[list(feature_columns)], num_iteration=booster.best_iteration or None
    )


def metrics(y_true: np.ndarray, y_score: np.ndarray, *, decile: float = 0.10) -> dict[str, float]:
    """AUC, PR-AUC, tasa base y lift del primer decil.

    `lift_top_decile` es cuantas veces mas positivos hay en el 10 % mejor puntuado que en
    la poblacion. Es la cifra que entiende un responsable de CRM: "de cada 100 cupones que
    mando a los que el modelo elige, acierto N veces mas que mandandolos al azar".
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    base = float(y_true.mean()) if y_true.size else float("nan")

    out = {
        "n": int(y_true.size),
        "base_rate": base,
        "auc": float("nan"),
        "pr_auc": float("nan"),
        "lift_top_decile": float("nan"),
    }
    # Con una sola clase presente las dos metricas son indefinidas; no se inventa un 0,5.
    if y_true.size == 0 or y_true.min() == y_true.max():
        return out

    out["auc"] = float(roc_auc_score(y_true, y_score))
    out["pr_auc"] = float(average_precision_score(y_true, y_score))

    k = max(1, int(round(decile * y_true.size)))
    top = np.argsort(-y_score, kind="stable")[:k]
    if base > 0:
        out["lift_top_decile"] = float(y_true[top].mean() / base)
    return out


def feature_importance(booster, *, top: int = 20) -> pd.DataFrame:
    """Importancia por ganancia."""
    return (
        pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance("gain"),
                "split": booster.feature_importance("split"),
            }
        )
        .sort_values("gain", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def save(booster, models_dir: Path, filename: str) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    booster.save_model(path.as_posix(), num_iteration=booster.best_iteration or None)
    return path


def load(models_dir: Path, filename: str):
    """Relee un modelo guardado. Lo usa la demo de la Fase 6, que no reentrena nada."""
    import lightgbm as lgb

    path = Path(models_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"No hay modelo de propension en {path}")
    return lgb.Booster(model_file=path.as_posix())

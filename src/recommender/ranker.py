"""Segunda etapa: el ranker LightGBM con objetivo `LambdaRank`.

## Por que LambdaRank y no un clasificador

Un clasificador binario ("¿entrara este producto en la cesta?") optimiza la probabilidad
de cada candidato por separado, y eso no es lo que se le pide al sistema: lo que importa es
el **orden dentro de una misma cesta**, porque solo se muestran 5 huecos. `LambdaRank`
optimiza directamente la NDCG de cada grupo, asi que penaliza colocar un acierto en el
puesto 6 mucho mas que dar una probabilidad mal calibrada.

El **grupo** es la query, es decir la cesta. Todas las filas de una cesta van juntas y en
orden contiguo: LightGBM lee los tamanos de grupo, no los identificadores, asi que el
DataFrame se ordena por `basket_id` antes de construir el `Dataset`.

## Grupos sin ningun acierto

Si ninguna de las fuentes propuso un solo producto del target, la query no tiene gradiente
que aportar y solo aporta ruido al entrenamiento; se descarta **del train**. En test se
mantiene: su NDCG@5 es 0 y forma parte honesta de la metrica, porque es un fallo real de la
primera etapa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.recommender.config import RankerConfig
from src.recommender.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS

# Nombre del fichero del modelo dentro de `models/`.
MODEL_FILENAME = "recommender_ranker_lgbm.txt"


def collect_for_ranking(df: DataFrame, *, with_label: bool = True) -> pd.DataFrame:
    """Trae la matriz de features al driver, ya ordenada por query.

    Tres decisiones que a este volumen son la diferencia entre funcionar y quedarse sin
    memoria (la matriz del ranker ronda los 2,2 M de filas x 54 columnas):

    - **El casteo a `float32` se hace en Spark**, no en pandas. Por defecto los `double`
      llegan como `float64` y ocupan el doble; convertir despues no ahorra nada, porque el
      pico de memoria ya se ha producido.
    - **El orden por query se hace en Spark.** LightGBM lee tamanos de grupo, no
      identificadores, asi que las filas de una cesta tienen que llegar contiguas; un
      `sort_values` en pandas sobre 2 M de filas duplica la memoria en el peor momento.
    - **La recogida va por trozos de `basket_id`.** `_collect_as_arrow` acumula en el heap
      del driver todos los lotes de lo que recoge. Con el historial as-of (punto A1) la
      matriz entera ya no cabia en 3-4 GB de driver, y con 6 GB la JVM mas la copia de
      Python agotaban un portatil de 16 GB. Por trozos, el driver solo aloja uno cada
      vez; las tablas de Arrow se concatenan sin copiar y se convierten una sola vez.
      Cada trozo va ordenado por `(basket_id, product_id)`, y los trozos se piden en
      orden, asi que la matriz final queda ordenada igual que con un `orderBy` global.
    - **La conversion a pandas usa `split_blocks` y `self_destruct`.** El `to_pandas()`
      normal consolida todas las columnas en un unico bloque de numpy, lo que exige tener
      a la vez la copia vieja y la nueva.
    """
    import pyarrow as pa

    selected = [F.col("basket_id"), F.col("product_id"), F.col("profile").cast("int")]
    selected += [F.col(c).cast("float").alias(c) for c in FEATURE_COLUMNS]
    if with_label:
        selected.append(F.col("label").cast("byte").alias("label"))

    # Cacheada: sin cache, cada trozo recalcularia la matriz entera.
    projected = df.select(*selected).cache()
    try:
        bounds = [None, *_basket_cuts(projected), None]
        tables = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            chunk = projected
            if lo is not None:
                chunk = chunk.filter(F.col("basket_id") > F.lit(lo))
            if hi is not None:
                chunk = chunk.filter(F.col("basket_id") <= F.lit(hi))
            batches = chunk.orderBy("basket_id", "product_id")._collect_as_arrow()  # noqa: SLF001
            if batches:
                tables.append(pa.Table.from_batches(batches, schema=batches[0].schema))
            del batches
    finally:
        projected.unpersist()

    if not tables:
        return pd.DataFrame(columns=[c.name for c in projected.schema])
    table = pa.concat_tables(tables)
    del tables
    return table.to_pandas(split_blocks=True, self_destruct=True)


# Trozos en que se recoge la matriz del ranker (ver `collect_for_ranking`).
COLLECT_CHUNKS = 8


def _basket_cuts(df: DataFrame) -> list[str]:
    """`COLLECT_CHUNKS - 1` cortes de `basket_id` que reparten las filas en trozos parecidos.

    Se calculan sobre los identificadores distintos ordenados (unas decenas de miles, que
    si caben en el driver), pesando cada cesta por su numero de filas.
    """
    counts = df.groupBy("basket_id").count().orderBy("basket_id").toPandas()
    if counts.empty:
        return []
    cumulative = counts["count"].cumsum().to_numpy()
    total = cumulative[-1]
    cuts = []
    for i in range(1, COLLECT_CHUNKS):
        pos = int(np.searchsorted(cumulative, total * i / COLLECT_CHUNKS))
        cuts.append(str(counts["basket_id"].iloc[min(pos, len(counts) - 1)]))
    return sorted(set(cuts))


def group_sizes(pdf: pd.DataFrame) -> np.ndarray:
    """Tamano de cada grupo, en el orden en que aparecen las cestas en el DataFrame."""
    return pdf.groupby("basket_id", sort=False).size().to_numpy()


def drop_groups_without_positives(pdf: pd.DataFrame) -> pd.DataFrame:
    """Quita del entrenamiento las cestas en las que ningun candidato es un acierto."""
    positives = pdf.groupby("basket_id")["label"].transform("max")
    return pdf.loc[positives > 0].reset_index(drop=True)


def train_ranker(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    cfg: RankerConfig,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    verbose_eval: int = 50,
):
    """Entrena el LambdaRank con parada temprana sobre la NDCG@5 de validacion.

    Args:
        train: Matriz de entrenamiento, ya ordenada por `basket_id`.
        valid: Matriz de validacion, de cestas distintas y posteriores no; misma ventana.
        cfg: Hiperparametros.
        feature_columns: Subconjunto de features a usar. Sirve para la ablacion sin
            senal de sesion sin tocar nada mas.
        verbose_eval: Cada cuantas iteraciones se imprime la metrica.

    Returns:
        `(booster, evals)` con el modelo entrenado y el historico de la metrica.
    """
    import lightgbm as lgb

    columns = list(feature_columns)
    categorical = [c for c in CATEGORICAL_FEATURES if c in columns]

    def make(pdf: pd.DataFrame, reference=None):
        return lgb.Dataset(
            pdf[columns],
            label=pdf["label"],
            group=group_sizes(pdf),
            categorical_feature=categorical,
            # LightGBM se queda con su copia binarizada; la matriz original
            # sigue viva en pandas y no hace falta que la duplique.
            free_raw_data=True,
            reference=reference,
        )

    dtrain = make(train)
    dvalid = make(valid, reference=dtrain)

    params = {
        "objective": cfg.objective,
        "metric": cfg.metric,
        "ndcg_eval_at": list(cfg.eval_at),
        # Relevancia binaria: el producto entra en la cesta o no entra.
        "label_gain": [0, 1],
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


def score(booster, pdf: pd.DataFrame, *, feature_columns: tuple[str, ...] = FEATURE_COLUMNS):
    """Puntua cada candidato. El score no es una probabilidad, solo sirve para ordenar."""
    return booster.predict(
        pdf[list(feature_columns)], num_iteration=booster.best_iteration or None
    )


def feature_importance(booster, *, top: int = 25) -> pd.DataFrame:
    """Importancia por ganancia, que es la que mide cuanto ayudo cada feature a ordenar."""
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


def save(booster, models_dir: Path, *, filename: str = MODEL_FILENAME) -> Path:
    """Guarda el modelo en el formato de texto de LightGBM (portable y diffeable)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    booster.save_model(path.as_posix(), num_iteration=booster.best_iteration or None)
    return path


def load(models_dir: Path, *, filename: str = MODEL_FILENAME):
    """Relee un modelo guardado. Lo usa la demo de la Fase 6, que no reentrena nada."""
    import lightgbm as lgb

    path = Path(models_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"No hay ranker entrenado en {path}")
    return lgb.Booster(model_file=path.as_posix())

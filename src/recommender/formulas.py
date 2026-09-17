"""Formulas de negocio compartidas por Spark y pandas (punto B2 del diagnostico).

Las features de reposicion y el orden de la fuente `hist` existen dos veces: en Spark
(`features.py`, `candidates.py`, `src/etl/repurchase.py`) y en pandas
(`src/serving/recommend.py`). Antes, cada formula estaba escrita en las dos, y cualquier
cambio (el punto A1, por ejemplo) habia que hacerlo dos veces y esperar a que el test de
paridad pillase la diferencia.

Aqui cada formula se escribe **una sola vez**, sobre un objeto de operaciones (`ops`)
que sabe hacer `where`, `coalesce` y `greatest` en su motor. La aritmetica (`+`, `*`,
`/`, `>=`) se escribe tal cual: funciona igual sobre una `Column` de Spark que sobre una
`Series` de pandas. `PANDAS_OPS` vive aqui; `SPARK_OPS` vive en `features.py`, para que
este modulo no importe PySpark (la demo sirve sin Spark).

Lo que **no** se comparte son los joins y agregaciones que preparan las entradas: esos
siguen duplicados, y los ata `tests/test_serving_parity.py` y los tests as-of de
`tests/test_asof_features.py`. Por que no se ha unificado todo en un solo motor esta
explicado en `docs/diagnostico-fase7.md`, punto B2.

## Ciclo de reposicion (Tarea 2)

El intervalo esperado entre compras de una categoria sale, por orden de preferencia:

1. de lo observado en ese cliente (media de dias entre compras, con al menos
   `MIN_OBSERVATIONS` dias de compra);
2. del intervalo tipico de la categoria ajustado por hogar
   (`HOUSEHOLD_BASE - HOUSEHOLD_SLOPE * tamano`).

Con dias de compra distintos y ordenados, la media de los huecos entre compras
consecutivas es `(ultima - primera) / (n - 1)`: la suma de los huecos es telescopica. Por
eso aqui no hace falta una ventana con `lag`.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

# Coeficientes del ajuste por tamano de hogar: factor 1,325 para un hogar de 1 y 0,70
# para uno de 6, el rango de `DATA_SPEC.md`. Hipotesis de dominio, no constantes del
# generador.
HOUSEHOLD_BASE = 1.45
HOUSEHOLD_SLOPE = 0.125

# Suelo del ciclo esperado: por corto que sea, nadie repone una categoria cada medio dia.
MIN_EXPECTED_DAYS = 2.0

# Dias de compra minimos para fiarse del intervalo observado (un intervalo necesita dos).
MIN_OBSERVATIONS = 2

# Tamano de hogar que se asume cuando falta el dato.
DEFAULT_HOUSEHOLD_SIZE = 1


class Ops(Protocol):
    """Las tres operaciones que la aritmetica no cubre, en cada motor."""

    def where(self, cond: Any, value: Any, otherwise: Any = None) -> Any: ...

    def coalesce(self, value: Any, fallback: Any) -> Any: ...

    def greatest(self, value: Any, floor: Any) -> Any: ...


class _PandasOps:
    """`Ops` sobre `pd.Series`, con la semantica de nulos de Spark.

    - `where` con condicion nula (NaN) cae en `otherwise`, como `F.when`.
    - `greatest` ignora los nulos, como `F.greatest`.
    """

    @staticmethod
    def where(cond: pd.Series, value: Any, otherwise: Any = None) -> pd.Series:
        fill = np.nan if otherwise is None else otherwise
        chosen = np.where(cond.fillna(False).to_numpy(dtype=bool), value, fill)
        return pd.Series(chosen, index=cond.index, dtype="float64")

    @staticmethod
    def coalesce(value: pd.Series, fallback: Any) -> pd.Series:
        return value.astype("float64").fillna(fallback)

    @staticmethod
    def greatest(value: pd.Series, floor: Any) -> pd.Series:
        return pd.Series(np.fmax(value.to_numpy(dtype="float64"), floor), index=value.index)


PANDAS_OPS: Ops = _PandasOps()


# --------------------------------------------------------------------------------------
# Ciclo de reposicion
# --------------------------------------------------------------------------------------
def household_factor(
    household_size: Any,
    ops: Ops,
    *,
    base: float = HOUSEHOLD_BASE,
    slope: float = HOUSEHOLD_SLOPE,
) -> Any:
    """Hogar mas grande, ciclo mas corto. Sin dato, se asume un hogar de 1."""
    return base - slope * ops.coalesce(household_size, DEFAULT_HOUSEHOLD_SIZE)


def mean_gap_days(n_purchase_days: Any, span_days: Any, ops: Ops) -> Any:
    """Media de dias entre compras: `(ultima - primera) / (n - 1)`; nula con una compra."""
    return ops.where(n_purchase_days >= 2, span_days / (n_purchase_days - 1))


def expected_repurchase_days(
    n_purchase_days: Any,
    mean_gap: Any,
    typical_days: Any,
    factor: Any,
    ops: Ops,
    *,
    min_observations: int = MIN_OBSERVATIONS,
) -> Any:
    """Intervalo esperado: el observado si hay evidencia, si no el tipico ajustado."""
    adjusted_typical = ops.greatest(typical_days * factor, MIN_EXPECTED_DAYS)
    observed = ops.where(n_purchase_days >= min_observations, mean_gap)
    return ops.greatest(ops.coalesce(observed, adjusted_typical), MIN_EXPECTED_DAYS)


def overdue_ratio(days_since: Any, expected_days: Any) -> Any:
    """Dias desde la ultima compra sobre el intervalo esperado. Por encima de 1, toca."""
    return days_since / expected_days


def is_due(ratio: Any, ops: Ops) -> Any:
    """`cat_due` como 0/1. Sin historial en la categoria (ratio nulo), no toca."""
    return ops.coalesce(ops.where(ratio >= 1.0, 1.0, 0.0), 0.0)


def category_need_score(n_purchase_days: Any, cat_due: Any) -> Any:
    """Cuanto necesita el cliente una categoria: su frecuencia, doblada si ya le toca.

    Es el criterio del baseline `personal_due` (`evaluate.baseline_personal_due`) y el de
    la feature `cat_due_rank`. `cat_due` tiene que llegar ya sin nulos (`is_due`). Solo
    usa aritmetica, asi que vale igual para numpy, pandas y una `Column` de Spark.
    """
    return n_purchase_days * (1.0 + cat_due)


# --------------------------------------------------------------------------------------
# Fuente `hist`
# --------------------------------------------------------------------------------------
def personal_score(n_baskets: Any, cat_due: Any, cat_overdue_ratio: Any, ops: Ops) -> Any:
    """Orden de la fuente `hist`: frecuencia como base y un empujon a lo que ya toca.

    Un producto que se compra mucho pero se acaba de reponer cede su sitio a otro que se
    compra menos pero lleva dos ciclos sin caer.
    """
    return n_baskets * (1.0 + ops.coalesce(cat_due, 0.0)) + ops.coalesce(cat_overdue_ratio, 0.0)

"""`Ops` de `formulas.py` sobre columnas de Spark.

Va en su propio modulo, y no en `formulas.py`, para que la ruta de serving en pandas no
importe PySpark; y no en `features.py`, para que el ETL (`src/etl/repurchase.py`) pueda
usarlo sin arrastrar el recomendador entero.
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import Column
from pyspark.sql import functions as F

from src.recommender.formulas import Ops


def _col(value: Any) -> Column:
    return value if isinstance(value, Column) else F.lit(value)


class _SparkOps:
    @staticmethod
    def where(cond: Column, value: Any, otherwise: Any = None) -> Column:
        expr = F.when(cond, _col(value))
        return expr if otherwise is None else expr.otherwise(_col(otherwise))

    @staticmethod
    def coalesce(value: Any, fallback: Any) -> Column:
        return F.coalesce(_col(value), _col(fallback))

    @staticmethod
    def greatest(value: Any, floor: Any) -> Column:
        return F.greatest(_col(value), _col(floor))


SPARK_OPS: Ops = _SparkOps()

"""Utilidades para construir DataFrames de Spark pequenos en los tests.

`spark.createDataFrame(lista_de_tuplas)` levanta un worker de Python, y ese worker esta
roto con PySpark 3.5 sobre Python 3.13 (ver `src/etl/session.PYTHON_313_NOTE`). El camino
que si funciona es pandas + Arrow, que viaja por la JVM. Este modulo encapsula esa
conversion, incluido el mapeo de tipos, para que los tests se escriban con diccionarios
normales y no se preocupen de dtypes.
"""

from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StructType,
    TimestampType,
)

# Tipos de pandas que sobreviven al viaje por Arrow conservando los nulos. Los dtypes
# nativos de numpy no valen: un entero con un None se convierte en float64 y Arrow ya no
# lo puede volver a meter en un IntegerType.
_PANDAS_DTYPES = {
    IntegerType: "Int32",
    LongType: "Int64",
    DoubleType: "float64",
    BooleanType: "boolean",
}


def _struct_from_ddl(ddl: str) -> StructType:
    """Convierte una cadena DDL en `StructType`.

    `StructType.fromDDL` solo existe a partir de PySpark 4; en la 3.5 que fija
    `constraints.txt` hay que tirar del parser interno.
    """
    try:
        return StructType.fromDDL(ddl)  # type: ignore[attr-defined]
    except AttributeError:
        from pyspark.sql.types import _parse_datatype_string

        parsed = _parse_datatype_string(ddl)
        if not isinstance(parsed, StructType):
            raise TypeError(f"El DDL {ddl!r} no describe una estructura de filas")
        return parsed


def spark_df(
    spark: SparkSession, rows: list[dict], schema: str | StructType
) -> DataFrame:
    """Crea un DataFrame a partir de una lista de diccionarios y un esquema DDL.

    Args:
        spark: Sesion activa.
        rows: Filas como diccionarios. Las claves que falten quedan a nulo.
        schema: Esquema en DDL (`"a string, b int"`) o un `StructType`.

    Returns:
        El DataFrame con el esquema pedido.
    """
    fields = _struct_from_ddl(schema) if isinstance(schema, str) else schema
    names = [f.name for f in fields.fields]

    if not rows:
        # Un DataFrame de pandas vacio no entra por el camino de Arrow y acaba
        # levantando un worker de Python. `range(0)` + columnas nulas casteadas se
        # resuelve entero en la JVM y da exactamente el mismo esquema.
        return spark.range(0).select(
            *[F.lit(None).cast(f.dataType).alias(f.name) for f in fields.fields]
        )

    pdf = pd.DataFrame(rows, columns=names)

    for field in fields.fields:
        dtype = _PANDAS_DTYPES.get(type(field.dataType))
        if dtype is not None:
            pdf[field.name] = pdf[field.name].astype(dtype)
        elif isinstance(field.dataType, DateType):
            pdf[field.name] = pd.to_datetime(pdf[field.name]).dt.date
        elif isinstance(field.dataType, TimestampType):
            pdf[field.name] = pd.to_datetime(pdf[field.name])
        else:
            pdf[field.name] = pdf[field.name].astype("object")

    return spark.createDataFrame(pdf, fields)


# Esquemas de las tablas tal como las consumen los modulos del ETL. Se declaran aqui, y no
# se importan de `src.etl.schemas`, para que un cambio accidental de esquema en produccion
# haga fallar los tests en vez de arrastrarlos sin que nadie se entere.
CUSTOMERS_DDL = (
    "customer_id string, signup_date date, country string, city string, "
    "household_size_est int, loyalty_tier string, preferred_channel string, "
    "churn_label boolean"
)
PRODUCTS_DDL = (
    "product_id string, department string, category string, brand string, "
    "is_private_label boolean, is_perishable boolean, unit_price double, "
    "pack_size int, typical_repurchase_days int"
)
PROMOTIONS_DDL = (
    "promotion_id string, product_id string, promo_type string, discount_value double, "
    "start_date date, end_date date"
)
BASKETS_DDL = (
    "basket_id string, customer_id string, channel string, basket_date timestamp, "
    "store_id string, total_amount double"
)
BASKET_ITEMS_DDL = (
    "basket_id string, product_id string, quantity int, unit_price_paid double, "
    "promotion_id string"
)
SESSIONS_DDL = (
    "session_id string, customer_id string, session_date timestamp, device_type string, "
    "converted boolean, basket_id string"
)
SESSION_EVENTS_DDL = (
    "session_id string, product_id string, event_type string, event_timestamp timestamp"
)

TABLE_DDL = {
    "customers": CUSTOMERS_DDL,
    "products": PRODUCTS_DDL,
    "promotions": PROMOTIONS_DDL,
    "baskets": BASKETS_DDL,
    "basket_items": BASKET_ITEMS_DDL,
    "sessions": SESSIONS_DDL,
    "session_events": SESSION_EVENTS_DDL,
}


def make_tables(spark: SparkSession, **rows: list[dict]) -> dict[str, DataFrame]:
    """Construye las 7 tablas a partir de las filas que se le pasen.

    Las tablas que no se mencionen quedan vacias pero con su esquema, que es lo que
    necesitan las reglas que cruzan tablas para no romperse.
    """
    return {name: spark_df(spark, rows.get(name, []), ddl) for name, ddl in TABLE_DDL.items()}


def collect_dicts(df: DataFrame, *order_by: str) -> list[dict]:
    """Materializa un DataFrame como lista de diccionarios, opcionalmente ordenado."""
    frame = df.orderBy(*order_by) if order_by else df
    return [row.asDict() for row in frame.collect()]


# --------------------------------------------------------------------------------------
# Vistas de `data/processed` que consumen las preguntas del EDA (`src/eda/questions.py`)
# --------------------------------------------------------------------------------------
# No es el esquema completo de las tablas procesadas: son **las columnas que leen las
# nueve preguntas**, que es el contrato que hay que fijar. Si el ETL dejara de producir
# alguna de ellas, o le cambiara el tipo, estos tests fallan antes que el notebook.
EDA_VIEW_DDL = {
    "customers": (
        "customer_id string, loyalty_tier string, household_size_est int, "
        "churn_label boolean"
    ),
    "products": (
        "product_id string, department string, category string, "
        "is_private_label boolean, typical_repurchase_days int"
    ),
    "promotions": (
        "promotion_id string, product_id string, promo_type string, "
        "start_date date, end_date date"
    ),
    "baskets": (
        "basket_id string, customer_id string, channel string, basket_date timestamp, "
        "basket_day date, total_amount double, n_lines int, n_units int, "
        "is_anonymous boolean"
    ),
    "basket_items": (
        "basket_id string, product_id string, quantity int, promotion_id string, "
        "line_amount double, is_promo boolean"
    ),
    "sessions": (
        "session_id string, customer_id string, device_type string, "
        "converted boolean, basket_id string"
    ),
    "session_events": (
        "session_id string, product_id string, event_type string, "
        "event_timestamp timestamp"
    ),
    "rfm": (
        "customer_id string, recency_days int, frequency int, monetary double, "
        "avg_ticket double, avg_days_between_baskets double, rfm_segment string, "
        "churn_label boolean, has_purchases boolean"
    ),
    "repurchase_features": (
        "customer_id string, category string, typical_repurchase_days int, "
        "observed_repurchase_days double, due_for_repurchase boolean"
    ),
    "affinity_category": (
        "antecedent string, consequent string, n_baskets_both bigint, "
        "support double, confidence double, lift double"
    ),
}


def register_eda_views(spark: SparkSession, **rows: list[dict]) -> dict[str, DataFrame]:
    """Registra como vistas temporales las tablas que necesita `src.eda.questions`.

    Las tablas que no se mencionen quedan vacias pero con su esquema, para que una
    consulta que las cruce no reviente por falta de vista.
    """
    views = {
        name: spark_df(spark, rows.get(name, []), ddl)
        for name, ddl in EDA_VIEW_DDL.items()
    }
    for name, df in views.items():
        df.createOrReplaceTempView(name)
    return views

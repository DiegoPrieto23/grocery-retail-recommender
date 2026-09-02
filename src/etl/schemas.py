"""Esquemas explicitos y lectura de las 7 tablas crudas de `data/raw`.

Los tipos son los de `DATA_SPEC.md`. Se declaran a mano en vez de dejar que Spark los
infiera por dos motivos: la inferencia obliga a una pasada extra sobre 3,1 M de lineas, y
sobre columnas con nulos (`store_id`, `promotion_id`, `sessions.basket_id`) puede resolver
tipos distintos segun el trozo de fichero que le toque leer.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Orden de dependencia: las dimensiones antes que los hechos.
TABLE_ORDER: tuple[str, ...] = (
    "customers",
    "products",
    "promotions",
    "baskets",
    "basket_items",
    "sessions",
    "session_events",
)


def _s(name: str) -> StructField:
    return StructField(name, StringType(), True)


RAW_SCHEMAS: dict[str, StructType] = {
    "customers": StructType(
        [
            _s("customer_id"),
            StructField("signup_date", DateType(), True),
            _s("country"),
            _s("city"),
            StructField("household_size_est", IntegerType(), True),
            _s("loyalty_tier"),
            _s("preferred_channel"),
            StructField("churn_label", BooleanType(), True),
        ]
    ),
    "products": StructType(
        [
            _s("product_id"),
            _s("department"),
            _s("category"),
            _s("brand"),
            StructField("is_private_label", BooleanType(), True),
            StructField("is_perishable", BooleanType(), True),
            StructField("unit_price", DoubleType(), True),
            StructField("pack_size", IntegerType(), True),
            StructField("typical_repurchase_days", IntegerType(), True),
        ]
    ),
    "promotions": StructType(
        [
            _s("promotion_id"),
            _s("product_id"),
            _s("promo_type"),
            StructField("discount_value", DoubleType(), True),
            StructField("start_date", DateType(), True),
            StructField("end_date", DateType(), True),
        ]
    ),
    "baskets": StructType(
        [
            _s("basket_id"),
            _s("customer_id"),
            _s("channel"),
            StructField("basket_date", TimestampType(), True),
            _s("store_id"),
            StructField("total_amount", DoubleType(), True),
        ]
    ),
    "basket_items": StructType(
        [
            _s("basket_id"),
            _s("product_id"),
            StructField("quantity", IntegerType(), True),
            StructField("unit_price_paid", DoubleType(), True),
            _s("promotion_id"),
        ]
    ),
    "sessions": StructType(
        [
            _s("session_id"),
            _s("customer_id"),
            StructField("session_date", TimestampType(), True),
            _s("device_type"),
            StructField("converted", BooleanType(), True),
            _s("basket_id"),
        ]
    ),
    "session_events": StructType(
        [
            _s("session_id"),
            _s("product_id"),
            _s("event_type"),
            StructField("event_timestamp", TimestampType(), True),
        ]
    ),
}

_CSV_OPTIONS = {
    "header": "true",
    "mode": "PERMISSIVE",
    "nullValue": "",
    "dateFormat": "yyyy-MM-dd",
    "timestampFormat": "yyyy-MM-dd HH:mm:ss",
}


def read_raw_table(spark: SparkSession, name: str, data_dir: str | Path) -> DataFrame:
    """Lee una tabla cruda de `data_dir/<name>.csv` con su esquema declarado."""
    if name not in RAW_SCHEMAS:
        raise KeyError(f"Tabla desconocida: {name!r}. Esperada una de {TABLE_ORDER}")
    path = Path(data_dir) / f"{name}.csv"
    return spark.read.options(**_CSV_OPTIONS).schema(RAW_SCHEMAS[name]).csv(path.as_posix())


def read_raw(spark: SparkSession, data_dir: str | Path = "data/raw") -> dict[str, DataFrame]:
    """Lee las 7 tablas crudas y las devuelve indexadas por nombre."""
    return {name: read_raw_table(spark, name, data_dir) for name in TABLE_ORDER}


def write_table(
    df: DataFrame, target: str | Path, *, engine: str = "auto"
) -> str:
    """Escribe un DataFrame en Parquet y devuelve la ruta escrita.

    En Windows, `df.write.parquet(...)` falla si falta `hadoop.dll` en `%HADOOP_HOME%\\bin`
    (`UnsatisfiedLinkError: NativeIO$Windows.access0`): el `FileOutputCommitter` de Hadoop
    consulta permisos POSIX a traves de esa libreria nativa. `winutils.exe` por si solo no
    basta. Por eso hay dos motores:

    - `"spark"`: escritura distribuida normal, a una carpeta con varios ficheros part-*.
    - `"pandas"`: se recogen los datos al driver en formato Arrow (sin pasar por el worker
      de Python) y se escribe un unico fichero con pyarrow. Es la salida de emergencia
      para local; a esta escala (3 M de lineas ~ 120 MB en Arrow) es incluso mas rapida.

    `"auto"` intenta Spark y cae a pandas solo ante ese fallo concreto, de forma que en la
    CI (Linux, sin el problema) la escritura sigue siendo la distribuida.

    Args:
        df: DataFrame a escribir.
        target: Ruta destino, sin extension. El motor `pandas` le anade `.parquet`.
        engine: `"auto"`, `"spark"` o `"pandas"`.

    Returns:
        La ruta realmente escrita (carpeta con Spark, fichero con pandas).
    """
    if engine not in {"auto", "spark", "pandas"}:
        raise ValueError(f"engine debe ser auto/spark/pandas, no {engine!r}")

    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)

    if engine in {"auto", "spark"}:
        try:
            df.write.mode("overwrite").parquet(path.as_posix())
            return path.as_posix()
        except Exception as exc:  # noqa: BLE001 - se reintenta con el otro motor
            if engine == "spark" or not _is_missing_hadoop_native(exc):
                raise

    return _write_with_pyarrow(df, path.with_suffix(".parquet"))


def _is_missing_hadoop_native(exc: BaseException) -> bool:
    """True si el fallo es el `hadoop.dll` ausente de Windows y no otra cosa."""
    return "NativeIO$Windows.access0" in str(exc) or "UnsatisfiedLinkError" in str(exc)


# Resultado del sondeo por SparkContext, para no repetirlo en cada tabla.
_ENGINE_PROBE: dict[str, str] = {}


def resolve_write_engine(spark: SparkSession) -> str:
    """Averigua una sola vez si la escritura nativa de Spark funciona en este entorno.

    Sin este sondeo, `engine="auto"` descubre el problema tabla a tabla, y cada intento
    fallido ha calculado antes el DataFrame entero: en un ETL de 10 tablas eso es
    recalcularlo casi todo dos veces. Se prueba con un DataFrame de una fila y se recuerda
    el veredicto.
    """
    key = spark.sparkContext.applicationId
    if key in _ENGINE_PROBE:
        return _ENGINE_PROBE[key]

    probe = Path(tempfile.mkdtemp(prefix="grocery_write_probe_"))
    try:
        spark.range(1).write.mode("overwrite").parquet((probe / "probe").as_posix())
        engine = "spark"
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de escritura descarta Spark
        if not _is_missing_hadoop_native(exc):
            raise
        engine = "pandas"
    finally:
        shutil.rmtree(probe, ignore_errors=True)

    _ENGINE_PROBE[key] = engine
    return engine


def _write_with_pyarrow(df: DataFrame, path: Path) -> str:
    """Recoge el DataFrame en Arrow y lo escribe como un unico fichero Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Un intento de escritura con Spark que aborta deja la carpeta destino vacia. Si se
    # queda ahi, `read_processed` la encontraria antes que el fichero y leeria la nada.
    stale = path.with_suffix("")
    if stale.is_dir():
        shutil.rmtree(stale, ignore_errors=True)

    batches = df._collect_as_arrow()  # noqa: SLF001 - unica via Arrow en PySpark 3.5
    if batches:
        table = pa.Table.from_batches(batches, schema=batches[0].schema)
    else:
        # Un DataFrame vacio no produce ni un lote, y sin lote no hay esquema Arrow del
        # que tirar. `toPandas()` sobre cero filas sigue devolviendo las columnas, que es
        # lo minimo para que el Parquet se pueda releer.
        table = pa.Table.from_pandas(df.toPandas(), preserve_index=False)
    pq.write_table(table, path.as_posix(), compression="snappy")
    return path.as_posix()


def write_processed(
    tables: dict[str, DataFrame],
    out_dir: str | Path = "data/processed",
    *,
    engine: str = "auto",
) -> dict[str, str]:
    """Escribe varias tablas en Parquet bajo `out_dir`. Devuelve las rutas escritas."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if engine == "auto" and tables:
        engine = resolve_write_engine(next(iter(tables.values())).sparkSession)
    return {name: write_table(df, out / name, engine=engine) for name, df in tables.items()}


def read_processed(
    spark: SparkSession, names: tuple[str, ...], in_dir: str | Path = "data/processed"
) -> dict[str, DataFrame]:
    """Lee de vuelta las tablas escritas por `write_processed`.

    Acepta indistintamente la carpeta que deja el motor `spark` y el fichero suelto que
    deja el motor `pandas`.
    """
    base = Path(in_dir)
    out: dict[str, DataFrame] = {}
    for name in names:
        folder, single = base / name, base / f"{name}.parquet"
        if folder.is_dir() and any(folder.iterdir()):
            out[name] = spark.read.parquet(folder.as_posix())
        elif single.is_file():
            out[name] = spark.read.parquet(single.as_posix())
        else:
            raise FileNotFoundError(f"No hay tabla procesada {name!r} en {base}")
    return out

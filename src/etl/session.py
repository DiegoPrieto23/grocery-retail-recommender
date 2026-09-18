"""Fabrica de la SparkSession del proyecto.

Centraliza la configuracion local para que ETL, tests y notebooks arranquen Spark
exactamente igual. Dos decisiones no obvias:

- **Zona horaria UTC.** Las marcas de tiempo del generador son "naive" (sin zona). Si la
  sesion usa la zona local, `2024-03-31 02:30` (hora que no existe en Europe/Madrid por
  el cambio horario) se desplaza sola y los `datediff` de RFM y de recompra dejan de
  cuadrar. Con UTC el calendario es continuo y las fechas viajan sin sorpresas.

  Efecto secundario que conviene conocer: al traer un `timestamp` al driver con
  `collect()` o `toPandas()`, PySpark lo convierte de la zona de la sesion a la del
  sistema operativo, asi que el `datetime` de Python sale desplazado respecto al CSV.
  El valor almacenado y todo el calculo en SQL (`HOUR()`, `to_date()`, `datediff()`) son
  correctos; lo unico que cambia es como se pinta al cruzar a Python. Por eso conviene
  extraer la parte de fecha u hora **dentro** de Spark antes de llevarse nada al driver.
- **Arrow activado.** Es el unico camino de ida y vuelta a pandas que no depende del
  worker de Python (ver `PYTHON_313_NOTE`), ademas de ser mas rapido para `toPandas()`
  en el notebook de EDA.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession

# --------------------------------------------------------------------------------------
# Aviso de entorno
# --------------------------------------------------------------------------------------
PYTHON_313_NOTE = """\
PySpark 3.5.x no soporta oficialmente Python 3.13: el worker de Python casca al arrancar
en Windows (OSError WinError 10038 sobre el socket). Todo el ETL de este paquete esta
escrito para ejecutarse integramente en la JVM -- DataFrame API, Spark SQL y MLlib -- sin
UDFs de Python, sin `.rdd` y sin `createDataFrame` sobre listas de Python, que son las
tres cosas que levantan un worker. Para construir DataFrames pequenos (tests, notebooks)
usar `spark.createDataFrame(pandas_df)` con Arrow, que va por la JVM.
"""

# Numero de particiones de shuffle por defecto en local: las 200 de fabrica generan
# cientos de ficheros diminutos y dominan el tiempo de ejecucion con este volumen.
DEFAULT_SHUFFLE_PARTITIONS = 16


def get_spark(
    app_name: str = "grocery-retail-etl",
    *,
    master: str = "local[*]",
    shuffle_partitions: int = DEFAULT_SHUFFLE_PARTITIONS,
    driver_memory: str = "4g",
    enable_ui: bool = False,
    work_dir: str | Path | None = None,
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    """Crea (o reutiliza) la SparkSession del proyecto.

    Args:
        app_name: Nombre de la aplicacion en la UI y en los logs.
        master: Master de Spark. En local siempre `local[*]`.
        shuffle_partitions: Particiones tras un shuffle. Ver DEFAULT_SHUFFLE_PARTITIONS.
        driver_memory: Memoria del driver. Solo aplica si la JVM aun no ha arrancado.
        enable_ui: Levantar la UI web de Spark (util para depurar, ruidosa en tests).
        work_dir: Directorio para el warehouse y el metastore Derby. Por defecto uno
            temporal, para no dejar `spark-warehouse/` ni `derby.log` en el repo.
        extra_conf: Configuracion adicional, aplicada al final.

    Returns:
        La SparkSession activa.
    """
    # El worker de Python debe ser el mismo interprete que el driver: si no coinciden,
    # Spark arranca el "python" del PATH, que puede ser otra version sin las librerias.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    work = Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "grocery_spark"
    work.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.ui.enabled", str(enable_ui).lower())
        # La barra de progreso ANSI ensucia la salida de pytest y de los notebooks.
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.warehouse.dir", str(work / "warehouse"))
        # Cota a la representacion en texto del plan. Por defecto no tiene ninguna, y en
        # las cadenas largas del recomendador (cinco fuentes unidas, historial as-of y la
        # matriz de features) el plan de la ejecucion adaptativa llega a cientos de MB de
        # texto: la JVM se quedaba sin heap en `QueryExecution.explainString`, construyendo
        # un `String` que nadie llega a leer, no procesando datos. Con el pool del punto M6
        # (234 candidatos por query) eso tumbaba el driver en la ventana de cold-start.
        # `explain()` sigue funcionando; solo se trunca a partir de este tamano.
        .config("spark.sql.maxPlanStringLength", "8192")
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={work}")
    )
    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

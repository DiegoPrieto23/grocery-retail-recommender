"""Tests de lectura de `data/raw` y de escritura/lectura de `data/processed`.

La escritura tiene dos motores (ver `src/etl/schemas.write_table`) porque en Windows falta
`hadoop.dll` y la escritura nativa de Spark no arranca. Estos tests fijan que el resultado
sea el mismo por los dos caminos y que el fallback no deje basura por medio.
"""

from __future__ import annotations

import pytest

from src.etl.schemas import (
    RAW_SCHEMAS,
    TABLE_ORDER,
    read_processed,
    read_raw,
    read_raw_table,
    resolve_write_engine,
    write_processed,
    write_table,
)
from tests.helpers import collect_dicts, spark_df

CSV_CUSTOMERS = """\
customer_id,signup_date,country,city,household_size_est,loyalty_tier,preferred_channel,churn_label
C000001,2024-01-15,Espana,Madrid,3,gold,app,False
C000002,2023-11-02,Espana,,1,bronze,store,True
"""

CSV_BASKETS = """\
basket_id,customer_id,channel,basket_date,store_id,total_amount
B0000001,C000001,app,2024-02-01 20:06:14,,23.33
B0000002,,store,2024-02-02 17:41:35,S026,16.69
"""


@pytest.fixture
def raw_dir(tmp_path):
    """Un `data/raw` mínimo con las dos tablas que ejercitan todos los tipos."""
    (tmp_path / "customers.csv").write_text(CSV_CUSTOMERS, encoding="utf-8")
    (tmp_path / "baskets.csv").write_text(CSV_BASKETS, encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------------------
# Lectura del crudo
# --------------------------------------------------------------------------------------


def test_el_esquema_declarado_es_el_que_se_aplica(spark, raw_dir):
    customers = read_raw_table(spark, "customers", raw_dir)

    assert customers.schema == RAW_SCHEMAS["customers"]


def test_los_tipos_se_resuelven_al_leer(spark, raw_dir):
    """Fechas, enteros y booleanos tienen que llegar tipados, no como texto."""
    rows = {r["customer_id"]: r for r in collect_dicts(read_raw_table(spark, "customers", raw_dir))}

    assert str(rows["C000001"]["signup_date"]) == "2024-01-15"
    assert rows["C000001"]["household_size_est"] == 3
    assert rows["C000002"]["churn_label"] is True


def test_la_cadena_vacia_del_csv_se_lee_como_nulo(spark, raw_dir):
    """`city` vacía en el CSV y `customer_id` de una compra anónima son NULL, no ''."""
    clientes = {r["customer_id"]: r for r in collect_dicts(read_raw_table(spark, "customers", raw_dir))}
    cestas = {r["basket_id"]: r for r in collect_dicts(read_raw_table(spark, "baskets", raw_dir))}

    assert clientes["C000002"]["city"] is None
    assert cestas["B0000002"]["customer_id"] is None
    assert cestas["B0000001"]["store_id"] is None


def test_las_marcas_de_tiempo_conservan_la_hora(spark, raw_dir):
    """La hora del CSV es la que ve Spark, que es la que usan `HOUR()` y `to_date()`.

    Se comprueba dentro de Spark a proposito. Al traer un `timestamp` al driver con
    `collect()`, PySpark lo pasa de la zona de la sesion (UTC) a la del sistema, asi que
    el `datetime` de Python sale desplazado. Es convencion de PySpark, no un fallo del
    dato: el valor almacenado y todo el calculo en SQL son correctos.
    """
    cestas = {
        r["basket_id"]: r
        for r in collect_dicts(
            read_raw_table(spark, "baskets", raw_dir).selectExpr(
                "basket_id",
                "cast(basket_date as string) AS ts",
                "cast(to_date(basket_date) as string) AS dia",
                "hour(basket_date) AS hora",
            )
        )
    }

    assert cestas["B0000001"]["ts"] == "2024-02-01 20:06:14"
    assert cestas["B0000001"]["dia"] == "2024-02-01"
    assert cestas["B0000001"]["hora"] == 20


def test_la_marca_de_tiempo_sobrevive_al_parquet(spark, tmp_path, raw_dir):
    """El viaje CSV -> Spark -> Parquet -> Spark no puede mover el reloj ni una hora."""
    original = read_raw_table(spark, "baskets", raw_dir)
    write_table(original, tmp_path / "baskets", engine="pandas")
    recuperada = read_processed(spark, ("baskets",), tmp_path)["baskets"]

    def horas(df):
        return collect_dicts(
            df.selectExpr("basket_id", "cast(basket_date as string) AS ts"), "basket_id"
        )

    assert horas(recuperada) == horas(original)


def test_tabla_desconocida(spark, raw_dir):
    with pytest.raises(KeyError, match="inventada"):
        read_raw_table(spark, "inventada", raw_dir)


def test_read_raw_espera_las_siete_tablas(spark, raw_dir):
    """Faltan cinco CSV: tiene que fallar al leer, no devolver tablas a medias."""
    assert set(TABLE_ORDER) == set(RAW_SCHEMAS)

    with pytest.raises(Exception):
        read_raw(spark, raw_dir)["products"].count()


# --------------------------------------------------------------------------------------
# Escritura y relectura de lo procesado
# --------------------------------------------------------------------------------------


@pytest.fixture
def sample(spark):
    return spark_df(
        spark,
        [
            {"id": "A", "n": 1, "valor": 1.5, "flag": True},
            {"id": "B", "n": 2, "valor": 2.5, "flag": False},
        ],
        "id string, n int, valor double, flag boolean",
    )


@pytest.mark.parametrize("engine", ["pandas", "auto"])
def test_ida_y_vuelta_por_los_dos_motores(spark, tmp_path, sample, engine):
    write_processed({"tabla": sample}, tmp_path, engine=engine)
    recuperada = read_processed(spark, ("tabla",), tmp_path)["tabla"]

    assert collect_dicts(recuperada, "id") == collect_dicts(sample, "id")


def test_los_tipos_sobreviven_al_parquet(spark, tmp_path, sample):
    write_processed({"tabla": sample}, tmp_path, engine="pandas")
    recuperada = read_processed(spark, ("tabla",), tmp_path)["tabla"]

    assert dict(recuperada.dtypes) == dict(sample.dtypes)


def test_una_tabla_vacia_se_escribe_y_se_relee(spark, tmp_path):
    """Sin filas no hay lotes de Arrow: el esquema tiene que salir igualmente."""
    vacia = spark_df(spark, [], "id string, n int")
    write_table(vacia, tmp_path / "vacia", engine="pandas")
    recuperada = read_processed(spark, ("vacia",), tmp_path)["vacia"]

    assert recuperada.count() == 0
    assert recuperada.columns == ["id", "n"]


def test_el_fallback_borra_la_carpeta_que_dejo_el_intento_fallido(spark, tmp_path, sample):
    """Spark aborta dejando la carpeta vacía; si sobrevive, la relectura leería la nada."""
    (tmp_path / "tabla").mkdir()
    write_table(sample, tmp_path / "tabla", engine="pandas")

    assert not (tmp_path / "tabla").exists()
    assert (tmp_path / "tabla.parquet").is_file()
    assert read_processed(spark, ("tabla",), tmp_path)["tabla"].count() == 2


def test_reescribir_sustituye_el_contenido(spark, tmp_path, sample):
    write_table(sample, tmp_path / "tabla", engine="pandas")
    write_table(sample.limit(1), tmp_path / "tabla", engine="pandas")

    assert read_processed(spark, ("tabla",), tmp_path)["tabla"].count() == 1


def test_una_tabla_que_no_existe(spark, tmp_path):
    with pytest.raises(FileNotFoundError, match="fantasma"):
        read_processed(spark, ("fantasma",), tmp_path)


def test_motor_desconocido(sample, tmp_path):
    with pytest.raises(ValueError, match="engine"):
        write_table(sample, tmp_path / "tabla", engine="mongo")


def test_el_motor_se_resuelve_una_sola_vez(spark):
    """El sondeo se cachea por aplicación: dos llamadas dan lo mismo sin volver a probar."""
    primero = resolve_write_engine(spark)

    assert primero in {"spark", "pandas"}
    assert resolve_write_engine(spark) == primero

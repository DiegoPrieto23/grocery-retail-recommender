"""Fixtures compartidas de los tests del generador."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_generation.generate_dataset import GeneratorConfig, generate

# Escala pequena: mantiene todas las reglas de negocio activas pero genera en segundos.
TEST_SCALE = 0.05


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Genera una vez el dataset de prueba y lo reparte a todos los tests."""
    out: Path = tmp_path_factory.mktemp("dataset")
    return generate(GeneratorConfig(out_dir=out, scale=TEST_SCALE))


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directorio con los CSV de un dataset de prueba ya escrito."""
    out: Path = tmp_path_factory.mktemp("dataset_csv")
    generate(GeneratorConfig(out_dir=out, scale=TEST_SCALE))
    return out


# --------------------------------------------------------------------------------------
# Fixtures de Spark (Fase 2)
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def spark():
    """SparkSession compartida por todos los tests del ETL.

    Arrancar la JVM cuesta ~10 s, asi que se levanta una sola vez por sesion de pytest.
    Dos particiones de shuffle: con DataFrames de decenas de filas, mas particiones solo
    anaden ficheros vacios y latencia.
    """
    from src.etl.session import get_spark

    session = get_spark("tests", master="local[2]", shuffle_partitions=2, driver_memory="1g")
    yield session
    session.stop()

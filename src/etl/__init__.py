"""ETL y feature engineering en PySpark (Fase 2 del ROADMAP).

Modulos:

- `session`: fabrica de la SparkSession del proyecto.
- `schemas`: esquemas de las tablas crudas, lectura y escritura.
- `cleaning`: limpieza de las 7 tablas y traza de lo corregido.
- `data_trust`: Data Trust Score del dataset (Tarea 1).
- `rfm`: RFM y segmentacion por cliente.
- `repurchase`: ciclo de recompra y `due_for_repurchase` (Tarea 2).
- `affinity`: afinidad de cesta por co-ocurrencia y FP-Growth.
- `run_etl`: orquestador que lo encadena todo.
"""

from src.etl.affinity import cooccurrence_affinity, expected_pairs_report, fpgrowth_rules
from src.etl.cleaning import CleaningConfig, CleaningReport, clean_all
from src.etl.data_trust import TrustReport, data_trust_score
from src.etl.repurchase import repurchase_features
from src.etl.rfm import rfm
from src.etl.schemas import read_raw, read_processed, write_processed
from src.etl.session import get_spark

__all__ = [
    "CleaningConfig",
    "CleaningReport",
    "TrustReport",
    "clean_all",
    "cooccurrence_affinity",
    "data_trust_score",
    "expected_pairs_report",
    "fpgrowth_rules",
    "get_spark",
    "read_processed",
    "read_raw",
    "repurchase_features",
    "rfm",
    "write_processed",
]

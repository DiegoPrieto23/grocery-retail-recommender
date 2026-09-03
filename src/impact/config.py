"""Supuestos economicos del resumen de impacto. Todos declarados, ninguno escondido.

Este fichero es al resumen de impacto lo que `src/nba/config.py` es a la politica: el
sitio donde vive **lo que no se mide**. La separacion importa, porque el resumen mezcla
tres cosas de naturaleza distinta y presentarlas juntas sin etiqueta es la forma habitual
de inflar un numero de negocio:

1. **Medido en el dato** (`pipeline.measure_baseline`): cestas al mes, cestas online al
   mes, importe medio de una linea de ticket y mezcla de departamentos. Salen de
   `data/processed` y se recalculan con el script.
2. **Medido en el modelo** (`reports/recommender/metrics.json`, `reports/nba/metrics.json`):
   `hit_rate@5` del ranker y del baseline de popularidad, y el valor incremental de la
   politica de NBA. Salen de ejecutar las Fases 3 y 4.
3. **Supuesto** (este fichero): el margen bruto por departamento y, sobre todo, la
   **tasa de incrementalidad**. Se barren en el informe.

## Por que la incrementalidad es el supuesto critico

El `hit_rate@5` mide **relevancia**, no causalidad: un acierto significa que el producto
recomendado estaba en la cesta de test, es decir, que el cliente **iba a comprarlo de
todas formas**. Recomendarselo no crea esa venta. Lo que puede crear valor es el margen
del hueco: que se lleve la unidad que se le habria olvidado, que la anada antes de cerrar
la cesta, o que compre la referencia sugerida en vez de otra.

`incremental_rate` es exactamente esa fraccion: de cada cesta en la que el sistema pone
delante algo relevante, cuantas acaban con **una unidad de mas**. Este dataset no permite
estimarla (haria falta un A/B con el panel apagado), asi que es un supuesto y el informe
lo barre entero. Lo que **no** depende de el es la mejora relativa sobre el baseline sin
aprendizaje, que se reporta aparte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.nba.config import MarginConfig

# Bases de referencia para poder leer las cifras a una escala reconocible. El supermercado
# simulado es pequeno (unas 13.700 cestas online al mes), asi que el informe da tambien el
# impacto normalizado a estas dos bases: cualquiera puede reescalarlo a la suya.
REFERENCE_ONLINE_BASKETS = 100_000
REFERENCE_CUSTOMERS = 100_000


@dataclass(frozen=True)
class ImpactConfig:
    """Supuestos y rutas del resumen de impacto."""

    # --- Supuesto 1: cuanta de la relevancia medida se convierte en venta extra --------
    # De cada cesta en la que el ranker acierta algo que el baseline no habria puesto,
    # que fraccion acaba con una unidad incremental. Ver el docstring del modulo.
    incremental_rate: float = 0.10
    # Barrido del supuesto anterior, de "casi nada" a "optimista pero defendible".
    incremental_rate_sweep: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.30)

    # --- Supuesto 2: margen bruto ------------------------------------------------------
    # El mismo mapa por departamento que usa la politica de la Fase 4, para que las dos
    # mitades del informe hablen de los mismos euros.
    margins: MarginConfig = field(default_factory=MarginConfig)

    # --- Supuesto 3: cadencia de campana ----------------------------------------------
    # La politica de NBA decide una accion por cliente en un corte. Su horizonte de
    # retencion son 4 semanas (`PolicyConfig.retention_weeks`), asi que una oleada al mes
    # es la lectura coherente: 12 al ano, sin componer el valor de retencion.
    nba_waves_per_year: int = 12

    # --- Rutas -------------------------------------------------------------------------
    processed_dir: Path = Path("data/processed")
    recommender_metrics: Path = Path("reports/recommender/metrics.json")
    nba_metrics: Path = Path("reports/nba/metrics.json")
    reports_dir: Path = Path("reports/impact")
    markdown_path: Path = Path("IMPACT.md")

    reference_online_baskets: int = REFERENCE_ONLINE_BASKETS
    reference_customers: int = REFERENCE_CUSTOMERS

    def __post_init__(self) -> None:
        if not 0.0 <= self.incremental_rate <= 1.0:
            raise ValueError("incremental_rate es una probabilidad: debe estar en [0, 1]")
        if self.nba_waves_per_year <= 0:
            raise ValueError("nba_waves_per_year debe ser positivo")

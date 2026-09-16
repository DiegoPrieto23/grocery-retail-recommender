"""Parametros de la Fase 3: ventanas temporales, tamanos y semillas.

Todo lo que cambia el resultado del recomendador vive aqui, para que el orquestador y los
tests hablen de las mismas cifras y para que una metrica del README se pueda recalcular
sabiendo solo esta configuracion.

## Las tres ventanas

El split es **temporal y a nivel de cesta**: ninguna cesta se parte entre train y test
(`CLAUDE.md`, "Splits y evaluacion"), y ademas ninguna fuente de candidatos ve nada
posterior a la cesta que tiene que predecir.

```
    |<---------- fuentes ---------->|<-- ranker -->|<-- test -->|
    2024-01-01                 2025-09-01     2025-11-01   2026-01-01
```

- **Ventana de fuentes** (`FIT_END` hacia atras): con ella se construyen popularidad,
  estacionalidad, afinidad, historial personal, recompra y ALS que alimentan a las
  *queries* de entrenamiento del ranker.
- **Ventana del ranker** (`FIT_END` a `TEST_START`): las cestas que sirven de *queries*
  para entrenar el LambdaRank.
- **Ventana de test** (`TEST_START` en adelante): las cestas ocultas. Para evaluarlas, las
  fuentes de candidatos se **vuelven a construir** con todo lo anterior a `TEST_START`,
  incluida la ventana del ranker.

Ese doble ajuste es lo que evita el error clasico: si el ranker se entrenase con features
calculadas sobre un historial que ya contiene la cesta objetivo, aprenderia que "lo que el
cliente ya compro" es la respuesta y en test se desplomaria. Con dos ajustes, las features
de entrenamiento y las de test miran ambas hacia atras desde el borde de su ventana.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

SEED = 42

# Bordes de las tres ventanas (inclusivo por la izquierda, exclusivo por la derecha).
FIT_END = dt.date(2025, 9, 1)
TEST_START = dt.date(2025, 11, 1)


@dataclass(frozen=True)
class CandidateConfig:
    """Cuantos candidatos aporta cada fuente antes de unirlas.

    Son topes por *query*, no por catalogo: el pool final ronda los 139 productos unicos
    de los 496 posibles (155 de 1.500 antes de la Fase 7a).

    Las cifras de esta nota se midieron en la Fase 3, sobre el dataset anterior a la 7a.
    Los topes no se han reajustado al catalogo nuevo: con ellos el `pool_recall` sube al
    76,9 % (`reports/recommender/metrics.md`).

    Los tamanos no son redondos por gusto, salen de medir el techo de cada fuente sobre
    las cestas de test. El historial personal es el caso claro: un cliente tiene 117
    productos distintos comprados de media (mediana 86), y de las lineas de una cesta
    futura solo el 29,7 % son productos que ya habia comprado. Cuantas de esas alcanza la
    fuente segun donde se corte:

    | `n_personal` | Lineas de test cubiertas |
    | ---: | ---: |
    | 25 | 6,2 % |
    | 40 | 8,6 % |
    | 60 | 11,6 % |
    | **80** | **14,2 %** |
    | 120 | 18,0 % |
    | 200 | 23,1 % |

    Cortar en 25 dejaba fuera tres cuartas partes de lo que el historial podia aportar. A
    partir de 80 la curva se aplana y cada candidato extra multiplica las filas que hay
    que puntuar (18.000 queries x 155 candidatos ya son 2,8 M de filas).

    **Y aun asi el techo no era este.** Pasar de 92 a 155 candidatos por cesta subio el
    `pool_recall` del 21,7 % al 31,1 % -- un 43 % mas de target alcanzable-- y el NDCG@5
    se quedo clavado en 0,034. Los candidatos que entran con el pool ampliado son
    productos que el ranker no sabe distinguir, porque en este dataset la eleccion de SKU
    dentro de una categoria es casi aleatoria (ver `evaluate.category_metrics` y la nota
    de la Fase 3 en `ROADMAP.md`). El sistema no esta limitado por la primera etapa: esta
    limitado por la senal que hay en el dato.
    """

    n_popularity: int = 60
    n_affinity_product: int = 30
    n_affinity_category: int = 10
    # Productos por categoria consecuente en la fuente de afinidad de categoria.
    n_products_per_category: int = 4
    n_personal: int = 80
    n_als: int = 50

    # Dias hacia atras que cuentan como "popularidad reciente".
    recent_days: int = 90


@dataclass(frozen=True)
class ALSConfig:
    """Hiperparametros del ALS de Spark MLlib (fuente colaborativa)."""

    rank: int = 32
    max_iter: int = 10
    reg_param: float = 0.05
    alpha: float = 20.0
    seed: int = SEED


@dataclass(frozen=True)
class RankerConfig:
    """Hiperparametros del LightGBM con objetivo `lambdarank`."""

    objective: str = "lambdarank"
    metric: str = "ndcg"
    eval_at: tuple[int, ...] = (5,)
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 800
    early_stopping_rounds: int = 50
    seed: int = SEED


@dataclass(frozen=True)
class RecommenderConfig:
    """Configuracion completa de la Fase 3."""

    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    predictions_dir: Path = Path("predictions")
    reports_dir: Path = Path("reports/recommender")

    fit_end: dt.date = FIT_END
    test_start: dt.date = TEST_START

    # Cuantas cestas se convierten en query. Muestrear no cambia el estimador de la
    # metrica (la muestra es determinista y sin sesgo por perfil), solo su varianza, y
    # divide por tres el coste de puntuar el pool de candidatos.
    n_train_queries: int = 12_000
    n_valid_queries: int = 2_500
    n_test_queries: int = 18_000

    # Recomendaciones que devuelve el sistema.
    top_k: int = 5

    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    als: ALSConfig = field(default_factory=ALSConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.fit_end >= self.test_start:
            raise ValueError("fit_end debe ser anterior a test_start")


# Tablas de `data/processed` que necesita la Fase 3.
REQUIRED_TABLES: tuple[str, ...] = (
    "baskets",
    "basket_items",
    "products",
    "promotions",
    "customers",
    "sessions",
    "session_events",
)

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

from src.recommender.schema import RELEVANCE_GAIN

SEED = 42

# Bordes de las tres ventanas (inclusivo por la izquierda, exclusivo por la derecha).
FIT_END = dt.date(2025, 9, 1)
TEST_START = dt.date(2025, 11, 1)


@dataclass(frozen=True)
class CandidateConfig:
    """Cuantos candidatos aporta cada fuente antes de unirlas.

    Son topes por *query*, no por catalogo: el pool final ronda los 234 productos unicos
    de los 496 posibles (139 antes del punto M6; 155 de 1.500 antes de la Fase 7a).

    ## Reajuste del punto M6

    Los topes venian de la Fase 3, medidos sobre un catalogo de 1.500 productos, y no se
    habian tocado desde entonces. El punto M6 los revisa sobre el catalogo de 496 y sobre
    las fuentes ya cambiadas por A1, A2 y A4. Medido con `pool_recall` y con la
    **cobertura de categorias** del pool sobre 6.000 cestas de test:

    | | pool | recall de SKU | categorias cubiertas | perfil 1 (SKU) | perfil 1 (cat.) |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | Topes de la Fase 3 | 141 | 82,7 % | 96,5 % | 55,1 % | 70,4 % |
    | Solo `n_popularity` = 150 | 205 | 89,8 % | 99,6 % | 74,9 % | 96,3 % |
    | **Servido** | **234** | **91,7 %** | **100 %** | **75,9 %** | **100 %** |

    Lo que estrangulaba al cold-start no era el numero de productos, era **cuantas
    categorias distintas tocaba el pool**: un cliente nuevo con el carrito vacio solo
    recibia alguna referencia del 70 % de las categorias que acabaria comprando, y eso es
    un techo duro sobre `cat_hit_rate@5`, la metrica principal. Ningun ranker lo recupera.

    La rama por categoria de la fuente `pop` (`fit_category_popularity`) lo cierra, pero
    **con un solo lider por categoria**. Con dos o tres pierde contra ampliar la
    popularidad global a igualdad de pool (a 205 candidatos: 70,7 % frente a 74,9 % en el
    perfil 1), porque gasta huecos en categorias que casi nunca aparecen mientras deja
    fuera la tercera y la cuarta referencia de las que si. Las dos ramas no compiten: la
    global reparte por demanda esperada y la de categoria cubre la cola por abajo.

    ## Por que cada tope esta donde esta

    Los tamanos salen de medir el techo de cada fuente, no de redondear. El historial
    personal es el caso claro: un cliente tiene 117 productos distintos comprados de media
    (mediana 86), y de las lineas de una cesta futura solo el 29,7 % son productos que ya
    habia comprado. Cuantas de esas alcanza la fuente segun donde se corte:

    | `n_personal` | Lineas de test cubiertas |
    | ---: | ---: |
    | 25 | 6,2 % |
    | 40 | 8,6 % |
    | 60 | 11,6 % |
    | 80 | 14,2 % |
    | **120** | **18,0 %** |
    | 200 | 23,1 % |

    Cada candidato extra multiplica las filas que hay que puntuar (18.000 queries x 234
    candidatos son 4,2 M de filas), asi que el criterio para subir un tope es que mueva el
    `pool_recall` de algun perfil, no el total.

    **Y el techo del sistema sigue sin estar aqui.** En la Fase 3, pasar de 92 a 155
    candidatos subio el `pool_recall` del 21,7 % al 31,1 % y el NDCG@5 se quedo clavado en
    0,034: los candidatos que entran con el pool ampliado son productos que el ranker no
    sabe distinguir, porque la eleccion de SKU dentro de una categoria es casi aleatoria
    (ver `evaluate.category_metrics`). Lo que el punto M6 arregla no es esa parte, es la
    cobertura de **categorias**, que si es una restriccion real sobre la metrica principal.
    Cuanto de esto llega a la cifra final esta en `reports/recommender/metrics.md`.
    """

    n_popularity: int = 150
    # Rama por categoria de la fuente `pop` (punto M6): los lideres de las
    # `n_pop_categories` categorias de mas peso esperado en el mes. Con 62, todas las del
    # catalogo. `0` la desactiva y deja solo la popularidad global de antes.
    n_pop_categories: int = 62
    n_pop_products_per_category: int = 1
    n_affinity_product: int = 50
    n_affinity_category: int = 15
    # Productos por categoria consecuente en la fuente de afinidad de categoria.
    n_products_per_category: int = 4
    n_personal: int = 120
    n_als: int = 80

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


# Las dos funciones objetivo del ranker (punto A4 del diagnostico).
RELEVANCE_GRADED = "graded"
RELEVANCE_BINARY_SKU = "sku"


# Como se separa la validacion del ranker dentro de su ventana (punto B4 del diagnostico).
VALID_HASH = "hash"
VALID_TEMPORAL = "temporal"
VALID_MODES: tuple[str, ...] = (VALID_HASH, VALID_TEMPORAL)


@dataclass(frozen=True)
class ValidationSplit:
    """De donde salen las queries con las que para el entrenamiento (punto B4).

    La parada temprana decide cuantos arboles se sirven, asi que la validacion deberia
    parecerse a lo que el modelo va a encontrar en produccion. En este proyecto eso es
    siempre **mas adelante en el tiempo**: el test empieza donde acaba la ventana del
    ranker.

    - `temporal` (servido): las cestas de los ultimos `days` dias de la ventana del
      ranker. El modelo para donde deja de mejorar sobre queries posteriores a las de
      entrenamiento, que es la misma direccion del desplazamiento hacia el test.
    - `hash`: `n_valid_queries` cestas muestreadas por hash de toda la ventana, lo que se
      hacia hasta el punto B4. Se mantiene como ablacion para poder medir la diferencia.

    Ninguno de los dos parte una cesta: la unidad es siempre el `basket_id`
    (`CLAUDE.md`, "Splits y evaluacion").
    """

    mode: str = VALID_TEMPORAL
    # Solo en modo `temporal`: ultimos dias de la ventana del ranker que hacen de
    # validacion. Dos semanas dejan ~3/4 de la ventana para entrenar y una muestra de
    # validacion del tamano del que habia por hash.
    days: int = 14

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"modo de validacion desconocido: {self.mode!r}")
        if self.days < 1:
            raise ValueError("days debe ser al menos 1")


@dataclass(frozen=True)
class RankerConfig:
    """Hiperparametros del LightGBM con objetivo `lambdarank`.

    `relevance` elige la etiqueta que optimiza la NDCG del LambdaRank:

    - `"graded"` (la servida desde el punto A4): columna `relevance` con 2 para el SKU
      exacto y 1 para la misma categoria, y ganancias `label_gain` = 0, 1, 3. Es la
      metrica principal del proyecto (`CHALLENGE.md`).
    - `"sku"`: la relevancia binaria de SKU exacto (columna `label`) con ganancias 0, 1,
      que es lo que se optimizaba hasta el punto A4. Se sigue entrenando como ablacion.

    Tambien decide que queries se descartan del entrenamiento por no tener gradiente:
    las que no tienen ningun candidato con relevancia positiva.
    """

    objective: str = "lambdarank"
    metric: str = "ndcg"
    eval_at: tuple[int, ...] = (5,)
    relevance: str = RELEVANCE_GRADED
    label_gain: tuple[float, ...] = RELEVANCE_GAIN
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

    def __post_init__(self) -> None:
        if self.relevance not in (RELEVANCE_GRADED, RELEVANCE_BINARY_SKU):
            raise ValueError(f"relevance desconocida: {self.relevance!r}")

    @property
    def label_column(self) -> str:
        """Columna de la matriz que hace de etiqueta."""
        return "relevance" if self.relevance == RELEVANCE_GRADED else "label"

    @property
    def gains(self) -> list[float]:
        """`label_gain` de LightGBM para la etiqueta elegida."""
        return list(self.label_gain) if self.relevance == RELEVANCE_GRADED else [0.0, 1.0]


@dataclass(frozen=True)
class RerankConfig:
    """Reglas que se aplican sobre el orden del ranker antes de cortar el top-k (punto A2).

    Por construccion del generador una cesta lleva casi siempre una sola linea por
    categoria (desde la Fase 8, las de exploracion llevan dos en ~12 % de las veces), asi
    que un segundo producto de la misma categoria en el top-k, o uno de una categoria que
    ya esta en el carrito, casi nunca puede acertar: son huecos regalados.

    - `max_per_category`: cuantas referencias de una misma categoria caben en el top-k
      (cuota). `None` desactiva la regla.
    - `exclude_cart_categories`: si se relegan las categorias que ya estan en el carrito.

    Ninguna regla elimina candidatos: los relega detras de los admisibles, en su orden de
    score, asi que la lista sigue teniendo `k` productos aunque el pool sea corto. Se
    aplica igual en `evaluate.top_k_predictions` y en `serving.rank_queries`
    (`src/recommender/rerank.py`).
    """

    max_per_category: int | None = 1
    exclude_cart_categories: bool = True

    @classmethod
    def off(cls) -> "RerankConfig":
        """Sin re-ranking: el top-k es el orden del ranker tal cual."""
        return cls(max_per_category=None, exclude_cart_categories=False)

    @property
    def active(self) -> bool:
        return self.max_per_category is not None or self.exclude_cart_categories


# Como se corta cada cesta en prefijo y target (punto M3, `splits.build_query_items`).
CUT_HEADLINE = "headline"
CUT_ALL_PREFIXES = "all_prefixes"
CUT_RANDOM_FRACTIONS = "random_fractions"
CUT_EMPTY_AND_HALF = "empty_and_half"
CUT_MODES: tuple[str, ...] = (
    CUT_HEADLINE,
    CUT_ALL_PREFIXES,
    CUT_RANDOM_FRACTIONS,
    CUT_EMPTY_AND_HALF,
)


@dataclass(frozen=True)
class CutPlan:
    """Cortes por cesta con los que se construyen las queries (punto M3).

    - `headline`: un corte por cesta, el de siempre. El hash del `basket_id` decide si el
      carrito va vacio o con la mitad de las lineas. Es la cifra de cabecera y la serie
      historica de `metrics.md`.
    - `all_prefixes`: todos los `k` en `1..n-1`, una query por corte.
    - `random_fractions`: `n_fractions` cortes por cesta con `k` uniforme en `1..n-1`,
      sorteado con un hash de (`basket_id`, `seed`, j). Si dos sorteos coinciden, el corte
      cuenta una sola vez.
    - `empty_and_half`: los dos cortes de cabecera a la vez, carrito vacio y mitad del
      ticket. Lo usa el sobremuestreo del cold-start (punto M4): cada cesta aporta una
      query al perfil 1 y otra al 2.
    """

    mode: str = CUT_HEADLINE
    n_fractions: int = 3
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.mode not in CUT_MODES:
            raise ValueError(f"modo de corte desconocido: {self.mode!r}")
        if self.n_fractions < 1:
            raise ValueError("n_fractions debe ser al menos 1")

    @property
    def one_per_basket(self) -> bool:
        """Si cada cesta da una sola query (y la clave de query es el `basket_id`)."""
        return self.mode == CUT_HEADLINE


@dataclass(frozen=True)
class BootstrapConfig:
    """Bootstrap de las metricas (punto M4, `evaluate.bootstrap_means`).

    Se remuestrean **cestas**, no queries: con varios cortes por cesta, las queries de una
    misma cesta estan correladas, y remuestrearlas por separado estrecharia el intervalo
    sin motivo. Con un corte por cesta las dos cosas coinciden.

    Con 1.000 remuestreos, el error Monte Carlo de un extremo del intervalo al 95 % es
    pequeno frente a su anchura. El p-valor mas pequeno que se puede reportar es
    `1 / (n_resamples + 1)`.
    """

    n_resamples: int = 1_000
    confidence: float = 0.95
    seed: int = SEED

    def __post_init__(self) -> None:
        if not 0 < self.confidence < 1:
            raise ValueError("confidence debe estar entre 0 y 1")
        if self.n_resamples < 1:
            raise ValueError("n_resamples debe ser al menos 1")


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
    # Cuantas cestas de mas se muestrean de la ventana del ranker para la validacion. Con
    # `ValidationSplit.mode = "temporal"` (punto B4) ya no decide cuales caen en
    # validacion -- eso lo hace la fecha--, solo el tamano total de la muestra.
    n_valid_queries: int = 2_500
    n_test_queries: int = 18_000

    # Recomendaciones que devuelve el sistema.
    top_k: int = 5

    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    als: ALSConfig = field(default_factory=ALSConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    # De donde sale la validacion de la parada temprana (punto B4).
    validation: ValidationSplit = field(default_factory=ValidationSplit)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    seed: int = SEED

    # Evaluacion con varios cortes por cesta (punto M3): cuantas cestas de test se
    # explotan y como. Con `all_prefixes` salen unas 4 queries por cesta (5,1 lineas de
    # media), asi que 3.000 cestas cuestan menos que las 18.000 queries de cabecera. Las
    # cestas se eligen con el mismo hash que la muestra de cabecera. `None` la desactiva.
    n_cut_baskets: int | None = 3_000
    cuts: CutPlan = field(default_factory=lambda: CutPlan(mode=CUT_ALL_PREFIXES))

    # Sobremuestreo del cold-start (punto M4): todas las cestas de la ventana de test de
    # clientes sin compras anteriores (o anonimas), con los dos cortes de cabecera.
    cold_start_oversample: bool = True

    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

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

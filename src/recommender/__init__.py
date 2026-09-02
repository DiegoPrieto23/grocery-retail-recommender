"""Recomendador de cesta en dos etapas (Fase 3 del ROADMAP, Tarea 3a de CHALLENGE.md).

Modulos:

- `config`: ventanas temporales, tamanos e hiperparametros.
- `splits`: split temporal por cesta y construccion de las *queries* (prefijo + target).
- `candidates`: primera etapa -- popularidad/estacionalidad, co-compra de producto y de
  categoria, historial personal con recompra, y ALS de Spark MLlib.
- `features`: matriz de features del par `(query, candidato)`.
- `ranker`: segunda etapa -- LightGBM con objetivo `LambdaRank`.
- `evaluate`: NDCG@5, Recall@5 y hit rate, desglosados por perfil de cliente.
- `pipeline`: orquestador de extremo a extremo.
- `demo_profiles`: un ejemplo real de cada uno de los cuatro perfiles.
"""

# Solo se reexporta lo que no comparte nombre con un submodulo: `from ... import evaluate`
# devolveria la funcion en vez del modulo `evaluate` y romperia a quien importe asi.
from src.recommender.config import RecommenderConfig
from src.recommender.splits import PROFILE_LABELS

__all__ = [
    "PROFILE_LABELS",
    "RecommenderConfig",
]

"""Next Best Action (Fase 4 del ROADMAP, Tarea 3b de CHALLENGE.md).

Modulos:

- `config`: cortes temporales, catalogo de acciones y **todos los supuestos economicos**.
  Es el primero que hay que leer: separa lo medido de lo supuesto.
- `targets`: las dos etiquetas de propension construidas en un corte, sin fuga.
- `features`: features de cliente y de cliente x categoria, mirando solo hacia atras.
- `propensity`: los dos LightGBM binarios y sus metricas (AUC, PR-AUC, lift).
- `policy`: la politica de valor esperado y las comparaciones contra las alternativas
  triviales, mas el barrido de sensibilidad del uplift supuesto.
- `pipeline`: orquestador de extremo a extremo.
"""

from src.nba.config import ACTIONS, NBAConfig

__all__ = [
    "ACTIONS",
    "NBAConfig",
]

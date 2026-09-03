"""Traduccion de las metricas de las Fases 3 y 4 a euros (Tarea 4 de `CHALLENGE.md`).

- `config.py`: los supuestos economicos, todos juntos y todos declarados.
- `model.py`: la aritmetica, en funciones puras y sin ficheros de por medio.
- `pipeline.py`: mide lo que se puede medir del dato, aplica el modelo y escribe
  `IMPACT.md` y `reports/impact/impact.json`.

Se ejecuta con `python -m src.impact.pipeline`.
"""

from src.impact.config import ImpactConfig

__all__ = ["ImpactConfig"]

"""Preguntas de negocio de la Tarea 1, como funciones y no como celdas de notebook.

`questions.py` contiene las nueve consultas del EDA (`notebooks/01_eda.ipynb`) y los
calculos derivados que antes vivian sueltos en el cuaderno. El notebook las **invoca**;
`src/eda/findings.py` las reutiliza para escribir el informe de hallazgos de negocio de
la Fase 5. Asi la logica esta en un sitio, tiene tests, y cualquier cifra publicada se
puede recalcular ejecutando un script.
"""

from src.eda.questions import REQUIRED_VIEWS, register_views

__all__ = ["REQUIRED_VIEWS", "register_views"]

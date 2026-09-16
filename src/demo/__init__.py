"""Piezas de presentacion de la demo (Fase 6b).

`catalog.py` resuelve lo que la app necesita para pintar un producto: nombre, foto,
precio, el motivo de una recomendacion y la lectura de una accion del NBA. Vive aqui y no
en `streamlit_app.py` para que se pueda probar sin levantar la app. `baskets.py` hace lo
mismo con las cestas reales de test y el acierto en dos niveles (categoria y SKU); lo
prueban `tests/test_demo_baskets.py` y `tests/test_demo_hits.py`.

La inferencia del recomendador no esta aqui, sino en `src/serving/`.
"""

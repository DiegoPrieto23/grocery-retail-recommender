# CLAUDE.md — convenciones del proyecto

Contexto para Claude Code al trabajar en este repo. Lee también `CHALLENGE.md` (el reto),
`DATA_SPEC.md` (esquema del dataset) y `ROADMAP.md` (fases) antes de generar código.

## Qué es este proyecto

Dataset sintético de retail de gran consumo (supermercado) + recomendador de cesta +
Next Best Action sobre un modelo de propensión. Es la "fase 2" del ciclo de vida del
cliente en retail que empezó con `sports-rental-analytics`, pero con un stack distinto
a propósito.

## Stack

- **PySpark** para generación a escala, ETL y feature engineering — no dbt en este proyecto.
- **Recomendador**: arquitectura de dos etapas. Candidatos con ALS (Spark MLlib) + co-compra
  + popularidad; ranking final con **LightGBM** (objetivo `LambdaRank`). Sin deep learning:
  es el patrón habitual en recomendadores de retail reales y corre bien en CPU.
- **NBA**: scikit-learn / LightGBM para el modelo de propensión.
- **Demo**: Streamlit para la app interactiva de la Fase 6b — es el entregable central del
  proyecto (Tarea 4 de `CHALLENGE.md`). Carga los modelos ya entrenados, no reentrena nada
  en caliente. Interfaz gráfica e intuitiva: productos mostrados como tarjetas con una foto
  real de su `visual_group` (una por grupo, no por producto ni generada por IA), nunca como
  tablas de texto plano ni iconos/emoji.
- **Preparación de imágenes** (Fase 6a, `assets/`): API de Pexels vía `requests`. La
  `PEXELS_API_KEY` vive en `.env` (cargada con `python-dotenv`), nunca hardcodeada ni
  comiteada — `.env` está en `.gitignore`; `.env.example` (sin valores reales) sí se
  comitea, como documentación de qué variables hacen falta. Es la única parte del proyecto
  con dependencia de internet; se ejecuta una vez y su resultado (`assets/` + el CSV de
  mapeo) se cachea/comitea, así que la demo en sí vuelve a arrancar en local sin red.
- **Estilo de la demo**: skill `developing-with-streamlit` (oficial, repo
  `streamlit/agent-skills`) para theming y estilizado — instalarla en `.claude/skills/`
  antes de empezar la Fase 6b. Complementarla con un tema propio en
  `.streamlit/config.toml` (colores, fuente) en vez de dejar el tema por defecto.
- Python 3.10+.

Un dashboard en Power BI estuvo planteado como pieza de BI Engineering, pero queda aparcado
por ahora a favor de la demo — ver "Fuera de alcance" en `ROADMAP.md`.

## Reproducibilidad

- Semilla fija (`SEED = 42`) propagada a numpy, `random` y Faker en el generador.
- Dos ejecuciones del generador deben producir ficheros idénticos (mismo criterio que en
  `sports-rental-analytics` — verificarlo con un test o un check de CI).
- Fijar versiones de librerías en `constraints.txt` una vez el dataset esté estable.
- Excepción: las imágenes de `assets/` (Fase 6a, vía Pexels) no son reproducibles por
  semilla — el resultado de una búsqueda puede cambiar con el tiempo. Por eso se descargan
  una sola vez y se tratan como un fixture cacheado, no como algo que se regenera en cada
  ejecución del pipeline.

## Estilo

- Nombres de columnas y tablas en `snake_case`, en inglés (como en `sports-rental-analytics`).
- Documentación (README, docstrings de alto nivel, comentarios de negocio) en español.
- Código y nombres de funciones/variables en inglés.
- Preferir funciones puras y testeables sobre notebooks monolíticos: la lógica de negocio
  vive en `src/`, los notebooks solo la invocan y visualizan.

## Splits y evaluación

- El split train/test del recomendador se hace por `basket_id` (o por `customer_id` si se
  quiere evaluar cold-start), nunca por fila suelta de `basket_items` — evitar fuga de datos.
- El split del modelo de propensión es temporal (entrenar con periodo pasado, evaluar con
  periodo posterior), no aleatorio.
- Cualquier métrica que se reporte en el README debe poder recalcularse ejecutando un
  script (`verify_*.py` o test), no copiarse a mano de un notebook — mismo criterio de
  verificación que en `sports-rental-analytics`.

## Al generar código nuevo

- Antes de añadir una tabla o columna nueva, comprobar que no rompe el esquema de
  `DATA_SPEC.md`; si hace falta cambiarlo, actualizar ese documento en el mismo cambio.
- Antes de marcar una fase del `ROADMAP.md` como hecha, dejar constancia de cómo se
  verifica (test, notebook, o script de verificación).
- No usar datos reales de clientes ni de ningún retailer concreto en ningún momento.

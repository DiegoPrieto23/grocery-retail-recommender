# Cómo arrancar el proyecto

Guía para alguien que acaba de clonar el repo. Si lo que buscas es **qué es** el proyecto,
empieza por el [`README.md`](README.md); si buscas **cómo funcionan los modelos** en lenguaje
llano, por [`docs/como-funcionan-recomendador-y-nba.md`](docs/como-funcionan-recomendador-y-nba.md).

> Este documento describía hasta la Fase 8 el arranque original con Claude Code, que ya no
> se parecía a lo que hay. Los prompts de construcción se conservan tal cual en
> [`docs/prompts-de-construccion.md`](docs/prompts-de-construccion.md); esto de aquí es la
> guía del pipeline actual (punto B3 del [diagnóstico](docs/diagnostico-fase7.md)).

## Lo que hace falta

- **Python 3.10 – 3.12.** PySpark 3.5 todavía no soporta 3.13.
- **Una JVM 17** (Temurin va bien). PySpark la necesita a partir de la Fase 2.
- Unos **8 GB de RAM libres** y unos **6 GB de disco** para los artefactos generados.
- **Ninguna clave ni conexión de red**, con una excepción: volver a descargar las fotos del
  catálogo de Pexels. Las fotos ya están en `assets/` y versionadas, así que no hace falta.

```bash
git clone <este-repo> && cd grocery-retail-recommender
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
```

`constraints.txt` fija las versiones. Instalar sin él funciona, pero entonces las cifras de
los informes pueden no salir idénticas.

## Comprobar que el entorno está bien, antes de gastar una hora

```bash
pytest -q                          # ~8 min, 453 tests
python -m src.pipeline list        # los 13 pasos y sus dependencias
python -m src.pipeline all --dry-run
```

En un repo recién clonado no hay `data/`, así que unos cuantos tests **se saltan solos**:
los que necesitan el bundle de serving o las tablas procesadas. Es esperado. Si pasan los
que sí corren, el entorno está bien.

## Generar todo

```bash
python -m src.pipeline all         # ~75 min
```

Un solo comando: `src/pipeline.py` conoce las dependencias entre pasos y los ordena. Lo que
deja, en orden:

| Paso | Deja en |
| --- | --- |
| `generate` | `data/raw/` — los 7 CSV, ~4,4 M de líneas de ticket |
| `verify-dataset` | `reports/etl/verify_dataset.json` |
| `etl` | `data/processed/` — 7 tablas limpias + 4 de features |
| `recommender` | `models/`, `predictions/`, `reports/recommender/` |
| `export-bundle` | `data/serving/` — lo que la demo necesita para correr sin Spark |
| `export-oracle` | `data/oracle/` — el techo teórico |
| `nba` | `models/`, `predictions/nba_actions.parquet`, `reports/nba/` |
| `impact` | `IMPACT.md` |
| `findings` | `reports/insights/` |
| `assets` | `assets/product_catalog.csv` |

Si algo falla a mitad, se retoma sin repetir lo hecho:

```bash
python -m src.pipeline all --from recommender
```

Y para probar el circuito entero en unos minutos en vez de en una hora, con un dataset
pequeño (es lo que corre el job de smoke en CI):

```bash
python -m src.pipeline all --scale 0.02
```

Ojo: a esa escala las **cifras no son las del README**. Sirve para comprobar que la cadena
funciona de punta a punta, no para reproducir resultados.

## Levantar la demo

```bash
streamlit run streamlit_app.py     # http://localhost:8501
```

Necesita `data/serving/` (paso `export-bundle`), `models/` y
`predictions/nba_actions.parquet`. Si falta algo, la app dice qué comando lo genera en vez
de reventar con un *stack trace*.

La demo **solo hace inferencia**: carga los modelos ya entrenados y no reentrena nada ni
llama a ninguna API.

## El dataset no se versiona, y aun así es reproducible

`data/` está en `.gitignore`. No se sube, se regenera: la semilla está fija (`SEED = 42`) y
propagada a numpy, `random` y Faker, con sub-*streams* por etapa. Dos ejecuciones dan
ficheros **byte a byte idénticos**, y eso lo comprueba un test comparando los `sha256` del
manifiesto (`data/raw/manifest.json`).

Hay dos excepciones declaradas:

- **Las fotos de `assets/`** salen de la API de Pexels y una búsqueda puede dar resultados
  distintos con el tiempo. Por eso se descargaron una vez y están versionadas.
- **El entrenamiento del ranker no es del todo determinista entre ejecuciones**: la parada
  temprana cae en un número de árboles distinto y la métrica se mueve en la tercera decimal.
  Está anotado como deuda en el [`README.md`](README.md#qué-viene-ahora).

## Si quieres tocar algo

- **Los supuestos de negocio** (márgenes, coste del cupón, uplifts, incrementalidad) están
  todos juntos y comentados en `src/nba/config.py` y `src/impact/config.py`. Son supuestos
  declarados, no estimaciones, y el porqué está en esos docstrings.
- **Los parámetros del recomendador** (ventanas temporales, topes de candidatos,
  hiperparámetros, re-ranking) están en `src/recommender/config.py`.
- **El dominio del generador** (categorías, misiones de compra, afinidades, estacionalidad)
  está en `data_generation/catalog.py`, y el esquema que produce, en
  [`DATA_SPEC.md`](DATA_SPEC.md).
- **La lógica de negocio vive en `src/`**, no en los notebooks ni en `streamlit_app.py`:
  eso último solo dibuja. Es la regla que fija [`CLAUDE.md`](CLAUDE.md) y lo que permite
  testear el porqué de una recomendación sin levantar la app.

Cambiar el generador obliga a rehacer todo lo que cuelga de él. Eso es un comando
(`python -m src.pipeline all`) y una hora, pero las cifras de los informes cambiarán: la
convención del repo es congelar las anteriores en un `reports/*/baseline_*.json` antes de
tocar nada, para poder comparar después.

## Extras opcionales

```bash
pip install jupyterlab             # para abrir notebooks/01_eda.ipynb
pip install mlflow                 # las Fases 3 y 4 registran cada ejecución
```

Sin MLflow instalado, el registro de experimentos es un *no-op* (`src/tracking.py`): no
falla ni pide nada.

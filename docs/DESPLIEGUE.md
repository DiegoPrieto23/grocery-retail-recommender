# Despliegue de la demo

Cómo se publica [`streamlit_app.py`](../streamlit_app.py) para que el enlace del
[`README`](../README.md) lleve a una app viva.

## Por qué no es GitHub Pages

GitHub Pages sirve **ficheros estáticos**: HTML, CSS, JavaScript y poco más. Una app de
Streamlit no es un fichero, es un **proceso de Python** que recibe cada interacción por
WebSocket, vuelve a ejecutar el script y devuelve el resultado. Sin un servidor que ejecute
Python no hay demo, así que Pages queda descartado por construcción, no por configuración.

Hay un camino que sí serviría en Pages —[stlite](https://github.com/whitphx/stlite), que
lleva Streamlit a WebAssembly con Pyodide— pero esta demo carga **LightGBM**, que no está
disponible en Pyodide. Sin el ranker no hay recomendador.

La plataforma equivalente, gratuita y conectada a este mismo repositorio, es **Streamlit
Community Cloud**.

## El problema: lo que la app necesita no cabe

La demo **solo hace inferencia**, pero esa inferencia necesita artefactos que produce un
pipeline de 75 minutos sobre una JVM. En Community Cloud eso no se puede ejecutar, así que
lo que la app lee tiene que estar en el repositorio — y el bundle completo no es razonable
subirlo ni cargarlo:

| | Bundle completo | En el repo |
| --- | ---: | ---: |
| `customer_lines` | 4.216.494 filas · 366 MB en RAM | ❌ |
| `als_topn` | 1.498.320 filas · 184 MB (×2, indexado) | ❌ |
| `customer_baskets` | 582.174 filas · 75 MB | ❌ |
| `data/processed/basket_items.parquet` | 4,3 M filas · 33 MB en disco | ❌ |
| **Total en memoria** | **826 MB** | |

Community Cloud asigna entre **690 MB y 2,7 GB de RAM** por app. Con el intérprete,
Streamlit, pandas y el *booster* encima, 826 MB de tablas dejan la app al borde del límite
todo el rato.

## La solución: recortar a quien la demo puede enseñar

El selector ofrece **20 clientes por escenario** (fiel / ocasional / en riesgo / todos), así
que como mucho se pueden elegir unas decenas de los 18.729. El otro 99 % del bundle es
historial de gente que nadie va a mirar.

`python -m src.serving.export_demo_bundle` escribe en `data/serving/demo/` esas mismas
tablas filtradas a los clientes ofrecidos, más las líneas de ticket de las cestas que se
pueden cargar. Es un paso más del pipeline (`export-demo-bundle`), no un script suelto.

| | Completo | Recortado |
| --- | ---: | ---: |
| `customer_lines` | 4.216.494 filas | 31.501 |
| `customer_baskets` | 582.174 filas | 4.361 |
| `als_topn` | 1.498.320 filas | 6.240 |
| `basket_items` | 4,3 M filas | 1.057 |
| **En disco** | 80 MB | **0,3 MB** |
| **En memoria** | 826 MB | **21 MB** |

Son **78 clientes y 106 cestas**: la unión de los cuatro escenarios.

### Lo que no se recorta, y por qué

Tres tablas parecen recortables y no lo son, porque deciden **quién** entra en el selector:

- `customer_stats` y `parity/queries.parquet` — los escenarios se definen por **cuantiles de
  la población elegible** (`src/demo/customers.py`), así que filtrarlas movería los umbrales
  y la app publicada acabaría ofreciendo a otra gente. Además es circular: el recorte se
  calcula a partir de ellas.
- `known_customers` — decide si un cliente es recurrente o *cold-start*.

Las tres juntas no llegan a 1,2 MB, así que se versionan enteras.

### Cómo lo resuelve la app, sin configuración

`src/serving/recommend.resolve_table` y `src/demo/baskets.basket_items` **prefieren siempre
la tabla completa y solo caen a la recortada si no está**. En local mandan las tablas
enteras —y los tests de paridad, que recorren clientes de todo el dataset, siguen midiendo
lo que medían—; en el despliegue solo está la recortada y la app arranca con ella. No hay
variable de entorno, ni modo demo, ni rama aparte.

Está verificado en los dos sentidos ([`tests/test_demo_bundle.py`](../tests/test_demo_bundle.py)):
que el recorte exporta exactamente los clientes que el selector ofrece, que exportar no
cambia esa lista, y que con solo los ficheros versionados el **top-5 sale idéntico** al del
bundle completo.

## Qué se versiona, en total

~4,5 MB, de los cuales 3,3 MB son las fotos del catálogo que ya estaban:

| Ruta | Tamaño | Para qué |
| --- | ---: | --- |
| `data/serving/` (sin las 3 tablas grandes) | 1,8 MB | Fuentes de candidatos, popularidad y afinidades |
| `data/serving/demo/` | 0,3 MB | El recorte: historial y líneas de los clientes ofrecidos |
| `data/processed/{customers,products,promotions}.parquet` | 0,3 MB | Atributos, catálogo y promociones vigentes |
| `models/recommender_ranker_lgbm.txt` | 0,9 MB | El ranker LambdaRank |
| `predictions/nba_actions.parquet` | 0,6 MB | La acción del NBA resuelta en el corte del 1-nov |
| `assets/` | 3,3 MB | Las 60 fotos del catálogo (ya estaban versionadas) |
| `reports/recommender/metrics.json` | — | Las cifras de referencia del banner (ya estaba versionada) |

Nada de eso es dataset crudo: `data/raw/` sigue sin versionarse y se sigue regenerando con
la semilla fija. Lo que se sube es **el artefacto de servicio**, que es lo que se subiría
también en un despliegue de verdad.

Si alguno falta, la app no revienta con un *stack trace*: dice qué comando lo genera.

## Publicar en Streamlit Community Cloud

1. Con el pipeline ya ejecutado en local, generar el recorte y empujar:

   ```bash
   python -m src.pipeline export-demo-bundle --only
   git add -A && git commit && git push
   ```

2. Entrar en [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub y
   autorizar el acceso al repositorio.
3. **Create app** → *Deploy a public app from GitHub*, con:
   - **Repository:** `DiegoPrieto23/grocery-retail-recommender`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** `grocery-retail-recommender` → queda
     `https://grocery-retail-recommender.streamlit.app`, que es la que el README ya enlaza.
     Si se elige otro subdominio hay que cambiar los tres enlaces del README.
4. Desplegar. El primer arranque tarda varios minutos: Community Cloud instala
   `requirements.txt` entero, PySpark incluido. La demo **no** lo usa en caliente, pero el
   fichero es uno solo para todo el proyecto y partirlo complicaría el arranque local y la
   CI a cambio de unos minutos que solo se pagan al construir.

A partir de ahí, cada `push` a `main` redespliega la app sola.

**No hay secretos que configurar.** La única parte del proyecto con red es la descarga de
fotos de Pexels (`src/catalog/`), que se ejecutó una vez y dejó `assets/` cacheado y
versionado. La app no necesita `PEXELS_API_KEY` ni ninguna otra credencial.

## Después de regenerar el dataset

Si se vuelve a ejecutar el pipeline, **hay que reexportar el recorte**: si no, el selector
de la app publicada ofrecería clientes cuyo historial ya no está en el bundle y se servirían
como si fueran nuevos, sin que nadie lo notara. Lo caza en CI el test
`test_el_recorte_versionado_cubre_a_todos_los_clientes_ofrecidos`, y `python -m src.pipeline all`
lo incluye como último paso.

## Comprobarlo antes de publicar

```bash
streamlit run streamlit_app.py     # http://localhost:8501
```

Ojo con el fallo típico: en local funciona porque hay artefactos que no están en el repo.
Para descartarlo, clonar el repo en otra carpeta y levantar la app ahí, sin ejecutar el
pipeline — que es exactamente lo que hace Community Cloud.

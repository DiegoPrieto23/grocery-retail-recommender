# README técnico

Cómo está construido el proyecto: arquitectura, pipeline, decisiones de ingeniería,
verificación y deudas abiertas. Si lo que buscas es **qué hace** el sistema, empieza por el
[`README.md`](README.md); si lo que quieres es **arrancarlo desde cero**, por
[`GETTING_STARTED.md`](GETTING_STARTED.md).

**Todas las cifras de este documento salen de artefactos que se regeneran**
(`reports/*/metrics.json`), nunca copiadas a mano de un notebook.

---

## Stack y por qué

| Pieza | Elección | Motivo |
| --- | --- | --- |
| Generación, ETL y features | **PySpark** | El volumen (4,4 M de líneas) da para ello, y el proyecto hermano ya usaba dbt: el objetivo era cubrir el otro stack |
| Candidatos colaborativos | **ALS** (Spark MLlib) | Entrena dentro del mismo motor, sin sacar la matriz de interacciones a memoria |
| Ranking | **LightGBM** `LambdaRank` | El patrón real de un recomendador de retail: corre en CPU, ordena listas y acepta features heterogéneas |
| Propensión | **LightGBM** binario | Mismo motor, tabular, con probabilidades calibrables |
| Registro de experimentos | **MLflow** (opcional) | `src/tracking.py` lo detecta; sin él, no-op |
| Demo | **Streamlit** | Solo inferencia, sin Spark y sin red |
| Imágenes del catálogo | **API de Pexels**, una sola vez | Única dependencia de internet del repo; el resultado se cachea y se versiona |

Sin *deep learning* a propósito: en este tamaño de catálogo y con estas features, un
gradient boosting sobre candidatos bien construidos es lo que se pondría en producción.

---

## El pipeline

`src/pipeline.py` es el **único punto de entrada**. Las dependencias entre pasos son datos,
no prosa: el orden se deriva de ellas y hay [tests](tests/test_pipeline.py) que comprueban
que el grafo es coherente.

```bash
python -m src.pipeline all                      # ~75 min: la cadena entera, en orden
python -m src.pipeline list                     # los 14 pasos, qué produce cada uno y qué tarda
python -m src.pipeline all --dry-run            # el plan, sin ejecutar nada
python -m src.pipeline nba                      # un paso y todo lo que necesita (generate → etl → nba)
python -m src.pipeline nba --only               # solo ese paso, dando por buenas sus entradas
python -m src.pipeline all --from recommender   # retomar desde la mitad
python -m src.pipeline all --scale 0.02         # la cadena en pequeño, para probarla de punta a punta
```

| Paso | Qué hace | Depende de | ~min |
| --- | --- | --- | ---: |
| `generate` | Los 7 CSV sintéticos, con la semilla fija | — | 3 |
| `verify-dataset` | Comprueba los patrones inyectados | `generate` | 1 |
| `etl` | Limpieza, Data Trust, RFM, recompra, afinidades | `generate` | 6 |
| `recommender` | Entrena y evalúa el sistema de dos etapas | `etl` | 33 |
| `demo-profiles` | Un caso legible de cada uno de los 4 perfiles | `recommender` | 1 |
| `export-bundle` | Vuelca las fuentes para servir sin Spark | `recommender` | 2 |
| `export-oracle` | Probabilidades reales de las cestas de test | `generate` | 15 |
| `verify-recommender` | Baselines, techo teórico e intervalos | `recommender`, `export-oracle` | 6 |
| `nba` | Propensión y política de valor esperado | `etl` | 7 |
| `verify-category-need` | La capa común recomendador/NBA | `nba` | 1 |
| `impact` | Traduce las métricas de los dos modelos a euros | `recommender`, `nba` | 0,1 |
| `findings` | Informe de hallazgos de negocio y sus figuras | `etl` | 1 |
| `assets` | Mapeo producto → foto (offline: no llama a Pexels) | `etl` | 0,2 |
| `export-demo-bundle` | Recorta el bundle a los clientes que la demo ofrece | `export-bundle`, `nba` | 0,5 |

**No es un motor de construcción.** No mira fechas de ficheros ni decide qué está al día.
Relanzar un paso lo repite, y con la semilla fija eso da lo mismo: solo cuesta tiempo. La
alternativa —saltarse pasos porque su salida parece reciente— es justo lo que deja que un
artefacto viejo sobreviva a un cambio de código sin que nadie se entere.

---

## Estructura del repositorio

```
grocery-retail-recommender/
├── data_generation/          # el dataset no se descarga, se genera
│   ├── catalog.py            #   parametrización de dominio: 62 categorías, lealtad, afinidad, estacionalidad
│   ├── generate_dataset.py   #   las 7 tablas + los defectos de calidad deliberados
│   ├── export_oracle.py      #   probabilidades reales de las cestas de test (fuera de data/raw)
│   └── verify_dataset.py     #   recalcula desde los CSV cada patrón que dice haber inyectado
├── src/
│   ├── pipeline.py           # orquestador: `python -m src.pipeline all`
│   ├── etl/                  # PySpark
│   │   ├── session.py        #   fábrica de la SparkSession (UTC, Arrow)
│   │   ├── schemas.py        #   esquemas explícitos, lectura y escritura
│   │   ├── cleaning.py       #   limpieza de las 7 tablas + traza de lo corregido
│   │   ├── data_trust.py     #   Data Trust Score
│   │   ├── rfm.py            #   RFM y segmento por cliente
│   │   ├── repurchase.py     #   due_for_repurchase
│   │   ├── affinity.py       #   co-ocurrencia y FP-Growth
│   │   └── run_etl.py        #   orquestador
│   ├── eda/                  # las 9 preguntas de negocio, como funciones
│   │   ├── questions.py      #   una función por pregunta, con tests
│   │   └── findings.py       #   el informe de hallazgos
│   ├── recommender/          # dos etapas
│   │   ├── config.py         #   ventanas temporales, tamaños de pool, hiperparámetros
│   │   ├── splits.py         #   split por cesta, prefijo/target y los 4 perfiles
│   │   ├── candidates.py     #   popularidad, co-compra (SKU y categoría), historial, ALS
│   │   ├── history.py        #   historial del cliente al día de cada cesta (as-of)
│   │   ├── formulas.py       #   fórmulas de reposición compartidas por Spark y pandas
│   │   ├── features.py       #   60 features del par (query, candidato) y relevancia graduada
│   │   ├── ranker.py         #   LightGBM LambdaRank
│   │   ├── evaluate.py       #   NDCG@5 graduada (principal), NDCG@5, Recall@5, F1@5 y baselines
│   │   ├── oracle.py         #   oráculo bayesiano: el techo teórico con los pesos del generador
│   │   ├── pipeline.py       #   orquestador
│   │   ├── verify_recommender_diagnostics.py  # baselines + techo teórico, reproducibles
│   │   └── demo_profiles.py  #   un caso legible de cada perfil
│   ├── nba/                  # propensión + política
│   │   ├── config.py         #   cortes, catálogo de acciones y TODOS los supuestos
│   │   ├── targets.py        #   las dos etiquetas, construidas por corte
│   │   ├── features.py       #   features de cliente y de cliente × categoría
│   │   ├── propensity.py     #   los dos LightGBM binarios y sus métricas
│   │   ├── policy.py         #   valor esperado, baselines y sensibilidad
│   │   ├── pipeline.py       #   orquestador
│   │   └── verify_category_need.py  # la capa común con el recomendador
│   ├── impact/               # de NDCG y AUC a euros
│   │   ├── config.py         #   los supuestos económicos, todos juntos
│   │   ├── model.py          #   la aritmética, en funciones puras
│   │   └── pipeline.py       #   mide, aplica y escribe IMPACT.md
│   ├── tracking.py           # MLflow opcional; sin él, no-op
│   ├── catalog/              # grupos visuales y fotos de Pexels (se ejecuta una vez)
│   ├── serving/              # inferencia del recomendador sin Spark
│   │   ├── export_bundle.py  #   vuelca las fuentes de candidatos ya ajustadas a data/serving/
│   │   ├── export_demo_bundle.py  # recorta ese bundle a los clientes que la demo ofrece
│   │   └── recommend.py      #   el mismo top-5 que el pipeline, con pandas
│   └── demo/                 # lo que la app pinta
│       ├── catalog.py        #   fichas de producto, buscador y el motivo de cada recomendación
│       ├── baskets.py        #   cestas reales de test y el contraste con lo que compró
│       └── customers.py      #   a quién ofrece el selector, y su escenario (fiel / ocasional / en riesgo)
├── streamlit_app.py          # la demo (solo dibuja; la lógica está en src/)
├── assets/                   # 60 fotos + el mapeo producto → foto (versionados)
├── notebooks/01_eda.ipynb    # reconocimiento de tablas + calidad + 9 preguntas de negocio
├── tests/                    # 510 tests
├── reports/etl/              # informes que genera run_etl (versionados)
├── reports/recommender/      # métricas, demo de los 4 perfiles, baselines/techo y referencias congeladas
├── reports/nba/              # métricas, barridos de sensibilidad y referencias congeladas
├── reports/impact/           # el detalle numérico de IMPACT.md
├── reports/insights/         # informe de hallazgos de negocio + sus 8 figuras
├── docs/HISTORIA.md          # la narrativa de cómo se construyó, paso a paso
├── docs/DESPLIEGUE.md        # cómo se publica la demo
├── docs/como-funcionan-recomendador-y-nba.md  # los dos modelos en lenguaje llano
├── docs/diagnostico-fase7.md # los puntos de mejora, con su estado
├── docs/CLEANING.md          # el porqué de cada decisión de limpieza
├── docs/VISUAL_CATALOG.md    # cómo se agruparon los productos y se eligieron las fotos
├── docs/prompts-de-construccion.md  # los prompts con los que se construyó el repo
├── docs/build_docx.py        # genera el .docx de negocio desde el Markdown
├── data/{raw,processed,serving}/  # generados; solo se versiona lo que la demo necesita en producción
├── IMPACT.md                 # el impacto de negocio estimado, en euros
├── DATA_SPEC.md              # esquema columna a columna, crudo y procesado
├── CHALLENGE.md              # el enunciado de partida
├── ROADMAP.md                # el plan de construcción y cómo se verificó cada punto
└── CLAUDE.md                 # convenciones del proyecto
```

---

## Decisiones que sostienen los resultados

### Splits: temporales y por cesta

| Modelo | Corte | Por qué así |
| --- | --- | --- |
| Recomendador | Fuentes de candidatos hasta el **2025-09-01**; ranker con cestas de sep-oct; test desde el **2025-11-01** | Cortar por `basket_id` evita que dos productos de la misma compra caigan a los dos lados (fuga trivial). Cortar además en el tiempo evita que el ranker aprenda de candidatos calculados con datos posteriores a la cesta que predice |
| NBA | Cuatro cortes de entrenamiento (abr → ago 2025), validación el **2025-09-15**, test el **2025-11-01** | El target es «qué pasa después de esta fecha»: un split aleatorio lo haría trivial |

Las fuentes de candidatos se **recalculan por ventana**, no una vez para todo el periodo.

### Features *as-of*

El historial del cliente se corta el día de cada cesta, no al inicio de la ventana. Está
implementado una sola vez (`src/recommender/history.py`) y se comprueba que Spark y pandas
dan lo mismo (`tests/test_asof_features.py`): la demo sirve con pandas y el pipeline entrena
con Spark, así que una divergencia ahí sería un sistema distinto en producción.

### Paridad entre entrenamiento y servicio

`src/serving/recommend.py` reproduce el top-5 del pipeline **sin Spark**, y
`tests/test_serving_parity.py` comprueba que salen los mismos productos, en el mismo orden y
con los mismos *scores*. Las fórmulas que las dos rutas comparten viven en un solo sitio
(`src/recommender/formulas.py`), no duplicadas.

### Lo que se despliega es un recorte, no otra cosa

El bundle de serving son 826 MB en memoria y el 99 % es historial de clientes que el
selector de la demo nunca ofrece. `src/serving/export_demo_bundle.py` lo recorta a los que
sí ofrece (78 clientes, 21 MB), y `resolve_table` prefiere la tabla completa cuando está: en
local manda el bundle entero, en el despliegue solo existe el recorte y la app arranca con
él sin ninguna configuración. Con solo los ficheros versionados, el top-5 sale **idéntico**
al del bundle completo. El detalle, en [`docs/DESPLIEGUE.md`](docs/DESPLIEGUE.md).

### Verificación en vez de afirmación

- `verify_dataset.py` recalcula desde los CSV cada patrón que el generador dice haber
  inyectado (afinidades, estacionalidad, fidelidad, embudo).
- `verify_recommender_diagnostics.py` recalcula baselines, techo teórico e intervalos.
- `verify_category_need.py` recalcula el efecto de la capa común sobre la política.
- `src/impact/pipeline.py` **escribe** `IMPACT.md`: las cifras de negocio no se editan a mano.

### Reproducibilidad

Semilla fija (`SEED = 42`) propagada a numpy, `random` y Faker, con sub-*streams* por etapa.
Dos ejecuciones del generador dan ficheros **byte a byte idénticos**, y un test lo comprueba
comparando los `sha256` del manifiesto. `constraints.txt` fija las versiones.

Dos excepciones declaradas: las fotos de `assets/` (salen de una búsqueda en Pexels, que
puede cambiar con el tiempo — se descargaron una vez y se versionan) y la parada temprana del
ranker, que no es determinista entre ejecuciones (ver **Deudas abiertas**).

---

## Tests

**510 tests** (`pytest`), verdes en CI sobre Ubuntu con Python 3.11 y JVM 17. Los que
necesitan artefactos que no se versionan se saltan solos en un repo recién clonado.

| Fichero | Tests | Qué fija |
| --- | ---: | --- |
| `test_generate_dataset.py` | 44 | Reproducibilidad, volúmenes, patrones del generador, fidelidad de marca, embudo online y estructura de la cesta (tamaño, misiones, sustitución, marca blanca) |
| `test_cleaning.py` | 23 | Cada regla de limpieza con un caso mínimo comprobable a mano |
| `test_data_trust.py` | 26 | Un defecto → la dimensión que le toca, sobre un dataset impecable |
| `test_eda_questions.py` | 39 | Las 9 preguntas de negocio, con las respuestas calculadas a mano |
| `test_repurchase.py` | 19 | Cadencias de 7 y 4 días, efecto del hogar, tolerancia |
| `test_affinity.py` | 13 | Soporte, confianza y lift calculados a mano sobre 10 cestas |
| `test_rfm.py` | 13 | Quintiles, segmentos y clientes sin compras |
| `test_schemas.py` | 16 | Tipos al leer, ida y vuelta a Parquet por los dos motores |
| `test_recommender.py` | 17 | Que no hay fuga: ni entre ventanas, ni del target al pool, ni de la sesión pasado el corte; y Precision/F1 a mano |
| `test_evaluation_robustness.py` | 19 | Varios cortes por cesta sin solapes, muestra por cesta, cold-start y el bootstrap contra casos conocidos |
| `test_cart_rerank.py` | 11 | Re-ranking por categoría y carrito, y features de carrito iguales en Spark y en pandas |
| `test_nba.py` | 25 | Que las features no miran tras el corte, y la aritmética del valor esperado a mano |
| `test_impact.py` | 18 | La cadena de multiplicaciones que lleva de `hit_rate@5` a euros |
| `test_tracking.py` | 19 | Qué se registra en MLflow, y que ni su ausencia ni sus fallos estorban |
| `test_catalog.py` | 30 | Grupos visuales, puntuación de fotos, huella perceptual y correspondencia con el catálogo actual |
| `test_serving_parity.py` | 6 | Que la inferencia sin Spark da el mismo top-5 que el pipeline |
| `test_demo_baskets.py` | 11 | Que la cesta real que siembra la demo es el carrito del corte, y lo que compara, el target del split |
| `test_demo_hits.py` | 12 | El acierto en dos niveles, sin perder nunca el número exacto |
| `test_demo_bundle.py` | 12 | Que el bundle recortado que se despliega ofrece los mismos clientes que el completo, y que en local manda la tabla entera |
| `test_recommender_diagnostics.py` | 27 | Baselines independientes del pool y el oráculo, contra una simulación por fuerza bruta del generador |
| `test_asof_features.py` | 8 | Historial as-of el día de cada cesta, igual en Spark y en pandas |
| `test_ranker_objective.py` | 11 | Relevancia graduada, `label_gain` y el ranking personal de categorías |

Dos criterios que se repiten en toda la suite:

- **Los tests de calidad no comprueban «el score bajó».** Parten de un dataset diminuto y
  perfecto que puntúa 100 e introducen **un solo** defecto, verificando que lo detecta la
  comprobación concreta que le corresponde.
- **Las funciones de negocio se ejercitan sobre datasets construidos para que cada cifra se
  pueda recalcular de cabeza**: cadencias de 7 y 4 días exactos, cinco cestas con importes
  redondos. Si un test falla, el número esperado se comprueba a mano en un minuto.

CI: [`ci.yml`](.github/workflows/ci.yml) (tests en cada push). La cadena entera a escala 0,02
se puede lanzar a mano con `python -m src.pipeline all --scale 0.02 --fast`.

---

## Notas de entorno

**PySpark 3.5 no soporta oficialmente Python 3.13**: el worker de Python casca al arrancar en
Windows. Por eso todo el ETL está escrito para ejecutarse **íntegramente en la JVM** —
DataFrame API, Spark SQL y MLlib, sin UDFs de Python, sin `.rdd` y sin `createDataFrame`
sobre listas, que son las tres cosas que levantan un worker. Para construir DataFrames
pequeños (tests, notebooks) se usa pandas + Arrow, que viaja por la JVM. La CI usa Python
3.11, donde nada de esto aplica.

**Escritura de Parquet en Windows**: si falta `hadoop.dll` en `%HADOOP_HOME%\bin`,
`df.write.parquet()` falla con `UnsatisfiedLinkError: NativeIO$Windows.access0`
(`winutils.exe` por sí solo no basta). El ETL lo detecta con un sondeo único y cae a escribir
con pyarrow desde el driver; a esta escala es incluso más rápido. Poniendo el `hadoop.dll`
que corresponda a tu `winutils.exe` vuelve a usarse la escritura nativa sin tocar nada.

**Zona horaria**: la sesión de Spark trabaja en UTC para que el calendario sea continuo y los
`datediff` no dependan del cambio de hora. Al traer un `timestamp` al driver con `collect()`,
PySpark lo convierte a la zona del sistema, así que el `datetime` de Python sale desplazado
respecto al CSV; el valor almacenado y todo el cálculo en SQL son correctos. Conviene extraer
la fecha o la hora **dentro** de Spark.

---

## Deudas abiertas

Ninguna bloqueante, todas medidas:

- **El «modelo de churn» es en realidad un modelo de inactividad a 4 semanas.** Un target
  honesto pediría una ventana más larga o condicionar por la cadencia de cada cliente.
- **El modelo de propensión no tiene features de calendario.** Por eso la política llega a
  elegir protector solar en noviembre. El índice estacional ya se calcula en el ETL: es
  meterlo en `src/nba/features.py`.
- **El entrenamiento del ranker no es determinista entre ejecuciones.** Con los mismos datos,
  la parada temprana cae en un número de árboles distinto (70 frente a 150) y la cabecera se
  mueve en la tercera decimal. Los intervalos de confianza lo cubren, pero una serie histórica
  limpia pediría fijar el orden de las filas y `deterministic=True` en LightGBM.
- **Ampliar el pool de candidatos no mejora lo que el sistema entrega.** Los topes se
  reajustaron al catálogo de 496 productos y el `pool_recall` pasó del 82,5 % al 91,8 % (y del
  55,3 % al 77,4 % en cliente nuevo con carrito vacío), pero la métrica principal se movió
  +0,03 pp. **El sistema no está limitado por la primera etapa**, sino por lo distinguibles
  que son entre sí las referencias de una misma categoría. Quien quiera mover la cifra tiene
  que mirar la segunda etapa.
- **El oráculo es caro y con menos muestra efectiva que antes.** Con cestas de hasta 50
  categorías, el muestreo por importancia condicionado al carrito pierde eficiencia: el tamaño
  efectivo mediano es de 151 muestras de 500, y en el 5 % peor baja a 11. El techo agregado
  aguanta el contraste contra lo realizado (máx |z| = 2,26), pero una query concreta puede
  tener bastante ruido; subir las muestras o usar remuestreo sistemático es la salida si
  alguna vez hace falta el techo *por query*.

**Fuera de alcance por decisión, no por abandono:** un dashboard en **Power BI** estuvo
planteado como pieza de BI Engineering y se aparcó a favor de la demo. Está declarado como
tal en el [`ROADMAP.md`](ROADMAP.md), donde también está el modelo dimensional que usaría.

---

## Para saber cómo se llegó hasta aquí

- [`docs/HISTORIA.md`](docs/HISTORIA.md) — la narrativa de construcción, con lo que se
  aprendió en cada paso y lo que hubo que rehacer.
- [`docs/diagnostico-fase7.md`](docs/diagnostico-fase7.md) — la auditoría del sistema y el
  estado de cada punto de mejora.
- [`ROADMAP.md`](ROADMAP.md) — el plan y, sobre todo, **cómo se verifica** cada punto.
- [`CHALLENGE.md`](CHALLENGE.md) — el enunciado del que salió todo.
- [`docs/prompts-de-construccion.md`](docs/prompts-de-construccion.md) — los prompts con los
  que se construyó el repo.

# Grocery Retail — Recommender & Next Best Action

Recomendador de cesta y capa de *Next Best Action* sobre un supermercado online simulado.
El proyecto empieza donde suelen empezar los problemas reales — **no hay dataset**: se
genera, con sus ciclos de reposición, su estacionalidad, su señal de abandono y sus
defectos de calidad deliberados — y sigue con el ETL, el modelado y la comunicación de
resultados, hasta una demo web que enseña el sistema funcionando.

Es la **fase 2 del ciclo de vida del cliente en retail** que empezó en
[`sports-rental-analytics`](https://github.com/DiegoPrieto23/sports-rental-analytics), con
un stack distinto a propósito: allí dbt sobre Databricks, aquí **PySpark** de punta a
punta.

> Todos los datos son **100 % sintéticos**. No contienen información real de ningún
> cliente, producto, tienda ni retailer.

---

## Estado actual

| Fase | Qué es | Estado |
| --- | --- | :---: |
| 0 · Setup | Estructura, entorno reproducible, CI | ✅ |
| 1 · Generador de datos | 7 tablas con reglas de negocio y defectos inyectados | ✅ |
| 2 · ETL y features | Limpieza, Data Trust Score, RFM, recompra, afinidad, EDA | ✅ |
| 3 · Recomendador de cesta | Candidatos (popularidad + co-compra + ALS) → ranker LightGBM | ✅ |
| 4 · Next Best Action | Modelo de propensión + política de valor esperado | ✅ |
| 5 · Empaquetado y storytelling | Impacto en €, informe de hallazgos, MLflow | ✅ |
| 6 · Demo web | Catálogo con fotos reales + Streamlit con la cesta en vivo | ✅ |
| 7 · Fidelidad de producto | Fidelidad de marca en el generador y todo lo anterior rehecho encima | ✅ |
| 8 · Estructura de la cesta | Misiones de compra, cola larga de tamaño y sustitución en el generador, con todo rehecho encima | ✅ |
| 9 · Cierre | Capa común recomendador/NBA, orquestador único, smoke E2E en CI y poda de documentación | ✅ |

El plan completo, con cómo se verifica cada punto, está en [`ROADMAP.md`](ROADMAP.md); el
enunciado del reto en [`CHALLENGE.md`](CHALLENGE.md); cómo se llegó hasta aquí, fase por
fase, en [`docs/HISTORIA.md`](docs/HISTORIA.md); y los puntos de mejora pendientes, con su
estado, en [`docs/diagnostico-fase7.md`](docs/diagnostico-fase7.md). **Todas las cifras de este README son las del dataset de la
Fase 8**; donde se comparan con las anteriores, se dice. El estado completo de antes de
esa regeneración está congelado en [`snapshots/pre-fase-8/`](snapshots/pre-fase-8/).

**Lo que ya se puede enseñar:** 4,4 M de líneas de ticket generadas de forma reproducible,
un ETL en PySpark que las limpia y documenta cada corrección, un Data Trust Score que pasa
de **91,38 (C) a 100,00 (A)**, cuatro tablas de features listas para modelar, un
[notebook de EDA](notebooks/01_eda.ipynb) con 9 preguntas de negocio resueltas en Spark SQL,
un recomendador de cesta de dos etapas que acierta la categoría en el **79,5 %** de las
cestas y el SKU exacto en el **61,6 %** (NDCG@5 graduada = 0,3015 [0,2981, 0,3052]), una política de Next Best
Action que decide en euros, dos documentos que traducen todo eso a negocio — el
[resumen de impacto](IMPACT.md) y el
[informe de hallazgos](reports/insights/business_findings.md) — y una
[demo en Streamlit](docs/HISTORIA.md#fase-6--demo-web) donde se ve todo funcionando sobre cestas reales.

---

## Cómo reproducirlo

Requiere **Python 3.10+** y una **JVM 17** (PySpark). Con Python 3.13 en Windows hay una
pega conocida: ver [Notas de entorno](#notas-de-entorno).

```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt

python -m src.pipeline all         # ~75 min: la cadena entera, en orden
streamlit run streamlit_app.py     # la demo, en http://localhost:8501
pytest                             # 489 tests
```

`src/pipeline.py` es el **único punto de entrada** (punto B1 del
[diagnóstico](docs/diagnostico-fase7.md)). Antes esto eran once comandos cuyo orden solo
estaba escrito aquí, en prosa; ahora las dependencias entre pasos son datos, el orden se
deriva de ellas y hay [tests](tests/test_pipeline.py) que comprueban que el grafo es
coherente. Lo que ofrece:

```bash
python -m src.pipeline list                     # los 13 pasos, qué produce cada uno y qué tarda
python -m src.pipeline all --dry-run            # el plan, sin ejecutar nada
python -m src.pipeline nba                      # un paso y todo lo que necesita (generate → etl → nba)
python -m src.pipeline nba --only               # solo ese paso, dando por buenas sus entradas
python -m src.pipeline all --from recommender   # retomar desde la mitad
python -m src.pipeline all --scale 0.02         # la cadena en pequeño (lo que corre el smoke de CI)
```

Los pasos, en orden, con lo que tarda cada uno a escala 1:

| Paso | Qué hace | Depende de | ~min |
| --- | --- | --- | ---: |
| `generate` | Los 7 CSV sintéticos, con la semilla fija | — | 3 |
| `verify-dataset` | Comprueba los patrones inyectados | `generate` | 1 |
| `etl` | Limpieza, Data Trust, RFM, recompra, afinidades (Tareas 1-2) | `generate` | 6 |
| `recommender` | Entrena y evalúa el sistema de dos etapas (Tarea 3a) | `etl` | 33 |
| `demo-profiles` | Un caso legible de cada uno de los 4 perfiles | `recommender` | 1 |
| `export-bundle` | Vuelca las fuentes para servir sin Spark | `recommender` | 2 |
| `export-oracle` | Probabilidades reales de las cestas de test | `generate` | 15 |
| `verify-recommender` | Baselines, techo teórico e intervalos | `recommender`, `export-oracle` | 6 |
| `nba` | Propensión y política de valor esperado (Tarea 3b) | `etl` | 7 |
| `verify-category-need` | La capa común de necesidad de categoría (punto M7) | `nba` | 1 |
| `impact` | Traduce las métricas de las dos tareas a euros | `recommender`, `nba` | 0,1 |
| `findings` | Informe de hallazgos de negocio y sus figuras | `etl` | 1 |
| `assets` | Mapeo producto → foto (offline: no llama a Pexels) | `etl` | 0,2 |

No es un motor de construcción: no mira fechas de ficheros ni decide qué está al día.
Relanzar un paso lo repite, y con la semilla fija eso da lo mismo — solo cuesta tiempo. La
alternativa, saltarse pasos porque su salida parece reciente, es justo lo que deja un
artefacto viejo sobrevivir a un cambio de código sin que nadie se entere.

El dataset **no se versiona** (`data/` está en `.gitignore`): se regenera con la semilla
fija y sale byte a byte idéntico. Lo que sí está en el repo son los informes de
`reports/`, el notebook ya ejecutado con sus figuras, los dos documentos de negocio
([`IMPACT.md`](IMPACT.md) y
[`reports/insights/business_findings.md`](reports/insights/business_findings.md)) y las
fotos del catálogo en `assets/`, que se descargaron una sola vez de Pexels y no se
regeneran (ver [Fase 6](docs/HISTORIA.md#fase-6--demo-web)).

Para abrir el notebook hace falta además `pip install jupyterlab`. Y con
`pip install mlflow`, las Fases 3 y 4 registran además cada ejecución como un experimento
— ver [Registro de experimentos](docs/HISTORIA.md#registro-de-experimentos-mlflow).

---

## Estructura del repositorio

```
grocery-retail-recommender/
├── data_generation/          # FASE 1 — el dataset no se descarga, se genera
│   ├── catalog.py            #   parametrización de dominio: 62 categorías, lealtad, afinidad, estacionalidad
│   ├── generate_dataset.py   #   las 7 tablas + los defectos de calidad deliberados
│   ├── export_oracle.py      #   probabilidades reales de las cestas de test, para el oráculo (fuera de data/raw)
│   └── verify_dataset.py     #   recalcula desde los CSV cada patrón que dice haber inyectado
├── src/
│   ├── pipeline.py           # FASE 9 — orquestador: `python -m src.pipeline all`
│   ├── etl/                  # FASE 2 — PySpark
│   │   ├── session.py        #   fábrica de la SparkSession (UTC, Arrow)
│   │   ├── schemas.py        #   esquemas explícitos, lectura y escritura
│   │   ├── cleaning.py       #   limpieza de las 7 tablas + traza de lo corregido
│   │   ├── data_trust.py     #   Data Trust Score (Tarea 1)
│   │   ├── rfm.py            #   RFM y segmento por cliente
│   │   ├── repurchase.py     #   due_for_repurchase (Tarea 2)
│   │   ├── affinity.py       #   co-ocurrencia y FP-Growth
│   │   └── run_etl.py        #   orquestador
│   ├── eda/                  # FASE 2/5 — las 9 preguntas de negocio, como funciones
│   │   ├── questions.py      #   Tarea 1: una función por pregunta, con tests
│   │   └── findings.py       #   el informe de hallazgos de la Fase 5
│   ├── recommender/          # FASE 3 — dos etapas
│   │   ├── config.py         #   ventanas temporales, tamaños de pool, hiperparámetros
│   │   ├── splits.py         #   split por cesta, prefijo/target y los 4 perfiles
│   │   ├── candidates.py     #   popularidad, co-compra (SKU y categoría), historial, ALS
│   │   ├── history.py        #   historial del cliente al día de cada cesta (as-of, punto A1)
│   │   ├── formulas.py       #   fórmulas de reposición compartidas por Spark y pandas (punto B2)
│   │   ├── features.py       #   60 features del par (query, candidato) y relevancia graduada
│   │   ├── ranker.py         #   LightGBM LambdaRank
│   │   ├── evaluate.py       #   NDCG@5 graduada (principal), NDCG@5, Recall@5, F1@5, SKU / categoría y baselines
│   │   ├── oracle.py         #   oráculo bayesiano: el techo teórico con los pesos reales del generador
│   │   ├── pipeline.py       #   orquestador
│   │   ├── verify_recommender_diagnostics.py  # baselines + techo teórico, reproducibles (diagnóstico A3/A6/M8)
│   │   └── demo_profiles.py  #   un caso legible de cada perfil
│   ├── nba/                  # FASE 4 — propensión + política
│   │   ├── config.py         #   cortes, catálogo de acciones y TODOS los supuestos
│   │   ├── targets.py        #   las dos etiquetas, construidas por corte
│   │   ├── features.py       #   features de cliente y de cliente × categoría
│   │   ├── propensity.py     #   los dos LightGBM binarios y sus métricas
│   │   ├── policy.py         #   valor esperado, baselines y sensibilidad
│   │   ├── pipeline.py       #   orquestador
│   │   └── verify_category_need.py  # la capa común con el recomendador (punto M7)
│   ├── impact/               # FASE 5 — de NDCG y AUC a euros
│   │   ├── config.py         #   los supuestos económicos, todos juntos
│   │   ├── model.py          #   la aritmética, en funciones puras
│   │   └── pipeline.py       #   mide, aplica y escribe IMPACT.md
│   ├── tracking.py           # FASE 5 — MLflow opcional; sin él, no-op
│   ├── catalog/              # FASE 6a — grupos visuales y fotos de Pexels (se ejecuta una vez)
│   ├── serving/              # FASE 6b — inferencia del recomendador sin Spark
│   │   ├── export_bundle.py  #   vuelca las fuentes de candidatos ya ajustadas a data/serving/
│   │   └── recommend.py      #   el mismo top-5 que el pipeline, con pandas
│   └── demo/                 # FASE 6b — lo que la app pinta
│       ├── catalog.py       #   fichas de producto, buscador y el motivo de cada recomendación
│       ├── baskets.py       #   cestas reales de test y el contraste con lo que compró
│       └── customers.py     #   a quién ofrece el selector, y su escenario (fiel / ocasional / en riesgo)
├── streamlit_app.py          # FASE 6b — la demo (solo dibuja; la lógica está en src/)
├── assets/                   # FASE 6a — 60 fotos + el mapeo producto → foto (versionados)
├── notebooks/01_eda.ipynb    # reconocimiento de tablas + calidad + 9 preguntas de negocio
├── tests/                    # 489 tests
├── reports/etl/              # informes que genera run_etl (versionados)
├── reports/recommender/      # métricas de la Fase 3, demo de los 4 perfiles, baselines/techo y las referencias congeladas
├── reports/nba/              # métricas de la Fase 4, barridos de sensibilidad y la referencia pre-Fase 7
├── reports/impact/           # el detalle numérico de IMPACT.md
├── reports/insights/         # informe de hallazgos de negocio + sus 8 figuras
├── docs/HISTORIA.md          # la narrativa fase por fase (lo que era la mitad del README)
├── docs/como-funcionan-recomendador-y-nba.md  # los dos modelos en lenguaje llano
├── docs/diagnostico-fase7.md # los puntos de mejora, con su estado
├── docs/CLEANING.md          # el porqué de cada decisión de limpieza
├── docs/VISUAL_CATALOG.md    # cómo se agruparon los productos y se eligieron las fotos
├── docs/prompts-de-construccion.md  # los prompts con los que se construyó el repo
├── docs/build_docx.py        # genera el .docx de negocio desde el Markdown
├── data/{raw,processed,serving}/  # generados, no versionados
├── IMPACT.md                 # el impacto de negocio estimado, en euros
├── DATA_SPEC.md              # esquema columna a columna, crudo y procesado
├── CHALLENGE.md              # el reto
├── ROADMAP.md                # plan por fases
└── CLAUDE.md                 # convenciones del proyecto
```

---

## Cómo funciona el sistema

Dos modelos que responden a dos preguntas distintas, y una capa que comparten.

### El recomendador de cesta (Tarea 3a)

**Pregunta:** qué productos añadirá el cliente a *esta* cesta. **Grano:** una query es una
cesta cortada en un instante. **Horizonte:** el resto de la cesta en curso.

Dos etapas, que es el patrón habitual en retail y corre en CPU:

1. **Candidatos** — cinco fuentes que dejan unos 234 productos de los 496 del catálogo:
   popularidad estacional (global y con un líder por categoría), co-compra de SKU,
   co-compra de categoría, historial personal con el ciclo de reposición, y ALS de Spark
   MLlib. Cubren el **100 %** de las categorías del objetivo en los cuatro perfiles, así
   que cualquier fallo de categoría es atribuible al ranker y no al pool.
2. **Ranking** — LightGBM `LambdaRank` sobre 60 features en cinco familias (fuente,
   cliente × producto, cliente × categoría, producto, contexto/sesión), con **relevancia
   graduada**: 2 si acierta el SKU exacto, 1 si acierta la categoría. Después, un
   re-ranking que deja como mucho una referencia por categoría y excluye lo que ya está en
   el carrito.

Las features del cliente son **as-of la fecha de la cesta**, no del inicio de la ventana:
una query del 20 de diciembre ve lo que ese cliente compró el 5, el 12 y el 18.

| Métrica (18.000 cestas de test) | Valor |
| --- | ---: |
| **NDCG@5 graduada** (métrica principal) | **0,3015** [0,2981, 0,3052] |
| Acierto de categoría @5 | 79,5 % |
| Acierto de SKU exacto @5 | 61,6 % |
| Del techo teórico de categoría (oráculo) | 94,9 % |
| Sobre el mejor baseline (reglas de asociación) | +3,6 pp [+3,0, +4,2] |

### El Next Best Action (Tarea 3b)

**Pregunta:** qué acción de negocio tomar con *este cliente*. **Grano:** cliente, y
cliente × categoría. **Horizonte:** 7 días para la compra, 28 para el abandono.

Dos modelos de propensión (`AUC 0,8532` de churn y `0,7591` de compra en categoría) y una
política que elige la acción de mayor **valor incremental sobre no hacer nada**:

    V(a)       = P(compra | a) × (margen − descuento(a)) − coste_envío(a)
    delta(a)   = V(a) − P(compra) × margen  +  retención(a)

Tres acciones: `ninguna_accion`, `recomendar_categoria` y `enviar_cupon_categoria`. Los
*uplifts* son **supuestos declarados**, no estimaciones — este dataset no permite
identificar el efecto causal de una acción —, y por eso el informe los acompaña de barridos
de sensibilidad que dicen a partir de qué supuesto la política deja de ganar.

| | Valor incremental |
| --- | ---: |
| No actuar nunca | 0 € |
| Cupón a todo el mundo | −5.032 € |
| Recomendar categoría a todo el mundo | +670 € |
| **Política de valor esperado** | **+3.078 €** |

### La capa que comparten: necesidad de categoría

`P(compra en categoría a 7 días)` del NBA y la "necesidad de categoría" del recomendador
son casi la misma cantidad con horizontes distintos. Desde el punto M7 comparten fórmula:
`recommender.formulas.category_need_weight`, sobre el `overdue_ratio` de la Tarea 2.

En el NBA entra ponderando la **retención**, y no es un adorno. El cupón cuesta 2,54 € y
persigue un margen de ~1,4 €, así que su término de cross-sell es negativo y *decreciente*
en la probabilidad de compra; como la retención no dependía de la categoría, el `argmax`
acababa eligiendo la categoría que **minimiza la fuga de descuento**, es decir, la que el
cliente casi seguro no iba a comprar. Medido:

| Categoría elegida para el cupón | Antes | Ahora |
| --- | ---: | ---: |
| Que el recomendador da por no vencida | 88,5 % | **23,2 %** |
| Recién repuesta | 73,3 % | **2,0 %** |
| Tasa real de compra a 7 días | 1,5 % | **4,3 %** |

Lo recalcula `python -m src.nba.verify_category_need`. El valor total de la política baja
de 3.842 € a 3.078 € justamente por esto: esos 764 € eran retención apuntada a ofertas que
no retienen a nadie.

### La demo

[Streamlit](streamlit_app.py), solo inferencia: carga los modelos ya entrenados y el bundle
de `data/serving/`, sin Spark y sin red. Cesta en vivo, top-5 en tarjetas con foto real del
`visual_group`, el motivo de cada recomendación sacado de las features con las que el ranker
ordenó, y — sobre cestas reales de test — si el cliente acabó comprando eso. El banner del
NBA dice **con qué fecha de corte** se decidió la acción, porque se resuelve una vez (1 de
noviembre) y se enseña junto a cestas posteriores.

La historia de cómo se llegó hasta aquí, fase por fase y con lo que se aprendió en cada
una, está en [`docs/HISTORIA.md`](docs/HISTORIA.md).

---

## Tests

**489 tests** (`pytest`), verdes en CI sobre Ubuntu con Python 3.11 y JVM 17. Los que
necesitan artefactos que no se versionan (el bundle de serving, las tablas procesadas) se
saltan solos en un repo recién clonado.

| Fichero | Tests | Qué fija |
| --- | ---: | --- |
| `test_generate_dataset.py` | 44 | Reproducibilidad, volúmenes, patrones del generador, fidelidad de marca, embudo online y la estructura de la cesta de la Fase 8 (tamaño, misiones, sustitución, marca blanca) |
| `test_cleaning.py` | 23 | Cada regla de limpieza con un caso mínimo comprobable a mano |
| `test_data_trust.py` | 26 | Un defecto → la dimensión que le toca, sobre un dataset impecable |
| `test_eda_questions.py` | 39 | Las 9 preguntas de negocio de la Tarea 1, con las respuestas calculadas a mano |
| `test_repurchase.py` | 19 | Cadencias de 7 y 4 días, efecto del hogar, tolerancia (Tarea 2) |
| `test_affinity.py` | 13 | Soporte, confianza y lift calculados a mano sobre 10 cestas |
| `test_rfm.py` | 13 | Quintiles, segmentos y clientes sin compras |
| `test_schemas.py` | 16 | Tipos al leer, ida y vuelta a Parquet por los dos motores |
| `test_recommender.py` | 17 | Que no hay fuga: ni entre ventanas, ni del target al pool, ni de la sesión pasado el corte; y Precision/F1 a mano |
| `test_evaluation_robustness.py` | 19 | Varios cortes por cesta sin solapes y con su propio `cut_ts`, muestra por cesta, cold-start, y el bootstrap (por cesta y pareado) contra casos conocidos |
| `test_cart_rerank.py` | 11 | Re-ranking por categoría y carrito, huecos regalados a mano, y features de carrito iguales en Spark y en pandas |
| `test_nba.py` | 25 | Que las features no miran tras el corte, y la aritmética del valor esperado a mano |
| `test_impact.py` | 18 | La cadena de multiplicaciones que lleva de `hit_rate@5` a euros |
| `test_tracking.py` | 19 | Qué se registra en MLflow, y que ni su ausencia ni sus fallos estorban |
| `test_catalog.py` | 30 | Grupos visuales, puntuación de fotos, huella perceptual y que el mapeo corresponde al catálogo actual |
| `test_serving_parity.py` | 6 | Que la inferencia sin Spark da el mismo top-5 que el pipeline, en el mismo orden y con los mismos scores |
| `test_demo_baskets.py` | 11 | Que la cesta real que siembra la demo es el carrito del corte, y lo que compara, el target del split |
| `test_demo_hits.py` | 12 | El acierto en dos niveles, sin perder nunca el número exacto, y la guarda contra un mapeo obsoleto |
| `test_recommender_diagnostics.py` | 27 | Los baselines independientes del pool y el oráculo: el muestreo secuencial por importancia contra una simulación por fuerza bruta del generador, y el contrato con lo que exporta `export_oracle` |
| `test_asof_features.py` | 8 | Historial as-of el día de cada cesta, igual en Spark y en pandas |
| `test_ranker_objective.py` | 11 | Relevancia graduada, `label_gain` y el ranking personal de categorías |

Dos criterios que se repiten en toda la suite:

- **Los tests de calidad no comprueban "el score bajó".** Parten de un dataset diminuto y
  perfecto que puntúa 100 e introducen **un solo** defecto, verificando que lo detecta la
  comprobación concreta que le corresponde.
- **Las funciones de las Tareas 1 y 2 se ejercitan sobre datasets construidos para que cada
  cifra se pueda recalcular de cabeza**: cadencias de 7 y 4 días exactos en el ciclo de
  recompra, cinco cestas con importes redondos en las preguntas de negocio. Si un test
  falla, el número esperado se comprueba a mano en un minuto.

---

## Notas de entorno

**PySpark 3.5 no soporta oficialmente Python 3.13**: el worker de Python casca al arrancar
en Windows. Por eso todo el ETL está escrito para ejecutarse **íntegramente en la JVM** —
DataFrame API, Spark SQL y MLlib, sin UDFs de Python, sin `.rdd` y sin `createDataFrame`
sobre listas, que son las tres cosas que levantan un worker. Para construir DataFrames
pequeños (tests, notebooks) se usa pandas + Arrow, que viaja por la JVM. La CI usa Python
3.11, donde nada de esto aplica.

**Escritura de Parquet en Windows**: si falta `hadoop.dll` en `%HADOOP_HOME%\bin`,
`df.write.parquet()` falla con `UnsatisfiedLinkError: NativeIO$Windows.access0`
(`winutils.exe` por sí solo no basta). El ETL lo detecta con un sondeo único y cae a
escribir con pyarrow desde el driver; a esta escala es incluso más rápido. Poniendo el
`hadoop.dll` que corresponda a tu `winutils.exe` vuelve a usarse la escritura nativa sin
tocar nada.

**Zona horaria**: la sesión de Spark trabaja en UTC para que el calendario sea continuo y
los `datediff` no dependan del cambio de hora. Al traer un `timestamp` al driver con
`collect()`, PySpark lo convierte a la zona del sistema, así que el `datetime` de Python
sale desplazado respecto al CSV; el valor almacenado y todo el cálculo en SQL son
correctos. Conviene extraer la fecha o la hora **dentro** de Spark.

---

## Qué viene ahora

Las ocho fases están cerradas. Un **dashboard en Power BI** estuvo planteado como pieza de
BI Engineering de la Tarea 4 y queda **fuera de alcance por ahora** a favor de la demo —
está declarado como tal en el [`ROADMAP.md`](ROADMAP.md), no es trabajo a medias.

Quedan tres deudas anotadas, ninguna bloqueante:

- **El "modelo de churn" es en realidad un modelo de inactividad a 4 semanas** (Fase 4). Un
  target honesto pediría una ventana más larga o condicionar por la cadencia de cada cliente.
- **El modelo de propensión no tiene features de calendario** (detectado en la Fase 5). Por
  eso la política elige protector solar en noviembre. El índice estacional ya se calcula en
  la Fase 2: es meterlo en `src/nba/features.py`.
- **El entrenamiento del ranker no es determinista entre ejecuciones.** Con los mismos
  datos, la parada temprana cae en un número de árboles distinto (70 frente a 150) y la
  cabecera se mueve en la tercera decimal. Los intervalos de confianza lo cubren, pero una
  serie histórica limpia pediría fijar el orden de las filas y `deterministic=True` en
  LightGBM.
- **Ampliar el pool de candidatos no mejora lo que el sistema entrega** (punto M6). Los
  topes sí se reajustaron al catálogo de 496 productos: el `pool_recall` pasó del 82,5 %
  al 91,8 % y, en el perfil de cliente nuevo con el carrito vacío, del 55,3 % al 77,4 %.
  La métrica principal se movió +0,03 pp. Es la misma conclusión que ya se midió en la
  Fase 3, ahora con el dataset y el ranker actuales: **el sistema no está limitado por la
  primera etapa**, sino por lo distinguibles que son entre sí las referencias de una misma
  categoría. Lo que sí arregló el punto es un techo real —la cobertura de *categorías* del
  pool en cold-start, del 70,4 % al 100 %—, y ese margen sigue sin convertirse en acierto.
  Quien quiera mover la cifra tiene que mirar la segunda etapa, no la primera.
- **El oráculo es más caro y con menos muestra efectiva que antes** (Fase 8). Con cestas de
  hasta 50 categorías, el muestreo por importancia condicionado al carrito pierde eficiencia:
  el tamaño efectivo mediano es de 151 muestras de 500, y en el 5 % peor baja a 11. El techo
  agregado aguanta el contraste contra lo realizado (máx |z| = 2,26), pero una query concreta
  puede tener bastante ruido; subir las muestras o usar remuestreo sistemático es la salida
  si alguna vez hace falta el techo *por query*.

---

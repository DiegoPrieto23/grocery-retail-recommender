# Puntos de mejora — diagnóstico del recomendador y del NBA

> Documento de diagnóstico, **sin cambios de código**. Revisión del estado tras la Fase 7
> (commit `83ab9f7`). El objetivo es explicar por qué el acierto de categoría del top-5 se
> ve tan a menudo como "0 de 5" y ordenar qué tocar primero.

## Resumen

1. **El "0 de 5" no es un caso raro: pasa en el 40 % de las cestas de test.** El sistema
   acierta al menos una categoría en el 60,0 % de las cestas (`cat_hit_rate@5`). Por
   perfil, las cestas con cero aciertos de categoría van del 32,5 % (perfil 3) al 57,7 %
   (perfil 2).
2. **No hay fuga de información ni desalineación de encoding.** El split es temporal y
   por cesta, las fuentes se reajustan en cada ventana, la sesión se corta en `cut_ts` y
   `category_idx` sale del orden alfabético de la misma tabla `products` en
   entrenamiento, test y serving. Este frente está bien resuelto.
3. **La causa principal es que el modelo trabaja con un historial desfasado.** Las
   features de cliente se congelan el día en que empieza la ventana. En el 76 % de las
   queries de test el cliente ya había hecho, de media, **3,5 compras** dentro de la
   ventana que el modelo no ve. Por eso el **15 % de los huecos del top-5** se gastan en
   categorías que el cliente compró en los 7 días anteriores, y en esos huecos la
   precisión es del 4,6 %, frente al 14,9 % del resto.
4. **A nivel de categoría, el modelo pierde contra baselines triviales.** "Las 5
   categorías más populares que no estén ya en el carrito" acierta en el 61,1 % de las
   cestas. "Las 5 categorías que más compra el cliente" acierta en el 65,1 %. El
   LambdaRank se queda en el 60,0 %. Donde sí gana es en el SKU exacto (49,5 % frente al
   38,7 % del mejor baseline). El ranker optimiza el SKU, mientras que la demo y el
   discurso de negocio hablan de categoría.
5. **El generador casi no pone estructura dentro de la cesta.** Solo hay 40 pares de
   categorías con lift > 1,5, de 3.780 posibles, y todos salen de los 10 pares
   inyectados y de los grupos bebé y mascota. El tamaño de cesta es de tipo Poisson
   (máximo 19, p99 = 11), y cada cesta lleva exactamente una línea por categoría. Hoy
   no se sabe qué parte del 40 % de fallos es entropía irreducible, porque falta un
   techo teórico (oráculo).

---

## Cómo se ha hecho el diagnóstico

- Lectura de `CHALLENGE.md`, `DATA_SPEC.md`, `ROADMAP.md`, `README.md`,
  `data_generation/`, `src/recommender/`, `src/nba/`, `src/serving/`, `src/demo/` y la CI.
- Análisis exploratorio sobre los artefactos locales de la Fase 7c:
  `data/processed/*.parquet`, `predictions/recommendations_test.parquet` (18.000 queries ×
  5), `predictions/recommender_test_context.parquet` y
  `predictions/recommender_test_queries.parquet`.
- **Aviso:** las cifras nuevas de este documento (baselines, historial desfasado,
  estructura de la cesta) salen de un script ad hoc que **no está en el repo**. Según el
  criterio de `CLAUDE.md`, antes de citarlas en el README hay que convertirlas en un
  `verify_*.py` o en un test (punto **M8**).

---

## 1. Generación de datos sintéticos

### A5 · La co-ocurrencia entre categorías es casi independiente — **ALTA**

**Problema.** En `_generate_baskets_and_items`, las categorías se sortean una a una con
pesos por cliente: afinidad lognormal × estacionalidad × ciclo de reposición. Esos pesos
solo se modifican con los 10 pares de `AFFINITY_PAIRS`, y solo cuando la disparadora sale
antes que la asociada. Fuera de eso, una cesta es un muestreo sin reemplazo e
independiente, condicionado al cliente. Medido sobre `affinity_category`, solo 40 de los
3.780 pares ordenados tienen lift > 1,5 y 56 tienen lift > 1,2. El top-25 lo ocupan
íntegramente los pares inyectados y los grupos bebé y mascota, que salen de los *gates*
de hogar y no de una afinidad de cesta.

**Por qué importa.** El caso de uso es la venta cruzada dentro de la cesta: "dado lo que
llevo, qué viene después". Si el contenido del carrito apenas informa sobre el resto, los
perfiles 2 y 4 solo pueden apoyarse en el historial y la popularidad. Encaja con lo
medido:

- el perfil 2 (nuevo, con carrito) es el peor, con `cat_hit_rate` del 42,3 %;
- el perfil 4 (recurrente, con carrito) rinde **menos** que el perfil 3 (recurrente, sin
  carrito): 53,3 % frente a 67,5 %.

Tener carrito debería ayudar, y aquí resta. Parte de la diferencia viene de que el target
es más pequeño con carrito (2,9 frente a 5,0 productos).

**Acción recomendada.**

- Introducir **misiones de compra** como variable latente de cada cesta: compra grande
  semanal, reposición rápida, desayuno, limpieza, cena o aperitivo, bebé, fiesta o
  barbacoa estacional. Cada misión tendría su mezcla de categorías y su distribución de
  tamaño, y la probabilidad de cada misión dependería del cliente.
- Ampliar la tabla de complementarios (desayuno, higiene, comida de mascota, recetas) y
  documentarla en `DATA_SPEC.md`.
- Hacer esto **calibrado contra el realismo, no contra la métrica**. Hay que dejar
  escrito qué lift se busca y por qué, igual que en la Fase 7a, para no fabricar un
  dataset hecho a medida del modelo.

### M1 · La distribución del tamaño de cesta no tiene cola larga — **MEDIA**

**Problema.** El número de líneas es `1 + Poisson(4 · f_hogar · spend)`, recortado a 20.
Medido: media 5,09, mediana 5, p99 = 11, máximo 19, coeficiente de variación 0,42 (el de
una Poisson con esa media sería 0,44). Solo el 0,03 % de las cestas tiene 15 líneas o
más. En gran consumo real conviven cestas de 1-3 artículos (reposición) con compras
semanales de 30-60, y la distribución tiene una cola larga bien marcada.

**Por qué importa.** El tamaño del target condiciona Recall, Precision y F1: la F1 de un
top-5 fijo está acotada por él. Además, sin cestas grandes no aparece el patrón de
"compra grande = todas las categorías que tocan", que es donde la señal de reposición
luce más.

**Acción recomendada.** Usar una binomial negativa o una mezcla por misión (punto A5) y
subir el tope de 20. Validar la forma en `verify_dataset.py`: cuantiles, coeficiente de
variación y porcentaje de cestas de más de 20 líneas.

### M2 · Sustitución y complementariedad pobres a nivel de producto — **MEDIA**

**Problema.**

- **Una línea por categoría por cesta** (`w[j] = 0` tras elegirla), verificado en el
  100 % de las 600.174 cestas. Nunca hay dos yogures distintos ni dos variedades de
  fruta en la misma cesta.
- **No hay sustitución entre categorías** (agua/refrescos, pollo/ternera,
  lavavajillas/limpiadores), ni sensibilidad al precio o preferencia por la marca blanca
  a nivel de cliente.
- **La sustitución dentro de una categoría sí existe**, a través de la fidelidad de
  marca y de las promociones de la competencia, que solo se llevan la parte no fiel.

**Por qué importa.** La regla "una por categoría" es una restricción dura que el modelo no
conoce (punto A2), y en la realidad no es tan estricta. La ausencia de sustitutos hace que
"llevo pollo" no reste probabilidad a "ternera", que es justo la señal negativa que un
recomendador de cesta debería aprender.

**Acción recomendada.** Añadir grupos de sustitución (llevar uno de ellos reduce el peso
de los demás) y permitir con baja probabilidad una segunda referencia en las categorías
de exploración. Añadir también una propensión a la marca blanca por cliente. Todo ello
documentado en `DATA_SPEC.md`.

### Lo que el generador ya hace bien (no tocar sin motivo)

- **Identidad de cliente con historial:** 20.000 clientes, unas 30 cestas por cliente y
  una cadencia de visita de unos 24 días.
- **Recompra de básicos:** ciclo propio por cliente y categoría
  (`typical_repurchase_days` × factor de hogar × ruido), con una penalización fuerte
  justo después de comprar (`ratio^1.8`, recortado a [0,03, 4]). Es el mecanismo más
  informativo del dataset, y justo el que el modelo ve peor (punto A1).
- **Fidelidad de marca por categoría,** con las bandas de hábito y exploración.
- **Embudo de sesión con fugas realistas,** estacionalidad, *gates* de hogar y churn
  progresivo.
- **Reproducibilidad byte a byte,** con sub-streams de semilla por etapa.

Nota menor: la tabla "Volumen de referencia" de `DATA_SPEC.md` sigue diciendo 300.000
cestas y unos 1,5 M de líneas. El generador produce 600.000 y 3,1 M (lo explica un
comentario de `GeneratorConfig`). Conviene alinear el documento (punto B3).

---

## 2. Metodología de evaluación

### A6 · No hay techo teórico: no se sabe cuánto es mejorable — **ALTA** (barato)

**Problema.** El generador conoce la probabilidad real de cada categoría en cada cesta:
afinidad × estacionalidad × ciclo, con los *gates* aplicados. Nadie la usa como
referencia, así que no se sabe si un 60 % de acierto de categoría está lejos o cerca del
máximo alcanzable.

**Por qué importa.** Sin techo, un "0 de 5" puede ser un fallo del modelo o la entropía
natural de un muestreo casi independiente de unas 4 categorías entre las ~40 activas de
cada cliente. Es la primera pregunta que hará cualquiera que vea la demo.

**Acción recomendada.** Exportar desde el generador, solo para las cestas de test, el
vector de pesos de categoría en el momento de la cesta, como columna oculta o fichero
aparte fuera de `data/raw`. Con él se calcula un **oráculo bayesiano**: el top-5 de
categorías por probabilidad real, excluyendo las del prefijo. Hay que reportar el
porcentaje del techo que alcanza cada sistema. Como el sorteo es secuencial, un
Monte Carlo sobre esos pesos da el techo esperado.

### A3 · Faltan baselines fuertes, y a nivel de categoría el modelo no los supera — **ALTA**

**Problema.** El único baseline es "popularidad reciente × estacionalidad", evaluado sobre
**el mismo pool** (`evaluate.popularity_baseline`). No hay baseline personal, ni de
reglas de asociación, ni de categoría. Medido sobre las mismas 18.000 queries, con 5
huecos y excluyendo las categorías ya en el carrito:

| Sistema | `cat_hit_rate@5` | `cat_precision@5` | `sku_hit_rate@5` | `sku_precision@5` |
| --- | ---: | ---: | ---: | ---: |
| LambdaRank (Fase 7c) | 0,600 | 0,176 | **0,495** | **0,128** |
| Top-5 categorías globales + referencia favorita del cliente | 0,611 | 0,179 | 0,296 | 0,068 |
| Top-5 categorías del cliente por frecuencia + referencia favorita | **0,651** | **0,196** | 0,387 | 0,094 |

`cat_hit_rate@5` por perfil del baseline personal: 0,566 / 0,442 / 0,735 / 0,579, frente
a 0,570 / 0,423 / 0,675 / 0,533 del modelo.

**Por qué importa.**

- Sin estos baselines el README no puede afirmar que el ranker aporta a nivel de
  categoría, y de hecho no aporta.
- El baseline personal es además un candidato serio a **fuente de candidatos o feature**,
  porque añade la frecuencia agregada por categoría y no por SKU.
- Faltan también reglas de asociación tipo Apriori para los perfiles 2 y 4, que es el
  baseline natural de "qué va con lo que llevo".

**Acción recomendada.** Añadir a `src/recommender/evaluate.py` una batería de baselines
independientes del pool:

- aleatorio;
- popularidad global;
- popularidad de categoría con la mejor referencia de cada una;
- frecuencia personal;
- frecuencia personal más `due_for_repurchase`;
- repetir la referencia favorita;
- reglas de asociación (`affinity_category` con confianza).

Evaluarlos todos a nivel SKU y categoría, y publicar la tabla en
`reports/recommender/metrics.md`. Congelar las cifras actuales en `baseline_*.json` antes
de tocar nada, como en la Fase 7.

### M3 · Un único corte por cesta, fijo y sin orden con significado — **MEDIA**

**Problema.** En `splits.build_query_items`, el hash del `basket_id` decide si el carrito
se evalúa vacío (50 %) o con `floor(n/2)` líneas (el otro 50 %). Hay **un solo corte por
cesta** y nunca una fracción aleatoria. El orden de las líneas tampoco significa nada: en
las cestas con sesión, el generador asigna los `add_to_cart` con una permutación
aleatoria (`rng.permutation`), y en las demás se ordena por hash. "Lo que viene después"
equivale, en la práctica, a "el resto de la cesta".

**Por qué importa.**

- La métrica no dice cómo evoluciona el acierto a medida que el carrito se llena, que es
  lo que se ve en la demo al añadir productos.
- El corte en la mitad es un punto concreto, no una muestra de todos los cortes posibles.
- Si algún día se quiere un "siguiente artículo" de verdad, el generador tendría que
  modelar la secuencia: por ejemplo, disparadora antes que asociada, o recorrido por
  secciones.

**Acción recomendada.**

- Evaluar varios cortes por cesta: todos los `k` en `1..n-1`, o `m` fracciones
  aleatorias con semilla, y desglosar las métricas por `prefix_size` y por fracción.
- Mantener el corte actual como cifra de cabecera para no romper la serie histórica.
- Documentar en `splits.py` y en el README que el orden no aporta información.

**Estado (Sesión 5).** Hecho. `CutPlan` (`config.py`) y `splits.build_query_items` admiten
cuatro modos: el corte de cabecera (sin cambios), todos los `k` en `1..n-1`, `m` fracciones
aleatorias con semilla (por hash, no por `F.rand`) y "carrito vacío y mitad". Con varios
cortes, la clave de la query pasa a ser `<basket_id>#k<k>` y la cesta real queda en
`source_basket_id`. El resto del pipeline no cambia. `reports/recommender/cuts.md`
desglosa por `prefix_size` y por fracción del ticket 12.400 queries de 2.947 cestas de la
muestra de test, con intervalos por cesta. `cat_hit_rate@5` pasa del 83,5 % con hasta el
25 % del ticket en el carrito al 39,5 % con más del 75 %, sobre todo porque queda menos por
adivinar (de 5,3 a 1,1 productos). La nota sobre el orden está en `splits.py` y en el
README. Los tests están en `tests/test_evaluation_robustness.py`.

### M4 · Sin intervalos de confianza, y el cold-start está poco representado — **MEDIA**

**Problema.** Todas las métricas son medias puntuales. Los perfiles 1 y 2 tienen solo 521
y 421 queries (el 5 % del test), porque en noviembre y diciembre casi todos los clientes
ya tienen historial y las cestas anónimas son el 3 %. Con n = 421, el error estándar de
un *hit rate* ronda los ±2,4 puntos. Tampoco hay contraste para ablaciones como "+5,1 %
por la señal de sesión".

**Por qué importa.** Se toman decisiones, y se escriben hallazgos en el README, con
diferencias que pueden caer dentro del ruido, sobre todo en cold-start.

**Acción recomendada.**

- Añadir un *bootstrap* pareado por query (IC al 95 % y p-valor de las diferencias entre
  sistemas) en `evaluate.summarise`.
- Sobremuestrear los perfiles 1 y 2 en el test, o añadir el **split por `customer_id`**
  que ya contempla `CLAUDE.md` para un cold-start real (clientes nunca vistos).

**Estado (Sesión 5).** Hecho.

- **Intervalos y contraste.** `evaluate.summarise` y `category_metrics` llevan intervalos
  al 95 % (bootstrap percentil, 1.000 remuestreos de cestas). `evaluate.paired_bootstrap`
  da la diferencia, su intervalo y el p-valor entre dos sistemas sobre las mismas cestas.
  `metrics.md` los aplica a la cabecera, al LambdaRank frente a la popularidad y a cada
  ablación. El verificador los aplica al LambdaRank frente a cada baseline A3. Se
  remuestrean cestas, no queries, para que los cortes de una misma cesta no estrechen el
  intervalo.
- **Cold-start: sobremuestreo, no split por cliente.** Los clientes sin compras antes de
  `test_start` ya quedan fuera de todas las fuentes y del entrenamiento del ranker, así
  que son clientes nunca vistos. Evaluar todas sus cestas (3.140, cada una con los dos
  cortes de cabecera: 6.092 queries) da un cold-start real sin reentrenar ni romper la
  serie. Un split por `customer_id` habría obligado a entrenar sin una parte de la base.
  Resultado: `cat_hit_rate@5` 59,5 % [57,9, 61,2] en el perfil 1 y 45,8 % [44,1, 47,6] en
  el 2, con intervalos de ±1,5 puntos frente a ±4–5 en la muestra.
- **Lo que confirman los intervalos.** El LambdaRank supera al mejor baseline en
  +3,9 pp [+3,3, +4,6] de categoría. La relevancia graduada gana +4,1 pp de categoría y
  pierde 0,9 de SKU, ambas fuera del ruido. El ranking personal de categorías no se
  distingue de cero (−0,1 pp [−0,5, +0,3]).
- **Un hallazgo de paso.** El entrenamiento del ranker no es determinista entre
  ejecuciones: reentrenar paró en 70 árboles en vez de 150 y movió la NDCG@5 graduada de
  0,2138 a 0,2148, dentro de su intervalo. Queda anotado en el README.
- **Un bug de paso.** `candidates_als` creaba su resultado vacío con
  `createDataFrame([])`, que levanta un worker de Python que en este entorno casca. Solo
  ocurría si ninguna query tenía cliente conocido, como en el cold-start sobremuestreado.
  Ahora el vacío se construye en la JVM.

### Fuga de datos y encoding — **verificado, sin incidencias**

- **Ventanas:** las fuentes se construyen con cestas anteriores a `FIT_END` (para el
  ranker) y a `TEST_START` (para el test). `customer_profile` recalcula el RFM en lugar
  de usar `data/processed/rfm`, que mira al final del dataset.
- **Carrito:** `in_cart` excluye del pool el prefijo y los `add_to_cart` anteriores al
  corte.
- **Sesión:** solo entran los eventos con `event_timestamp <= cut_ts`.
- **Encoding:** `category_idx` y `department_idx` salen del orden alfabético
  (`index_products`) sobre el mismo `products` limpio (62 categorías normalizadas). El
  serving usa el `products_indexed` exportado. El índice de ALS se ajusta junto con su
  modelo. `FEATURE_COLUMNS` vive en `schema.py` y lo comparten Spark y pandas, y la
  paridad la fija `test_serving_parity.py`.
- Por tanto, el acierto bajo **no** se explica por fuga ni por desalineación de
  vocabulario.

---

## 3. Modelo de recomendación

**Arquitectura actual.** Tiene dos etapas.

- **Candidatos:** cinco fuentes que producen unos 139 candidatos por query sobre 496
  productos, con un `pool_recall` del 76,9 %:
  - `pop`: popularidad × estacionalidad, 60 productos;
  - `aff`: co-compra de SKU, 30;
  - `cataff`: co-compra de categoría × 4 líderes por categoría;
  - `hist`: historial más `due_for_repurchase`, 80;
  - `als`: ALS implícito, 50.
- **Ranking:** LightGBM LambdaRank (87 árboles) sobre 54 features en 5 familias: fuente,
  cliente × producto, cliente × categoría, producto y contexto/sesión. La relevancia es
  binaria: SKU exacto sí o no.
- **Uso del historial:** es la base del modelo. `hist_rank` es con diferencia la primera
  feature por ganancia, y la fuente `hist` está detrás del 98-99 % de las
  recomendaciones en los perfiles recurrentes.

### A1 · Historial congelado al inicio de la ventana: las features no son point-in-time — **ALTA** (causa principal)

**Problema.** `customer_products`, `customer_stats` y `repurchase` se calculan una sola
vez con las cestas anteriores al inicio de la ventana: 1-sep para el entrenamiento y
1-nov para el test. `cat_days_since` y `cat_overdue_ratio` se recalculan contra
`basket_day`, pero partiendo de una **última compra desfasada**. Una query del 20-dic no
ve lo que el mismo cliente compró el 5, el 12 o el 18 de diciembre. Medido:

- el **75,9 %** de las queries identificadas tiene al menos una cesta propia invisible
  dentro de la ventana, y **3,49 de media**;
- el **24,8 %** de las líneas del target son de categorías que el cliente ya había
  comprado en la ventana;
- el **43,1 %** de los huecos del top-5 caen en categorías compradas en la ventana antes
  de la query;
- el **15,3 %** de los huecos son categorías compradas **en los últimos 7 días**, que el
  generador penaliza con fuerza (`ratio^1.8`, mínimo 0,03). En esos huecos la precisión
  SKU es del **4,6 %**, frente al **14,9 %** de los huecos sin compra en la ventana. Si se
  amplía a 14 días, son el 25,8 % de los huecos, con una precisión del 7,4 %.

El modelo ve como "vencida" una categoría que el cliente acaba de reponer, y la recomienda.

**Por qué importa.** El ciclo de reposición es el mecanismo generativo con más señal, y
es exactamente el que se mide mal. No es una fuga, porque entrenamiento y test comparten
el sesgo, pero sí es una pérdida de información que un sistema real no tendría: en
producción el historial se actualiza a diario. Afecta a los perfiles 3 y 4, que son el
95 % del test.

**Acción recomendada.** Calcular las features de cliente × categoría y cliente × producto
**as-of** la fecha de la cesta, con todas las cestas del cliente anteriores a
`basket_day` (o a `cut_ts`):

- en Spark, con una ventana ordenada por fecha sobre el historial del cliente, o con un
  *as-of join* contra una tabla de eventos cliente × categoría;
- también la fuente `hist` y el `due_for_repurchase` que usa para ordenar;
- ALS y afinidad pueden seguir congeladas por ventana, porque es el patrón habitual de
  reentrenamiento periódico.

Esto obliga a replicar la lógica en `src/serving/recommend.py` (punto B2). Hay que
volver a medir con los baselines de A3, que también deben ser as-of para que la
comparación sea justa.

**Estado (Sesión 3): hecho.** `src/recommender/history.py` agrega, para cada query,
todas las cestas de su cliente con `basket_day` estrictamente anterior. Es un *as-of join*
por `customer_id` con filtro de fecha. La réplica en pandas está en
`serving.asof_history`. Resultado sobre las mismas 18.000 queries, con el modelo
anterior congelado en `reports/recommender/baseline_pre_a1.json`:

| | Antes | Después |
| --- | ---: | ---: |
| `cat_hit_rate@5` | 0,6168 | 0,6743 |
| `sku_hit_rate@5` | 0,4947 | 0,5488 |
| Huecos en categorías compradas hace ≤ 7 días (precisión SKU) | 15,6 % (4,7 %) | 2,4 % (16,7 %) |
| Huecos en categorías compradas hace ≤ 14 días (precisión SKU) | 27,2 % (7,7 %) | 9,9 % (16,3 %) |
| Mejor baseline (as-of) menos LambdaRank | +4,4 pp (congelado) | +0,5 pp |

Las cifras de antes difieren un poco de las de arriba porque aquí el modelo de referencia
ya incluye A2. Las recalculan `python -m src.recommender.pipeline` y
`python -m src.recommender.verify_recommender_diagnostics`.

### A2 · El ranker no sabe qué categorías hay ya en el carrito, ni hay reglas post-ranking — **ALTA** (rápido)

**Problema.** `in_cart` excluye el **mismo SKU**, pero no la misma categoría, y no existe
ninguna feature del tipo "la categoría del candidato ya está en el carrito". Por
construcción del generador (una línea por categoría), un candidato de una categoría ya
presente es siempre un fallo. Medido en los perfiles 2 y 4:

- el **2,9 %** de los huecos del top-5 va a categorías ya en el carrito, con **0 %** de
  aciertos;
- el **12,8 %** de las cestas con carrito pierde al menos un hueco así.

Además no hay ninguna restricción de diversidad: el top-5 tiene 4,88 categorías distintas
de media, y un 12 % de las listas repite categoría. Solo una de las repetidas puede
acertar.

**Por qué importa.** Son huecos regalados, y en la demo se notan mucho: añades leche y te
recomienda otra leche.

**Acción recomendada.**

- Añadir features de carrito: `cat_in_cart`, `dept_share_in_cart` y el número de líneas
  del carrito en el mismo departamento. Así el modelo lo aprende, también cuando el
  generador relaje la regla (punto M2).
- Añadir un re-ranking final configurable: como máximo una referencia por categoría en el
  top-5 (tipo MMR o por cuota), y exclusión opcional de las categorías del carrito.
- Aplicarlo igual en `evaluate.top_k_predictions` y en `serving.rank_queries`.

### A4 · El objetivo del ranker no coincide con la métrica que se comunica — **ALTA**

**Problema.** El LambdaRank optimiza NDCG@5 con relevancia binaria de **SKU exacto**. La
demo y el relato de negocio ponen delante el **acierto de categoría** ("X de 5 acertaron
la categoría"). Optimizar el SKU empuja hacia productos con referencia segura (categorías
de hábito que el cliente compra mucho) y no hacia las categorías más probables, lo que
explica que pierda contra los baselines de categoría (punto A3). Además, `category_idx`
(62 niveles, tratada como categórica) es la segunda feature por ganancia con 811 *splits*:
el modelo aprende en buena medida priors por categoría.

**Por qué importa.** Mientras no se fije cuál es la métrica principal (SKU o categoría),
cualquier mejora en una puede empeorar la otra sin que nadie lo vea.

**Acción recomendada.**

1. Decidir la métrica principal y dejarla escrita en `CHALLENGE.md` y en el README.
   Propuesta: NDCG@5 con **relevancia graduada**, 2 para el SKU exacto y 1 para la
   misma categoría (`label_gain = [0, 1, 3]`). Premia ambos niveles y mantiene el SKU
   por delante.
2. Valorar una arquitectura **jerárquica** que imite el proceso generativo: primero un
   modelo de "necesidad de categoría" por cliente, cesta y momento (multietiqueta,
   dominado por el ciclo de reposición y la afinidad), después la referencia dentro de
   la categoría (favorita, promoción, popularidad). Encaja con el mercado real y con la
   foto por `visual_group` de la demo.
3. Recuperar la frecuencia personal agregada por categoría como feature
   (`cat_n_purchase_days` ya existe, pero no hay un ranking personal de categorías).

**Estado (Sesión 4).** Hechos el 1 y el 3; el 2, valorado y descartado por ahora.
Cifras sobre las mismas 18.000 cestas de test, frente a la misma arquitectura reentrenada
con relevancia binaria:

- **Métrica principal:** NDCG@5 graduada, escrita en `CHALLENGE.md` y en el README.
  `evaluate.category_metrics` la calcula para todos los sistemas.
- **Relevancia graduada:** `cat_hit_rate@5` 0,6790 → 0,7173 (+3,8 pp),
  `sku_hit_rate@5` 0,5491 → 0,5398 (−0,9 pp), NDCG@5 graduada 0,2086 → 0,2138. El
  LambdaRank queda 3,8 pp por delante del mejor baseline de categoría, con el 91,8 % del
  techo de categoría y el 91,2 % del de SKU.
- **Ranking personal de categorías** (`cat_freq_rank`, `cat_freq_share`,
  `cat_due_rank`): puestos 5, 8 y 10 de 60 por ganancia, pero sin mejora medible en la
  ablación (−0,06 pp de categoría). La señal ya estaba en las features de reposición.
- **Arquitectura jerárquica:** no se implementa. Queda a 6,4 pp del techo, y parte de esa
  distancia es error de estimación irreducible. La valoración y un esbozo del diseño
  están en `ROADMAP.md`, "Objetivo del ranker y arquitectura jerárquica".

Referencia del antes: `reports/recommender/baseline_pre_a4.json`. Lo recalculan
`python -m src.recommender.pipeline` y `verify_recommender_diagnostics`; los tests están
en `tests/test_ranker_objective.py`.

### M6 · Topes de candidatos sin reajustar, y un pool pobre en cold-start — **MEDIA-BAJA**

Deuda ya anotada en el README. El perfil 1 tiene un pool de **60 productos**, solo la
fuente `pop`, y un `pool_recall` del 38,2 %. Con 496 productos, el pool de popularidad
puede crecer (o ir por categoría: los líderes de las N categorías más probables del mes)
sin coste relevante. Conviene reajustar los topes después de A1, A2 y A4, no antes.

### B4 · Validación del ranker en la misma ventana que el entrenamiento — **BAJA**

La parada temprana usa 2.500 queries de sep-oct muestreadas por hash, la misma ventana
que el entrenamiento. Es aceptable, pero una validación temporal (las últimas 2 semanas
de la ventana) sería más fiel al desplazamiento que hay hacia el test de noviembre y
diciembre.

---

## 4. Separación conceptual recomendador vs NBA

**Estado.** En la documentación y en el código están bien diferenciados: módulos,
pipelines, modelos e informes separados.

| | Recomendador (Tarea 3a) | NBA (Tarea 3b) |
| --- | --- | --- |
| Pregunta | Qué productos añadirá a *esta* cesta | Qué acción de negocio tomar con *este cliente* |
| Grano | Query = cesta × instante de corte | Cliente (churn) y cliente × categoría (compra) |
| Horizonte | El resto de la cesta en curso | 7 días (compra) y 28 días (churn) |
| Datos | Historial + carrito + sesión | Historial hasta el corte |
| Métrica | NDCG/Recall/Precision/F1@5, hit rate | AUC/PR-AUC + valor esperado frente a políticas triviales |

### M7 · Ambigüedades en los puntos de contacto — **MEDIA** (baja para la calidad de las recomendaciones)

- **Nombre engañoso.** La acción `recomendar_producto` del NBA no usa el recomendador ni
  elige un producto: actúa a nivel de categoría con un uplift supuesto. Conviene
  renombrarla (p. ej. `recomendar_categoria`) o conectarla de verdad con el top-N del
  recomendador para esa categoría.
- **Incoherencia temporal en la demo.** El banner NBA se lee de
  `predictions/nba_actions.parquet` (corte del 1-nov), pero aparece junto a cestas reales
  de noviembre y diciembre, y al lado de un recomendador que razona con la fecha de la
  cesta. Hay que mostrar la fecha de corte en el banner, o recalcular el NBA a la fecha
  de la cesta.
- **Solapamiento que se desaprovecha.** `P(compra en categoría a 7 días)` del NBA y la
  "necesidad de categoría" del recomendador (punto A4) son casi la misma cantidad con
  horizontes distintos. Una capa común de *category need*, compartida o usada como
  feature, reduciría la duplicación y daría coherencia: el NBA no ofrecería un cupón de
  una categoría que el recomendador considera recién repuesta.
- **Deudas ya conocidas.** El "churn" es en realidad inactividad a 4 semanas (tasa base
  0,48), y el modelo no tiene features de calendario (protector solar en noviembre).

---

## 5. Estructura, documentación y reproducibilidad

### B1 · No hay un orquestador único, y la CI no ejecuta el pipeline — **MEDIA**

**Problema.**

- Reproducir el proyecto exige unos 11 comandos en orden, con unos 40 minutos de cómputo.
- La CI solo lanza `pytest`. Los tests que necesitan el bundle
  (`test_serving_parity.py`, `test_demo_baskets.py`) y parte de `test_catalog.py` **se
  saltan** en un clon limpio, así que la paridad Spark/pandas y las cifras del README
  nunca se comprueban en CI.

**Acción recomendada.**

- Añadir un único punto de entrada (`python -m src.pipeline all` o un `Makefile`/`justfile`)
  con las dependencias entre pasos.
- Añadir un trabajo de CI de *smoke* que ejecute toda la cadena con
  `generate_dataset --scale 0.02` y los tests de paridad sobre ese bundle. Puede ir en un
  job nocturno si es lento.
- Valorar la escritura de un manifiesto con los hashes de los artefactos intermedios.

### B2 · La lógica de features está duplicada entre Spark y pandas — **BAJA-MEDIA**

`src/recommender/features.py` y `src/serving/recommend.py` implementan las mismas 54
features dos veces. Lo mitiga `test_serving_parity.py`, pero **cada cambio de A1, A2 y
A4 cuesta el doble** y el test de paridad no corre en CI (punto B1). Alternativas:

- calcular las features con una sola implementación (pandas/polars o DuckDB, que a esta
  escala cabe en memoria);
- o, como mínimo, extraer las fórmulas a funciones puras compartidas.

**Estado (Sesión 3, junto con A1): se ha hecho lo segundo, no lo primero.** Las fórmulas
de negocio (ciclo de reposición, `overdue_ratio`, `cat_due` y el orden de la fuente
`hist`) viven una sola vez en `src/recommender/formulas.py`. Se escriben sobre un objeto
de operaciones (`where`, `coalesce`, `greatest`) que tiene versión de pandas y de Spark
(`formulas_spark.py`), y la aritmética funciona igual sobre una `Column` que sobre una
`Series`. Las usan el ETL de la Tarea 2 (`src/etl/repurchase.py`), el historial as-of de
Spark (`history.py`, `candidates.py`) y el serving. Lo que sigue duplicado son los joins y
agregaciones que preparan las entradas; los atan `test_serving_parity.py` y
`tests/test_asof_features.py`.

Unificar el motor no compensa en esta sesión, por tres motivos:

- `CLAUDE.md` fija PySpark para el feature engineering. Cambiarlo es una decisión de
  stack que corresponde al dueño del proyecto, no un efecto secundario de A1.
- El ALS es de Spark MLlib, así que el entrenamiento seguiría necesitando Spark.
  Unificar solo las features obliga a reescribir unas 1.000 líneas (fuentes, matriz,
  diagnóstico) y a volver a validarlas contra las cifras actuales.
- Ni DuckDB ni polars son dependencias del proyecto.

Si se retoma, la opción más natural es **DuckDB** con las features escritas una vez en SQL,
ejecutado sobre parquet en el entrenamiento y sobre pandas en el serving. Así desaparece
la réplica y el test de paridad deja de ser el único seguro.

### B3 · Documentación extensa, con restos desfasados — **BAJA**

- `README.md` (1.014 líneas) y `ROADMAP.md` (695) mezclan estado actual con la narrativa
  histórica de las fases. Cuesta ver de un vistazo cómo está hoy el proyecto.
- Quedan cifras antiguas en docstrings y notas. El docstring de `CandidateConfig` sigue
  razonando con los números de la Fase 3 (117 productos por cliente, 29,7 %…), y la
  Fase 6a del ROADMAP menciona 1.500 productos. La tabla "Volumen de referencia" de
  `DATA_SPEC.md` no cuadra con el generador.
- `GETTING_STARTED.md` es el arranque de la Fase 0 y no describe el pipeline actual.
- `docs/Como funcionan el recomendador y el Next Best Action.docx` es un binario que no
  se puede comparar en un diff.

**Acción recomendada.**

- Separar el README en "estado actual" (breve) y un `docs/HISTORIA.md` o changelog.
- Revisar los docstrings con cifras.
- Alinear el volumen de `DATA_SPEC.md`.
- Pasar el `.docx` a Markdown, o documentar de dónde se genera.

### M8 · Convertir este diagnóstico en código verificable — **MEDIA** (requisito previo)

Según `CLAUDE.md`, ninguna cifra puede ir al README sin un script que la recalcule. Las
cifras nuevas de este documento (baselines, historial desfasado, huecos desperdiciados,
estructura de co-ocurrencia, forma de la distribución de tamaño) deberían vivir en un
`verify_recommender_diagnostics.py`, o como secciones nuevas de
`reports/recommender/metrics.md` y de `verify_dataset.py`, antes de usarlas para decidir.

---

## Tabla de prioridades

| ID | Área | Punto | Prioridad | Coste estimado | Impacto esperado en recomendaciones |
| --- | --- | --- | :---: | :---: | --- |
| A1 | Modelo | Features point-in-time (historial as-of fecha de cesta) | **Alta** | Medio-alto | Alto en perfiles 3-4 (95 % del test) |
| A2 | Modelo | Features de carrito + re-ranking por categoría | **Alta** | Bajo | Medio (~3-5 % de huecos recuperables, muy visible en la demo) |
| A3 | Evaluación | Batería de baselines SKU y categoría | **Alta** | Bajo | Indirecto: hace medible todo lo demás |
| A4 | Modelo | Alinear objetivo (relevancia graduada / modelo jerárquico) | **Alta** | Medio | Alto en acierto de categoría |
| A5 | Datos | Misiones de compra y más complementariedad | **Alta** | Alto (rehacer fases) | Alto en perfiles 2-4 |
| A6 | Evaluación | Oráculo / techo teórico desde el generador | **Alta** | Bajo | Indirecto: dice cuánto queda por ganar |
| M1 | Datos | Tamaño de cesta con cola larga | Media | Medio | Medio |
| M2 | Datos | Sustitución entre categorías, varias SKU por categoría, marca blanca | Media | Medio | Medio |
| M3 | Evaluación | Varios cortes por cesta, desglose por `prefix_size` | Media | Bajo | Indirecto |
| M4 | Evaluación | Bootstrap/IC y cold-start bien representado | Media | Bajo | Indirecto |
| M6 | Modelo | Reajustar topes de candidatos (cold-start) | Media-baja | Bajo | Bajo-medio en perfil 1 |
| M7 | NBA | Nombre de la acción, fecha de corte en demo, capa común de necesidad | Media | Bajo-medio | Bajo (coherencia) |
| M8 | Gobierno | Codificar el diagnóstico en `verify_*` | Media | Bajo | Requisito previo |
| B1 | Repro | Orquestador único + smoke E2E en CI | Media | Medio | — |
| B2 | Repro | Unificar features Spark/pandas | Baja-media | Medio | — (abarata A1/A2/A4) |
| B3 | Docs | Poda y actualización de documentación | Baja | Bajo | — |
| B4 | Modelo | Validación temporal del ranker | Baja | Bajo | Bajo |

## Orden sugerido de abordaje

La idea es **medir antes de tocar**, luego arreglar el modelo sobre el dataset actual
(barato y comparable) y **agrupar todos los cambios del generador en una sola
regeneración**, porque cada una obliga a rehacer las Fases 2-6, como pasó en la Fase 7.

1. **Instrumentar (M8 + A3 + A6).** Congelar las métricas actuales en un `baseline_*.json`,
   añadir la batería de baselines y el oráculo, y convertir las cifras de este documento
   en script. Ahí se sabrá si el 60 % está lejos del techo.
2. **Arreglos rápidos de ranking (A2).** Features de carrito y re-ranking por categoría.
   Volver a medir.
3. **Features point-in-time (A1).** Es el cambio de más impacto esperado. Conviene
   hacerlo junto con **B2** o justo después, para no duplicar el trabajo en el serving.
4. **Alinear el objetivo (A4).** Relevancia graduada primero, que es barato; valorar el
   modelo jerárquico si la distancia con el oráculo lo justifica.
5. **Evaluación robusta (M3 + M4).** Varios cortes, intervalos de confianza y cold-start
   por cliente. A partir de aquí, las ablaciones se reportan con su intervalo.
6. **Nueva versión del generador (A5 + M1 + M2), en una sola "Fase 8".** Misiones,
   cola larga y sustitución, calibradas contra el realismo y documentadas en
   `DATA_SPEC.md`. Regenerar, rehacer las Fases 2-6 y comparar contra los baseline
   congelados, como se hizo en la 7.
7. **Reajuste fino (M6, B4).** Topes de candidatos y validación temporal sobre el
   dataset nuevo.
8. **NBA y cierre (M7, B1, B3).** Renombrar y conectar la acción con el recomendador,
   mostrar la fecha de corte en la demo, montar el orquestador y el smoke en CI, y
   podar la documentación.

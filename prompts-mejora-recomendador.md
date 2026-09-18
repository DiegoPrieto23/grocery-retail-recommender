# Prompts de trabajo — mejoras del recomendador y del NBA

> Basado en `puntos-de-mejora.md` (diagnóstico tras la Fase 7, commit `83ab9f7`).
> Cada sección de abajo es un prompt autocontenido, pensado para pegarse en una sesión
> nueva de Claude Code sobre el repo `grocery-retail-recommender`. Siguen el orden
> sugerido en el propio diagnóstico: medir antes de tocar, arreglos baratos primero,
> el cambio de más impacto (A1) a continuación, y todos los cambios del generador
> agrupados en una única "Fase 8" al final.

## Antes de empezar

1. Copia `puntos-de-mejora.md` al repo, por ejemplo en `docs/diagnostico-fase7.md`, para
   que las sesiones puedan referenciarlo directamente en vez de tener que pegarlo entero
   cada vez.
2. Cada prompt da por hecho que sigue vigente la norma de `CLAUDE.md`: ninguna cifra
   nueva va al README sin un script `verify_*.py` (o un test) que la recalcule.
3. No hace falta pegar todo el fichero de golpe: cada `## Sesión N` es independiente y
   asume que las anteriores ya están mergeadas (los prompts lo indican cuando dependen
   de algo previo).
4. Antes de tocar el modelo o el generador, congela las métricas y artefactos actuales
   (`baseline_*.json` o similar) para poder comparar antes/después, tal como se hizo en
   la Fase 7.

---

## Sesión 1 — Instrumentar: baselines, oráculo y diagnóstico verificable (M8, A3, A6)

**Objetivo:** antes de tocar nada del modelo, dejar medible si el 60 % de
`cat_hit_rate@5` está lejos o cerca del máximo alcanzable, y con qué se compara.

**Prompt:**
```
Estamos en el repo grocery-retail-recommender. Lee docs/diagnostico-fase7.md (puntos
A3, A6 y M8) y CLAUDE.md antes de nada.

Quiero convertir en código verificable las cifras nuevas del diagnóstico (que hoy
salen de un script ad hoc fuera del repo). Trabaja en este orden:

1. Congela las métricas actuales del sistema (LambdaRank Fase 7c) en un
   baseline_pre_diagnostico.json o similar, si no existe ya un snapshot equivalente.

2. Añade a src/recommender/evaluate.py una batería de baselines independientes del
   pool de candidatos, evaluados sobre las mismas queries de test (a nivel SKU y a
   nivel categoría):
   - aleatorio
   - popularidad global
   - popularidad de categoría + mejor referencia de esa categoría
   - frecuencia personal por categoría + referencia favorita
   - frecuencia personal + due_for_repurchase
   - repetir la referencia favorita del cliente
   - reglas de asociación a partir de affinity_category, con un umbral de confianza

   Publica la tabla resultante en reports/recommender/metrics.md, junto a las cifras
   del LambdaRank actual, desglosada también por perfil (1-4).

3. Implementa el oráculo bayesiano de A6: exporta desde el generador, solo para las
   cestas de test, el vector de pesos de categoría en el momento de la cesta (fuera de
   data/raw, como columna oculta o fichero aparte). Con ese vector, calcula el top-5 de
   categorías por probabilidad real (excluyendo el prefijo) y repórtalo como el techo
   teórico. Si el sorteo es secuencial, usa Monte Carlo sobre esos pesos para estimar
   el techo esperado. Reporta qué % del techo alcanza cada sistema (LambdaRank y cada
   baseline).

4. Escribe todo esto como un script verify_recommender_diagnostics.py (o como
   secciones nuevas de verify_dataset.py / evaluate.py, lo que encaje mejor con la
   estructura actual del repo), de forma que las cifras se puedan regenerar con un
   comando y no dependan de un script ad hoc.

No toques el modelo ni el generador en esta sesión: es solo instrumentación y
medición. Al final quiero saber, con cifras reproducibles: cuánto le saca el mejor
baseline al LambdaRank en cat_hit_rate@5, y qué % del techo teórico alcanza cada uno.
```

**Qué debe quedar terminado:** tabla de baselines en `reports/recommender/metrics.md`,
oráculo/techo teórico calculado y reportado, y todo reproducible vía script (no ad hoc).

---

## Sesión 2 — Arreglos rápidos de ranking: carrito y diversidad (A2)

**Depende de:** nada, pero conviene tener ya la Sesión 1 para medir el antes/después
contra los baselines.

**Prompt:**
```
Repo grocery-retail-recommender. Lee docs/diagnostico-fase7.md, punto A2, antes de
nada. El ranker no sabe qué categorías hay ya en el carrito (in_cart solo excluye el
mismo SKU) y no hay ninguna restricción de diversidad, así que un top-5 puede repetir
categoría o recomendar una categoría que ya está en la cesta (huecos regalados: 0% de
acierto posible ahí).

Quiero dos cambios:

1. Features de carrito nuevas en src/recommender/features.py:
   - cat_in_cart (la categoría del candidato ya está en el carrito, sí/no)
   - dept_share_in_cart (proporción del carrito en el mismo departamento que el
     candidato)
   - número de líneas del carrito en el mismo departamento que el candidato
   Añádelas a FEATURE_COLUMNS en schema.py y replica el cálculo en
   src/serving/recommend.py para mantener la paridad (revisa test_serving_parity.py).

2. Re-ranking final configurable, aplicado igual en evaluate.top_k_predictions y en
   serving.rank_queries:
   - como máximo una referencia por categoría en el top-5 (tipo MMR o por cuota)
   - exclusión opcional (flag) de categorías ya presentes en el carrito

Reentrena si hace falta, y vuelve a correr la evaluación completa (incluidos los
baselines de la Sesión 1 si ya existen) para medir el impacto en cat_hit_rate@5,
sku_hit_rate@5 y en el % de huecos "regalados" por categoría repetida o ya en carrito.
Actualiza reports/recommender/metrics.md con el antes/después.
```

**Qué debe quedar terminado:** features de carrito añadidas y replicadas en serving,
re-ranking con cuota por categoría, métricas actualizadas mostrando el impacto.

---

## Sesión 3 — Features point-in-time del historial (A1 + B2)

**Depende de:** ninguna sesión anterior estrictamente, pero es el cambio de más
impacto esperado — hazlo cuando puedas dedicarle una sesión larga.

**Prompt:**
```
Repo grocery-retail-recommender. Lee docs/diagnostico-fase7.md, punto A1 (causa
principal del diagnóstico) y B2, antes de nada.

Problema: customer_products, customer_stats y repurchase se calculan una sola vez con
las cestas anteriores al inicio de la ventana (1-sep entrenamiento, 1-nov test). Una
query del 20-dic no ve las compras del cliente del 5, 12 o 18 de diciembre. Esto hace
que el modelo recomiende como "vencidas" categorías que el cliente ya repuso hace
pocos días, justo donde el generador penaliza más fuerte.

Quiero recalcular las features de cliente × categoría y cliente × producto as-of la
fecha de cada cesta (basket_day o cut_ts), usando todas las cestas del cliente
anteriores a esa fecha:
- en Spark, con una ventana ordenada por fecha sobre el historial del cliente, o con
  un as-of join contra una tabla de eventos cliente × categoría
- incluye también la fuente de candidatos hist y el cálculo de due_for_repurchase que
  usa para ordenar
- ALS y afinidad de categoría pueden seguir congeladas por ventana (es el patrón
  habitual de reentrenamiento periódico); no hace falta tocarlas

Esto obliga a replicar la lógica en src/serving/recommend.py. Aprovecha para abordar
B2 a la vez si es razonable: valora si compensa unificar la implementación de
features (pandas/polars/DuckDB) en vez de mantener Spark y pandas por separado, dado
que cada cambio de A1 hay que hacerlo dos veces. Si no compensa en esta sesión,
como mínimo extrae las fórmulas a funciones puras compartidas entre ambos.

Verifica la paridad Spark/pandas con test_serving_parity.py después del cambio.

Reentrena el ranker y vuelve a correr toda la evaluación (baselines de la Sesión 1
incluidos, recalculados también as-of para que la comparación sea justa). Reporta el
cambio en cat_hit_rate@5 y sku_hit_rate@5, y en concreto el cambio en precisión sobre
los huecos que antes caían en categorías repuestas en los últimos 7-14 días.
```

**Qué debe quedar terminado:** features as-of implementadas en Spark y replicadas en
serving, paridad verificada, baselines recalculados as-of, métricas comparadas
antes/después.

---

## Sesión 4 — Alinear el objetivo del ranker (A4)

**Depende de:** Sesión 3 (A1) recomendable antes, para que la comparación de objetivo
no esté contaminada por el historial desfasado.

**Prompt:**
```
Repo grocery-retail-recommender. Lee docs/diagnostico-fase7.md, punto A4, antes de
nada. El LambdaRank optimiza NDCG@5 con relevancia binaria de SKU exacto, pero la demo
y el discurso de negocio hablan de acierto de categoría, y ahí el modelo pierde contra
baselines triviales.

Quiero, en este orden:

1. Decidir y dejar escrita en CHALLENGE.md y en el README cuál es la métrica
   principal. Propuesta a implementar primero, por ser barata: NDCG@5 con relevancia
   graduada, label_gain = [0, 1, 3] (2 para el SKU exacto, 1 para la misma categoría
   que el target aunque no sea el SKU exacto). Reentrena con esa función objetivo y
   compara cat_hit_rate@5 y sku_hit_rate@5 antes/después.

2. Recupera como feature la frecuencia personal agregada por categoría a nivel de
   ranking de categorías del cliente (cat_n_purchase_days ya existe, pero falta un
   ranking personal de categorías explícito). Añádela a FEATURE_COLUMNS y mide su
   importancia.

3. No implementes todavía la arquitectura jerárquica (modelo de necesidad de
   categoría + modelo de referencia dentro de categoría) — solo valora por escrito, en
   un apartado del README o de ROADMAP.md, si la distancia al oráculo (Sesión 1, punto
   A6) la justifica. Si la justifica claramente, propone el diseño pero no lo
   implementes en esta sesión.

Actualiza reports/recommender/metrics.md con el nuevo objetivo y compáralo contra el
LambdaRank de SKU puro y contra los baselines de categoría.
```

**Qué debe quedar terminado:** métrica principal decidida y documentada, ranker
reentrenado con relevancia graduada, comparación publicada, valoración escrita (no
implementación) de la arquitectura jerárquica.

---

## Sesión 5 — Evaluación robusta: cortes, intervalos y cold-start (M3 + M4)

**Depende de:** ninguna estrictamente, pero tiene más sentido una vez estabilizado el
modelo (después de A1/A2/A4), para que las ablaciones que mida ya sean las buenas.

**Prompt:**
```
Repo grocery-retail-recommender. Lee docs/diagnostico-fase7.md, puntos M3 y M4, antes
de nada.

Dos mejoras de metodología de evaluación, independientes entre sí:

1. (M3) Hoy solo hay un corte por cesta (mitad del carrito, decidido por hash de
   basket_id). Añade en splits.py la posibilidad de evaluar varios cortes por cesta:
   todos los k en 1..n-1, o m fracciones aleatorias con semilla fija. Desglosa las
   métricas por prefix_size (o por fracción) en un reporte nuevo, y deja el corte
   actual como cifra de cabecera para no romper la serie histórica de métricas ya
   publicadas. Documenta en splits.py y en el README que el orden de las líneas no
   aporta información (se asigna por permutación aleatoria en el generador).

2. (M4) Añade un bootstrap pareado por query en evaluate.summarise, que calcule
   intervalos de confianza al 95% y p-valor para las diferencias entre sistemas
   (LambdaRank vs. cada baseline, y para cualquier ablación que se quiera reportar en
   el futuro). Además, dado que los perfiles 1 y 2 (cold-start) son solo ~5% del test
   (521 y 421 queries), añade o bien sobremuestreo de esos perfiles en el test, o
   implementa el split adicional por customer_id que ya contempla CLAUDE.md para un
   cold-start real con clientes nunca vistos. Elige la opción que encaje mejor con la
   estructura actual de splits.py y justifícalo brevemente en el commit.

Actualiza reports/recommender/metrics.md para que las cifras de cabecera lleven su
intervalo de confianza a partir de ahora.
```

**Qué debe quedar terminado:** evaluación multi-corte con desglose por `prefix_size`,
bootstrap con IC/p-valor en `evaluate.summarise`, cold-start mejor representado.

---

## Sesión 6 — Fase 8 del generador: misiones de compra y sustitución (A5 + M1 + M2)

**Depende de:** conviene tener ya la Sesión 1 (baselines + oráculo) para poder
comparar la Fase 8 contra algo, y es la sesión más cara: regenerar obliga a rehacer
las Fases 2-6. Agrupa aquí TODOS los cambios de generador para no regenerar varias
veces.

**Prompt:**
```
Repo grocery-retail-recommender. Lee docs/diagnostico-fase7.md, puntos A5, M1 y M2, y
también la sección "Lo que el generador ya hace bien (no tocar sin motivo)", antes de
nada. Esta es la Fase 8: agrupo aquí todos los cambios del generador para regenerar
una sola vez.

Congela antes de nada los artefactos y métricas actuales (dataset, modelo, baselines,
oráculo) en una carpeta de snapshot "pre-fase-8", igual que se hizo en la Fase 7.

Cambios a implementar en data_generation/ y a documentar en DATA_SPEC.md:

1. (A5) Misiones de compra como variable latente de cada cesta: compra grande semanal,
   reposición rápida, desayuno, limpieza, cena o aperitivo, bebé, fiesta/barbacoa
   estacional. Cada misión con su propia mezcla de categorías y distribución de
   tamaño; la probabilidad de cada misión depende del cliente. Amplía también la tabla
   de complementarios (desayuno, higiene, comida de mascota, recetas). Antes de tocar
   el código, escribe en DATA_SPEC.md qué lift de co-ocurrencia se busca y por qué —
   esto se calibra contra el realismo de una cesta de supermercado real, no contra la
   métrica del modelo.

2. (M1) Sustituye la Poisson recortada del tamaño de cesta por una binomial negativa o
   una mezcla por misión (puede salir directamente del punto 1), y sube el tope de 20
   líneas. Valida la forma en verify_dataset.py: cuantiles, coeficiente de variación y
   % de cestas de más de 20 líneas.

3. (M2) Añade grupos de sustitución entre categorías (agua/refrescos, pollo/ternera,
   lavavajillas/limpiadores...): llevar una reduce el peso de las demás del grupo.
   Permite con baja probabilidad una segunda referencia dentro de una categoría en las
   bandas de exploración (hoy es siempre una línea por categoría). Añade una
   propensión a la marca blanca por cliente.

No toques nada de lo que ya funciona bien (identidad de cliente, ciclo de recompra,
fidelidad de marca, embudo de sesión, reproducibilidad por semillas) salvo lo
estrictamente necesario para encajar los puntos de arriba.

Una vez implementado: regenera el dataset completo, rehaz las Fases 2-6 (features,
candidatos, entrenamiento, evaluación), y compara contra el snapshot pre-fase-8 y
contra los baselines/oráculo congelados en la Sesión 1. Publica el nuevo lift medido
en affinity_category (target: bastantes más de 40 pares con lift > 1,5) y las nuevas
métricas del modelo en reports/recommender/metrics.md.
```

**Qué debe quedar terminado:** generador con misiones de compra, tamaño de cesta con
cola larga, sustitución entre categorías y marca blanca; dataset regenerado; Fases 2-6
rehechas; comparación contra el snapshot previo y contra los baselines/oráculo.

---

## Sesión 7 — Reajuste fino: candidatos y validación temporal (M6 + B4)

**Depende de:** hazla después de A1, A2 y A4 (Sesiones 2-4); no tiene sentido antes.

**Prompt:**
```
Repo grocery-retail-recommender. Lee docs/diagnostico-fase7.md, puntos M6 y B4, antes
de nada.

1. (M6) El perfil 1 (cold-start puro) tiene un pool de candidatos de solo 60
   productos, con la única fuente pop, y un pool_recall bajo. Con 496 productos en
   catálogo, amplía el pool de popularidad para ese perfil, o constrúyelo por
   categoría (líderes de las N categorías más probables del mes) en vez de solo por
   popularidad global. Reajusta también los topes del resto de fuentes de candidatos
   (pop, aff, cataff, hist, als) a la luz de los cambios ya hechos en A1/A2/A4, y mide
   el nuevo pool_recall por perfil.

2. (B4) La parada temprana del ranker usa 2.500 queries de sep-oct muestreadas por
   hash, la misma ventana que el entrenamiento. Cambia a una validación temporal:
   las últimas 2 semanas de la ventana de entrenamiento, para que sea más fiel al
   desplazamiento hacia el test de noviembre-diciembre. Compara el resultado del
   modelo con parada temprana temporal frente al actual.

Actualiza reports/recommender/metrics.md con el pool_recall nuevo por perfil y con la
comparación de validación temporal vs. por hash.
```

**Qué debe quedar terminado:** topes de candidatos reajustados y `pool_recall`
recalculado por perfil, validación temporal implementada y comparada.

---

## Sesión 8 — NBA y cierre: coherencia, orquestador y documentación (M7 + B1 + B3)

**Depende de:** conviene hacerla última, ya que da por hecho que el resto de cambios
del recomendador (A1-A4, Fase 8) ya están mergeados y estables.

**Prompt:**
```
Repo grocery-retail-recommender. Lee docs/diagnostico-fase7.md, puntos M7, B1 y B3,
antes de nada. Esta sesión es de cierre: coherencia recomendador/NBA, reproducibilidad
y poda de documentación. No toques el modelo del recomendador ni el generador.

1. (M7) NBA:
   - Renombra la acción recomendar_producto (que hoy actúa a nivel de categoría con
     un uplift supuesto, sin usar el recomendador) a algo como recomendar_categoria,
     o conéctala de verdad con el top-N del recomendador para esa categoría — decide
     cuál de las dos opciones encaja mejor con el estado actual del código y hazlo.
   - En la demo, el banner de NBA se lee de un corte fijo (1-nov) pero se muestra
     junto a cestas reales de fechas posteriores. Añade la fecha de corte visible en
     el banner, o recalcula el NBA a la fecha de la cesta que se está viendo.
   - Valora si compensa una capa común de "category need" compartida entre
     P(compra en categoría a 7 días) del NBA y la necesidad de categoría del
     recomendador (relacionado con A4): si el recomendador ya sabe que una categoría
     se acaba de reponer, el NBA no debería ofrecer un cupón de esa categoría. Si
     compensa, impleméntalo; si no, documenta por qué no en ROADMAP.md.

2. (B1) Añade un único punto de entrada para reproducir el proyecto (por ejemplo
   python -m src.pipeline all, o un Makefile/justfile) con las dependencias entre
   pasos explícitas, sustituyendo a la secuencia actual de ~11 comandos manuales.
   Añade un job de CI de smoke test que ejecute toda la cadena con
   generate_dataset --scale 0.02 y corra ahí los tests de paridad Spark/pandas
   (test_serving_parity.py, test_demo_baskets.py, test_catalog.py), que hoy se saltan
   en un clon limpio. Puede ir como job nocturno si es lento para correr en cada push.

3. (B3) Separa el README en un "estado actual" breve y un docs/HISTORIA.md (o
   changelog) con la narrativa de fases. Revisa y corrige los docstrings con cifras
   desfasadas (CandidateConfig con números de la Fase 3, ROADMAP con la Fase 6a
   desfasada). Alinea la tabla "Volumen de referencia" de DATA_SPEC.md con el volumen
   real que produce el generador. Actualiza GETTING_STARTED.md para que describa el
   pipeline actual, no el de la Fase 0. Pasa docs/Como funcionan el recomendador y el
   Next Best Action.docx a Markdown, o documenta claramente de dónde se genera ese
   fichero.
```

**Qué debe quedar terminado:** NBA renombrado/conectado y con fecha de corte visible,
orquestador único con smoke test en CI, documentación podada y alineada con el estado
real del proyecto.

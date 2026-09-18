# ROADMAP

Plan de trabajo por fases. Cada fase debería cerrar con algo demostrable (un notebook, un
test que pasa, un fichero de salida) antes de pasar a la siguiente.

## Fase 0 — Setup

- [x] Estructura de carpetas del repo (ver `CHALLENGE.md`)
- [x] Entorno: `requirements.txt` / `pyproject.toml`, PySpark local
- [x] `constraints.txt` con versiones fijas (numpy, pandas, faker, pyspark) para
      reproducibilidad, como en `sports-rental-analytics`
- [x] CI mínimo: un workflow que instale dependencias y ejecute los tests

## Fase 1 — Generador de datos

- [x] `generate_dataset.py`: `customers`, `products`, `promotions`, `baskets`,
      `basket_items`, `sessions`, `session_events` según `DATA_SPEC.md`
- [x] Semilla fija (seed=42) propagada a numpy/random/Faker
- [x] Reglas de negocio: ciclos de reposición, afinidad de cesta, estacionalidad, uplift de
      promoción, señal de churn progresiva
- [x] Problemas de calidad deliberados
- [x] Diccionario de datos autogenerado (`output/README.md`, como en el otro proyecto)
- [x] Verificación: dos ejecuciones producen el mismo hash

## Fase 2 — ETL y feature engineering (PySpark)

Se ejecuta entera con `python -m src.etl.run_etl`, que deja las tablas en
`data/processed/` y los informes en `reports/etl/`.

- [x] Limpieza: duplicados, nulos, categorías inconsistentes, outliers — documentado
      · `src/etl/cleaning.py` · criterios en `docs/CLEANING.md` · recuento reproducible en
      `reports/etl/cleaning_report.md` · verificado por `tests/test_cleaning.py`
- [x] Data Trust Score (Tarea 1)
      · `src/etl/data_trust.py` · 79 comprobaciones en 5 dimensiones · **89,99 (C) sobre el
      crudo → 100,00 (A) sobre el limpio** · `reports/etl/data_trust.md` · verificado por
      `tests/test_data_trust.py`
- [x] Notebook de EDA: preguntas de negocio de la Tarea 1 resueltas con Spark SQL y
      visualizaciones, una por pregunta
      · `notebooks/01_eda.ipynb` · 9 preguntas, cada una con su `spark.sql` y su figura ·
      ejecutado de principio a fin, con salidas guardadas
- [x] RFM por cliente
      · `src/etl/rfm.py` · tabla `data/processed/rfm` · verificado por `tests/test_rfm.py`
- [x] Función `due_for_repurchase` por cliente-categoría (Tarea 2)
      · `src/etl/repurchase.py` · tabla `data/processed/repurchase_features` · verificado
      por `tests/test_repurchase.py`
- [x] Afinidad de cesta (co-ocurrencia / FP-Growth) como tabla intermedia para el
      recomendador
      · `src/etl/affinity.py` · tablas `affinity_category` y `affinity_product` · los 10
      pares de `DATA_SPEC.md` salen con lift > 1,8 y coinciden con
      `data_generation/verify_dataset.py` hasta el segundo decimal ·
      `reports/etl/affinity_expected_pairs.md` · verificado por `tests/test_affinity.py`

## Fase 3 — Recomendador de cesta

Se ejecuta entera con `python -m src.recommender.pipeline`, que deja el ranker en
`models/`, las predicciones en `predictions/` y los informes en `reports/recommender/`.
`python -m src.recommender.demo_profiles` traduce esas salidas a un caso legible de cada
perfil.

- [x] Split train/test por `basket_id` (no por fila) para evitar fuga de datos
      · `src/recommender/splits.py` · **temporal y por cesta**, en tres ventanas: fuentes
      hasta 2025-09-01, queries del ranker sep-oct 2025, test desde 2025-11-01. Las
      fuentes de candidatos se **reajustan por ventana**, así que ninguna ve el futuro de
      la cesta que predice · verificado por `tests/test_recommender.py`
- [x] Generación de candidatos: popularidad/estacionalidad, co-compra (Fase 2), ALS
      (Spark MLlib)
      · `src/recommender/candidates.py` · cinco fuentes (se añadió co-compra por
      **categoría**, que cubre la cola larga donde `affinity_product` no llega) · 155
      candidatos por cesta de media, `pool_recall` = 31,1 %
- [x] Feature engineering para el ranker: señal de cada fuente de candidatos,
      recency/frequency, promoción, popularidad reciente, señal de sesión
      · `src/recommender/features.py` · 60 features en 6 familias (la de carrito, del punto A2,
      y el ranking personal de categorías del punto A4 dentro de la de cliente × categoría)
      · desde el punto A1, el historial personal (fuente `hist`, `due_for_repurchase` y
      las features de cliente) se calcula **as-of el día de cada cesta**
      (`src/recommender/history.py`), con las fórmulas compartidas con la demo en
      `formulas.py` · verificado por `tests/test_asof_features.py` (corte estricto y
      paridad Spark/pandas), `tests/test_serving_parity.py` y la sección de huecos por
      recencia de `verify_recommender_diagnostics`
- [x] **Deuda detectada en la Fase 2 (Q9 del EDA): la señal de sesión estaba contaminada.**
      **Resuelta arreglando el generador.** El problema era peor de lo anotado: no sólo
      `add_to_cart`, también los `view` cubrían el 100 % de la cesta. `_generate_sessions`
      produce ahora abandono de carrito a nivel de línea, productos que sólo se miran, y
      un retardo entre ver y añadir — `P(en la cesta | add_to_cart)` = 87,9 %,
      `P(en la cesta | view)` = 58,6 %, `P(visto | en la cesta)` = 85,0 % · documentado en
      `DATA_SPEC.md` → "Embudo online" · fijado por dos tests en
      `tests/test_generate_dataset.py` · la señal se consume con un corte estricto en
      `cut_ts` y aporta **+13 % de NDCG@5** sobre la ablación que la excluye
- [x] Ranking: LightGBM con objetivo `LambdaRank` sobre los candidatos
      · `src/recommender/ranker.py` · parada temprana sobre NDCG@5 de validación · desde el
      punto A4, con **relevancia graduada** (2 SKU exacto, 1 misma categoría,
      `label_gain = [0, 1, 3]`), que es la métrica principal fijada en `CHALLENGE.md` ·
      verificado por `tests/test_ranker_objective.py` y la sección "Objetivo del ranker"
      de `reports/recommender/metrics.md`, que reentrena la variante binaria como ablación
- [x] Lógica de combinación para los 4 perfiles de cliente (Tarea 3a)
      · el ranker es **el mismo** para los cuatro; lo que cambia es qué fuentes tienen
      señal. Medido sobre el top-5: el perfil 1 se cubre 100 % con popularidad, el 4
      combina las cinco · `reports/recommender/profiles_demo.md`
- [x] Evaluación: NDCG@5, Recall@5 sobre el test
      · `src/recommender/evaluate.py` · **NDCG@5 = 0,0343 · Recall@5 = 0,0332 ·
      hit_rate@5 = 11,8 %** sobre 18.000 cestas · baseline de popularidad 0,0200 ·
      `reports/recommender/metrics.md`

### Lo que limita la métrica (hallazgo de la fase)

El NDCG@5 de SKU es bajo y **no es por la primera etapa**. Dos medidas lo demuestran:

1. **Categoría vs SKU.** El sistema acierta la categoría en el **51,4 %** de las cestas y
   el SKU exacto sólo en el 11,8 %. De los aciertos de categoría, apenas el 13 % son
   además el SKU correcto.
2. **Ampliar el pool no ayuda.** Subir de 92 a 155 candidatos por cesta llevó el
   `pool_recall` del 21,7 % al 31,1 % (+43 % de target alcanzable) y el NDCG@5 no se movió
   (0,0344 → 0,0343).

La causa está en el generador: dentro de una categoría hay ~24 referencias y la elección
es casi aleatoria. Un cliente con tres o más compras en una categoría compra **0,86
referencias distintas por compra** — es decir, casi nunca repite SKU — y sólo el 29,7 % de
las líneas de una cesta futura son productos que ya había comprado, frente al **88,6 %**
cuando se mira la categoría. La fidelidad está en la categoría, no en la referencia.

- [x] **Deuda para más adelante: dar fidelidad de marca/SKU al generador.** En gran consumo
      real un cliente repite referencia (siempre la misma leche), y ahí es donde un
      recomendador de SKU tiene margen. Implica tocar la elección de producto dentro de
      categoría en `_generate_baskets_and_items`, regenerar el dataset (cambian los
      `sha256` de `baskets` y `basket_items`) y rehacer la Fase 2 y sus informes.
      · **La parte de generador está hecha en la Fase 7a**: 8 referencias por categoría y
      lealtad por categoría, con la repetición de SKU subiendo de 0,850 a 0,440 referencias
      distintas por compra. Falta rehacer las Fases 2, 3, 4 y 6a sobre el dataset nuevo
      (7b-7e), que es donde se verá si el NDCG@5 de SKU sube.
      · **Subió**: la 7c reentrena sobre el dataset nuevo y el NDCG@5 pasa de 0,0343 a
      0,1752, con el hit_rate de SKU de 11,8 % a 49,5 %. La 7d rehace la Fase 4 sin cambios
      notables, y la 7e rehace la 6a y lleva el acierto de categoría a la demo. Cerrada.

### Objetivo del ranker y arquitectura jerárquica (punto A4)

El punto A4 de `docs/diagnostico-fase7.md` hizo dos cambios y dejó una tercera idea solo
valorada, sin implementar. Todas las cifras salen de `python -m src.recommender.pipeline`
y `python -m src.recommender.verify_recommender_diagnostics`, sobre las mismas 18.000
cestas de test.

- [x] Métrica principal decidida y documentada: **NDCG@5 graduada** (`CHALLENGE.md`,
      README) · `evaluate.category_metrics` la calcula para todos los sistemas, oráculos y
      baselines incluidos
- [x] Ranker reentrenado con relevancia graduada · `cat_hit_rate@5` 0,6790 → **0,7173**
      (+3,8 pp), `sku_hit_rate@5` 0,5491 → 0,5398 (−0,9 pp), NDCG@5 graduada 0,2086 →
      **0,2138**, frente a la misma arquitectura entrenada con relevancia binaria. El
      LambdaRank pasa a ir **3,8 pp por delante** del mejor baseline de categoría (antes
      iba 0,5 pp por detrás)
- [x] Ranking personal de categorías como features (`cat_freq_rank`, `cat_freq_share`,
      `cat_due_rank`, en `schema.CUSTOMER_CATEGORY_RANK_FEATURES`), as-of y con réplica en
      la demo · puestos 5, 8 y 10 de 60 por ganancia, **pero la ablación no mide mejora**
      (`cat_hit_rate@5` −0,06 pp, `sku_hit_rate@5` −0,21 pp sin ellas): el árbol ya sacaba
      esa señal de `cat_n_purchase_days`, `cat_due` y `hist_rank`, y las nuevas le sustituyen
      splits sin añadir información. Se mantienen porque son baratas y legibles, pero
      quitarlas es una opción legítima si se quiere simplificar
- [ ] Arquitectura jerárquica (modelo de necesidad de categoría + modelo de referencia
      dentro de la categoría) · **valorada, no implementada; hoy no está justificada**

**Distancia al oráculo después de A4** (techo de la Sesión 1, punto A6):

| | LambdaRank servido | Oráculo de categoría | Oráculo de SKU | % del mejor techo |
| --- | ---: | ---: | ---: | ---: |
| `cat_hit_rate@5` | 0,7173 | 0,7816 | 0,7309 | 91,8 % |
| `cat_precision@5` | 0,2265 | 0,2660 | 0,2339 | 85,2 % |
| `sku_hit_rate@5` | 0,5398 | 0,5542 | 0,5920 | 91,2 % |
| NDCG@5 graduada | 0,2138 | 0,2308 | 0,2340 | 91,4 % |

Por perfil, el hueco en `cat_hit_rate@5` es parecido en los cuatro (5,8 pp en el 1, 6,7
en el 2, 6,3 en el 3 y 6,6 en el 4): no hay un perfil donde falle algo estructural.

**Por qué no está justificada todavía.**

1. **El margen es pequeño y no todo es recuperable.** Quedan 6,4 pp de `cat_hit_rate@5`
   (y 0,020 de NDCG graduada). El oráculo conoce los pesos reales del generador para cada
   cliente; cualquier modelo que solo vea el pasado los estima con ~30 cestas por cliente,
   así que parte de esa distancia es error de estimación que ninguna arquitectura quita.
2. **Lo que la jerarquía aportaría ya lo ha aportado el objetivo.** La idea es que el
   modelo aprenda explícitamente *qué categoría toca*. La relevancia graduada ya le da esa
   señal, y ha cerrado 3,8 de los 10,3 pp que había hasta el techo. Que el ranking personal
   de categorías no sume nada apunta en la misma dirección: al ranker no le falta
   información de necesidad de categoría que un modelo aparte fuera a darle.
3. **La jerarquía no resuelve el compromiso SKU/categoría, lo hace explícito.** Incluso los
   dos oráculos se reparten el terreno: el de categoría gana 5,1 pp de categoría y pierde
   3,8 de SKU frente al de SKU. El LambdaRank graduado está a 1,4 pp del oráculo de
   categoría en SKU y a 6,4 en categoría, y la palanca para moverse por esa frontera ya
   existe (`label_gain`).
4. **Cuesta mucho.** Dos modelos, dos matrices (query × categoría y query × referencia),
   una regla para combinarlos, y todo por duplicado en Spark y en la demo (punto B2).

**Qué haría cambiar la decisión.** Retomarla después de la Fase 8 del generador (A5:
misiones de compra y complementariedad), si con el dato nuevo la distancia al oráculo de
categoría vuelve a pasar de ~10 pp, o si `cat_precision@5` sigue por debajo del 90 % del
techo después de reajustar los candidatos (M6), que es más barato. Antes conviene probar
lo barato: barrer `label_gain` (p. ej. `[0, 1, 2]` o `[0, 2, 3]`) para situar el punto en
la frontera SKU/categoría que se quiera comunicar.

**Respuesta (Sesión 7, tras M6).** La condición **se cumple**: reajustados los candidatos,
`cat_precision@5` está en el **87,5 %** del techo (0,3295 de 0,3765), todavía por debajo
del 90 %. Apenas se movió con el pool ampliado (86,8 % → 87,5 %), que es lo esperable
ahora que se sabe que la primera etapa no era la restricción: el pool cubre el **100 %**
de las categorías del target en los cuatro perfiles y aun así `cat_hit_rate@5` se queda en
el 79,8 % (95,2 % del techo). Lo que falta está entero en la segunda etapa. Sigue siendo
válido probar antes lo barato (barrer `label_gain`) que montar la jerarquía.

**Esbozo, por si se retoma.** (1) Modelo de necesidad: LightGBM binario (o LambdaRank con
grupo = cesta) sobre filas query × categoría fuera del carrito, con las features de
cliente × categoría as-of, estacionalidad, afinidad con el carrito y sesión agregada por
categoría; etiqueta = la categoría está en el resto de la cesta. (2) Modelo de referencia:
dentro de cada categoría, un ranker sobre sus 8 referencias con fidelidad
(`hist_n_baskets`, `hist_days_since`), promoción, precio y popularidad; etiqueta = SKU
comprado, condicionado a que la categoría se compró. (3) Combinación: top-5 de
categorías por `P(categoría)`, cada una con su mejor referencia (la forma del oráculo de
categoría), o por `P(categoría) × P(referencia)` si se quiere empujar el SKU (la del
oráculo de SKU). Se evaluaría con la misma NDCG graduada y contra los mismos techos.

### Evaluación robusta (puntos M3 y M4)

Todas las cifras salen de `python -m src.recommender.pipeline` (`metrics.md` y `cuts.md`)
y de `python -m src.recommender.verify_recommender_diagnostics`. Los tests están en
`tests/test_evaluation_robustness.py`.

- [x] Varios cortes por cesta (`CutPlan`: todos los `k` en `1..n-1` o `m` fracciones con
      semilla) con el corte de siempre como cabecera · desglose por `prefix_size` y por
      fracción del ticket en `reports/recommender/cuts.md` (12.400 queries, 2.947 cestas)
      · el orden de las líneas, documentado como sin información en `splits.py` y el README
- [x] Intervalos de confianza de las cifras de cabecera (bootstrap por cesta) · NDCG@5
      graduada **0,2148 [0,2120, 0,2177]**
- [x] Bootstrap pareado entre sistemas (`evaluate.paired_bootstrap`) · LambdaRank frente a
      la popularidad y cada ablación en `metrics.md`, y frente a cada baseline A3 en la
      sección de diagnóstico · mejor baseline: **+3,9 pp [+3,3, +4,6]** de categoría
- [x] Cold-start bien representado: todas las cestas de la ventana de test sin historial
      previo (3.140 cestas, 6.092 queries) en vez de las 942 de la muestra · se eligió
      sobremuestrear y no un split por `customer_id` porque esos clientes ya quedan fuera
      de todo entrenamiento
- [ ] Entrenamiento determinista del ranker · reentrenar con el mismo código movió la
      cabecera de 0,2138 a 0,2148 (70 árboles frente a 150). Los intervalos lo cubren,
      pero una serie limpia pediría fijar el orden de las filas y `deterministic=True`

## Fase 4 — Next Best Action

Se ejecuta entera con `python -m src.nba.pipeline`, que deja los dos modelos en `models/`,
la tabla `customer_id → acción → valor esperado` en `predictions/nba_actions.parquet` y los
informes en `reports/nba/`.

- [x] Target de propensión: compra en categoría en próximos 7 días / churn en 4 semanas
      · `src/nba/targets.py` · **no se usa `customers.churn_label` como target**: está
      definido respecto al final del dataset (60 días sin comprar hasta 2025-12-31), así
      que en un corte de abril sería adivinar algo de ocho meses después, y como *feature*
      sería fuga. El churn se construye por corte de forma observacional; `churn_label`
      queda para una comprobación de cordura, que da **83,4 % de coincidencia** ·
      verificado por `tests/test_nba.py`
- [x] Modelo (LightGBM o logística) + validación temporal (no aleatoria, para no filtrar
      futuro)
      · `src/nba/propensity.py` · dos binarios · cuatro cortes de entrenamiento
      (abr–ago 2025), validación en 2025-09-15, test en 2025-11-01 · las features de cada
      corte sólo miran compras anteriores a él, fijado por un test que añade una compra
      posterior y comprueba que ninguna feature se mueve
- [x] Catálogo de acciones y coste/margen asociado por acción
      · `src/nba/config.py` · `ninguna_accion`, `recomendar_producto`,
      `enviar_cupon_categoria` · margen bruto **por departamento** (18 % Frescos → 35 %
      Droguería/Higiene) · el coste se parte en dos: `send_cost` se paga siempre,
      `discount` sólo si el cliente compra · valor facial del cupón anclado a la media real
      de los cupones de `promotions` (2,54 €)
- [x] Política de valor esperado (Tarea 3b)
      · `src/nba/policy.py` · todo se mide **incremental sobre no actuar**, así
      `ninguna_accion` vale 0 por construcción y una acción sólo gana si su efecto paga su
      coste · actúa sobre el **75,6 %** de los clientes, no sobre todos
- [x] Evaluación: AUC/PR-AUC del modelo, uplift de valor esperado de la política frente a
      "no actuar" y "actuar siempre"
      · **Churn 4 semanas: AUC = 0,8531 · PR-AUC = 0,8556** (tasa base 0,4759) ·
      **Compra en categoría 7 días: AUC = 0,7634 · PR-AUC = 0,2190** (tasa base 0,0619) ·
      política **+4.012 €** frente a "no actuar", **+3.082 €** frente a la mejor
      alternativa trivial · `reports/nba/metrics.md`

### Lo que no es medible con este dataset (hallazgo de la fase)

`P(conversión | acción)` **no es identificable aquí**. El generador aplica su
`PROMO_UPLIFT = 3.0` al reparto de cuota *dentro* de una categoría (qué SKU se elige), no a
la probabilidad de comprar la categoría ni a la de volver: el tratamiento nunca varía. Las
propensiones sí se miden; el efecto de cada acción es un **supuesto declarado** en
`config.py`, y el informe lo barre en vez de esconderlo.

Barrer el supuesto cambió la lectura del resultado. El barrido obvio —el uplift de
conversión del cupón— resultó ser el parámetro equivocado: pasar de 1,00 a 2,00 mueve el
total de 3.938 € a 4.721 €. La razón es económica: con un cupón de 2,54 € persiguiendo un
margen esperado de ~1,4 €, **el descuento es mayor que el margen**, así que por cross-sell
el cupón destruye valor haga lo que haga la conversión. Todo lo que aporta viene de la
retención. El barrido que importa es el de `churn_reduction`:

| `churn_reduction` | Valor de la política | Cupones repartidos |
| ---: | ---: | ---: |
| **0,00** (no retiene a nadie) | **1.210 €** | 0 % |
| 0,05 | 1.734 € | 41,7 % |
| **0,10** (el supuesto) | **4.012 €** | 64,8 % |
| 0,20 | 8.862 € | 71,7 % |

La conclusión robusta es la primera fila: **incluso suponiendo que el cupón no retenga a
nadie, la política sigue ganando** (1.210 € frente a 930 € de "recomendar siempre" y 0 € de
"no actuar"), y en ese escenario deja de repartir cupones por completo. Lo que depende del
supuesto es el tamaño del premio, no el signo.

- [ ] **Deuda: el "modelo de churn" es en realidad un modelo de inactividad a 4 semanas.**
      Su tasa base es 0,4759 porque la cadencia media de visita ronda los 24 días, así que
      no comprar en 4 semanas le pasa a media base sin ser abandono. Las features
      dominantes (`n_baskets_90d`, `avg_days_between_baskets`, `recency_days`) confirman
      que buena parte de lo que acierta es frecuencia de compra. Un target honesto de churn
      pediría una ventana más larga o condicionar por la cadencia propia de cada cliente.
- [ ] **Deuda: el modelo de propensión no tiene features de calendario** (detectada en la
      Fase 5, hallazgo 8 de `reports/insights/business_findings.md`). Por eso la política
      elige protector solar en noviembre, que vende 5,5 veces menos que en junio. El índice
      estacional ya se calcula en la Fase 2 (`src/eda/questions.py`, Q3): es meterlo en
      `src/nba/features.py`.

## Fase 5 — Empaquetado y storytelling

- [x] README principal con arquitectura, resultados y cómo reproducir
      · `README.md` cubre las Fases 0-7 con las cifras del dataset de la Fase 7 (7e), con
      secciones propias para la demo (Fase 6) y para el antes y el después de la fidelidad
      de marca (Fase 7). Cada cifra sale de un script: `verify_dataset`, `run_etl`,
      `recommender.pipeline`, `nba.pipeline`, `impact.pipeline`, `eda.findings` o el
      notebook re-ejecutado
- [x] Diagrama ER (Mermaid) del modelo relacional de las 7 tablas, con sus claves y
      relaciones, incluido en el README
      · sección "Modelo relacional" del `README.md`; las 14 relaciones están además
      verificadas empíricamente en `notebooks/01_eda.ipynb` (0 claves huérfanas)
- [x] Resumen de impacto de negocio (medio folio): NDCG@5 y uplift de NBA traducidos a
      impacto estimado (ej. cross-sell extra en €/mes)
      · `IMPACT.md`, que escribe `python -m src.impact.pipeline` · verificado por
      `tests/test_impact.py` · regenerado en la 7e: 289.430 €/año por 100.000 clientes
      y 100.000 cestas online/mes (antes 262.777 €), con las cifras previas congeladas en
      `reports/impact/baseline_fase5.json` · regenerado tras los puntos A1–A4 y M3–M4 del
      diagnóstico: **296.614 €/año** (el ranker acierta en el 54,2 % de las cestas)
- [x] Notebook o informe con los hallazgos de negocio (estilo "vistazo al análisis" del
      otro proyecto)
      · `reports/insights/business_findings.md` + 8 figuras, que escribe
      `python -m src.eda.findings` · regenerado en la 7e; el hallazgo 6 se reescribió porque
      su texto fijo ("el generador elige casi al azar") dejó de ser cierto, y ahora cuenta
      el antes y el después leyendo `reports/recommender/baseline_fase3.json`
- [x] Tests de las funciones de Tarea 1 y Tarea 2
      · `tests/test_eda_questions.py` (39, las 9 preguntas con respuesta a mano) y
      `tests/test_repurchase.py` (19)
- [x] (Opcional) MLflow para registrar experimentos del recomendador y de propensión —
      aporta un ángulo de MLOps/BI que no está en los otros dos proyectos
      · `src/tracking.py`, opcional y a prueba de fallos · verificado por
      `tests/test_tracking.py` (19)

## Fase 6a — Preparación visual del catálogo

Paso previo a la app, se ejecuta una sola vez con `python -m src.catalog.build_assets`.
Es la única parte del proyecto que necesita internet — el resultado se cachea en `assets/`
y a partir de ahí todo vuelve a ser local. La lógica vive en `src/catalog/` y está
verificada por `tests/test_catalog.py` (29 tests; 30 desde la 7e, que la rehízo sobre el
catálogo de 496 productos — ver más abajo).

- [x] `.env` con `PEXELS_API_KEY` (en `.gitignore`, nunca comiteado) + `.env.example` sin
      valores reales, comiteado como documentación de qué variable hace falta
      · `.gitignore` ignora `.env` y `.env.*` con la excepción `!.env.example` ·
      comprobado con `git ls-files`: `.env` **no** está trackeado, `.env.example` sí y va
      con la clave vacía
- [x] Analizar `products` (department, category, brand) y decidir si `category` ya es
      suficientemente granular o hace falta una columna `visual_group` nueva — ni tan
      amplia como el departamento ni tan específica como el SKU (p.ej. `leche_entera`,
      `yogur_griego`, `salmón`, no `Lácteos` ni un `product_id` concreto)
      · **`category` es el nivel correcto** (1.500 productos en 62 categorías, 24 por
      grupo de media): `department` son 8 valores demasiado amplios y `brand` son 144
      razones sociales de Faker sin aspecto propio. `visual_group` se mantiene como
      columna propia y no como alias porque es un slug ASCII usable como nombre de
      fichero y porque el mapa es 62 → 60, no 1:1 · razonamiento en el docstring de
      `src/catalog/visual_groups.py` y en `docs/VISUAL_CATALOG.md`
- [x] CSV `visual_group, search_term` (término de búsqueda en inglés, que es donde Pexels
      tiene mejor cobertura, aunque el resto del proyecto esté en español)
      · `assets/visual_groups.csv` · 60 filas
- [x] Revisar recuento de productos por `visual_group` y fusionar los grupos demasiado
      pequeños
      · `MIN_GROUP_SIZE = 15` y la lista `MERGES` en `src/catalog/visual_groups.py` · dos
      fusiones aplicadas (Bacalao → `pescado_blanco`, Limpiacristales →
      `limpiadores_hogar`), cada una con su motivo escrito · el criterio exige además una
      hermana que represente **el mismo objeto físico** en el mismo departamento, así que
      grupos pequeños como turrón, torrijas, cava, marisco o protector solar se quedan
      sin fusionar a propósito
- [x] Descargar vía la API de Pexels (`PEXELS_API_KEY` en variable de entorno, nunca
      hardcodeada) una imagen representativa por `visual_group` — priorizando fondo limpio
      o blanco, aspecto ecommerce/supermercado, sin personas, sin composiciones complejas,
      sin fotos de cocina o restaurante
      · `src/catalog/pexels.py` · `score_photo` **descarta** la foto si el texto
      alternativo contiene una palabra de `PEOPLE_WORDS` y puntúa el resto por brillo,
      vocabulario de estudio/producto, y penalización de escena, composición y
      panorámicas · búsqueda en cascada (`color=white` primero, luego sin filtro) · la
      foto elegida, su autor y su puntuación quedan en `assets/image_credits.csv`
- [x] Guardar las imágenes en `assets/` (`assets/leche_entera.jpg`, etc.)
      · 60 `.jpg`, uno por `visual_group` · `src/catalog/image_hash.py` usa una huella
      perceptual para que dos grupos no acaben con la misma foto
- [x] CSV final `product_id, product_name, visual_group, image_path` — `product_name` se
      deriva de department/category/brand (el dataset no tiene un nombre de producto
      propio; no hace falta tocar `DATA_SPEC.md` ni el generador para esto)
      · `assets/product_catalog.csv` · 1.500 filas, **0 con `image_path` vacío**
- [x] Comitear `assets/` y los CSV de mapeo — no se regeneran en cada ejecución, se tratan
      como un fixture cacheado (los resultados de búsqueda de Pexels no son reproducibles
      por semilla)
      · 63 ficheros trackeados en `assets/` (60 fotos + `visual_groups.csv`,
      `product_catalog.csv`, `image_credits.csv`)

Verificado además **de punta a punta sobre la app**: ejecutando `streamlit_app.py` con
`AppTest`, las fotos que sirve son byte-idénticas a los ficheros de `assets/` (13 fotos
distintas en la primera carga, 0 recodificadas, 0 inventadas).

## Fase 6b — Demo web interactiva

Es el entregable central del proyecto: la pieza que hace tangibles el recomendador (Fase 3)
y el NBA (Fase 4) para cualquiera que la abra, sin tener que leer métricas.

Se ejecuta con:

```bash
.venv/Scripts/streamlit run streamlit_app.py     # Windows
.venv/bin/streamlit run streamlit_app.py         # macOS / Linux
```

`streamlit_app.py` solo dibuja; la lógica vive en `src/serving/` (inferencia del
recomendador fuera de Spark) y `src/demo/` (catálogo, buscador y el motivo de cada
recomendación), que es lo que permite probarla sin levantar la app.

**Nota sobre cómo salió**: estaba planeada en tres versiones incrementales, una por sesión.
En la práctica las tres aterrizaron juntas en el commit `55116b3`, así que los checkboxes
de abajo se marcan sobre lo verificado, no sobre el orden en que se hizo.

### V1 — Carrito

- [x] Comprobar que `streamlit`, `python-dotenv` y `requests` están en `requirements.txt`
      (no estaban en el `requirements.txt` original de la Fase 0) e instalarlos; verificar
      que `streamlit run` arranca antes de seguir
      · `requests` y `python-dotenv` entraron con la Fase 6a; **`streamlit` faltaba en
      `requirements.txt` y en `constraints.txt`** y se añadió después, fijado a `1.63.0`
      con sus transitivas · `streamlit run` verificado: el servidor responde `200` en `/`
      y `ok` en `/_stcore/health`
- [x] Instalar la skill `developing-with-streamlit` (repo `streamlit/agent-skills`) en
      `.claude/skills/`
- [x] Tema propio en `.streamlit/config.toml` (colores, fuente) en vez del tema por defecto
      · paleta de supermercado (verde `#1a7f4b` de frescos, ámbar de promoción), fuente
      Inter y colores semánticos reutilizados por las insignias de la app · todo en tokens
      del tema y **sin CSS inyectado**, que apuntaría a clases internas de Streamlit
- [x] Selector de cliente: existente (`customer_id` real) o "nuevo simulado" (sin historial)
      · `st.segmented_control` Recurrente / Nuevo · en modo recurrente se ofrecen los
      clientes con más historial y se muestra su ficha (cestas, referencias, ticket medio)
- [x] Catálogo de productos visual: tarjetas con la foto real de su `visual_group` (Fase
      6a, vía `image_path`), nombre, categoría y precio
      · buscador sobre todo el catálogo (AND entre palabras, insensible a tildes) y
      navegación por departamento → categoría · el escaparate de un departamento enseña
      **una referencia por categoría**, porque la foto es una por `visual_group` y si no
      saldría la misma imagen quince veces
- [x] La cesta (cargada o simulada) se muestra con el mismo estilo de tarjeta que el
      catálogo
      · misma función `product_card`, cambiando solo el botón (Añadir / Quitar)
- [x] ~~Sin llamadas al recomendador ni al NBA todavía~~ — restricción de andamiaje de la
      V1, superada: la app llegó con las tres versiones a la vez

### V2 — Recomendaciones en vivo

- [x] Integrar el pipeline de candidatos + ranking (Fase 3, ya entrenado): al cambiar la
      cesta, se recalcula y muestra el top-5 de recomendaciones
      · `src/serving/recommend.py` reimplementa la inferencia **sin Spark** sobre el bundle
      de `data/serving/` · la equivalencia con el pipeline offline no se supone: la fija
      `tests/test_serving_parity.py` con 6 tests (mismas cestas, mismo top-5 producto a
      producto, mismo orden, mismos scores del booster, mismas fuentes atribuidas)
- [x] Las recomendaciones se muestran como tarjetas con foto (mismo estilo que el catálogo)
      y un motivo breve cuando se pueda derivar de las features del ranker (ej. "porque te
      toca reponerlo", "co-compra habitual con lo que llevas")
      · `explain()` en `src/demo/catalog.py` · el motivo sale de las **mismas columnas que
      el ranker usó para ordenar** (`cat_due`, `cat_overdue_ratio`, las cinco `src_*`), no
      de una racionalización escrita a posteriori · la reposición manda sobre la fuente
- [x] Verificar que los 4 perfiles de cliente funcionan correctamente (nuevo sin cesta,
      nuevo con cesta, recurrente sin cesta, recurrente con cesta)
      · comprobado sobre el bundle real: los 4 devuelven 5 recomendaciones con foto (5/5)
      y **activan fuentes progresivamente**, que es justo lo que la Fase 3 predecía:

      | Perfil | Fuentes con señal | Motivos que salen |
      | --- | --- | --- |
      | 1 · nuevo, sin cesta | `src_pop` | top ventas ahora |
      | 2 · nuevo, con cesta | `src_aff`, `src_cataff`, `src_pop` | va con tu cesta |
      | 3 · recurrente, sin cesta | `src_hist`, `src_pop` | lo compras a menudo · te toca reponerlo |
      | 4 · recurrente, con cesta | las cinco | encaja con tu cesta · lo compras a menudo · te toca reponerlo |

- [x] ~~Sigue sin NBA~~ — restricción de andamiaje, superada por el mismo motivo que en la V1

### V3 — Next Best Action

- [x] Cargar los modelos de propensión (Fase 4) y calcular la acción recomendada para el
      cliente activo
      · la app lee la tabla ya resuelta `predictions/nba_actions.parquet` (que produce
      `python -m src.nba.pipeline`) en vez de invocar los modelos en caliente: la política
      es determinista dado el corte, así que recalcularla por pulsación no cambiaría el
      resultado y sí la latencia
- [x] Banner de Next Best Action visualmente destacado (color/icono), no solo texto plano
      · `nba_banner()` · título con icono y color por acción, categoría objetivo, valor
      esperado en € y las dos probabilidades (compra 7 d, churn 4 sem) como métricas · un
      cliente nuevo no entra en la política y el banner lo dice explícitamente en vez de
      quedarse vacío
- [x] La demo completa (V1+V2+V3) solo hace inferencia sobre los modelos ya guardados en
      `models/` y las imágenes ya descargadas en `assets/` — no reentrena nada ni vuelve a
      llamar a Pexels
      · verificado ejecutando la app con `AppTest`: corre sin excepciones y las fotos que
      sirve son **byte-idénticas** a los ficheros de `assets/` (13 distintas en la primera
      carga, 0 recodificadas) · sin tráfico a Pexels: `src/catalog/` no se importa desde
      la app

## Fase 7 — Fidelidad de producto y comparación con Kaggle

Motivada por un hallazgo de uso real: en la demo, el hit-rate de SKU exacto (11,8%,
Fase 3) se ve como "0 de 5" en la mayoría de pruebas manuales, y da la sensación de que el
sistema no acierta nada — aunque a nivel de categoría sí acierta el 51,4% de las veces. El
objetivo es doble: (a) que el dato tenga más señal real a nivel de SKU (el generador
elegía casi al azar entre ~24 referencias por categoría, así que ningún modelo podía
predecir bien algo que era casi aleatorio por construcción — resuelto en 7a), y (b) que la
demo comunique el
acierto de forma honesta, mostrando categoría y SKU exacto por separado en vez de solo el
exacto.

Cambia el dataset (nuevos `product_id`, nuevos hashes), así que obliga a rehacer las Fases
2, 3, 4 y 6a sobre los datos nuevos. Se divide en sub-fases para que cada una quepa en una
sesión de la suscripción Pro.

### 7a — Generador: fidelidad de marca + catálogo reducido

Hecha. Todo lo que se reporta aquí lo recalcula `python -m data_generation.verify_dataset`
(secciones `fidelidad_de_sku`, `uplift_promocion` y `senal_churn`) y lo fijan cuatro tests
nuevos en `tests/test_generate_dataset.py`. El dataset regenerado tiene **`product_id` y
`sha256` nuevos**: las Fases 2, 3, 4 y 6a siguen colgando de los números viejos hasta que
se rehagan en 7b-7e.

- [x] Reducir el número de referencias por categoría de ~24 a un catálogo más curado
      · `catalog.PRODUCTS_PER_CATEGORY = 8`, el mismo surtido en todas las categorías:
      62 × 8 = **496 productos** (antes 1.500 repartidos en proporción a la popularidad de
      la categoría) · el reparto plano es deliberado: si las categorías pequeñas se
      quedaran en 2 referencias serían triviales de predecir y las grandes seguirían
      siendo ruido · verificado por `test_surtido_de_ocho_referencias_por_categoria`
- [x] Añadir fidelidad de marca/SKU: la primera compra de un cliente en una categoría le
      asigna una referencia "preferida"; las compras siguientes en esa categoría la repiten
      con una probabilidad alta, variable por tipo de categoría
      · `Category.loyalty` en `data_generation/catalog.py` (**40 categorías de hábito**
      0,75-0,85 y **22 exploratorias** 0,25-0,40; `validate_catalog()` rechaza cualquier
      valor fuera de las dos bandas) + reparto de cuota dentro de la categoría en
      `_generate_baskets_and_items` · la preferencia se sortea por popularidad en la
      primera compra y no se reasigna; la variedad sale del hueco `1 - loyalty` ·
      verificado por `test_fidelidad_de_marca_hace_que_el_cliente_repita_referencia` y
      `test_cada_categoria_declara_su_banda_de_lealtad`
- [x] Documentar el nuevo parámetro de lealtad y el catálogo reducido en `DATA_SPEC.md`
      · sección **"Fidelidad de marca (detalle)"** con la asignación categoría a categoría,
      el criterio de cada banda y el efecto medido; nota de tamaño de surtido bajo la tabla
      `products`; volumen de referencia actualizado a 496
- [x] Regenerar el dataset con la semilla fija y verificar que sigue siendo reproducible
      · dos ejecuciones completas con `seed=42` dan los mismos `sha256` en las 7 tablas
      (`basket_items` `e84c387f…`, `products` `5c180e8a…`) · 3.103.685 líneas sobre 600.174
      cestas · fijado también por `test_dos_ejecuciones_producen_el_mismo_hash`
- [x] Verificar que la repetición de SKU dentro de categoría subió de forma clara
      · **0,850 → 0,440 referencias distintas por compra** (−48 %) y la cuota de la
      referencia favorita **0,301 → 0,697**, sobre los 357.195 pares cliente-categoría con
      tres o más compras. Por banda: hábito **0,351** distintas y **0,840** de cuota
      favorita —justo dentro de la banda declarada—, exploración 0,571 y 0,485. El 0,86 que
      documentaba la Fase 3 se recalcula ahora como 0,850 con la misma definición

#### Efectos colaterales medidos

- **El uplift de promoción observado baja de 2,99 a 1,97.** Es consecuencia directa, no un
  fallo: el multiplicador sigue siendo exactamente `PROMO_UPLIFT` pero se aplica *encima*
  del reparto por hábito, así que una promoción solo puede llevarse la parte no fiel de la
  categoría. Es además lo que pasa en gran consumo real. Queda documentado en `DATA_SPEC.md`.
- **La rampa de churn sigue intacta** (frecuencia 0,79 frente a 0,985 de los activos;
  ticket 0,972 frente a 1,012), pero la comprobación del ticket a `TEST_SCALE` resultó ser
  ruido: con ~170 churners el estimador se mueve varios puntos según la semilla y con la 42
  llegaba a invertir el orden. Se partió en dos tests: la frecuencia se sigue comprobando a
  `TEST_SCALE` y el ticket pasa a `dataset_dir_grande` (~700 churners), donde el orden sale
  estable con cualquier semilla.
- Afinidad de cesta, estacionalidad, ciclos de reposición e integridad referencial no se
  mueven: los diez pares de `DATA_SPEC.md` siguen con su lift y la correlación de rangos de
  los ciclos es 0,992.

### 7b — Rehacer la Fase 2 (ETL) sobre el dataset nuevo

Hecha. Se re-ejecuta entera con `python -m src.etl.run_etl` (234 s en local), que reescribe
`data/processed/` y `reports/etl/`. No hizo falta tocar una línea del ETL: el cambio de la
7a no altera el esquema, así que la misma lógica corre sobre los datos nuevos tal cual.

- [x] Volver a ejecutar limpieza, Data Trust Score, RFM, `due_for_repurchase` y afinidad de
      cesta sobre el dataset regenerado
      · **Data Trust Score 91,42 (C) → 100,00 (A)**, con 9 de 79 comprobaciones fallando
      sobre el crudo (antes 10 de 79 y 89,99). La nota del crudo sube 1,43 puntos por una
      sola razón: `customers.signup_within_period` pasa de 3 filas malas a 0. No es un
      defecto que se haya dejado de inyectar, es un **colateral** del que sí se inyecta —
      al mover el alta de 58 clientes a "unos días después de su primera compra", antes
      3 de esos saltos caían más allá del 2025-12-31 y con el reparto nuevo ninguno lo
      hace. Los 58 fallos de `signup_before_first_purchase` siguen ahí. El limpio vuelve a
      dar 100,00 en las cinco dimensiones · `reports/etl/data_trust.md`
      · **Limpieza**: mismos recuentos salvo donde el catálogo encogió. `basket_items`
      3.103.685 → 3.057.825 (45.860 duplicados, 1,478 %; 12.314 cantidades negativas,
      0,397 % — antes 45.859 y 12.308). `products` baja de 1.500 a 496 filas, así que sus
      defectos escalan con ella: 20 grafías de categoría (4,03 %, antes 60 sobre 1.500 =
      4,00 %) y 5 marcas nulas (1,01 %, antes 17 = 1,13 %). `baskets` 1.192 importes
      recalculados, idéntico. `customers` 154 ciudades nulas y 58 altas corregidas (antes
      163 y 58). `session_events` 79 duplicados sobre 897.674 (antes 33 sobre 900.143) ·
      `reports/etl/cleaning_report.md`
      · **Features**: `rfm` 20.000, `repurchase_features` 608.885 con el **52,4 %** de los
      pares vencidos (mismas cifras que antes), `affinity_category` 3.780 pares y
      `affinity_product` 8.438 (antes 9.281 — son menos consecuentes porque hay 496
      productos en vez de 1.500, no porque se haya perdido señal)
      · los 110 tests de `test_cleaning.py`, `test_data_trust.py`, `test_rfm.py`,
      `test_repurchase.py`, `test_affinity.py` y `test_schemas.py` siguen pasando
- [x] Comprobar que los 10 pares de afinidad de `DATA_SPEC.md` siguen saliendo con lift
      alto (no debería haber cambiado mucho, pero verificarlo)
      · **no se ha movido nada**: `reports/etl/affinity_expected_pairs.md` sale
      byte-idéntico al de la Fase 2 original, con los mismos lift, soporte y confianza
      hasta el segundo decimal. Era lo esperable — la fidelidad de marca de la 7a decide
      *qué referencia* se lleva el cliente dentro de una categoría ya elegida, y la
      afinidad se mide entre categorías, así que la señal vive en una capa que el cambio
      no toca
      · comprobación adicional sobre la tabla nueva: cada uno de los 10 consecuentes es el
      **puesto 1 por lift** de sus 61 candidatos para esa categoría disparadora. El rango
      va de 1,84 (Pan → Embutido) a 13,39 (Pañales → Toallitas), y los desvíos frente al
      objetivo son los mismos de siempre (los pares de categorías grandes se quedan por
      debajo porque compiten con el resto de la cesta, Pañales se dispara porque su
      marginal es minúscula)

**README, al final de la Fase 7 (hecho).** El `README.md` citaba las cifras viejas (1.500
productos, Data Trust 89,99, 10 de 79), y no se tocó aquí porque también citaba las de las
Fases 3, 4 y 6. Se actualizó de una vez tras la 7e. Al hacerlo aparecieron cuatro piezas
que tampoco se habían rehecho y que el README cita: `IMPACT.md`, el informe de hallazgos,
el notebook de EDA y `docs/CLEANING.md`. Las tres primeras se regeneraron con su script
(el notebook, re-ejecutado entero con `jupyter nbconvert --execute`, y sus 6 celdas de
texto con cifras escritas a mano, corregidas) y las cifras de `docs/CLEANING.md` se
recalcularon sobre `data/raw/`. El filtro IQR que cita no tenía script; ahora el documento
dice qué valla usa (`Q3 + 3·IQR`: 3.477 cestas).

### 7c — Rehacer la Fase 3 (recomendador): reentrenar, F1@5 y comparación con Kaggle

Hecha. Se re-ejecuta entera con `python -m src.recommender.pipeline` (23 min en local), que
reescribe `models/recommender_ranker_lgbm.txt`, `predictions/recommend*.parquet` y
`reports/recommender/`. Todas las cifras de abajo salen de `reports/recommender/metrics.md`,
incluida la comparación con la Fase 3, que el pipeline recalcula contra
`reports/recommender/baseline_fase3.json` (el `metrics.json` de la Fase 3 congelado con
`git show b3c29ba:reports/recommender/metrics.json`).

- [x] Reentrenar el pipeline completo (candidatos + ranker) sobre el dataset nuevo
      · mismo código, mismas ventanas, mismos tamaños de muestra (14.500 queries de
      ranker, 18.000 de test) · LambdaRank de **87 árboles** (antes 84), NDCG@5 de
      validación 0,2166 (antes 0,0836) · pool de **139 candidatos** por cesta sobre 496 productos
      (antes 155 sobre 1.500), `pool_recall` **31,1 % → 76,9 %** · los topes de
      `CandidateConfig` **no se han reajustado** al catálogo nuevo
      · derivados regenerados en la misma pasada: `reports/recommender/profiles_demo.md`
      (`python -m src.recommender.demo_profiles`) y el bundle de `data/serving/`
      (`python -m src.serving.export_bundle`), para que `tests/test_serving_parity.py`
      compare contra el modelo nuevo
- [x] Añadir Precision@5 y F1@5 a la evaluación (ya se tiene Recall@5; F1@5 es su media
      armónica con Precision@5)
      · `src/recommender/evaluate.py` · `precision@5` por cesta (aciertos / 5) y dos
      variantes de F1: **`f1@5`**, la media armónica de Precision@5 y Recall@5 medios (la
      definición de esta fase, cifra de cabecera), y **`f1@5_por_cesta`**, la media del F1
      de cada cesta, que es como agregaba Instacart · verificado por
      `test_precision_y_f1_contra_el_calculo_a_mano` en `tests/test_recommender.py`, que
      además fija un caso donde las dos variantes no coinciden
      · **Precision@5 = 0,1282 · F1@5 = 0,1454 · F1@5 por cesta = 0,1367**
- [x] Comparar el F1@5 con el 1er puesto de la competición de Kaggle "Instacart Market
      Basket Analysis" (F1 ≈ 0,41) — documentando explícitamente que no es una comparación
      100% equivalente: Instacart predice solo recompras con un conjunto de tamaño variable
      optimizado por F1-maximization, mientras que aquí se predice un top-5 fijo que mezcla
      recompra con descubrimiento (popularidad/co-compra/ALS) — dejarlo dicho en el
      informe, no solo el número
      · sección **"F1@5 frente a Kaggle"** de `reports/recommender/metrics.md`: 0,1454
      (0,1367 por cesta) frente a ≈ 0,41, con la salvedad escrita al lado de la tabla.
      Además de las dos diferencias pedidas (solo recompra frente a recompra +
      descubrimiento; tamaño variable con F1-maximization frente a top-5 fijo), recoge
      otras tres: el ranker optimiza NDCG y no F1, aquí se predice a mitad de cesta, e
      Instacart promediaba el F1 por pedido
- [x] Verificar si el SKU-hit_rate@5, NDCG@5 y el ratio SKU/categoría mejoraron de forma
      clara frente a los números viejos (11,8% / 0,0343 / 51,4% categoría vs 11,8% SKU)
      · **sí, con claridad**:

      | Métrica | Fase 3 | Fase 7c | Cambio |
      | --- | ---: | ---: | ---: |
      | NDCG@5 | 0,0343 | **0,1752** | ×5,1 |
      | Recall@5 | 0,0332 | **0,1678** | ×5,1 |
      | Precision@5 | 0,0251 | **0,1282** | ×5,1 |
      | F1@5 | 0,0286 | **0,1454** | ×5,1 |
      | hit_rate@5 (SKU) | 11,8 % | **49,5 %** | ×4,2 |
      | hit_rate@5 (categoría) | 51,4 % | **60,0 %** | ×1,2 |
      | SKU / categoría (hit_rate) | 23,0 % | **82,4 %** | ×3,6 |
      | SKU / categoría (precision) | 13,7 % | **72,7 %** | ×5,3 |

      La brecha categoría/SKU casi se cierra: la categoría apenas mejora y el SKU se
      multiplica, que es exactamente el efecto que buscaba la 7a. Mejoran los cuatro
      perfiles, incluido el cold-start (perfil 1: NDCG@5 0,0212 → 0,1185)

#### Lo que hay que tener en cuenta al leer la mejora

- **Parte de la subida es el catálogo, no el modelo.** Con 496 productos en vez de 1.500,
  cualquier ordenación acierta más. El baseline de popularidad sobre el mismo pool pasa de
  0,0200 a **0,0868** de NDCG@5 (×4,3). Lo que sí es mérito del ranker es la distancia
  a ese baseline, que crece de **×1,71 a ×2,02**.
- **El sistema se apoya ahora en el historial personal.** En los perfiles recurrentes, la
  fuente `hist` está detrás del 98–99 % de las recomendaciones (antes el 42–49 %;
  `profiles_demo.md`). Es la consecuencia natural de la fidelidad de marca.
- **La señal de sesión pesa menos.** Pasa de +13,2 % a **+5,1 %** de NDCG@5 sobre la
  ablación sin sesión. Con el historial prediciendo bien la referencia, lo que el cliente
  mira en la web aporta menos información nueva. La cifra de +13 % de la Fase 3 queda
  como histórica.
- **El F1@5 no está cerca de 0,41, y no debería leerse como un fallo.** Aparte de las
  diferencias de planteamiento, un top-5 fijo contra 3,9 productos por adivinar de media
  pone techo a Precision y Recall a la vez.

### 7d — Rehacer la Fase 4 (NBA) sobre el dataset nuevo

Hecha. Se re-ejecuta entera con `python -m src.nba.pipeline` (6 min en local), que
reescribe `models/nba_*_lgbm.txt`, `predictions/nba_actions.parquet` y `reports/nba/`.
La comparación con la Fase 4 la recalcula el propio pipeline (sección **"Frente a la Fase 4
original"** de `reports/nba/metrics.md`) contra `reports/nba/baseline_fase4.json`, el
`metrics.json` de la Fase 4 congelado con `git show 2327f25:reports/nba/metrics.json`.

- [x] Reentrenar los dos modelos de propensión y recalcular la política de valor esperado
      sobre el dataset regenerado
      · mismo código, mismos cortes y mismos supuestos · churn con **17 árboles** (antes
      15), compra en categoría con **97** (antes 67) · mismas filas que antes: 34.557 de
      entrenamiento de churn, 726.764 pares de categoría, 18.729 clientes y 370.300 pares
      en test — la 7a cambia *qué referencia* se compra, no cuándo ni en qué categoría,
      así que las cabeceras de cesta y las etiquetas no se mueven · la comprobación de
      cordura contra `churn_label` sigue en **83,4 %** · los 73 tests de `test_nba.py`,
      `test_tracking.py`, `test_impact.py` y `test_demo_baskets.py` siguen pasando
- [x] Comprobar que las conclusiones de la Fase 4 (AUC, el barrido de sensibilidad de
      `churn_reduction`) se mantienen razonablemente estables — si cambian mucho, avisar
      antes de darlo por bueno
      · **estables, sin desvíos notables**:

      | Métrica | Fase 4 | Fase 7d |
      | --- | ---: | ---: |
      | Churn 4 semanas: AUC / PR-AUC | 0,8531 / 0,8556 | **0,8527 / 0,8550** |
      | Compra categoría 7 días: AUC / PR-AUC | 0,7634 / 0,2190 | **0,7630 / 0,2185** |
      | Política frente a no actuar | 4.012 € | **3.938 €** (−1,9 %) |
      | Ventaja sobre la mejor alternativa trivial | 3.082 € | **3.030 €** |
      | Clientes con acción | 75,6 % | **75,2 %** |

      | `churn_reduction` | Fase 4 | Fase 7d | Cupones 7d |
      | ---: | ---: | ---: | ---: |
      | 0,00 | 1.210 € | **1.189 €** | 0 % |
      | 0,05 | 1.734 € | **1.706 €** | 43,7 % |
      | 0,10 | 4.012 € | **3.938 €** | 66,2 % |
      | 0,20 | 8.862 € | **8.662 €** | 72,0 % |

      Las conclusiones de la Fase 4 se sostienen tal cual: con `churn_reduction = 0` la
      política sigue ganando (1.189 € frente a 907 € de "recomendar siempre") y deja de
      repartir cupones, y las tres features dominantes del churn siguen siendo
      `n_baskets_90d`, `avg_days_between_baskets` y `recency_days`. La pequeña bajada de
      valor se ve también en "cupón a todos" (−1.430 € → −1.592 €) y "recomendar siempre"
      (930 € → 907 €): afecta a todas las políticas, no solo a la de valor esperado. Era lo esperable: el NBA trabaja a nivel de cliente y
      categoría, no de SKU

### 7e — Rehacer la Fase 6a y arreglar la Fase 6b

Hecha. Se verifica con `pytest tests/test_catalog.py tests/test_demo_hits.py
tests/test_demo_baskets.py tests/test_serving_parity.py` (59 tests, ninguno saltado) y
arrancando la demo. Con esto, y con el `README.md` ya actualizado (ver la nota de la 7b),
**la Fase 7 queda cerrada**.

- [x] Rehacer la Fase 6a (los `product_id` cambiaron): revisar `visual_group`/`search_term`
      y regenerar el CSV de imágenes — reutilizar `assets/` existente donde el
      `visual_group` no haya cambiado, para no volver a llamar a Pexels de más
      · **los 60 `visual_group` y sus términos siguen valiendo**: las 62 categorías no
      cambiaron y el mapa va por categoría, así que `visual_groups.csv`, las 60 fotos e
      `image_credits.csv` quedan byte-idénticos · **0 llamadas a Pexels**:
      `python -m src.catalog.build_assets --offline` · `assets/product_catalog.csv` pasa
      de 1.500 a **496 filas, 0 sin foto**, 10 nombres desempatados con ordinal (antes 177)
      · **el CSV viejo rompía la demo sin avisar**: con ids correlativos seguía cruzando con
      los 496 nuevos, pero solo el 1,6 % caía en su grupo, así que se pintaban nombres y
      fotos de otra categoría. Ahora lo impiden `check_catalog_matches` (la demo falla con
      el comando que lo arregla; comprobado contra el CSV del commit `e719f62`) y
      `test_el_csv_final_corresponde_al_catalogo_actual`
      · con 8 referencias por categoría, `MIN_GROUP_SIZE = 15` ya no discrimina (58 de 60
      grupos quedan por debajo); las dos fusiones se mantienen por el criterio de mismo
      objeto físico, y se deja escrito en `visual_groups.py` y `docs/VISUAL_CATALOG.md`
- [x] En la demo (Fase 6b), cambiar el indicador de acierto: en vez de solo "X de 5
      recomendaciones estaban en lo que añadiste después" (SKU exacto), mostrar también el
      acierto de categoría por separado — algo como "X de 5 acertaron la categoría, de esas
      Y acertaron el producto exacto". No ocultar el número de SKU exacto, solo dar más
      contexto honesto junto a él
      · `score_hits` / `HitSummary` en `src/demo/baskets.py`, con **la misma definición que
      `category_metrics`** del informe · la frase es "**X de 5** acertaron la categoría de
      lo que el cliente añadió después; de esas, **Y** eran el producto exacto", con los
      dos números en negrita · un acierto exacto se cuenta siempre, aunque faltara su
      categoría en el mapa, así que el número de SKU no puede perderse por el camino
      · las tarjetas distinguen "lo compró de verdad" (verde) de "categoría acertada"
      (amarillo), y el desplegable del target marca lo mismo desde el otro lado
      · debajo, el acierto medio del test (**0,88 de 5** por categoría y **0,64** exacto;
      **60,0 %** y **49,5 %** de cestas con algún acierto), leído de
      `reports/recommender/metrics.json` por `load_reference_hit_rates` en vez de escrito
      a mano. Sustituye al texto que citaba las cifras de la Fase 3 (11,8 % / 51,4 %)
      · verificado por los 12 tests de `tests/test_demo_hits.py`
- [x] Re-apuntar la demo a los modelos y al CSV de imágenes nuevos
      · la app lee rutas fijas que las 7c y 7d ya sobrescribieron
      (`models/recommender_ranker_lgbm.txt`, `data/serving/` con 496 productos indexados,
      `predictions/nba_actions.parquet`), así que no hizo falta cambiar ninguna ruta; lo que
      estaba desfasado era el CSV de imágenes · la paridad del serving con el ranker nuevo
      la sigue fijando `tests/test_serving_parity.py`
- [x] Comprobar que `streamlit run` arranca sin errores
      · `AppTest` cargando 18 cestas reales de 6 clientes: 0 excepciones, y salen los tres
      casos (aciertos exactos, solo de categoría y ninguno) · `streamlit run` real:
      `/_stcore/health` → `ok`, `/` → `200`, y abierta en Chromium sin cabeza: 0
      excepciones de Streamlit, 0 errores de JavaScript, las fotos de las tarjetas casan
      con la categoría del producto

## Fase 8 — Estructura de la cesta: misiones, cola larga y sustitución

Puntos **A5**, **M1** y **M2** de `docs/diagnostico-fase7.md`. El diagnóstico dejó claro que
el generador casi no ponía estructura *dentro* de la cesta: solo 40 de los 3.780 pares de
categorías tenían lift > 1,5, el tamaño era una Poisson recortada a 20 líneas y cada cesta
llevaba exactamente una línea por categoría. Con eso, "dado lo que llevo, qué viene
después" tenía poco que aprender. Como cada regeneración obliga a rehacer las Fases 2-6,
los tres cambios van juntos en una sola pasada.

Antes de tocar nada se congeló el estado completo en `snapshots/pre-fase-8/` (dataset,
`data/processed`, oráculo, bundle de la demo, modelos, predicciones e informes), con un
`MANIFEST.json` de hashes, y las métricas en `baseline_pre_fase8.json` del recomendador,
del NBA y del impacto. Los tres guardan además los hashes de las 7 tablas del dataset con
las que se midieron: los informes solo pintan las comparaciones antes/después de A1, A2 y
A4 cuando el snapshot corresponde al dataset en uso, para no comparar cifras de datasets
distintos sin darse cuenta.

### 8a — Generador

Hecha. Todo lo que se reporta aquí lo recalcula `python -m data_generation.verify_dataset`
(secciones `tamano_de_cesta`, `lineas_por_categoria`, `coocurrencia_de_categorias` y
`propension_marca_blanca`, volcadas en `reports/etl/verify_dataset.json`) y lo fijan seis
tests nuevos en `tests/test_generate_dataset.py`. El dataset regenerado tiene **nuevos
`sha256` en `baskets`, `basket_items`, `sessions` y `session_events`**; `products`,
`customers` y `promotions` salen idénticos, así que la Fase 6a (fotos y catálogo visual)
no hubo que rehacerla.

- [x] **Misiones de compra como variable latente** (A5)
      · `catalog.MISSIONS`: 7 misiones (compra semanal, reposición, desayuno, limpieza e
      higiene, cena o aperitivo, bebé, fiesta estacional), cada una con su perfil de
      categorías y su distribución de tamaño · la probabilidad de cada misión depende del
      hogar, del canal, del *gate* de bebé, del día de la semana y del mes, y cada cliente
      tiene su propio reparto (Dirichlet de concentración 20) · las cestas anónimas usan el
      reparto medio de la población
- [x] **Tabla de complementarios ampliada** (A5)
      · de 10 a **30 pares** (`catalog.COMPLEMENT_PAIRS`): desayuno, higiene, comida de
      mascota y recetas · una categoría puede disparar varias asociadas
      (harina → huevos y azúcar) · el multiplicador aplicado no cambia (`AFFINITY_CALIBRATION
      = 3,8`)
- [x] **Objetivos escritos antes de tocar el código**
      · `DATA_SPEC.md`, sección "Estructura de la cesta: qué co-ocurrencia se busca y por
      qué", con el punto de partida, cómo es una cesta real, los rangos objetivo y lo que
      no debe moverse · tres objetivos se revisaron **después** de medir (el estimador del
      lift controlado, la mediana del lift crudo y los sustitutos que comparten misión), y
      queda escrito por qué en esa misma sección
- [x] **Tamaño de cesta con cola larga** (M1)
      · `1 + BinomialNegativa(media de la misión × factor de hogar y churn, r)`, tope de 50
      categorías en vez de 20 líneas

      | | Fase 7a | Fase 8 | Objetivo |
      | --- | ---: | ---: | :---: |
      | Líneas por cesta: media | 5,08 | **7,13** | 7-10 |
      | Coeficiente de variación | 0,42 | **1,04** | 0,75-1,10 |
      | p50 / p90 / p99 / máximo | 5 / 8 / 11 / 19 | **5 / 17 / 36 / 58** | p99 30-50 |
      | Cestas de 1 a 3 líneas | 24,6 % | **37,9 %** | 25-40 % |
      | Cestas de más de 20 líneas | 0,00 % | **6,6 %** | 4-12 % |

- [x] **Sustitución, segunda referencia y marca blanca** (M2)
      · 7 grupos de sustitución (`catalog.SUBSTITUTION_GROUPS`, factores 0,25-0,35) ·
      segunda referencia con probabilidad 0,12 en las categorías de exploración
      (**11,9 %** de las categorías de exploración de una cesta acaban con dos referencias,
      0 % en las de hábito) · propensión a marca blanca por cliente `LogNormal(0; 0,8)`:
      cuota p10-p90 entre clientes **0,12-0,44** (antes 0,17-0,33), con una varianza entre
      clientes 11 veces la del azar (antes 2,2)
      · los sustitutos que no comparten misión bajan directamente (agua → refrescos, lift
      controlado **0,43**; pollo → pescado **0,47**); los que sí la comparten se verifican
      contra el contrafactual: el mismo generador sin grupos los deja entre 1,8 y 3,2 veces
      más juntos (`test_la_sustitucion_resta_frente_a_no_tenerla`)
- [x] **Estructura de la cesta medida** (A5)

      | | Fase 7a | Fase 8 | Objetivo |
      | --- | ---: | ---: | :---: |
      | Pares con lift crudo > 1,5 (de 3.782) | 40 | **3.372** | muchos más de 40 |
      | Pares con lift controlado por tamaño > 1,5 | 42 | **404** | 150-450 |
      | Mediana del lift crudo | 1,00 | **2,45** | se reporta |

      El lift crudo se dispara porque con cestas de tamaño muy variable *cualquier* par
      co-ocurre más que por azar: en una compra semanal está casi todo. Por eso el objetivo
      se fijó sobre el lift **controlado por tamaño** (cada par dentro de su banda de
      tamaño, ponderando bandas por número de cestas), que es el que dice si hay estructura
      de verdad. Los dos los publica `reports/etl/affinity_expected_pairs.md` (crudo, desde
      `affinity_category`) y `verify_dataset` (controlado).
- [x] **Reproducibilidad byte a byte**
      · dos ejecuciones completas con `seed = 42` dan los mismos `sha256` en las 7 tablas
      (`basket_items` `7317aff5…`, `products` `5c180e8a…`) · el registrador del oráculo
      sigue sin cambiar ni un byte: `export_oracle` valida los hashes contra
      `data/raw/manifest.json` antes de escribir
- [x] **Lo que no se ha tocado**
      · identidad y cadencia del cliente, ciclo de reposición (correlación de rangos
      spec/observado **0,985**, antes 0,992), fidelidad de marca (cuota de la referencia
      favorita en hábito **0,839**, dentro de la banda declarada 0,75-0,85), embudo de
      sesión, estacionalidad y churn progresivo (ratio de frecuencia 0,79 en los que
      abandonan frente a 0,985 en los activos) · el uplift de promoción observado baja de
      1,97 a **1,88**: las cestas grandes diluyen el efecto, igual que en la realidad

### 8b — Rehacer la Fase 2 (ETL) sobre el dataset nuevo

Hecha. Se re-ejecuta entera con `python -m src.etl.run_etl` (6 min en local). No hizo falta
tocar la lógica: el cambio de la 8a no altera el esquema.

- [x] Limpieza, Data Trust Score, RFM, `due_for_repurchase` y afinidad sobre el dataset
      regenerado
      · **Data Trust Score 91,38 (C) → 100,00 (A)** (antes 91,42 → 100,00), con las mismas
      9 de 79 comprobaciones fallando en el crudo · `basket_items` 4.357.007 → 4.292.495
      tras quitar 64.512 duplicados (1,48 %); 17.253 cantidades negativas corregidas
      (0,40 %) · `baskets` 1.208 importes recalculados · `affinity_category` **3.782 pares**
      (antes 3.780: ahora todos los pares llegan al soporte mínimo)
- [x] **Un hallazgo del ETL que la cola larga hace más fuerte.** La valla de valores
      extremos sobre `total_amount` (`Q3 + 3·IQR`) pasa de marcar 3.477 cestas (0,58 %) a
      **21.318 (3,55 %)**, con 1.208 outliers inyectados: 17 de cada 18 serían compras
      grandes legítimas. Es el argumento de `docs/CLEANING.md` para recalcular el importe
      desde el detalle en vez de filtrar por umbral, y ahora se ve mucho mejor
- [x] Publicar el lift medido en `affinity_category`
      · `reports/etl/affinity_expected_pairs.md` trae los **30 complementos declarados** y
      una sección nueva de **estructura global**: 3.372 de 3.782 pares ordenados (89,2 %)
      con lift > 1,5, 1.282 por encima de 3, mediana 2,45 · el matiz del tamaño de cesta y
      el lift controlado están en `verify_dataset` y en `DATA_SPEC.md`

### 8c — Rehacer la Fase 3 (recomendador)

Hecha. `python -m src.recommender.pipeline` (33 min) y
`python -m src.recommender.verify_recommender_diagnostics` (6 min, con
`python -m data_generation.export_oracle` antes, 14 min). Todas las cifras salen de
`reports/recommender/metrics.md`.

- [x] Reentrenar candidatos y ranker sobre el dataset nuevo
      · mismo código y mismas ventanas · LambdaRank de **168 árboles** (antes 70), NDCG@5
      graduada de validación **0,3377** (antes 0,2503) · pool de 141,4 candidatos por cesta
      con `pool_recall` **77,8 % → 82,5 %**
- [x] Comparar contra el snapshot pre-Fase 8

      | Métrica | Fase 7 (dataset 7a) | Fase 8 | Cambio |
      | --- | ---: | ---: | ---: |
      | NDCG@5 graduada | 0,2148 | **0,3015** | ×1,40 |
      | `cat_hit_rate@5` | 0,7181 | **0,7954** | +7,73 pp |
      | `sku_hit_rate@5` | 0,5416 | **0,6157** | +7,41 pp |
      | F1@5 | 0,1649 | **0,2165** | ×1,31 |
      | Productos por adivinar (media) | 3,93 | 5,55 | ×1,41 |

      **Cuidado al leerlo:** son cestas distintas de datasets distintos y el target es más
      grande, así que una parte de la subida es mecánica. Lo que sí es comparable es el
      **% del techo teórico**, y ahí el LambdaRank pasa del **91,9 % al 94,9 %** en
      categoría (y del 91,5 % al 92,2 % en SKU).
- [x] Comprobar que la estructura nueva es aprendible desde la cesta (objetivo de A5)
      · el perfil 2 (cliente nuevo **con carrito**), que era el peor con diferencia, pasa
      de **0,4466 a 0,7088** de `cat_hit_rate@5`, y el perfil 4 (recurrente con carrito) de
      0,6648 a 0,7932 · el baseline de **reglas de asociación** —el que solo mira lo que
      hay en el carrito— pasa del **78,0 % al 90,6 % del techo**, y se convierte en el mejor
      baseline: antes la co-ocurrencia no enseñaba nada
- [x] Techo teórico con el oráculo rehecho
      · el sorteo ya no es una carrera de relojes con pares disparadora → asociada, así que
      `src/recommender/oracle.py` pasa a **muestreo secuencial por importancia**: replica el
      sorteo paso a paso (misión latente incluida, con su posterior) forzando que el carrito
      quede dentro · lo valida un test contra una simulación por fuerza bruta del generador
      (7 casos: sustitutos, cadenas y tope de tamaño) y, sobre el dataset real, la
      comprobación de que el techo realizado coincide con el esperado (máx |z| = 2,26 en
      20 contrastes)
      · techo `cat_hit_rate@5` **0,8382** (antes 0,7816); LambdaRank **+3,61 pp
      [+3,04, +4,17]** sobre el mejor baseline, p < 0,001
- [x] Evaluación robusta sobre el dataset nuevo
      · cortes por cesta: 12.947 queries de 1.800 cestas (`cuts.md`) · cold-start
      sobremuestreado: 3.140 cestas / 5.802 queries, `cat_hit_rate@5` **0,5289 → 0,7192**
- [x] **Un fallo de paridad de paso.** `tests/test_serving_parity.py` empezó a fallar por
      0,033 de score en **1 de cada 10.000** filas, con el top-5 idéntico. Las 63 features
      eran iguales hasta el último decimal: lo que cambiaba era el tipo. El pipeline
      castea la matriz a `float32` en Spark (`ranker.collect_for_ranking`) y el serving
      puntuaba en `float64`, así que un valor un epsilon por encima del umbral de un corte
      caía al otro lado al redondear. Con más árboles (168) y más datos acabó por aflorar.
      Arreglado en `serving.score_matrix`, que ahora pasa las features en `float32`

### 8d — Rehacer la Fase 4 (NBA)

Hecha. `python -m src.nba.pipeline` (6,5 min). La comparación la recalcula el propio
pipeline en la sección "Frente al dataset anterior a la Fase 8" de
`reports/nba/metrics.md`, contra `reports/nba/baseline_pre_fase8.json`.

- [x] Reentrenar los dos modelos de propensión y recalcular la política
      · churn con **44 árboles** (antes 17), compra en categoría con 93 (antes 97) · mismas
      cestas y mismos cortes: la Fase 8 cambia *qué* lleva cada cesta, no cuándo compra
      cada cliente, así que el maestro (18.729 clientes en test) no se mueve
- [x] Comprobar que las conclusiones se mantienen

      | Métrica | Fase 7d | Fase 8 |
      | --- | ---: | ---: |
      | Churn 4 semanas: AUC / PR-AUC | 0,8527 / 0,8550 | **0,8532 / 0,8576** |
      | Compra categoría 7 días: AUC / PR-AUC | 0,7630 / 0,2185 | **0,7591 / 0,2479** |
      | Tasa base de compra en categoría | 0,0619 | **0,0755** |
      | Política frente a no actuar | 3.938 € | **3.842 €** |
      | Clientes con acción | 75,2 % | **74,6 %** |

      Las conclusiones aguantan. Dos matices que sí cambian y tienen explicación: la tasa
      base de compra en categoría sube (más categorías por cesta ⇒ más pares positivos), lo
      que empuja la PR-AUC arriba y el AUC ligeramente abajo; y "cupón a todos" empeora de
      −1.591 € a **−3.745 €**, porque con más compras hay más cupones redimidos y el
      descuento sigue siendo mayor que el margen que persigue. La política de valor
      esperado, que es la que decide a quién, apenas se mueve.

### 8e — Impacto, hallazgos y demo

- [x] `python -m src.impact.pipeline` → `IMPACT.md` y `reports/impact/impact.json`
      · **279.114 €/año** por cada 100.000 clientes y 100.000 cestas online al mes (antes
      286.612 €)
      · **El cross-sell baja de 44.327 € a 32.964 €/año aunque el recomendador acierta
      más**, y es la lectura más interesante de la fase: el impacto se mide como
      *incremental sobre la popularidad*, y en un dataset con estructura la popularidad
      también acierta mucho más (`hit_rate` de SKU **0,270 → 0,408**). El acierto absoluto
      del modelo sube (0,542 → 0,616) y la distancia se estrecha. Con datos más realistas,
      el mérito atribuible al ranker es menor de lo que parecía
- [x] `python -m src.eda.findings` → informe de hallazgos y 8 figuras regeneradas
- [x] Demo (Fases 6a y 6b) · `products.csv` sale **idéntico** (el generador no toca el
      catálogo), así que las 60 fotos, `visual_groups.csv` e `image_credits.csv` siguen
      valiendo y no hubo ni una llamada a Pexels · la demo lee el bundle nuevo
      (`python -m src.serving.export_bundle`) y las cifras de referencia de
      `reports/recommender/metrics.json`

## Fuera de alcance (por ahora)

- Dashboard en Power BI (Tarea 4 de `CHALLENGE.md`) — el foco pasa a la demo de la Fase 6;
  se retoma más adelante si el proyecto da más de sí
- Servir el modelo en producción/cloud como API pública (la demo de la Fase 6 es solo local)
- Multi-país (se queda en España para esta primera versión)
- Optimización de precios/promociones (posible fase futura si el proyecto da más de sí)

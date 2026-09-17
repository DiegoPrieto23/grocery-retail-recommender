# Recomendador de cesta (Fase 3, Tarea 3a)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra de este informe se copia a
mano: se recalcula ejecutando ese comando.

## Montaje

| | |
| --- | --- |
| Fuentes de candidatos (ranker) | cestas anteriores a 2025-09-01 |
| Queries de entrenamiento | 2025-09-01 a 2025-11-01 (11,936 cestas con algun candidato relevante en el pool; 11,611 con el SKU exacto) |
| Fuentes de candidatos (test) | cestas anteriores a 2025-11-01 |
| Queries de test | desde 2025-11-01 (18,000 cestas) |
| Ranker | LightGBM `lambdarank`, 150 arboles |
| Objetivo | NDCG@5 con relevancia graduada: 2 SKU exacto, 1 misma categoria (`label_gain` = [0.0, 1.0, 3.0]) |
| NDCG@5 graduada de validacion | 0.2506 |

El split es temporal **y por cesta**: ninguna cesta se reparte entre train y test, y las
fuentes de candidatos se reajustan para cada ventana con solo el pasado de esa ventana.

## Metrica principal

La metrica principal del recomendador es la **NDCG@5 con relevancia graduada**
(`CHALLENGE.md`, punto A4 de `docs/diagnostico-fase7.md`): 3 puntos por hueco si es el SKU
exacto, 1 si solo acierta la categoria. Es la que optimiza el LambdaRank. Se lee siempre
junto a `cat_hit_rate@5` (lo que ensena la demo) y `sku_hit_rate@5`.

| grupo | n_queries | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 |
| --- | --- | --- | --- | --- |
| total | 18000 | 0.2138 | 0.7173 | 0.5398 |
| 1 - nuevo, carrito vacio | 521 | 0.1532 | 0.5969 | 0.3973 |
| 2 - nuevo, con articulos | 421 | 0.1121 | 0.4632 | 0.2375 |
| 3 - recurrente, carrito vacio | 8713 | 0.2384 | 0.7896 | 0.6190 |
| 4 - recurrente, con articulos | 8345 | 0.1971 | 0.6622 | 0.4814 |

## Objetivo del ranker (punto A4)

Hasta el punto A4 el LambdaRank optimizaba la NDCG@5 con relevancia binaria de SKU exacto. Ahora optimiza la graduada (`RankerConfig.relevance`) y tiene tres features nuevas de ranking personal de categorias (`CUSTOMER_CATEGORY_RANK_FEATURES`: `cat_freq_rank`, `cat_freq_share`, `cat_due_rank`). Las variantes se entrenan con las mismas queries y se evaluan sobre el mismo pool, con el mismo re-ranking servido.

| Variante | NDCG@5 graduada | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 SKU | cat_precision@5 | sku_precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Modelo en disco antes de A4 (congelado) | — | 0.6743 | 0.5488 | 0.2028 | 0.2021 | 0.1482 |
| Relevancia binaria de SKU (objetivo anterior) | 0.2086 | 0.6790 | 0.5491 | 0.2045 | 0.2051 | 0.1488 |
| Relevancia graduada, sin ranking personal | 0.2145 | 0.7179 | 0.5419 | 0.2008 | 0.2265 | 0.1454 |
| **Relevancia graduada + ranking personal (servido)** | **0.2138** | **0.7173** | **0.5398** | **0.1993** | **0.2265** | **0.1445** |

Cambiar solo el objetivo (fila de relevancia binaria a graduada sin ranking) mueve cat_hit_rate@5 **+3.89 pp** y sku_hit_rate@5 **-0.71 pp**. Anadir el ranking personal mueve cat_hit_rate@5 **-0.06 pp** y sku_hit_rate@5 **-0.21 pp**. En conjunto, frente al objetivo anterior: cat_hit_rate@5 **+3.83 pp**, sku_hit_rate@5 **-0.92 pp**.

La fila congelada es el modelo que habia en disco antes de este cambio (`reports/recommender/baseline_pre_a4.json`, 2026-09-17, commit `e0411d4`), que no media la NDCG graduada. La fila "relevancia binaria" lo reentrena con el codigo actual, que ya incluye el ranking personal entre sus features, asi que puede diferir un poco.

### Por perfil: relevancia binaria de SKU frente a la graduada servida

| Grupo | cat_hit_rate@5 SKU | graduada | cambio | sku_hit_rate@5 SKU | graduada | cambio | NDCG@5 graduada SKU | graduada |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 0.6790 | 0.7173 | +3.83 pp | 0.5491 | 0.5398 | -0.92 pp | 0.2086 | 0.2138 |
| 1 - nuevo, carrito vacio | 0.5854 | 0.5969 | +1.15 pp | 0.3935 | 0.3973 | +0.38 pp | 0.1518 | 0.1532 |
| 2 - nuevo, con articulos | 0.4561 | 0.4632 | +0.71 pp | 0.2470 | 0.2375 | -0.95 pp | 0.1120 | 0.1121 |
| 3 - recurrente, carrito vacio | 0.7483 | 0.7896 | +4.13 pp | 0.6261 | 0.6190 | -0.71 pp | 0.2318 | 0.2384 |
| 4 - recurrente, con articulos | 0.6237 | 0.6622 | +3.85 pp | 0.4936 | 0.4814 | -1.22 pp | 0.1928 | 0.1971 |

### Importancia del ranking personal de categorias

Puesto entre todas las features del modelo servido (ganancia).

| Feature | Puesto por ganancia | Ganancia | % de la ganancia total | Splits |
| --- | ---: | ---: | ---: | ---: |
| `cat_freq_share` | 5 de 60 | 11696 | 4.5% | 534 |
| `cat_due_rank` | 8 de 60 | 7003 | 2.7% | 171 |
| `cat_freq_rank` | 10 de 60 | 6153 | 2.4% | 173 |

La comparacion con los baselines de categoria y con el techo teorico esta en la seccion de diagnostico, al final (`verify_recommender_diagnostics`).

## Resultado a nivel de SKU

Las metricas de SKU exacto de siempre (NDCG@5 binaria, recall, precision, F1).

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.1993 | 0.1902 | 0.1445 | 0.1642 | 0.1546 | 0.5398 | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.1309 | 0.1151 | 0.0971 | 0.1053 | 0.0988 | 0.3973 | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.1003 | 0.1091 | 0.0546 | 0.0728 | 0.0709 | 0.2375 | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.2128 | 0.1799 | 0.1753 | 0.1775 | 0.1686 | 0.6190 | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.1945 | 0.2097 | 0.1198 | 0.1525 | 0.1477 | 0.4814 | 2.8506 |

## Comparacion

Mismo pool de candidatos, distinta forma de ordenarlo. Es lo que aisla la aportacion del
ranker de la de la primera etapa.

| Sistema | NDCG@5 | Recall@5 | Precision@5 | F1@5 | hit_rate@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (sin aprendizaje) | 0.0881 | 0.0793 | 0.0618 | 0.0695 | 0.2701 |
| LambdaRank sin senal de sesion | 0.1932 | 0.1856 | 0.1420 | 0.1609 | 0.5347 |
| LambdaRank con relevancia binaria de SKU (objetivo anterior) | 0.2045 | 0.1947 | 0.1488 | 0.1687 | 0.5491 |
| **LambdaRank completo** | **0.1993** | **0.1902** | **0.1445** | **0.1642** | **0.5398** |

### Por perfil, sin senal de sesion

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.1932 | 0.1856 | 0.1420 | 0.1609 | 0.1515 | 0.5347 | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.1322 | 0.1159 | 0.0983 | 0.1063 | 0.0999 | 0.4012 | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.0926 | 0.1011 | 0.0523 | 0.0689 | 0.0671 | 0.2352 | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.2142 | 0.1815 | 0.1762 | 0.1788 | 0.1696 | 0.6223 | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.1802 | 0.1986 | 0.1136 | 0.1445 | 0.1400 | 0.4666 | 2.8506 |

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

| grupo | n_queries | ndcg_graded@5 | cat_hit_rate@5 | cat_precision@5 | sku_hit_rate@5 | sku_precision@5 |
| --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.2138 | 0.7173 | 0.2265 | 0.5398 | 0.1445 |
| 1 - nuevo, carrito vacio | 521 | 0.1532 | 0.5969 | 0.1820 | 0.3973 | 0.0971 |
| 2 - nuevo, con articulos | 421 | 0.1121 | 0.4632 | 0.1197 | 0.2375 | 0.0546 |
| 3 - recurrente, carrito vacio | 8713 | 0.2384 | 0.7896 | 0.2728 | 0.6190 | 0.1753 |
| 4 - recurrente, con articulos | 8345 | 0.1971 | 0.6622 | 0.1864 | 0.4814 | 0.1198 |

El sistema acierta la categoria en el **71.7%** de las cestas y el SKU exacto en el **54.0%**; el cociente entre las dos es **75.3%**. La distancia entre las dos columnas mide cuanto del error esta en *elegir la referencia* y no en *saber que categoria toca*. Desde la Fase 7a el surtido es de 8 referencias por categoria y el cliente repite su referencia preferida con la lealtad de la categoria (`DATA_SPEC.md`, "Fidelidad de marca").

## Historial al dia de la cesta (punto A1)

Las features de cliente x producto, cliente x categoria y cliente, y la fuente `hist` con su `due_for_repurchase`, se calculan con todas las cestas del cliente anteriores al dia de cada query (`src/recommender/history.py`), tambien las de dentro de la ventana. ALS, popularidad, afinidades y el perfil siguen congelados al inicio de cada ventana.

"Antes" es el modelo con el historial congelado al inicio de la ventana (`reports/recommender/baseline_pre_a1.json`, 2026-09-17, commit `bd1adc6`), sobre las mismas queries de test.

| Grupo | cat_hit_rate@5 antes | despues | cambio | sku_hit_rate@5 antes | despues | cambio | NDCG@5 antes | despues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 0.6168 | 0.7173 | +10.05 pp | 0.4947 | 0.5398 | +4.52 pp | 0.1759 | 0.1993 |
| 1 - nuevo, carrito vacio | 0.5624 | 0.5969 | +3.45 pp | 0.3724 | 0.3973 | +2.50 pp | 0.1204 | 0.1309 |
| 2 - nuevo, con articulos | 0.4371 | 0.4632 | +2.61 pp | 0.2114 | 0.2375 | +2.61 pp | 0.0851 | 0.1003 |
| 3 - recurrente, carrito vacio | 0.6848 | 0.7896 | +10.48 pp | 0.5741 | 0.6190 | +4.49 pp | 0.1881 | 0.2128 |
| 4 - recurrente, con articulos | 0.5583 | 0.6622 | +10.39 pp | 0.4337 | 0.4814 | +4.77 pp | 0.1713 | 0.1945 |

El detalle de los huecos que caian en categorias recien repuestas esta en la seccion de diagnostico (`verify_recommender_diagnostics`).

## Carrito y diversidad (punto A2)

Con una linea por categoria en cada cesta, un hueco del top-5 se *regala* si su categoria ya esta en el carrito (no puede acertar) o ya salio mas arriba en la lista (de las dos, como mucho acierta una). Dos piezas atacan el problema: tres features de carrito (`cat_in_cart`, `dept_n_in_cart`, `dept_share_in_cart`) y un re-ranking final (`src/recommender/rerank.py`). Reglas servidas: **como maximo 1 referencia(s) por categoria, categorias del carrito relegadas** (`RecommenderConfig.rerank`), las mismas en esta evaluacion, en las tablas de arriba y en la demo.

| Variante | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 | Huecos regalados | Huecos en cat. del carrito (perfiles 2 y 4) | Listas con carrito afectadas | Listas con cat. repetida | Categorias distintas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sin features de carrito, sin re-ranking (antes) | 0.6587 | 0.5236 | 0.1910 | 19.2% | 7.0% | 67.2% | 55.3% | 4.16 |
| Sin features de carrito + re-ranking servido | 0.7156 | 0.5419 | 0.2010 | 0.0% | 0.0% | 0.0% | 0.0% | 5.00 |
| Con features de carrito, sin re-ranking | 0.6468 | 0.5178 | 0.1889 | 22.7% | 0.0% | 69.4% | 65.0% | 3.87 |
| Con features de carrito + 1 por categoria | 0.7176 | 0.5398 | 0.1993 | 0.0% | 0.0% | 0.1% | 0.0% | 5.00 |
| **Con features de carrito + 1 por categoria + exclusion del carrito** | **0.7173** | **0.5398** | **0.1993** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **5.00** |

Las filas "sin features de carrito" usan un LambdaRank entrenado aparte con las mismas queries y sin esas tres columnas. Los huecos en categorias del carrito se miden contra el prefijo del ticket; la regla de exclusion usa el carrito en el corte, que ademas incluye los `add_to_cart` de sesion.

Frente a la fila "antes" (mismo entrenamiento, sin las tres features ni reglas), el sistema servido (en negrita) cambia cat_hit_rate@5 en **+5.86 pp** y sku_hit_rate@5 en **+1.62 pp**, y los huecos regalados pasan del 19.2% al 0.0%.

Referencia congelada del modelo que habia en disco antes de este cambio (`reports/recommender/baseline_pre_diagnostico.json`, 2026-09-17): cat_hit_rate@5 0.6003, sku_hit_rate@5 0.4948, NDCG@5 0.1752. La fila "antes" lo reentrena con el codigo actual y puede diferir un poco: el muestreo de columnas de LightGBM depende del numero de features.

### Huecos regalados del sistema servido, por perfil

| grupo | huecos_en_carrito@5 | huecos_repetidos@5 | huecos_regalados@5 | listas_con_regalo@5 |
| --- | --- | --- | --- | --- |
| total | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 - nuevo, carrito vacio | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 - nuevo, con articulos | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 - recurrente, carrito vacio | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 - recurrente, con articulos | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| con carrito (perfiles 2 y 4) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## F1@5 frente a Kaggle "Instacart Market Basket Analysis"

| | F1 |
| --- | ---: |
| Este sistema, F1@5 (media armonica de Precision@5 y Recall@5 medios) | 0.1642 |
| Este sistema, F1@5 por cesta (media del F1 de cada cesta) | 0.1546 |
| Instacart, 1er puesto (aprox.) | 0.41 |

**No es el mismo benchmark y las cifras no se deben leer como una comparacion directa.**
Se ponen juntas solo como orden de magnitud, con estas diferencias de planteamiento:

- **Que se predice.** Instacart pide solo *recompras*: que productos que el usuario ya
  compro antes estaran en su siguiente pedido. Aqui el top-5 mezcla recompra con
  *descubrimiento* (popularidad, co-compra, ALS), y el target incluye productos que el
  cliente no habia comprado nunca.
- **Tamano de la lista.** En Instacart cada pedido recibe un conjunto de **tamano
  variable**, elegido para maximizar el F1 esperado de ese pedido (F1-maximization),
  incluida la opcion de predecir "ninguno". Aqui la lista es **siempre de 5**: con
  3.9 productos por adivinar de media, la Precision@5 y el
  Recall@5 estan acotados por el propio formato, acierte lo que acierte el modelo.
- **Que se optimiza.** El ranker se entrena con LambdaRank para NDCG@5 graduada (SKU y
  categoria), no para F1.
- **Contexto.** Aqui se predice a mitad de cesta: lo que ya esta en el carrito queda fuera
  del target. Instacart predice el pedido entero.
- **Agregacion.** Instacart promediaba el F1 de cada pedido; la variante mas cercana es la
  segunda fila ("por cesta"), no la de cabecera.


## Frente a la Fase 3 original

La Fase 3 se entreno sobre el dataset anterior a la Fase 7a (1.500 productos, ~24 referencias por categoria y eleccion de SKU casi aleatoria). Sus cifras estan congeladas en `reports/recommender/baseline_fase3.json`. Mismo codigo, mismas ventanas y mismo numero de queries de test; cambia el dato.

| Metrica | Fase 3 (dataset viejo) | Fase 7c (dataset nuevo) | Cambio |
| --- | ---: | ---: | ---: |
| NDCG@5 | 0.0343 | 0.1993 | 5.81x |
| Recall@5 | 0.0332 | 0.1902 | 5.73x |
| Precision@5 | 0.0251 | 0.1445 | 5.76x |
| F1@5 | 0.0286 | 0.1642 | 5.75x |
| hit_rate@5 (SKU) | 0.1184 | 0.5398 | 4.56x |
| hit_rate@5 (categoria) | 0.5144 | 0.7173 | 1.39x |
| SKU / categoria (hit_rate) | 0.2302 | 0.7526 | 3.27x |
| SKU / categoria (precision) | 0.1371 | 0.6377 | 4.65x |

Un matiz al leerlo: el catalogo pasa de 1.500 a 496 productos, asi que un top-5 al azar tambien acierta mas que antes. El baseline de popularidad de la tabla de comparacion, sobre el mismo pool, es lo que aisla lo que aporta el ranker.

## Techo de la primera etapa

Que parte del target llego siquiera al pool de candidatos. Lo que no esta aqui, el ranker
no lo puede recuperar.

| grupo | n_queries | pool_recall | pool_size_medio |
| --- | --- | --- | --- |
| total | 18000 | 0.7781 | 140.2235 |
| 1 - nuevo, carrito vacio | 521 | 0.4104 | 63.9060 |
| 2 - nuevo, con articulos | 421 | 0.5291 | 110.6152 |
| 3 - recurrente, carrito vacio | 8713 | 0.7786 | 123.3732 |
| 4 - recurrente, con articulos | 8345 | 0.8132 | 164.0753 |

## Que features usa el ranker

Importancia por ganancia, las 25 primeras.

| feature | gain | split |
| --- | --- | --- |
| hist_rank | 59874.5966 | 219 |
| category_idx | 42899.0912 | 1334 |
| cat_overdue_ratio | 24633.5856 | 717 |
| cat_days_since | 19448.4681 | 791 |
| cat_freq_share | 11695.7990 | 534 |
| hist_n_baskets | 9716.5466 | 143 |
| sess_secs_since_view | 8167.7117 | 204 |
| cat_due_rank | 7003.1273 | 171 |
| cat_in_cart | 6912.7384 | 141 |
| cat_freq_rank | 6153.0926 | 173 |
| is_known_customer | 4870.2965 | 51 |
| aff_lift_max | 4756.4370 | 238 |
| cust_n_products | 4591.1027 | 368 |
| typical_repurchase_days | 4045.4679 | 181 |
| cust_recency_days | 3728.8115 | 396 |
| cust_avg_ticket | 3553.6475 | 463 |
| cust_frequency | 3506.7177 | 248 |
| cat_expected_days | 3404.7716 | 413 |
| hist_days_since | 2959.9095 | 177 |
| cat_n_purchase_days | 2313.2926 | 212 |
| prod_pop_all | 1746.9950 | 82 |
| cataff_conf_max | 1704.4026 | 140 |
| prefix_size | 1440.7764 | 163 |
| cataff_lift_max | 1435.5515 | 160 |
| aff_conf_sum | 1254.2093 | 82 |

<!-- diagnostics:start -->
## Diagnostico: baselines independientes del pool y techo teorico

Generado por `python -m src.recommender.verify_recommender_diagnostics` (2026-09-17), sobre las 18,000 queries de test de las predicciones en disco (sha256 de `recommendations_test.parquet`: `ae91ab3b4781`). El oraculo necesita antes `python -m data_generation.export_oracle`. Esta seccion no la reescribe el pipeline: si se reentrena, hay que volver a lanzar el verificador. Puntos A3, A4, A6 y M8 de `docs/diagnostico-fase7.md`. La metrica principal es la NDCG@5 graduada (3 por SKU exacto, 1 por categoria; `evaluate.category_metrics`).

### Lectura rapida

- **Mejor baseline en cat_hit_rate@5:** Frecuencia personal x due_for_repurchase + referencia favorita, 0.6789 frente a 0.7173 del LambdaRank: -3.8 pp (el LambdaRank va por delante).
- **Techo teorico de cat_hit_rate@5:** 0.7816. El LambdaRank alcanza el 91.8% y el mejor baseline el 86.9%.
- **Techo de sku_hit_rate@5:** 0.5920. El LambdaRank alcanza el 91.2%.
- **Objetivo del ranker (A4):** con relevancia binaria de SKU, cat_hit_rate@5 0.6790 (+0.0 pp frente al mejor baseline); con la graduada servida, 0.7173 (+3.8 pp). sku_hit_rate@5 pasa de 0.5491 a 0.5398 y la NDCG@5 graduada de 0.2086 a 0.2138.

### Todos los sistemas, total

Las columnas "% techo" dividen por el oraculo correspondiente sobre las mismas queries. "Huecos regalados" es la parte del top-5 en una categoria que ya esta en el carrito o repetida mas arriba en la lista (`evaluate.wasted_slot_metrics`, punto A2). Los baselines de categoria no regalan ninguno por construccion; el LambdaRank los evita con el re-ranking de `RecommenderConfig.rerank`.

| Sistema | NDCG@5 graduada | cat_hit_rate@5 | % techo | cat_precision@5 | sku_hit_rate@5 | % techo SKU | sku_precision@5 | NDCG@5 SKU | Huecos regalados |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.2308 | 0.7816 | — | 0.2660 | 0.5542 | — | 0.1495 | 0.2004 | 0.0% |
| **Oraculo de SKU (techo)** | 0.2340 | 0.7309 | — | 0.2339 | 0.5920 | — | 0.1661 | 0.2269 | 0.0% |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.2138 | 0.7173 | 91.8% | 0.2265 | 0.5398 | 91.2% | 0.1445 | 0.1993 | 0.0% |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.2086 | 0.6790 | 86.9% | 0.2051 | 0.5491 | 92.7% | 0.1488 | 0.2045 | 0.0% |
| LambdaRank antes de A4 (relevancia binaria, congelado) | 0.2059 | 0.6743 | 86.3% | 0.2021 | 0.5488 | 92.7% | 0.1482 | 0.2028 | 0.0% |
| LambdaRank antes de A1 (historial congelado) | 0.1791 | 0.6168 | 78.9% | 0.1754 | 0.4947 | 83.6% | 0.1285 | 0.1759 | 0.0% |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.1680 | 0.6789 | 86.9% | 0.2089 | 0.4214 | 71.2% | 0.1047 | 0.1374 | 0.0% |
| Frecuencia personal por categoria + referencia favorita | 0.1564 | 0.6489 | 83.0% | 0.1957 | 0.3994 | 67.5% | 0.0971 | 0.1278 | 0.0% |
| Popularidad de categoria + referencia lider | 0.1237 | 0.6109 | 78.2% | 0.1791 | 0.2651 | 44.8% | 0.0605 | 0.0867 | 0.0% |
| Reglas de asociacion de categoria + referencia lider | 0.1222 | 0.6096 | 78.0% | 0.1782 | 0.2586 | 43.7% | 0.0591 | 0.0835 | 0.0% |
| Repetir las referencias favoritas del cliente | 0.1616 | 0.5969 | 76.4% | 0.1739 | 0.4481 | 75.7% | 0.1125 | 0.1512 | 1.4% |
| Popularidad global (SKU) | 0.1198 | 0.5799 | 74.2% | 0.1666 | 0.2662 | 45.0% | 0.0606 | 0.0871 | 0.5% |
| Aleatorio | 0.0301 | 0.2754 | 35.2% | 0.0641 | 0.0408 | 6.9% | 0.0082 | 0.0102 | 2.8% |

Cifras congeladas del LambdaRank en `reports/recommender/baseline_pre_diagnostico.json` (2026-09-17, commit `83ab9f7`): cat_hit_rate@5 0.6003, sku_hit_rate@5 0.4948. **Las predicciones en disco ya no son las congeladas**: la fila del LambdaRank de arriba es la version actual.

### cat_hit_rate@5 por perfil (entre parentesis, % del techo del perfil)

| Sistema | Total | Perfil 1 - nuevo, carrito vacio | Perfil 2 - nuevo, con articulos | Perfil 3 - recurrente, carrito vacio | Perfil 4 - recurrente, con articulos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.7816 | 0.6545 | 0.5297 | 0.8522 | 0.7286 |
| **Oraculo de SKU (techo)** | 0.7309 | 0.6372 | 0.4632 | 0.8057 | 0.6723 |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.7173 (91.8%) | 0.5969 (91.2%) | 0.4632 (87.4%) | 0.7896 (92.7%) | 0.6622 (90.9%) |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.6790 (86.9%) | 0.5854 (89.4%) | 0.4561 (86.1%) | 0.7483 (87.8%) | 0.6237 (85.6%) |
| LambdaRank antes de A4 (relevancia binaria, congelado) | 0.6743 (86.3%) | 0.5835 (89.1%) | 0.4418 (83.4%) | 0.7470 (87.7%) | 0.6158 (84.5%) |
| LambdaRank antes de A1 (historial congelado) | 0.6168 (78.9%) | 0.5624 (85.9%) | 0.4371 (82.5%) | 0.6848 (80.4%) | 0.5583 (76.6%) |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.6789 (86.9%) | 0.5547 (84.8%) | 0.4086 (77.1%) | 0.7690 (90.2%) | 0.6062 (83.2%) |
| Frecuencia personal por categoria + referencia favorita | 0.6489 (83.0%) | 0.5413 (82.7%) | 0.4038 (76.2%) | 0.7360 (86.4%) | 0.5771 (79.2%) |
| Popularidad de categoria + referencia lider | 0.6109 (78.2%) | 0.5662 (86.5%) | 0.4418 (83.4%) | 0.7007 (82.2%) | 0.5285 (72.5%) |
| Reglas de asociacion de categoria + referencia lider | 0.6096 (78.0%) | 0.5662 (86.5%) | 0.4347 (82.1%) | 0.7007 (82.2%) | 0.5261 (72.2%) |
| Repetir las referencias favoritas del cliente | 0.5969 (76.4%) | 0.5298 (80.9%) | 0.3753 (70.9%) | 0.6845 (80.3%) | 0.5208 (71.5%) |
| Popularidad global (SKU) | 0.5799 (74.2%) | 0.5643 (86.2%) | 0.4204 (79.4%) | 0.6654 (78.1%) | 0.4997 (68.6%) |
| Aleatorio | 0.2754 (35.2%) | 0.2764 (42.2%) | 0.1591 (30.0%) | 0.3324 (39.0%) | 0.2218 (30.4%) |

### sku_hit_rate@5 por perfil (entre parentesis, % del techo de SKU del perfil)

| Sistema | Total | Perfil 1 - nuevo, carrito vacio | Perfil 2 - nuevo, con articulos | Perfil 3 - recurrente, carrito vacio | Perfil 4 - recurrente, con articulos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.5542 | 0.4203 | 0.2494 | 0.6311 | 0.4975 |
| **Oraculo de SKU (techo)** | 0.5920 | 0.4587 | 0.2589 | 0.6837 | 0.5214 |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.5398 (91.2%) | 0.3973 (86.6%) | 0.2375 (91.7%) | 0.6190 (90.5%) | 0.4814 (92.3%) |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.5491 (92.7%) | 0.3935 (85.8%) | 0.2470 (95.4%) | 0.6261 (91.6%) | 0.4936 (94.7%) |
| LambdaRank antes de A4 (relevancia binaria, congelado) | 0.5488 (92.7%) | 0.3954 (86.2%) | 0.2399 (92.7%) | 0.6288 (92.0%) | 0.4905 (94.1%) |
| LambdaRank antes de A1 (historial congelado) | 0.4947 (83.6%) | 0.3724 (81.2%) | 0.2114 (81.7%) | 0.5741 (84.0%) | 0.4337 (83.2%) |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.4214 (71.2%) | 0.3129 (68.2%) | 0.1853 (71.6%) | 0.5027 (73.5%) | 0.3552 (68.1%) |
| Frecuencia personal por categoria + referencia favorita | 0.3994 (67.5%) | 0.2994 (65.3%) | 0.1829 (70.6%) | 0.4715 (69.0%) | 0.3413 (65.5%) |
| Popularidad de categoria + referencia lider | 0.2651 (44.8%) | 0.3090 (67.4%) | 0.1853 (71.6%) | 0.3219 (47.1%) | 0.2070 (39.7%) |
| Reglas de asociacion de categoria + referencia lider | 0.2586 (43.7%) | 0.3090 (67.4%) | 0.1686 (65.1%) | 0.3219 (47.1%) | 0.1938 (37.2%) |
| Repetir las referencias favoritas del cliente | 0.4481 (75.7%) | 0.3033 (66.1%) | 0.1734 (67.0%) | 0.5369 (78.5%) | 0.3783 (72.6%) |
| Popularidad global (SKU) | 0.2662 (45.0%) | 0.3090 (67.4%) | 0.1829 (70.6%) | 0.3208 (46.9%) | 0.2108 (40.4%) |
| Aleatorio | 0.0408 (6.9%) | 0.0230 (5.0%) | 0.0285 (11.0%) | 0.0487 (7.1%) | 0.0344 (6.6%) |

### Huecos en categorias recien compradas (punto A1)

Cada hueco del top-5 se clasifica por los dias desde que el cliente compro por ultima vez su categoria, con **todo** su historial anterior al dia de la cesta (lo viera o no el modelo). El generador castiga con fuerza reponer justo despues de comprar, asi que los huecos de los primeros dias casi nunca aciertan. "En la ventana" es desde el inicio del test. Precision = parte de esos huecos que acierta el SKU o la categoria.

| Huecos en categorias compradas... | LambdaRank antes de A1 (historial congelado): % huecos | SKU prec. | cat. prec. | LambdaRank servido: relevancia graduada (predicciones en disco): % huecos | SKU prec. | cat. prec. | Frecuencia personal x due_for_repurchase + referencia favorita: % huecos | SKU prec. | cat. prec. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <= 7 dias | 15.6% | 4.7% | 7.2% | 2.7% | 15.0% | 28.8% | 14.7% | 5.0% | 10.6% |
| <= 14 dias | 27.2% | 7.7% | 11.5% | 11.3% | 15.5% | 27.9% | 24.6% | 7.7% | 16.1% |
| 8-14 dias | 11.6% | 11.7% | 17.2% | 8.6% | 15.7% | 27.6% | 10.0% | 11.7% | 24.3% |
| comprada en la ventana, antes de la cesta | 41.7% | 10.5% | 14.9% | 29.5% | 15.6% | 26.6% | 36.9% | 9.8% | 20.3% |
| sin compra en la ventana | 58.3% | 14.5% | 19.4% | 70.5% | 14.0% | 21.0% | 63.1% | 10.8% | 21.2% |
| total | 100.0% | 12.9% | 17.5% | 100.0% | 14.4% | 22.7% | 100.0% | 10.5% | 20.9% |

**Queries donde el modelo de antes gastaba huecos en categorias recien compradas.** Para cada umbral, las queries cuya lista antes de A1 tenia al menos un hueco en una categoria comprada hace <= d dias, y como les va a cada sistema (lista entera de 5).

| Umbral | Sistema | Queries | Huecos recientes | sku_precision@5 | cat_precision@5 | sku_hit_rate@5 | cat_hit_rate@5 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <= 7 dias | LambdaRank servido: relevancia graduada (predicciones en disco) | 6,693 | 7.1% | 0.1508 | 0.2417 | 0.5567 | 0.7457 |
| <= 7 dias | LambdaRank antes de A1 (historial congelado) | 6,693 | 41.9% | 0.1140 | 0.1551 | 0.4538 | 0.5684 |
| <= 7 dias | Frecuencia personal x due_for_repurchase + referencia favorita | 6,693 | 36.4% | 0.0990 | 0.2013 | 0.4028 | 0.6607 |
| <= 14 dias | LambdaRank servido: relevancia graduada (predicciones en disco) | 9,987 | 19.9% | 0.1484 | 0.2378 | 0.5522 | 0.7404 |
| <= 14 dias | LambdaRank antes de A1 (historial congelado) | 9,987 | 49.0% | 0.1195 | 0.1645 | 0.4715 | 0.5936 |
| <= 14 dias | Frecuencia personal x due_for_repurchase + referencia favorita | 9,987 | 42.2% | 0.1026 | 0.2084 | 0.4165 | 0.6782 |

Las predicciones de antes son las congeladas en `baseline_pre_a1.json` (2026-09-17, commit `bd1adc6`).

### Como se construye el techo

El generador exporta, para cada cesta de test, el peso de cada categoria antes del primer sorteo (afinidad x estacionalidad x ciclo de reposicion) y la referencia mas probable de cada una (`data/oracle/`, fuera de `data/raw`). Como el sorteo es secuencial y sin reemplazo, la probabilidad de que cada categoria este en el resto de la cesta, dado el carrito, se calcula por Monte Carlo: una carrera de relojes exponenciales con los lifts de afinidad, condicionada al carrito por muestreo por importancia (exacta salvo ruido de muestreo; detalle en `src/recommender/oracle.py`). El **oraculo de categoria** recomienda las 5 categorias de mas peso fuera del carrito (con el lift de las disparadoras del carrito aplicado), cada una con su referencia mas probable: es la lista optima salvo en los 10 pares de afinidad, asi que su valor es una cota inferior muy ajustada del maximo. El **oraculo de SKU** ordena por `P(categoria en el resto) x P(mejor referencia)`, estimada con la mitad de las muestras y medida con la otra. Ningun sistema que solo vea el pasado puede superarlos en valor esperado: lo que queda entre el techo y el 100 % es entropia del generador.

El realizado y el esperado deben coincidir salvo ruido (ver comprobaciones):

| Techo | Realizado (estas queries) | Esperado (Monte Carlo) |
| --- | ---: | ---: |
| cat_hit_rate@5 (oraculo de categoria) | 0.7816 | 0.7783 ± 0.0055 |
| cat_precision@5 (oraculo de categoria) | 0.2660 | 0.2662 |
| sku_hit_rate@5 (oraculo de SKU) | 0.5920 | 0.5861 |
| sku_precision@5 (oraculo de SKU) | 0.1661 | 0.1656 |

Muestras por query: 1,000 (mitad de evaluacion: tamano efectivo mediano 1000, percentil 5 50; queries con peso de respaldo: 7). Reglas de asociacion: 489 de 3772 pares con confianza >= 0.10 y lift > 1.

### Techo esperado por perfil

| grupo | cat_hit_rate@5 | cat_precision@5 | sku_hit_rate@5 | sku_precision@5 |
| --- | --- | --- | --- | --- |
| total | 0.7783 | 0.2662 | 0.5861 | 0.1656 |
| 1 - nuevo, carrito vacio | 0.6537 | 0.2136 | 0.4198 | 0.1103 |
| 2 - nuevo, con articulos | 0.5201 | 0.1352 | 0.2687 | 0.0611 |
| 3 - recurrente, carrito vacio | 0.8478 | 0.3256 | 0.6804 | 0.2084 |
| 4 - recurrente, con articulos | 0.7264 | 0.2141 | 0.5141 | 0.1296 |

### Comprobaciones

- OK — La ablacion con relevancia binaria cubre las mismas queries que el LambdaRank
- OK — Las predicciones de referencia (antes de A4) son las congeladas y cubren las mismas queries (sha256 = el de baseline_pre_a4.json)
- OK — Una linea por categoria: el target en categorias coincide con n_target
- OK — LambdaRank recalculado desde predictions/ = reports/recommender/metrics.json (0.7173 frente a 0.7173)
- OK — Techo realizado compatible con el esperado en las 4 metricas, total y por perfil (max |z| <= 3.5) (peor: sku_hit_p1, z = +1.55)
- OK — Ningun sistema supera al oraculo de categoria en cat_hit_rate (mejor sistema lambdarank = 0.7173)
- OK — Las predicciones de referencia (antes de A1) son las congeladas y cubren las mismas queries (sha256 = el de baseline_pre_a1.json)
<!-- diagnostics:end -->

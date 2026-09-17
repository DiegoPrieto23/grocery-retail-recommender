# Recomendador de cesta (Fase 3, Tarea 3a)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra de este informe se copia a
mano: se recalcula ejecutando ese comando.

## Montaje

| | |
| --- | --- |
| Fuentes de candidatos (ranker) | cestas anteriores a 2025-09-01 |
| Queries de entrenamiento | 2025-09-01 a 2025-11-01 (11,906 cestas con algun candidato relevante en el pool; 11,594 con el SKU exacto) |
| Fuentes de candidatos (test) | cestas anteriores a 2025-11-01 |
| Queries de test | desde 2025-11-01 (18,000 cestas) |
| Ranker | LightGBM `lambdarank`, 168 arboles |
| Objetivo | NDCG@5 con relevancia graduada: 2 SKU exacto, 1 misma categoria (`label_gain` = [0.0, 1.0, 3.0]) |
| NDCG@5 graduada de validacion | 0.3377 |

El split es temporal **y por cesta**: ninguna cesta se reparte entre train y test, y las
fuentes de candidatos se reajustan para cada ventana con solo el pasado de esa ventana.

## Metrica principal

La metrica principal del recomendador es la **NDCG@5 con relevancia graduada**
(`CHALLENGE.md`, punto A4 de `docs/diagnostico-fase7.md`): 3 puntos por hueco si es el SKU
exacto, 1 si solo acierta la categoria. Es la que optimiza el LambdaRank. Se lee siempre
junto a `cat_hit_rate@5` (lo que ensena la demo) y `sku_hit_rate@5`.

| grupo | n_queries | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 |
| --- | --- | --- | --- | --- |
| total | 18000 | 0.3015 [0.2981, 0.3052] | 0.7954 [0.7898, 0.8011] | 0.6157 [0.6088, 0.6228] |
| 1 - nuevo, carrito vacio | 554 | 0.2330 [0.2161, 0.2523] | 0.7112 [0.6751, 0.7473] | 0.4892 [0.4495, 0.5307] |
| 2 - nuevo, con articulos | 388 | 0.2172 [0.1957, 0.2392] | 0.7088 [0.6623, 0.7552] | 0.4304 [0.3840, 0.4794] |
| 3 - recurrente, carrito vacio | 9460 | 0.3148 [0.3099, 0.3200] | 0.8057 [0.7980, 0.8139] | 0.6420 [0.6327, 0.6520] |
| 4 - recurrente, con articulos | 7598 | 0.2942 [0.2885, 0.2990] | 0.7932 [0.7843, 0.8028] | 0.6016 [0.5904, 0.6123] |

Entre corchetes, el intervalo de confianza al 95%: bootstrap percentil con 1,000 remuestreos de cestas (semilla 42; `evaluate.bootstrap_means`). En el grupo mas pequeno (2 - nuevo, con articulos, n = 388) el intervalo de cat_hit_rate@5 mide 9.3 puntos: las diferencias entre perfiles pequenos se leen con eso delante. Los perfiles 1 y 2, con muchas mas queries, estan en "Cold-start sobremuestreado".

El acierto segun cuantas lineas lleva ya el carrito (varios cortes por cesta, punto M3) esta en [`cuts.md`](cuts.md).

## Objetivo del ranker (punto A4)

Hasta el punto A4 el LambdaRank optimizaba la NDCG@5 con relevancia binaria de SKU exacto. Ahora optimiza la graduada (`RankerConfig.relevance`) y tiene tres features nuevas de ranking personal de categorias (`CUSTOMER_CATEGORY_RANK_FEATURES`: `cat_freq_rank`, `cat_freq_share`, `cat_due_rank`). Las variantes se entrenan con las mismas queries y se evaluan sobre el mismo pool, con el mismo re-ranking servido.

| Variante | NDCG@5 graduada | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 SKU | cat_precision@5 | sku_precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Relevancia binaria de SKU (objetivo anterior) | 0.2970 | 0.7758 | 0.6226 | 0.2874 | 0.3074 | 0.2063 |
| Relevancia graduada, sin ranking personal | 0.3014 | 0.7941 | 0.6147 | 0.2818 | 0.3266 | 0.2009 |
| **Relevancia graduada + ranking personal (servido)** | **0.3015** | **0.7954** | **0.6157** | **0.2815** | **0.3269** | **0.2008** |

Cambiar solo el objetivo (fila de relevancia binaria a graduada sin ranking) mueve cat_hit_rate@5 **+1.83 pp** y sku_hit_rate@5 **-0.79 pp**. Anadir el ranking personal mueve cat_hit_rate@5 **+0.13 pp** y sku_hit_rate@5 **+0.09 pp**. En conjunto, frente al objetivo anterior: cat_hit_rate@5 **+1.97 pp**, sku_hit_rate@5 **-0.69 pp**.



### Por perfil: relevancia binaria de SKU frente a la graduada servida

| Grupo | cat_hit_rate@5 SKU | graduada | cambio | sku_hit_rate@5 SKU | graduada | cambio | NDCG@5 graduada SKU | graduada |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 0.7758 | 0.7954 | +1.97 pp | 0.6226 | 0.6157 | -0.69 pp | 0.2970 | 0.3015 |
| 1 - nuevo, carrito vacio | 0.7040 | 0.7112 | +0.72 pp | 0.4928 | 0.4892 | -0.36 pp | 0.2353 | 0.2330 |
| 2 - nuevo, con articulos | 0.6907 | 0.7088 | +1.80 pp | 0.4149 | 0.4304 | +1.55 pp | 0.2147 | 0.2172 |
| 3 - recurrente, carrito vacio | 0.7838 | 0.8057 | +2.19 pp | 0.6479 | 0.6420 | -0.59 pp | 0.3095 | 0.3148 |
| 4 - recurrente, con articulos | 0.7753 | 0.7932 | +1.79 pp | 0.6112 | 0.6016 | -0.96 pp | 0.2901 | 0.2942 |

### Importancia del ranking personal de categorias

Puesto entre todas las features del modelo servido (ganancia).

| Feature | Puesto por ganancia | Ganancia | % de la ganancia total | Splits |
| --- | ---: | ---: | ---: | ---: |
| `cat_freq_share` | 5 de 60 | 14732 | 4.4% | 616 |
| `cat_due_rank` | 7 de 60 | 7658 | 2.3% | 224 |
| `cat_freq_rank` | 16 de 60 | 4175 | 1.3% | 169 |

La comparacion con los baselines de categoria y con el techo teorico esta en la seccion de diagnostico, al final (`verify_recommender_diagnostics`).

## Resultado a nivel de SKU

Las metricas de SKU exacto de siempre (NDCG@5 binaria, recall, precision, F1), con su
intervalo de confianza.

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.2815 [0.2777, 0.2857] | 0.2350 [0.2309, 0.2393] | 0.2008 [0.1980, 0.2040] | 0.2165 [0.2137, 0.2196] | 0.1807 [0.1782, 0.1832] | 0.6157 [0.6088, 0.6228] | 5.5474 |
| 1 - nuevo, carrito vacio | 554 | 0.2122 [0.1912, 0.2341] | 0.1810 [0.1601, 0.2044] | 0.1386 [0.1260, 0.1538] | 0.1570 [0.1427, 0.1725] | 0.1285 [0.1166, 0.1413] | 0.4892 [0.4495, 0.5307] | 5.0162 |
| 2 - nuevo, con articulos | 388 | 0.1965 [0.1712, 0.2221] | 0.1898 [0.1630, 0.2172] | 0.1134 [0.0990, 0.1289] | 0.1420 [0.1245, 0.1596] | 0.1265 [0.1109, 0.1425] | 0.4304 [0.3840, 0.4794] | 3.3608 |
| 3 - recurrente, carrito vacio | 9460 | 0.2899 [0.2844, 0.2958] | 0.2246 [0.2194, 0.2301] | 0.2171 [0.2131, 0.2215] | 0.2208 [0.2168, 0.2252] | 0.1794 [0.1763, 0.1830] | 0.6420 [0.6327, 0.6520] | 6.7346 |
| 4 - recurrente, con articulos | 7598 | 0.2804 [0.2741, 0.2867] | 0.2542 [0.2481, 0.2609] | 0.1894 [0.1845, 0.1937] | 0.2171 [0.2122, 0.2216] | 0.1888 [0.1846, 0.1926] | 0.6016 [0.5904, 0.6123] | 4.2197 |

## Comparacion

Mismo pool de candidatos, distinta forma de ordenarlo. Es lo que aisla la aportacion del
ranker de la de la primera etapa.

| Sistema | NDCG@5 | Recall@5 | Precision@5 | F1@5 | hit_rate@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (sin aprendizaje) | 0.1565 | 0.1273 | 0.1088 | 0.1173 | 0.4082 |
| LambdaRank sin senal de sesion | 0.2753 | 0.2301 | 0.1967 | 0.2121 | 0.6096 |
| LambdaRank con relevancia binaria de SKU (objetivo anterior) | 0.2874 | 0.2379 | 0.2063 | 0.2210 | 0.6226 |
| **LambdaRank completo** | **0.2815** | **0.2350** | **0.2008** | **0.2165** | **0.6157** |

### Diferencias con intervalo de confianza (bootstrap pareado)

Diferencia = LambdaRank servido - el otro sistema, sobre las mismas queries y con el mismo re-ranking. Cada remuestreo sortea las mismas cestas para los dos sistemas (`evaluate.paired_bootstrap`, 1,000 remuestreos). El p-valor es bilateral y no esta corregido por comparaciones multiples. Las tasas van en puntos porcentuales. El desglose por perfil esta en `metrics.json` (`comparisons`).

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (mismo pool) | +0.0927 [+0.0900, +0.0956] | < 0.001 | +7.10 pp [+6.49, +7.67] | < 0.001 | +20.74 pp [+20.06, +21.58] | < 0.001 | +0.1250 [+0.1212, +0.1288] | < 0.001 |
| LambdaRank sin senal de sesion | +0.0050 [+0.0038, +0.0063] | < 0.001 | +0.42 pp [+0.16, +0.68] | 0.008 | +0.61 pp [+0.28, +0.96] | 0.002 | +0.0062 [+0.0046, +0.0078] | < 0.001 |
| LambdaRank sin features de carrito | +0.0024 [+0.0013, +0.0036] | < 0.001 | +0.51 pp [+0.19, +0.82] | 0.003 | +0.43 pp [+0.09, +0.77] | 0.011 | +0.0007 [-0.0007, +0.0023] | 0.386 |
| LambdaRank con relevancia binaria de SKU | +0.0045 [+0.0031, +0.0059] | < 0.001 | +1.97 pp [+1.59, +2.37] | < 0.001 | -0.69 pp [-1.08, -0.27] | 0.002 | -0.0059 [-0.0077, -0.0041] | < 0.001 |
| LambdaRank sin ranking personal de categorias | +0.0000 [-0.0011, +0.0012] | 0.939 | +0.13 pp [-0.14, +0.41] | 0.342 | +0.09 pp [-0.25, +0.43] | 0.597 | -0.0003 [-0.0017, +0.0012] | 0.715 |

### Por perfil, sin senal de sesion

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.2753 | 0.2301 | 0.1967 | 0.2121 | 0.1769 | 0.6096 | 5.5474 |
| 1 - nuevo, carrito vacio | 554 | 0.2077 | 0.1817 | 0.1350 | 0.1549 | 0.1260 | 0.4801 | 5.0162 |
| 2 - nuevo, con articulos | 388 | 0.1922 | 0.1865 | 0.1119 | 0.1398 | 0.1245 | 0.4253 | 3.3608 |
| 3 - recurrente, carrito vacio | 9460 | 0.2897 | 0.2241 | 0.2163 | 0.2201 | 0.1789 | 0.6418 | 6.7346 |
| 4 - recurrente, con articulos | 7598 | 0.2666 | 0.2433 | 0.1811 | 0.2076 | 0.1808 | 0.5883 | 4.2197 |

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

| grupo | n_queries | ndcg_graded@5 | cat_hit_rate@5 | cat_precision@5 | sku_hit_rate@5 | sku_precision@5 |
| --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.3015 [0.2981, 0.3052] | 0.7954 [0.7898, 0.8011] | 0.3269 [0.3231, 0.3308] | 0.6157 [0.6088, 0.6228] | 0.2008 [0.1980, 0.2040] |
| 1 - nuevo, carrito vacio | 554 | 0.2330 [0.2161, 0.2523] | 0.7112 [0.6751, 0.7473] | 0.2560 [0.2361, 0.2758] | 0.4892 [0.4495, 0.5307] | 0.1386 [0.1260, 0.1538] |
| 2 - nuevo, con articulos | 388 | 0.2172 [0.1957, 0.2392] | 0.7088 [0.6623, 0.7552] | 0.2289 [0.2088, 0.2495] | 0.4304 [0.3840, 0.4794] | 0.1134 [0.0990, 0.1289] |
| 3 - recurrente, carrito vacio | 9460 | 0.3148 [0.3099, 0.3200] | 0.8057 [0.7980, 0.8139] | 0.3545 [0.3489, 0.3604] | 0.6420 [0.6327, 0.6520] | 0.2171 [0.2131, 0.2215] |
| 4 - recurrente, con articulos | 7598 | 0.2942 [0.2885, 0.2990] | 0.7932 [0.7843, 0.8028] | 0.3026 [0.2971, 0.3077] | 0.6016 [0.5904, 0.6123] | 0.1894 [0.1845, 0.1937] |

El sistema acierta la categoria en el **79.5%** de las cestas y el SKU exacto en el **61.6%**; el cociente entre las dos es **77.4%**. La distancia entre las dos columnas mide cuanto del error esta en *elegir la referencia* y no en *saber que categoria toca*. Desde la Fase 7a el surtido es de 8 referencias por categoria y el cliente repite su referencia preferida con la lealtad de la categoria (`DATA_SPEC.md`, "Fidelidad de marca").



## Carrito y diversidad (punto A2)

Con (casi siempre) una linea por categoria en cada cesta, un hueco del top-5 se *regala* si su categoria ya esta en el carrito (no puede acertar) o ya salio mas arriba en la lista (de las dos, como mucho acierta una). Dos piezas atacan el problema: tres features de carrito (`cat_in_cart`, `dept_n_in_cart`, `dept_share_in_cart`) y un re-ranking final (`src/recommender/rerank.py`). Reglas servidas: **como maximo 1 referencia(s) por categoria, categorias del carrito relegadas** (`RecommenderConfig.rerank`), las mismas en esta evaluacion, en las tablas de arriba y en la demo.

| Variante | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 | Huecos regalados | Huecos en cat. del carrito (perfiles 2 y 4) | Listas con carrito afectadas | Listas con cat. repetida | Categorias distintas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sin features de carrito, sin re-ranking (antes) | 0.7196 | 0.5953 | 0.2634 | 31.1% | 13.5% | 84.8% | 77.9% | 3.65 |
| Sin features de carrito + re-ranking servido | 0.7903 | 0.6114 | 0.2808 | 0.0% | 0.0% | 0.0% | 0.0% | 5.00 |
| Con features de carrito, sin re-ranking | 0.7182 | 0.6022 | 0.2670 | 32.8% | 1.4% | 83.6% | 83.6% | 3.38 |
| Con features de carrito + 1 por categoria | 0.7967 | 0.6159 | 0.2819 | 1.3% | 2.9% | 13.8% | 0.0% | 5.00 |
| **Con features de carrito + 1 por categoria + exclusion del carrito** | **0.7954** | **0.6157** | **0.2815** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **5.00** |

Las filas "sin features de carrito" usan un LambdaRank entrenado aparte con las mismas queries y sin esas tres columnas. Los huecos en categorias del carrito se miden contra el prefijo del ticket; la regla de exclusion usa el carrito en el corte, que ademas incluye los `add_to_cart` de sesion.

Frente a la fila "antes" (mismo entrenamiento, sin las tres features ni reglas), el sistema servido (en negrita) cambia cat_hit_rate@5 en **+7.59 pp** y sku_hit_rate@5 en **+2.04 pp**, y los huecos regalados pasan del 31.1% al 0.0%.



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
| Este sistema, F1@5 (media armonica de Precision@5 y Recall@5 medios) | 0.2165 |
| Este sistema, F1@5 por cesta (media del F1 de cada cesta) | 0.1807 |
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
  5.5 productos por adivinar de media, la Precision@5 y el
  Recall@5 estan acotados por el propio formato, acierte lo que acierte el modelo.
- **Que se optimiza.** El ranker se entrena con LambdaRank para NDCG@5 graduada (SKU y
  categoria), no para F1.
- **Contexto.** Aqui se predice a mitad de cesta: lo que ya esta en el carrito queda fuera
  del target. Instacart predice el pedido entero.
- **Agregacion.** Instacart promediaba el F1 de cada pedido; la variante mas cercana es la
  segunda fila ("por cesta"), no la de cabecera.


## Frente al dataset anterior a la Fase 8

La Fase 8 regenera el dataset con misiones de compra, tamano de cesta con cola larga, sustitucion entre categorias, segunda referencia en las categorias de exploracion y propension a la marca blanca (`DATA_SPEC.md`, "Estructura de la cesta"). Mismo codigo, mismas ventanas y mismo numero de queries de test; cambia el dato. Las cifras de antes estan congeladas en `reports/recommender/baseline_pre_fase8.json` (2026-09-17, commit `0a49c2e`), y el informe completo de entonces, con sus comparaciones antes/despues de los puntos A1, A2 y A4, en `snapshots/pre-fase-8/reports/recommender/metrics.md`.

| Metrica | Fase 7 (dataset 7a) | Fase 8 | Cambio |
| --- | ---: | ---: | ---: |
| NDCG@5 graduada | 0.2148 | 0.3015 | 1.40x |
| cat_hit_rate@5 | 0.7181 | 0.7954 | +7.73 pp |
| sku_hit_rate@5 | 0.5416 | 0.6157 | +7.41 pp |
| NDCG@5 SKU | 0.2006 | 0.2815 | 1.40x |
| Recall@5 | 0.1910 | 0.2350 | 1.23x |
| Precision@5 | 0.1451 | 0.2008 | 1.38x |
| F1@5 | 0.1649 | 0.2165 | 1.31x |
| Productos por adivinar (media) | 3.9327 | 5.5474 | 1.41x |

Son cestas distintas de datasets distintos, asi que no hay contraste pareado. Tampoco se puede leer el cambio como mejor o peor modelo: el target cambia de tamano y de estructura. Lo que dice cuanto del techo alcanza cada sistema en cada dataset esta en la seccion de diagnostico.

### Por perfil

| Grupo | n_queries 7a | Fase 8 | cat_hit_rate@5 7a | Fase 8 | sku_hit_rate@5 7a | Fase 8 | NDCG@5 graduada 7a | Fase 8 | Productos por adivinar 7a | Fase 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 18000 | 18000 | 0.7181 | 0.7954 | 0.5416 | 0.6157 | 0.2148 | 0.3015 | 3.93 | 5.55 |
| 1 - nuevo, carrito vacio | 521 | 554 | 0.5969 | 0.7112 | 0.4031 | 0.4892 | 0.1520 | 0.2330 | 4.17 | 5.02 |
| 2 - nuevo, con articulos | 421 | 388 | 0.4466 | 0.7088 | 0.2257 | 0.4304 | 0.1110 | 0.2172 | 2.34 | 3.36 |
| 3 - recurrente, carrito vacio | 8713 | 9460 | 0.7895 | 0.8057 | 0.6186 | 0.6420 | 0.2393 | 0.3148 | 5.03 | 6.73 |
| 4 - recurrente, con articulos | 8345 | 7598 | 0.6648 | 0.7932 | 0.4857 | 0.6016 | 0.1984 | 0.2942 | 2.85 | 4.22 |

## Frente a la Fase 3 original

La Fase 3 se entreno sobre el dataset anterior a la Fase 7a (1.500 productos, ~24 referencias por categoria y eleccion de SKU casi aleatoria). Sus cifras estan congeladas en `reports/recommender/baseline_fase3.json`. Mismo codigo, mismas ventanas y mismo numero de queries de test; cambia el dato.

| Metrica | Fase 3 (dataset viejo) | Fase 7c (dataset nuevo) | Cambio |
| --- | ---: | ---: | ---: |
| NDCG@5 | 0.0343 | 0.2815 | 8.21x |
| Recall@5 | 0.0332 | 0.2350 | 7.08x |
| Precision@5 | 0.0251 | 0.2008 | 8.01x |
| F1@5 | 0.0286 | 0.2165 | 7.58x |
| hit_rate@5 (SKU) | 0.1184 | 0.6157 | 5.20x |
| hit_rate@5 (categoria) | 0.5144 | 0.7954 | 1.55x |
| SKU / categoria (hit_rate) | 0.2302 | 0.7740 | 3.36x |
| SKU / categoria (precision) | 0.1371 | 0.6142 | 4.48x |

Un matiz al leerlo: el catalogo pasa de 1.500 a 496 productos, asi que un top-5 al azar tambien acierta mas que antes. El baseline de popularidad de la tabla de comparacion, sobre el mismo pool, es lo que aisla lo que aporta el ranker.

## Cold-start sobremuestreado (punto M4)

En la muestra de cabecera el cold-start son pocas queries (554 en el perfil 1 y 388 en el perfil 2). Aqui entran **todas** las cestas de la ventana de test sin compras anteriores a 2025-11-01: 3,140 cestas (1,495 anonimas y el resto de 496 clientes nuevos), cada una con los dos cortes de cabecera (carrito vacio y mitad del ticket): 5,802 queries. Las queries de una misma cesta estan correladas, asi que el bootstrap remuestrea cestas.

Estos clientes no tienen ninguna cesta antes del inicio del test, asi que no han entrado en ninguna fuente de candidatos (popularidad, afinidades, ALS) ni en el entrenamiento del ranker. Son clientes nunca vistos por ningun modelo. Lo unico que se sabe de ellos es su historial as-of dentro de la propia ventana, igual que en produccion. Mismos modelos y mismas fuentes que la cabecera.

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| perfiles 1 y 2 | 5802 | 3140 | 0.2228 [0.2161, 0.2302] | 0.7192 [0.7057, 0.7332] | 0.4555 [0.4414, 0.4717] | 0.1943 [0.1864, 0.2026] | 0.1700 [0.1624, 0.1780] | 4.5789 |
| 1 - nuevo, carrito vacio | 3140 | 3140 | 0.2429 [0.2351, 0.2517] | 0.7395 [0.7232, 0.7561] | 0.5054 [0.4876, 0.5233] | 0.2087 [0.2004, 0.2182] | 0.1691 [0.1608, 0.1777] | 5.5634 |
| 2 - nuevo, con articulos | 2662 | 2662 | 0.1991 [0.1914, 0.2070] | 0.6953 [0.6784, 0.7126] | 0.3967 [0.3779, 0.4155] | 0.1772 [0.1671, 0.1867] | 0.1710 [0.1604, 0.1814] | 3.4177 |

Entre corchetes, el intervalo de confianza al 95%: bootstrap percentil con 1,000 remuestreos de cestas (semilla 42; `evaluate.bootstrap_means`).

### Frente a los demas sistemas, por perfil

Diferencia = LambdaRank servido - el otro sistema (bootstrap pareado por cesta).

**1 - nuevo, carrito vacio.**

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (mismo pool) | +0.0167 [+0.0129, +0.0203] | < 0.001 | +2.07 pp [+1.05, +3.03] | < 0.001 | +3.89 pp [+2.74, +5.00] | < 0.001 | +0.0213 [+0.0165, +0.0261] | < 0.001 |
| LambdaRank sin senal de sesion | +0.0023 [+0.0005, +0.0042] | 0.018 | -0.06 pp [-0.73, +0.54] | 0.879 | +0.41 pp [-0.25, +1.08] | 0.272 | +0.0037 [+0.0016, +0.0062] | 0.002 |
| LambdaRank sin features de carrito | -0.0001 [-0.0019, +0.0021] | 0.957 | -0.25 pp [-0.96, +0.38] | 0.465 | +0.73 pp [+0.06, +1.46] | 0.046 | +0.0020 [-0.0003, +0.0047] | 0.113 |
| LambdaRank con relevancia binaria de SKU | -0.0001 [-0.0023, +0.0021] | 0.912 | +0.25 pp [-0.38, +0.89] | 0.444 | -0.86 pp [-1.50, -0.16] | 0.009 | -0.0039 [-0.0065, -0.0011] | 0.006 |
| LambdaRank sin ranking personal de categorias | -0.0009 [-0.0028, +0.0010] | 0.347 | +0.03 pp [-0.64, +0.70] | 0.967 | +0.16 pp [-0.57, +0.86] | 0.655 | +0.0008 [-0.0017, +0.0034] | 0.481 |

**2 - nuevo, con articulos.**

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (mismo pool) | +0.0274 [+0.0222, +0.0321] | < 0.001 | +5.86 pp [+4.51, +7.36] | < 0.001 | +5.63 pp [+4.21, +7.14] | < 0.001 | +0.0283 [+0.0216, +0.0352] | < 0.001 |
| LambdaRank sin senal de sesion | +0.0061 [+0.0035, +0.0086] | < 0.001 | +0.49 pp [-0.19, +1.20] | 0.203 | +1.24 pp [+0.30, +2.10] | 0.008 | +0.0082 [+0.0043, +0.0120] | 0.002 |
| LambdaRank sin features de carrito | +0.0012 [-0.0012, +0.0038] | 0.344 | +0.23 pp [-0.53, +1.09] | 0.613 | +0.23 pp [-0.68, +1.13] | 0.644 | +0.0006 [-0.0034, +0.0044] | 0.772 |
| LambdaRank con relevancia binaria de SKU | +0.0028 [-0.0003, +0.0057] | 0.078 | +1.84 pp [+0.83, +2.85] | 0.002 | -0.26 pp [-1.28, +0.64] | 0.652 | -0.0003 [-0.0047, +0.0039] | 0.887 |
| LambdaRank sin ranking personal de categorias | -0.0001 [-0.0026, +0.0022] | 0.948 | +0.04 pp [-0.71, +0.79] | 0.944 | +0.49 pp [-0.30, +1.28] | 0.254 | +0.0017 [-0.0021, +0.0053] | 0.363 |


## Techo de la primera etapa

Que parte del target llego siquiera al pool de candidatos. Lo que no esta aqui, el ranker
no lo puede recuperar.

| grupo | n_queries | pool_recall | pool_size_medio |
| --- | --- | --- | --- |
| total | 18000 | 0.8245 | 141.3904 |
| 1 - nuevo, carrito vacio | 554 | 0.5533 | 64.5325 |
| 2 - nuevo, con articulos | 388 | 0.6256 | 114.9536 |
| 3 - recurrente, carrito vacio | 9460 | 0.8302 | 124.9186 |
| 4 - recurrente, con articulos | 7598 | 0.8474 | 168.8530 |

## Que features usa el ranker

Importancia por ganancia, las 25 primeras.

| feature | gain | split |
| --- | --- | --- |
| category_idx | 99842.2167 | 1553 |
| hist_rank | 70311.3716 | 259 |
| cat_overdue_ratio | 26058.5815 | 869 |
| cat_days_since | 15257.5715 | 832 |
| cat_freq_share | 14731.5080 | 616 |
| hist_n_baskets | 7870.7784 | 137 |
| cat_due_rank | 7657.6997 | 224 |
| cat_in_cart | 7624.1887 | 272 |
| typical_repurchase_days | 6451.1697 | 211 |
| prefix_size | 5590.9380 | 408 |
| dept_share_in_cart | 5242.7116 | 334 |
| cust_recency_days | 4508.7350 | 434 |
| cust_frequency | 4460.4566 | 317 |
| sess_secs_since_view | 4285.3377 | 123 |
| cust_avg_ticket | 4237.5486 | 498 |
| cat_freq_rank | 4175.3812 | 169 |
| cat_expected_days | 3841.5515 | 468 |
| aff_conf_sum | 3728.7056 | 104 |
| prod_pop_all | 3453.5709 | 71 |
| dept_n_in_cart | 3429.3057 | 150 |
| cataff_conf_max | 3133.2299 | 150 |
| cust_n_products | 3131.8558 | 342 |
| aff_lift_max | 3092.4995 | 133 |
| cat_n_purchase_days | 2091.0511 | 232 |
| hist_units | 1936.7693 | 128 |

<!-- diagnostics:start -->
## Diagnostico: baselines independientes del pool y techo teorico

Generado por `python -m src.recommender.verify_recommender_diagnostics` (2026-09-17), sobre las 18,000 queries de test de las predicciones en disco (sha256 de `recommendations_test.parquet`: `4e502725e32a`). El oraculo necesita antes `python -m data_generation.export_oracle`. Esta seccion no la reescribe el pipeline: si se reentrena, hay que volver a lanzar el verificador. Puntos A3, A4, A6 y M8 de `docs/diagnostico-fase7.md`. La metrica principal es la NDCG@5 graduada (3 por SKU exacto, 1 por categoria; `evaluate.category_metrics`).

### Lectura rapida

- **Mejor baseline en cat_hit_rate@5:** Reglas de asociacion de categoria + referencia lider, 0.7594 frente a 0.7954 del LambdaRank: -3.6 pp, el LambdaRank va por delante (LambdaRank - baseline: +3.61 pp [+3.04, +4.17], p < 0.001).
- **Techo teorico de cat_hit_rate@5:** 0.8382. El LambdaRank alcanza el 94.9% y el mejor baseline el 90.6%.
- **Techo de sku_hit_rate@5:** 0.6677. El LambdaRank alcanza el 92.2%.
- **Objetivo del ranker (A4):** con relevancia binaria de SKU, cat_hit_rate@5 0.7758 (+1.6 pp frente al mejor baseline); con la graduada servida, 0.7954 (+3.6 pp). sku_hit_rate@5 pasa de 0.6226 a 0.6157 y la NDCG@5 graduada de 0.2970 a 0.3015.

### Todos los sistemas, total

Las columnas "% techo" dividen por el oraculo correspondiente sobre las mismas queries. "Huecos regalados" es la parte del top-5 en una categoria que ya esta en el carrito o repetida mas arriba en la lista (`evaluate.wasted_slot_metrics`, punto A2). Los baselines de categoria no regalan ninguno por construccion; el LambdaRank los evita con el re-ranking de `RecommenderConfig.rerank`.

| Sistema | NDCG@5 graduada | cat_hit_rate@5 | % techo | cat_precision@5 | sku_hit_rate@5 | % techo SKU | sku_precision@5 | NDCG@5 SKU | Huecos regalados |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.3310 | 0.8382 | — | 0.3762 | 0.6425 | — | 0.2124 | 0.2941 | 0.0% |
| **Oraculo de SKU (techo)** | 0.3406 | 0.8292 | — | 0.3490 | 0.6677 | — | 0.2354 | 0.3283 | 0.0% |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.3015 | 0.7954 | 94.9% | 0.3269 | 0.6157 | 92.2% | 0.2008 | 0.2815 | 0.0% |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.2970 | 0.7758 | 92.6% | 0.3074 | 0.6226 | 93.3% | 0.2063 | 0.2874 | 0.0% |
| Reglas de asociacion de categoria + referencia lider | 0.2117 | 0.7594 | 90.6% | 0.3067 | 0.4012 | 60.1% | 0.1058 | 0.1475 | 0.0% |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.2479 | 0.7493 | 89.4% | 0.3032 | 0.5276 | 79.0% | 0.1555 | 0.2093 | 0.0% |
| Frecuencia personal por categoria + referencia favorita | 0.2414 | 0.7405 | 88.3% | 0.2960 | 0.5120 | 76.7% | 0.1489 | 0.2029 | 0.0% |
| Popularidad de categoria + referencia lider | 0.2039 | 0.7383 | 88.1% | 0.2972 | 0.3949 | 59.1% | 0.1043 | 0.1417 | 0.0% |
| Popularidad global (SKU) | 0.2104 | 0.7173 | 85.6% | 0.2909 | 0.4136 | 61.9% | 0.1103 | 0.1572 | 2.5% |
| Repetir las referencias favoritas del cliente | 0.2455 | 0.6989 | 83.4% | 0.2701 | 0.5441 | 81.5% | 0.1656 | 0.2295 | 3.6% |
| Aleatorio | 0.0395 | 0.3137 | 37.4% | 0.0882 | 0.0515 | 7.7% | 0.0108 | 0.0126 | 2.9% |

### Frente al dataset anterior a la Fase 8

Los mismos sistemas, medidos con el mismo codigo sobre el dataset de la Fase 7a (`reports/recommender/baseline_pre_fase8.json`, 2026-09-17, commit `0a49c2e`; el resto de artefactos de ese momento esta en `snapshots/pre-fase-8/`). Son queries distintas de datasets distintos: la comparacion es de nivel, no pareada. El techo tambien cambia, asi que la columna que dice cuanto mejora cada sistema *respecto a lo alcanzable* es el % del techo.

| Sistema | NDCG@5 graduada antes | ahora | cat_hit_rate@5 antes | ahora | % techo antes | ahora | sku_hit_rate@5 antes | ahora | % techo SKU antes | ahora |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.2308 | 0.3310 | 0.7816 | 0.8382 | — | — | 0.5542 | 0.6425 | — | — |
| **Oraculo de SKU (techo)** | 0.2340 | 0.3406 | 0.7309 | 0.8292 | — | — | 0.5920 | 0.6677 | — | — |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.2148 | 0.3015 | 0.7181 | 0.7954 | 91.9% | 94.9% | 0.5416 | 0.6157 | 91.5% | 92.2% |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.2083 | 0.2970 | 0.6775 | 0.7758 | 86.7% | 92.6% | 0.5510 | 0.6226 | 93.1% | 93.3% |
| Reglas de asociacion de categoria + referencia lider | 0.1222 | 0.2117 | 0.6096 | 0.7594 | 78.0% | 90.6% | 0.2586 | 0.4012 | 43.7% | 60.1% |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.1680 | 0.2479 | 0.6789 | 0.7493 | 86.9% | 89.4% | 0.4214 | 0.5276 | 71.2% | 79.0% |
| Frecuencia personal por categoria + referencia favorita | 0.1564 | 0.2414 | 0.6489 | 0.7405 | 83.0% | 88.3% | 0.3994 | 0.5120 | 67.5% | 76.7% |
| Popularidad de categoria + referencia lider | 0.1237 | 0.2039 | 0.6109 | 0.7383 | 78.2% | 88.1% | 0.2651 | 0.3949 | 44.8% | 59.1% |
| Popularidad global (SKU) | 0.1198 | 0.2104 | 0.5799 | 0.7173 | 74.2% | 85.6% | 0.2662 | 0.4136 | 45.0% | 61.9% |
| Repetir las referencias favoritas del cliente | 0.1616 | 0.2455 | 0.5969 | 0.6989 | 76.4% | 83.4% | 0.4481 | 0.5441 | 75.7% | 81.5% |
| Aleatorio | 0.0301 | 0.0395 | 0.2754 | 0.3137 | 35.2% | 37.4% | 0.0408 | 0.0515 | 6.9% | 7.7% |

### LambdaRank frente a cada sistema (bootstrap pareado, punto M4)

Diferencia = LambdaRank servido - el otro sistema, sobre las mismas queries. Cada remuestreo sortea las mismas cestas para los dos (1,000 remuestreos, intervalo percentil al 95%; `evaluate.paired_bootstrap`). El p-valor es bilateral y no esta corregido por comparaciones multiples. Las tasas van en puntos porcentuales. El desglose por perfil esta en `diagnostics.json` (`comparisons`).

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | +0.0045 [+0.0031, +0.0059] | < 0.001 | +1.97 pp [+1.59, +2.37] | < 0.001 | -0.69 pp [-1.08, -0.27] | 0.002 |
| Reglas de asociacion de categoria + referencia lider | +0.0898 [+0.0870, +0.0929] | < 0.001 | +3.61 pp [+3.04, +4.17] | < 0.001 | +21.44 pp [+20.73, +22.32] | < 0.001 |
| Frecuencia personal x due_for_repurchase + referencia favorita | +0.0536 [+0.0511, +0.0562] | < 0.001 | +4.61 pp [+4.07, +5.12] | < 0.001 | +8.81 pp [+8.20, +9.43] | < 0.001 |
| Frecuencia personal por categoria + referencia favorita | +0.0601 [+0.0577, +0.0628] | < 0.001 | +5.49 pp [+4.94, +6.01] | < 0.001 | +10.37 pp [+9.76, +10.98] | < 0.001 |
| Popularidad de categoria + referencia lider | +0.0976 [+0.0948, +0.1009] | < 0.001 | +5.72 pp [+5.11, +6.26] | < 0.001 | +22.08 pp [+21.36, +22.94] | < 0.001 |
| Popularidad global (SKU) | +0.0911 [+0.0885, +0.0941] | < 0.001 | +7.81 pp [+7.20, +8.39] | < 0.001 | +20.21 pp [+19.47, +20.99] | < 0.001 |
| Repetir las referencias favoritas del cliente | +0.0559 [+0.0535, +0.0584] | < 0.001 | +9.66 pp [+9.15, +10.20] | < 0.001 | +7.16 pp [+6.62, +7.71] | < 0.001 |
| Aleatorio | +0.2620 [+0.2587, +0.2656] | < 0.001 | +48.17 pp [+47.37, +48.99] | < 0.001 | +56.42 pp [+55.69, +57.14] | < 0.001 |

### cat_hit_rate@5 por perfil (entre parentesis, % del techo del perfil)

| Sistema | Total | Perfil 1 - nuevo, carrito vacio | Perfil 2 - nuevo, con articulos | Perfil 3 - recurrente, carrito vacio | Perfil 4 - recurrente, con articulos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.8382 | 0.7437 | 0.7861 | 0.8410 | 0.8443 |
| **Oraculo de SKU (techo)** | 0.8292 | 0.7148 | 0.7680 | 0.8314 | 0.8379 |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.7954 (94.9%) | 0.7112 (95.6%) | 0.7088 (90.2%) | 0.8057 (95.8%) | 0.7932 (94.0%) |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.7758 (92.6%) | 0.7040 (94.7%) | 0.6907 (87.9%) | 0.7838 (93.2%) | 0.7753 (91.8%) |
| Reglas de asociacion de categoria + referencia lider | 0.7594 (90.6%) | 0.7058 (94.9%) | 0.7345 (93.4%) | 0.7739 (92.0%) | 0.7465 (88.4%) |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.7493 (89.4%) | 0.6733 (90.5%) | 0.6521 (83.0%) | 0.7813 (92.9%) | 0.7201 (85.3%) |
| Frecuencia personal por categoria + referencia favorita | 0.7405 (88.3%) | 0.6733 (90.5%) | 0.6469 (82.3%) | 0.7763 (92.3%) | 0.7056 (83.6%) |
| Popularidad de categoria + referencia lider | 0.7383 (88.1%) | 0.7058 (94.9%) | 0.6804 (86.6%) | 0.7739 (92.0%) | 0.6993 (82.8%) |
| Popularidad global (SKU) | 0.7173 (85.6%) | 0.7022 (94.4%) | 0.6443 (82.0%) | 0.7540 (89.7%) | 0.6765 (80.1%) |
| Repetir las referencias favoritas del cliente | 0.6989 (83.4%) | 0.6751 (90.8%) | 0.6211 (79.0%) | 0.7340 (87.3%) | 0.6608 (78.3%) |
| Aleatorio | 0.3137 (37.4%) | 0.2852 (38.3%) | 0.2345 (29.8%) | 0.3500 (41.6%) | 0.2747 (32.5%) |

### sku_hit_rate@5 por perfil (entre parentesis, % del techo de SKU del perfil)

| Sistema | Total | Perfil 1 - nuevo, carrito vacio | Perfil 2 - nuevo, con articulos | Perfil 3 - recurrente, carrito vacio | Perfil 4 - recurrente, con articulos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.6425 | 0.5018 | 0.4948 | 0.6575 | 0.6416 |
| **Oraculo de SKU (techo)** | 0.6677 | 0.5144 | 0.5129 | 0.6866 | 0.6632 |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.6157 (92.2%) | 0.4892 (95.1%) | 0.4304 (83.9%) | 0.6420 (93.5%) | 0.6016 (90.7%) |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.6226 (93.3%) | 0.4928 (95.8%) | 0.4149 (80.9%) | 0.6479 (94.4%) | 0.6112 (92.2%) |
| Reglas de asociacion de categoria + referencia lider | 0.4012 (60.1%) | 0.4224 (82.1%) | 0.3943 (76.9%) | 0.4412 (64.3%) | 0.3502 (52.8%) |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.5276 (79.0%) | 0.4314 (83.9%) | 0.4021 (78.4%) | 0.5705 (83.1%) | 0.4875 (73.5%) |
| Frecuencia personal por categoria + referencia favorita | 0.5120 (76.7%) | 0.4278 (83.2%) | 0.3892 (75.9%) | 0.5569 (81.1%) | 0.4685 (70.6%) |
| Popularidad de categoria + referencia lider | 0.3949 (59.1%) | 0.4224 (82.1%) | 0.3763 (73.4%) | 0.4412 (64.3%) | 0.3361 (50.7%) |
| Popularidad global (SKU) | 0.4136 (61.9%) | 0.4458 (86.7%) | 0.3789 (73.9%) | 0.4575 (66.6%) | 0.3583 (54.0%) |
| Repetir las referencias favoritas del cliente | 0.5441 (81.5%) | 0.4477 (87.0%) | 0.3892 (75.9%) | 0.5877 (85.6%) | 0.5046 (76.1%) |
| Aleatorio | 0.0515 (7.7%) | 0.0307 (6.0%) | 0.0335 (6.5%) | 0.0612 (8.9%) | 0.0419 (6.3%) |

### Huecos en categorias recien compradas (punto A1)

Cada hueco del top-5 se clasifica por los dias desde que el cliente compro por ultima vez su categoria, con **todo** su historial anterior al dia de la cesta (lo viera o no el modelo). El generador castiga con fuerza reponer justo despues de comprar, asi que los huecos de los primeros dias casi nunca aciertan. "En la ventana" es desde el inicio del test. Precision = parte de esos huecos que acierta el SKU o la categoria.

| Huecos en categorias compradas... | LambdaRank servido: relevancia graduada (predicciones en disco): % huecos | SKU prec. | cat. prec. | Frecuencia personal x due_for_repurchase + referencia favorita: % huecos | SKU prec. | cat. prec. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <= 7 dias | 9.2% | 20.5% | 37.2% | 20.7% | 11.5% | 23.7% |
| <= 14 dias | 22.2% | 21.2% | 37.6% | 33.2% | 14.3% | 28.9% |
| 8-14 dias | 13.0% | 21.7% | 37.9% | 12.5% | 19.0% | 37.4% |
| comprada en la ventana, antes de la cesta | 40.8% | 21.3% | 35.9% | 46.4% | 15.8% | 31.1% |
| sin compra en la ventana | 59.2% | 19.2% | 30.4% | 53.6% | 15.4% | 29.7% |
| total | 100.0% | 20.1% | 32.7% | 100.0% | 15.6% | 30.3% |

### Como se construye el techo

El generador exporta, para cada cesta de test, el peso de cada categoria antes de aplicar la mision (afinidad x estacionalidad x ciclo de reposicion), la probabilidad de cada mision de compra en esa visita, su factor de tamano y la referencia mas probable de cada categoria (`data/oracle/`, fuera de `data/raw`); el manifiesto trae los perfiles de mision, el tamano de cada una y la matriz de complementos y sustitutos. El oraculo conoce el carrito y cuantas categorias distintas tiene la cesta, pero no la mision. La probabilidad de que cada categoria este en el resto se calcula por Monte Carlo con muestreo secuencial por importancia: replica el sorteo del generador paso a paso, forzando que el carrito quede dentro, y pondera cada muestra por su probabilidad real (exacto salvo ruido de muestreo; detalle en `src/recommender/oracle.py`). El **oraculo de categoria** recomienda las 5 categorias de mas probabilidad fuera del carrito, cada una con su referencia mas probable; el **oraculo de SKU** ordena por `P(categoria en el resto) x P(mejor referencia en la cesta)`. Las dos listas se eligen con la mitad de las muestras y se miden con la otra, asi que el techo es una cota inferior muy ajustada del maximo. Ningun sistema que solo vea el pasado puede superarlos en valor esperado: lo que queda entre el techo y el 100 % es entropia del generador.

El realizado y el esperado deben coincidir salvo ruido (ver comprobaciones):

| Techo | Realizado (estas queries) | Esperado (Monte Carlo) |
| --- | ---: | ---: |
| cat_hit_rate@5 (oraculo de categoria) | 0.8382 | 0.8385 ± 0.0048 |
| cat_precision@5 (oraculo de categoria) | 0.3762 | 0.3765 |
| sku_hit_rate@5 (oraculo de SKU) | 0.6677 | 0.6661 |
| sku_precision@5 (oraculo de SKU) | 0.2354 | 0.2350 |

Muestras por query: 1,000 (mitad de evaluacion: tamano efectivo mediano 151, percentil 5 11; queries sin ninguna muestra valida: 0). Reglas de asociacion: 3031 de 3782 pares con confianza >= 0.10 y lift > 1.

### Techo esperado por perfil

| grupo | cat_hit_rate@5 | cat_precision@5 | sku_hit_rate@5 | sku_precision@5 |
| --- | --- | --- | --- | --- |
| total | 0.8385 | 0.3765 | 0.6661 | 0.2350 |
| 1 - nuevo, carrito vacio | 0.7424 | 0.2965 | 0.5111 | 0.1558 |
| 2 - nuevo, con articulos | 0.7591 | 0.2634 | 0.4590 | 0.1239 |
| 3 - recurrente, carrito vacio | 0.8386 | 0.4055 | 0.6851 | 0.2574 |
| 4 - recurrente, con articulos | 0.8495 | 0.3520 | 0.6644 | 0.2187 |

### Comprobaciones

- OK — La ablacion con relevancia binaria cubre las mismas queries que el LambdaRank
- OK — Target en categorias <= target en lineas (segunda referencia de la Fase 8) (17.8% de queries con dos lineas de una categoria en el target; 8.7% con una categoria a los dos lados del corte)
- OK — LambdaRank recalculado desde predictions/ = reports/recommender/metrics.json (0.7954 frente a 0.7954)
- OK — Techo realizado compatible con el esperado en las 4 metricas, total y por perfil (max |z| <= 3.5) (peor: sku_precision_p2, z = +2.26)
- OK — Ningun sistema supera al oraculo de categoria en cat_hit_rate (mejor sistema lambdarank = 0.7954)
<!-- diagnostics:end -->

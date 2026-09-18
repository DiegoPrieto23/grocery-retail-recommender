# Recomendador de cesta (Fase 3, Tarea 3a)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra de este informe se copia a
mano: se recalcula ejecutando ese comando.

## Montaje

| | |
| --- | --- |
| Fuentes de candidatos (ranker) | cestas anteriores a 2025-09-01 |
| Queries de entrenamiento | 2025-09-01 a 2025-11-01 (11,196 cestas con algun candidato relevante en el pool; 11,059 con el SKU exacto) |
| Fuentes de candidatos (test) | cestas anteriores a 2025-11-01 |
| Queries de test | desde 2025-11-01 (18,000 cestas) |
| Ranker | LightGBM `lambdarank`, 119 arboles |
| Objetivo | NDCG@5 con relevancia graduada: 2 SKU exacto, 1 misma categoria (`label_gain` = [0.0, 1.0, 3.0]) |
| NDCG@5 graduada de validacion | 0.3216 |

El split es temporal **y por cesta**: ninguna cesta se reparte entre train y test, y las
fuentes de candidatos se reajustan para cada ventana con solo el pasado de esa ventana.

## Metrica principal

La metrica principal del recomendador es la **NDCG@5 con relevancia graduada**
(`CHALLENGE.md`, punto A4 de `docs/diagnostico-fase7.md`): 3 puntos por hueco si es el SKU
exacto, 1 si solo acierta la categoria. Es la que optimiza el LambdaRank. Se lee siempre
junto a `cat_hit_rate@5` (lo que ensena la demo) y `sku_hit_rate@5`.

| grupo | n_queries | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 |
| --- | --- | --- | --- | --- |
| total | 18000 | 0.3018 [0.2983, 0.3057] | 0.7981 [0.7926, 0.8042] | 0.6122 [0.6052, 0.6192] |
| 1 - nuevo, carrito vacio | 554 | 0.2312 [0.2140, 0.2506] | 0.6968 [0.6588, 0.7365] | 0.4747 [0.4350, 0.5181] |
| 2 - nuevo, con articulos | 388 | 0.2176 [0.1963, 0.2398] | 0.7191 [0.6753, 0.7629] | 0.4175 [0.3686, 0.4665] |
| 3 - recurrente, carrito vacio | 9460 | 0.3145 [0.3095, 0.3196] | 0.8080 [0.8004, 0.8163] | 0.6408 [0.6313, 0.6508] |
| 4 - recurrente, con articulos | 7598 | 0.2954 [0.2898, 0.3006] | 0.7972 [0.7886, 0.8060] | 0.5965 [0.5850, 0.6069] |

Entre corchetes, el intervalo de confianza al 95%: bootstrap percentil con 1,000 remuestreos de cestas (semilla 42; `evaluate.bootstrap_means`). En el grupo mas pequeno (2 - nuevo, con articulos, n = 388) el intervalo de cat_hit_rate@5 mide 8.8 puntos: las diferencias entre perfiles pequenos se leen con eso delante. Los perfiles 1 y 2, con muchas mas queries, estan en "Cold-start sobremuestreado".

El acierto segun cuantas lineas lleva ya el carrito (varios cortes por cesta, punto M3) esta en [`cuts.md`](cuts.md).

## Objetivo del ranker (punto A4)

Hasta el punto A4 el LambdaRank optimizaba la NDCG@5 con relevancia binaria de SKU exacto. Ahora optimiza la graduada (`RankerConfig.relevance`) y tiene tres features nuevas de ranking personal de categorias (`CUSTOMER_CATEGORY_RANK_FEATURES`: `cat_freq_rank`, `cat_freq_share`, `cat_due_rank`). Las variantes se entrenan con las mismas queries y se evaluan sobre el mismo pool, con el mismo re-ranking servido.

| Variante | NDCG@5 graduada | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 SKU | cat_precision@5 | sku_precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Relevancia binaria de SKU (objetivo anterior) | 0.2967 | 0.7774 | 0.6219 | 0.2868 | 0.3083 | 0.2056 |
| Relevancia graduada, sin ranking personal | 0.3021 | 0.7990 | 0.6137 | 0.2809 | 0.3294 | 0.1993 |
| **Relevancia graduada + ranking personal (servido)** | **0.3018** | **0.7981** | **0.6122** | **0.2799** | **0.3295** | **0.1988** |

Cambiar solo el objetivo (fila de relevancia binaria a graduada sin ranking) mueve cat_hit_rate@5 **+2.16 pp** y sku_hit_rate@5 **-0.82 pp**. Anadir el ranking personal mueve cat_hit_rate@5 **-0.09 pp** y sku_hit_rate@5 **-0.15 pp**. En conjunto, frente al objetivo anterior: cat_hit_rate@5 **+2.07 pp**, sku_hit_rate@5 **-0.97 pp**.



### Por perfil: relevancia binaria de SKU frente a la graduada servida

| Grupo | cat_hit_rate@5 SKU | graduada | cambio | sku_hit_rate@5 SKU | graduada | cambio | NDCG@5 graduada SKU | graduada |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 0.7774 | 0.7981 | +2.07 pp | 0.6219 | 0.6122 | -0.97 pp | 0.2967 | 0.3018 |
| 1 - nuevo, carrito vacio | 0.7022 | 0.6968 | -0.54 pp | 0.4910 | 0.4747 | -1.62 pp | 0.2325 | 0.2312 |
| 2 - nuevo, con articulos | 0.6649 | 0.7191 | +5.41 pp | 0.4201 | 0.4175 | -0.26 pp | 0.2098 | 0.2176 |
| 3 - recurrente, carrito vacio | 0.7872 | 0.8080 | +2.08 pp | 0.6466 | 0.6408 | -0.58 pp | 0.3089 | 0.3145 |
| 4 - recurrente, con articulos | 0.7765 | 0.7972 | +2.07 pp | 0.6110 | 0.5965 | -1.45 pp | 0.2905 | 0.2954 |

### Importancia del ranking personal de categorias

Puesto entre todas las features del modelo servido (ganancia).

| Feature | Puesto por ganancia | Ganancia | % de la ganancia total | Splits |
| --- | ---: | ---: | ---: | ---: |
| `cat_due_rank` | 4 de 60 | 15873 | 4.3% | 189 |
| `cat_freq_share` | 6 de 60 | 12362 | 3.3% | 406 |
| `cat_freq_rank` | 23 de 60 | 2415 | 0.7% | 114 |

La comparacion con los baselines de categoria y con el techo teorico esta en la seccion de diagnostico, al final (`verify_recommender_diagnostics`).

## Resultado a nivel de SKU

Las metricas de SKU exacto de siempre (NDCG@5 binaria, recall, precision, F1), con su
intervalo de confianza.

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.2799 [0.2760, 0.2842] | 0.2338 [0.2297, 0.2383] | 0.1988 [0.1960, 0.2022] | 0.2149 [0.2120, 0.2181] | 0.1794 [0.1768, 0.1820] | 0.6122 [0.6052, 0.6192] | 5.5474 |
| 1 - nuevo, carrito vacio | 554 | 0.2112 [0.1916, 0.2335] | 0.1827 [0.1613, 0.2075] | 0.1329 [0.1198, 0.1477] | 0.1538 [0.1399, 0.1694] | 0.1251 [0.1132, 0.1381] | 0.4747 [0.4350, 0.5181] | 5.0162 |
| 2 - nuevo, con articulos | 388 | 0.1923 [0.1657, 0.2195] | 0.1908 [0.1637, 0.2196] | 0.1155 [0.1005, 0.1320] | 0.1439 [0.1256, 0.1629] | 0.1277 [0.1116, 0.1448] | 0.4175 [0.3686, 0.4665] | 3.3608 |
| 3 - recurrente, carrito vacio | 9460 | 0.2879 [0.2823, 0.2935] | 0.2231 [0.2178, 0.2288] | 0.2149 [0.2107, 0.2192] | 0.2189 [0.2151, 0.2232] | 0.1781 [0.1750, 0.1816] | 0.6408 [0.6313, 0.6508] | 6.7346 |
| 4 - recurrente, con articulos | 7598 | 0.2795 [0.2731, 0.2859] | 0.2530 [0.2467, 0.2595] | 0.1880 [0.1833, 0.1922] | 0.2157 [0.2108, 0.2200] | 0.1876 [0.1833, 0.1913] | 0.5965 [0.5850, 0.6069] | 4.2197 |

## Comparacion

Mismo pool de candidatos, distinta forma de ordenarlo. Es lo que aisla la aportacion del
ranker de la de la primera etapa.

| Sistema | NDCG@5 | Recall@5 | Precision@5 | F1@5 | hit_rate@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (sin aprendizaje) | 0.1565 | 0.1273 | 0.1088 | 0.1173 | 0.4082 |
| LambdaRank sin senal de sesion | 0.2737 | 0.2291 | 0.1950 | 0.2107 | 0.6058 |
| LambdaRank con relevancia binaria de SKU (objetivo anterior) | 0.2868 | 0.2376 | 0.2056 | 0.2205 | 0.6219 |
| **LambdaRank completo** | **0.2799** | **0.2338** | **0.1988** | **0.2149** | **0.6122** |

### Diferencias con intervalo de confianza (bootstrap pareado)

Diferencia = LambdaRank servido - el otro sistema, sobre las mismas queries y con el mismo re-ranking. Cada remuestreo sortea las mismas cestas para los dos sistemas (`evaluate.paired_bootstrap`, 1,000 remuestreos). El p-valor es bilateral y no esta corregido por comparaciones multiples. Las tasas van en puntos porcentuales. El desglose por perfil esta en `metrics.json` (`comparisons`).

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (mismo pool) | +0.0930 [+0.0902, +0.0959] | < 0.001 | +7.37 pp [+6.78, +7.95] | < 0.001 | +20.39 pp [+19.67, +21.17] | < 0.001 | +0.1234 [+0.1195, +0.1274] | < 0.001 |
| LambdaRank sin senal de sesion | +0.0052 [+0.0040, +0.0063] | < 0.001 | +0.36 pp [+0.11, +0.59] | 0.003 | +0.64 pp [+0.33, +0.96] | < 0.001 | +0.0062 [+0.0046, +0.0077] | < 0.001 |
| LambdaRank sin features de carrito | +0.0023 [+0.0012, +0.0033] | < 0.001 | +0.48 pp [+0.20, +0.78] | < 0.001 | +0.14 pp [-0.18, +0.49] | 0.404 | +0.0003 [-0.0012, +0.0017] | 0.731 |
| LambdaRank con relevancia binaria de SKU | +0.0051 [+0.0037, +0.0065] | < 0.001 | +2.07 pp [+1.68, +2.43] | < 0.001 | -0.97 pp [-1.38, -0.53] | < 0.001 | -0.0069 [-0.0087, -0.0050] | < 0.001 |
| LambdaRank sin ranking personal de categorias | -0.0003 [-0.0014, +0.0008] | 0.593 | -0.09 pp [-0.36, +0.18] | 0.557 | -0.15 pp [-0.49, +0.18] | 0.397 | -0.0010 [-0.0024, +0.0004] | 0.218 |
| LambdaRank con validacion por hash (antes del punto B4) | -0.0001 [-0.0011, +0.0009] | 0.834 | +0.24 pp [-0.02, +0.51] | 0.077 | -0.15 pp [-0.48, +0.18] | 0.389 | -0.0009 [-0.0023, +0.0006] | 0.212 |

### Por perfil, sin senal de sesion

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.2737 | 0.2291 | 0.1950 | 0.2107 | 0.1757 | 0.6058 | 5.5474 |
| 1 - nuevo, carrito vacio | 554 | 0.2086 | 0.1798 | 0.1321 | 0.1523 | 0.1238 | 0.4693 | 5.0162 |
| 2 - nuevo, con articulos | 388 | 0.1861 | 0.1855 | 0.1098 | 0.1379 | 0.1223 | 0.4046 | 3.3608 |
| 3 - recurrente, carrito vacio | 9460 | 0.2869 | 0.2216 | 0.2136 | 0.2175 | 0.1768 | 0.6365 | 6.7346 |
| 4 - recurrente, con articulos | 7598 | 0.2665 | 0.2443 | 0.1809 | 0.2079 | 0.1808 | 0.5878 | 4.2197 |

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

| grupo | n_queries | ndcg_graded@5 | cat_hit_rate@5 | cat_precision@5 | sku_hit_rate@5 | sku_precision@5 |
| --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.3018 [0.2983, 0.3057] | 0.7981 [0.7926, 0.8042] | 0.3295 [0.3257, 0.3336] | 0.6122 [0.6052, 0.6192] | 0.1988 [0.1960, 0.2022] |
| 1 - nuevo, carrito vacio | 554 | 0.2312 [0.2140, 0.2506] | 0.6968 [0.6588, 0.7365] | 0.2513 [0.2314, 0.2711] | 0.4747 [0.4350, 0.5181] | 0.1329 [0.1198, 0.1477] |
| 2 - nuevo, con articulos | 388 | 0.2176 [0.1963, 0.2398] | 0.7191 [0.6753, 0.7629] | 0.2361 [0.2165, 0.2567] | 0.4175 [0.3686, 0.4665] | 0.1155 [0.1005, 0.1320] |
| 3 - recurrente, carrito vacio | 9460 | 0.3145 [0.3095, 0.3196] | 0.8080 [0.8004, 0.8163] | 0.3568 [0.3512, 0.3628] | 0.6408 [0.6313, 0.6508] | 0.2149 [0.2107, 0.2192] |
| 4 - recurrente, con articulos | 7598 | 0.2954 [0.2898, 0.3006] | 0.7972 [0.7886, 0.8060] | 0.3059 [0.3004, 0.3112] | 0.5965 [0.5850, 0.6069] | 0.1880 [0.1833, 0.1922] |

El sistema acierta la categoria en el **79.8%** de las cestas y el SKU exacto en el **61.2%**; el cociente entre las dos es **76.7%**. La distancia entre las dos columnas mide cuanto del error esta en *elegir la referencia* y no en *saber que categoria toca*. Desde la Fase 7a el surtido es de 8 referencias por categoria y el cliente repite su referencia preferida con la lealtad de la categoria (`DATA_SPEC.md`, "Fidelidad de marca").



## Carrito y diversidad (punto A2)

Con (casi siempre) una linea por categoria en cada cesta, un hueco del top-5 se *regala* si su categoria ya esta en el carrito (no puede acertar) o ya salio mas arriba en la lista (de las dos, como mucho acierta una). Dos piezas atacan el problema: tres features de carrito (`cat_in_cart`, `dept_n_in_cart`, `dept_share_in_cart`) y un re-ranking final (`src/recommender/rerank.py`). Reglas servidas: **como maximo 1 referencia(s) por categoria, categorias del carrito relegadas** (`RecommenderConfig.rerank`), las mismas en esta evaluacion, en las tablas de arriba y en la demo.

| Variante | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 | Huecos regalados | Huecos en cat. del carrito (perfiles 2 y 4) | Listas con carrito afectadas | Listas con cat. repetida | Categorias distintas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sin features de carrito, sin re-ranking (antes) | 0.7167 | 0.5907 | 0.2604 | 33.0% | 13.8% | 86.2% | 79.8% | 3.55 |
| Sin features de carrito + re-ranking servido | 0.7933 | 0.6107 | 0.2796 | 0.0% | 0.0% | 0.0% | 0.0% | 5.00 |
| Con features de carrito, sin re-ranking | 0.7192 | 0.5984 | 0.2653 | 33.5% | 1.5% | 86.8% | 86.1% | 3.35 |
| Con features de carrito + 1 por categoria | 0.7992 | 0.6128 | 0.2804 | 1.3% | 3.0% | 14.2% | 0.0% | 5.00 |
| **Con features de carrito + 1 por categoria + exclusion del carrito** | **0.7981** | **0.6122** | **0.2799** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **5.00** |

Las filas "sin features de carrito" usan un LambdaRank entrenado aparte con las mismas queries y sin esas tres columnas. Los huecos en categorias del carrito se miden contra el prefijo del ticket; la regla de exclusion usa el carrito en el corte, que ademas incluye los `add_to_cart` de sesion.

Frente a la fila "antes" (mismo entrenamiento, sin las tres features ni reglas), el sistema servido (en negrita) cambia cat_hit_rate@5 en **+8.14 pp** y sku_hit_rate@5 en **+2.15 pp**, y los huecos regalados pasan del 33.0% al 0.0%.



### Huecos regalados del sistema servido, por perfil

| grupo | huecos_en_carrito@5 | huecos_repetidos@5 | huecos_regalados@5 | listas_con_regalo@5 |
| --- | --- | --- | --- | --- |
| total | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 - nuevo, carrito vacio | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 - nuevo, con articulos | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 - recurrente, carrito vacio | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 - recurrente, con articulos | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| con carrito (perfiles 2 y 4) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## De donde sale la validacion de la parada temprana (punto B4)

La parada temprana decide cuantos arboles se sirven. Hasta el punto B4 la validacion eran 2,500 cestas muestreadas por hash de **la misma ventana** que el entrenamiento, asi que el modelo paraba donde dejaba de mejorar sobre queries contemporaneas. El test, en cambio, esta siempre mas adelante en el tiempo. Ahora la validacion son los ultimos 14 dias de la ventana del ranker (del 2025-10-18 al 2025-10-31): mismo sentido de desplazamiento que hacia el test. Las dos variantes salen de la misma matriz de features y se evaluan sobre las mismas queries de test, con el mismo re-ranking.

Un matiz al leerlo: cambiar el criterio cambia **las dos** partes a la vez, porque lo que no cae en validacion entrena. La diferencia en test no es solo "donde para", tambien es "con que cestas aprende"; separarlas pediria fijar el numero de arboles a mano y no es lo que se quiere medir aqui.

| Variante | De donde sale la validacion | Cestas de train | Cestas de validacion | Arboles | NDCG@5 de su validacion |
| --- | --- | ---: | ---: | ---: | ---: |
| Temporal (servida) | ultimos 14 dias de la ventana (2025-10-18 a 2025-10-31) | 11,196 | 3,304 | 119 | 0.3216 |
| Por hash (anterior) | 2,500 cestas por hash, de toda la ventana | 12,000 | 2,500 | 140 | 0.3279 |

Las dos NDCG de validacion no son comparables entre si: cada una se mide sobre su propia muestra, y la temporal cae sobre cestas mas dificiles. Lo que se compara es el resultado en test.

### Sobre las 18,000 queries de test

| Variante | NDCG@5 graduada | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 SKU |
| --- | ---: | ---: | ---: | ---: |
| Parada temprana con validacion por hash (anterior) | 0.3019 | 0.7957 | 0.6137 | 0.2808 |
| **Parada temprana con validacion temporal (servido)** | **0.3018** | **0.7981** | **0.6122** | **0.2799** |

La validacion temporal para en 119 arboles frente a 140 (-21) y mueve la metrica principal -0.0001:

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LambdaRank con validacion por hash (antes del punto B4) | -0.0001 [-0.0011, +0.0009] | 0.834 | +0.24 pp [-0.02, +0.51] | 0.077 | -0.15 pp [-0.48, +0.18] | 0.389 | -0.0009 [-0.0023, +0.0006] | 0.212 |

**Lectura.** El cambio es metodologicamente el correcto --- parar sobre queries posteriores a las de entrenamiento se parece mas a lo que el modelo encuentra en test --- pero **no se mide ninguna diferencia**: el intervalo de la metrica principal cruza el cero con holgura. Tambien lo cruzan las otras tres metricas y los cuatro perfiles (el desglose esta en `metrics.json`, `comparisons`; la unica celda con p < 0,05 es una de veinte comparaciones sin corregir). Conviene ademas leerlo con la no-determinacion de la parada temprana delante: entre dos ejecuciones con los mismos datos el numero de arboles se mueve solo (deuda anotada en el README), asi que una diferencia de esta talla no seria atribuible al criterio de validacion aunque la hubiera. Se sirve la temporal por ser la mas defendible, no porque rinda mas.

## F1@5 frente a Kaggle "Instacart Market Basket Analysis"

| | F1 |
| --- | ---: |
| Este sistema, F1@5 (media armonica de Precision@5 y Recall@5 medios) | 0.2149 |
| Este sistema, F1@5 por cesta (media del F1 de cada cesta) | 0.1794 |
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
| NDCG@5 graduada | 0.2148 | 0.3018 | 1.40x |
| cat_hit_rate@5 | 0.7181 | 0.7981 | +8.00 pp |
| sku_hit_rate@5 | 0.5416 | 0.6122 | +7.06 pp |
| NDCG@5 SKU | 0.2006 | 0.2799 | 1.40x |
| Recall@5 | 0.1910 | 0.2338 | 1.22x |
| Precision@5 | 0.1451 | 0.1988 | 1.37x |
| F1@5 | 0.1649 | 0.2149 | 1.30x |
| Productos por adivinar (media) | 3.9327 | 5.5474 | 1.41x |

Son cestas distintas de datasets distintos, asi que no hay contraste pareado. Tampoco se puede leer el cambio como mejor o peor modelo: el target cambia de tamano y de estructura. Lo que dice cuanto del techo alcanza cada sistema en cada dataset esta en la seccion de diagnostico.

### Por perfil

| Grupo | n_queries 7a | Fase 8 | cat_hit_rate@5 7a | Fase 8 | sku_hit_rate@5 7a | Fase 8 | NDCG@5 graduada 7a | Fase 8 | Productos por adivinar 7a | Fase 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 18000 | 18000 | 0.7181 | 0.7981 | 0.5416 | 0.6122 | 0.2148 | 0.3018 | 3.93 | 5.55 |
| 1 - nuevo, carrito vacio | 521 | 554 | 0.5969 | 0.6968 | 0.4031 | 0.4747 | 0.1520 | 0.2312 | 4.17 | 5.02 |
| 2 - nuevo, con articulos | 421 | 388 | 0.4466 | 0.7191 | 0.2257 | 0.4175 | 0.1110 | 0.2176 | 2.34 | 3.36 |
| 3 - recurrente, carrito vacio | 8713 | 9460 | 0.7895 | 0.8080 | 0.6186 | 0.6408 | 0.2393 | 0.3145 | 5.03 | 6.73 |
| 4 - recurrente, con articulos | 8345 | 7598 | 0.6648 | 0.7972 | 0.4857 | 0.5965 | 0.1984 | 0.2954 | 2.85 | 4.22 |

## Frente a la Fase 3 original

La Fase 3 se entreno sobre el dataset anterior a la Fase 7a (1.500 productos, ~24 referencias por categoria y eleccion de SKU casi aleatoria). Sus cifras estan congeladas en `reports/recommender/baseline_fase3.json`. Mismo codigo, mismas ventanas y mismo numero de queries de test; cambia el dato.

| Metrica | Fase 3 (dataset viejo) | Fase 7c (dataset nuevo) | Cambio |
| --- | ---: | ---: | ---: |
| NDCG@5 | 0.0343 | 0.2799 | 8.16x |
| Recall@5 | 0.0332 | 0.2338 | 7.04x |
| Precision@5 | 0.0251 | 0.1988 | 7.93x |
| F1@5 | 0.0286 | 0.2149 | 7.52x |
| hit_rate@5 (SKU) | 0.1184 | 0.6122 | 5.17x |
| hit_rate@5 (categoria) | 0.5144 | 0.7981 | 1.55x |
| SKU / categoria (hit_rate) | 0.2302 | 0.7670 | 3.33x |
| SKU / categoria (precision) | 0.1371 | 0.6035 | 4.40x |

Un matiz al leerlo: el catalogo pasa de 1.500 a 496 productos, asi que un top-5 al azar tambien acierta mas que antes. El baseline de popularidad de la tabla de comparacion, sobre el mismo pool, es lo que aisla lo que aporta el ranker.

## Cold-start sobremuestreado (punto M4)

En la muestra de cabecera el cold-start son pocas queries (554 en el perfil 1 y 388 en el perfil 2). Aqui entran **todas** las cestas de la ventana de test sin compras anteriores a 2025-11-01: 3,140 cestas (1,495 anonimas y el resto de 496 clientes nuevos), cada una con los dos cortes de cabecera (carrito vacio y mitad del ticket): 5,802 queries. Las queries de una misma cesta estan correladas, asi que el bootstrap remuestrea cestas.

Estos clientes no tienen ninguna cesta antes del inicio del test, asi que no han entrado en ninguna fuente de candidatos (popularidad, afinidades, ALS) ni en el entrenamiento del ranker. Son clientes nunca vistos por ningun modelo. Lo unico que se sabe de ellos es su historial as-of dentro de la propia ventana, igual que en produccion. Mismos modelos y mismas fuentes que la cabecera.

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| perfiles 1 y 2 | 5802 | 3140 | 0.2228 [0.2158, 0.2301] | 0.7229 [0.7096, 0.7368] | 0.4529 [0.4374, 0.4692] | 0.1935 [0.1854, 0.2018] | 0.1718 [0.1640, 0.1797] | 4.5789 |
| 1 - nuevo, carrito vacio | 3140 | 3140 | 0.2409 [0.2332, 0.2494] | 0.7360 [0.7201, 0.7522] | 0.4943 [0.4768, 0.5134] | 0.2072 [0.1987, 0.2166] | 0.1696 [0.1616, 0.1786] | 5.5634 |
| 2 - nuevo, con articulos | 2662 | 2662 | 0.2014 [0.1932, 0.2095] | 0.7074 [0.6897, 0.7243] | 0.4042 [0.3835, 0.4223] | 0.1775 [0.1669, 0.1877] | 0.1745 [0.1637, 0.1851] | 3.4177 |

Entre corchetes, el intervalo de confianza al 95%: bootstrap percentil con 1,000 remuestreos de cestas (semilla 42; `evaluate.bootstrap_means`).

### Frente a los demas sistemas, por perfil

Diferencia = LambdaRank servido - el otro sistema (bootstrap pareado por cesta).

**1 - nuevo, carrito vacio.**

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (mismo pool) | +0.0147 [+0.0101, +0.0190] | < 0.001 | +1.72 pp [+0.51, +2.90] | 0.003 | +2.77 pp [+1.34, +4.11] | < 0.001 | +0.0197 [+0.0144, +0.0251] | < 0.001 |
| LambdaRank sin senal de sesion | +0.0015 [+0.0002, +0.0028] | 0.023 | -0.06 pp [-0.38, +0.25] | 0.770 | +0.45 pp [+0.00, +0.89] | 0.070 | +0.0020 [+0.0004, +0.0039] | 0.023 |
| LambdaRank sin features de carrito | +0.0007 [-0.0010, +0.0024] | 0.450 | +0.00 pp [-0.54, +0.57] | 1.000 | -0.32 pp [-0.92, +0.22] | 0.299 | +0.0006 [-0.0013, +0.0027] | 0.584 |
| LambdaRank con relevancia binaria de SKU | -0.0008 [-0.0031, +0.0017] | 0.551 | -0.10 pp [-0.99, +0.83] | 0.877 | -1.50 pp [-2.39, -0.51] | 0.005 | -0.0045 [-0.0073, -0.0013] | 0.002 |
| LambdaRank sin ranking personal de categorias | -0.0001 [-0.0021, +0.0019] | 0.935 | -0.38 pp [-1.02, +0.22] | 0.256 | +0.38 pp [-0.32, +1.08] | 0.326 | +0.0025 [+0.0000, +0.0050] | 0.050 |
| LambdaRank con validacion por hash (antes del punto B4) | -0.0039 [-0.0059, -0.0018] | < 0.001 | -0.70 pp [-1.53, +0.13] | 0.121 | -1.37 pp [-2.20, -0.57] | 0.002 | -0.0021 [-0.0045, +0.0006] | 0.096 |

**2 - nuevo, con articulos.**

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (mismo pool) | +0.0297 [+0.0245, +0.0353] | < 0.001 | +7.06 pp [+5.56, +8.64] | < 0.001 | +6.39 pp [+4.88, +7.85] | < 0.001 | +0.0286 [+0.0217, +0.0360] | < 0.001 |
| LambdaRank sin senal de sesion | +0.0062 [+0.0036, +0.0089] | < 0.001 | +0.45 pp [-0.23, +1.09] | 0.201 | +1.54 pp [+0.71, +2.37] | < 0.001 | +0.0077 [+0.0041, +0.0116] | < 0.001 |
| LambdaRank sin features de carrito | +0.0033 [+0.0005, +0.0062] | 0.032 | +1.01 pp [+0.15, +1.92] | 0.038 | +0.11 pp [-0.86, +1.05] | 0.877 | +0.0020 [-0.0015, +0.0059] | 0.288 |
| LambdaRank con relevancia binaria de SKU | +0.0053 [+0.0022, +0.0085] | 0.005 | +3.79 pp [+2.55, +5.03] | < 0.001 | -0.26 pp [-1.24, +0.71] | 0.648 | -0.0030 [-0.0067, +0.0008] | 0.126 |
| LambdaRank sin ranking personal de categorias | -0.0002 [-0.0025, +0.0025] | 0.862 | -0.11 pp [-0.83, +0.64] | 0.824 | +0.15 pp [-0.68, +1.01] | 0.751 | +0.0008 [-0.0025, +0.0044] | 0.653 |
| LambdaRank con validacion por hash (antes del punto B4) | -0.0006 [-0.0029, +0.0019] | 0.608 | +0.00 pp [-0.79, +0.75] | 1.000 | +0.23 pp [-0.64, +1.02] | 0.662 | -0.0003 [-0.0038, +0.0033] | 0.870 |


## Techo de la primera etapa

Que parte del target llego siquiera al pool de candidatos. Lo que no esta aqui, el ranker
no lo puede recuperar.

`pool_recall` es a nivel de SKU; `cat_pool_recall`, a nivel de **categoria** (que parte de
las categorias del target tiene al menos un representante en el pool). La segunda es el
techo de `cat_hit_rate@5`, y por tanto el de la metrica principal. No son lo mismo: un
pool puede llevar muchas referencias de pocas categorias y quedarse corto justo donde
importa. Era el caso en cold-start antes del punto M6 (`docs/diagnostico-fase7.md`).

| grupo | n_queries | pool_recall | cat_pool_recall | pool_size_medio |
| --- | --- | --- | --- | --- |
| total | 18000 | 0.9179 | 1.0000 | 233.7123 |
| 1 - nuevo, carrito vacio | 554 | 0.7742 | 1.0000 | 166.8213 |
| 2 - nuevo, con articulos | 388 | 0.8381 | 1.0000 | 220.5747 |
| 3 - recurrente, carrito vacio | 9460 | 0.9169 | 1.0000 | 215.4355 |
| 4 - recurrente, con articulos | 7598 | 0.9337 | 1.0000 | 262.0163 |

El pool alcanza el **100.0%** de las categorias del target, en los cuatro perfiles, y el sistema acierta alguna en el **79.8%** de las cestas. Esa distancia no la explica la primera etapa: son candidatos que si llegaron al pool y el ranker no subio al top-5.

## Que features usa el ranker

Importancia por ganancia, las 25 primeras.

| feature | gain | split |
| --- | --- | --- |
| category_idx | 110557.7454 | 1138 |
| hist_rank | 99820.3177 | 214 |
| cat_overdue_ratio | 24421.4638 | 622 |
| cat_due_rank | 15872.9493 | 189 |
| cat_days_since | 13643.3971 | 613 |
| cat_freq_share | 12362.1005 | 406 |
| cat_in_cart | 7698.2156 | 271 |
| typical_repurchase_days | 7024.7036 | 199 |
| hist_n_baskets | 5456.0553 | 96 |
| aff_conf_sum | 5447.9674 | 98 |
| cust_n_products | 4959.0164 | 271 |
| aff_lift_max | 4684.1322 | 113 |
| prefix_size | 4642.0989 | 300 |
| sess_secs_since_view | 4285.5903 | 113 |
| dept_share_in_cart | 4123.6489 | 254 |
| prod_pop_all | 4080.8302 | 55 |
| src_hist | 3914.1674 | 6 |
| cust_frequency | 3770.6465 | 199 |
| cust_recency_days | 3359.1570 | 254 |
| cataff_conf_max | 3218.2260 | 128 |
| cust_avg_ticket | 2815.9844 | 310 |
| dept_n_in_cart | 2736.5617 | 120 |
| cat_freq_rank | 2415.2775 | 114 |
| cat_expected_days | 2322.8779 | 257 |
| is_known_customer | 2006.3913 | 24 |

<!-- diagnostics:start -->
## Diagnostico: baselines independientes del pool y techo teorico

Generado por `python -m src.recommender.verify_recommender_diagnostics` (2026-09-18), sobre las 18,000 queries de test de las predicciones en disco (sha256 de `recommendations_test.parquet`: `5786394eb6f4`). El oraculo necesita antes `python -m data_generation.export_oracle`. Esta seccion no la reescribe el pipeline: si se reentrena, hay que volver a lanzar el verificador. Puntos A3, A4, A6 y M8 de `docs/diagnostico-fase7.md`. La metrica principal es la NDCG@5 graduada (3 por SKU exacto, 1 por categoria; `evaluate.category_metrics`).

### Lectura rapida

- **Mejor baseline en cat_hit_rate@5:** Reglas de asociacion de categoria + referencia lider, 0.7594 frente a 0.7981 del LambdaRank: -3.9 pp, el LambdaRank va por delante (LambdaRank - baseline: +3.87 pp [+3.36, +4.42], p < 0.001).
- **Techo teorico de cat_hit_rate@5:** 0.8382. El LambdaRank alcanza el 95.2% y el mejor baseline el 90.6%.
- **Techo de sku_hit_rate@5:** 0.6677. El LambdaRank alcanza el 91.7%.
- **Objetivo del ranker (A4):** con relevancia binaria de SKU, cat_hit_rate@5 0.7774 (+1.8 pp frente al mejor baseline); con la graduada servida, 0.7981 (+3.9 pp). sku_hit_rate@5 pasa de 0.6219 a 0.6122 y la NDCG@5 graduada de 0.2967 a 0.3018.

### Todos los sistemas, total

Las columnas "% techo" dividen por el oraculo correspondiente sobre las mismas queries. "Huecos regalados" es la parte del top-5 en una categoria que ya esta en el carrito o repetida mas arriba en la lista (`evaluate.wasted_slot_metrics`, punto A2). Los baselines de categoria no regalan ninguno por construccion; el LambdaRank los evita con el re-ranking de `RecommenderConfig.rerank`.

| Sistema | NDCG@5 graduada | cat_hit_rate@5 | % techo | cat_precision@5 | sku_hit_rate@5 | % techo SKU | sku_precision@5 | NDCG@5 SKU | Huecos regalados |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.3310 | 0.8382 | — | 0.3762 | 0.6425 | — | 0.2124 | 0.2941 | 0.0% |
| **Oraculo de SKU (techo)** | 0.3406 | 0.8292 | — | 0.3490 | 0.6677 | — | 0.2354 | 0.3283 | 0.0% |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.3018 | 0.7981 | 95.2% | 0.3295 | 0.6122 | 91.7% | 0.1988 | 0.2799 | 0.0% |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.2967 | 0.7774 | 92.7% | 0.3083 | 0.6219 | 93.1% | 0.2056 | 0.2868 | 0.0% |
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
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.2148 | 0.3018 | 0.7181 | 0.7981 | 91.9% | 95.2% | 0.5416 | 0.6122 | 91.5% | 91.7% |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.2083 | 0.2967 | 0.6775 | 0.7774 | 86.7% | 92.7% | 0.5510 | 0.6219 | 93.1% | 93.1% |
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
| LambdaRank con relevancia binaria de SKU (ablacion A4) | +0.0051 [+0.0037, +0.0065] | < 0.001 | +2.07 pp [+1.68, +2.43] | < 0.001 | -0.97 pp [-1.38, -0.53] | < 0.001 |
| Reglas de asociacion de categoria + referencia lider | +0.0900 [+0.0873, +0.0933] | < 0.001 | +3.87 pp [+3.36, +4.42] | < 0.001 | +21.09 pp [+20.38, +21.91] | < 0.001 |
| Frecuencia personal x due_for_repurchase + referencia favorita | +0.0539 [+0.0516, +0.0564] | < 0.001 | +4.88 pp [+4.36, +5.42] | < 0.001 | +8.46 pp [+7.88, +9.05] | < 0.001 |
| Frecuencia personal por categoria + referencia favorita | +0.0604 [+0.0581, +0.0629] | < 0.001 | +5.76 pp [+5.21, +6.32] | < 0.001 | +10.02 pp [+9.43, +10.63] | < 0.001 |
| Popularidad de categoria + referencia lider | +0.0979 [+0.0951, +0.1011] | < 0.001 | +5.98 pp [+5.42, +6.58] | < 0.001 | +21.73 pp [+21.01, +22.54] | < 0.001 |
| Popularidad global (SKU) | +0.0914 [+0.0886, +0.0944] | < 0.001 | +8.08 pp [+7.50, +8.69] | < 0.001 | +19.86 pp [+19.13, +20.64] | < 0.001 |
| Repetir las referencias favoritas del cliente | +0.0562 [+0.0539, +0.0587] | < 0.001 | +9.92 pp [+9.36, +10.45] | < 0.001 | +6.81 pp [+6.25, +7.39] | < 0.001 |
| Aleatorio | +0.2623 [+0.2590, +0.2659] | < 0.001 | +48.44 pp [+47.62, +49.29] | < 0.001 | +56.07 pp [+55.33, +56.83] | < 0.001 |

### cat_hit_rate@5 por perfil (entre parentesis, % del techo del perfil)

| Sistema | Total | Perfil 1 - nuevo, carrito vacio | Perfil 2 - nuevo, con articulos | Perfil 3 - recurrente, carrito vacio | Perfil 4 - recurrente, con articulos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.8382 | 0.7437 | 0.7861 | 0.8410 | 0.8443 |
| **Oraculo de SKU (techo)** | 0.8292 | 0.7148 | 0.7680 | 0.8314 | 0.8379 |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.7981 (95.2%) | 0.6968 (93.7%) | 0.7191 (91.5%) | 0.8080 (96.1%) | 0.7972 (94.4%) |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.7774 (92.7%) | 0.7022 (94.4%) | 0.6649 (84.6%) | 0.7872 (93.6%) | 0.7765 (92.0%) |
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
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.6122 (91.7%) | 0.4747 (92.3%) | 0.4175 (81.4%) | 0.6408 (93.3%) | 0.5965 (89.9%) |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.6219 (93.1%) | 0.4910 (95.4%) | 0.4201 (81.9%) | 0.6466 (94.2%) | 0.6110 (92.1%) |
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
| <= 7 dias | 9.6% | 20.0% | 37.0% | 20.7% | 11.5% | 23.7% |
| <= 14 dias | 22.8% | 20.9% | 37.5% | 33.2% | 14.3% | 28.9% |
| 8-14 dias | 13.3% | 21.5% | 37.8% | 12.5% | 19.0% | 37.4% |
| comprada en la ventana, antes de la cesta | 41.1% | 21.1% | 36.1% | 46.4% | 15.8% | 31.1% |
| sin compra en la ventana | 58.9% | 19.0% | 30.7% | 53.6% | 15.4% | 29.7% |
| total | 100.0% | 19.9% | 32.9% | 100.0% | 15.6% | 30.3% |

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
- OK — LambdaRank recalculado desde predictions/ = reports/recommender/metrics.json (0.7981 frente a 0.7981)
- OK — Techo realizado compatible con el esperado en las 4 metricas, total y por perfil (max |z| <= 3.5) (peor: sku_precision_p2, z = +2.26)
- OK — Ningun sistema supera al oraculo de categoria en cat_hit_rate (mejor sistema lambdarank = 0.7981)
<!-- diagnostics:end -->

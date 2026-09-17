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
| Ranker | LightGBM `lambdarank`, 70 arboles |
| Objetivo | NDCG@5 con relevancia graduada: 2 SKU exacto, 1 misma categoria (`label_gain` = [0.0, 1.0, 3.0]) |
| NDCG@5 graduada de validacion | 0.2503 |

El split es temporal **y por cesta**: ninguna cesta se reparte entre train y test, y las
fuentes de candidatos se reajustan para cada ventana con solo el pasado de esa ventana.

## Metrica principal

La metrica principal del recomendador es la **NDCG@5 con relevancia graduada**
(`CHALLENGE.md`, punto A4 de `docs/diagnostico-fase7.md`): 3 puntos por hueco si es el SKU
exacto, 1 si solo acierta la categoria. Es la que optimiza el LambdaRank. Se lee siempre
junto a `cat_hit_rate@5` (lo que ensena la demo) y `sku_hit_rate@5`.

| grupo | n_queries | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 |
| --- | --- | --- | --- | --- |
| total | 18000 | 0.2148 [0.2120, 0.2177] | 0.7181 [0.7114, 0.7248] | 0.5416 [0.5348, 0.5491] |
| 1 - nuevo, carrito vacio | 521 | 0.1520 [0.1366, 0.1666] | 0.5969 [0.5528, 0.6392] | 0.4031 [0.3589, 0.4434] |
| 2 - nuevo, con articulos | 421 | 0.1110 [0.0943, 0.1278] | 0.4466 [0.3967, 0.4964] | 0.2257 [0.1853, 0.2637] |
| 3 - recurrente, carrito vacio | 8713 | 0.2393 [0.2350, 0.2436] | 0.7895 [0.7809, 0.7984] | 0.6186 [0.6085, 0.6284] |
| 4 - recurrente, con articulos | 8345 | 0.1984 [0.1940, 0.2029] | 0.6648 [0.6556, 0.6749] | 0.4857 [0.4751, 0.4963] |

Entre corchetes, el intervalo de confianza al 95%: bootstrap percentil con 1,000 remuestreos de cestas (semilla 42; `evaluate.bootstrap_means`). En el grupo mas pequeno (2 - nuevo, con articulos, n = 421) el intervalo de cat_hit_rate@5 mide 10.0 puntos: las diferencias entre perfiles pequenos se leen con eso delante. Los perfiles 1 y 2, con muchas mas queries, estan en "Cold-start sobremuestreado".

El acierto segun cuantas lineas lleva ya el carrito (varios cortes por cesta, punto M3) esta en [`cuts.md`](cuts.md).

## Objetivo del ranker (punto A4)

Hasta el punto A4 el LambdaRank optimizaba la NDCG@5 con relevancia binaria de SKU exacto. Ahora optimiza la graduada (`RankerConfig.relevance`) y tiene tres features nuevas de ranking personal de categorias (`CUSTOMER_CATEGORY_RANK_FEATURES`: `cat_freq_rank`, `cat_freq_share`, `cat_due_rank`). Las variantes se entrenan con las mismas queries y se evaluan sobre el mismo pool, con el mismo re-ranking servido.

| Variante | NDCG@5 graduada | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 SKU | cat_precision@5 | sku_precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Modelo en disco antes de A4 (congelado) | — | 0.6743 | 0.5488 | 0.2028 | 0.2021 | 0.1482 |
| Relevancia binaria de SKU (objetivo anterior) | 0.2083 | 0.6775 | 0.5510 | 0.2048 | 0.2036 | 0.1491 |
| Relevancia graduada, sin ranking personal | 0.2149 | 0.7192 | 0.5403 | 0.2012 | 0.2270 | 0.1454 |
| **Relevancia graduada + ranking personal (servido)** | **0.2148** | **0.7181** | **0.5416** | **0.2006** | **0.2267** | **0.1451** |

Cambiar solo el objetivo (fila de relevancia binaria a graduada sin ranking) mueve cat_hit_rate@5 **+4.17 pp** y sku_hit_rate@5 **-1.07 pp**. Anadir el ranking personal mueve cat_hit_rate@5 **-0.11 pp** y sku_hit_rate@5 **+0.13 pp**. En conjunto, frente al objetivo anterior: cat_hit_rate@5 **+4.06 pp**, sku_hit_rate@5 **-0.94 pp**.

La fila congelada es el modelo que habia en disco antes de este cambio (`reports/recommender/baseline_pre_a4.json`, 2026-09-17, commit `e0411d4`), que no media la NDCG graduada. La fila "relevancia binaria" lo reentrena con el codigo actual, que ya incluye el ranking personal entre sus features, asi que puede diferir un poco.

### Por perfil: relevancia binaria de SKU frente a la graduada servida

| Grupo | cat_hit_rate@5 SKU | graduada | cambio | sku_hit_rate@5 SKU | graduada | cambio | NDCG@5 graduada SKU | graduada |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 0.6775 | 0.7181 | +4.06 pp | 0.5510 | 0.5416 | -0.94 pp | 0.2083 | 0.2148 |
| 1 - nuevo, carrito vacio | 0.5777 | 0.5969 | +1.92 pp | 0.3916 | 0.4031 | +1.15 pp | 0.1485 | 0.1520 |
| 2 - nuevo, con articulos | 0.4418 | 0.4466 | +0.48 pp | 0.2257 | 0.2257 | +0.00 pp | 0.1065 | 0.1110 |
| 3 - recurrente, carrito vacio | 0.7490 | 0.7895 | +4.05 pp | 0.6309 | 0.6186 | -1.23 pp | 0.2328 | 0.2393 |
| 4 - recurrente, con articulos | 0.6210 | 0.6648 | +4.39 pp | 0.4939 | 0.4857 | -0.83 pp | 0.1915 | 0.1984 |

### Importancia del ranking personal de categorias

Puesto entre todas las features del modelo servido (ganancia).

| Feature | Puesto por ganancia | Ganancia | % de la ganancia total | Splits |
| --- | ---: | ---: | ---: | ---: |
| `cat_freq_share` | 6 de 60 | 9261 | 4.2% | 266 |
| `cat_due_rank` | 9 de 60 | 6607 | 3.0% | 83 |
| `cat_freq_rank` | 10 de 60 | 5230 | 2.4% | 88 |

La comparacion con los baselines de categoria y con el techo teorico esta en la seccion de diagnostico, al final (`verify_recommender_diagnostics`).

## Resultado a nivel de SKU

Las metricas de SKU exacto de siempre (NDCG@5 binaria, recall, precision, F1), con su
intervalo de confianza.

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.2006 [0.1975, 0.2039] | 0.1910 [0.1878, 0.1944] | 0.1451 [0.1429, 0.1473] | 0.1649 [0.1624, 0.1674] | 0.1552 [0.1529, 0.1576] | 0.5416 [0.5348, 0.5491] | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.1317 [0.1153, 0.1479] | 0.1175 [0.1026, 0.1331] | 0.0990 [0.0871, 0.1106] | 0.1075 [0.0944, 0.1200] | 0.1008 [0.0885, 0.1119] | 0.4031 [0.3589, 0.4434] | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.0994 [0.0796, 0.1199] | 0.1062 [0.0859, 0.1278] | 0.0532 [0.0432, 0.0632] | 0.0709 [0.0572, 0.0846] | 0.0689 [0.0556, 0.0825] | 0.2257 [0.1853, 0.2637] | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.2136 [0.2091, 0.2181] | 0.1801 [0.1764, 0.1840] | 0.1756 [0.1719, 0.1790] | 0.1778 [0.1742, 0.1814] | 0.1687 [0.1654, 0.1720] | 0.6186 [0.6085, 0.6284] | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.1965 [0.1912, 0.2018] | 0.2112 [0.2057, 0.2168] | 0.1209 [0.1178, 0.1239] | 0.1538 [0.1499, 0.1577] | 0.1489 [0.1452, 0.1526] | 0.4857 [0.4751, 0.4963] | 2.8506 |

## Comparacion

Mismo pool de candidatos, distinta forma de ordenarlo. Es lo que aisla la aportacion del
ranker de la de la primera etapa.

| Sistema | NDCG@5 | Recall@5 | Precision@5 | F1@5 | hit_rate@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (sin aprendizaje) | 0.0881 | 0.0793 | 0.0618 | 0.0695 | 0.2701 |
| LambdaRank sin senal de sesion | 0.1938 | 0.1854 | 0.1422 | 0.1610 | 0.5343 |
| LambdaRank con relevancia binaria de SKU (objetivo anterior) | 0.2048 | 0.1952 | 0.1491 | 0.1691 | 0.5510 |
| **LambdaRank completo** | **0.2006** | **0.1910** | **0.1451** | **0.1649** | **0.5416** |

### Diferencias con intervalo de confianza (bootstrap pareado)

Diferencia = LambdaRank servido - el otro sistema, sobre las mismas queries y con el mismo re-ranking. Cada remuestreo sortea las mismas cestas para los dos sistemas (`evaluate.paired_bootstrap`, 1,000 remuestreos). El p-valor es bilateral y no esta corregido por comparaciones multiples. Las tasas van en puntos porcentuales. El desglose por perfil esta en `metrics.json` (`comparisons`).

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (mismo pool) | +0.0944 [+0.0916, +0.0972] | < 0.001 | +13.56 pp [+12.81, +14.37] | < 0.001 | +27.15 pp [+26.28, +28.04] | < 0.001 | +0.1125 [+0.1091, +0.1158] | < 0.001 |
| LambdaRank sin senal de sesion | +0.0056 [+0.0045, +0.0068] | < 0.001 | +0.36 pp [+0.01, +0.69] | 0.046 | +0.73 pp [+0.36, +1.08] | 0.002 | +0.0069 [+0.0055, +0.0084] | < 0.001 |
| LambdaRank sin features de carrito | +0.0009 [-0.0002, +0.0018] | 0.089 | +0.47 pp [+0.11, +0.80] | 0.017 | +0.03 pp [-0.38, +0.39] | 0.876 | -0.0005 [-0.0017, +0.0007] | 0.402 |
| LambdaRank con relevancia binaria de SKU | +0.0065 [+0.0053, +0.0079] | < 0.001 | +4.06 pp [+3.60, +4.53] | < 0.001 | -0.94 pp [-1.41, -0.49] | < 0.001 | -0.0041 [-0.0057, -0.0025] | < 0.001 |
| LambdaRank sin ranking personal de categorias | -0.0001 [-0.0014, +0.0010] | 0.817 | -0.11 pp [-0.48, +0.26] | 0.571 | +0.13 pp [-0.28, +0.49] | 0.545 | -0.0005 [-0.0020, +0.0007] | 0.435 |

### Por perfil, sin senal de sesion

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.1938 | 0.1854 | 0.1422 | 0.1610 | 0.1515 | 0.5343 | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.1339 | 0.1195 | 0.1010 | 0.1094 | 0.1026 | 0.4088 | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.0868 | 0.0915 | 0.0466 | 0.0617 | 0.0601 | 0.2090 | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.2147 | 0.1813 | 0.1765 | 0.1789 | 0.1697 | 0.6225 | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.1810 | 0.1986 | 0.1138 | 0.1447 | 0.1401 | 0.4664 | 2.8506 |

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

| grupo | n_queries | ndcg_graded@5 | cat_hit_rate@5 | cat_precision@5 | sku_hit_rate@5 | sku_precision@5 |
| --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.2148 [0.2120, 0.2177] | 0.7181 [0.7114, 0.7248] | 0.2267 [0.2239, 0.2294] | 0.5416 [0.5348, 0.5491] | 0.1451 [0.1429, 0.1473] |
| 1 - nuevo, carrito vacio | 521 | 0.1520 [0.1366, 0.1666] | 0.5969 [0.5528, 0.6392] | 0.1781 [0.1620, 0.1939] | 0.4031 [0.3589, 0.4434] | 0.0990 [0.0871, 0.1106] |
| 2 - nuevo, con articulos | 421 | 0.1110 [0.0943, 0.1278] | 0.4466 [0.3967, 0.4964] | 0.1145 [0.1007, 0.1292] | 0.2257 [0.1853, 0.2637] | 0.0532 [0.0432, 0.0632] |
| 3 - recurrente, carrito vacio | 8713 | 0.2393 [0.2350, 0.2436] | 0.7895 [0.7809, 0.7984] | 0.2730 [0.2687, 0.2774] | 0.6186 [0.6085, 0.6284] | 0.1756 [0.1719, 0.1790] |
| 4 - recurrente, con articulos | 8345 | 0.1984 [0.1940, 0.2029] | 0.6648 [0.6556, 0.6749] | 0.1871 [0.1840, 0.1907] | 0.4857 [0.4751, 0.4963] | 0.1209 [0.1178, 0.1239] |

El sistema acierta la categoria en el **71.8%** de las cestas y el SKU exacto en el **54.2%**; el cociente entre las dos es **75.4%**. La distancia entre las dos columnas mide cuanto del error esta en *elegir la referencia* y no en *saber que categoria toca*. Desde la Fase 7a el surtido es de 8 referencias por categoria y el cliente repite su referencia preferida con la lealtad de la categoria (`DATA_SPEC.md`, "Fidelidad de marca").

## Historial al dia de la cesta (punto A1)

Las features de cliente x producto, cliente x categoria y cliente, y la fuente `hist` con su `due_for_repurchase`, se calculan con todas las cestas del cliente anteriores al dia de cada query (`src/recommender/history.py`), tambien las de dentro de la ventana. ALS, popularidad, afinidades y el perfil siguen congelados al inicio de cada ventana.

"Antes" es el modelo con el historial congelado al inicio de la ventana (`reports/recommender/baseline_pre_a1.json`, 2026-09-17, commit `bd1adc6`), sobre las mismas queries de test.

| Grupo | cat_hit_rate@5 antes | despues | cambio | sku_hit_rate@5 antes | despues | cambio | NDCG@5 antes | despues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total | 0.6168 | 0.7181 | +10.13 pp | 0.4947 | 0.5416 | +4.69 pp | 0.1759 | 0.2006 |
| 1 - nuevo, carrito vacio | 0.5624 | 0.5969 | +3.45 pp | 0.3724 | 0.4031 | +3.07 pp | 0.1204 | 0.1317 |
| 2 - nuevo, con articulos | 0.4371 | 0.4466 | +0.95 pp | 0.2114 | 0.2257 | +1.43 pp | 0.0851 | 0.0994 |
| 3 - recurrente, carrito vacio | 0.6848 | 0.7895 | +10.47 pp | 0.5741 | 0.6186 | +4.45 pp | 0.1881 | 0.2136 |
| 4 - recurrente, con articulos | 0.5583 | 0.6648 | +10.65 pp | 0.4337 | 0.4857 | +5.20 pp | 0.1713 | 0.1965 |

El detalle de los huecos que caian en categorias recien repuestas esta en la seccion de diagnostico (`verify_recommender_diagnostics`).

## Carrito y diversidad (punto A2)

Con una linea por categoria en cada cesta, un hueco del top-5 se *regala* si su categoria ya esta en el carrito (no puede acertar) o ya salio mas arriba en la lista (de las dos, como mucho acierta una). Dos piezas atacan el problema: tres features de carrito (`cat_in_cart`, `dept_n_in_cart`, `dept_share_in_cart`) y un re-ranking final (`src/recommender/rerank.py`). Reglas servidas: **como maximo 1 referencia(s) por categoria, categorias del carrito relegadas** (`RecommenderConfig.rerank`), las mismas en esta evaluacion, en las tablas de arriba y en la demo.

| Variante | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 | Huecos regalados | Huecos en cat. del carrito (perfiles 2 y 4) | Listas con carrito afectadas | Listas con cat. repetida | Categorias distintas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sin features de carrito, sin re-ranking (antes) | 0.6601 | 0.5233 | 0.1914 | 18.2% | 7.0% | 66.4% | 53.9% | 4.21 |
| Sin features de carrito + re-ranking servido | 0.7134 | 0.5412 | 0.2011 | 0.0% | 0.0% | 0.0% | 0.0% | 5.00 |
| Con features de carrito, sin re-ranking | 0.6570 | 0.5237 | 0.1918 | 20.8% | 0.0% | 66.7% | 63.0% | 3.96 |
| Con features de carrito + 1 por categoria | 0.7182 | 0.5416 | 0.2006 | 0.0% | 0.0% | 0.1% | 0.0% | 5.00 |
| **Con features de carrito + 1 por categoria + exclusion del carrito** | **0.7181** | **0.5416** | **0.2006** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **5.00** |

Las filas "sin features de carrito" usan un LambdaRank entrenado aparte con las mismas queries y sin esas tres columnas. Los huecos en categorias del carrito se miden contra el prefijo del ticket; la regla de exclusion usa el carrito en el corte, que ademas incluye los `add_to_cart` de sesion.

Frente a la fila "antes" (mismo entrenamiento, sin las tres features ni reglas), el sistema servido (en negrita) cambia cat_hit_rate@5 en **+5.81 pp** y sku_hit_rate@5 en **+1.83 pp**, y los huecos regalados pasan del 18.2% al 0.0%.

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
| Este sistema, F1@5 (media armonica de Precision@5 y Recall@5 medios) | 0.1649 |
| Este sistema, F1@5 por cesta (media del F1 de cada cesta) | 0.1552 |
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
| NDCG@5 | 0.0343 | 0.2006 | 5.85x |
| Recall@5 | 0.0332 | 0.1910 | 5.75x |
| Precision@5 | 0.0251 | 0.1451 | 5.79x |
| F1@5 | 0.0286 | 0.1649 | 5.77x |
| hit_rate@5 (SKU) | 0.1184 | 0.5416 | 4.57x |
| hit_rate@5 (categoria) | 0.5144 | 0.7181 | 1.40x |
| SKU / categoria (hit_rate) | 0.2302 | 0.7541 | 3.28x |
| SKU / categoria (precision) | 0.1371 | 0.6401 | 4.67x |

Un matiz al leerlo: el catalogo pasa de 1.500 a 496 productos, asi que un top-5 al azar tambien acierta mas que antes. El baseline de popularidad de la tabla de comparacion, sobre el mismo pool, es lo que aisla lo que aporta el ranker.

## Cold-start sobremuestreado (punto M4)

En la muestra de cabecera el cold-start son pocas queries (521 en el perfil 1 y 421 en el perfil 2). Aqui entran **todas** las cestas de la ventana de test sin compras anteriores a 2025-11-01: 3,140 cestas (1,495 anonimas y el resto de 496 clientes nuevos), cada una con los dos cortes de cabecera (carrito vacio y mitad del ticket): 6,092 queries. Las queries de una misma cesta estan correladas, asi que el bootstrap remuestrea cestas.

Estos clientes no tienen ninguna cesta antes del inicio del test, asi que no han entrado en ninguna fuente de candidatos (popularidad, afinidades, ALS) ni en el entrenamiento del ranker. Son clientes nunca vistos por ningun modelo. Lo unico que se sabe de ellos es su historial as-of dentro de la propia ventana, igual que en produccion. Mismos modelos y mismas fuentes que la cabecera.

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| perfiles 1 y 2 | 6092 | 3140 | 0.1333 [0.1276, 0.1391] | 0.5289 [0.5133, 0.5440] | 0.3285 [0.3127, 0.3428] | 0.1180 [0.1116, 0.1244] | 0.1147 [0.1082, 0.1210] | 3.3733 |
| 1 - nuevo, carrito vacio | 3140 | 3140 | 0.1494 [0.1434, 0.1555] | 0.5952 [0.5790, 0.6118] | 0.3876 [0.3704, 0.4038] | 0.1274 [0.1206, 0.1338] | 0.1112 [0.1050, 0.1172] | 4.2309 |
| 2 - nuevo, con articulos | 2952 | 2952 | 0.1161 [0.1103, 0.1224] | 0.4583 [0.4411, 0.4756] | 0.2656 [0.2503, 0.2818] | 0.1079 [0.1012, 0.1157] | 0.1184 [0.1109, 0.1267] | 2.4610 |

Entre corchetes, el intervalo de confianza al 95%: bootstrap percentil con 1,000 remuestreos de cestas (semilla 42; `evaluate.bootstrap_means`).

### Frente a los demas sistemas, por perfil

Diferencia = LambdaRank servido - el otro sistema (bootstrap pareado por cesta).

**1 - nuevo, carrito vacio.**

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (mismo pool) | +0.0249 [+0.0203, +0.0296] | < 0.001 | +3.18 pp [+1.53, +4.84] | < 0.001 | +8.25 pp [+6.78, +9.87] | < 0.001 | +0.0296 [+0.0248, +0.0346] | < 0.001 |
| LambdaRank sin senal de sesion | +0.0006 [-0.0008, +0.0020] | 0.416 | +0.03 pp [-0.73, +0.76] | 0.963 | +0.16 pp [-0.45, +0.80] | 0.657 | +0.0009 [-0.0008, +0.0026] | 0.292 |
| LambdaRank sin features de carrito | -0.0005 [-0.0022, +0.0011] | 0.588 | -0.35 pp [-1.05, +0.38] | 0.388 | -0.19 pp [-0.92, +0.45] | 0.616 | -0.0000 [-0.0019, +0.0017] | 0.995 |
| LambdaRank con relevancia binaria de SKU | +0.0033 [+0.0011, +0.0057] | 0.005 | +0.19 pp [-0.80, +1.24] | 0.730 | -0.45 pp [-1.31, +0.45] | 0.351 | +0.0007 [-0.0017, +0.0033] | 0.590 |
| LambdaRank sin ranking personal de categorias | +0.0002 [-0.0019, +0.0022] | 0.875 | -0.10 pp [-0.89, +0.67] | 0.857 | +0.06 pp [-0.67, +0.80] | 0.898 | +0.0007 [-0.0015, +0.0030] | 0.549 |

**2 - nuevo, con articulos.**

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p | ndcg@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (mismo pool) | +0.0275 [+0.0227, +0.0327] | < 0.001 | +4.23 pp [+2.51, +6.06] | < 0.001 | +7.32 pp [+5.86, +8.88] | < 0.001 | +0.0326 [+0.0267, +0.0386] | < 0.001 |
| LambdaRank sin senal de sesion | +0.0052 [+0.0029, +0.0077] | < 0.001 | +0.17 pp [-0.68, +1.15] | 0.724 | +1.15 pp [+0.44, +1.96] | 0.005 | +0.0076 [+0.0047, +0.0108] | < 0.001 |
| LambdaRank sin features de carrito | -0.0003 [-0.0022, +0.0018] | 0.785 | -0.10 pp [-0.98, +0.75] | 0.866 | -0.03 pp [-0.75, +0.75] | 0.963 | -0.0006 [-0.0030, +0.0021] | 0.672 |
| LambdaRank con relevancia binaria de SKU | +0.0036 [+0.0010, +0.0063] | 0.014 | -0.14 pp [-1.22, +1.02] | 0.847 | -0.64 pp [-1.49, +0.30] | 0.174 | +0.0013 [-0.0019, +0.0050] | 0.456 |
| LambdaRank sin ranking personal de categorias | +0.0002 [-0.0019, +0.0024] | 0.814 | -0.95 pp [-1.90, +0.00] | 0.058 | +0.17 pp [-0.64, +0.98] | 0.696 | +0.0011 [-0.0017, +0.0040] | 0.449 |


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
| hist_rank | 59144.3962 | 128 |
| category_idx | 38312.8745 | 711 |
| cat_overdue_ratio | 22254.7706 | 380 |
| cat_days_since | 16600.0462 | 459 |
| hist_n_baskets | 9521.9184 | 124 |
| cat_freq_share | 9261.3332 | 266 |
| sess_secs_since_view | 7856.3732 | 156 |
| cat_in_cart | 6767.7755 | 125 |
| cat_due_rank | 6607.3911 | 83 |
| cat_freq_rank | 5229.8296 | 88 |
| is_known_customer | 4602.4211 | 45 |
| aff_lift_max | 4264.1563 | 157 |
| typical_repurchase_days | 3920.5979 | 124 |
| cust_n_products | 2642.8706 | 129 |
| cust_frequency | 2583.9762 | 96 |
| hist_days_since | 2277.7461 | 85 |
| cust_recency_days | 1718.1457 | 118 |
| prod_pop_all | 1525.5423 | 51 |
| cat_expected_days | 1245.3104 | 111 |
| cataff_conf_max | 1157.6332 | 64 |
| cust_avg_ticket | 1132.5926 | 104 |
| aff_conf_sum | 1100.3845 | 56 |
| cat_n_purchase_days | 1057.8356 | 65 |
| hist_units | 973.2104 | 37 |
| sess_viewed | 853.1224 | 24 |

<!-- diagnostics:start -->
## Diagnostico: baselines independientes del pool y techo teorico

Generado por `python -m src.recommender.verify_recommender_diagnostics` (2026-09-17), sobre las 18,000 queries de test de las predicciones en disco (sha256 de `recommendations_test.parquet`: `ac3f78bdf4fa`). El oraculo necesita antes `python -m data_generation.export_oracle`. Esta seccion no la reescribe el pipeline: si se reentrena, hay que volver a lanzar el verificador. Puntos A3, A4, A6 y M8 de `docs/diagnostico-fase7.md`. La metrica principal es la NDCG@5 graduada (3 por SKU exacto, 1 por categoria; `evaluate.category_metrics`).

### Lectura rapida

- **Mejor baseline en cat_hit_rate@5:** Frecuencia personal x due_for_repurchase + referencia favorita, 0.6789 frente a 0.7181 del LambdaRank: -3.9 pp, el LambdaRank va por delante (LambdaRank - baseline: +3.92 pp [+3.25, +4.59], p < 0.001).
- **Techo teorico de cat_hit_rate@5:** 0.7816. El LambdaRank alcanza el 91.9% y el mejor baseline el 86.9%.
- **Techo de sku_hit_rate@5:** 0.5920. El LambdaRank alcanza el 91.5%.
- **Objetivo del ranker (A4):** con relevancia binaria de SKU, cat_hit_rate@5 0.6775 (-0.1 pp frente al mejor baseline); con la graduada servida, 0.7181 (+3.9 pp). sku_hit_rate@5 pasa de 0.5510 a 0.5416 y la NDCG@5 graduada de 0.2083 a 0.2148.

### Todos los sistemas, total

Las columnas "% techo" dividen por el oraculo correspondiente sobre las mismas queries. "Huecos regalados" es la parte del top-5 en una categoria que ya esta en el carrito o repetida mas arriba en la lista (`evaluate.wasted_slot_metrics`, punto A2). Los baselines de categoria no regalan ninguno por construccion; el LambdaRank los evita con el re-ranking de `RecommenderConfig.rerank`.

| Sistema | NDCG@5 graduada | cat_hit_rate@5 | % techo | cat_precision@5 | sku_hit_rate@5 | % techo SKU | sku_precision@5 | NDCG@5 SKU | Huecos regalados |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.2308 | 0.7816 | — | 0.2660 | 0.5542 | — | 0.1495 | 0.2004 | 0.0% |
| **Oraculo de SKU (techo)** | 0.2340 | 0.7309 | — | 0.2339 | 0.5920 | — | 0.1661 | 0.2269 | 0.0% |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.2148 | 0.7181 | 91.9% | 0.2267 | 0.5416 | 91.5% | 0.1451 | 0.2006 | 0.0% |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.2083 | 0.6775 | 86.7% | 0.2036 | 0.5510 | 93.1% | 0.1491 | 0.2048 | 0.0% |
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

### LambdaRank frente a cada sistema (bootstrap pareado, punto M4)

Diferencia = LambdaRank servido - el otro sistema, sobre las mismas queries. Cada remuestreo sortea las mismas cestas para los dos (1,000 remuestreos, intervalo percentil al 95%; `evaluate.paired_bootstrap`). El p-valor es bilateral y no esta corregido por comparaciones multiples. Las tasas van en puntos porcentuales. El desglose por perfil esta en `diagnostics.json` (`comparisons`).

| Frente a | ndcg_graded@5: diferencia [IC 95 %] | p | cat_hit_rate@5: diferencia [IC 95 %] | p | sku_hit_rate@5: diferencia [IC 95 %] | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | +0.0065 [+0.0053, +0.0079] | < 0.001 | +4.06 pp [+3.60, +4.53] | < 0.001 | -0.94 pp [-1.41, -0.49] | < 0.001 |
| LambdaRank antes de A4 (relevancia binaria, congelado) | +0.0089 [+0.0074, +0.0103] | < 0.001 | +4.38 pp [+3.86, +4.94] | < 0.001 | -0.73 pp [-1.22, -0.29] | 0.003 |
| LambdaRank antes de A1 (historial congelado) | +0.0357 [+0.0339, +0.0377] | < 0.001 | +10.13 pp [+9.50, +10.78] | < 0.001 | +4.69 pp [+4.11, +5.31] | < 0.001 |
| Frecuencia personal x due_for_repurchase + referencia favorita | +0.0468 [+0.0446, +0.0491] | < 0.001 | +3.92 pp [+3.25, +4.59] | < 0.001 | +12.02 pp [+11.32, +12.67] | < 0.001 |
| Frecuencia personal por categoria + referencia favorita | +0.0584 [+0.0562, +0.0608] | < 0.001 | +6.92 pp [+6.29, +7.62] | < 0.001 | +14.22 pp [+13.52, +14.93] | < 0.001 |
| Popularidad de categoria + referencia lider | +0.0912 [+0.0884, +0.0938] | < 0.001 | +10.72 pp [+9.91, +11.52] | < 0.001 | +27.65 pp [+26.83, +28.54] | < 0.001 |
| Reglas de asociacion de categoria + referencia lider | +0.0926 [+0.0898, +0.0954] | < 0.001 | +10.85 pp [+10.07, +11.62] | < 0.001 | +28.30 pp [+27.49, +29.13] | < 0.001 |
| Repetir las referencias favoritas del cliente | +0.0532 [+0.0512, +0.0555] | < 0.001 | +12.12 pp [+11.47, +12.74] | < 0.001 | +9.34 pp [+8.73, +9.98] | < 0.001 |
| Popularidad global (SKU) | +0.0950 [+0.0922, +0.0976] | < 0.001 | +13.82 pp [+13.03, +14.65] | < 0.001 | +27.53 pp [+26.69, +28.42] | < 0.001 |
| Aleatorio | +0.1847 [+0.1817, +0.1877] | < 0.001 | +44.27 pp [+43.41, +45.14] | < 0.001 | +50.07 pp [+49.28, +50.84] | < 0.001 |

### cat_hit_rate@5 por perfil (entre parentesis, % del techo del perfil)

| Sistema | Total | Perfil 1 - nuevo, carrito vacio | Perfil 2 - nuevo, con articulos | Perfil 3 - recurrente, carrito vacio | Perfil 4 - recurrente, con articulos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.7816 | 0.6545 | 0.5297 | 0.8522 | 0.7286 |
| **Oraculo de SKU (techo)** | 0.7309 | 0.6372 | 0.4632 | 0.8057 | 0.6723 |
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.7181 (91.9%) | 0.5969 (91.2%) | 0.4466 (84.3%) | 0.7895 (92.6%) | 0.6648 (91.2%) |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.6775 (86.7%) | 0.5777 (88.3%) | 0.4418 (83.4%) | 0.7490 (87.9%) | 0.6210 (85.2%) |
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
| **LambdaRank servido: relevancia graduada (predicciones en disco)** | 0.5416 (91.5%) | 0.4031 (87.9%) | 0.2257 (87.2%) | 0.6186 (90.5%) | 0.4857 (93.2%) |
| LambdaRank con relevancia binaria de SKU (ablacion A4) | 0.5510 (93.1%) | 0.3916 (85.4%) | 0.2257 (87.2%) | 0.6309 (92.3%) | 0.4939 (94.7%) |
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
| <= 7 dias | 15.6% | 4.7% | 7.2% | 3.0% | 14.7% | 27.8% | 14.7% | 5.0% | 10.6% |
| <= 14 dias | 27.2% | 7.7% | 11.5% | 12.0% | 15.2% | 27.2% | 24.6% | 7.7% | 16.1% |
| 8-14 dias | 11.6% | 11.7% | 17.2% | 9.1% | 15.3% | 27.0% | 10.0% | 11.7% | 24.3% |
| comprada en la ventana, antes de la cesta | 41.7% | 10.5% | 14.9% | 30.2% | 15.4% | 26.2% | 36.9% | 9.8% | 20.3% |
| sin compra en la ventana | 58.3% | 14.5% | 19.4% | 69.8% | 14.1% | 21.1% | 63.1% | 10.8% | 21.2% |
| total | 100.0% | 12.9% | 17.5% | 100.0% | 14.5% | 22.7% | 100.0% | 10.5% | 20.9% |

**Queries donde el modelo de antes gastaba huecos en categorias recien compradas.** Para cada umbral, las queries cuya lista antes de A1 tenia al menos un hueco en una categoria comprada hace <= d dias, y como les va a cada sistema (lista entera de 5).

| Umbral | Sistema | Queries | Huecos recientes | sku_precision@5 | cat_precision@5 | sku_hit_rate@5 | cat_hit_rate@5 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <= 7 dias | LambdaRank servido: relevancia graduada (predicciones en disco) | 6,693 | 7.8% | 0.1502 | 0.2400 | 0.5573 | 0.7433 |
| <= 7 dias | LambdaRank antes de A1 (historial congelado) | 6,693 | 41.9% | 0.1140 | 0.1551 | 0.4538 | 0.5684 |
| <= 7 dias | Frecuencia personal x due_for_repurchase + referencia favorita | 6,693 | 36.4% | 0.0990 | 0.2013 | 0.4028 | 0.6607 |
| <= 14 dias | LambdaRank servido: relevancia graduada (predicciones en disco) | 9,987 | 21.1% | 0.1487 | 0.2373 | 0.5527 | 0.7395 |
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
- OK — LambdaRank recalculado desde predictions/ = reports/recommender/metrics.json (0.7181 frente a 0.7181)
- OK — Techo realizado compatible con el esperado en las 4 metricas, total y por perfil (max |z| <= 3.5) (peor: sku_hit_p1, z = +1.55)
- OK — Ningun sistema supera al oraculo de categoria en cat_hit_rate (mejor sistema lambdarank = 0.7181)
- OK — Las predicciones de referencia (antes de A1) son las congeladas y cubren las mismas queries (sha256 = el de baseline_pre_a1.json)
<!-- diagnostics:end -->

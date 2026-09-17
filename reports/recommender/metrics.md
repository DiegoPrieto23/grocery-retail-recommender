# Recomendador de cesta (Fase 3, Tarea 3a)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra de este informe se copia a
mano: se recalcula ejecutando ese comando.

## Montaje

| | |
| --- | --- |
| Fuentes de candidatos (ranker) | cestas anteriores a 2025-09-01 |
| Queries de entrenamiento | 2025-09-01 a 2025-11-01 (11,587 cestas con acierto en el pool) |
| Fuentes de candidatos (test) | cestas anteriores a 2025-11-01 |
| Queries de test | desde 2025-11-01 (18,000 cestas) |
| Ranker | LightGBM `lambdarank`, 169 arboles |
| NDCG@5 de validacion | 0.2203 |

El split es temporal **y por cesta**: ninguna cesta se reparte entre train y test, y las
fuentes de candidatos se reajustan para cada ventana con solo el pasado de esa ventana.

## Resultado

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.1759 | 0.1681 | 0.1285 | 0.1457 | 0.1370 | 0.4947 | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.1204 | 0.1068 | 0.0860 | 0.0953 | 0.0893 | 0.3724 | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.0851 | 0.0957 | 0.0475 | 0.0635 | 0.0614 | 0.2114 | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.1881 | 0.1591 | 0.1568 | 0.1579 | 0.1500 | 0.5741 | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.1713 | 0.1851 | 0.1058 | 0.1346 | 0.1303 | 0.4337 | 2.8506 |

## Comparacion

Mismo pool de candidatos, distinta forma de ordenarlo. Es lo que aisla la aportacion del
ranker de la de la primera etapa.

| Sistema | NDCG@5 | Recall@5 | Precision@5 | F1@5 | hit_rate@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (sin aprendizaje) | 0.0881 | 0.0793 | 0.0618 | 0.0695 | 0.2701 |
| LambdaRank sin senal de sesion | 0.1690 | 0.1630 | 0.1256 | 0.1419 | 0.4870 |
| **LambdaRank completo** | **0.1759** | **0.1681** | **0.1285** | **0.1457** | **0.4947** |

### Por perfil, sin senal de sesion

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.1690 | 0.1630 | 0.1256 | 0.1419 | 0.1334 | 0.4870 | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.1197 | 0.1081 | 0.0845 | 0.0948 | 0.0885 | 0.3685 | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.0814 | 0.0945 | 0.0475 | 0.0632 | 0.0613 | 0.2185 | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.1891 | 0.1594 | 0.1569 | 0.1581 | 0.1501 | 0.5717 | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.1555 | 0.1737 | 0.0995 | 0.1265 | 0.1224 | 0.4195 | 2.8506 |

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

| grupo | n_queries | cat_hit_rate@5 | cat_precision@5 | sku_hit_rate@5 | sku_precision@5 |
| --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.6168 | 0.1754 | 0.4947 | 0.1285 |
| 1 - nuevo, carrito vacio | 521 | 0.5624 | 0.1585 | 0.3724 | 0.0860 |
| 2 - nuevo, con articulos | 421 | 0.4371 | 0.1050 | 0.2114 | 0.0475 |
| 3 - recurrente, carrito vacio | 8713 | 0.6848 | 0.2081 | 0.5741 | 0.1568 |
| 4 - recurrente, con articulos | 8345 | 0.5583 | 0.1458 | 0.4337 | 0.1058 |

El sistema acierta la categoria en el **61.7%** de las cestas y el SKU exacto en el **49.5%**; el cociente entre las dos es **80.2%**. La distancia entre las dos columnas mide cuanto del error esta en *elegir la referencia* y no en *saber que categoria toca*. Desde la Fase 7a el surtido es de 8 referencias por categoria y el cliente repite su referencia preferida con la lealtad de la categoria (`DATA_SPEC.md`, "Fidelidad de marca").

## Carrito y diversidad (punto A2)

Con una linea por categoria en cada cesta, un hueco del top-5 se *regala* si su categoria ya esta en el carrito (no puede acertar) o ya salio mas arriba en la lista (de las dos, como mucho acierta una). Dos piezas atacan el problema: tres features de carrito (`cat_in_cart`, `dept_n_in_cart`, `dept_share_in_cart`) y un re-ranking final (`src/recommender/rerank.py`). Reglas servidas: **como maximo 1 referencia(s) por categoria, categorias del carrito relegadas** (`RecommenderConfig.rerank`), las mismas en esta evaluacion, en las tablas de arriba y en la demo.

| Variante | cat_hit_rate@5 | sku_hit_rate@5 | NDCG@5 | Huecos regalados | Huecos en cat. del carrito (perfiles 2 y 4) | Listas con carrito afectadas | Listas con cat. repetida | Categorias distintas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sin features de carrito, sin re-ranking (antes) | 0.6015 | 0.4947 | 0.1741 | 4.2% | 3.0% | 28.3% | 13.6% | 4.86 |
| Sin features de carrito + re-ranking servido | 0.6118 | 0.4954 | 0.1758 | 0.0% | 0.0% | 0.0% | 0.0% | 5.00 |
| Con features de carrito, sin re-ranking | 0.6092 | 0.4959 | 0.1757 | 3.3% | 0.0% | 20.1% | 15.2% | 4.83 |
| Con features de carrito + 1 por categoria | 0.6169 | 0.4947 | 0.1759 | 0.0% | 0.0% | 0.1% | 0.0% | 5.00 |
| **Con features de carrito + 1 por categoria + exclusion del carrito** | **0.6168** | **0.4947** | **0.1759** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **5.00** |

Las filas "sin features de carrito" usan un LambdaRank entrenado aparte con las mismas queries y sin esas tres columnas. Los huecos en categorias del carrito se miden contra el prefijo del ticket; la regla de exclusion usa el carrito en el corte, que ademas incluye los `add_to_cart` de sesion.

Frente a la fila "antes" (mismo entrenamiento, sin las tres features ni reglas), el sistema servido (en negrita) cambia cat_hit_rate@5 en **+1.53 pp** y sku_hit_rate@5 en **+0.00 pp**, y los huecos regalados pasan del 4.2% al 0.0%.

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
| Este sistema, F1@5 (media armonica de Precision@5 y Recall@5 medios) | 0.1457 |
| Este sistema, F1@5 por cesta (media del F1 de cada cesta) | 0.1370 |
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
- **Que se optimiza.** El ranker se entrena con LambdaRank para NDCG@5, no para F1.
- **Contexto.** Aqui se predice a mitad de cesta: lo que ya esta en el carrito queda fuera
  del target. Instacart predice el pedido entero.
- **Agregacion.** Instacart promediaba el F1 de cada pedido; la variante mas cercana es la
  segunda fila ("por cesta"), no la de cabecera.


## Frente a la Fase 3 original

La Fase 3 se entreno sobre el dataset anterior a la Fase 7a (1.500 productos, ~24 referencias por categoria y eleccion de SKU casi aleatoria). Sus cifras estan congeladas en `reports/recommender/baseline_fase3.json`. Mismo codigo, mismas ventanas y mismo numero de queries de test; cambia el dato.

| Metrica | Fase 3 (dataset viejo) | Fase 7c (dataset nuevo) | Cambio |
| --- | ---: | ---: | ---: |
| NDCG@5 | 0.0343 | 0.1759 | 5.13x |
| Recall@5 | 0.0332 | 0.1681 | 5.07x |
| Precision@5 | 0.0251 | 0.1285 | 5.13x |
| F1@5 | 0.0286 | 0.1457 | 5.10x |
| hit_rate@5 (SKU) | 0.1184 | 0.4947 | 4.18x |
| hit_rate@5 (categoria) | 0.5144 | 0.6168 | 1.20x |
| SKU / categoria (hit_rate) | 0.2302 | 0.8019 | 3.48x |
| SKU / categoria (precision) | 0.1371 | 0.7328 | 5.35x |

Un matiz al leerlo: el catalogo pasa de 1.500 a 496 productos, asi que un top-5 al azar tambien acierta mas que antes. El baseline de popularidad de la tabla de comparacion, sobre el mismo pool, es lo que aisla lo que aporta el ranker.

## Techo de la primera etapa

Que parte del target llego siquiera al pool de candidatos. Lo que no esta aqui, el ranker
no lo puede recuperar.

| grupo | n_queries | pool_recall | pool_size_medio |
| --- | --- | --- | --- |
| total | 18000 | 0.7690 | 139.3371 |
| 1 - nuevo, carrito vacio | 521 | 0.3820 | 60.0000 |
| 2 - nuevo, con articulos | 421 | 0.5052 | 107.6532 |
| 3 - recurrente, carrito vacio | 8713 | 0.7689 | 122.5585 |
| 4 - recurrente, con articulos | 8345 | 0.8066 | 163.4072 |

## Que features usa el ranker

Importancia por ganancia, las 25 primeras.

| feature | gain | split |
| --- | --- | --- |
| hist_rank | 84374.0671 | 471 |
| category_idx | 45332.4455 | 1308 |
| hist_n_baskets | 29848.8744 | 382 |
| sess_secs_since_view | 16953.6174 | 273 |
| src_hist | 15806.8137 | 5 |
| prod_pop_all | 12907.8679 | 204 |
| cust_n_products | 12035.2167 | 458 |
| cat_n_purchase_days | 8112.7309 | 280 |
| prod_pop_recent | 7825.2285 | 164 |
| cust_frequency | 7582.5316 | 336 |
| hist_days_since | 6680.4447 | 481 |
| cat_days_since | 6109.5323 | 478 |
| cat_overdue_ratio | 5991.7209 | 486 |
| pop_rank | 5939.8303 | 137 |
| cust_recency_days | 5456.9979 | 415 |
| cat_in_cart | 5016.6893 | 147 |
| pop_score | 4960.1132 | 177 |
| aff_lift_max | 4742.0450 | 313 |
| cust_avg_ticket | 4403.3908 | 449 |
| cat_expected_days | 3908.6568 | 397 |
| als_score | 3900.7353 | 391 |
| is_on_promo | 3570.9461 | 100 |
| aff_conf_sum | 3551.8124 | 236 |
| prod_seasonal_index | 2508.2608 | 265 |
| promo_discount | 2467.7611 | 90 |

<!-- diagnostics:start -->
## Diagnostico: baselines independientes del pool y techo teorico

Generado por `python -m src.recommender.verify_recommender_diagnostics` (2026-09-17), sobre las 18,000 queries de test de las predicciones en disco (sha256 de `recommendations_test.parquet`: `02092ed97cc5`). El oraculo necesita antes `python -m data_generation.export_oracle`. Esta seccion no la reescribe el pipeline: si se reentrena, hay que volver a lanzar el verificador. Puntos A3, A6 y M8 de `docs/diagnostico-fase7.md`.

### Lectura rapida

- **Mejor baseline en cat_hit_rate@5:** Frecuencia personal por categoria + referencia favorita, 0.6613 frente a 0.6168 del LambdaRank: +4.4 pp a favor del baseline.
- **Techo teorico de cat_hit_rate@5:** 0.7816. El LambdaRank alcanza el 78.9% y el mejor baseline el 84.6%.
- **Techo de sku_hit_rate@5:** 0.5920. El LambdaRank alcanza el 83.6%.

### Todos los sistemas, total

Las columnas "% techo" dividen por el oraculo correspondiente sobre las mismas queries. "Huecos regalados" es la parte del top-5 en una categoria que ya esta en el carrito o repetida mas arriba en la lista (`evaluate.wasted_slot_metrics`, punto A2). Los baselines de categoria no regalan ninguno por construccion; el LambdaRank los evita con el re-ranking de `RecommenderConfig.rerank`.

| Sistema | cat_hit_rate@5 | % techo | cat_precision@5 | sku_hit_rate@5 | % techo SKU | sku_precision@5 | NDCG@5 | Huecos regalados |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.7816 | — | 0.2660 | 0.5542 | — | 0.1495 | 0.2004 | 0.0% |
| **Oraculo de SKU (techo)** | 0.7309 | — | 0.2339 | 0.5920 | — | 0.1661 | 0.2269 | 0.0% |
| **LambdaRank (predicciones en disco)** | 0.6168 | 78.9% | 0.1754 | 0.4947 | 83.6% | 0.1285 | 0.1759 | 0.0% |
| Frecuencia personal por categoria + referencia favorita | 0.6613 | 84.6% | 0.2011 | 0.4069 | 68.7% | 0.0998 | 0.1314 | 0.0% |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.6594 | 84.4% | 0.1990 | 0.4038 | 68.2% | 0.0987 | 0.1299 | 0.0% |
| Repetir las referencias favoritas del cliente | 0.6120 | 78.3% | 0.1798 | 0.4561 | 77.0% | 0.1148 | 0.1554 | 1.5% |
| Popularidad de categoria + referencia lider | 0.6109 | 78.2% | 0.1791 | 0.2651 | 44.8% | 0.0605 | 0.0867 | 0.0% |
| Reglas de asociacion de categoria + referencia lider | 0.6096 | 78.0% | 0.1782 | 0.2586 | 43.7% | 0.0591 | 0.0835 | 0.0% |
| Popularidad global (SKU) | 0.5799 | 74.2% | 0.1666 | 0.2662 | 45.0% | 0.0606 | 0.0871 | 0.5% |
| Aleatorio | 0.2754 | 35.2% | 0.0641 | 0.0408 | 6.9% | 0.0082 | 0.0102 | 2.8% |

Cifras congeladas del LambdaRank en `reports/recommender/baseline_pre_diagnostico.json` (2026-09-17, commit `83ab9f7`): cat_hit_rate@5 0.6003, sku_hit_rate@5 0.4948. **Las predicciones en disco ya no son las congeladas**: la fila del LambdaRank de arriba es la version actual.

### cat_hit_rate@5 por perfil (entre parentesis, % del techo del perfil)

| Sistema | Total | Perfil 1 - nuevo, carrito vacio | Perfil 2 - nuevo, con articulos | Perfil 3 - recurrente, carrito vacio | Perfil 4 - recurrente, con articulos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.7816 | 0.6545 | 0.5297 | 0.8522 | 0.7286 |
| **Oraculo de SKU (techo)** | 0.7309 | 0.6372 | 0.4632 | 0.8057 | 0.6723 |
| **LambdaRank (predicciones en disco)** | 0.6168 (78.9%) | 0.5624 (85.9%) | 0.4371 (82.5%) | 0.6848 (80.4%) | 0.5583 (76.6%) |
| Frecuencia personal por categoria + referencia favorita | 0.6613 (84.6%) | 0.5662 (86.5%) | 0.4418 (83.4%) | 0.7458 (87.5%) | 0.5901 (81.0%) |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.6594 (84.4%) | 0.5662 (86.5%) | 0.4418 (83.4%) | 0.7472 (87.7%) | 0.5847 (80.2%) |
| Repetir las referencias favoritas del cliente | 0.6120 (78.3%) | 0.5643 (86.2%) | 0.4204 (79.4%) | 0.6987 (82.0%) | 0.5341 (73.3%) |
| Popularidad de categoria + referencia lider | 0.6109 (78.2%) | 0.5662 (86.5%) | 0.4418 (83.4%) | 0.7007 (82.2%) | 0.5285 (72.5%) |
| Reglas de asociacion de categoria + referencia lider | 0.6096 (78.0%) | 0.5662 (86.5%) | 0.4347 (82.1%) | 0.7007 (82.2%) | 0.5261 (72.2%) |
| Popularidad global (SKU) | 0.5799 (74.2%) | 0.5643 (86.2%) | 0.4204 (79.4%) | 0.6654 (78.1%) | 0.4997 (68.6%) |
| Aleatorio | 0.2754 (35.2%) | 0.2764 (42.2%) | 0.1591 (30.0%) | 0.3324 (39.0%) | 0.2218 (30.4%) |

### sku_hit_rate@5 por perfil (entre parentesis, % del techo de SKU del perfil)

| Sistema | Total | Perfil 1 - nuevo, carrito vacio | Perfil 2 - nuevo, con articulos | Perfil 3 - recurrente, carrito vacio | Perfil 4 - recurrente, con articulos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.5542 | 0.4203 | 0.2494 | 0.6311 | 0.4975 |
| **Oraculo de SKU (techo)** | 0.5920 | 0.4587 | 0.2589 | 0.6837 | 0.5214 |
| **LambdaRank (predicciones en disco)** | 0.4947 (83.6%) | 0.3724 (81.2%) | 0.2114 (81.7%) | 0.5741 (84.0%) | 0.4337 (83.2%) |
| Frecuencia personal por categoria + referencia favorita | 0.4069 (68.7%) | 0.3090 (67.4%) | 0.1853 (71.6%) | 0.4789 (70.1%) | 0.3491 (67.0%) |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.4038 (68.2%) | 0.3090 (67.4%) | 0.1853 (71.6%) | 0.4788 (70.0%) | 0.3424 (65.7%) |
| Repetir las referencias favoritas del cliente | 0.4561 (77.0%) | 0.3090 (67.4%) | 0.1829 (70.6%) | 0.5465 (79.9%) | 0.3845 (73.8%) |
| Popularidad de categoria + referencia lider | 0.2651 (44.8%) | 0.3090 (67.4%) | 0.1853 (71.6%) | 0.3219 (47.1%) | 0.2070 (39.7%) |
| Reglas de asociacion de categoria + referencia lider | 0.2586 (43.7%) | 0.3090 (67.4%) | 0.1686 (65.1%) | 0.3219 (47.1%) | 0.1938 (37.2%) |
| Popularidad global (SKU) | 0.2662 (45.0%) | 0.3090 (67.4%) | 0.1829 (70.6%) | 0.3208 (46.9%) | 0.2108 (40.4%) |
| Aleatorio | 0.0408 (6.9%) | 0.0230 (5.0%) | 0.0285 (11.0%) | 0.0487 (7.1%) | 0.0344 (6.6%) |

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

- OK — Una linea por categoria: el target en categorias coincide con n_target
- OK — LambdaRank recalculado desde predictions/ = reports/recommender/metrics.json (0.6168 frente a 0.6168)
- OK — Techo realizado compatible con el esperado en las 4 metricas, total y por perfil (max |z| <= 3.5) (peor: sku_hit_p1, z = +1.55)
- OK — Ningun sistema supera al oraculo de categoria en cat_hit_rate (mejor sistema personal_frequency = 0.6613)
<!-- diagnostics:end -->

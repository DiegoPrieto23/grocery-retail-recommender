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
| Ranker | LightGBM `lambdarank`, 87 arboles |
| NDCG@5 de validacion | 0.2166 |

El split es temporal **y por cesta**: ninguna cesta se reparte entre train y test, y las
fuentes de candidatos se reajustan para cada ventana con solo el pasado de esa ventana.

## Resultado

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.1752 | 0.1678 | 0.1282 | 0.1454 | 0.1367 | 0.4948 | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.1185 | 0.1070 | 0.0856 | 0.0951 | 0.0891 | 0.3724 | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.0835 | 0.0924 | 0.0456 | 0.0611 | 0.0593 | 0.2090 | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.1886 | 0.1593 | 0.1569 | 0.1581 | 0.1502 | 0.5752 | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.1694 | 0.1844 | 0.1050 | 0.1338 | 0.1296 | 0.4328 | 2.8506 |

## Comparacion

Mismo pool de candidatos, distinta forma de ordenarlo. Es lo que aisla la aportacion del
ranker de la de la primera etapa.

| Sistema | NDCG@5 | Recall@5 | Precision@5 | F1@5 | hit_rate@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (sin aprendizaje) | 0.0868 | 0.0783 | 0.0611 | 0.0686 | 0.2673 |
| LambdaRank sin senal de sesion | 0.1667 | 0.1606 | 0.1241 | 0.1400 | 0.4832 |
| **LambdaRank completo** | **0.1752** | **0.1678** | **0.1282** | **0.1454** | **0.4948** |

### Por perfil, sin senal de sesion

| grupo | n_queries | ndcg@5 | recall@5 | precision@5 | f1@5 | f1@5_por_cesta | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.1667 | 0.1606 | 0.1241 | 0.1400 | 0.1316 | 0.4832 | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.1199 | 0.1031 | 0.0871 | 0.0944 | 0.0890 | 0.3551 | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.0779 | 0.0871 | 0.0428 | 0.0574 | 0.0557 | 0.2019 | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.1881 | 0.1593 | 0.1568 | 0.1580 | 0.1500 | 0.5737 | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.1518 | 0.1693 | 0.0964 | 0.1229 | 0.1189 | 0.4109 | 2.8506 |

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

| grupo | n_queries | cat_hit_rate@5 | cat_precision@5 | sku_hit_rate@5 | sku_precision@5 |
| --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.6003 | 0.1764 | 0.4948 | 0.1282 |
| 1 - nuevo, carrito vacio | 521 | 0.5701 | 0.1597 | 0.3724 | 0.0856 |
| 2 - nuevo, con articulos | 421 | 0.4228 | 0.1097 | 0.2090 | 0.0456 |
| 3 - recurrente, carrito vacio | 8713 | 0.6754 | 0.2102 | 0.5752 | 0.1569 |
| 4 - recurrente, con articulos | 8345 | 0.5328 | 0.1456 | 0.4328 | 0.1050 |

El sistema acierta la categoria en el **60.0%** de las cestas y el SKU exacto en el **49.5%**; el cociente entre las dos es **82.4%**. La distancia entre las dos columnas mide cuanto del error esta en *elegir la referencia* y no en *saber que categoria toca*. Desde la Fase 7a el surtido es de 8 referencias por categoria y el cliente repite su referencia preferida con la lealtad de la categoria (`DATA_SPEC.md`, "Fidelidad de marca").

## F1@5 frente a Kaggle "Instacart Market Basket Analysis"

| | F1 |
| --- | ---: |
| Este sistema, F1@5 (media armonica de Precision@5 y Recall@5 medios) | 0.1454 |
| Este sistema, F1@5 por cesta (media del F1 de cada cesta) | 0.1367 |
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
| NDCG@5 | 0.0343 | 0.1752 | 5.11x |
| Recall@5 | 0.0332 | 0.1678 | 5.06x |
| Precision@5 | 0.0251 | 0.1282 | 5.11x |
| F1@5 | 0.0286 | 0.1454 | 5.09x |
| hit_rate@5 (SKU) | 0.1184 | 0.4948 | 4.18x |
| hit_rate@5 (categoria) | 0.5144 | 0.6003 | 1.17x |
| SKU / categoria (hit_rate) | 0.2302 | 0.8242 | 3.58x |
| SKU / categoria (precision) | 0.1371 | 0.7266 | 5.30x |

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
| hist_rank | 90368.6487 | 222 |
| category_idx | 42011.6916 | 811 |
| hist_n_baskets | 20951.4951 | 244 |
| sess_secs_since_view | 16276.1651 | 231 |
| src_hist | 14024.1772 | 6 |
| prod_pop_all | 12704.1900 | 135 |
| cust_n_products | 11010.5746 | 281 |
| cat_n_purchase_days | 7336.4265 | 199 |
| pop_score | 6057.5926 | 98 |
| cust_frequency | 6022.9253 | 227 |
| pop_rank | 5083.8064 | 63 |
| prod_pop_recent | 5018.0350 | 72 |
| aff_lift_max | 4347.2485 | 208 |
| cat_days_since | 4304.9225 | 261 |
| cat_overdue_ratio | 4160.4213 | 241 |
| hist_days_since | 3751.5081 | 203 |
| is_on_promo | 3656.0422 | 86 |
| hist_units | 3003.7098 | 95 |
| promo_discount | 2468.5317 | 79 |
| cust_recency_days | 2440.6536 | 165 |
| aff_conf_sum | 2421.1462 | 135 |
| cat_expected_days | 2281.9635 | 171 |
| cust_avg_ticket | 2133.6958 | 153 |
| als_score | 1843.9430 | 138 |
| n_sources | 1758.9496 | 46 |

<!-- diagnostics:start -->
## Diagnostico: baselines independientes del pool y techo teorico

Generado por `python -m src.recommender.verify_recommender_diagnostics` (2026-09-17), sobre las 18,000 queries de test de las predicciones en disco (sha256 de `recommendations_test.parquet`: `6fcf750fbae0`). El oraculo necesita antes `python -m data_generation.export_oracle`. Esta seccion no la reescribe el pipeline: si se reentrena, hay que volver a lanzar el verificador. Puntos A3, A6 y M8 de `docs/diagnostico-fase7.md`.

### Lectura rapida

- **Mejor baseline en cat_hit_rate@5:** Frecuencia personal por categoria + referencia favorita, 0.6613 frente a 0.6003 del LambdaRank: +6.1 pp a favor del baseline.
- **Techo teorico de cat_hit_rate@5:** 0.7816. El LambdaRank alcanza el 76.8% y el mejor baseline el 84.6%.
- **Techo de sku_hit_rate@5:** 0.5920. El LambdaRank alcanza el 83.6%.

### Todos los sistemas, total

Las columnas "% techo" dividen por el oraculo correspondiente sobre las mismas queries. Los baselines nunca recomiendan una categoria que ya esta en el carrito; el LambdaRank si puede hacerlo (punto A2).

| Sistema | cat_hit_rate@5 | % techo | cat_precision@5 | sku_hit_rate@5 | % techo SKU | sku_precision@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.7816 | — | 0.2660 | 0.5542 | — | 0.1495 | 0.2004 |
| **Oraculo de SKU (techo)** | 0.7309 | — | 0.2339 | 0.5920 | — | 0.1661 | 0.2269 |
| **LambdaRank (Fase 7c, predicciones en disco)** | 0.6003 | 76.8% | 0.1764 | 0.4948 | 83.6% | 0.1282 | 0.1752 |
| Frecuencia personal por categoria + referencia favorita | 0.6613 | 84.6% | 0.2011 | 0.4069 | 68.7% | 0.0998 | 0.1314 |
| Frecuencia personal x due_for_repurchase + referencia favorita | 0.6594 | 84.4% | 0.1990 | 0.4038 | 68.2% | 0.0987 | 0.1299 |
| Repetir las referencias favoritas del cliente | 0.6120 | 78.3% | 0.1798 | 0.4561 | 77.0% | 0.1148 | 0.1554 |
| Popularidad de categoria + referencia lider | 0.6109 | 78.2% | 0.1791 | 0.2651 | 44.8% | 0.0605 | 0.0867 |
| Reglas de asociacion de categoria + referencia lider | 0.6096 | 78.0% | 0.1782 | 0.2586 | 43.7% | 0.0591 | 0.0835 |
| Popularidad global (SKU) | 0.5799 | 74.2% | 0.1666 | 0.2662 | 45.0% | 0.0606 | 0.0871 |
| Aleatorio | 0.2754 | 35.2% | 0.0641 | 0.0408 | 6.9% | 0.0082 | 0.0102 |

Cifras congeladas del LambdaRank en `reports/recommender/baseline_pre_diagnostico.json` (2026-09-17, commit `83ab9f7`): cat_hit_rate@5 0.6003, sku_hit_rate@5 0.4948. Las predicciones en disco las reproducen exactamente.

### cat_hit_rate@5 por perfil (entre parentesis, % del techo del perfil)

| Sistema | Total | Perfil 1 - nuevo, carrito vacio | Perfil 2 - nuevo, con articulos | Perfil 3 - recurrente, carrito vacio | Perfil 4 - recurrente, con articulos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Oraculo de categoria (techo)** | 0.7816 | 0.6545 | 0.5297 | 0.8522 | 0.7286 |
| **Oraculo de SKU (techo)** | 0.7309 | 0.6372 | 0.4632 | 0.8057 | 0.6723 |
| **LambdaRank (Fase 7c, predicciones en disco)** | 0.6003 (76.8%) | 0.5701 (87.1%) | 0.4228 (79.8%) | 0.6754 (79.3%) | 0.5328 (73.1%) |
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
| **LambdaRank (Fase 7c, predicciones en disco)** | 0.4948 (83.6%) | 0.3724 (81.2%) | 0.2090 (80.7%) | 0.5752 (84.1%) | 0.4328 (83.0%) |
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
- OK — LambdaRank recalculado desde predictions/ = reports/recommender/metrics.json (0.6003 frente a 0.6003)
- OK — Techo realizado compatible con el esperado en las 4 metricas, total y por perfil (max |z| <= 3.5) (peor: sku_hit_p1, z = +1.55)
- OK — Ningun sistema supera al oraculo de categoria en cat_hit_rate (mejor sistema personal_frequency = 0.6613)
<!-- diagnostics:end -->

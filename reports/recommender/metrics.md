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

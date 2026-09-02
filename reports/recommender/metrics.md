# Recomendador de cesta (Fase 3, Tarea 3a)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra de este informe se copia a
mano: se recalcula ejecutando ese comando.

## Montaje

| | |
| --- | --- |
| Fuentes de candidatos (ranker) | cestas anteriores a 2025-09-01 |
| Queries de entrenamiento | 2025-09-01 a 2025-11-01 (8,549 cestas con acierto en el pool) |
| Fuentes de candidatos (test) | cestas anteriores a 2025-11-01 |
| Queries de test | desde 2025-11-01 (18,000 cestas) |
| Ranker | LightGBM `lambdarank`, 84 arboles |
| NDCG@5 de validacion | 0.0836 |

El split es temporal **y por cesta**: ninguna cesta se reparte entre train y test, y las
fuentes de candidatos se reajustan para cada ventana con solo el pasado de esa ventana.

## Resultado

| grupo | n_queries | ndcg@5 | recall@5 | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.0343 | 0.0332 | 0.1184 | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.0212 | 0.0191 | 0.0729 | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.0149 | 0.0168 | 0.0451 | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.0352 | 0.0304 | 0.1414 | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.0351 | 0.0378 | 0.1009 | 2.8506 |

## Comparacion

Mismo pool de candidatos, distinta forma de ordenarlo. Es lo que aisla la aportacion del
ranker de la de la primera etapa.

| Sistema | NDCG@5 | Recall@5 | hit_rate@5 |
| --- | ---: | ---: | ---: |
| Popularidad reciente x estacionalidad (sin aprendizaje) | 0.0200 | 0.0221 | 0.0831 |
| LambdaRank sin senal de sesion | 0.0303 | 0.0302 | 0.1107 |
| **LambdaRank completo** | **0.0343** | **0.0332** | **0.1184** |

### Por perfil, sin senal de sesion

| grupo | n_queries | ndcg@5 | recall@5 | hit_rate@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.0303 | 0.0302 | 0.1107 | 3.9327 |
| 1 - nuevo, carrito vacio | 521 | 0.0204 | 0.0213 | 0.0749 | 4.1651 |
| 2 - nuevo, con articulos | 421 | 0.0145 | 0.0164 | 0.0404 | 2.3444 |
| 3 - recurrente, carrito vacio | 8713 | 0.0345 | 0.0299 | 0.1400 | 5.0319 |
| 4 - recurrente, con articulos | 8345 | 0.0273 | 0.0317 | 0.0859 | 2.8506 |

## SKU o categoria: donde falla exactamente

La misma lista, puntuada dos veces. "Acierta la categoria" significa que el producto
recomendado pertenece a una categoria que el cliente si acabo comprando, aunque la
referencia concreta fuera otra.

| grupo | n_queries | cat_hit_rate@5 | cat_precision@5 | sku_hit_rate@5 | sku_precision@5 |
| --- | --- | --- | --- | --- | --- |
| total | 18000 | 0.5144 | 0.1828 | 0.1184 | 0.0251 |
| 1 - nuevo, carrito vacio | 521 | 0.4645 | 0.1336 | 0.0729 | 0.0154 |
| 2 - nuevo, con articulos | 421 | 0.2898 | 0.1074 | 0.0451 | 0.0090 |
| 3 - recurrente, carrito vacio | 8713 | 0.6110 | 0.2196 | 0.1414 | 0.0300 |
| 4 - recurrente, con articulos | 8345 | 0.4279 | 0.1513 | 0.1009 | 0.0213 |

La distancia entre las dos columnas es la respuesta a por que el NDCG@5 de SKU es bajo:
el sistema **si sabe que categoria toca**, y falla al elegir cual de las ~24 referencias de
esa categoria. En este dataset ese segundo paso esta cerca del azar por construccion --
un cliente con tres o mas compras en una categoria compra 0,86 referencias distintas por
compra, es decir casi nunca repite SKU--, asi que el techo de la metrica de SKU lo pone el
generador, no el modelo. Ver la nota de la Fase 3 en `ROADMAP.md`.

## Techo de la primera etapa

Que parte del target llego siquiera al pool de candidatos. Lo que no esta aqui, el ranker
no lo puede recuperar.

| grupo | n_queries | pool_recall | pool_size_medio |
| --- | --- | --- | --- |
| total | 18000 | 0.3109 | 155.1491 |
| 1 - nuevo, carrito vacio | 521 | 0.1630 | 60.0000 |
| 2 - nuevo, con articulos | 421 | 0.2177 | 97.5321 |
| 3 - recurrente, carrito vacio | 8713 | 0.2996 | 142.4367 |
| 4 - recurrente, con articulos | 8345 | 0.3365 | 177.2693 |

## Que features usa el ranker

Importancia por ganancia, las 25 primeras.

| feature | gain | split |
| --- | --- | --- |
| category_idx | 30927.2127 | 682 |
| sess_secs_since_view | 9609.1249 | 72 |
| cat_days_since | 7567.2839 | 312 |
| prod_pop_all | 7383.5090 | 147 |
| cat_overdue_ratio | 6792.2103 | 270 |
| cat_expected_days | 6018.0669 | 267 |
| cust_avg_ticket | 5757.5693 | 266 |
| cust_recency_days | 5598.7113 | 269 |
| hist_days_since | 5380.8287 | 253 |
| cust_n_products | 4640.0053 | 205 |
| cust_frequency | 4321.3056 | 191 |
| als_score | 4064.9503 | 186 |
| als_rank | 4005.2492 | 185 |
| prod_pop_recent | 3712.3610 | 96 |
| hist_rank | 3533.8470 | 167 |
| aff_lift_max | 3327.9575 | 144 |
| prod_month_rank | 3290.7126 | 77 |
| cat_n_purchase_days | 3271.6208 | 128 |
| pop_score | 3243.4929 | 110 |
| is_on_promo | 2790.6237 | 53 |
| prod_seasonal_index | 2583.5448 | 128 |
| cataff_lift_max | 2503.6806 | 109 |
| aff_conf_sum | 2360.4281 | 106 |
| cataff_conf_max | 2169.4033 | 87 |
| basket_dow | 2118.9309 | 105 |

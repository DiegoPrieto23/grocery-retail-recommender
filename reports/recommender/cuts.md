# Evaluacion con varios cortes por cesta (punto M3)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra se copia a mano. La cifra de
cabecera del recomendador sigue siendo la de [`metrics.md`](metrics.md): un corte por cesta
(carrito vacio o mitad del ticket), para no romper la serie historica. Este informe
responde a otra pregunta: **como cambia el acierto a medida que se llena el carrito**.

## Montaje

| | |
| --- | --- |
| Cortes | todos los cortes `k` = 1..n-1 de cada cesta (`CutPlan(mode="all_prefixes")`) |
| Cestas | 1,800 de la ventana de test, desde 2025-11-01 (cestas de 2 o mas lineas) |
| Queries | 12,947 (7.2 por cesta) |
| Cestas dentro de la muestra de cabecera | si: son las primeras de la misma lista |
| Modelo y fuentes | los de la cabecera (mismo LambdaRank, mismo re-ranking) |
| Intervalos | bootstrap por cesta, 1,000 remuestreos, 95% |

**El orden de las lineas no aporta informacion.** En las cestas con sesion, el generador
asigna los `add_to_cart` con una permutacion aleatoria del ticket, y en las demas el orden
lo pone un hash (`src/recommender/splits.py`). El prefijo de tamano `k` es, a efectos
practicos, un subconjunto aleatorio de `k` lineas. Un corte "tardio" no es "el final de la
compra": es una cesta con mas contexto y menos que adivinar.

Las queries de una misma cesta estan correladas, asi que `n_cestas` (y no `n_queries`) es
el tamano efectivo de cada fila. Dos efectos mecanicos al leer las tablas:

- al crecer `k` quedan menos productos por adivinar (`n_target_medio`), y el
  `hit_rate` y el `recall` se mueven aunque el modelo no cambie;
- las filas altas de `prefix_size` solo tienen cestas largas, que son otra poblacion.

La fraccion del ticket (`prefix_size / n_items`) compara cestas de distinto tamano en el
mismo punto de la compra, y corrige en parte lo segundo.

## Lectura rapida

Con 1 linea en el carrito, NDCG@5 graduada 0.3310 y cat_hit_rate@5
0.8183. Con 10+ lineas, 0.4655 y 0.8832
(n_target medio 7.2 frente a 9.4).

## Todos los cortes

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| todos los cortes | 12947 | 1800 | 0.3834 [0.3662, 0.4008] | 0.8387 [0.8261, 0.8501] | 0.6979 [0.6799, 0.7169] | 0.3461 [0.3289, 0.3639] | 0.2149 [0.2062, 0.2234] | 8.4433 |

## Por `prefix_size` (lineas ya en el carrito)

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1800 | 1800 | 0.3310 [0.3192, 0.3428] | 0.8183 [0.8006, 0.8356] | 0.6500 [0.6278, 0.6733] | 0.3005 [0.2875, 0.3141] | 0.2157 [0.2037, 0.2279] | 7.1928 |
| 2 | 1511 | 1511 | 0.3307 [0.3190, 0.3442] | 0.8187 [0.7995, 0.8372] | 0.6545 [0.6327, 0.6823] | 0.2982 [0.2848, 0.3144] | 0.2110 [0.1980, 0.2242] | 7.3772 |
| 3 | 1261 | 1261 | 0.3379 [0.3230, 0.3531] | 0.8224 [0.8002, 0.8430] | 0.6542 [0.6273, 0.6796] | 0.3008 [0.2848, 0.3167] | 0.2074 [0.1932, 0.2203] | 7.6416 |
| 4 | 1059 | 1059 | 0.3414 [0.3252, 0.3568] | 0.8244 [0.8007, 0.8470] | 0.6610 [0.6336, 0.6912] | 0.3056 [0.2881, 0.3227] | 0.2065 [0.1913, 0.2226] | 7.9084 |
| 5 | 886 | 886 | 0.3494 [0.3316, 0.3672] | 0.8262 [0.7991, 0.8510] | 0.6546 [0.6230, 0.6851] | 0.3159 [0.2964, 0.3358] | 0.2077 [0.1908, 0.2244] | 8.2573 |
| 6 | 748 | 748 | 0.3511 [0.3298, 0.3709] | 0.8088 [0.7781, 0.8342] | 0.6604 [0.6270, 0.6939] | 0.3120 [0.2903, 0.3329] | 0.1868 [0.1701, 0.2031] | 8.5963 |
| 7 | 625 | 625 | 0.3670 [0.3449, 0.3914] | 0.7936 [0.7600, 0.8240] | 0.6800 [0.6432, 0.7168] | 0.3239 [0.3002, 0.3474] | 0.1851 [0.1690, 0.2019] | 9.0912 |
| 8 | 541 | 541 | 0.3834 [0.3592, 0.4081] | 0.8170 [0.7856, 0.8521] | 0.7079 [0.6710, 0.7468] | 0.3444 [0.3197, 0.3713] | 0.1939 [0.1753, 0.2125] | 9.3475 |
| 9 | 465 | 465 | 0.4009 [0.3753, 0.4273] | 0.8301 [0.7914, 0.8624] | 0.7355 [0.6925, 0.7785] | 0.3676 [0.3409, 0.3962] | 0.2061 [0.1860, 0.2290] | 9.7118 |
| 10+ | 4051 | 412 | 0.4655 [0.4323, 0.4957] | 0.8832 [0.8614, 0.9014] | 0.7722 [0.7383, 0.8016] | 0.4231 [0.3880, 0.4545] | 0.2357 [0.2208, 0.2510] | 9.4320 |

## Por fraccion del ticket ya en el carrito

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (0%, 25%] | 2941 | 1261 | 0.4902 [0.4712, 0.5088] | 0.9514 [0.9420, 0.9609] | 0.8477 [0.8293, 0.8655] | 0.4153 [0.3951, 0.4359] | 0.1550 [0.1488, 0.1617] | 15.3159 |
| (25%, 50%] | 4014 | 1800 | 0.4119 [0.3916, 0.4313] | 0.8829 [0.8690, 0.8964] | 0.7439 [0.7223, 0.7649] | 0.3656 [0.3454, 0.3855] | 0.2040 [0.1957, 0.2129] | 9.2362 |
| (50%, 75%] | 3433 | 1511 | 0.3552 [0.3329, 0.3759] | 0.8211 [0.8024, 0.8386] | 0.6604 [0.6323, 0.6856] | 0.3258 [0.3041, 0.3465] | 0.2308 [0.2201, 0.2413] | 5.9650 |
| (75%, 100%] | 2559 | 1059 | 0.2540 [0.2343, 0.2729] | 0.6635 [0.6357, 0.6901] | 0.5041 [0.4724, 0.5337] | 0.2633 [0.2419, 0.2852] | 0.2793 [0.2598, 0.2977] | 2.6256 |

## Referencia: el corte de cabecera sobre las mismas cestas

Las mismas cestas, evaluadas con el corte de siempre (una query por cesta: la mitad con el
carrito vacio y la otra mitad con la mitad del ticket).

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cabecera: carrito vacio | 888 | 888 | 0.3384 [0.3222, 0.3555] | 0.8649 [0.8423, 0.8863] | 0.6926 [0.6599, 0.7218] | 0.2898 [0.2721, 0.3077] | 0.1860 [0.1747, 0.1987] | 8.0405 |
| cabecera: mitad del ticket | 912 | 912 | 0.3034 [0.2877, 0.3208] | 0.7982 [0.7752, 0.8246] | 0.6217 [0.5910, 0.6546] | 0.2909 [0.2720, 0.3103] | 0.2581 [0.2389, 0.2768] | 4.3980 |
| cabecera: las dos | 1800 | 1800 | 0.3206 [0.3094, 0.3327] | 0.8311 [0.8128, 0.8478] | 0.6567 [0.6328, 0.6789] | 0.2904 [0.2771, 0.3043] | 0.2225 [0.2105, 0.2341] | 6.1950 |

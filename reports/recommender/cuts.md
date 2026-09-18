# Evaluacion con varios cortes por cesta (punto M3)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra se copia a mano. La cifra de
cabecera del recomendador sigue siendo la de [`metrics.md`](metrics.md): un corte por cesta
(carrito vacio o mitad del ticket), para no romper la serie historica. Este informe
responde a otra pregunta: **como cambia el acierto a medida que se llena el carrito**.

## Montaje

| | |
| --- | --- |
| Cortes | todos los cortes `k` = 1..n-1 de cada cesta (`CutPlan(mode="all_prefixes")`) |
| Cestas | 2,712 de la ventana de test, desde 2025-11-01 (cestas de 2 o mas lineas) |
| Queries | 19,108 (7.0 por cesta) |
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

Con 1 linea en el carrito, NDCG@5 graduada 0.3286 y cat_hit_rate@5
0.8164. Con 10+ lineas, 0.4599 y 0.8844
(n_target medio 7.0 frente a 9.5).

## Todos los cortes

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| todos los cortes | 19108 | 2712 | 0.3821 [0.3673, 0.3956] | 0.8420 [0.8310, 0.8514] | 0.6971 [0.6803, 0.7122] | 0.3441 [0.3296, 0.3582] | 0.2150 [0.2078, 0.2216] | 8.3796 |

## Por `prefix_size` (lineas ya en el carrito)

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2712 | 2712 | 0.3286 [0.3189, 0.3375] | 0.8164 [0.8016, 0.8304] | 0.6401 [0.6220, 0.6574] | 0.2973 [0.2860, 0.3075] | 0.2162 [0.2060, 0.2259] | 7.0457 |
| 2 | 2269 | 2269 | 0.3344 [0.3243, 0.3445] | 0.8197 [0.8052, 0.8352] | 0.6615 [0.6430, 0.6814] | 0.3041 [0.2940, 0.3151] | 0.2174 [0.2074, 0.2289] | 7.2261 |
| 3 | 1880 | 1880 | 0.3387 [0.3269, 0.3504] | 0.8271 [0.8090, 0.8436] | 0.6521 [0.6298, 0.6724] | 0.3025 [0.2882, 0.3153] | 0.2128 [0.2015, 0.2243] | 7.5144 |
| 4 | 1574 | 1574 | 0.3410 [0.3288, 0.3543] | 0.8202 [0.8018, 0.8399] | 0.6423 [0.6207, 0.6658] | 0.3045 [0.2917, 0.3197] | 0.2050 [0.1934, 0.2176] | 7.7808 |
| 5 | 1301 | 1301 | 0.3506 [0.3354, 0.3651] | 0.8271 [0.8055, 0.8470] | 0.6541 [0.6280, 0.6779] | 0.3165 [0.2993, 0.3323] | 0.2036 [0.1904, 0.2178] | 8.2037 |
| 6 | 1087 | 1087 | 0.3602 [0.3437, 0.3768] | 0.8270 [0.8050, 0.8500] | 0.6743 [0.6477, 0.7020] | 0.3230 [0.3051, 0.3408] | 0.1985 [0.1849, 0.2126] | 8.6219 |
| 7 | 927 | 927 | 0.3672 [0.3495, 0.3859] | 0.8123 [0.7875, 0.8371] | 0.6882 [0.6602, 0.7184] | 0.3285 [0.3100, 0.3470] | 0.1972 [0.1826, 0.2129] | 8.9374 |
| 8 | 780 | 780 | 0.3873 [0.3665, 0.4060] | 0.8346 [0.8077, 0.8603] | 0.7154 [0.6833, 0.7462] | 0.3471 [0.3259, 0.3670] | 0.1948 [0.1803, 0.2112] | 9.4333 |
| 9 | 678 | 678 | 0.4050 [0.3810, 0.4261] | 0.8422 [0.8127, 0.8658] | 0.7404 [0.7065, 0.7729] | 0.3690 [0.3438, 0.3908] | 0.2044 [0.1859, 0.2222] | 9.7021 |
| 10+ | 5900 | 602 | 0.4599 [0.4368, 0.4854] | 0.8844 [0.8683, 0.8995] | 0.7736 [0.7487, 0.7971] | 0.4142 [0.3889, 0.4397] | 0.2291 [0.2169, 0.2423] | 9.4871 |

## Por fraccion del ticket ya en el carrito

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (0%, 25%] | 4338 | 1880 | 0.4896 [0.4730, 0.5046] | 0.9514 [0.9436, 0.9587] | 0.8456 [0.8290, 0.8606] | 0.4148 [0.3971, 0.4311] | 0.1566 [0.1513, 0.1621] | 15.2003 |
| (25%, 50%] | 5950 | 2712 | 0.4087 [0.3930, 0.4241] | 0.8835 [0.8732, 0.8936] | 0.7403 [0.7238, 0.7566] | 0.3610 [0.3443, 0.3765] | 0.2035 [0.1964, 0.2106] | 9.1358 |
| (50%, 75%] | 5065 | 2269 | 0.3538 [0.3379, 0.3695] | 0.8274 [0.8132, 0.8417] | 0.6654 [0.6432, 0.6864] | 0.3242 [0.3077, 0.3393] | 0.2324 [0.2233, 0.2419] | 5.9159 |
| (75%, 100%] | 3755 | 1574 | 0.2539 [0.2396, 0.2687] | 0.6692 [0.6465, 0.6909] | 0.4999 [0.4734, 0.5264] | 0.2627 [0.2471, 0.2794] | 0.2770 [0.2613, 0.2932] | 2.6250 |

## Referencia: el corte de cabecera sobre las mismas cestas

Las mismas cestas, evaluadas con el corte de siempre (una query por cesta: la mitad con el
carrito vacio y la otra mitad con la mitad del ticket).

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cabecera: carrito vacio | 1352 | 1352 | 0.3357 [0.3224, 0.3496] | 0.8536 [0.8350, 0.8713] | 0.6864 [0.6620, 0.7108] | 0.2881 [0.2743, 0.3023] | 0.1857 [0.1752, 0.1956] | 8.0104 |
| cabecera: mitad del ticket | 1360 | 1360 | 0.2942 [0.2802, 0.3075] | 0.7824 [0.7595, 0.8037] | 0.5963 [0.5691, 0.6199] | 0.2789 [0.2629, 0.2938] | 0.2478 [0.2324, 0.2629] | 4.2662 |
| cabecera: las dos | 2712 | 2712 | 0.3149 [0.3058, 0.3240] | 0.8178 [0.8024, 0.8326] | 0.6412 [0.6228, 0.6578] | 0.2835 [0.2732, 0.2934] | 0.2168 [0.2080, 0.2253] | 6.1327 |

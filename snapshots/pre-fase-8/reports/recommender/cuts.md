# Evaluacion con varios cortes por cesta (punto M3)

Generado por `python -m src.recommender.pipeline`. Ninguna cifra se copia a mano. La cifra de
cabecera del recomendador sigue siendo la de [`metrics.md`](metrics.md): un corte por cesta
(carrito vacio o mitad del ticket), para no romper la serie historica. Este informe
responde a otra pregunta: **como cambia el acierto a medida que se llena el carrito**.

## Montaje

| | |
| --- | --- |
| Cortes | todos los cortes `k` = 1..n-1 de cada cesta (`CutPlan(mode="all_prefixes")`) |
| Cestas | 2,947 de la ventana de test, desde 2025-11-01 (cestas de 2 o mas lineas) |
| Queries | 12,400 (4.2 por cesta) |
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

Con 1 linea en el carrito, NDCG@5 graduada 0.2250 y cat_hit_rate@5
0.7340. Con 10+ lineas, 0.1801 y 0.5714
(n_target medio 4.2 frente a 1.9).

## Todos los cortes

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| todos los cortes | 12400 | 2947 | 0.1989 [0.1921, 0.2061] | 0.6512 [0.6372, 0.6652] | 0.4761 [0.4606, 0.4926] | 0.1959 [0.1877, 0.2046] | 0.2047 [0.1967, 0.2133] | 3.1248 |

## Por `prefix_size` (lineas ya en el carrito)

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2947 | 2947 | 0.2250 [0.2177, 0.2323] | 0.7340 [0.7173, 0.7503] | 0.5623 [0.5439, 0.5809] | 0.2068 [0.1983, 0.2152] | 0.1888 [0.1811, 0.1967] | 4.2077 |
| 2 | 2727 | 2727 | 0.2063 [0.1984, 0.2139] | 0.6821 [0.6656, 0.7001] | 0.5053 [0.4866, 0.5259] | 0.1957 [0.1864, 0.2045] | 0.1942 [0.1852, 0.2036] | 3.4664 |
| 3 | 2313 | 2313 | 0.1906 [0.1829, 0.1987] | 0.6308 [0.6118, 0.6515] | 0.4544 [0.4358, 0.4752] | 0.1877 [0.1783, 0.1975] | 0.2002 [0.1891, 0.2115] | 2.9079 |
| 4 | 1728 | 1728 | 0.1798 [0.1708, 0.1903] | 0.5914 [0.5700, 0.6152] | 0.4167 [0.3952, 0.4410] | 0.1834 [0.1720, 0.1960] | 0.2055 [0.1920, 0.2199] | 2.5538 |
| 5 | 1182 | 1182 | 0.1799 [0.1671, 0.1918] | 0.5770 [0.5499, 0.6049] | 0.4027 [0.3739, 0.4298] | 0.1954 [0.1781, 0.2130] | 0.2253 [0.2063, 0.2442] | 2.2716 |
| 6 | 721 | 721 | 0.1804 [0.1651, 0.1948] | 0.5895 [0.5534, 0.6255] | 0.4064 [0.3689, 0.4397] | 0.1937 [0.1729, 0.2128] | 0.2320 [0.2077, 0.2557] | 2.0846 |
| 7 | 386 | 386 | 0.1851 [0.1653, 0.2066] | 0.5959 [0.5440, 0.6451] | 0.4197 [0.3679, 0.4689] | 0.2064 [0.1795, 0.2343] | 0.2547 [0.2193, 0.2902] | 2.0259 |
| 8 | 194 | 194 | 0.1975 [0.1665, 0.2253] | 0.6031 [0.5361, 0.6702] | 0.4381 [0.3660, 0.5052] | 0.2229 [0.1784, 0.2669] | 0.2583 [0.2107, 0.3083] | 2.0412 |
| 9 | 104 | 104 | 0.1850 [0.1433, 0.2224] | 0.5865 [0.4904, 0.6731] | 0.4135 [0.3173, 0.5000] | 0.2068 [0.1483, 0.2586] | 0.2512 [0.1838, 0.3142] | 1.9423 |
| 10+ | 98 | 53 | 0.1801 [0.1219, 0.2391] | 0.5714 [0.4600, 0.6604] | 0.3980 [0.2796, 0.5044] | 0.2071 [0.1281, 0.2893] | 0.2588 [0.1685, 0.3538] | 1.9184 |

## Por fraccion del ticket ya en el carrito

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (0%, 25%] | 2723 | 2313 | 0.2661 [0.2568, 0.2745] | 0.8347 [0.8194, 0.8490] | 0.6618 [0.6424, 0.6815] | 0.2332 [0.2233, 0.2428] | 0.1872 [0.1803, 0.1943] | 5.2549 |
| (25%, 50%] | 4239 | 2947 | 0.2166 [0.2091, 0.2246] | 0.7179 [0.7023, 0.7335] | 0.5360 [0.5181, 0.5529] | 0.2027 [0.1943, 0.2113] | 0.2011 [0.1931, 0.2093] | 3.5284 |
| (50%, 75%] | 3506 | 2727 | 0.1676 [0.1598, 0.1753] | 0.5693 [0.5506, 0.5888] | 0.3862 [0.3673, 0.4049] | 0.1773 [0.1673, 0.1872] | 0.2111 [0.1996, 0.2225] | 2.0924 |
| (75%, 100%] | 1932 | 1728 | 0.1224 [0.1128, 0.1322] | 0.3949 [0.3717, 0.4186] | 0.2464 [0.2265, 0.2682] | 0.1625 [0.1469, 0.1773] | 0.2254 [0.2072, 0.2451] | 1.1108 |

## Referencia: el corte de cabecera sobre las mismas cestas

Las mismas cestas, evaluadas con el corte de siempre (una query por cesta: la mitad con el
carrito vacio y la otra mitad con la mitad del ticket).

| grupo | n_queries | n_cestas | ndcg_graded@5 | cat_hit_rate@5 | sku_hit_rate@5 | ndcg@5 | recall@5 | n_target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cabecera: carrito vacio | 1485 | 1485 | 0.2406 [0.2304, 0.2508] | 0.7818 [0.7616, 0.8013] | 0.6121 [0.5892, 0.6357] | 0.2118 [0.2011, 0.2227] | 0.1710 [0.1626, 0.1798] | 5.1704 |
| cabecera: mitad del ticket | 1462 | 1462 | 0.1956 [0.1854, 0.2062] | 0.6648 [0.6388, 0.6895] | 0.4815 [0.4555, 0.5069] | 0.1928 [0.1804, 0.2050] | 0.2060 [0.1928, 0.2183] | 2.8618 |
| cabecera: las dos | 2947 | 2947 | 0.2183 [0.2106, 0.2257] | 0.7238 [0.7085, 0.7394] | 0.5473 [0.5287, 0.5646] | 0.2024 [0.1939, 0.2108] | 0.1884 [0.1805, 0.1961] | 4.0251 |

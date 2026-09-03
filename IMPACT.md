# Impacto de negocio estimado

De NDCG@5 y AUC a euros. Lo escribe `python -m src.impact.pipeline`, que lee las metricas
de [`reports/recommender/metrics.json`](reports/recommender/metrics.json) y
[`reports/nba/metrics.json`](reports/nba/metrics.json) y mide el resto sobre
`data/processed`. Ninguna cifra esta tecleada a mano; el detalle completo queda en
[`reports/impact/impact.json`](reports/impact/impact.json).

> **Lo que se mide y lo que se supone.** El `hit_rate@5` mide **relevancia**, no
> causalidad: un acierto significa que el producto estaba en la cesta de test, o sea que
> el cliente iba a comprarlo igualmente. Convertir eso en venta extra exige un supuesto
> — la **incrementalidad** — que este dataset no puede estimar, porque haria falta un A/B
> con el panel de recomendaciones apagado. Va declarado en `src/impact/config.py` y
> barrido entero mas abajo. Lo unico que no depende de ningun supuesto es la mejora
> relativa sobre el baseline sin aprendizaje.

## El negocio simulado, medido

| Magnitud | Valor |
| --- | ---: |
| Periodo | 24 meses |
| Cestas totales | 600.174 (25.007 al mes) |
| Cestas online (`app` + `web`) | 327.624 (13.651 al mes) |
| Clientes con al menos una compra | 19.225 |
| Lineas por cesta | 5,09 |
| Importe medio de una linea | 5,66 € |
| Margen bruto mezclado | 23,9 % |

El margen mezclado pondera el mapa por departamento de la Fase 4 (18 % en Frescos, 35 % en
Drogueria e Higiene) por lo que vende cada uno. Frescos es un tercio de la venta, asi que
tira del promedio hacia abajo.

## 1. El recomendador: cross-sell

El ranker acierta algo en el **11,8 %** de las cestas de test,
frente al **8,3 %** del baseline sin aprendizaje (popularidad
reciente x indice estacional). Son **3,53 puntos** mas: un
**42 % mas de cestas con una sugerencia relevante**. Esa
cifra no lleva ningun supuesto dentro.

Traducida a euros sobre una base de referencia de **100.000 cestas online al mes**,
con la incrementalidad al 10 %:

| Paso | Valor |
| --- | ---: |
| Cestas al mes con un acierto que el baseline no daba | 3.528 |
| x incrementalidad (10 %) = unidades extra al mes | 353 |
| x importe medio de linea = **venta extra al mes** | **1.996 €** |
| x margen bruto = **margen extra al mes** | **477 €** |
| **Margen extra al ano** | **5.721 €** |

Sobre el supermercado simulado tal cual (13.651 cestas
online al mes) son 65 € al mes.

### Barrido del supuesto

Por 100.000 cestas online al mes:

| Incrementalidad | Unidades extra/mes | Venta extra/mes | Margen/mes | Margen/ano |
| ---: | ---: | ---: | ---: | ---: |
| 2 % | 71 | 399 € | 95 € | 1.144 € |
| 5 % | 176 | 998 € | 238 € | 2.861 € |
| 10 % | 353 | 1.996 € | 477 € | 5.721 € |
| 20 % | 706 | 3.992 € | 954 € | 11.443 € |
| 30 % | 1.058 | 5.988 € | 1.430 € | 17.164 € |

La lectura honesta: **el cross-sell del recomendador vale miles de euros al ano por cada
100.000 cestas online al mes, no cientos de miles**. Y no es un defecto del calculo,
es el modelo: con un NDCG@5 de 0,0343 no da para mas. El
[README](README.md#por-qué-el-número-es-bajo-no-es-el-modelo-es-el-dato) explica por que
— el generador elige el SKU dentro de la categoria casi al azar — y por que el mismo
sistema acierta la **categoria** en el 51,4 % de las cestas. Con fidelidad de marca en el
dato, la misma arquitectura tendria bastante mas recorrido.

## 2. El Next Best Action

Aqui no hay que traducir nada: la politica de la Fase 4 ya decide en euros de valor
incremental esperado sobre no actuar. Sobre los 18.729 clientes del
corte de test, una oleada de campana vale **4.012 €**
(0,214 € por cliente), de los que
**3.082 € los aporta elegir a quien** y no la accion
en si: la mejor campana no segmentada se queda en
930 €.

Con una oleada al mes (12 al ano, que es lo coherente con el
horizonte de retencion de 4 semanas), por **100.000 clientes activos**:

| Escenario | Por oleada | Al ano |
| --- | ---: | ---: |
| Politica de valor esperado | 21.421 € | 257.056 € |
| Suelo: el cupon no retiene a nadie | 6.458 € | 77.499 € |
| Mejor alternativa trivial | 4.966 € | 59.587 € |

La fila que sostiene el caso es la segunda. El barrido de la Fase 4 muestra que **incluso
suponiendo que el cupon no retenga a nadie la politica sigue ganando**, y que en ese
escenario deja de repartir cupones por completo. Lo que depende del supuesto es el tamano
del premio, no el signo.

## 3. Las dos piezas juntas

Por 100.000 clientes activos y 100.000 cestas online al mes:

| Pieza | Al mes | Al ano |
| --- | ---: | ---: |
| Cross-sell del recomendador (margen) | 477 € | 5.721 € |
| Politica de Next Best Action | 21.421 € | 257.056 € |
| **Total** | **21.898 €** | **262.777 €** |

Las dos cifras no son homogeneas y conviene no sumarlas a la ligera: la del recomendador
es margen bruto sobre venta incremental y descansa en un supuesto de incrementalidad; la
del NBA es valor esperado neto de coste de campana y descansa en un supuesto de efecto de
la accion. Coinciden en la parte economica — el mismo mapa de margen por departamento — y
en que las dos son **conservadoras por construccion**: el recomendador se compara contra
un baseline que ya funciona, y la politica contra la mejor de las alternativas triviales,
no contra no hacer nada.

El reparto tambien dice donde esta hoy el proyecto: **casi todo el valor viene del NBA**,
no del recomendador. En parte tiene sentido — la politica decide sobre el cliente entero y
el recomendador solo sobre cinco huecos de una cesta — pero sobre todo es consecuencia del
techo de dato descrito arriba.

## Que haria falta para afinar esto

1. **Un A/B**, que es lo unico que convierte la incrementalidad de supuesto en estimacion.
2. **Fidelidad de marca en el generador**: es el techo del recomendador y esta anotado como
   deuda en el [`ROADMAP.md`](ROADMAP.md).
3. **Un target de churn condicionado a la cadencia de cada cliente**: hoy el modelo de la
   Fase 4 predice inactividad a 4 semanas, que con una cadencia media de 24 dias le pasa a
   media base sin ser abandono.

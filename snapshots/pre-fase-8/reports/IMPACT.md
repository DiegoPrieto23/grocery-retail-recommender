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
| Importe medio de una linea | 5,62 € |
| Margen bruto mezclado | 24,2 % |

El margen mezclado pondera el mapa por departamento de la Fase 4 (18 % en Frescos, 35 % en
Drogueria e Higiene) por lo que vende cada uno. Frescos es un tercio de la venta, asi que
tira del promedio hacia abajo.

## 1. El recomendador: cross-sell

El ranker acierta algo en el **54,2 %** de las cestas de test,
frente al **27,0 %** del baseline sin aprendizaje (popularidad
reciente x indice estacional). Son **27,15 puntos** mas: un
**101 % mas de cestas con una sugerencia relevante**. Esa
cifra no lleva ningun supuesto dentro.

Traducida a euros sobre una base de referencia de **100.000 cestas online al mes**,
con la incrementalidad al 10 %:

| Paso | Valor |
| --- | ---: |
| Cestas al mes con un acierto que el baseline no daba | 27.150 |
| x incrementalidad (10 %) = unidades extra al mes | 2.715 |
| x importe medio de linea = **venta extra al mes** | **15.265 €** |
| x margen bruto = **margen extra al mes** | **3.694 €** |
| **Margen extra al ano** | **44.327 €** |

Sobre el supermercado simulado tal cual (13.651 cestas
online al mes) son 504 € al mes.

### Barrido del supuesto

Por 100.000 cestas online al mes:

| Incrementalidad | Unidades extra/mes | Venta extra/mes | Margen/mes | Margen/ano |
| ---: | ---: | ---: | ---: | ---: |
| 2 % | 543 | 3.053 € | 739 € | 8.865 € |
| 5 % | 1.358 | 7.633 € | 1.847 € | 22.164 € |
| 10 % | 2.715 | 15.265 € | 3.694 € | 44.327 € |
| 20 % | 5.430 | 30.530 € | 7.388 € | 88.655 € |
| 30 % | 8.145 | 45.795 € | 11.082 € | 132.982 € |

La lectura honesta: **el cross-sell del recomendador vale 44.327 € al ano por cada 100.000 cestas online al mes**, frente a 252.286 € del NBA por cada 100.000 clientes.

Antes de la Fase 7 eran **5.721 €**, con el ranker acertando en el
11,8 % de las cestas. El codigo es el mismo; lo que cambio es el dato. El
generador original elegia la referencia dentro de la categoria casi al azar entre ~24, asi
que ningun modelo podia acertar el SKU; con fidelidad de marca y 8 referencias por
categoria el mismo sistema acierta en el 54,2 %
([`ROADMAP.md`](ROADMAP.md), Fase 7). Parte de la subida es el catalogo mas pequeno
— tambien el baseline acierta mas —, y por eso la fila que cuenta es la de cestas con un
acierto **que el baseline no daba**. Las cifras de antes estan congeladas en
[`reports/impact/baseline_fase5.json`](reports/impact/baseline_fase5.json).

## 2. El Next Best Action

Aqui no hay que traducir nada: la politica de la Fase 4 ya decide en euros de valor
incremental esperado sobre no actuar. Sobre los 18.729 clientes del
corte de test, una oleada de campana vale **3.938 €**
(0,210 € por cliente), de los que
**3.030 € los aporta elegir a quien** y no la accion
en si: la mejor campana no segmentada se queda en
907 €.

Con una oleada al mes (12 al ano, que es lo coherente con el
horizonte de retencion de 4 semanas), por **100.000 clientes activos**:

| Escenario | Por oleada | Al ano |
| --- | ---: | ---: |
| Politica de valor esperado | 21.024 € | 252.286 € |
| Suelo: el cupon no retiene a nadie | 6.346 € | 76.158 € |
| Mejor alternativa trivial | 4.843 € | 58.118 € |

La fila que sostiene el caso es la segunda. El barrido de la Fase 4 muestra que **incluso
suponiendo que el cupon no retenga a nadie la politica sigue ganando**, y que en ese
escenario deja de repartir cupones por completo. Lo que depende del supuesto es el tamano
del premio, no el signo.

## 3. Las dos piezas juntas

Por 100.000 clientes activos y 100.000 cestas online al mes:

| Pieza | Al mes | Al ano |
| --- | ---: | ---: |
| Cross-sell del recomendador (margen) | 3.694 € | 44.327 € |
| Politica de Next Best Action | 21.024 € | 252.286 € |
| **Total** | **24.718 €** | **296.614 €** |

Las dos cifras no son homogeneas y conviene no sumarlas a la ligera: la del recomendador
es margen bruto sobre venta incremental y descansa en un supuesto de incrementalidad; la
del NBA es valor esperado neto de coste de campana y descansa en un supuesto de efecto de
la accion. Coinciden en la parte economica — el mismo mapa de margen por departamento — y
en que las dos son **conservadoras por construccion**: el recomendador se compara contra
un baseline que ya funciona, y la politica contra la mejor de las alternativas triviales,
no contra no hacer nada.

El reparto tambien dice donde esta hoy el proyecto: el NBA pone el **85 %** del total y el recomendador el 15 %. En parte tiene sentido — la politica decide sobre el cliente entero y el recomendador solo sobre cinco huecos de una cesta — y en parte es historia: antes de la Fase 7 el recomendador ponia solo el
2 %, porque el dato no le dejaba acertar la referencia.

## Que haria falta para afinar esto

1. **Un A/B**, que es lo unico que convierte la incrementalidad de supuesto en estimacion.
2. **Features de calendario en el modelo de propension**: sin ellas la politica puede
   empujar una categoria de temporada fuera de temporada (ver el informe de hallazgos).
3. **Un target de churn condicionado a la cadencia de cada cliente**: hoy el modelo de la
   Fase 4 predice inactividad a 4 semanas, que con una cadencia media de 24 dias le pasa a
   media base sin ser abandono.

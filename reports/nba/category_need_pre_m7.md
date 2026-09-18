# Capa comun de necesidad de categoria (punto M7)

> Lo recalcula `python -m src.nba.verify_category_need`. Ninguna cifra de este
> fichero esta escrita a mano.

Corte de test: **2025-11-01** · horizonte de la etiqueta: 7 dias · pares con compra en los ultimos 180 dias: **426.285**

## 1. El ciclo de reposicion predice tambien la etiqueta del NBA

`overdue_ratio` es la senal con la que el recomendador decide si una categoria
toca. La etiqueta del NBA es otra cosa -- comprar la categoria en los 7 dias
siguientes --, y aun asi la ordena: por eso la capa se puede compartir.

| banda de `overdue_ratio` | pares | tasa real de compra a 7 d |
| --- | ---: | ---: |
| [0.0, 0.25) | 98.591 | 4,6% |
| [0.25, 0.5) | 80.156 | 7,3% |
| [0.5, 0.75) | 59.281 | 8,6% |
| [0.75, 1.0) | 42.670 | 9,6% |
| [1.0, 1.5) | 53.948 | 9,7% |
| [1.5, 2.0) | 29.362 | 9,4% |
| [2.0, 3.0) | 28.334 | 8,9% |
| [3.0, inf) | 33.943 | 6,0% |

La tasa sube del 4,6% en la primera
banda al 9,7% en `[1.0, 1.5)` y vuelve a
bajar al 6,0% en la cola. Por eso
`category_need_weight` se satura en 1 en vez de seguir creciendo: pasada la
banda del ciclo, un ratio alto es una categoria abandonada, no una necesidad.

## 2. A que categoria apunta cada accion

| accion | n | `overdue` mediano | % no vencida | % recien repuesta | tasa real 7 d |
| --- | ---: | ---: | ---: | ---: | ---: |
| `enviar_cupon_categoria` | 10.943 | 0.24 | 88,5% | 73,3% | 1,5% |
| `recomendar_producto` | 3.037 | 1.30 | 38,1% | 15,8% | 5,6% |

Referencia del pool completo de candidatos: el 65,8%
de los pares no esta vencido y el 41,9% esta recien
repuesto, con una tasa real del 7,5%. Una accion que
apunte por encima de esas cifras esta eligiendo peor que el azar del pool.

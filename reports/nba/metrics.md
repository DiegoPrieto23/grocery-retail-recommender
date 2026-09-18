# Next Best Action (Fase 4, Tarea 3b)

Generado por `python -m src.nba.pipeline`. Ninguna cifra de este informe se copia a mano:
se recalcula ejecutando ese comando.

## Montaje

| | |
| --- | --- |
| Cortes de entrenamiento | 2025-04-01, 2025-05-15, 2025-07-01, 2025-08-15 |
| Corte de validacion | 2025-09-15 |
| Corte de test | 2025-11-01 |
| Horizonte de compra en categoria | 7 dias |
| Horizonte de churn | 28 dias |
| Filas de entrenamiento (churn) | 34,557 |
| Filas de entrenamiento (categoria) | 838,369 |
| Clientes en test | 18,729 |
| Pares cliente-categoria en test | 426,285 |
| Clientes sin ninguna categoria accionable | 3,141 |

Los clientes sin categoria accionable son los que no han comprado **nada** en los
180 dias previos al corte. No generan pares, asi que ninguna
politica puede actuar sobre ellos; entran igualmente en la tabla con `ninguna_accion` para
que el denominador sea el maestro completo y no el subconjunto comodo.

El split es **temporal**: los cortes de entrenamiento son anteriores al de validacion y
ninguno de los dos alcanza la ventana de test. Las features de cada corte se calculan solo
con compras anteriores a el.

## Modelos de propension

| Modelo | n | tasa base | AUC | PR-AUC | lift decil 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Churn a 4 semanas | 18,729 | 0.4759 | 0.8532 | 0.8576 | 2.08x |
| Compra en categoria a 7 dias | 426,285 | 0.0755 | 0.7591 | 0.2479 | 3.43x |

El PR-AUC se lee contra la tasa base, no contra 0,5: un modelo aleatorio da exactamente la
tasa base. El lift del primer decil es la lectura de negocio: cuantas veces mas positivos
hay en el 10 % mejor puntuado que en la poblacion.

**Como leer el modelo de churn.** Su tasa base es alta porque en este dataset la cadencia
media de visita ronda los 24 dias: no comprar en 4 semanas le pasa a mucha gente y no
siempre significa abandono. Las features que mas pesan lo confirman -- `n_baskets_90d`,
`avg_days_between_baskets`, `recency_days` --, asi que buena parte de lo que el modelo
acierta es **frecuencia de compra**, no intencion de irse. El AUC es real, pero llamarlo
"modelo de churn" concede mas de lo que hace: es un modelo de inactividad a 4 semanas. La
senal de abandono de verdad (la rampa de decaimiento de `DATA_SPEC.md`) esta en los
`trend_*`, que aportan bastante menos.

### Comprobacion de cordura del target de churn

El churn se construye por corte (no compra en los 28 dias siguientes,
es decir 4 semanas), no se toma de `customers.churn_label`, que esta definido
respecto al final del
dataset y seria fuga. En el corte de test las dos ventanas casi coinciden, asi que deben
parecerse: **coinciden en el 83.4%** de los 18,729
clientes (47.6% de churn observado frente a
31.5% de la etiqueta de `DATA_SPEC.md`).

## La politica frente a las alternativas triviales

Valor **incremental** sobre no hacer nada, en euros de margen esperado. `ninguna_accion`
vale 0 por construccion, asi que una accion solo suma si su efecto paga su coste.

| politica | n_clientes | n_acciones | pct_accion | valor_total | valor_por_cliente | uplift_vs_no_actuar |
| --- | --- | --- | --- | --- | --- | --- |
| no actuar siempre | 18729 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| actuar siempre: recomendar_categoria | 18729 | 15588 | 0.8323 | 669.8880 | 0.0358 | 669.8880 |
| actuar siempre: enviar_cupon_categoria | 18729 | 15588 | 0.8323 | -5032.1985 | -0.2687 | -5032.1985 |
| politica de valor esperado | 18729 | 13896 | 0.7420 | 3077.8120 | 0.1643 | 3077.8120 |

La politica actua sobre el **74.2%** de los clientes, no sobre
todos, y ahi esta su ventaja: mandar el cupon a todo el mundo cuesta 8110 EUR
de valor esperado frente a elegir a quien.

### Reparto de acciones

| action | cuota |
| --- | --- |
| enviar_cupon_categoria | 0.4725 |
| recomendar_categoria | 0.2695 |
| ninguna_accion | 0.2580 |

## Sensibilidad al supuesto de uplift

`P(conversion | accion)` **no es estimable con este dataset**: el generador aplica su
`PROMO_UPLIFT` al reparto de cuota dentro de una categoria, nunca a la probabilidad de
comprarla ni a la de volver, asi que el tratamiento nunca varia. El uplift del cupon es un
supuesto declarado en `config.py`. Esta tabla dice cuanto depende la conclusion de el.

### Uplift de conversion del cupon

| uplift_cupon | valor_politica | valor_cupon_a_todos | pct_accion | pct_cupon |
| --- | --- | --- | --- | --- |
| 1.0000 | 2973.9783 | -3402.6969 | 0.7420 | 0.4457 |
| 1.1000 | 2969.6057 | -3868.6832 | 0.7420 | 0.4459 |
| 1.2000 | 2991.7032 | -4334.6695 | 0.7420 | 0.4534 |
| 1.3500 | 3077.8120 | -5032.1985 | 0.7420 | 0.4725 |
| 1.5000 | 3228.1314 | -5721.0455 | 0.7420 | 0.4986 |
| 1.7500 | 3581.2174 | -6836.9970 | 0.7420 | 0.5376 |
| 2.0000 | 4002.7889 | -7919.4264 | 0.7421 | 0.5658 |

### Reduccion de churn del cupon

Este es el barrido que de verdad importa, y conviene decir por que. Con el cupon anclado a
su valor real (2,54 EUR) y un margen esperado por categoria de alrededor de 1,4 EUR, el
descuento **es mayor que el margen que persigue**: por el lado del cross-sell el cupon
destruye valor haga lo que haga la conversion. Se ve en la tabla de arriba, donde pasar el
uplift de 1,00 a 2,00 apenas mueve el total. Todo lo que aporta el cupon viene del termino
de retencion, asi que es `churn_reduction` lo que hay que auditar.

| reduccion_churn | valor_politica | valor_cupon_a_todos | pct_accion | pct_cupon |
| --- | --- | --- | --- | --- |
| 0.0000 | 1157.9242 | -8463.6140 | 0.7420 | 0.0001 |
| 0.0200 | 1158.4048 | -7777.3309 | 0.7420 | 0.0011 |
| 0.0500 | 1454.4933 | -6747.9062 | 0.7420 | 0.1805 |
| 0.1000 | 3077.8120 | -5032.1985 | 0.7420 | 0.4725 |
| 0.1500 | 5155.5759 | -3316.4908 | 0.7420 | 0.5918 |
| 0.2000 | 7359.0614 | -1600.7830 | 0.7420 | 0.6429 |

La fila de `reduccion_churn = 0` es la lectura pesimista: lo que queda cuando se supone que
el cupon no retiene a nadie.

Lo que **no** es un supuesto es el reparto: con los efectos fijados, toda la diferencia
entre la politica y "actuar siempre" viene de acertar a quien, y eso es merito del modelo.

## Frente al dataset anterior a la Fase 8

La Fase 8 regenera el dataset con misiones de compra, cestas con cola larga, sustitucion entre categorias y propension a la marca blanca. Sus cifras estan congeladas en `reports/nba/baseline_pre_fase8.json`. Mismo codigo, mismos cortes y mismos supuestos; cambia el dato.

| Metrica | Fase 7d (dataset viejo) | Fase 8 (dataset nuevo) |
| --- | ---: | ---: |
| Churn 4 semanas: base_rate | 0.4759 | 0.4759 |
| Churn 4 semanas: auc | 0.8527 | 0.8532 |
| Churn 4 semanas: pr_auc | 0.8550 | 0.8576 |
| Churn 4 semanas: lift decil 1 | 2.07x | 2.08x |
| Compra categoria: base_rate | 0.0619 | 0.0755 |
| Compra categoria: auc | 0.7630 | 0.7591 |
| Compra categoria: pr_auc | 0.2185 | 0.2479 |
| Compra categoria: lift decil 1 | 3.50x | 3.43x |
| Politica: valor frente a no actuar | 3,938 EUR | 3,078 EUR |
| Politica: ventaja sobre la mejor trivial | 3,030 EUR | 2,408 EUR |
| Politica: clientes con accion | 75.2% | 74.2% |

### Barrido de `churn_reduction` (Fase 7d frente a Fase 8)

| reduccion_churn | valor Fase 7d | valor Fase 8 | cupones Fase 7d | cupones Fase 8 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 1,189 EUR | 1,158 EUR | 0.0% | 0.0% |
| 0.02 | 1,189 EUR | 1,158 EUR | 0.1% | 0.1% |
| 0.05 | 1,706 EUR | 1,454 EUR | 43.7% | 18.0% |
| 0.10 | 3,938 EUR | 3,078 EUR | 66.2% | 47.2% |
| 0.15 | 6,291 EUR | 5,156 EUR | 70.3% | 59.2% |
| 0.20 | 8,662 EUR | 7,359 EUR | 72.0% | 64.3% |

## Frente a la Fase 4 original

La Fase 4 se entreno sobre el dataset anterior a la Fase 7a (1.500 productos, sin fidelidad de marca). Sus cifras estan congeladas en `reports/nba/baseline_fase4.json`. Mismo codigo, mismos cortes y mismos supuestos; cambia el dato.

| Metrica | Fase 4 (dataset viejo) | Fase 7d (dataset nuevo) |
| --- | ---: | ---: |
| Churn 4 semanas: base_rate | 0.4759 | 0.4759 |
| Churn 4 semanas: auc | 0.8531 | 0.8532 |
| Churn 4 semanas: pr_auc | 0.8556 | 0.8576 |
| Churn 4 semanas: lift decil 1 | 2.07x | 2.08x |
| Compra categoria: base_rate | 0.0619 | 0.0755 |
| Compra categoria: auc | 0.7634 | 0.7591 |
| Compra categoria: pr_auc | 0.2190 | 0.2479 |
| Compra categoria: lift decil 1 | 3.51x | 3.43x |
| Politica: valor frente a no actuar | 4,012 EUR | 3,078 EUR |
| Politica: ventaja sobre la mejor trivial | 3,082 EUR | 2,408 EUR |
| Politica: clientes con accion | 75.6% | 74.2% |

### Barrido de `churn_reduction` (Fase 4 frente a Fase 7d)

| reduccion_churn | valor Fase 4 | valor Fase 7d | cupones Fase 4 | cupones Fase 7d |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 1,210 EUR | 1,158 EUR | 0.0% | 0.0% |
| 0.02 | 1,210 EUR | 1,158 EUR | 0.1% | 0.1% |
| 0.05 | 1,734 EUR | 1,454 EUR | 41.7% | 18.0% |
| 0.10 | 4,012 EUR | 3,078 EUR | 64.8% | 47.2% |
| 0.15 | 6,427 EUR | 5,156 EUR | 69.7% | 59.2% |
| 0.20 | 8,862 EUR | 7,359 EUR | 71.6% | 64.3% |

## Que features usan los modelos

### Churn a 4 semanas

| feature | gain | split |
| --- | --- | --- |
| n_baskets_90d | 38118.7888 | 72 |
| recency_days | 37462.5109 | 298 |
| avg_days_between_baskets | 31897.7413 | 432 |
| spend_90d | 5612.0461 | 150 |
| frequency | 2713.8896 | 118 |
| spend_28d | 1732.3097 | 171 |
| signup_tenure_days | 1578.1570 | 236 |
| tenure_days | 1208.3888 | 164 |
| spend_prev_28d | 951.7907 | 145 |
| avg_ticket | 890.1452 | 154 |
| monetary | 871.0402 | 133 |
| promo_line_share | 715.2525 | 118 |
| n_products | 668.1434 | 96 |
| n_categories | 540.9727 | 84 |
| loyalty_tier_idx | 537.8151 | 81 |
| trend_ticket | 418.3817 | 68 |
| n_baskets_28d | 386.2318 | 30 |
| trend_spend | 314.3461 | 53 |
| preferred_channel_idx | 299.0457 | 48 |
| household_size_est | 172.2345 | 33 |

### Compra en categoria

| feature | gain | split |
| --- | --- | --- |
| cat_n_purchase_days | 159802.7508 | 87 |
| cat_observed_repurchase_days | 144195.4383 | 98 |
| recency_days | 42709.4692 | 449 |
| spend_28d | 37894.2961 | 459 |
| signup_tenure_days | 32183.1312 | 376 |
| cat_overdue_ratio | 32118.8448 | 272 |
| cat_typical_repurchase_days | 30830.1005 | 313 |
| cat_expected_repurchase_days | 30073.8246 | 53 |
| avg_days_between_baskets | 25794.7490 | 309 |
| avg_ticket | 20301.1109 | 309 |
| promo_line_share | 18475.2296 | 281 |
| spend_90d | 17882.7207 | 256 |
| tenure_days | 17011.6424 | 255 |
| spend_prev_28d | 14806.3770 | 215 |
| n_products | 14528.9174 | 203 |
| trend_ticket | 13745.4641 | 206 |
| n_baskets_28d | 12685.3720 | 112 |
| trend_spend | 12676.8976 | 191 |
| monetary | 11445.4021 | 164 |
| frequency | 10748.5110 | 160 |

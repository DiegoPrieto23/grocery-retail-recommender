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
| Filas de entrenamiento (categoria) | 726,764 |
| Clientes en test | 18,729 |
| Pares cliente-categoria en test | 370,300 |
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
| Churn a 4 semanas | 18,729 | 0.4759 | 0.8527 | 0.8550 | 2.07x |
| Compra en categoria a 7 dias | 370,300 | 0.0619 | 0.7630 | 0.2185 | 3.50x |

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
| actuar siempre: recomendar_producto | 18729 | 15588 | 0.8323 | 907.0725 | 0.0484 | 907.0725 |
| actuar siempre: enviar_cupon_categoria | 18729 | 15588 | 0.8323 | -1591.4925 | -0.0850 | -1591.4925 |
| politica de valor esperado | 18729 | 14090 | 0.7523 | 3937.5604 | 0.2102 | 3937.5604 |

La politica actua sobre el **75.2%** de los clientes, no sobre
todos, y ahi esta su ventaja: mandar el cupon a todo el mundo cuesta 5529 EUR
de valor esperado frente a elegir a quien.

### Reparto de acciones

| action | cuota |
| --- | --- |
| enviar_cupon_categoria | 0.6621 |
| ninguna_accion | 0.2477 |
| recomendar_producto | 0.0902 |

## Sensibilidad al supuesto de uplift

`P(conversion | accion)` **no es estimable con este dataset**: el generador aplica su
`PROMO_UPLIFT` al reparto de cuota dentro de una categoria, nunca a la probabilidad de
comprarla ni a la de volver, asi que el tratamiento nunca varia. El uplift del cupon es un
supuesto declarado en `config.py`. Esta tabla dice cuanto depende la conclusion de el.

### Uplift de conversion del cupon

| uplift_cupon | valor_politica | valor_cupon_a_todos | pct_accion | pct_cupon |
| --- | --- | --- | --- | --- |
| 1.0000 | 3865.6453 | -430.7736 | 0.7523 | 0.6586 |
| 1.1000 | 3866.5078 | -762.6142 | 0.7523 | 0.6585 |
| 1.2000 | 3876.7224 | -1094.4548 | 0.7523 | 0.6597 |
| 1.3500 | 3937.5604 | -1591.4925 | 0.7523 | 0.6621 |
| 1.5000 | 4067.3511 | -2084.2251 | 0.7523 | 0.6660 |
| 1.7500 | 4382.5846 | -2890.9899 | 0.7523 | 0.6749 |
| 2.0000 | 4760.5521 | -3675.9827 | 0.7524 | 0.6848 |

### Reduccion de churn del cupon

Este es el barrido que de verdad importa, y conviene decir por que. Con el cupon anclado a
su valor real (2,54 EUR) y un margen esperado por categoria de alrededor de 1,4 EUR, el
descuento **es mayor que el margen que persigue**: por el lado del cross-sell el cupon
destruye valor haga lo que haga la conversion. Se ve en la tabla de arriba, donde pasar el
uplift de 1,00 a 2,00 apenas mueve el total. Todo lo que aporta el cupon viene del termino
de retencion, asi que es `churn_reduction` lo que hay que auditar.

| reduccion_churn | valor_politica | valor_cupon_a_todos | pct_accion | pct_cupon |
| --- | --- | --- | --- | --- |
| 0.0000 | 1188.6342 | -6357.0116 | 0.7523 | 0.0000 |
| 0.0200 | 1188.9809 | -5403.9078 | 0.7523 | 0.0010 |
| 0.0500 | 1706.1675 | -3974.2521 | 0.7523 | 0.4367 |
| 0.1000 | 3937.5604 | -1591.4925 | 0.7523 | 0.6621 |
| 0.1500 | 6290.7683 | 791.2670 | 0.7523 | 0.7030 |
| 0.2000 | 8661.5814 | 3174.0265 | 0.7528 | 0.7205 |

La fila de `reduccion_churn = 0` es la lectura pesimista: lo que queda cuando se supone que
el cupon no retiene a nadie.

Lo que **no** es un supuesto es el reparto: con los efectos fijados, toda la diferencia
entre la politica y "actuar siempre" viene de acertar a quien, y eso es merito del modelo.

## Frente a la Fase 4 original

La Fase 4 se entreno sobre el dataset anterior a la Fase 7a (1.500 productos, sin fidelidad de marca). Sus cifras estan congeladas en `reports/nba/baseline_fase4.json`. Mismo codigo, mismos cortes y mismos supuestos; cambia el dato.

| Metrica | Fase 4 (dataset viejo) | Fase 7d (dataset nuevo) |
| --- | ---: | ---: |
| Churn 4 semanas: base_rate | 0.4759 | 0.4759 |
| Churn 4 semanas: auc | 0.8531 | 0.8527 |
| Churn 4 semanas: pr_auc | 0.8556 | 0.8550 |
| Churn 4 semanas: lift decil 1 | 2.07x | 2.07x |
| Compra categoria: base_rate | 0.0619 | 0.0619 |
| Compra categoria: auc | 0.7634 | 0.7630 |
| Compra categoria: pr_auc | 0.2190 | 0.2185 |
| Compra categoria: lift decil 1 | 3.51x | 3.50x |
| Politica: valor frente a no actuar | 4,012 EUR | 3,938 EUR |
| Politica: ventaja sobre la mejor trivial | 3,082 EUR | 3,030 EUR |
| Politica: clientes con accion | 75.6% | 75.2% |

### Barrido de `churn_reduction`

| reduccion_churn | valor Fase 4 | valor Fase 7d | cupones Fase 4 | cupones Fase 7d |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 1,210 EUR | 1,189 EUR | 0.0% | 0.0% |
| 0.02 | 1,210 EUR | 1,189 EUR | 0.1% | 0.1% |
| 0.05 | 1,734 EUR | 1,706 EUR | 41.7% | 43.7% |
| 0.10 | 4,012 EUR | 3,938 EUR | 64.8% | 66.2% |
| 0.15 | 6,427 EUR | 6,291 EUR | 69.7% | 70.3% |
| 0.20 | 8,862 EUR | 8,662 EUR | 71.6% | 72.0% |

## Que features usan los modelos

### Churn a 4 semanas

| feature | gain | split |
| --- | --- | --- |
| n_baskets_90d | 25262.5355 | 24 |
| avg_days_between_baskets | 23062.2003 | 179 |
| recency_days | 20800.3455 | 123 |
| spend_90d | 15762.5590 | 60 |
| frequency | 1194.6965 | 36 |
| spend_28d | 979.9661 | 65 |
| n_products | 735.9670 | 47 |
| signup_tenure_days | 673.0201 | 83 |
| tenure_days | 623.1358 | 76 |
| monetary | 508.3049 | 44 |
| avg_ticket | 416.0513 | 73 |
| loyalty_tier_idx | 353.8592 | 48 |
| promo_line_share | 337.7837 | 59 |
| n_categories | 255.6725 | 29 |
| spend_prev_28d | 237.9817 | 34 |
| trend_ticket | 169.9552 | 24 |
| n_baskets_28d | 133.7066 | 6 |
| household_size_est | 73.3022 | 14 |
| preferred_channel_idx | 62.5417 | 12 |
| trend_baskets | 56.5217 | 8 |

### Compra en categoria

| feature | gain | split |
| --- | --- | --- |
| cat_observed_repurchase_days | 128669.2666 | 133 |
| cat_n_purchase_days | 55276.5105 | 96 |
| cat_overdue_ratio | 44827.7279 | 370 |
| cat_typical_repurchase_days | 33363.4157 | 438 |
| spend_28d | 27335.0061 | 359 |
| recency_days | 21537.3424 | 378 |
| avg_days_between_baskets | 17021.0747 | 298 |
| spend_90d | 14522.8082 | 266 |
| cat_days_since_last_purchase | 14029.4468 | 199 |
| signup_tenure_days | 12919.4988 | 350 |
| cat_expected_repurchase_days | 12050.3826 | 56 |
| tenure_days | 10742.7740 | 327 |
| avg_ticket | 8846.3748 | 296 |
| promo_line_share | 8164.3869 | 259 |
| spend_prev_28d | 7851.8499 | 227 |
| trend_ticket | 6611.6503 | 205 |
| n_baskets_28d | 6084.4606 | 77 |
| trend_spend | 5933.9323 | 183 |
| n_baskets_90d | 5905.8074 | 145 |
| n_products | 5770.4150 | 181 |

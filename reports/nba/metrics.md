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
| Churn a 4 semanas | 18,729 | 0.4759 | 0.8531 | 0.8556 | 2.07x |
| Compra en categoria a 7 dias | 370,300 | 0.0619 | 0.7634 | 0.2190 | 3.51x |

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
| actuar siempre: recomendar_producto | 18729 | 15588 | 0.8323 | 930.0018 | 0.0497 | 930.0018 |
| actuar siempre: enviar_cupon_categoria | 18729 | 15588 | 0.8323 | -1430.3508 | -0.0764 | -1430.3508 |
| politica de valor esperado | 18729 | 14161 | 0.7561 | 4011.9997 | 0.2142 | 4011.9997 |

La politica actua sobre el **75.6%** de los clientes, no sobre
todos, y ahi esta su ventaja: mandar el cupon a todo el mundo cuesta 5442 EUR
de valor esperado frente a elegir a quien.

### Reparto de acciones

| action | cuota |
| --- | --- |
| enviar_cupon_categoria | 0.6476 |
| ninguna_accion | 0.2439 |
| recomendar_producto | 0.1085 |

## Sensibilidad al supuesto de uplift

`P(conversion | accion)` **no es estimable con este dataset**: el generador aplica su
`PROMO_UPLIFT` al reparto de cuota dentro de una categoria, nunca a la probabilidad de
comprarla ni a la de volver, asi que el tratamiento nunca varia. El uplift del cupon es un
supuesto declarado en `config.py`. Esta tabla dice cuanto depende la conclusion de el.

### Uplift de conversion del cupon

| uplift_cupon | valor_politica | valor_cupon_a_todos | pct_accion | pct_cupon |
| --- | --- | --- | --- | --- |
| 1.0000 | 3938.0634 | -265.0322 | 0.7561 | 0.6445 |
| 1.1000 | 3939.2672 | -597.9804 | 0.7561 | 0.6443 |
| 1.2000 | 3952.6050 | -930.9286 | 0.7561 | 0.6451 |
| 1.3500 | 4011.9997 | -1430.3508 | 0.7561 | 0.6476 |
| 1.5000 | 4123.9311 | -1929.0381 | 0.7561 | 0.6524 |
| 1.7500 | 4393.7165 | -2748.1575 | 0.7561 | 0.6621 |
| 2.0000 | 4721.1899 | -3544.8422 | 0.7561 | 0.6715 |

### Reduccion de churn del cupon

Este es el barrido que de verdad importa, y conviene decir por que. Con el cupon anclado a
su valor real (2,54 EUR) y un margen esperado por categoria de alrededor de 1,4 EUR, el
descuento **es mayor que el margen que persigue**: por el lado del cross-sell el cupon
destruye valor haga lo que haga la conversion. Se ve en la tabla de arriba, donde pasar el
uplift de 1,00 a 2,00 apenas mueve el total. Todo lo que aporta el cupon viene del termino
de retencion, asi que es `churn_reduction` lo que hay que auditar.

| reduccion_churn | valor_politica | valor_cupon_a_todos | pct_accion | pct_cupon |
| --- | --- | --- | --- | --- |
| 0.0000 | 1209.5732 | -6331.2402 | 0.7561 | 0.0000 |
| 0.0200 | 1210.0794 | -5351.0623 | 0.7561 | 0.0014 |
| 0.0500 | 1734.0268 | -3880.7955 | 0.7561 | 0.4172 |
| 0.1000 | 4011.9997 | -1430.3508 | 0.7561 | 0.6476 |
| 0.1500 | 6426.7205 | 1020.0939 | 0.7561 | 0.6968 |
| 0.2000 | 8862.4818 | 3470.5385 | 0.7563 | 0.7165 |

La fila de `reduccion_churn = 0` es la lectura pesimista: lo que queda cuando se supone que
el cupon no retiene a nadie.

Lo que **no** es un supuesto es el reparto: con los efectos fijados, toda la diferencia
entre la politica y "actuar siempre" viene de acertar a quien, y eso es merito del modelo.

## Que features usan los modelos

### Churn a 4 semanas

| feature | gain | split |
| --- | --- | --- |
| n_baskets_90d | 25678.7167 | 29 |
| avg_days_between_baskets | 21136.9012 | 169 |
| recency_days | 16488.2140 | 99 |
| spend_90d | 16406.3567 | 65 |
| frequency | 1162.7533 | 33 |
| spend_28d | 990.3241 | 56 |
| monetary | 641.7852 | 47 |
| signup_tenure_days | 612.8706 | 73 |
| n_products | 551.7295 | 42 |
| tenure_days | 495.5596 | 56 |
| n_categories | 391.7476 | 33 |
| loyalty_tier_idx | 346.9111 | 44 |
| avg_ticket | 296.1929 | 49 |
| promo_line_share | 210.0556 | 34 |
| trend_ticket | 181.8403 | 27 |
| spend_prev_28d | 141.3312 | 22 |
| n_baskets_28d | 81.1674 | 10 |
| trend_spend | 72.7569 | 11 |
| preferred_channel_idx | 58.6549 | 11 |
| household_size_est | 51.1486 | 10 |

### Compra en categoria

| feature | gain | split |
| --- | --- | --- |
| cat_observed_repurchase_days | 125914.7185 | 112 |
| cat_n_purchase_days | 57027.3817 | 90 |
| cat_overdue_ratio | 44195.4756 | 346 |
| cat_typical_repurchase_days | 30387.1128 | 362 |
| spend_28d | 25069.1335 | 266 |
| recency_days | 18996.6407 | 305 |
| avg_days_between_baskets | 15189.4887 | 215 |
| cat_days_since_last_purchase | 13217.6838 | 172 |
| cat_expected_repurchase_days | 12748.1240 | 59 |
| spend_90d | 11252.2458 | 171 |
| signup_tenure_days | 8878.5158 | 215 |
| tenure_days | 7634.4687 | 200 |
| promo_line_share | 5710.1838 | 169 |
| n_baskets_28d | 5646.9308 | 62 |
| spend_prev_28d | 5241.7140 | 135 |
| avg_ticket | 5201.4500 | 155 |
| trend_ticket | 4877.8924 | 133 |
| monetary | 3909.9877 | 105 |
| n_products | 3906.5636 | 107 |
| n_baskets_90d | 3573.4592 | 68 |

# Decisiones de limpieza (Fase 2)

Qué se corrige de las 7 tablas crudas, con qué regla y **por qué esa y no otra**. El código
está en `src/etl/cleaning.py`; las cifras las regenera `python -m src.etl.run_etl` en
`reports/etl/cleaning_report.md`. Este documento explica el criterio, no repite los números.

Los defectos no son accidentales: `DATA_SPEC.md` los inyecta a propósito en la Fase 1 para
que haya algo real que limpiar. Eso permite comparar lo corregido con lo inyectado y saber
si la limpieza acierta, algo que con dato real nunca se puede.

---

## Principios

1. **Un nulo esperado no es un defecto.** `baskets.customer_id` nulo es una compra anónima,
   `baskets.store_id` nulo es una compra online y `basket_items.promotion_id` nulo es una
   línea sin promoción. Los tres son legítimos y se conservan. Descartar las anónimas
   (3 % de los tickets, casi todas de tienda física) sesgaría a la baja el canal `store`.
2. **Corregir antes que descartar.** Se tira una fila sólo cuando no aporta nada
   recuperable (una línea con 0 unidades). Si el valor correcto se puede deducir del propio
   dato, se deduce.
3. **El defecto no se borra, se etiqueta.** Cada corrección deja una columna testigo
   (`city_is_missing`, `signup_date_corrected`, `total_amount_raw`...). Así el Data Trust
   Score del dato limpio sube porque el dato es *usable*, no porque se haya perdido la
   historia de lo que pasaba.
4. **El ETL no mira al generador.** Ninguna regla importa el catálogo de la Fase 1: la
   grafía canónica de una categoría se deduce de los propios datos. Si el ETL usara el
   catálogo, "limpiar" sería copiar la respuesta.

---

## Tabla por tabla

### `products` — categorías inconsistentes y marcas sin informar

**Problema.** El 4 % de los productos escribe su categoría con otra grafía: `LECHE`,
`leche`, `Fruta ` (con espacios sobrantes), `Salsa  de  tomate` (con espacio doble
interno). 82 grafías distintas para 62 categorías reales (eran 116 con el catálogo de
1.500 productos anterior a la Fase 7a).

**Regla.** Se agrupan las grafías por su clave normalizada
(`lower(trim(colapsar espacios))`) y cada grupo se unifica a **la grafía mayoritaria**.

**Por qué esa.** Las alternativas fallan:

- *Recapitalizar con `initcap`* destrozaría `Torrijas y bolleria de Cuaresma` →
  `Torrijas Y Bolleria De Cuaresma`.
- *Tomar la lista del generador* acoplaría el ETL a la Fase 1 y haría trampa: en un caso
  real no existe esa lista.
- *Quedarse con la clave normalizada* (`leche`) daría un resultado limpio pero feo en un
  informe o un dashboard.

La mayoría funciona porque el ruido es minoritario por construcción (4 %): la grafía buena
siempre gana. **Comprobado:** las 62 grafías elegidas coinciden una a una con el catálogo
del generador, sin haberlo consultado.

**Empates.** Cuando dos grafías empatan en frecuencia, el orden alfabético elige mal:
`LECHE` va antes que `Leche` (las mayúsculas son menores en ASCII) y `Salsa  de  tomate`
antes que `Salsa de tomate` (el espacio va antes que cualquier letra). Por eso el desempate
penaliza primero los espacios sobrantes y después estar todo en mayúsculas o todo en
minúsculas. Con el volumen real no hay empates, pero la regla no puede depender de eso.

**`brand` nula** (1,0 %): se rellena con `Sin marca` y se marca en `brand_is_missing`. No
se descarta el producto: su categoría, precio y ventas son válidos, y perderlos por no
saber la marca dejaría huecos en toda la venta.

---

### `basket_items` — el orden importa

Aquí están las dos reglas que más consecuencias tienen, y el orden entre ellas no es
intercambiable.

#### 1. Primero el signo: `quantity = abs(quantity)`

**Problema.** El 0,4 % de las líneas tiene cantidad negativa. `DATA_SPEC.md` las describe
como "devoluciones mal codificadas", que admite dos lecturas: o son devoluciones reales
apuntadas donde no toca, o son ventas normales a las que se les invirtió el signo.

**Evidencia.** Una devolución real tiene enfrente la venta original que anula. De las
17.253 líneas negativas, sólo 518 tienen una línea positiva del mismo producto en la misma
cesta — y esas 518 se explican solas por el mecanismo de duplicados (ver abajo). El 97 %
restante no anula nada: son ventas mal firmadas.

**Regla.** Se corrige el signo y se marca en `quantity_sign_corrected`. Descartarlas
perdería un 0,4 % de la demanda y, peor, la perdería de forma no aleatoria: sesgaría los
ciclos de recompra de la Tarea 2 en las categorías afectadas.

Es una decisión discutible, así que es un parámetro:
`CleaningConfig(negative_quantity_policy="drop")` las elimina, para quien prefiera la
lectura conservadora.

#### 2. Después el duplicado: se elimina la fila repetida entera

**Problema.** El 1,5 % de las líneas está duplicada (un TPV que emite la misma línea dos
veces).

**Por qué en este orden.** El generador duplica primero y ensucia el signo después. Cuando
la línea invertida es la copia de una duplicada, el par queda como `(+2, −2)` y **ya no es
un duplicado exacto**: sobrevive a cualquier `dropDuplicates`.

| Orden | Filas resultantes | `(basket_id, product_id)` duplicados |
| --- | ---: | ---: |
| Deduplicar y luego corregir el signo | 3.058.211 | **386** |
| Corregir el signo y luego deduplicar | 3.057.825 | **0** |

Esas 386 líneas fantasma inflarían la cesta y romperían el supuesto de "una línea por cesta
y producto" del que depende toda la afinidad de la Fase 3. El ETL cuenta explícitamente
cuántos duplicados afloran gracias al orden, y hay un test que lo fija
(`test_el_duplicado_con_el_signo_invertido_colapsa`).

**Detalle de implementación.** La deduplicación agrupa en vez de usar `dropDuplicates`,
porque las dos copias difieren en `quantity_sign_corrected` y `dropDuplicates` se quedaría
con una cualquiera: el flag saldría distinto en cada ejecución.

**Líneas con 0 unidades** se descartan: no son una venta. En este dataset no hay ninguna.

---

### `baskets` — el importe se recalcula

**Problema.** `total_amount` no cuadra con la suma de sus líneas en el 12,2 % de los
tickets, por tres motivos a la vez: líneas duplicadas, cantidades negativas y un 0,2 % de
importes multiplicados por 20-60x (los outliers inyectados).

**Regla.** `total_amount` se recalcula como la suma de `line_amount` de las líneas **ya
limpias**, tal y como pide `DATA_SPEC.md`. El importe original se guarda en
`total_amount_raw`.

**Por qué recalcular en vez de filtrar outliers.** Un filtro por IQR sobre `total_amount`,
incluso con la valla de valores extremos (`Q3 + 3·IQR` sobre el crudo), marca **21.318
cestas (3,55 %)** cuando los outliers inyectados son 1.208 (0,2 %): 17 de cada 18 serían
cestas grandes perfectamente legítimas. Con la valla habitual de `1,5·IQR` serían 53.444
(8,9 %). El recálculo no necesita adivinar, porque el dato correcto está en el detalle del
ticket.

Desde la Fase 8 el argumento es mucho más fuerte que antes: con el tamaño de cesta de cola
larga (compras semanales de 30-50 líneas conviviendo con reposiciones de 1-3), la
distribución del importe tiene una cola legítima enorme, y la valla marcaba 3.477 cestas
(0,58 %) sobre el dataset anterior frente a estas 21.318. Un umbral estadístico sobre el
importe no distingue "ticket inflado por un error" de "compra grande del sábado".

**Y sale redondo:** después de limpiar las líneas, quedan exactamente **1.208 cestas**
(0,201 %) cuya cabecera no cuadra con su detalle — justo los outliers inyectados, ni una
más. El resto de descuadres los causaban los duplicados y los signos. Se marcan en
`total_amount_is_outlier` con la regla "la cabecera supera 1,5x el importe recalculado";
como el multiplicador inyectado es de 20-60x, cualquier corte entre 1,5 y 10 aísla el mismo
conjunto, así que la regla no es frágil al umbral.

Se añaden `n_lines`, `n_units`, `basket_day` e `is_anonymous`, que usan RFM y el EDA.

---

### `customers` — altas imposibles y ciudades sin informar

**`signup_date` posterior a la primera compra** (58 clientes, 0,29 %). Un alta posterior a
una compra del propio cliente no puede ser: la compra es el hecho duro y la fecha de alta,
el dato administrativo. Se corrige a `min(signup_date, primera compra)` y se marca en
`signup_date_corrected`, conservando el original en `signup_date_raw`.

La misma regla arregla de paso los 3 clientes cuya alta caía **después del fin del periodo
observado** (enero de 2026, con datos hasta diciembre de 2025): los tres tienen compras, así
que la corrección los devuelve dentro de rango.

Dejarlo sin corregir rompería la antigüedad del cliente (`tenure`), que es una feature del
modelo de propensión de la Fase 4.

**`city` nula** (0,8 %): se rellena con `Desconocida` y se marca en `city_is_missing`. Se
rellena en vez de dejarla nula porque es un atributo de dimensión que se usa para agrupar,
y un nulo desaparece silenciosamente de un `GROUP BY` o de un segmento de dashboard.

---

### `promotions`, `sessions`, `session_events` — poco que hacer

- **`promotions`**: se marca `date_range_invalid` cuando `start_date > end_date`, pero **no
  se borra la fila**. Borrarla dejaría huérfana cualquier línea de ticket que la
  referencie, y cambiar una integridad referencial intacta por una rota es un mal negocio.
  En este dataset no hay ninguna; la regla está por si el generador cambia.
- **`sessions`**: `converted` pasa a derivarse de `basket_id IS NOT NULL` en vez de creerse
  la columna. Aquí las dos ya coinciden, pero es la clase de incoherencia que aparece en
  cuanto el dato viene de dos sistemas distintos.
- **`session_events`**: se eliminan 25 eventos duplicados exactos (mismo producto, mismo
  evento, mismo instante).

---

## Lo que la limpieza **no** hace

- **No elimina cestas anónimas.** Son el 3 % y son ventas reales.
- **No imputa la marca ni la ciudad con un modelo.** Con un 1 % de huecos, el ruido que
  metería una imputación supera lo que aporta. Se etiqueta y se decide aguas abajo.
- **No recorta los tickets grandes legítimos.** Sólo se corrige lo que contradice al
  detalle del ticket.
- **No toca `products.unit_price` cuando difiere de `unit_price_paid`.** Es la diferencia
  esperada por promoción, no un error, y es señal útil para el recomendador.

---

## Cómo se verifica

| Qué | Dónde |
| --- | --- |
| Cada regla, con un caso mínimo | `tests/test_cleaning.py` |
| Data Trust Score antes y después | `reports/etl/data_trust.md` |
| Recuento de lo corregido | `reports/etl/cleaning_report.md` |
| Que los patrones de negocio sobreviven | `python -m data_generation.verify_dataset` |

El lift de afinidad medido sobre el dato limpio con Spark coincide hasta el segundo decimal
con el que mide `data_generation/verify_dataset.py` en pandas sobre el dato crudo. Son dos
implementaciones independientes: que coincidan es la señal de que la limpieza no ha movido
la señal de negocio.

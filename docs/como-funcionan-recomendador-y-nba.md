# Cómo funcionan el recomendador y el Next Best Action

> **De dónde sale este documento.** Era un `.docx` escrito a mano
> (`docs/Como funcionan el recomendador y el Next Best Action.docx`), y se pasó a Markdown
> en el punto B3 del [diagnóstico](diagnostico-fase7.md). El motivo no es estético: un
> binario no se puede comparar en un *diff*, así que sus cifras envejecieron en silencio
> durante tres fases. Cuando se convirtió, decía que el `NDCG@5` era 0,0343 y que el
> sistema acertaba el SKU en el 11,8 % de las cestas — números de la Fase 5, cuando los
> reales ya eran 0,3015 y 61,6 %.
>
> **Este Markdown es ahora la fuente, y el `.docx` sale de él** con
> `python docs/build_docx.py`. No es una copia que haya que acordarse de actualizar: para
> que el `.docx` cambie tiene que cambiar este fichero, así que no puede volver a
> desfasarse por su cuenta.
>
> Las cifras salen a su vez de `reports/recommender/metrics.json` y
> `reports/nba/metrics.json`, que reescriben `python -m src.pipeline recommender` y
> `python -m src.pipeline nba`. Si vuelven a no cuadrar, la culpa es de este fichero y no
> de los informes.

Este documento explica, en lenguaje llano, qué hacen los dos modelos del proyecto
**grocery-retail-recommender**: qué buscan en los datos, cómo se preparan esos datos, cómo
se dividen en train/test y por qué así, y qué significan las métricas que reporta cada uno.
Para el detalle fase a fase está [`HISTORIA.md`](HISTORIA.md); para el estado actual, el
[`README.md`](../README.md).

Los dos modelos comparten la misma base de datos preparada (Fase 2) y a partir de ahí
divergen: el recomendador decide **qué productos** mostrar dentro de una cesta en curso; el
NBA decide **qué acción de negocio** tomar con cada cliente, al margen de si está comprando
en ese momento o no. Desde el punto M7 comparten además una capa: la **necesidad de
categoría**, explicada al final.

## Glosario

Términos que aparecen a lo largo del documento, explicados sin dar por hecho que se conocen:

- **NBA (Next Best Action):** en vez de tratar a todos los clientes igual, decidir para cada uno cuál es la mejor acción de negocio a tomar ahora mismo — recomendarle algo, ofrecerle un descuento, o directamente no hacer nada porque no compensa.

- **Churn:** el abandono de un cliente — deja de comprar. Se suele expresar como una probabilidad: "este cliente tiene un 70% de probabilidad de abandonar en las próximas 4 semanas".

- **Modelo de propensión:** un modelo que no predice un hecho seguro, sino una probabilidad de que algo ocurra (que el cliente compre, que abandone). "Propensión a comprar" = qué tan probable es que compre.

- **Recall@5:** de todo lo que un cliente acabó comprando de verdad, qué fracción estaba entre los 5 productos que el sistema recomendó. Mide si el sistema "no se deja fuera" lo importante.

- **NDCG@5:** parecido al Recall, pero además de acertar importa en qué posición aciertas — acertar en el primer puesto vale más que acertar en el quinto. Es la métrica principal del recomendador porque en una lista de 5 productos, el orden importa (lo primero se ve más).

- **Hit rate@5:** en qué porcentaje de cestas al menos uno de los 5 productos recomendados fue efectivamente comprado. Es la métrica más fácil de explicar, aunque la menos exigente de las tres.

- **AUC:** mide qué tan bien un modelo distingue entre dos clases (por ejemplo, "va a abandonar" vs "no va a abandonar"). Va de 0,5 (el modelo acierta como si tirara una moneda al aire) a 1,0 (acierta siempre). Por encima de 0,8 se considera un modelo sólido.

- **PR-AUC:** una variante del AUC más adecuada cuando lo que se quiere predecir es poco frecuente (por ejemplo, solo el 6% de los clientes compra una categoría concreta en 7 días). En esos casos el AUC normal puede parecer bueno aunque el modelo no sirva demasiado; el PR-AUC es más honesto en ese escenario.

- **RFM:** Recencia, Frecuencia y Monetario — tres números por cliente (cuándo compró por última vez, con qué frecuencia compra, cuánto se gasta de media). Es la forma más clásica de resumir el comportamiento de un cliente en pocas cifras.

- **ALS (Alternating Least Squares):** una técnica de "filtrado colaborativo" — encuentra patrones del tipo "clientes que compran parecido a ti también compraron esto", sin necesidad de saber nada sobre el contenido de los productos.

- **LightGBM / LambdaRank:** LightGBM es un algoritmo de árboles de decisión muy usado en la industria por ser rápido y preciso con datos tabulares. LambdaRank es el modo de entrenarlo específicamente para ordenar una lista (en vez de solo predecir un número o una categoría), que es justo lo que hace falta para decidir el orden del top-5.

- **Valor esperado:** el resultado promedio de una decisión si se repitiera muchas veces, teniendo en cuenta tanto la probabilidad de que algo funcione como su coste. Es la idea detrás de la política del NBA: no se trata de qué acción "podría" funcionar, sino de cuál gana más dinero de media una vez se descuenta lo que cuesta intentarla.

- **Cliente nuevo vs. recurrente (cold start):** un cliente nuevo no tiene historial de compras que analizar — es lo que en la industria se llama el problema de "arranque en frío". El recomendador lo resuelve apoyándose en popularidad y estacionalidad en vez de en historial personal.

## 1. La base común: qué se le hace al dato antes de modelar (Fase 2)

Ni el recomendador ni el NBA entrenan sobre el dato crudo. Antes pasa por:

- **Limpieza:** duplicados, nulos, categorías mal escritas, cantidades negativas y outliers de importe — documentado en `docs/CLEANING.md`. El **Data Trust Score** (79 comprobaciones en 5 dimensiones) sube de **91,38 (C) en crudo a 100,00 (A) en limpio**, así que hay una medida objetiva de que la limpieza cumple su función, no solo una afirmación.

- **RFM por cliente:** la base de la segmentación y de varias features de ambos modelos.

- `due_for_repurchase`: por cliente y categoría, si ya ha pasado el intervalo típico de recompra de esa categoría para ese cliente concreto (no un intervalo genérico de la categoría). Alimenta tanto al recomendador (repescar lo que "toca") como al NBA (propensión de compra a 7 días).

- **Afinidad de cesta:** qué categorías y qué productos aparecen juntos en una cesta más de lo que el azar explicaría (co-ocurrencia / FP-Growth). Los 30 pares que se diseñaron a propósito en `DATA_SPEC.md` (cerveza+snacks, pañales+toallitas...) salen con el lift que se buscaba, confirmando que la señal de negocio que se inyectó en el generador sintético sí llega intacta hasta esta tabla.

Estas cuatro salidas son las que consumen después el recomendador y el NBA — ninguno de los dos vuelve a tocar el dato crudo.

## 2. Recomendador de cesta

### Qué busca el modelo en los datos

La idea central es que **una cesta se explica por varias señales distintas a la vez**, y ninguna sola basta: un cliente nuevo no tiene historial que mirar, uno recurrente sí; una cesta vacía no tiene con qué calcular afinidad, una cesta con productos sí. Por eso el sistema no es un único modelo, sino dos etapas:

**Etapa 1 — Generación de candidatos. **Cinco fuentes independientes proponen productos candidatos para una cesta dada, cada una mirando algo distinto en el dato: popularidad y estacionalidad (el único candidato disponible si no hay ni cliente conocido ni cesta empezada), co-compra por producto y por categoría (esta última cubre la "cola larga" donde la afinidad exacta no tiene suficientes datos), ALS (la señal colaborativa: "clientes parecidos a ti compraron esto"), e historial personal combinado con `due_for_repurchase`. Entre las cinco, generan un **pool de ~234 candidatos por cesta** de media, de los 496 del catálogo.

**Etapa 2 — Ranking. **Un **LightGBM** con objetivo de ranking (LambdaRank) reordena ese pool y se queda con el top-5. No decide qué candidatos existen (eso ya lo hizo la etapa 1); decide en qué orden importan, usando **60 features en 5 familias**: la señal de cada fuente de candidatos, recencia/frecuencia del cliente con esa categoría o producto, si está en promoción, popularidad reciente, y señal de sesión (qué ha visto o añadido al carrito antes de completar la cesta).

La señal de sesión mereció una mención aparte: en la Fase 2 se detectó que estaba contaminada (view y `add_to_cart` cubrían el 100% de la cesta, es decir, no aportaba nada que no se supiera ya). Se corrigió el generador para que hubiera abandono de carrito a nivel de línea y productos que solo se miran sin comprarse. Una vez arreglada, esa señal aporta **+13% de NDCG@5** frente a no usarla — la prueba de que la señal es real y no ruido.

Los **4 perfiles de cliente** del reto no son 4 modelos distintos: es **el mismo ranker** para los cuatro. Lo que cambia es qué fuentes de candidatos tienen señal disponible en cada caso — el perfil 1 (cliente nuevo, cesta vacía) se cubre 100% con popularidad; el perfil 4 (cliente recurrente, cesta en curso) combina las cinco fuentes a la vez.

### Train / test

El split es **temporal y por cesta**, no aleatorio y no por fila: las fuentes de candidatos se calculan con datos hasta el 2025-09-01; las cestas que sirven de ejemplos de entrenamiento del ranker son de septiembre-octubre; el test son cestas reales desde el 2025-11-01, nunca vistas en ninguna de las dos fases anteriores.

Dos motivos para esto. Cortar por `basket_id` y no por línea suelta evita que dos productos de la misma cesta acaben uno en train y otro en test (fuga trivial). Cortar además en el tiempo evita que el ranker aprenda de candidatos calculados con datos "del futuro" respecto a la cesta que está prediciendo — las fuentes de candidatos se **recalculan por ventana** para que ninguna vea por delante.

### Métricas y resultado

La métrica principal es la **NDCG@5 con relevancia graduada**: vale 2 si el sistema acierta
la referencia exacta y 1 si acierta la categoría. Se eligió así en el punto A4 porque la
demo y el discurso de negocio hablan de categoría mientras que el ranker optimizaba solo el
SKU, y mientras no se fije cuál manda, una mejora en una puede empeorar la otra sin que
nadie lo vea.

| Métrica (18.000 cestas de test) | Valor |
| --- | ---: |
| **NDCG@5 graduada** (principal) | **0,3015** [0,2981, 0,3052] |
| Acierto de categoría @5 | 79,5 % |
| Acierto de SKU exacto @5 | 61,6 % |
| Del techo teórico de categoría (oráculo) | 94,9 % |
| Sobre el mejor baseline (reglas de asociación) | +3,6 pp [+3,0, +4,2] |

Los corchetes son intervalos al 95 % por *bootstrap* de cestas (1.000 remuestreos). No son
decoración: los perfiles de cliente nuevo tienen pocas cestas, y sin intervalo es fácil
contar como mejora algo que cabe dentro del ruido.

El "techo teórico" es un oráculo que conoce las probabilidades reales con las que el
generador construyó cada cesta (`src/recommender/oracle.py`). Sirve para responder a la
primera pregunta que hace cualquiera que ve un 79,5 %: cuánto de lo que falta es mejorable
y cuánto es azar irreducible. La respuesta es que el sistema se queda al **94,9 %** de lo
alcanzable.

### El hallazgo honesto de esta fase

Durante tres fases, el `NDCG@5` de SKU fue bajísimo (0,0343) y **no era culpa del modelo**.
El sistema acertaba la categoría en el 51,4 % de las cestas y el SKU exacto solo en el
11,8 %; ampliar el pool de candidatos de 92 a 155 subía el recall del pool del 21,7 % al
31,1 % y **no movía el NDCG** (0,0344 → 0,0343).

La causa estaba en el generador: dentro de una categoría había ~24 referencias (el catálogo era de 1.500 productos) y la
elección entre ellas era casi aleatoria, así que el cliente sintético era fiel a la
categoría pero no a la marca. Ningún ajuste del ranker podía superar ese techo.

**Se arregló donde estaba el problema, en el dato** (Fase 7): fidelidad de marca por cliente
y categoría, y catálogo reducido a 496 referencias. Con el mismo código de modelado, el
acierto de SKU pasó del 11,8 % al 49,5 % y el `NDCG@5` se multiplicó por cinco. Las mejoras
posteriores de la segunda etapa (historial *as-of*, features de carrito, relevancia
graduada) lo llevaron al 61,6 % actual.

La lección se dejó escrita porque se repite: **antes de tocar el modelo, comprobar que la
métrica no está midiendo una limitación del dato**. El punto M6 volvió a confirmarlo más
tarde — ampliar el pool de 141 a 234 candidatos subió el techo de la primera etapa 9,3 pp y
movió la métrica final +0,03 pp.

## 3. Next Best Action

### Qué busca el modelo en los datos

Aquí la pregunta no es "qué va a comprar" sino "qué le pasa a este cliente y qué merece la pena hacer al respecto". Dos modelos de propensión (LightGBM o logística, uno por target): compra en una categoría concreta en los próximos 7 días, y churn en las próximas 4 semanas (definido como inactividad, ver más abajo).

Un detalle importante: **no se usa **customers.churn_label** como target de entrenamiento**. Esa etiqueta está definida respecto al final del dataset completo (60 días sin comprar hasta 2025-12-31), así que entrenar con ella en un corte de, por ejemplo, abril equivaldría a pedirle al modelo que adivinara algo que ocurre ocho meses después — y usarla como feature sería fuga directa de información del futuro. En su lugar, el churn se construye de forma observacional en cada corte temporal, y churn_label se reserva solo como comprobación de cordura al final (da un 83,4% de coincidencia con lo que predice el modelo).

### Train / test

**Split temporal**, no aleatorio: cuatro cortes de entrenamiento entre abril y agosto de 2025, validación en 2025-09-15, test en 2025-11-01. Las features de cada corte solo miran compras **anteriores** a ese corte — se verifica con un test que añade una compra posterior a propósito y comprueba que ninguna feature se mueve, para pillar cualquier fuga futura de forma automática.

### La política de decisión

Con las dos probabilidades ya calculadas, una **política de valor esperado** decide la mejor acción de un catálogo fijo (ninguna_acción, recomendar_producto, enviar_cupón_categoría):

acción* = argmax_a ( P(conversión | acción=a) × margen_esperado(a) − coste(a) )

El margen es específico por departamento (18% en Frescos, hasta 35% en Droguería/Higiene) y el coste se separa en dos partes: el coste de enviar la comunicación (siempre se paga) y el del descuento (solo si el cliente compra). Todo se mide **incremental sobre no actuar**, así que ninguna_acción vale 0 por construcción. Con esto, la política actúa sobre el 75,6% de los clientes — no sobre todos, y eso es una decisión del modelo, no un límite técnico.

### Métricas y resultado

| | Valor |
| --- | ---: |
| Churn a 4 semanas | AUC **0,8532** · PR-AUC 0,8576 (tasa base 0,4759) |
| Compra en categoría a 7 días | AUC **0,7591** · PR-AUC 0,2479 (tasa base 0,0755) |

Con una tasa base del 7,5 %, en el segundo modelo el PR-AUC es la métrica que de verdad
importa: un AUC decente puede convivir con un modelo que no sirva para priorizar.

Y en euros de valor incremental sobre no hacer nada:

| Política | Valor |
| --- | ---: |
| No actuar nunca | 0 € |
| Cupón a todo el mundo | −5.032 € |
| Recomendar categoría a todo el mundo | +670 € |
| **Política de valor esperado** | **+3.078 €** |

### El hallazgo honesto de esta fase

P(conversión | acción) **no se puede medir con este dataset**: el generador aplica su uplift de promoción al reparto de qué SKU se elige dentro de una cesta, no a si el cliente compra la categoría o si vuelve a comprar. La propensión sí se mide; el efecto de cada acción es un **supuesto declarado** en `src/nba/config.py`, sometido a un barrido de sensibilidad sobre `churn_reduction` (cuánto reduce el cupón la probabilidad de abandono):

| `churn_reduction` | Valor de la política | Cupones repartidos |
| --- | ---: | ---: |
| 0,00 (no retiene a nadie) | 1.158 € | 0,0% |
| 0,02 | 1.158 € | 0,1% |
| 0,05 | 1.454 € | 18,0% |
| 0,10 (el supuesto usado) | 3.078 € | 47,2% |
| 0,15 | 5.156 € | 59,2% |
| 0,20 | 7.359 € | 64,3% |

La fila que importa es la primera: **incluso en el escenario más pesimista, donde el cupón no retiene a nadie, la política sigue ganando** (1.158 € frente a 0 € de no actuar), y en ese caso deja de repartir cupones por completo. El supuesto concreto cambia **cuánto** gana la política, no si conviene usarla.

## 4. La capa que comparten: necesidad de categoría

`P(compra en categoría a 7 días)` del NBA y la "necesidad de categoría" del recomendador son
casi la misma cantidad con horizontes distintos. Desde el punto M7 comparten literalmente la
misma función, `recommender.formulas.category_need_weight`, sobre el `overdue_ratio` de la
Tarea 2.

Que se pueda compartir no era evidente, así que se midió antes de hacerlo: el
`overdue_ratio` ordena también la etiqueta del NBA, que es otra cosa. La tasa real de compra
a 7 días va del **4,6 %** en categorías recién repuestas al **9,7 %** en las que ya tocan, y
vuelve a bajar al 6,0 % en la cola de categorías abandonadas.

Y hacía falta. El **88,5 %** de los cupones caía en categorías que el recomendador da por no
vencidas, con una tasa real de compra del **1,5 %** — peor que el 7,5 % de elegir al azar
del pool. La causa no era el modelo de propensión, era la política:

- el término de *cross-sell* del cupón es **decreciente** en `p_purchase`, porque el
  descuento de 2,54 € supera el margen de ~1,4 € que persigue: cuanto más segura es la
  compra, más dinero se deja en la mesa;
- el término de retención **no dependía de la categoría**;
- así que el `argmax` sobre categorías no maximizaba lo que le sirve al cliente, sino lo que
  **minimiza la fuga de descuento**: la categoría que el cliente casi seguro no iba a
  comprar.

La relevancia entra ahora ponderando la retención — una oferta de algo que el cliente acaba
de reponer no retiene a nadie — y el reparto cambia:

| Categoría elegida para el cupón | Antes | Ahora |
| --- | ---: | ---: |
| Que el recomendador da por no vencida | 88,5 % | **23,2 %** |
| Recién repuesta | 73,3 % | **2,0 %** |
| Tasa real de compra a 7 días | 1,5 % | **4,3 %** |

El valor total de la política baja de 3.842 € a 3.078 €, y **esa bajada es el resultado
correcto**: los 764 € de diferencia eran retención apuntada a ofertas que no retienen a
nadie. Lo recalcula `python -m src.nba.verify_category_need`.

## En una frase

**Recomendador:** cinco fuentes proponen candidatos por razones distintas (qué se vende, qué compró antes, qué compra gente parecida, qué le toca reponer), y un único LightGBM los reordena con 60 features — acierta la categoría en el 79,5 % de las cestas y la referencia exacta en el 61,6 %, al 94,9 % del techo que marca un oráculo con las probabilidades reales del generador.

**NBA:** dos modelos de propensión (compra próxima, churn) alimentan una política de valor esperado que decide la mejor acción por cliente — el resultado económico depende de un supuesto de negocio que no se puede medir con este dataset, así que en vez de ocultarlo, el proyecto lo somete a un barrido de sensibilidad y muestra que la conclusión aguanta incluso en el escenario más desfavorable.

**Y lo que comparten:** una sola definición de "a este cliente le toca esta categoría", que el recomendador usa para ordenar candidatos y el NBA para no gastar un cupón en algo que el cliente acaba de reponer.

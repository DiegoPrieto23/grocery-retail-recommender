# DATA_SPEC — Esquema del dataset sintético

Detalle columna a columna para `data_generation/generate_dataset.py`. Los rangos y
distribuciones son punto de partida: ajustar al generar y comprobar que el Data Trust Score
y las correlaciones salen coherentes, igual que en `sports-rental-analytics`.

---

## `customers`

| Columna              | Tipo   | Notas                                                              |
| --------------------- | ------- | -------------------------------------------------------------------- |
| `customer_id`          | string  | PK                                                                    |
| `signup_date`          | date    | Fecha de alta                                                         |
| `country`              | string  | España (regiones simuladas) — se puede ampliar más adelante          |
| `city`                 | string  |                                                                        |
| `household_size_est`   | int     | 1-6, afecta a la cantidad y frecuencia de compra                     |
| `loyalty_tier`         | string  | `bronze` / `silver` / `gold`                                          |
| `preferred_channel`    | string  | `app` / `web` / `store`                                               |
| `churn_label`          | boolean | Derivado: 1 si no ha comprado en los últimos 60 días del periodo simulado |

---

## `products`

| Columna            | Tipo    | Notas                                                                 |
| -------------------- | -------- | ------------------------------------------------------------------------ |
| `product_id`          | string   | PK                                                                        |
| `department`          | string   | Frescos, Despensa, Bebidas, Droguería, Higiene, Bebé, Mascotas, Congelados |
| `category`            | string   | Subcategoría dentro del departamento                                     |
| `brand`               | string   |                                                                            |
| `is_private_label`    | boolean  | Marca blanca                                                              |
| `is_perishable`       | boolean  | Afecta a frecuencia de recompra y merma                                  |
| `unit_price`          | float    |                                                                            |
| `pack_size`           | int      |                                                                            |
| `typical_repurchase_days` | int | Intervalo típico de recompra de la categoría (usado por el generador y por Tarea 2) |

**Tamaño del surtido: 8 referencias por categoría** (`catalog.PRODUCTS_PER_CATEGORY`), las
mismas en todas — 62 categorías × 8 = **496 productos**. El catálogo original repartía
1.500 productos de forma proporcional a la popularidad de la categoría (~24 referencias
por categoría), y el hallazgo de la Fase 3 (ver `ROADMAP.md`) fue que eso ponía un techo
artificial al recomendador: con 24 referencias equivalentes y elegidas casi al azar,
acertar el SKU exacto era casi imposible por construcción. Un surtido curado de 8
referencias es el orden de magnitud que ve un cliente delante del lineal de una categoría
concreta, y deja que la señal de SKU la ponga la fidelidad de marca (más abajo) en vez del
tamaño del surtido.

---

## `promotions`

| Columna         | Tipo   | Notas                                    |
| ----------------- | ------- | ------------------------------------------- |
| `promotion_id`     | string  | PK                                           |
| `product_id`       | string  | FK a `products`                              |
| `promo_type`       | string  | `2x1` / `discount_pct` / `coupon`            |
| `discount_value`   | float   |                                               |
| `start_date`       | date    |                                               |
| `end_date`         | date    |                                               |

---

## `baskets` (cabecera de ticket)

| Columna         | Tipo    | Notas                                     |
| ----------------- | -------- | --------------------------------------------- |
| `basket_id`        | string   | PK                                             |
| `customer_id`      | string   | FK a `customers` — nulo si compra anónima      |
| `channel`          | string   | `app` / `web` / `store`                        |
| `basket_date`      | datetime |                                                 |
| `store_id`         | string   | Nulo si el canal no es `store`                 |
| `total_amount`     | float    | Suma de `basket_items` — se recalcula en ETL   |

---

## `basket_items` (línea de ticket)

| Columna              | Tipo   | Notas                                  |
| ---------------------- | ------- | ------------------------------------------ |
| `basket_id`            | string  | FK a `baskets`                              |
| `product_id`            | string  | FK a `products`                             |
| `quantity`              | int     |                                              |
| `unit_price_paid`       | float   | Puede diferir de `unit_price` si hay promo  |
| `promotion_id`          | string  | Nulo si no aplica                           |

---

## `sessions` (canal online)

| Columna         | Tipo    | Notas                                             |
| ----------------- | -------- | ----------------------------------------------------- |
| `session_id`       | string   | PK                                                      |
| `customer_id`      | string   | Nulo si es visitante no identificado                    |
| `session_date`     | datetime |                                                          |
| `device_type`      | string   |                                                          |
| `converted`        | boolean  | Si terminó en `basket_id`                                |
| `basket_id`        | string   | FK a `baskets`, nulo si no convirtió                     |

## `session_events`

| Columna         | Tipo     | Notas                              |
| ----------------- | --------- | -------------------------------------- |
| `session_id`       | string    | FK a `sessions`                          |
| `product_id`        | string    | FK a `products`                          |
| `event_type`        | string    | `view` / `add_to_cart`                    |
| `event_timestamp`   | datetime  |                                            |

---

## Patrones a inyectar (resumen operativo)

- **Ciclo de reposición**: para cada `(customer_id, category)`, generar compras espaciadas
  ≈ `typical_repurchase_days` del producto, con ruido gaussiano, escalado por
  `household_size_est` (hogares más grandes, ciclos más cortos).
- **Uplift de promoción**: multiplicar la probabilidad base de compra de un producto por un
  factor >1 mientras `promotion_id` esté activa.

### Estructura de la cesta: qué co-ocurrencia se busca y por qué (Fase 8)

Esta sección se escribió **antes** de tocar el generador de la Fase 8 (puntos A5, M1 y M2
de `docs/diagnostico-fase7.md`). Fija contra qué se calibra: el realismo de una cesta de
supermercado, **no** la métrica del recomendador. Si el modelo mejora o empeora con el
dataset nuevo, es un resultado, no un objetivo.

**Punto de partida (dataset de la Fase 7a).** Una cesta era un muestreo casi independiente
de categorías condicionado al cliente: solo 40 de los 3.780 pares ordenados de
`affinity_category` tenían lift > 1,5, y todos venían de los 10 pares de la tabla de
abajo o de los grupos bebé y mascota, que salen de los *gates* de hogar y no de la cesta.
El tamaño era una Poisson recortada (media 5,1 líneas, coeficiente de variación 0,42,
p99 = 11, máximo 19), con una sola línea por categoría. Cifras congeladas en
`snapshots/pre-fase-8/reports/etl/verify_dataset.json`.

**Cómo es una cesta real de gran consumo.** Tres rasgos que el dataset no tenía:

1. **Las visitas tienen una misión.** Conviven la compra grande semanal (20-40 categorías),
   la reposición rápida de frescos (pan, leche, fruta: 1-3 artículos), el desayuno, la
   limpieza del hogar, la cena o el aperitivo, la compra del bebé y la fiesta o barbacoa
   de temporada. La misión explica la mayor parte de la co-ocurrencia: el café va con la
   leche porque las dos son desayuno, no por una regla par a par.
2. **El tamaño tiene cola larga.** En los datasets públicos de referencia (Instacart, por
   ejemplo, con ~10 productos por pedido de media y pedidos de más de 50) la desviación
   típica del tamaño es del orden de la media, no de su raíz cuadrada como en una Poisson.
3. **Hay sustitutos, no solo complementos.** Quien lleva pollo para la cena lleva menos
   ternera; quien carga agua lleva menos refrescos. Un buen recomendador de cesta tiene
   que poder aprender esa señal negativa.

**Un matiz de medida que condiciona el objetivo.** El lift crudo de `affinity_category`
mezcla la afinidad con el **tamaño**: en una compra de 30 categorías aparece casi todo, así
que con una distribución de tamaño realista *cualquier* par co-ocurre más que por azar.
Con un coeficiente de variación `cv`, el lift de dos categorías poco frecuentes y sin
ninguna relación ronda `1 + cv²` (≈ 1,5-1,8 con `cv` entre 0,7 y 0,9). Por eso se miden
dos cosas en `verify_dataset.py` (sección `coocurrencia_de_categorias`):

- el **lift crudo**, idéntico al de `affinity_category`, que es lo que ve el recomendador;
- el **lift controlado por tamaño**: co-ocurrencia observada frente a la esperada si,
  dentro de cada banda de tamaño, las categorías fueran independientes, dividido por la
  mediana de todos los pares (sortear sin reemplazo un número fijo de categorías ya
  introduce una ligera dependencia negativa). Un 1 es "lo normal para una cesta de ese
  tamaño". Es el que dice si hay estructura de verdad.

**Objetivos.** Rangos, no puntos: se da por bueno cualquier valor dentro.

| Qué | Antes (7a) | Objetivo | Por qué |
| --- | ---: | ---: | --- |
| Líneas por cesta: media | 5,1 | 7-10 | Orden de magnitud de los datasets públicos, con la compra en tienda (más pequeña) mezclada |
| Líneas por cesta: mediana | 5 | 4-7 | La mayoría de visitas son pequeñas |
| Líneas por cesta: coeficiente de variación | 0,42 | 0,75-1,10 | Cola larga: desviación del orden de la media |
| Líneas por cesta: p99 | 11 | 30-50 | Compras semanales grandes |
| Cestas de más de 20 líneas | 0 % | 4-12 % | La compra grande es una minoría de las visitas |
| Cestas de 1 a 3 líneas | 24,6 % | 25-40 % | La reposición rápida es la visita más frecuente |
| Pares con lift **controlado** > 1,5 | 38 | 150-450 (4-12 %) | Misiones (desayuno, limpieza, cena, bebé, fiesta) y complementos. Por encima del 12 %, las cestas serían plantillas |
| Pares con lift **crudo** > 1,5 | 40 | muchos más (cientos o miles) | Inevitable con cola larga de tamaño: no es un objetivo en sí, se reporta para que se lea con el matiz de arriba |
| Mediana del lift crudo | 1,00 | se reporta (ver nota) | El efecto tamaño: no es un objetivo |
| Complementos declarados: lift crudo | 1,8-13 | ≥ 1,8 (bebé y mascota pueden pasar de 10 por el *gate*) | Receta o rutina: se compran juntos varias veces más que por azar |
| Complementos declarados: lift controlado | 1,6-10,7 | ≥ 1,4 | Que la relación sobreviva al quitar el tamaño |
| Sustitutos declarados: lift controlado frente al mismo generador sin grupos | — | ≤ 0,85 en todos los pares | Llevar uno resta al otro, sin llegar a excluirlo |
| Sustitutos que no comparten misión: lift controlado | 0,87-0,91 (sin grupos) | < 1 | Sin una misión que los junte, se ve la señal negativa directamente |
| Categorías de exploración con dos referencias en la misma cesta | 0 % | 5-15 % | Dos yogures o dos frutas distintas son normales en la cesta real; en las de hábito, no |
| Categorías de hábito con dos referencias | 0 % | 0 % | La fidelidad de marca de la 7a no se toca |
| Cuota de marca blanca: p10-p90 entre clientes | 0,17-0,33 | claramente más ancho (p. ej. 0,10-0,45) | Hay hogares que buscan precio y hogares que no |

**Qué se revisó al medir, y por qué.** Tres cosas de esta sección cambiaron después de la
primera calibración, y ninguna para acercar una métrica del modelo:

- *El estimador del lift controlado.* La primera versión ponderaba cada banda de tamaño por
  sus co-ocurrencias (tipo Mantel-Haenszel). Así, casi todo el peso caía en las compras
  semanales, que es justo donde no hay estructura: en cestas de 3-12 categorías el par
  café-cereales tenía lift 2-3, y el estimador lo dejaba en 1,0. Ahora cada banda pesa por
  sus cestas, y las cestas de una sola categoría (que no dicen nada de pares) quedan fuera.
- *La mediana del lift crudo.* Se había fijado en 1,1-1,6 pensando en categorías de
  frecuencia media. Con misiones, las categorías de acopio (sal, harina, limpiacristales)
  aparecen casi solo en la compra semanal, y su lift crudo con cualquier otra de acopio
  sube a 3-6 sin que haya una relación entre ellas. Eso también pasa en una cesta real,
  así que la mediana cruda (≈ 2,5) se reporta y no se persigue.
- *Los sustitutos que comparten misión.* Lavavajillas y lejía están en el núcleo de la
  misión de limpieza; pizza y precocinados, en el de la cena. La misión los junta más de
  lo que el grupo los separa, y su lift controlado queda por encima de 1 aunque el grupo
  les reste. El objetivo pasa a medirse contra el mismo generador sin grupos (lo comprueba
  `test_la_sustitucion_resta_frente_a_no_tenerla`), y el "< 1" se exige a los que no
  comparten misión.

Lo que **no** debe moverse (sección "Lo que el generador ya hace bien" del diagnóstico):
identidad y cadencia de visita del cliente, ciclo de reposición (la correlación de rangos
de `ciclos_reposicion` debe seguir por encima de 0,95), fidelidad de marca (cuota de la
favorita en hábito dentro de 0,75-0,85), embudo de sesión, estacionalidad, churn
progresivo y reproducibilidad byte a byte.

### Misiones de compra (detalle)

Cada cesta tiene una **misión latente** que el dataset no publica (solo la conoce el
generador y, para el techo teórico, el oráculo). La misión decide dos cosas: el **perfil de
categorías** (un multiplicador sobre el peso de cada categoría) y la **distribución del
tamaño**. Todo lo demás se mantiene: el peso de partida de cada categoría sigue siendo
afinidad del cliente × estacionalidad × ciclo de reposición, con los *gates* de hogar, y la
misión solo lo reescala.

| Misión | Categorías núcleo (multiplicador) | Resto | Categorías distintas: 1 + BN(media, r) | Peso base |
| --- | --- | ---: | --- | ---: |
| `compra_semanal` | Todas; despensa, droguería e higiene ×1,3 (se hace acopio); caprichos ×0,8 | ×1,0 | media 15, r = 4 | 0,12 |
| `reposicion` | Pan, leche ×3; fruta, verdura ×2,5; huevos, yogur ×2; agua, embutido ×1,5 | ×0,20 | media 2,5, r = 2 | 0,36 |
| `desayuno` | Leche, café, cereales ×3; galletas, zumos ×2,5; yogur, pan, azúcar ×2; fruta, huevos ×1,2 | ×0,15 | media 3,2, r = 3 | 0,10 |
| `limpieza_hogar` | Detergente, suavizante, lavavajillas, lejía, limpiacristales, bolsas ×3; papel higiénico ×2,5; gel, champú, acondicionador, pasta de dientes, desodorante ×2; higiene femenina ×1,5 | ×0,15 | media 3,5, r = 3 | 0,09 |
| `cena_aperitivo` | Pizza, precocinados, snacks, cerveza ×3; vino, queso, embutido, refrescos, palomitas ×2,5; pan, helados ×2; conservas de pescado ×1,5 | ×0,20 | media 4,0, r = 3 | 0,15 |
| `bebe` | Pañales, toallitas, leche infantil, potitos ×4; fruta, yogur ×1,5; leche, gel ×1,2 | ×0,20 | media 3,0, r = 3 | 0,22 si hay bebé; 0 si no |
| `fiesta` | Cerveza, snacks ×3,5; ternera, pollo, refrescos, cava, turrón, marisco ×3; vino ×2,5; agua, pan, embutido, queso, helados ×2 | ×0,20 | media 7, r = 3 | 0,02; ×3 de junio a septiembre (barbacoa), ×4 en diciembre |

**Probabilidad de cada misión por cliente.** Parte del peso base y se ajusta con el hogar
(la compra semanal pesa más cuanto más grande es el hogar, `0,5 + 0,2 × miembros`; la
reposición, menos, `1,3 − 0,1 × miembros`), con el canal (online ×1,5 en la compra
semanal: el pedido a domicilio es la compra grande) y con los *gates* (sin bebé no hay
misión `bebe`). Encima, cada cliente tiene su propio reparto, sorteado con una Dirichlet
de concentración 20 alrededor de ese perfil: hay hogares de compra semanal y hogares de
ir cada día a por pan. En cada visita se aplican además el día de la semana (viernes y
sábado ×1,6 para la compra semanal) y el mes (la fiesta). Las cestas anónimas usan el
reparto medio de la población.

**Tamaño.** El número de categorías distintas es `1 + BinomialNegativa(media × f, r)`, con
`f` el mismo factor de antes (hogar × rampa de churn) y un tope de **50 categorías**
(antes, 20 líneas). La binomial negativa de cada misión y la mezcla entre misiones dan la
cola larga. Los parámetros exactos viven en `data_generation/catalog.py` (`MISSIONS`).

### Sustitución entre categorías (detalle)

Al entrar en la cesta una categoría de un grupo, el peso del resto de categorías del grupo
se multiplica por el factor (menor que 1). Actúa igual que los complementos, pero hacia
abajo, y en las dos direcciones.

| Grupo | Categorías | Factor | Motivo |
| --- | --- | ---: | --- |
| Bebida fría | Agua, Refrescos | 0,35 | Se carga una de las dos |
| Proteína principal | Carne de pollo, Carne de ternera, Pescado blanco | 0,30 | Una para la comida o la cena de ese día |
| Limpieza de cocina | Lavavajillas, Lejía y limpiadores | 0,25 | Reposición alterna del armario de limpieza |
| Capricho dulce | Galletas, Chocolate y huevos de Pascua | 0,35 | Un capricho por visita |
| Cena congelada | Pizza congelada, Precocinados congelados | 0,30 | Una solución rápida por noche |
| Picoteo | Snacks y aperitivos, Palomitas de microondas | 0,35 | Uno para la noche de sofá |
| Alcohol de mesa | Cerveza, Vino | 0,35 | Una bebida para la comida |

### Varias referencias por categoría y marca blanca (detalle)

- **Segunda referencia.** En las categorías de la banda de exploración (lealtad 0,25-0,40),
  cuando la categoría sale, con probabilidad **0,12** entra además una segunda referencia
  distinta de la misma categoría (dos yogures, dos frutas). No consume un hueco de
  categoría: el número de categorías distintas sigue siendo el de la misión. En las de
  hábito sigue habiendo una sola línea.
- **Propensión a la marca blanca.** Cada cliente tiene un multiplicador `ρ ~ LogNormal(0;
  0,8)` sobre el peso de las referencias de marca blanca. Se aplica en la parte de la
  elección que reparte la popularidad: en la primera compra de la categoría (la que fija la
  referencia preferida) y en el hueco `1 - lealtad`. La fidelidad de marca no cambia: la
  referencia preferida sigue llevándose su cuota. Las cestas anónimas usan `ρ = 1`.

### Afinidad de cesta (detalle)

Al construir cada `basket_items`, si ya hay un producto de la categoría "disparadora" en la
cesta, aplicar el lift indicado a la probabilidad de añadir un producto de la categoría
asociada (en vez de la probabilidad marginal). Sirve de señal real para el candidato de
co-compra del recomendador.

| Categoría disparadora | Categoría asociada     | Lift objetivo | Motivo                  |
| ----------------------- | ------------------------ | --------------- | -------------------------- |
| Cerveza                  | Snacks / aperitivos       | 3.0x            | consumo social              |
| Pasta                    | Salsa de tomate            | 2.5x            | receta directa              |
| Pañales                  | Toallitas húmedas          | 4.0x            | cesta de bebé                |
| Café                     | Azúcar / edulcorante       | 2.0x            | hábito de desayuno           |
| Pan de molde/barra       | Embutido / fiambre         | 2.2x            | bocadillo                    |
| Cereales                 | Leche                      | 2.8x            | desayuno                     |
| Vino                     | Queso                      | 2.5x            | maridaje                     |
| Detergente                | Suavizante                 | 3.5x            | rutina de lavado             |
| Champú                    | Acondicionador              | 3.0x            | rutina de higiene            |
| Palomitas de microondas   | Refrescos                   | 2.0x            | noche de peli                |

**Ampliación de la Fase 8** (punto A5): veinte pares más, en los cuatro grupos que pedía el
diagnóstico.

| Grupo | Categoría disparadora | Categoría asociada | Lift objetivo | Motivo |
| --- | --- | --- | ---: | --- |
| Desayuno | Café | Leche | 2.0x | café con leche |
| Desayuno | Galletas | Leche | 2.0x | desayuno y merienda |
| Desayuno | Cereales | Yogur | 2.0x | desayuno |
| Desayuno | Pan | Aceite de oliva | 2.0x | tostada |
| Higiene | Gel de ducha | Champú | 2.2x | rutina de ducha |
| Higiene | Pasta de dientes | Desodorante | 2.0x | neceser |
| Higiene | Papel higiénico | Lejía y limpiadores | 2.0x | limpieza del baño |
| Mascota | Comida para gato | Arena para gato | 3.0x | mismo animal |
| Mascota | Comida para perro | Bolsas de basura | 2.0x | paseo del perro |
| Receta | Pasta | Queso | 2.0x | pasta gratinada |
| Receta | Arroz | Marisco | 2.5x | paella |
| Receta | Legumbres | Embutido y fiambre | 2.0x | cocido y fabada |
| Receta | Carne de pollo | Verdura | 2.0x | plato de diario |
| Receta | Pescado blanco | Verdura | 2.0x | plato de diario |
| Receta | Harina | Huevos | 2.5x | repostería |
| Receta | Harina | Azúcar y edulcorante | 2.5x | repostería |
| Receta | Aceite de oliva | Sal y especias | 2.2x | despensa de cocina |
| Aperitivo y fiesta | Pizza congelada | Refrescos | 2.0x | cena rápida |
| Aperitivo y fiesta | Cava y espumosos | Turrón y mazapán | 2.5x | Navidad |
| Aperitivo y fiesta | Carne de ternera | Cerveza | 2.0x | barbacoa |

La tabla completa vive en `catalog.COMPLEMENT_PAIRS` (los 10 primeros siguen también en
`catalog.AFFINITY_PAIRS`, que es la que comprueba el informe del ETL). Una categoría puede
disparar varias asociadas (harina → huevos y azúcar). El multiplicador que aplica el
generador no cambia respecto a la Fase 7a: `1 + (objetivo − 1) × 3,8`
(`AFFINITY_CALIBRATION`).

Desde la Fase 8, "lift objetivo" es la **fuerza nominal** del par, no el lift crudo que se
va a medir. Con misiones y cola larga de tamaño, el lift crudo sale por encima en las
categorías poco frecuentes que se concentran en la compra semanal, y por debajo en las
muy frecuentes (pan, leche), que ya están en media cesta. Lo que se exige a cada par es el
lift **controlado** por tamaño (tabla de objetivos de arriba), y
`python -m data_generation.verify_dataset` reporta los dos.

### Fidelidad de marca (detalle)

La **primera** compra de un cliente en una categoría le fija una referencia preferida
(sorteada entre las 8 de la categoría, con peso proporcional a su popularidad, y con el
uplift de promoción de ese día si lo hay). A partir de ahí, cada compra suya en esa
categoría se lleva esa misma referencia con probabilidad `Category.loyalty`, y el
`1 - loyalty` restante se reparte por popularidad entre el resto del surtido. La
preferencia no caduca ni se reasigna: la variedad sale del propio hueco de exploración.

La lealtad es **por categoría**, no global, y cae en una de dos bandas:

| Banda | Lealtad | Criterio |
| --- | ---: | --- |
| Hábito | 0,75 - 0,85 | El cliente compra "su" marca y cambiarla tiene coste: mismo café, mismo detergente, mismo champú, misma comida del perro. Droguería, higiene, bebé, mascotas, bebidas de marca y despensa envasada. |
| Exploración | 0,25 - 0,40 | Manda el producto concreto del día, no la marca: frescos que se eligen por aspecto o corte, caprichos que rotan por sabor y compras de ocasión. |

Asignación completa (el valor entre bandas gradúa cuánto: 0,85 para las categorías donde
cambiar de marca es más raro, 0,25 donde la elección es más libre):

| Departamento | Hábito (0,75-0,85) | Exploración (0,25-0,40) |
| --- | --- | --- |
| Frescos | Leche 0,80, Yogur 0,75, Huevos 0,78 | Queso 0,35, Pan 0,40, Embutido y fiambre 0,35, Fruta 0,25, Verdura 0,25, Carne de pollo 0,35, Carne de ternera 0,30, Pescado blanco 0,28, Torrijas y bollería de Cuaresma 0,30, Bacalao 0,30 |
| Despensa | Pasta 0,75, Salsa de tomate 0,78, Arroz 0,80, Legumbres 0,75, Conservas de pescado 0,75, Aceite de oliva 0,82, Café 0,85, Azúcar y edulcorante 0,82, Cereales 0,75, Harina 0,80, Sal y especias 0,80, Sopas y caldos 0,75 | Galletas 0,40, Snacks y aperitivos 0,25, Palomitas de microondas 0,35, Chocolate y huevos de Pascua 0,30, Turrón y mazapán 0,30 |
| Bebidas | Agua 0,85, Refrescos 0,85, Zumos 0,75, Cerveza 0,80 | Vino 0,30, Cava y espumosos 0,30 |
| Droguería | Detergente 0,85, Suavizante 0,82, Lavavajillas 0,80, Lejía y limpiadores 0,78, Limpiacristales 0,78, Bolsas de basura 0,80, Protector solar 0,75 | — |
| Higiene | Papel higiénico 0,82, Champú 0,85, Acondicionador 0,85, Gel de ducha 0,82, Pasta de dientes 0,85, Desodorante 0,85, Higiene femenina 0,85 | — |
| Bebé | Pañales 0,85, Toallitas húmedas 0,82, Leche infantil 0,85, Potitos 0,75 | — |
| Mascotas | Comida para perro 0,85, Comida para gato 0,85, Arena para gato 0,82 | — |
| Congelados | — | Verduras congeladas 0,40, Pizza congelada 0,35, Precocinados congelados 0,30, Helados 0,25, Marisco 0,30 |

40 categorías de hábito y 22 exploratorias. `validate_catalog()` falla si alguna lealtad
cae fuera de las dos bandas, para que no se cuele un valor intermedio sin criterio.

**Efecto medido** (`fidelidad_de_sku` de `python -m data_generation.verify_dataset`, sobre
los pares cliente-categoría con tres o más compras):

| Métrica | Antes (24 refs, sin fidelidad) | Ahora (8 refs + fidelidad) |
| --- | ---: | ---: |
| Referencias distintas por compra | 0,850 | **0,440** |
| Cuota de la referencia favorita | 0,301 | **0,697** |
| Referencias distintas por compra — hábito | 0,859 | 0,351 |
| Referencias distintas por compra — exploración | 0,837 | 0,571 |
| Cuota de la referencia favorita — hábito | 0,310 | 0,840 |

La cuota de la favorita en las categorías de hábito (0,840) aterriza dentro de la banda
declarada, que es la comprobación de que el parámetro hace lo que dice.

**Interacción con el uplift de promoción.** El uplift se aplica *encima* del reparto por
hábito, así que una promoción de la competencia solo puede llevarse la parte no fiel de la
categoría. El multiplicador sigue siendo exactamente `PROMO_UPLIFT` sobre esa parte, pero
el uplift **observado** a nivel de categoría baja: **1,97** frente al 2,99 que medía el
dataset anterior. Es lo que pasa en gran consumo — promocionar contra un hábito rinde
menos que promocionar donde nadie tiene marca fija —, sigue siendo señal de sobra para el
recomendador, y la cifra la recalcula `python -m data_generation.verify_dataset`.

### Estacionalidad (detalle)

Multiplicador aplicado a la probabilidad base de compra de la categoría durante el mes
pico. Fuera de esos meses, multiplicador = 1.0.

| Categoría                  | Mes(es) pico          | Multiplicador |
| ----------------------------- | ------------------------ | --------------- |
| Turrón / mazapán                | Diciembre                 | 8.0x            |
| Marisco (fresco/congelado)      | Diciembre                 | 3.0x            |
| Cava / espumoso                 | Diciembre                 | 5.0x            |
| Helados                          | Junio - Agosto             | 4.0x            |
| Protector solar (droguería)      | Junio - Agosto             | 6.0x            |
| Torrijas / bollería de Cuaresma  | Marzo - Abril               | 5.0x            |
| Bacalao                          | Marzo - Abril               | 3.0x            |
| Chocolate / huevos de Pascua      | Marzo - Abril               | 3.0x            |
| Sopas / caldos                    | Noviembre - Febrero          | 1.8x            |

### Embudo online (detalle)

`sessions` y `session_events` son el contexto de la "cesta en curso" de la Tarea 3a, y por
eso **no pueden ser el ticket escrito de otra forma**: si cada `add_to_cart` acabara en la
compra, la señal de sesión sería una copia del target y el recomendador daría métricas
falsas. El generador introduce las tres fugas que tiene un embudo real:

| Parámetro | Valor | Efecto medido |
| --- | ---: | --- |
| `session_item_browse_rate` | 0,85 | Solo el 85 % de las líneas del ticket pasa por la web (el resto entra por lista de la compra o en tienda) → `P(visto \| en la cesta) < 1` |
| `session_abandoned_adds` | 0,60 | Productos por sesión que se añaden al carrito y se quedan ahí → `P(en la cesta \| add_to_cart) = 87,9 %` |
| `session_view_only` | 2,50 | Productos por sesión que se miran y no se añaden → `P(en la cesta \| view) = 58,6 %` |
| `session_add_lag_s` | (20, 240) s | Cada `add_to_cart` va por detrás de su `view` |

El retardo entre ver y añadir es lo que hace la señal **utilizable** en vez de tautológica:
al cortar una sesión en un instante `t`, hay productos ya vistos y todavía no añadidos —
justo los que el recomendador tiene que adivinar. Medido sobre el dataset, en un corte a
mitad de cesta ~24 % de los productos que faltan por añadir ya han sido vistos, compitiendo
con ~1,6 productos vistos que nunca se añadirán.

Las sesiones que no convierten solo dejan vistas y algún `add_to_cart` suelto.

- **Churn progresivo**: para clientes marcados como `churn_label = 1`, reducir
  gradualmente frecuencia de compra y `total_amount` medio en las 6-8 semanas previas a su
  última compra, en vez de cortar en seco — así el patrón es aprendible por un modelo.
- **Calidad del dato**: ~1-2 % duplicados en `basket_items`, nulos en `customer_id` de
  `baskets` (compras anónimas legítimas, no descartar), categorías con mayúsculas/espacios
  inconsistentes, algunas `quantity` negativas (devoluciones mal codificadas) y outliers de
  `total_amount`.

## Volumen de referencia (ajustable)

Los de la ejecución con `seed = 42` y `scale = 1.0`, que es la que hay en `data/raw`.

| Tabla            | Filas |
| ------------------ | -------------- |
| `customers`         | 20.000         |
| `products`           | 496            |
| `promotions`         | 300            |
| `baskets`            | 600.174        |
| `basket_items`       | 4.357.007      |
| `sessions`           | 150.000        |
| `session_events`     | 1.106.225      |

`baskets` son 600.000 y no las 300.000 que decía esta tabla hasta la Fase 8 (punto B3 del
diagnóstico): con 20.000 clientes, 300.000 cestas son ~15 compras por cliente en dos años
—una visita cada ~73 días—, y a esa cadencia los ciclos de reposición no son observables.
`basket_items` pasa de 3,1 a 4,4 millones con las misiones de compra de la Fase 8.

## Tablas de `data/processed/` (Fase 2)

Todo lo anterior describe `data/raw/`, tal y como lo escribe el generador. La Fase 2 deja
en `data/processed/` una versión limpia de las 7 tablas más 4 tablas de features, en
Parquet, con `python -m src.etl.run_etl`. El porqué de cada corrección está en
`docs/CLEANING.md`.

Las tablas limpias **conservan todas las columnas de la tabla cruda** con el mismo nombre y
tipo; lo que se añade son columnas testigo (que dejan rastro de lo corregido) y columnas
derivadas de conveniencia. Ninguna columna cruda desaparece.

### Columnas añadidas a las tablas limpias

| Tabla            | Columna                    | Tipo    | Para qué                                                   |
| ----------------- | --------------------------- | -------- | ------------------------------------------------------------ |
| `customers`       | `signup_date_raw`           | date     | Alta original, antes de corregirla                           |
| `customers`       | `signup_date_corrected`     | boolean  | El alta era posterior a la primera compra                    |
| `customers`       | `city_is_missing`           | boolean  | La ciudad venía nula y se rellenó con `Desconocida`          |
| `customers`       | `first_purchase_date`       | date     | Primera compra observada (nula si nunca compró)              |
| `customers`       | `last_purchase_date`        | date     | Última compra observada                                      |
| `products`        | `category_raw`              | string   | Grafía original de la categoría, antes de normalizar         |
| `products`        | `brand_is_missing`          | boolean  | La marca venía nula y se rellenó con `Sin marca`             |
| `promotions`      | `date_range_invalid`        | boolean  | `start_date > end_date`                                      |
| `promotions`      | `duration_days`             | int      | Días de vigencia, extremos incluidos                         |
| `baskets`         | `basket_day`                | date     | `basket_date` truncada al día                                |
| `baskets`         | `total_amount_raw`          | double   | Importe declarado en la cabecera, antes de recalcular        |
| `baskets`         | `total_amount_is_outlier`   | boolean  | La cabecera superaba 1,5x el importe real de las líneas      |
| `baskets`         | `n_lines`                   | int      | Líneas del ticket tras limpiar                               |
| `baskets`         | `n_units`                   | int      | Unidades del ticket tras limpiar                             |
| `baskets`         | `is_anonymous`              | boolean  | Compra sin `customer_id` (legítima)                          |
| `basket_items`    | `line_amount`               | double   | `quantity * unit_price_paid`                                 |
| `basket_items`    | `is_promo`                  | boolean  | La línea llevaba promoción                                   |
| `basket_items`    | `quantity_sign_corrected`   | boolean  | La cantidad venía negativa y se corrigió el signo            |
| `sessions`        | `session_day`               | date     | `session_date` truncada al día                               |
| `sessions`        | `is_anonymous`              | boolean  | Visitante no identificado                                    |

`baskets.total_amount` deja de ser el valor crudo: pasa a ser el importe **recalculado**
desde `basket_items`, como pide la propia nota de la tabla cruda. El original sigue
disponible en `total_amount_raw`.

### `rfm` — una fila por cliente

| Columna                    | Tipo    | Notas                                                          |
| --------------------------- | -------- | ---------------------------------------------------------------- |
| `customer_id`               | string   | PK. Aparecen todos los clientes, hayan comprado o no             |
| `recency_days`              | int      | Días desde la última compra hasta `reference_date`               |
| `frequency`                 | int      | Cestas distintas del cliente                                     |
| `monetary`                  | double   | Suma de `total_amount` recalculado                               |
| `avg_ticket`                | double   | Importe medio por cesta                                          |
| `total_units`               | int      | Unidades compradas en total                                      |
| `first_purchase_date`       | date     | Primera compra                                                   |
| `last_purchase_date`        | date     | Última compra                                                    |
| `tenure_days`               | int      | Días desde la primera compra                                     |
| `avg_days_between_baskets`  | double   | Cadencia media de visita (nula con una sola compra)              |
| `reference_date`            | date     | Fecha de corte; por defecto el último día con compras            |
| `r_score` / `f_score` / `m_score` | int | Quintiles 1-5. Score alto = mejor en las tres                    |
| `rfm_score`                 | string   | Concatenación de los tres, p. ej. `555`                          |
| `rfm_segment`               | string   | Campeones / Fieles / Prometedores / Necesitan atención / En riesgo / Hibernando / Sin compras |
| `has_purchases`             | boolean  | False para clientes de alta sin ninguna compra                   |
| `loyalty_tier`, `signup_date`, `churn_label` | | Heredadas de `customers`                          |

### `repurchase_features` — una fila por cliente y categoría (Tarea 2)

| Columna                     | Tipo    | Notas                                                         |
| ---------------------------- | -------- | --------------------------------------------------------------- |
| `customer_id`, `category`    | string   | PK compuesta                                                     |
| `last_purchase_date`         | date     | Última compra de esa categoría por ese cliente                   |
| `first_purchase_date`        | date     | Primera compra de esa categoría                                  |
| `n_purchase_days`            | int      | Días distintos con compra (no líneas: dos cestas el mismo día cuentan una vez) |
| `observed_repurchase_days`   | double   | Media de días entre compras; nula con menos de `min_observations` |
| `stddev_gap_days`            | double   | Desviación de esos intervalos: mide lo regular que es el cliente |
| `typical_repurchase_days`    | int      | Intervalo típico de la categoría, desde `products`               |
| `household_factor`           | double   | Ajuste por tamaño de hogar (1,325 para hogar de 1; 0,70 para 6)  |
| `expected_repurchase_days`   | double   | El que se usa para decidir: observado si es fiable, si no el típico ajustado |
| `reference_date`             | date     | Fecha de corte                                                   |
| `days_since_last_purchase`   | int      | Días transcurridos                                               |
| `overdue_ratio`              | double   | `days_since / expected`. Por encima de 1 va con retraso           |
| `due_for_repurchase`         | boolean  | El flag de la Tarea 2                                            |

### `affinity_category` y `affinity_product` — una fila por par ordenado

`affinity_category` cruza categorías (es el grano de la tabla de afinidad de este mismo
documento); `affinity_product` cruza `product_id` y guarda sólo los 20 mejores consecuentes
por producto, que es lo que consumirá el generador de candidatos de la Fase 3.

| Columna                  | Tipo   | Notas                                                     |
| ------------------------- | ------- | ------------------------------------------------------------ |
| `antecedent`              | string  | Categoría o producto disparador                             |
| `consequent`              | string  | Categoría o producto asociado                               |
| `total_baskets`           | long    | Cestas consideradas en el cálculo                           |
| `n_baskets_antecedent`    | long    | Cestas que contienen el antecedente                         |
| `n_baskets_consequent`    | long    | Cestas que contienen el consecuente                         |
| `n_baskets_both`          | long    | Cestas que contienen los dos                                |
| `support`                 | double  | `n_both / total`                                            |
| `confidence`              | double  | `n_both / n_antecedent`, es decir P(consecuente si antecedente) |
| `lift`                    | double  | `confidence / (n_consequent / total)`. Es la métrica de la tabla de afinidad de arriba |
| `jaccard`                 | double  | `n_both / (n_antecedent + n_consequent − n_both)`           |
| `rank`                    | int     | Sólo en `affinity_product`: puesto del consecuente por lift  |

El par se guarda en las dos direcciones: `support`, `lift` y `jaccard` son simétricos,
`confidence` no.

## Oráculo del recomendador (`data/oracle/`, fuera del dataset)

No es una tabla del dataset: es información privilegiada que solo conoce el generador y
que ningún sistema real tendría. La escribe `python -m data_generation.export_oracle` y la
lee únicamente `src/recommender/oracle.py`, para calcular el techo teórico del
recomendador (punto A6 de `docs/diagnostico-fase7.md`). Vive fuera de `data/raw` y fuera
de git para que ninguna feature pueda leerla por error.

El exportador vuelve a ejecutar el generador con un registrador que no consume
aleatoriedad y comprueba que los sha256 de las 7 tablas coinciden con
`data/raw/manifest.json` antes de escribir nada: el dataset sale idéntico byte a byte con
o sin registrador.

`category_weights.parquet` — una fila por cesta con fecha `>= 2025-11-01` (la ventana de
test) y categoría con peso > 0:

| Columna | Tipo | Notas |
| --- | --- | --- |
| `basket_id` | string | FK a `baskets` |
| `category` | string | Nombre canónico del catálogo (el de `data/processed`) |
| `weight` | double | Peso de la categoría antes del primer sorteo de la cesta: afinidad del cliente × estacionalidad × ciclo de reposición, con los gates de hogar. Sin normalizar |
| `best_product_id` | string | Referencia más probable de la categoría en esa cesta (fidelidad de marca y promociones del día) |
| `best_product_prob` | double | Probabilidad de esa referencia si la categoría sale |

`manifest.json` guarda la semilla, la escala, la fecha de corte, el orden de las
categorías, los pares de afinidad con el lift **aplicado** y los hashes contra los que se
validó.

## Modelo dimensional para Power BI (Fase 5)

Star schema exportado a `reports/powerbi/` (Parquet o CSV) para que el proyecto Power BI lo
cargue vía Power Query, en local — sin conexión a ninguna base de datos en la nube.

### Dimensiones

| Tabla             | Columnas clave                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| `dim_customers`       | `customer_id` (PK), `loyalty_tier`, `household_size_est`, `preferred_channel`, `city`, `churn_label`     |
| `dim_products`         | `product_id` (PK), `department`, `category`, `brand`, `is_private_label`, `is_perishable`                |
| `dim_date`              | `date` (PK), `year`, `month`, `month_name`, `quarter`, `is_weekend` — para time intelligence               |
| `dim_actions`            | `action_id` (PK), `action_name`, `send_cost`, `discount`, `conversion_uplift`, `churn_reduction` — catálogo de acciones del NBA (Fase 4) |

### Hechos

| Tabla                    | Grano                                     | Columnas clave                                                                              |
| --------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `fact_basket_items`           | 1 fila por línea de ticket                     | `basket_id`, `customer_id` (FK), `product_id` (FK), `date_id` (FK), `quantity`, `unit_price_paid`, `is_promo` |
| `fact_recommendations`         | 1 fila por cesta de test evaluada               | `basket_id`, `customer_id` (FK), `customer_profile` (1-4), `ndcg_at_5`, `recall_at_5`                     |
| `fact_nba`                      | 1 fila por cliente y fecha de evaluación         | `customer_id` (FK), `date_id` (FK), `recommended_action_id` (FK a `dim_actions`), `category`, `p_purchase`, `p_churn`, `expected_value` |

Todas las FK son las columnas que usará el modelo tabular (TMDL) de Power BI para definir
las relaciones — no hace falta resolver los JOINs de antemano, eso lo hace el propio
semantic model.

`fact_nba` sale de `predictions/nba_actions.parquet`, que genera `python -m src.nba.pipeline`.
Dos precisiones sobre su esquema, que cambió al construir la Fase 4:

- **Hay dos propensiones, no una.** La Tarea 3b entrena dos modelos con horizontes
  distintos (`p_purchase` a 7 días sobre el par cliente-categoría, `p_churn` a 4 semanas
  sobre el cliente), y la política usa los dos: el primero para el valor de cross-sell y el
  segundo para el de retención. Colapsarlos en un único `propensity_score` perdería
  justamente la parte que hace no trivial a la política.
- **El coste de una acción se parte en dos.** `send_cost` se paga siempre que se ejecuta la
  acción; `discount` sólo si el cliente compra, porque es valor facial de cupón que se
  descuenta del ticket. Tratarlos como un único `cost` penalizaría de más a los clientes de
  baja propensión. `conversion_uplift` y `churn_reduction` son **supuestos declarados**, no
  estimaciones: este dataset no permite identificar el efecto causal de una acción (ver la
  cabecera de `src/nba/config.py`).

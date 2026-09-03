# Hallazgos de negocio

Ocho cosas que este dataset y estos modelos dicen sobre el negocio, y que cambian una
decision. No es el EDA pregunta a pregunta -- eso esta en
[`notebooks/01_eda.ipynb`](../../notebooks/01_eda.ipynb) -- sino lo que queda al cruzar el
dato de la Fase 2 con lo que hacen el recomendador (Fase 3) y la politica de Next Best
Action (Fase 4).

Lo escribe `python -m src.eda.findings`. Las consultas son las mismas funciones que usa el
notebook ([`src/eda/questions.py`](../../src/eda/questions.py), con tests en
[`tests/test_eda_questions.py`](../../tests/test_eda_questions.py)); los resultados de
modelo salen de `reports/*/metrics.json` y `predictions/*.parquet`. Ninguna cifra esta
tecleada a mano.

> Datos **100 % sinteticos**. Los "hallazgos" lo son sobre un supermercado simulado; lo
> que se puede llevar uno de aqui es el metodo, no las cifras.

---

## 1. El calendario mueve el surtido mas que el cliente

![Estacionalidad](01_estacionalidad.png)

9 categorias tienen un pico estacional claro. El mas fuerte es
**Turron y mazapan**, que en el mes 12 pesa
**4,45 veces** lo que pesa un mes cualquiera dentro de
la venta total. No es que en diciembre se venda mas de todo: el indice esta calculado sobre
la *cuota del mes*, precisamente para separar las dos cosas.

**Por que importa.** Un recomendador que solo mire el historico personal del cliente nunca
propondra turron a tiempo, porque nadie lo compro en noviembre. Es la razon de que la
fuente de candidatos "popularidad x indice estacional" de la Fase 3 sea la unica que cubre
al 100 % al cliente nuevo con la cesta vacia. Y es tambien, como se vera en el hallazgo 8,
lo que le falta al modelo de propension.

---

## 2. Lo que se compra junto no siempre es complementariedad

![Afinidad de cesta](02_afinidad.png)

Los 10 pares que el generador declara aparecen todos con lift por encima de 1
(10 de 10), pero el mas extremo se sale de escala:
**Panales -> Toallitas humedas** mide **13,39** frente a un
objetivo de 4,0.

La explicacion no es que la regla se aplicara mal, sino que **las dos categorias estan
restringidas a hogares con bebe**. El lift observado suma dos efectos: la complementariedad
real y la composicion de la clientela. Se ve claro en que todo el bloque de bebe -- leche
infantil, potitos -- sube al ranking sin que exista ninguna regla que lo una.

**Por que importa.** Para recomendar da igual: la senal es util venga de donde venga. Para
*decidir un surtido o un lineal* no da igual en absoluto, porque colocar toallitas al lado
de los panales no hara que las compre quien no tiene bebe. Es la diferencia entre una
correlacion que sirve para predecir y una que sirve para intervenir.

---

## 3. La palanca sobre un cliente valioso es la frecuencia, no el ticket

![Frecuencia frente a ticket](03_frecuencia_vs_ticket.png)

Los clientes `gold` son el **13,2 %** de la base y traen el
**21,7 %** de la facturacion, un indice de
1,65x. Lo interesante es de donde sale ese indice: el ticket medio
es practicamente identico en los tres niveles (29,17 EUR en gold
frente a 28,99 EUR en bronze). **La diferencia esta entera en la
frecuencia**: 48 compras frente a 23.

El tamano del hogar, en cambio, si mueve el ticket: de 23,77 EUR
en un hogar de una persona a 38,60 EUR en uno de
6.

**Por que importa.** Una campana que persiga subir el ticket medio de un cliente fiel esta
atacando la variable que no se mueve. Lo que se mueve es cuando vuelve, y eso es justamente
lo que hace accionable el ciclo de reposicion del hallazgo siguiente.

---

## 4. El ciclo de reposicion es real, y escala con el hogar

![Ciclo de recompra](04_recompra.png)

El ciclo observado reproduce el teorico con una correlacion de rangos de Spearman de
**0,978**, y se acorta de forma monotona al crecer el hogar en
las cuatro categorias dibujadas. Hoy, **52,4 %** de los
608.885 pares cliente-categoria tienen la recompra vencida, y eso alcanza
a 19.147 clientes.

La nube cae por debajo de la diagonal en los ciclos largos, y tiene explicacion: el
intervalo observado esta **truncado por la frecuencia de visita**. Nadie puede comprar
detergente cada 45 dias si solo pisa la tienda cada 60.

**Por que importa.** Es lo que convierte `due_for_repurchase` (Tarea 2) en una feature y no
en una corazonada, y lo que justifica escalar el ciclo esperado por `household_size_est` en
vez de usar un intervalo unico por categoria.

---

## 5. La sesion online no es el ticket, y por poco lo fue

![Embudo online](05_embudo.png)

150.000 sesiones, 35,0 % de conversion, y una
diferencia de solo 0,38 puntos entre el mejor y el peor
dispositivo: **el dispositivo, por si solo, no es una feature con senal**.

Lo que si la tiene es el panel de la derecha. De lo que se anade al carrito acaba en el
ticket el **87,9 %**; de lo que solo se mira, el
**58,6 %**; y en sentido contrario, solo el
**85,0 %** de las lineas del ticket dejo rastro online.

**Por que importa.** Las tres cifras estaban en el 100 % en la primera version del
generador, que emitia un `add_to_cart` por cada producto de la cesta y ninguno mas: la
sesion **era** el ticket escrito de otra forma. Usarla como feature habria dado un NDCG@5
espectacular y falso. Se arreglo el generador con abandono de carrito, productos que solo
se miran y un retardo entre ver y anadir. Es el hallazgo que mas trabajo ahorro: una fuga
de target encontrada antes de entrenar, no despues de presentar el resultado.

---

## 6. El recomendador sabe que necesitas; no sabe que referencia

![Recomendador por perfil](06_recomendador.png)

NDCG@5 = **0,0343** frente a 0,0200 del
baseline sin aprendizaje. El numero es bajo, y el panel de la derecha dice por que: el
sistema acierta la **categoria** en el **51,4 %** de las cestas y el **SKU**
solo en el **11,8 %**.

No es la primera etapa: el pool cubre ya el 31,1 % del target, y
ampliarlo de 92 a 155 candidatos por cesta no movio el NDCG. Es el dato: dentro de una
categoria hay unas 24 referencias y el generador elige casi al azar.

Por perfil, el peor es **2 - nuevo, con articulos** (NDCG@5 =
0,0149). Es el unico que no puede tirar ni de historial ni de ALS, y
encima su cesta ya va por la mitad, asi que lo facil de acertar ya esta dentro.

**Por que importa.** En gran consumo, acertar la categoria **es** util: si el cliente va a
comprar leche, recomendarle una leche sirve aunque no sea la referencia exacta. El proyecto
esta midiendo con la metrica mas dura de las dos y aun asi el techo esta en el dato. Darle
fidelidad de marca al generador es la deuda numero uno del [`ROADMAP.md`](../../ROADMAP.md).

---

## 7. La politica no responde a quien se va, sino a quien todavia vale algo

![NBA por segmento](07_nba_a_quien.png)

Este es el hallazgo menos intuitivo del proyecto. La correlacion entre el valor esperado de
la accion y la probabilidad de churn es **-0,63** -- **negativa** --
y con el gasto de los ultimos 90 dias, **+0,96**.

| Segmento RFM | clientes | P(churn) media | gasto 90d (EUR) | valor (EUR/cliente) | con accion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Campeones | 5.085 | 0,308 | 248 | 0,428 | 99,7 % |
| Fieles | 3.925 | 0,402 | 121 | 0,250 | 96,4 % |
| En riesgo | 3.112 | 0,471 | 57 | 0,127 | 58,5 % |
| Prometedores | 1.125 | 0,486 | 56 | 0,125 | 85,7 % |
| Necesitan atencion | 990 | 0,498 | 52 | 0,119 | 83,2 % |
| Hibernando | 4.492 | 0,549 | 19 | 0,045 | 37,9 % |

En el decil de mas riesgo de fuga la politica actua solo sobre el
**24 %** de los clientes y saca
0,007 EUR por cabeza; en el decil de menos riesgo actua sobre
todos y saca 0,727 EUR. El mejor segmento es
**Campeones** (0,428 EUR por cliente) y el peor,
**Hibernando** (0,045 EUR).

**Por que importa.** Un modelo de churn con AUC 0,85 invita a una conclusion que la
economia no sostiene: perseguir al que mas riesgo tiene. El valor de retener es
`P(churn) x valor_de_retener`, y en un cliente hibernado el segundo factor es casi cero,
porque no queda nada que salvar. **Un buen modelo de churn no es, por si solo, una politica
de retencion**: hace falta multiplicarlo por lo que el cliente todavia vale, y eso invierte
el orden de la lista.

---

## 8. La politica se va al margen, y sin calendario eso tiene un coste

![NBA por departamento](08_nba_margen.png)

Drogueria e Higiene son el **17,2 %** de la venta y se llevan el
**53,6 %** de las acciones. Frescos, que es mas de un tercio de
la venta, se queda en un indice de **0,23**.

| Departamento | de las acciones | de la venta | margen | indice |
| --- | ---: | ---: | ---: | ---: |
| Frescos | 8,4 % | 36,2 % | 18 % | 0,23 |
| Bebe | 2,4 % | 3,9 % | 20 % | 0,61 |
| Congelados | 6,0 % | 8,3 % | 22 % | 0,71 |
| Bebidas | 4,4 % | 9,8 % | 24 % | 0,45 |
| Despensa | 15,2 % | 19,6 % | 25 % | 0,78 |
| Mascotas | 9,9 % | 5,0 % | 30 % | 2,00 |
| Drogueria | 28,8 % | 7,7 % | 35 % | 3,75 |
| Higiene | 24,9 % | 9,5 % | 35 % | 2,62 |

Es aritmetica, no capricho: el cupon vale 2,54 EUR y hay que pagarlo con el margen del
ticket que provoque. Con un 18 % de margen en Frescos harian falta mas de 14 EUR de compra
solo para empatar; con un 35 % en Drogueria bastan 7,26 EUR.

**Y aqui aparece el punto ciego.** La categoria mas elegida por la politica es
**Protector solar** (1.848 veces), que en el mes del corte
(mes 11) vende **5,5 veces menos** que en su mes
pico (mes 6). El modelo de propension no tiene ni una feature de
calendario -- ni mes, ni indice estacional, ni nada -- asi que no puede descontar una
categoria de temporada fuera de temporada, y la economia (precio unitario alto por un 35 %
de margen) hace el resto.

**Por que importa.** Es un fallo barato de arreglar y caro de no ver: meter el indice
estacional de la Fase 2 en las features del modelo de propension. Queda anotado como deuda
en el [`ROADMAP.md`](../../ROADMAP.md).

---

## Lo que estos ocho tienen en comun

Cinco de los ocho son del mismo tipo: **una cifra que parecia buena o mala estaba midiendo
otra cosa**. El lift de 13 que era composicion de clientela; el solapamiento del 100 % que
era el target disfrazado; el NDCG bajo que era el techo del dato y no del modelo; el AUC
alto que no basta para una politica; la categoria mas recomendada, que estaba fuera de
temporada.

Ninguno se ve mirando la metrica sola. Todos aparecieron al preguntar **de donde sale este
numero** y encontrar que la respuesta no era la esperada.

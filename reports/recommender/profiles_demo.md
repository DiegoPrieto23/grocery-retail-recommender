# Los cuatro perfiles de cliente, con un caso real de cada uno

Generado por `python -m src.recommender.demo_profiles` sobre las cestas de test de
la Fase 3. Sirve para ver **la mecanica**: que fuentes se activan en cada perfil y
como el ranker las arbitra. Las metricas agregadas -- las cifras honestas -- estan en
`reports/recommender/metrics.md`; cada caso lleva ademas el `hit_rate` real de su
perfil al lado, para no confundir un ejemplo con un resultado.

## De donde sale cada recomendacion

Proporcion del top-5 que propuso cada fuente, por perfil. Es la tesis de la Tarea 3a
en una tabla: el ranker es el mismo en las cuatro filas, lo que cambia es quien tiene
algo que decir.

| Perfil | popularidad/estacionalidad | co-compra (producto) | co-compra (categoria) | historial + recompra | ALS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 - nuevo, carrito vacio | 91% | 0% | 0% | 21% | 0% |
| 2 - nuevo, con articulos | 88% | 14% | 29% | 19% | 0% |
| 3 - recurrente, carrito vacio | 51% | 0% | 0% | 99% | 39% |
| 4 - recurrente, con articulos | 49% | 12% | 15% | 99% | 38% |

---

### 1 · nuevo, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 39.5% de las 521 cestas._

- **Cesta**: `B0570723`, canal `app`, 2025-12-05
- **Cliente**: `C003414` - bronze, hogar de 4, Santander
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (7): Leche - Marca Blanca Basico (P00006); Cereales - Logística Higueras & Asociados S.C.P (P00170); Zumos - Carmela Dalmau Landa S.Coop. (P00258); Fruta - Inmobiliaria Luján S.Coop. (P00049); Verdura - Farmaceútica Revilla S.A. (P00060); Conservas de pescado - Marca Blanca Bio (P00139); Legumbres - Marca Blanca Bio (P00133)
- **Resultado**: NDCG@5 = 0.316, Recall@5 = 0.286 (2 de 7)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Cafe - Inversiones Sebastián S.Com. (P00155) | historial + recompra | no |
| 2 | Vino - Hervia y Medina S.L. (P00276) | historial + recompra | no |
| 3 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad | SI |
| 4 | Cereales - Logística Higueras & Asociados S.C.P (P00170) | historial + recompra | SI |
| 5 | Agua - Inmobiliaria RPIA S.L. (P00243) | popularidad/estacionalidad, en promocion | no |

---

### 2 · nuevo, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 24.0% de las 421 cestas._

- **Cesta**: `B0582622`, canal `web`, 2025-12-16
- **Cliente**: compra anonima (sin `customer_id`)
- **Ya en el carrito** (2): Snacks y aperitivos - Familia Julián S.A. (P00188); Fruta - Hervia y Medina S.L. (P00054)
- **Falta por anadir** (2): Leche - Marca Blanca Basico (P00006); Zumos - Marca Blanca Basico (P00259)
- **Resultado**: NDCG@5 = 0.387, Recall@5 = 0.500 (1 de 2)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Salsa de tomate - Jordán y asociados S.C.P (P00113) | popularidad/estacionalidad, en promocion | no |
| 2 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, co-compra (categoria) | SI |
| 3 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad | no |
| 4 | Verdura - Hnos Trillo S.A. (P00058) | popularidad/estacionalidad, co-compra (categoria) | no |
| 5 | Embutido y fiambre - Infraestructuras Carlos S.L.N.E (P00044) | popularidad/estacionalidad, co-compra (categoria) | no |

---

### 3 · recurrente, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 62.9% de las 8,713 cestas._

- **Cesta**: `B0554521`, canal `store`, 2025-11-17
- **Cliente**: `C010112` - silver, hogar de 1, Sevilla
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (5): Verdura - Comercial ZRYW S.L.L. (P00061); Agua - Giménez y Torrents S.A.T. (P00241); Aceite de oliva - Marca Blanca Bio (P00148); Leche - Marca Blanca Basico (P00006); Fruta - Familia Barrera S.Com. (P00050)
- **Resultado**: NDCG@5 = 0.339, Recall@5 = 0.200 (1 de 5)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, historial + recompra, ALS, en promocion | SI |
| 2 | Pasta - Fábrica KNI S.Coop. (P00106) | popularidad/estacionalidad, historial + recompra | no |
| 3 | Refrescos - Farmaceútica del Norte S.A. (P00254) | popularidad/estacionalidad, historial + recompra | no |
| 4 | Cafe - Banco BID S.L. (P00157) | historial + recompra, ALS | no |
| 5 | Gel de ducha - Pinilla y asociados S.L. (P00375) | historial + recompra | no |

---

### 4 · recurrente, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 49.0% de las 8,345 cestas._

- **Cesta**: `B0560043`, canal `store`, 2025-11-23
- **Cliente**: `C009788` - bronze, hogar de 3, Gijon
- **Ya en el carrito** (1): Legumbres - Infraestructuras Carlos S.L.N.E (P00135)
- **Falta por anadir** (2): Refrescos - Grupo Gómez S.A. (P00252); Palomitas de microondas - Cerdá y Serrano S.A. (P00197)
- **Resultado**: NDCG@5 = 0.387, Recall@5 = 0.500 (1 de 2)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Huevos - Construcción Inteligentes S.L.L. (P00026) | historial + recompra | no |
| 2 | Refrescos - Grupo Gómez S.A. (P00252) | popularidad/estacionalidad, historial + recompra, ALS | SI |
| 3 | Detergente - Marca Blanca Bio (P00293) | historial + recompra | no |
| 4 | Arroz - Marca Blanca Seleccion (P00127) | historial + recompra, en promocion | no |
| 5 | Cafe - Cerdá y Serrano S.A. (P00153) | historial + recompra | no |

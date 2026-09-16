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
| 1 - nuevo, carrito vacio | 100% | 0% | 0% | 0% | 0% |
| 2 - nuevo, con articulos | 92% | 20% | 33% | 0% | 0% |
| 3 - recurrente, carrito vacio | 57% | 0% | 0% | 99% | 42% |
| 4 - recurrente, con articulos | 56% | 15% | 17% | 98% | 41% |

---

### 1 · nuevo, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 37.2% de las 521 cestas._

- **Cesta**: `B0554100`, canal `store`, 2025-11-17
- **Cliente**: compra anonima (sin `customer_id`)
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (6): Leche - Marca Blanca Basico (P00006); Agua - Inmobiliaria RPIA S.L. (P00243); Verduras congeladas - Gonzalez y Yáñez S.Com. (P00461); Cerveza - Marca Blanca Basico (P00269); Panales - Marca Blanca Seleccion (P00406); Refrescos - Farmaceútica Vera & Asociados S.L.N.E (P00250)
- **Resultado**: NDCG@5 = 0.339, Recall@5 = 0.167 (1 de 6)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, en promocion | SI |
| 2 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad | no |
| 3 | Verdura - Hnos Trillo S.A. (P00058) | popularidad/estacionalidad | no |
| 4 | Chocolate y huevos de Pascua - Noemí Ramírez Falcón S.A. (P00222) | popularidad/estacionalidad, en promocion | no |
| 5 | Embutido y fiambre - Infraestructuras Carlos S.L.N.E (P00044) | popularidad/estacionalidad | no |

---

### 2 · nuevo, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 20.9% de las 421 cestas._

- **Cesta**: `B0562753`, canal `store`, 2025-11-27
- **Cliente**: compra anonima (sin `customer_id`)
- **Ya en el carrito** (1): Embutido y fiambre - Infraestructuras Carlos S.L.N.E (P00044)
- **Falta por anadir** (1): Chocolate y huevos de Pascua - Banca Privada OLMJ S.L.N.E (P00221)
- **Resultado**: NDCG@5 = 0.387, Recall@5 = 1.000 (1 de 1)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad, co-compra (producto), co-compra (categoria) | no |
| 2 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, co-compra (categoria), en promocion | no |
| 3 | Verdura - Hnos Trillo S.A. (P00058) | popularidad/estacionalidad, co-compra (categoria) | no |
| 4 | Pan - Belda & Asociados S.C.P (P00034) | popularidad/estacionalidad, co-compra (producto), co-compra (categoria) | no |
| 5 | Chocolate y huevos de Pascua - Banca Privada OLMJ S.L.N.E (P00221) | popularidad/estacionalidad, co-compra (categoria), en promocion | SI |

---

### 3 · recurrente, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 57.5% de las 8,713 cestas._

- **Cesta**: `B0595134`, canal `app`, 2025-12-26
- **Cliente**: `C019738` - bronze, hogar de 2, Valencia
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (6): Precocinados congelados - Consultoría del Norte S.A. (P00480); Yogur - Alimentación Villanueva & Asociados S.A. (P00014); Pasta - Fábrica KNI S.Coop. (P00106); Fruta - Hervia y Medina S.L. (P00054); Salsa de tomate - Jordán y asociados S.C.P (P00113); Pizza congelada - Restauración XRI S.Com. (P00472)
- **Resultado**: NDCG@5 = 0.316, Recall@5 = 0.333 (2 de 6)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Jordán y asociados S.C.P (P00001) | popularidad/estacionalidad, historial + recompra, ALS | no |
| 2 | Agua - Giménez y Torrents S.A.T. (P00241) | historial + recompra | no |
| 3 | Yogur - Alimentación Villanueva & Asociados S.A. (P00014) | popularidad/estacionalidad, historial + recompra, ALS | SI |
| 4 | Salsa de tomate - Jordán y asociados S.C.P (P00113) | popularidad/estacionalidad, historial + recompra, ALS, en promocion | SI |
| 5 | Refrescos - Pancho Mayoral Goñi S.A.T. (P00255) | historial + recompra | no |

---

### 4 · recurrente, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 43.3% de las 8,345 cestas._

- **Cesta**: `B0588915`, canal `store`, 2025-12-20
- **Cliente**: `C019768` - gold, hogar de 4, Madrid
- **Ya en el carrito** (4): Huevos - Marca Blanca Seleccion (P00031); Cafe - Transportes Bou S.L. (P00158); Champu - Sáenz y asociados S.Com. (P00360); Leche - Marca Blanca Basico (P00006)
- **Falta por anadir** (5): Cerveza - Infraestructuras Carlos S.L.N.E (P00268); Pan - Marca Blanca Bio (P00040); Verdura - Gonzalez y Yáñez S.Com. (P00063); Gel de ducha - Pinilla y asociados S.L. (P00375); Conservas de pescado - Marca Blanca Bio (P00139)
- **Resultado**: NDCG@5 = 0.360, Recall@5 = 0.400 (2 de 5)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Yogur - Alimentación Villanueva & Asociados S.A. (P00014) | popularidad/estacionalidad, historial + recompra | no |
| 2 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad, historial + recompra | SI |
| 3 | Refrescos - Farmaceútica del Norte S.A. (P00254) | popularidad/estacionalidad, historial + recompra | no |
| 4 | Cerveza - Infraestructuras Carlos S.L.N.E (P00268) | historial + recompra | SI |
| 5 | Zumos - Carmela Dalmau Landa S.Coop. (P00258) | historial + recompra | no |

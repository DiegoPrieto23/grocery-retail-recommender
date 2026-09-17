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
| 1 - nuevo, carrito vacio | 91% | 0% | 0% | 24% | 0% |
| 2 - nuevo, con articulos | 84% | 19% | 22% | 21% | 0% |
| 3 - recurrente, carrito vacio | 60% | 0% | 0% | 99% | 46% |
| 4 - recurrente, con articulos | 58% | 11% | 17% | 98% | 43% |

---

### 1 · nuevo, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 40.3% de las 521 cestas._

- **Cesta**: `B0554100`, canal `store`, 2025-11-17
- **Cliente**: compra anonima (sin `customer_id`)
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (6): Leche - Marca Blanca Basico (P00006); Agua - Inmobiliaria RPIA S.L. (P00243); Verduras congeladas - Gonzalez y Yáñez S.Com. (P00461); Cerveza - Marca Blanca Basico (P00269); Panales - Marca Blanca Seleccion (P00406); Refrescos - Farmaceútica Vera & Asociados S.L.N.E (P00250)
- **Resultado**: NDCG@5 = 0.339, Recall@5 = 0.167 (1 de 6)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, en promocion | SI |
| 2 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad | no |
| 3 | Cafe - Banco Segovia & Asociados S.L.N.E (P00156) | popularidad/estacionalidad | no |
| 4 | Verdura - Hnos Trillo S.A. (P00058) | popularidad/estacionalidad | no |
| 5 | Agua - Comercializadora VCRE S.A. (P00245) | popularidad/estacionalidad | no |

---

### 2 · nuevo, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 22.6% de las 421 cestas._

- **Cesta**: `B0590400`, canal `store`, 2025-12-23
- **Cliente**: compra anonima (sin `customer_id`)
- **Ya en el carrito** (1): Palomitas de microondas - Distribuciones FBQ S.L. (P00200)
- **Falta por anadir** (2): Cereales - Logística Higueras & Asociados S.C.P (P00170); Leche - Marca Blanca Basico (P00006)
- **Resultado**: NDCG@5 = 0.387, Recall@5 = 0.500 (1 de 2)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Refrescos - Grupo Gómez S.A. (P00252) | popularidad/estacionalidad, co-compra (producto), co-compra (categoria) | no |
| 2 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad | SI |
| 3 | Salsa de tomate - Jordán y asociados S.C.P (P00113) | popularidad/estacionalidad, en promocion | no |
| 4 | Fruta - Fábrica Navarro S.A. (P00053) | popularidad/estacionalidad, en promocion | no |
| 5 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad | no |

---

### 3 · recurrente, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 61.9% de las 8,713 cestas._

- **Cesta**: `B0558686`, canal `app`, 2025-11-22
- **Cliente**: `C000557` - gold, hogar de 2, Malaga
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (6): Leche - Marca Blanca Basico (P00006); Cereales - Farmaceútica EJ S.A.U (P00169); Pasta - Tecnologías Iberia S.L. (P00112); Legumbres - Marca Blanca Bio (P00130); Queso - Jose Francisco Sobrino Pinilla S.Com. (P00021); Embutido y fiambre - Marca Blanca Seleccion (P00048)
- **Resultado**: NDCG@5 = 0.339, Recall@5 = 0.167 (1 de 6)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, historial + recompra, ALS, en promocion | SI |
| 2 | Refrescos - Grupo Gómez S.A. (P00252) | popularidad/estacionalidad, historial + recompra, ALS | no |
| 3 | Agua - Iniesta y Bilbao S.Coop. (P00247) | historial + recompra | no |
| 4 | Huevos - Marca Blanca Seleccion (P00031) | historial + recompra, ALS | no |
| 5 | Embutido y fiambre - Infraestructuras Carlos S.L.N.E (P00044) | popularidad/estacionalidad, historial + recompra, ALS | no |

---

### 4 · recurrente, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 48.6% de las 8,345 cestas._

- **Cesta**: `B0571401`, canal `store`, 2025-12-05
- **Cliente**: `C012462` - gold, hogar de 3, Madrid
- **Ya en el carrito** (2): Carne de pollo - Infraestructuras Castellana S.L. (P00067); Gel de ducha - Marca Blanca Basico (P00369)
- **Falta por anadir** (2): Bolsas de basura - Banco Camps y asociados S.Com. (P00332); Agua - Giménez y Torrents S.A.T. (P00241)
- **Resultado**: NDCG@5 = 0.387, Recall@5 = 0.500 (1 de 2)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Huevos - Hotel Carrillo & Asociados S.Com. (P00032) | popularidad/estacionalidad, historial + recompra | no |
| 2 | Agua - Giménez y Torrents S.A.T. (P00241) | historial + recompra, ALS | SI |
| 3 | Verdura - Hnos Trillo S.A. (P00058) | popularidad/estacionalidad, historial + recompra, ALS | no |
| 4 | Papel higienico - Dalmau y Solsona S.L.N.E (P00345) | historial + recompra | no |
| 5 | Yogur - Alimentación Villanueva & Asociados S.A. (P00014) | popularidad/estacionalidad, historial + recompra | no |

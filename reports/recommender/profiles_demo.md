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
| 1 - nuevo, carrito vacio | 93% | 0% | 0% | 22% | 0% |
| 2 - nuevo, con articulos | 85% | 20% | 22% | 20% | 0% |
| 3 - recurrente, carrito vacio | 59% | 0% | 0% | 99% | 46% |
| 4 - recurrente, con articulos | 57% | 12% | 17% | 98% | 43% |

---

### 1 · nuevo, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 39.7% de las 521 cestas._

- **Cesta**: `B0551369`, canal `store`, 2025-11-14
- **Cliente**: compra anonima (sin `customer_id`)
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (5): Snacks y aperitivos - Marca Blanca Seleccion (P00190); Suavizante - Pancho Mayoral Goñi S.A.T. (P00304); Embutido y fiambre - Infraestructuras Carlos S.L.N.E (P00044); Yogur - Marca Blanca Bio (P00012); Leche - Marca Blanca Basico (P00006)
- **Resultado**: NDCG@5 = 0.339, Recall@5 = 0.200 (1 de 5)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, en promocion | SI |
| 2 | Agua - Comercializadora VCRE S.A. (P00245) | popularidad/estacionalidad, en promocion | no |
| 3 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad | no |
| 4 | Verdura - Hnos Trillo S.A. (P00058) | popularidad/estacionalidad | no |
| 5 | Snacks y aperitivos - Marca Blanca Basico (P00186) | popularidad/estacionalidad | no |

---

### 2 · nuevo, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 23.8% de las 421 cestas._

- **Cesta**: `B0563541`, canal `store`, 2025-11-27
- **Cliente**: `C017986` - bronze, hogar de 2, Gijon
- **Ya en el carrito** (2): Embutido y fiambre - Infraestructuras Carlos S.L.N.E (P00044); Protector solar - Marca Blanca Bio (P00341)
- **Falta por anadir** (2): Vino - Hervia y Medina S.L. (P00276); Leche - Marca Blanca Basico (P00006)
- **Resultado**: NDCG@5 = 0.387, Recall@5 = 0.500 (1 de 2)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad, co-compra (producto), co-compra (categoria) | no |
| 2 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, en promocion | SI |
| 3 | Agua - Inmobiliaria RPIA S.L. (P00243) | popularidad/estacionalidad, en promocion | no |
| 4 | Chocolate y huevos de Pascua - Banca Privada OLMJ S.L.N.E (P00221) | popularidad/estacionalidad, en promocion | no |
| 5 | Verdura - Hnos Trillo S.A. (P00058) | popularidad/estacionalidad | no |

---

### 3 · recurrente, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 61.9% de las 8,713 cestas._

- **Cesta**: `B0553851`, canal `app`, 2025-11-16
- **Cliente**: `C004433` - bronze, hogar de 5, Gijon
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (6): Helados - Fabián Falcó Martin S.A.T. (P00484); Verdura - Tecnologías Tejero y asociados S.Com. (P00057); Leche - Marca Blanca Basico (P00006); Pan - Belda & Asociados S.C.P (P00034); Embutido y fiambre - Infraestructuras Carlos S.L.N.E (P00044); Arroz - Comercializadora Donoso & Asociados S.A. (P00123)
- **Resultado**: NDCG@5 = 0.339, Recall@5 = 0.167 (1 de 6)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, historial + recompra, en promocion | SI |
| 2 | Refrescos - Grupo Gómez S.A. (P00252) | popularidad/estacionalidad, historial + recompra | no |
| 3 | Helados - Industrias EGVE S.Coop. (P00487) | historial + recompra | no |
| 4 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad, historial + recompra | no |
| 5 | Verdura - Soluciones RYD S.L.L. (P00064) | popularidad/estacionalidad, historial + recompra | no |

---

### 4 · recurrente, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 48.1% de las 8,345 cestas._

- **Cesta**: `B0568144`, canal `app`, 2025-12-02
- **Cliente**: `C014993` - silver, hogar de 2, Almeria
- **Ya en el carrito** (2): Arroz - Meléndez & Asociados S.A. (P00121); Chocolate y huevos de Pascua - Banca Privada OLMJ S.L.N.E (P00221)
- **Falta por anadir** (2): Carne de pollo - Farmaceútica del Norte S.A. (P00066); Panales - Perea y Ríos S.L.N.E (P00405)
- **Resultado**: NDCG@5 = 0.387, Recall@5 = 0.500 (1 de 2)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, co-compra (producto), historial + recompra, ALS | no |
| 2 | Panales - Perea y Ríos S.L.N.E (P00405) | historial + recompra, ALS | SI |
| 3 | Potitos - Comercializadora Donoso & Asociados S.A. (P00431) | historial + recompra, ALS, en promocion | no |
| 4 | Refrescos - Marca Blanca Basico (P00251) | historial + recompra, ALS | no |
| 5 | Agua - Inmobiliaria RPIA S.L. (P00243) | popularidad/estacionalidad, historial + recompra, ALS, en promocion | no |

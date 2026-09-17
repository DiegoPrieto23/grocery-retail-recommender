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
| 1 - nuevo, carrito vacio | 95% | 0% | 0% | 24% | 0% |
| 2 - nuevo, con articulos | 87% | 9% | 9% | 24% | 0% |
| 3 - recurrente, carrito vacio | 74% | 0% | 0% | 99% | 39% |
| 4 - recurrente, con articulos | 67% | 8% | 8% | 97% | 35% |

---

### 1 · nuevo, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 48.9% de las 554 cestas._

- **Cesta**: `B0581394`, canal `web`, 2025-12-14
- **Cliente**: `C010618` - silver, hogar de 3, Sevilla
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (6): Toallitas humedas - Hervia y Medina S.L. (P00414); Leche - Farmaceútica Vera & Asociados S.L.N.E (P00002); Potitos - Comercializadora Donoso & Asociados S.A. (P00431); Panales - Hermanos Tejedor S.A. (P00401); Pizza congelada - Soluciones RYD S.L.L. (P00467); Pan - Marca Blanca Seleccion (P00035)
- **Resultado**: NDCG@5 = 0.360, Recall@5 = 0.333 (2 de 6)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad, historial + recompra | no |
| 2 | Leche - Farmaceútica Vera & Asociados S.L.N.E (P00002) | popularidad/estacionalidad, historial + recompra | SI |
| 3 | Verdura - Hnos Trillo S.A. (P00058) | popularidad/estacionalidad | no |
| 4 | Toallitas humedas - Hervia y Medina S.L. (P00414) | historial + recompra | SI |
| 5 | Agua - Inmobiliaria RPIA S.L. (P00243) | popularidad/estacionalidad, historial + recompra | no |

---

### 2 · nuevo, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 43.0% de las 388 cestas._

- **Cesta**: `B0544278`, canal `store`, 2025-11-06
- **Cliente**: compra anonima (sin `customer_id`)
- **Ya en el carrito** (1): Refrescos - Grupo Gómez S.A. (P00252)
- **Falta por anadir** (1): Embutido y fiambre - Infraestructuras Carlos S.L.N.E (P00044)
- **Resultado**: NDCG@5 = 0.431, Recall@5 = 1.000 (1 de 1)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad | no |
| 2 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad | no |
| 3 | Verdura - Hnos Trillo S.A. (P00058) | popularidad/estacionalidad | no |
| 4 | Embutido y fiambre - Infraestructuras Carlos S.L.N.E (P00044) | popularidad/estacionalidad | SI |
| 5 | Yogur - Alimentación Villanueva & Asociados S.A. (P00014) | popularidad/estacionalidad | no |

---

### 3 · recurrente, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 64.2% de las 9,460 cestas._

- **Cesta**: `B0581042`, canal `web`, 2025-12-13
- **Cliente**: `C018326` - gold, hogar de 1, Sevilla
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (4): Detergente - Marca Blanca Bio (P00293); Suavizante - Marca Blanca Basico (P00299); Huevos - Inversiones Real y asociados S.Com. (P00025); Fruta - Marca Blanca Seleccion (P00051)
- **Resultado**: NDCG@5 = 0.414, Recall@5 = 0.500 (2 de 4)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Basico (P00006) | popularidad/estacionalidad, historial + recompra | no |
| 2 | Huevos - Inversiones Real y asociados S.Com. (P00025) | popularidad/estacionalidad, historial + recompra | SI |
| 3 | Pan - Marca Blanca Bio (P00040) | popularidad/estacionalidad, historial + recompra | no |
| 4 | Fruta - Marca Blanca Seleccion (P00051) | popularidad/estacionalidad, historial + recompra, ALS | SI |
| 5 | Yogur - Marca Blanca Basico (P00013) | historial + recompra | no |

---

### 4 · recurrente, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 60.2% de las 7,598 cestas._

- **Cesta**: `B0589394`, canal `store`, 2025-12-22
- **Cliente**: `C000300` - bronze, hogar de 3, Alicante
- **Ya en el carrito** (3): Legumbres - Farmaceútica del Norte S.A. (P00131); Verdura - Hnos Trillo S.A. (P00058); Fruta - Fábrica Navarro S.A. (P00053)
- **Falta por anadir** (3): Pan - Ani Criado Barberá S.L.N.E (P00033); Embutido y fiambre - Desarrollo Tudela & Asociados S.L.N.E (P00041); Agua - Comercializadora VCRE S.A. (P00245)
- **Resultado**: NDCG@5 = 0.437, Recall@5 = 0.667 (2 de 3)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Comercial Bru y asociados S.L.L. (P00004) | popularidad/estacionalidad, historial + recompra, ALS | no |
| 2 | Embutido y fiambre - Marca Blanca Seleccion (P00048) | popularidad/estacionalidad, historial + recompra, ALS | no |
| 3 | Pan - Ani Criado Barberá S.L.N.E (P00033) | popularidad/estacionalidad, historial + recompra, ALS | SI |
| 4 | Agua - Comercializadora VCRE S.A. (P00245) | popularidad/estacionalidad, historial + recompra | SI |
| 5 | Huevos - Inversiones Real y asociados S.Com. (P00025) | popularidad/estacionalidad, historial + recompra | no |

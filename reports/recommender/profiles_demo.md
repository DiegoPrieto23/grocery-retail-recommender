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
| 2 - nuevo, con articulos | 84% | 47% | 48% | 0% | 0% |
| 3 - recurrente, carrito vacio | 86% | 0% | 0% | 49% | 65% |
| 4 - recurrente, con articulos | 79% | 34% | 39% | 42% | 57% |

---

### 1 · nuevo, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 7.3% de las 521 cestas._

- **Cesta**: `B0584988`, canal `store`, 2025-12-18
- **Cliente**: `C004451` - bronze, hogar de 2, Palma
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (8): Higiene femenina - Marca Blanca Basico (P01194); Pan - Hermanos Tejedor S.A. (P00141); Embutido y fiambre - Desarrollo Tudela & Asociados S.L.N.E (P00175); Fruta - Restauración del Sur S.Com. (P00231); Galletas - Construcción Inteligentes S.L.L. (P00624); Conservas de pescado - Escobar y asociados S.A. (P00488); Arroz - Victorino Alemany Buendía S.Com. (P00430); Pescado blanco - Hermanos Madrid S.A.T. (P00327)
- **Resultado**: NDCG@5 = 0.214, Recall@5 = 0.125 (1 de 8)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Snacks y aperitivos - Marca Blanca Seleccion (P00645) | popularidad/estacionalidad | no |
| 2 | Galletas - Construcción Inteligentes S.L.L. (P00624) | popularidad/estacionalidad | SI |
| 3 | Leche - Finanzas Globales S.A. (P00008) | popularidad/estacionalidad | no |
| 4 | Lavavajillas - Comercial Bru y asociados S.L.L. (P00973) | popularidad/estacionalidad | no |
| 5 | Leche - Marca Blanca Bio (P00018) | popularidad/estacionalidad | no |

---

### 2 · nuevo, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 4.5% de las 421 cestas._

- **Cesta**: `B0594066`, canal `web`, 2025-12-26
- **Cliente**: `C005383` - silver, hogar de 1, Barcelona
- **Ya en el carrito** (2): Agua - Caballero y Palomar S.L.L. (P00794); Azucar y edulcorante - Segura & Asociados S.Com. (P00566)
- **Falta por anadir** (2): Embutido y fiambre - Desarrollo Tudela & Asociados S.L.N.E (P00175); Leche - Finanzas Globales S.A. (P00008)
- **Resultado**: NDCG@5 = 0.264, Recall@5 = 0.500 (1 de 2)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Cafe - Servicios DGC S.A. (P00522) | popularidad/estacionalidad, co-compra (producto), co-compra (categoria) | no |
| 2 | Chocolate y huevos de Pascua - Carmela Dalmau Landa S.Coop. (P00724) | popularidad/estacionalidad, co-compra (producto) | no |
| 3 | Cafe - Hermanos Tejedor S.A. (P00527) | popularidad/estacionalidad, co-compra (producto), co-compra (categoria) | no |
| 4 | Leche - Finanzas Globales S.A. (P00008) | popularidad/estacionalidad, co-compra (producto) | SI |
| 5 | Cafe - Grupo Antón S.L. (P00548) | co-compra (producto), co-compra (categoria) | no |

---

### 3 · recurrente, carrito vacio

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 14.1% de las 8,713 cestas._

- **Cesta**: `B0555118`, canal `web`, 2025-11-18
- **Cliente**: `C003736` - silver, hogar de 2, Palma
- **Ya en el carrito** (0): _vacio_
- **Falta por anadir** (6): Acondicionador - Distribuciones FBQ S.L. (P01113); Cerveza - Fabián Falcó Martin S.A.T. (P00883); Galletas - Construcción Inteligentes S.L.L. (P00624); Helados - Restauración SL S.L.L. (P01479); Pescado blanco - Marca Blanca Seleccion (P00342); Verdura - Ortiz y asociados S.C.P (P00272)
- **Resultado**: NDCG@5 = 0.214, Recall@5 = 0.167 (1 de 6)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Leche - Marca Blanca Bio (P00018) | popularidad/estacionalidad, ALS, en promocion | no |
| 2 | Galletas - Construcción Inteligentes S.L.L. (P00624) | popularidad/estacionalidad, ALS | SI |
| 3 | Pasta - Desarrollo Tudela & Asociados S.L.N.E (P00391) | popularidad/estacionalidad, historial + recompra, en promocion | no |
| 4 | Salsa de tomate - Hnos Amores S.A. (P00411) | popularidad/estacionalidad | no |
| 5 | Embutido y fiambre - Compañía Hurtado & Asociados S.Coop. (P00182) | popularidad/estacionalidad | no |

---

### 4 · recurrente, con articulos

_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto del perfil acierta al menos un producto el 10.1% de las 8,345 cestas._

- **Cesta**: `B0593936`, canal `app`, 2025-12-26
- **Cliente**: `C003759` - silver, hogar de 2, Sevilla
- **Ya en el carrito** (3): Cereales - Marca Blanca Basico (P00581); Agua - Familia Barrera S.Com. (P00768); Yogur - Marca Blanca Seleccion (P00049)
- **Falta por anadir** (3): Verdura - Transportes Haro y asociados S.Com. (P00269); Cafe - Belda & Asociados S.C.P (P00542); Embutido y fiambre - Compañía Hurtado & Asociados S.Coop. (P00182)
- **Resultado**: NDCG@5 = 0.296, Recall@5 = 0.333 (1 de 3)

| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |
| ---: | --- | --- | :---: |
| 1 | Embutido y fiambre - Instalaciones Cabrera & Asociados S.Com. (P00180) | popularidad/estacionalidad, co-compra (categoria), historial + recompra, ALS | no |
| 2 | Embutido y fiambre - Compañía Hurtado & Asociados S.Coop. (P00182) | popularidad/estacionalidad, co-compra (categoria), historial + recompra, ALS | SI |
| 3 | Fruta - Grupo Giménez S.A. (P00202) | popularidad/estacionalidad, co-compra (categoria), historial + recompra, ALS | no |
| 4 | Verdura - Alimentación Campillo S.L. (P00253) | popularidad/estacionalidad, co-compra (categoria), ALS | no |
| 5 | Leche - Marca Blanca Bio (P00018) | popularidad/estacionalidad, co-compra (producto), co-compra (categoria), ALS | no |

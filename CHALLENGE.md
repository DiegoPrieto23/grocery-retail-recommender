# Grocery Retail — Recommender & Next Best Action

## 🌐 Background

Un supermercado online quiere personalizar la experiencia de compra: recomendar productos
durante la cesta en curso y decidir, cliente a cliente, cuál es la siguiente acción de
negocio con más valor esperado (recomendar, ofrecer un cupón, o no actuar). A diferencia de
un e-commerce de moda, aquí la unidad de análisis no es una sesión de navegación con un
único producto de interés, sino una **cesta** con varios artículos, con fuerte estacionalidad,
ciclos de reposición predecibles (la leche se repite cada pocos días, el detergente cada
mes y medio) y una señal de churn que se puede leer en la caída progresiva de frecuencia y
ticket antes del abandono.

El reto (autoimpuesto) tiene tres partes: generar el dato sintético, construir un
recomendador de cesta, y construir una capa de Next Best Action sobre un modelo de
propensión.

Todos los datos son **100 % sintéticos**. No contienen información real de clientes,
productos ni tiendas.

---

## 🗂️ Dataset

El dataset no se descarga: lo genera `data_generation/generate_dataset.py`, con semilla fija
para que sea reproducible. El detalle columna a columna vive en `DATA_SPEC.md`; aquí solo el
resumen de tablas y para qué sirve cada una.

| Tabla            | Grano                              | Para qué sirve                                      |
| ----------------- | ----------------------------------- | ----------------------------------------------------- |
| `customers`       | 1 fila por cliente                  | Segmentación, RFM, target de churn                    |
| `products`        | 1 fila por producto                 | Contenido para el recomendador (departamento, marca…) |
| `promotions`       | 1 fila por promoción activa/pasada  | Uplift de campañas, feature para NBA                  |
| `baskets`          | 1 fila por ticket (online o tienda) | Cabecera de la transacción                            |
| `basket_items`     | 1 fila por línea de ticket          | Afinidad de cesta, entrenamiento del recomendador      |
| `sessions`         | 1 fila por sesión online            | Contexto de una cesta en curso                        |
| `session_events`   | 1 fila por evento dentro de sesión  | Vistas / add-to-cart antes de completar la cesta       |

### Reglas de negocio que codifica el generador

- **Ciclos de reposición** por categoría y tamaño de hogar (ej. leche cada 5-7 días,
  papel higiénico cada ~30, detergente cada ~45).
- **Afinidad de cesta**: pares/tríos de categorías que aparecen juntos con más frecuencia
  de la esperada por azar (cerveza + snacks, pasta + salsa, pañales + toallitas).
- **Estacionalidad**: Navidad (turrón, marisco), verano (helados, protector solar),
  Semana Santa (torrijas, bacalao).
- **Uplift de promoción**: un producto en promoción tiene más probabilidad de entrar en
  la cesta que el mismo producto sin promocionar.
- **Señal de churn**: antes de un abandono, frecuencia de compra y ticket medio decaen de
  forma progresiva en las N semanas previas — es el target del modelo de propensión.
- **Calidad del dato (problemas deliberados)**: duplicados, nulos, categorías mal escritas,
  cantidades negativas, fechas inconsistentes, outliers de importe — igual de espíritu que
  en `sports-rental-analytics`, para practicar limpieza real.

---

## 🎯 Tareas

### Tarea 1 — Exploración y calidad de dato

Preguntas de negocio sobre `customers`, `products` y `basket_items` (equivalente a las Q1–Q7
del hackathon de Inditex, pero sobre este dominio), más una función que calcule el
**Data Trust Score** del dataset generado, reutilizando el enfoque de `sports-rental-analytics`.

El resultado se documenta en un **notebook de EDA** (`notebooks/`) que resuelve cada
pregunta con **Spark SQL** (vistas temporales + `spark.sql(...)`) acompañada de una
**visualización** (matplotlib/seaborn/plotly) — una consulta y un gráfico por pregunta, no
solo una tabla de números.

### Tarea 2 — Función de features

Dada una tabla con el formato de `basket_items` + `customers`, devolver por cliente y
categoría: última fecha de compra, frecuencia media de recompra observada, y un flag
`due_for_repurchase` (si ya ha pasado el intervalo típico de recompra de esa categoría para
ese cliente). Esta función alimenta tanto al recomendador (repescar productos "tocan") como
al NBA.

### Tarea 3a — Recomendador de cesta

Dada una cesta en curso (0 o más artículos ya añadidos) y un `customer_id` que puede o no
tener historial, recomendar **5 productos**. Arquitectura de **dos etapas**, como en los
recomendadores de retail reales — evita entrenar un modelo end-to-end sobre todo el
catálogo y es ligero en cómputo (sin deep learning, sin GPU):

1. **Generación de candidatos** (varias fuentes, cada una aporta un pool):
   - Popularidad/estacionalidad (fallback, siempre disponible)
   - Co-compra / afinidad de cesta (tabla de la Fase 2)
   - ALS (Spark MLlib) sobre `customer_id × product_id`, señal colaborativa personalizada
2. **Ranking**: un **LightGBM** con objetivo de ranking (`LambdaRank`) reordena el pool de
   candidatos usando features (score de cada fuente, recency/frequency del cliente con esa
   categoría, si el producto está en promoción, popularidad reciente, señal de sesión si la
   hay) y devuelve el top-5 final.

Cubrir los mismos cuatro perfiles que el reto de Inditex, adaptados — lo que cambia entre
perfiles es qué fuentes de candidatos tienen señal disponible, no el ranker, que es el mismo
para los cuatro:

1. Cliente nuevo, sin artículos aún en la cesta → candidatos de popularidad/estacionalidad.
2. Cliente nuevo, con algún artículo ya en la cesta → candidatos de co-compra.
3. Cliente recurrente, sin artículos aún en la cesta → candidatos del historial personal +
   productos "due for repurchase".
4. Cliente recurrente, con artículos en la cesta → combinación de las tres fuentes.

**Métrica:** NDCG@5 y Recall@5 sobre cestas reales ocultadas en el split de test (holdout
por `basket_id`, nunca visto en entrenamiento).

### Tarea 3b — Next Best Action

Un modelo de **propensión** (LightGBM o logística) estima, por cliente, la probabilidad de:
(a) comprar en una categoría concreta en los próximos 7 días, y (b) hacer churn en las
próximas 4 semanas. Sobre esas probabilidades, una **política de valor esperado** simple
decide la mejor acción de un catálogo fijo:

```
acción*  =  argmax_a  ( P(conversión | acción=a) × margen_esperado − coste(a) )
```

Catálogo de acciones de ejemplo: `recomendar_producto`, `enviar_cupón_categoría`,
`ninguna_acción`. El resultado es una tabla `customer_id → acción recomendada → valor
esperado`.

**Métrica:** AUC / PR-AUC del modelo de propensión, y una evaluación tipo negocio (uplift de
valor esperado de la política frente a "no actuar" o "actuar siempre"), sobre un periodo de
test simulado.

---

### Tarea 4 — Storytelling y demo

Más allá de los notebooks y las métricas, el proyecto se cierra con tres piezas orientadas
a comunicar el trabajo — ninguna reentrena nada, todas consumen lo ya construido en las
Tareas 1-3:

- **Resumen de impacto de negocio** (medio folio, no un proyecto aparte): traducir NDCG@5 y
  el uplift del NBA a impacto estimado — por ejemplo, si el recomendador sube el cross-sell
  un X %, sobre Y cestas/mes son Z € extra.
- **Preparación visual del catálogo** (paso previo, una sola vez, con conexión a internet):
  analizar `products` y agrupar los productos en `visual_group` — categorías visuales
  reutilizables, ni tan amplias como el departamento ni tan específicas como el SKU (p.ej.
  `leche_entera`, `yogur_griego`, `salmón`), buscar y descargar vía la API de Pexels una
  foto real representativa de cada `visual_group` (fondo limpio, estilo ecommerce, sin
  personas ni composiciones complejas), y generar el mapeo `product_id → visual_group →
  image_path`. Es la única parte del proyecto que sale del entorno 100 % offline/sintético
  — se ejecuta una vez, las imágenes se cachean en `assets/` y a partir de ahí la demo
  vuelve a ser local.
- **Demo web interactiva** (en local, Streamlit): simulación de cesta de la compra donde, al
  añadir productos, aparecen en vivo las recomendaciones del sistema (Tarea 3a) y la próxima
  mejor acción para ese cliente (Tarea 3b), cubriendo los 4 perfiles de cliente de forma
  interactiva. Es el entregable central de esta tarea: gráfica e intuitiva, con los
  productos mostrados como tarjetas con la foto real de su `visual_group` (de la
  preparación anterior) — sin iconos, sin SVGs, sin imágenes generadas por IA.

Un dashboard en Power BI estuvo planteado aquí como pieza de BI Engineering, pero queda
aparcado por ahora a favor de dedicar el esfuerzo a la demo — ver "Fuera de alcance" en
`ROADMAP.md`.

---

## 📂 Estructura de repositorio (objetivo)

```
grocery-retail-recommender/
├── data_generation/
│   └── generate_dataset.py
├── data/{raw,processed}/
├── src/
│   ├── etl/                # PySpark: limpieza, RFM, ciclo de reposición
│   ├── recommender/        # candidatos (co-compra/ALS) + ranking + evaluación
│   └── nba/                # modelo de propensión + política de acción
├── notebooks/               # exploración y EDA puntual
├── models/  predictions/
├── tests/
├── CHALLENGE.md             # este documento
├── DATA_SPEC.md             # esquema detallado del dataset
├── ROADMAP.md               # plan de trabajo por fases
├── CLAUDE.md                # convenciones para trabajar con Claude Code
└── README.md
```

---

## 💫 Guías

- Stack: **PySpark** para ETL y feature engineering, **MLlib / scikit-learn / LightGBM**
  para modelado — sin dbt esta vez, para variar de stack frente a los otros dos proyectos.
- Reproducibilidad: semilla fija en el generador (como en `sports-rental-analytics`).
- Todo el dato es sintético; ningún dato real de clientes.

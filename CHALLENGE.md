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

### Tarea 2 — Función de features

Dada una tabla con el formato de `basket_items` + `customers`, devolver por cliente y
categoría: última fecha de compra, frecuencia media de recompra observada, y un flag
`due_for_repurchase` (si ya ha pasado el intervalo típico de recompra de esa categoría para
ese cliente). Esta función alimenta tanto al recomendador (repescar productos "tocan") como
al NBA.

### Tarea 3a — Recomendador de cesta

Dada una cesta en curso (0 o más artículos ya añadidos) y un `customer_id` que puede o no
tener historial, recomendar **5 productos**. Cubrir los mismos cuatro perfiles que el reto
de Inditex, adaptados:

1. Cliente nuevo, sin artículos aún en la cesta → fallback por popularidad/estacionalidad.
2. Cliente nuevo, con algún artículo ya en la cesta → afinidad de cesta (co-compra).
3. Cliente recurrente, sin artículos aún en la cesta → historial personal + repescar
   productos "due for repurchase".
4. Cliente recurrente, con artículos en la cesta → combinar afinidad de cesta con
   preferencias históricas.

**Métrica:** NDCG@5 y Recall@5 sobre cestas reales ocultadas en el split de test.

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

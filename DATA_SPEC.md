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
- **Churn progresivo**: para clientes marcados como `churn_label = 1`, reducir
  gradualmente frecuencia de compra y `total_amount` medio en las 6-8 semanas previas a su
  última compra, en vez de cortar en seco — así el patrón es aprendible por un modelo.
- **Calidad del dato**: ~1-2 % duplicados en `basket_items`, nulos en `customer_id` de
  `baskets` (compras anónimas legítimas, no descartar), categorías con mayúsculas/espacios
  inconsistentes, algunas `quantity` negativas (devoluciones mal codificadas) y outliers de
  `total_amount`.

## Volumen de referencia (ajustable)

| Tabla            | Filas aprox. |
| ------------------ | -------------- |
| `customers`         | 20.000         |
| `products`           | 1.500          |
| `promotions`         | 300            |
| `baskets`            | 300.000        |
| `basket_items`       | ~1.500.000     |
| `sessions`           | 150.000        |
| `session_events`     | ~900.000       |

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

## Modelo dimensional para Power BI (Fase 5)

Star schema exportado a `reports/powerbi/` (Parquet o CSV) para que el proyecto Power BI lo
cargue vía Power Query, en local — sin conexión a ninguna base de datos en la nube.

### Dimensiones

| Tabla             | Columnas clave                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| `dim_customers`       | `customer_id` (PK), `loyalty_tier`, `household_size_est`, `preferred_channel`, `city`, `churn_label`     |
| `dim_products`         | `product_id` (PK), `department`, `category`, `brand`, `is_private_label`, `is_perishable`                |
| `dim_date`              | `date` (PK), `year`, `month`, `month_name`, `quarter`, `is_weekend` — para time intelligence               |
| `dim_actions`            | `action_id` (PK), `action_name`, `cost`, `expected_margin` — catálogo de acciones del NBA                   |

### Hechos

| Tabla                    | Grano                                     | Columnas clave                                                                              |
| --------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `fact_basket_items`           | 1 fila por línea de ticket                     | `basket_id`, `customer_id` (FK), `product_id` (FK), `date_id` (FK), `quantity`, `unit_price_paid`, `is_promo` |
| `fact_recommendations`         | 1 fila por cesta de test evaluada               | `basket_id`, `customer_id` (FK), `customer_profile` (1-4), `ndcg_at_5`, `recall_at_5`                     |
| `fact_nba`                      | 1 fila por cliente y fecha de evaluación         | `customer_id` (FK), `date_id` (FK), `recommended_action_id` (FK a `dim_actions`), `propensity_score`, `expected_value` |

Todas las FK son las columnas que usará el modelo tabular (TMDL) de Power BI para definir
las relaciones — no hace falta resolver los JOINs de antemano, eso lo hace el propio
semantic model.

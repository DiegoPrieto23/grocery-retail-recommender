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

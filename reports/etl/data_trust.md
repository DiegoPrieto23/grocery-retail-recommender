# Data Trust Score (Tarea 1)

| Dataset | Score | Nota |
| --- | ---: | :---: |
| `data/raw` (crudo) | 89.99 | C |
| `data/processed` (limpio) | 100.00 | A |

## Antes de limpiar

**Data Trust Score: 89.99 / 100 (nota C)** — 69 de 79 comprobaciones sin una sola fila mala.

| Dimension | Puntuacion | Tasa de filas | Tasa de reglas | Reglas KO |
| --- | ---: | ---: | ---: | ---: |
| `completeness` | 97.41 | 99.95 % | 94.87 % | 2 / 39 |
| `uniqueness` | 81.07 | 99.63 % | 62.50 % | 3 / 8 |
| `validity` | 93.61 | 99.73 % | 87.50 % | 2 / 16 |
| `consistency` | 77.88 | 98.62 % | 57.14 % | 3 / 7 |
| `integrity` | 100.00 | 100.00 % | 100.00 % | 0 / 9 |

| Tabla | Dimension | Comprobacion | Filas KO | Total | % OK |
| --- | --- | --- | ---: | ---: | ---: |
| `baskets` | consistency | `total_amount` cuadra con la suma de sus lineas | 56.340 | 600.174 | 90.613 % |
| `products` | validity | `category` sin mayusculas ni espacios inconsistentes | 60 | 1.500 | 96.000 % |
| `basket_items` | uniqueness | Una sola linea por cesta y producto | 45.859 | 3.103.684 | 98.522 % |
| `basket_items` | uniqueness | Sin filas duplicadas exactas | 45.506 | 3.103.684 | 98.534 % |
| `products` | completeness | `brand` informada | 17 | 1.500 | 98.867 % |
| `customers` | completeness | `city` informada | 163 | 20.000 | 99.185 % |
| `basket_items` | validity | `quantity` mayor que 0 | 12.308 | 3.103.684 | 99.603 % |
| `customers` | consistency | `signup_date` anterior o igual a la primera compra | 58 | 20.000 | 99.710 % |
| `customers` | consistency | `signup_date` no posterior al fin del periodo observado | 3 | 20.000 | 99.985 % |
| `session_events` | uniqueness | Sin filas duplicadas exactas | 25 | 945.676 | 99.997 % |
| `basket_items` | integrity | `basket_id` existe en `basket_id` | 0 | 3.103.684 | 100.000 % |
| `basket_items` | completeness | `basket_id` informada | 0 | 3.103.684 | 100.000 % |
| `basket_items` | integrity | `product_id` existe en `product_id` | 0 | 3.103.684 | 100.000 % |
| `basket_items` | completeness | `product_id` informada | 0 | 3.103.684 | 100.000 % |
| `basket_items` | consistency | La promocion aplicada estaba vigente el dia de la cesta | 0 | 83.214 | 100.000 % |
| `basket_items` | integrity | `promotion_id` existe en `promotion_id` | 0 | 83.214 | 100.000 % |
| `basket_items` | completeness | `quantity` informada | 0 | 3.103.684 | 100.000 % |
| `basket_items` | completeness | `unit_price_paid` informada | 0 | 3.103.684 | 100.000 % |
| `basket_items` | validity | `unit_price_paid` mayor que 0 | 0 | 3.103.684 | 100.000 % |
| `baskets` | completeness | `basket_date` informada | 0 | 600.174 | 100.000 % |
| `baskets` | completeness | `basket_id` informada | 0 | 600.174 | 100.000 % |
| `baskets` | uniqueness | `basket_id` sin repetir | 0 | 600.174 | 100.000 % |
| `baskets` | validity | `channel` en app/web/store | 0 | 600.174 | 100.000 % |
| `baskets` | completeness | `channel` informada | 0 | 600.174 | 100.000 % |
| `baskets` | integrity | `customer_id` existe en `customer_id` | 0 | 582.174 | 100.000 % |
| `baskets` | consistency | `store_id` informado si y solo si el canal es `store` | 0 | 600.174 | 100.000 % |
| `baskets` | completeness | `total_amount` informada | 0 | 600.174 | 100.000 % |
| `baskets` | validity | `total_amount` mayor que 0 | 0 | 600.174 | 100.000 % |
| `customers` | completeness | `churn_label` informada | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `country` informada | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `customer_id` informada | 0 | 20.000 | 100.000 % |
| `customers` | uniqueness | `customer_id` sin repetir | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `household_size_est` informada | 0 | 20.000 | 100.000 % |
| `customers` | validity | `household_size_est` entre 1 y 6 | 0 | 20.000 | 100.000 % |
| `customers` | validity | `loyalty_tier` en bronze/silver/gold | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `loyalty_tier` informada | 0 | 20.000 | 100.000 % |
| `customers` | validity | `preferred_channel` en app/web/store | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `preferred_channel` informada | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `signup_date` informada | 0 | 20.000 | 100.000 % |
| `products` | completeness | `category` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `department` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `is_perishable` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `is_private_label` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `pack_size` informada | 0 | 1.500 | 100.000 % |
| `products` | validity | `pack_size` mayor que 0 | 0 | 1.500 | 100.000 % |
| `products` | completeness | `product_id` informada | 0 | 1.500 | 100.000 % |
| `products` | uniqueness | `product_id` sin repetir | 0 | 1.500 | 100.000 % |
| `products` | validity | `typical_repurchase_days` mayor que 0 | 0 | 1.500 | 100.000 % |
| `products` | completeness | `typical_repurchase_days` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `unit_price` informada | 0 | 1.500 | 100.000 % |
| `products` | validity | `unit_price` mayor que 0 | 0 | 1.500 | 100.000 % |
| `promotions` | validity | `start_date` anterior o igual a `end_date` | 0 | 300 | 100.000 % |
| `promotions` | completeness | `discount_value` informada | 0 | 300 | 100.000 % |
| `promotions` | validity | `discount_value` mayor que 0 | 0 | 300 | 100.000 % |
| `promotions` | completeness | `end_date` informada | 0 | 300 | 100.000 % |
| `promotions` | integrity | `product_id` existe en `product_id` | 0 | 300 | 100.000 % |
| `promotions` | completeness | `product_id` informada | 0 | 300 | 100.000 % |
| `promotions` | validity | `promo_type` en 2x1/discount_pct/coupon | 0 | 300 | 100.000 % |
| `promotions` | completeness | `promo_type` informada | 0 | 300 | 100.000 % |
| `promotions` | completeness | `promotion_id` informada | 0 | 300 | 100.000 % |
| `promotions` | uniqueness | `promotion_id` sin repetir | 0 | 300 | 100.000 % |
| `promotions` | completeness | `start_date` informada | 0 | 300 | 100.000 % |
| `session_events` | consistency | El evento no es anterior al inicio de su sesion | 0 | 945.676 | 100.000 % |
| `session_events` | completeness | `event_timestamp` informada | 0 | 945.676 | 100.000 % |
| `session_events` | validity | `event_type` en view/add_to_cart | 0 | 945.676 | 100.000 % |
| `session_events` | completeness | `event_type` informada | 0 | 945.676 | 100.000 % |
| `session_events` | integrity | `product_id` existe en `product_id` | 0 | 945.676 | 100.000 % |
| `session_events` | completeness | `product_id` informada | 0 | 945.676 | 100.000 % |
| `session_events` | integrity | `session_id` existe en `session_id` | 0 | 945.676 | 100.000 % |
| `session_events` | completeness | `session_id` informada | 0 | 945.676 | 100.000 % |
| `sessions` | integrity | `basket_id` existe en `basket_id` | 0 | 52.500 | 100.000 % |
| `sessions` | consistency | `converted` coincide con tener `basket_id` | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `converted` informada | 0 | 150.000 | 100.000 % |
| `sessions` | integrity | `customer_id` existe en `customer_id` | 0 | 115.521 | 100.000 % |
| `sessions` | validity | `device_type` en mobile/desktop/tablet | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `device_type` informada | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `session_date` informada | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `session_id` informada | 0 | 150.000 | 100.000 % |
| `sessions` | uniqueness | `session_id` sin repetir | 0 | 150.000 | 100.000 % |

## Despues de limpiar

**Data Trust Score: 100.00 / 100 (nota A)** — 79 de 79 comprobaciones sin una sola fila mala.

| Dimension | Puntuacion | Tasa de filas | Tasa de reglas | Reglas KO |
| --- | ---: | ---: | ---: | ---: |
| `completeness` | 100.00 | 100.00 % | 100.00 % | 0 / 39 |
| `uniqueness` | 100.00 | 100.00 % | 100.00 % | 0 / 8 |
| `validity` | 100.00 | 100.00 % | 100.00 % | 0 / 16 |
| `consistency` | 100.00 | 100.00 % | 100.00 % | 0 / 7 |
| `integrity` | 100.00 | 100.00 % | 100.00 % | 0 / 9 |

| Tabla | Dimension | Comprobacion | Filas KO | Total | % OK |
| --- | --- | --- | ---: | ---: | ---: |
| `basket_items` | integrity | `basket_id` existe en `basket_id` | 0 | 3.057.825 | 100.000 % |
| `basket_items` | completeness | `basket_id` informada | 0 | 3.057.825 | 100.000 % |
| `basket_items` | uniqueness | Una sola linea por cesta y producto | 0 | 3.057.825 | 100.000 % |
| `basket_items` | uniqueness | Sin filas duplicadas exactas | 0 | 3.057.825 | 100.000 % |
| `basket_items` | integrity | `product_id` existe en `product_id` | 0 | 3.057.825 | 100.000 % |
| `basket_items` | completeness | `product_id` informada | 0 | 3.057.825 | 100.000 % |
| `basket_items` | consistency | La promocion aplicada estaba vigente el dia de la cesta | 0 | 81.975 | 100.000 % |
| `basket_items` | integrity | `promotion_id` existe en `promotion_id` | 0 | 81.975 | 100.000 % |
| `basket_items` | completeness | `quantity` informada | 0 | 3.057.825 | 100.000 % |
| `basket_items` | validity | `quantity` mayor que 0 | 0 | 3.057.825 | 100.000 % |
| `basket_items` | completeness | `unit_price_paid` informada | 0 | 3.057.825 | 100.000 % |
| `basket_items` | validity | `unit_price_paid` mayor que 0 | 0 | 3.057.825 | 100.000 % |
| `baskets` | completeness | `basket_date` informada | 0 | 600.174 | 100.000 % |
| `baskets` | completeness | `basket_id` informada | 0 | 600.174 | 100.000 % |
| `baskets` | uniqueness | `basket_id` sin repetir | 0 | 600.174 | 100.000 % |
| `baskets` | validity | `channel` en app/web/store | 0 | 600.174 | 100.000 % |
| `baskets` | completeness | `channel` informada | 0 | 600.174 | 100.000 % |
| `baskets` | integrity | `customer_id` existe en `customer_id` | 0 | 582.174 | 100.000 % |
| `baskets` | consistency | `store_id` informado si y solo si el canal es `store` | 0 | 600.174 | 100.000 % |
| `baskets` | consistency | `total_amount` cuadra con la suma de sus lineas | 0 | 600.174 | 100.000 % |
| `baskets` | completeness | `total_amount` informada | 0 | 600.174 | 100.000 % |
| `baskets` | validity | `total_amount` mayor que 0 | 0 | 600.174 | 100.000 % |
| `customers` | completeness | `churn_label` informada | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `city` informada | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `country` informada | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `customer_id` informada | 0 | 20.000 | 100.000 % |
| `customers` | uniqueness | `customer_id` sin repetir | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `household_size_est` informada | 0 | 20.000 | 100.000 % |
| `customers` | validity | `household_size_est` entre 1 y 6 | 0 | 20.000 | 100.000 % |
| `customers` | validity | `loyalty_tier` en bronze/silver/gold | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `loyalty_tier` informada | 0 | 20.000 | 100.000 % |
| `customers` | validity | `preferred_channel` en app/web/store | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `preferred_channel` informada | 0 | 20.000 | 100.000 % |
| `customers` | consistency | `signup_date` anterior o igual a la primera compra | 0 | 20.000 | 100.000 % |
| `customers` | completeness | `signup_date` informada | 0 | 20.000 | 100.000 % |
| `customers` | consistency | `signup_date` no posterior al fin del periodo observado | 0 | 20.000 | 100.000 % |
| `products` | completeness | `brand` informada | 0 | 1.500 | 100.000 % |
| `products` | validity | `category` sin mayusculas ni espacios inconsistentes | 0 | 1.500 | 100.000 % |
| `products` | completeness | `category` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `department` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `is_perishable` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `is_private_label` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `pack_size` informada | 0 | 1.500 | 100.000 % |
| `products` | validity | `pack_size` mayor que 0 | 0 | 1.500 | 100.000 % |
| `products` | completeness | `product_id` informada | 0 | 1.500 | 100.000 % |
| `products` | uniqueness | `product_id` sin repetir | 0 | 1.500 | 100.000 % |
| `products` | validity | `typical_repurchase_days` mayor que 0 | 0 | 1.500 | 100.000 % |
| `products` | completeness | `typical_repurchase_days` informada | 0 | 1.500 | 100.000 % |
| `products` | completeness | `unit_price` informada | 0 | 1.500 | 100.000 % |
| `products` | validity | `unit_price` mayor que 0 | 0 | 1.500 | 100.000 % |
| `promotions` | validity | `start_date` anterior o igual a `end_date` | 0 | 300 | 100.000 % |
| `promotions` | completeness | `discount_value` informada | 0 | 300 | 100.000 % |
| `promotions` | validity | `discount_value` mayor que 0 | 0 | 300 | 100.000 % |
| `promotions` | completeness | `end_date` informada | 0 | 300 | 100.000 % |
| `promotions` | integrity | `product_id` existe en `product_id` | 0 | 300 | 100.000 % |
| `promotions` | completeness | `product_id` informada | 0 | 300 | 100.000 % |
| `promotions` | validity | `promo_type` en 2x1/discount_pct/coupon | 0 | 300 | 100.000 % |
| `promotions` | completeness | `promo_type` informada | 0 | 300 | 100.000 % |
| `promotions` | completeness | `promotion_id` informada | 0 | 300 | 100.000 % |
| `promotions` | uniqueness | `promotion_id` sin repetir | 0 | 300 | 100.000 % |
| `promotions` | completeness | `start_date` informada | 0 | 300 | 100.000 % |
| `session_events` | consistency | El evento no es anterior al inicio de su sesion | 0 | 945.651 | 100.000 % |
| `session_events` | completeness | `event_timestamp` informada | 0 | 945.651 | 100.000 % |
| `session_events` | validity | `event_type` en view/add_to_cart | 0 | 945.651 | 100.000 % |
| `session_events` | completeness | `event_type` informada | 0 | 945.651 | 100.000 % |
| `session_events` | uniqueness | Sin filas duplicadas exactas | 0 | 945.651 | 100.000 % |
| `session_events` | integrity | `product_id` existe en `product_id` | 0 | 945.651 | 100.000 % |
| `session_events` | completeness | `product_id` informada | 0 | 945.651 | 100.000 % |
| `session_events` | integrity | `session_id` existe en `session_id` | 0 | 945.651 | 100.000 % |
| `session_events` | completeness | `session_id` informada | 0 | 945.651 | 100.000 % |
| `sessions` | integrity | `basket_id` existe en `basket_id` | 0 | 52.500 | 100.000 % |
| `sessions` | consistency | `converted` coincide con tener `basket_id` | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `converted` informada | 0 | 150.000 | 100.000 % |
| `sessions` | integrity | `customer_id` existe en `customer_id` | 0 | 115.521 | 100.000 % |
| `sessions` | validity | `device_type` en mobile/desktop/tablet | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `device_type` informada | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `session_date` informada | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `session_id` informada | 0 | 150.000 | 100.000 % |
| `sessions` | uniqueness | `session_id` sin repetir | 0 | 150.000 | 100.000 % |

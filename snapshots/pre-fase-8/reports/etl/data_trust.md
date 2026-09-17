# Data Trust Score (Tarea 1)

| Dataset | Score | Nota |
| --- | ---: | :---: |
| `data/raw` (crudo) | 91.42 | C |
| `data/processed` (limpio) | 100.00 | A |

## Antes de limpiar

**Data Trust Score: 91.42 / 100 (nota C)** — 70 de 79 comprobaciones sin una sola fila mala.

| Dimension | Puntuacion | Tasa de filas | Tasa de reglas | Reglas KO |
| --- | ---: | ---: | ---: | ---: |
| `completeness` | 97.41 | 99.95 % | 94.87 % | 2 / 39 |
| `uniqueness` | 81.07 | 99.63 % | 62.50 % | 3 / 8 |
| `validity` | 93.61 | 99.72 % | 87.50 % | 2 / 16 |
| `consistency` | 85.02 | 98.62 % | 71.43 % | 2 / 7 |
| `integrity` | 100.00 | 100.00 % | 100.00 % | 0 / 9 |

| Tabla | Dimension | Comprobacion | Filas KO | Total | % OK |
| --- | --- | --- | ---: | ---: | ---: |
| `baskets` | consistency | `total_amount` cuadra con la suma de sus lineas | 56.205 | 600.174 | 90.635 % |
| `products` | validity | `category` sin mayusculas ni espacios inconsistentes | 20 | 496 | 95.968 % |
| `basket_items` | uniqueness | Una sola linea por cesta y producto | 45.860 | 3.103.685 | 98.522 % |
| `basket_items` | uniqueness | Sin filas duplicadas exactas | 45.474 | 3.103.685 | 98.535 % |
| `products` | completeness | `brand` informada | 5 | 496 | 98.992 % |
| `customers` | completeness | `city` informada | 154 | 20.000 | 99.230 % |
| `basket_items` | validity | `quantity` mayor que 0 | 12.314 | 3.103.685 | 99.603 % |
| `customers` | consistency | `signup_date` anterior o igual a la primera compra | 58 | 20.000 | 99.710 % |
| `session_events` | uniqueness | Sin filas duplicadas exactas | 79 | 897.674 | 99.991 % |
| `basket_items` | integrity | `basket_id` existe en `basket_id` | 0 | 3.103.685 | 100.000 % |
| `basket_items` | completeness | `basket_id` informada | 0 | 3.103.685 | 100.000 % |
| `basket_items` | integrity | `product_id` existe en `product_id` | 0 | 3.103.685 | 100.000 % |
| `basket_items` | completeness | `product_id` informada | 0 | 3.103.685 | 100.000 % |
| `basket_items` | consistency | La promocion aplicada estaba vigente el dia de la cesta | 0 | 158.226 | 100.000 % |
| `basket_items` | integrity | `promotion_id` existe en `promotion_id` | 0 | 158.226 | 100.000 % |
| `basket_items` | completeness | `quantity` informada | 0 | 3.103.685 | 100.000 % |
| `basket_items` | completeness | `unit_price_paid` informada | 0 | 3.103.685 | 100.000 % |
| `basket_items` | validity | `unit_price_paid` mayor que 0 | 0 | 3.103.685 | 100.000 % |
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
| `customers` | consistency | `signup_date` no posterior al fin del periodo observado | 0 | 20.000 | 100.000 % |
| `products` | completeness | `category` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `department` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `is_perishable` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `is_private_label` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `pack_size` informada | 0 | 496 | 100.000 % |
| `products` | validity | `pack_size` mayor que 0 | 0 | 496 | 100.000 % |
| `products` | completeness | `product_id` informada | 0 | 496 | 100.000 % |
| `products` | uniqueness | `product_id` sin repetir | 0 | 496 | 100.000 % |
| `products` | validity | `typical_repurchase_days` mayor que 0 | 0 | 496 | 100.000 % |
| `products` | completeness | `typical_repurchase_days` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `unit_price` informada | 0 | 496 | 100.000 % |
| `products` | validity | `unit_price` mayor que 0 | 0 | 496 | 100.000 % |
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
| `session_events` | consistency | El evento no es anterior al inicio de su sesion | 0 | 897.674 | 100.000 % |
| `session_events` | completeness | `event_timestamp` informada | 0 | 897.674 | 100.000 % |
| `session_events` | validity | `event_type` en view/add_to_cart | 0 | 897.674 | 100.000 % |
| `session_events` | completeness | `event_type` informada | 0 | 897.674 | 100.000 % |
| `session_events` | integrity | `product_id` existe en `product_id` | 0 | 897.674 | 100.000 % |
| `session_events` | completeness | `product_id` informada | 0 | 897.674 | 100.000 % |
| `session_events` | integrity | `session_id` existe en `session_id` | 0 | 897.674 | 100.000 % |
| `session_events` | completeness | `session_id` informada | 0 | 897.674 | 100.000 % |
| `sessions` | integrity | `basket_id` existe en `basket_id` | 0 | 52.500 | 100.000 % |
| `sessions` | consistency | `converted` coincide con tener `basket_id` | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `converted` informada | 0 | 150.000 | 100.000 % |
| `sessions` | integrity | `customer_id` existe en `customer_id` | 0 | 115.340 | 100.000 % |
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
| `basket_items` | consistency | La promocion aplicada estaba vigente el dia de la cesta | 0 | 155.937 | 100.000 % |
| `basket_items` | integrity | `promotion_id` existe en `promotion_id` | 0 | 155.937 | 100.000 % |
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
| `products` | completeness | `brand` informada | 0 | 496 | 100.000 % |
| `products` | validity | `category` sin mayusculas ni espacios inconsistentes | 0 | 496 | 100.000 % |
| `products` | completeness | `category` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `department` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `is_perishable` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `is_private_label` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `pack_size` informada | 0 | 496 | 100.000 % |
| `products` | validity | `pack_size` mayor que 0 | 0 | 496 | 100.000 % |
| `products` | completeness | `product_id` informada | 0 | 496 | 100.000 % |
| `products` | uniqueness | `product_id` sin repetir | 0 | 496 | 100.000 % |
| `products` | validity | `typical_repurchase_days` mayor que 0 | 0 | 496 | 100.000 % |
| `products` | completeness | `typical_repurchase_days` informada | 0 | 496 | 100.000 % |
| `products` | completeness | `unit_price` informada | 0 | 496 | 100.000 % |
| `products` | validity | `unit_price` mayor que 0 | 0 | 496 | 100.000 % |
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
| `session_events` | consistency | El evento no es anterior al inicio de su sesion | 0 | 897.595 | 100.000 % |
| `session_events` | completeness | `event_timestamp` informada | 0 | 897.595 | 100.000 % |
| `session_events` | validity | `event_type` en view/add_to_cart | 0 | 897.595 | 100.000 % |
| `session_events` | completeness | `event_type` informada | 0 | 897.595 | 100.000 % |
| `session_events` | uniqueness | Sin filas duplicadas exactas | 0 | 897.595 | 100.000 % |
| `session_events` | integrity | `product_id` existe en `product_id` | 0 | 897.595 | 100.000 % |
| `session_events` | completeness | `product_id` informada | 0 | 897.595 | 100.000 % |
| `session_events` | integrity | `session_id` existe en `session_id` | 0 | 897.595 | 100.000 % |
| `session_events` | completeness | `session_id` informada | 0 | 897.595 | 100.000 % |
| `sessions` | integrity | `basket_id` existe en `basket_id` | 0 | 52.500 | 100.000 % |
| `sessions` | consistency | `converted` coincide con tener `basket_id` | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `converted` informada | 0 | 150.000 | 100.000 % |
| `sessions` | integrity | `customer_id` existe en `customer_id` | 0 | 115.340 | 100.000 % |
| `sessions` | validity | `device_type` en mobile/desktop/tablet | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `device_type` informada | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `session_date` informada | 0 | 150.000 | 100.000 % |
| `sessions` | completeness | `session_id` informada | 0 | 150.000 | 100.000 % |
| `sessions` | uniqueness | `session_id` sin repetir | 0 | 150.000 | 100.000 % |

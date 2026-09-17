# Informe de limpieza (Fase 2)

Generado por `python -m src.etl.run_etl`. El porque de cada regla esta en `docs/CLEANING.md`.

| Tabla | Problema | Regla aplicada | Filas afectadas | % | Filas antes | Filas despues |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `products` | product_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 496 | 496 |
| `products` | category con mayusculas/espacios inconsistentes | Se unifica a la grafia mas frecuente de su clave normalizada | 20 | 4.032 % | 496 | 496 |
| `products` | brand nula | Se rellena con 'Sin marca' y se marca en brand_is_missing | 5 | 1.008 % | 496 | 496 |
| `basket_items` | quantity negativa | Se corrige el signo: quantity = abs(quantity) | 17.253 | 0.396 % | 4.357.007 | 4.357.007 |
| `basket_items` | linea de ticket duplicada | Se elimina la fila repetida entera, despues de corregir el signo | 64.512 | 1.481 % | 4.357.007 | 4.292.495 |
| `basket_items` | duplicados que solo afloran tras corregir el signo | Diagnostico: (dedup tras signo) - (dedup sin corregir el signo) | 518 | 0.012 % | 4.357.007 | 4.292.495 |
| `basket_items` | quantity nula o cero | Se descarta: una linea sin unidades no es una venta | 0 | 0.000 % | 4.292.495 | 4.292.495 |
| `baskets` | basket_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 600.174 | 600.174 |
| `baskets` | total_amount no cuadra con las lineas | Se recalcula como suma de line_amount de basket_items ya limpias | 1.208 | 0.201 % | 600.174 | 600.174 |
| `baskets` | outlier de total_amount | Se marca en total_amount_is_outlier si la cabecera supera 1.5x el importe recalculado | 1.208 | 0.201 % | 600.174 | 600.174 |
| `baskets` | cesta sin lineas tras la limpieza | Se conserva con importe 0: la cabecera sigue siendo una visita real | 0 | 0.000 % | 600.174 | 600.174 |
| `customers` | customer_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 20.000 | 20.000 |
| `customers` | city nula | Se rellena con 'Desconocida' y se marca en city_is_missing | 154 | 0.770 % | 20.000 | 20.000 |
| `customers` | signup_date posterior a la primera compra | Se corrige a la fecha de la primera compra observada | 58 | 0.290 % | 20.000 | 20.000 |
| `promotions` | promotion_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 300 | 300 |
| `promotions` | ventana de vigencia invertida | Se marca en date_range_invalid; no se borra para no romper las FK | 0 | 0.000 % | 300 | 300 |
| `sessions` | session_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 150.000 | 150.000 |
| `sessions` | converted incoherente con basket_id | converted pasa a derivarse de basket_id IS NOT NULL | 0 | 0.000 % | 150.000 | 150.000 |
| `session_events` | evento duplicado exacto | Se elimina la fila repetida entera | 85 | 0.008 % | 1.106.225 | 1.106.140 |

### Categorias normalizadas (20 grafias corregidas)

| Grafia original | Canonica |
| --- | --- |
| `  Aceite de oliva` | `Aceite de oliva` |
| `  Harina` | `Harina` |
| `  Yogur` | `Yogur` |
| `Aceite de oliva  ` | `Aceite de oliva` |
| `Carne  de  pollo` | `Carne de pollo` |
| `Cerveza  ` | `Cerveza` |
| `Chocolate y huevos de Pascua  ` | `Chocolate y huevos de Pascua` |
| `Comida para gato  ` | `Comida para gato` |
| `Comida para perro  ` | `Comida para perro` |
| `HUEVOS` | `Huevos` |
| `Limpiacristales  ` | `Limpiacristales` |
| `Marisco  ` | `Marisco` |
| `PASTA DE DIENTES` | `Pasta de dientes` |
| `Pizza congelada  ` | `Pizza congelada` |
| `Toallitas  humedas` | `Toallitas humedas` |
| `YOGUR` | `Yogur` |
| `ZUMOS` | `Zumos` |
| `gel de ducha` | `Gel de ducha` |
| `helados` | `Helados` |
| `zumos` | `Zumos` |

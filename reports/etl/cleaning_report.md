# Informe de limpieza (Fase 2)

Generado por `python -m src.etl.run_etl`. El porque de cada regla esta en `docs/CLEANING.md`.

| Tabla | Problema | Regla aplicada | Filas afectadas | % | Filas antes | Filas despues |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `products` | product_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 1.500 | 1.500 |
| `products` | category con mayusculas/espacios inconsistentes | Se unifica a la grafia mas frecuente de su clave normalizada | 60 | 4.000 % | 1.500 | 1.500 |
| `products` | brand nula | Se rellena con 'Sin marca' y se marca en brand_is_missing | 17 | 1.133 % | 1.500 | 1.500 |
| `basket_items` | quantity negativa | Se corrige el signo: quantity = abs(quantity) | 12.308 | 0.397 % | 3.103.684 | 3.103.684 |
| `basket_items` | linea de ticket duplicada | Se elimina la fila repetida entera, despues de corregir el signo | 45.859 | 1.478 % | 3.103.684 | 3.057.825 |
| `basket_items` | duplicados que solo afloran tras corregir el signo | Diagnostico: (dedup tras signo) - (dedup sin corregir el signo) | 353 | 0.011 % | 3.103.684 | 3.057.825 |
| `basket_items` | quantity nula o cero | Se descarta: una linea sin unidades no es una venta | 0 | 0.000 % | 3.057.825 | 3.057.825 |
| `baskets` | basket_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 600.174 | 600.174 |
| `baskets` | total_amount no cuadra con las lineas | Se recalcula como suma de line_amount de basket_items ya limpias | 1.192 | 0.199 % | 600.174 | 600.174 |
| `baskets` | outlier de total_amount | Se marca en total_amount_is_outlier si la cabecera supera 1.5x el importe recalculado | 1.192 | 0.199 % | 600.174 | 600.174 |
| `baskets` | cesta sin lineas tras la limpieza | Se conserva con importe 0: la cabecera sigue siendo una visita real | 0 | 0.000 % | 600.174 | 600.174 |
| `customers` | customer_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 20.000 | 20.000 |
| `customers` | city nula | Se rellena con 'Desconocida' y se marca en city_is_missing | 163 | 0.815 % | 20.000 | 20.000 |
| `customers` | signup_date posterior a la primera compra | Se corrige a la fecha de la primera compra observada | 58 | 0.290 % | 20.000 | 20.000 |
| `promotions` | promotion_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 300 | 300 |
| `promotions` | ventana de vigencia invertida | Se marca en date_range_invalid; no se borra para no romper las FK | 0 | 0.000 % | 300 | 300 |
| `sessions` | session_id duplicado | Se conserva una fila por PK | 0 | 0.000 % | 150.000 | 150.000 |
| `sessions` | converted incoherente con basket_id | converted pasa a derivarse de basket_id IS NOT NULL | 0 | 0.000 % | 150.000 | 150.000 |
| `session_events` | evento duplicado exacto | Se elimina la fila repetida entera | 33 | 0.004 % | 900.143 | 900.110 |

### Categorias normalizadas (54 grafias corregidas)

| Grafia original | Canonica |
| --- | --- |
| `  Cereales` | `Cereales` |
| `  Comida para gato` | `Comida para gato` |
| `  Conservas de pescado` | `Conservas de pescado` |
| `  Galletas` | `Galletas` |
| `  Leche` | `Leche` |
| `  Marisco` | `Marisco` |
| `  Papel higienico` | `Papel higienico` |
| `  Potitos` | `Potitos` |
| `  Sal y especias` | `Sal y especias` |
| `  Vino` | `Vino` |
| `  Yogur` | `Yogur` |
| `  Zumos` | `Zumos` |
| `ARROZ` | `Arroz` |
| `Agua  ` | `Agua` |
| `Arena  para  gato` | `Arena para gato` |
| `BOLSAS DE BASURA` | `Bolsas de basura` |
| `CAVA Y ESPUMOSOS` | `Cava y espumosos` |
| `CHAMPU` | `Champu` |
| `Carne  de  ternera` | `Carne de ternera` |
| `DESODORANTE` | `Desodorante` |
| `FRUTA` | `Fruta` |
| `Fruta  ` | `Fruta` |
| `Galletas  ` | `Galletas` |
| `Harina  ` | `Harina` |
| `Helados  ` | `Helados` |
| `LECHE` | `Leche` |
| `LEGUMBRES` | `Legumbres` |
| `Lavavajillas  ` | `Lavavajillas` |
| `PAN` | `Pan` |
| `PANALES` | `Panales` |
| `PAPEL HIGIENICO` | `Papel higienico` |
| `PASTA` | `Pasta` |
| `Palomitas  de  microondas` | `Palomitas de microondas` |
| `Panales  ` | `Panales` |
| `Papel higienico  ` | `Papel higienico` |
| `Precocinados  congelados` | `Precocinados congelados` |
| `Protector solar  ` | `Protector solar` |
| `Refrescos  ` | `Refrescos` |
| `Snacks  y  aperitivos` | `Snacks y aperitivos` |
| `Suavizante  ` | `Suavizante` |
| `TOALLITAS HUMEDAS` | `Toallitas humedas` |
| `Toallitas humedas  ` | `Toallitas humedas` |
| `VERDURA` | `Verdura` |
| `VINO` | `Vino` |
| `acondicionador` | `Acondicionador` |
| `bolsas de basura` | `Bolsas de basura` |
| `panales` | `Panales` |
| `papel higienico` | `Papel higienico` |
| `pasta` | `Pasta` |
| `pizza congelada` | `Pizza congelada` |
| `precocinados congelados` | `Precocinados congelados` |
| `salsa de tomate` | `Salsa de tomate` |
| `verdura` | `Verdura` |
| `vino` | `Vino` |

# Afinidad de cesta: pares esperados de DATA_SPEC.md

Generado por `python -m src.etl.run_etl`. Desde la Fase 8, "lift objetivo" es la fuerza nominal del par y el lift medido es el crudo de `affinity_category`, que sube con el tamano de la cesta (ver `DATA_SPEC.md`, "Afinidad de cesta").

| Disparadora | Asociada | Lift objetivo | Lift medido | Soporte | Confianza | Medido/objetivo |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cerveza | Snacks y aperitivos | 3.0 | 3.29 | 8.83 % | 68.27 % | 1.10x |
| Pasta | Salsa de tomate | 2.5 | 4.98 | 6.18 % | 55.33 % | 1.99x |
| Panales | Toallitas humedas | 4.0 | 16.32 | 2.47 % | 69.04 % | 4.08x |
| Cafe | Azucar y edulcorante | 2.0 | 4.34 | 5.14 % | 40.96 % | 2.17x |
| Pan | Embutido y fiambre | 2.2 | 1.64 | 23.06 % | 52.34 % | 0.74x |
| Cereales | Leche | 2.8 | 1.91 | 9.15 % | 81.01 % | 0.68x |
| Vino | Queso | 2.5 | 3.31 | 5.15 % | 65.84 % | 1.32x |
| Detergente | Suavizante | 3.5 | 5.73 | 4.99 % | 55.97 % | 1.64x |
| Champu | Acondicionador | 3.0 | 6.29 | 4.13 % | 47.94 % | 2.10x |
| Palomitas de microondas | Refrescos | 2.0 | 3.09 | 3.25 % | 60.93 % | 1.54x |
| Cafe | Leche | 2.0 | 1.84 | 9.79 % | 78.05 % | 0.92x |
| Galletas | Leche | 2.0 | 1.79 | 8.72 % | 76.09 % | 0.90x |
| Cereales | Yogur | 2.0 | 2.40 | 6.93 % | 61.38 % | 1.20x |
| Pan | Aceite de oliva | 2.0 | 1.74 | 7.97 % | 18.09 % | 0.87x |
| Gel de ducha | Champu | 2.2 | 5.12 | 3.71 % | 44.13 % | 2.33x |
| Pasta de dientes | Desodorante | 2.0 | 5.91 | 2.68 % | 39.70 % | 2.95x |
| Papel higienico | Lejia y limpiadores | 2.0 | 4.49 | 4.75 % | 39.72 % | 2.25x |
| Comida para gato | Arena para gato | 3.0 | 21.22 | 1.11 % | 43.95 % | 7.07x |
| Comida para perro | Bolsas de basura | 2.0 | 5.16 | 0.92 % | 36.50 % | 2.58x |
| Pasta | Queso | 2.0 | 3.06 | 6.81 % | 60.98 % | 1.53x |
| Arroz | Marisco | 2.5 | 6.53 | 2.52 % | 30.76 % | 2.61x |
| Legumbres | Embutido y fiambre | 2.0 | 2.21 | 5.35 % | 70.79 % | 1.11x |
| Carne de pollo | Verdura | 2.0 | 2.09 | 7.78 % | 65.53 % | 1.04x |
| Harina | Huevos | 2.5 | 3.14 | 2.78 % | 67.55 % | 1.26x |
| Harina | Azucar y edulcorante | 2.5 | 5.88 | 2.28 % | 55.47 % | 2.35x |
| Aceite de oliva | Sal y especias | 2.2 | 6.07 | 3.37 % | 32.35 % | 2.76x |
| Pescado blanco | Verdura | 2.0 | 2.22 | 4.61 % | 69.65 % | 1.11x |
| Pizza congelada | Refrescos | 2.0 | 3.10 | 6.63 % | 61.24 % | 1.55x |
| Cava y espumosos | Turron y mazapan | 2.5 | 17.81 | 0.41 % | 26.79 % | 7.13x |
| Carne de ternera | Cerveza | 2.0 | 3.17 | 3.24 % | 40.94 % | 1.58x |

## Estructura global de `affinity_category`

Pares ordenados con soporte suficiente: **3.782**.

| Umbral | Pares con lift por encima | % |
| --- | ---: | ---: |
| lift > 1.2 | 3.714 | 98.2% |
| lift > 1.5 | 3.372 | 89.2% |
| lift > 3.0 | 1.282 | 33.9% |
| lift < 0.8 | 0 | 0.0% |

Mediana del lift: 2.45. El lift crudo mezcla la afinidad con el tamano de la cesta; el controlado por tamano esta en `python -m data_generation.verify_dataset` (seccion `coocurrencia_de_categorias`).

### Los 20 pares de mas lift

| Antecedente | Consecuente | Lift | Soporte |
| --- | --- | ---: | ---: |
| Arena para gato | Comida para gato | 21.22 | 1.11 % |
| Comida para gato | Arena para gato | 21.22 | 1.11 % |
| Cava y espumosos | Turron y mazapan | 17.81 | 0.41 % |
| Turron y mazapan | Cava y espumosos | 17.81 | 0.41 % |
| Panales | Toallitas humedas | 16.32 | 2.47 % |
| Toallitas humedas | Panales | 16.32 | 2.47 % |
| Arena para gato | Comida para perro | 13.32 | 0.69 % |
| Comida para perro | Arena para gato | 13.32 | 0.69 % |
| Leche infantil | Potitos | 12.74 | 0.90 % |
| Potitos | Leche infantil | 12.74 | 0.90 % |
| Leche infantil | Toallitas humedas | 12.31 | 1.20 % |
| Toallitas humedas | Leche infantil | 12.31 | 1.20 % |
| Potitos | Toallitas humedas | 12.12 | 1.58 % |
| Toallitas humedas | Potitos | 12.12 | 1.58 % |
| Leche infantil | Panales | 12.03 | 0.99 % |
| Panales | Leche infantil | 12.03 | 0.99 % |
| Panales | Potitos | 11.78 | 1.30 % |
| Potitos | Panales | 11.78 | 1.30 % |
| Comida para gato | Comida para perro | 11.35 | 0.72 % |
| Comida para perro | Comida para gato | 11.35 | 0.72 % |

# ROADMAP

Plan de trabajo por fases. Cada fase debería cerrar con algo demostrable (un notebook, un
test que pasa, un fichero de salida) antes de pasar a la siguiente.

## Fase 0 — Setup

- [ ] Estructura de carpetas del repo (ver `CHALLENGE.md`)
- [ ] Entorno: `requirements.txt` / `pyproject.toml`, PySpark local
- [ ] `constraints.txt` con versiones fijas (numpy, pandas, faker, pyspark) para
      reproducibilidad, como en `sports-rental-analytics`
- [ ] CI mínimo: un workflow que instale dependencias y ejecute los tests

## Fase 1 — Generador de datos

- [ ] `generate_dataset.py`: `customers`, `products`, `promotions`, `baskets`,
      `basket_items`, `sessions`, `session_events` según `DATA_SPEC.md`
- [ ] Semilla fija (seed=42) propagada a numpy/random/Faker
- [ ] Reglas de negocio: ciclos de reposición, afinidad de cesta, estacionalidad, uplift de
      promoción, señal de churn progresiva
- [ ] Problemas de calidad deliberados
- [ ] Diccionario de datos autogenerado (`output/README.md`, como en el otro proyecto)
- [ ] Verificación: dos ejecuciones producen el mismo hash

## Fase 2 — ETL y feature engineering (PySpark)

- [ ] Limpieza: duplicados, nulos, categorías inconsistentes, outliers — documentado
- [ ] Data Trust Score (Tarea 1)
- [ ] RFM por cliente
- [ ] Función `due_for_repurchase` por cliente-categoría (Tarea 2)
- [ ] Afinidad de cesta (co-ocurrencia / FP-Growth) como tabla intermedia para el
      recomendador

## Fase 3 — Recomendador de cesta

- [ ] Split train/test por `basket_id` (no por fila) para evitar fuga de datos
- [ ] Baseline: popularidad por departamento/estacionalidad
- [ ] Candidatos por co-compra (a partir de la Fase 2)
- [ ] Candidatos personalizados (ALS sobre `customer_id × product_id`, o similar)
- [ ] Lógica de combinación para los 4 perfiles de cliente (Tarea 3a)
- [ ] Evaluación: NDCG@5, Recall@5 sobre el test

## Fase 4 — Next Best Action

- [ ] Target de propensión: compra en categoría en próximos 7 días / churn en 4 semanas
- [ ] Modelo (LightGBM o logística) + validación temporal (no aleatoria, para no filtrar
      futuro)
- [ ] Catálogo de acciones y coste/margen asociado por acción
- [ ] Política de valor esperado (Tarea 3b)
- [ ] Evaluación: AUC/PR-AUC del modelo, uplift de valor esperado de la política frente a
      "no actuar" y "actuar siempre"

## Fase 5 — Empaquetado y storytelling

- [ ] README principal con arquitectura, resultados y cómo reproducir
- [ ] Notebook o informe con los hallazgos de negocio (estilo "vistazo al análisis" del
      otro proyecto)
- [ ] Tests de las funciones de Tarea 1 y Tarea 2
- [ ] (Opcional) MLflow para registrar experimentos del recomendador y de propensión —
      aporta un ángulo de MLOps/BI que no está en los otros dos proyectos

## Fuera de alcance (por ahora)

- Servir el modelo como API o demo en tiempo real
- Multi-país (se queda en España para esta primera versión)
- Optimización de precios/promociones (posible Fase 6 si el proyecto da más de sí)

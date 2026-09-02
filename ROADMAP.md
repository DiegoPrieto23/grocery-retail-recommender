# ROADMAP

Plan de trabajo por fases. Cada fase debería cerrar con algo demostrable (un notebook, un
test que pasa, un fichero de salida) antes de pasar a la siguiente.

## Fase 0 — Setup

- [x] Estructura de carpetas del repo (ver `CHALLENGE.md`)
- [x] Entorno: `requirements.txt` / `pyproject.toml`, PySpark local
- [x] `constraints.txt` con versiones fijas (numpy, pandas, faker, pyspark) para
      reproducibilidad, como en `sports-rental-analytics`
- [x] CI mínimo: un workflow que instale dependencias y ejecute los tests

## Fase 1 — Generador de datos

- [x] `generate_dataset.py`: `customers`, `products`, `promotions`, `baskets`,
      `basket_items`, `sessions`, `session_events` según `DATA_SPEC.md`
- [x] Semilla fija (seed=42) propagada a numpy/random/Faker
- [x] Reglas de negocio: ciclos de reposición, afinidad de cesta, estacionalidad, uplift de
      promoción, señal de churn progresiva
- [x] Problemas de calidad deliberados
- [x] Diccionario de datos autogenerado (`output/README.md`, como en el otro proyecto)
- [x] Verificación: dos ejecuciones producen el mismo hash

## Fase 2 — ETL y feature engineering (PySpark)

Se ejecuta entera con `python -m src.etl.run_etl`, que deja las tablas en
`data/processed/` y los informes en `reports/etl/`.

- [x] Limpieza: duplicados, nulos, categorías inconsistentes, outliers — documentado
      · `src/etl/cleaning.py` · criterios en `docs/CLEANING.md` · recuento reproducible en
      `reports/etl/cleaning_report.md` · verificado por `tests/test_cleaning.py`
- [x] Data Trust Score (Tarea 1)
      · `src/etl/data_trust.py` · 79 comprobaciones en 5 dimensiones · **89,99 (C) sobre el
      crudo → 100,00 (A) sobre el limpio** · `reports/etl/data_trust.md` · verificado por
      `tests/test_data_trust.py`
- [x] Notebook de EDA: preguntas de negocio de la Tarea 1 resueltas con Spark SQL y
      visualizaciones, una por pregunta
      · `notebooks/01_eda.ipynb` · 9 preguntas, cada una con su `spark.sql` y su figura ·
      ejecutado de principio a fin, con salidas guardadas
- [x] RFM por cliente
      · `src/etl/rfm.py` · tabla `data/processed/rfm` · verificado por `tests/test_rfm.py`
- [x] Función `due_for_repurchase` por cliente-categoría (Tarea 2)
      · `src/etl/repurchase.py` · tabla `data/processed/repurchase_features` · verificado
      por `tests/test_repurchase.py`
- [x] Afinidad de cesta (co-ocurrencia / FP-Growth) como tabla intermedia para el
      recomendador
      · `src/etl/affinity.py` · tablas `affinity_category` y `affinity_product` · los 10
      pares de `DATA_SPEC.md` salen con lift > 1,8 y coinciden con
      `data_generation/verify_dataset.py` hasta el segundo decimal ·
      `reports/etl/affinity_expected_pairs.md` · verificado por `tests/test_affinity.py`

## Fase 3 — Recomendador de cesta

- [ ] Split train/test por `basket_id` (no por fila) para evitar fuga de datos
- [ ] Generación de candidatos: popularidad/estacionalidad, co-compra (Fase 2), ALS
      (Spark MLlib)
- [ ] Feature engineering para el ranker: señal de cada fuente de candidatos,
      recency/frequency, promoción, popularidad reciente, señal de sesión
- [ ] **Deuda detectada en la Fase 2 (Q9 del EDA): la señal de sesión está contaminada.**
      El generador emite un `add_to_cart` por cada producto de la cesta y ninguno más, así
      que el solapamiento entre carrito y ticket es del 100 % y usarlo como feature
      filtraría el target. Antes de entrenar hay que elegir: dejar `add_to_cart` fuera del
      ranker, o arreglar el generador para que haya carritos abandonados. Los eventos
      `view` sí tienen ruido real y son utilizables.
- [ ] Ranking: LightGBM con objetivo `LambdaRank` sobre los candidatos
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
- [ ] Diagrama ER (Mermaid) del modelo relacional de las 7 tablas, con sus claves y
      relaciones, incluido en el README
- [ ] Resumen de impacto de negocio (medio folio): NDCG@5 y uplift de NBA traducidos a
      impacto estimado (ej. cross-sell extra en €/mes)
- [ ] Exportar a `reports/powerbi/` el modelo dimensional de `DATA_SPEC.md` (`dim_customers`,
      `dim_products`, `dim_date`, `dim_actions`, `fact_basket_items`,
      `fact_recommendations`, `fact_nba`) en Parquet o CSV
- [ ] Generar el proyecto Power BI (`.pbip`): modelo TMDL con las relaciones del star schema
      y medidas DAX básicas, e informe PBIR con Power Query apuntando a los ficheros
      locales (parámetro de carpeta, sin BBDD en la nube) — se abre directo en Power BI
      Desktop
- [ ] Notebook o informe con los hallazgos de negocio (estilo "vistazo al análisis" del
      otro proyecto)
- [ ] Tests de las funciones de Tarea 1 y Tarea 2
- [ ] (Opcional) MLflow para registrar experimentos del recomendador y de propensión —
      aporta un ángulo de MLOps/BI que no está en los otros dos proyectos

## Fase 6 — Demo web interactiva

- [ ] App Streamlit en local: selector de cliente (existente / nuevo simulado) para cubrir
      los 4 perfiles del recomendador
- [ ] Simulación de cesta: añadir productos y ver el top-5 de recomendaciones (Fase 3)
      actualizarse en vivo
- [ ] Banner de Next Best Action (Fase 4) para el cliente simulado
- [ ] La demo solo hace inferencia sobre los modelos ya guardados en `models/`, no
      reentrena nada
- [ ] README corto de la demo: cómo lanzarla en local

## Fuera de alcance (por ahora)

- Servir el modelo en producción/cloud como API pública (la demo de la Fase 6 es solo local)
- Multi-país (se queda en España para esta primera versión)
- Optimización de precios/promociones (posible fase futura si el proyecto da más de sí)

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

Se ejecuta entera con `python -m src.recommender.pipeline`, que deja el ranker en
`models/`, las predicciones en `predictions/` y los informes en `reports/recommender/`.
`python -m src.recommender.demo_profiles` traduce esas salidas a un caso legible de cada
perfil.

- [x] Split train/test por `basket_id` (no por fila) para evitar fuga de datos
      · `src/recommender/splits.py` · **temporal y por cesta**, en tres ventanas: fuentes
      hasta 2025-09-01, queries del ranker sep-oct 2025, test desde 2025-11-01. Las
      fuentes de candidatos se **reajustan por ventana**, así que ninguna ve el futuro de
      la cesta que predice · verificado por `tests/test_recommender.py`
- [x] Generación de candidatos: popularidad/estacionalidad, co-compra (Fase 2), ALS
      (Spark MLlib)
      · `src/recommender/candidates.py` · cinco fuentes (se añadió co-compra por
      **categoría**, que cubre la cola larga donde `affinity_product` no llega) · 155
      candidatos por cesta de media, `pool_recall` = 31,1 %
- [x] Feature engineering para el ranker: señal de cada fuente de candidatos,
      recency/frequency, promoción, popularidad reciente, señal de sesión
      · `src/recommender/features.py` · 54 features en 5 familias
- [x] **Deuda detectada en la Fase 2 (Q9 del EDA): la señal de sesión estaba contaminada.**
      **Resuelta arreglando el generador.** El problema era peor de lo anotado: no sólo
      `add_to_cart`, también los `view` cubrían el 100 % de la cesta. `_generate_sessions`
      produce ahora abandono de carrito a nivel de línea, productos que sólo se miran, y
      un retardo entre ver y añadir — `P(en la cesta | add_to_cart)` = 87,9 %,
      `P(en la cesta | view)` = 58,6 %, `P(visto | en la cesta)` = 85,0 % · documentado en
      `DATA_SPEC.md` → "Embudo online" · fijado por dos tests en
      `tests/test_generate_dataset.py` · la señal se consume con un corte estricto en
      `cut_ts` y aporta **+13 % de NDCG@5** sobre la ablación que la excluye
- [x] Ranking: LightGBM con objetivo `LambdaRank` sobre los candidatos
      · `src/recommender/ranker.py` · 84 árboles, parada temprana sobre NDCG@5 de validación
- [x] Lógica de combinación para los 4 perfiles de cliente (Tarea 3a)
      · el ranker es **el mismo** para los cuatro; lo que cambia es qué fuentes tienen
      señal. Medido sobre el top-5: el perfil 1 se cubre 100 % con popularidad, el 4
      combina las cinco · `reports/recommender/profiles_demo.md`
- [x] Evaluación: NDCG@5, Recall@5 sobre el test
      · `src/recommender/evaluate.py` · **NDCG@5 = 0,0343 · Recall@5 = 0,0332 ·
      hit_rate@5 = 11,8 %** sobre 18.000 cestas · baseline de popularidad 0,0200 ·
      `reports/recommender/metrics.md`

### Lo que limita la métrica (hallazgo de la fase)

El NDCG@5 de SKU es bajo y **no es por la primera etapa**. Dos medidas lo demuestran:

1. **Categoría vs SKU.** El sistema acierta la categoría en el **51,4 %** de las cestas y
   el SKU exacto sólo en el 11,8 %. De los aciertos de categoría, apenas el 13 % son
   además el SKU correcto.
2. **Ampliar el pool no ayuda.** Subir de 92 a 155 candidatos por cesta llevó el
   `pool_recall` del 21,7 % al 31,1 % (+43 % de target alcanzable) y el NDCG@5 no se movió
   (0,0344 → 0,0343).

La causa está en el generador: dentro de una categoría hay ~24 referencias y la elección
es casi aleatoria. Un cliente con tres o más compras en una categoría compra **0,86
referencias distintas por compra** — es decir, casi nunca repite SKU — y sólo el 29,7 % de
las líneas de una cesta futura son productos que ya había comprado, frente al **88,6 %**
cuando se mira la categoría. La fidelidad está en la categoría, no en la referencia.

- [ ] **Deuda para más adelante: dar fidelidad de marca/SKU al generador.** En gran consumo
      real un cliente repite referencia (siempre la misma leche), y ahí es donde un
      recomendador de SKU tiene margen. Implica tocar la elección de producto dentro de
      categoría en `_generate_baskets_and_items`, regenerar el dataset (cambian los
      `sha256` de `baskets` y `basket_items`) y rehacer la Fase 2 y sus informes.

## Fase 4 — Next Best Action

- [ ] Target de propensión: compra en categoría en próximos 7 días / churn en 4 semanas
- [ ] Modelo (LightGBM o logística) + validación temporal (no aleatoria, para no filtrar
      futuro)
- [ ] Catálogo de acciones y coste/margen asociado por acción
- [ ] Política de valor esperado (Tarea 3b)
- [ ] Evaluación: AUC/PR-AUC del modelo, uplift de valor esperado de la política frente a
      "no actuar" y "actuar siempre"

## Fase 5 — Empaquetado y storytelling

- [~] README principal con arquitectura, resultados y cómo reproducir
      · `README.md` cubre ya las Fases 0-3 (generación, lógica inyectada, ETL, features,
      EDA, recomendador con su NDCG@5 y el diagnóstico SKU/categoría, tests y notas de
      entorno). Falta añadir los resultados de las Fases 4-6 (uplift del NBA, Power BI y
      demo) según vayan saliendo.
- [x] Diagrama ER (Mermaid) del modelo relacional de las 7 tablas, con sus claves y
      relaciones, incluido en el README
      · sección "Modelo relacional" del `README.md`; las 14 relaciones están además
      verificadas empíricamente en `notebooks/01_eda.ipynb` (0 claves huérfanas)
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

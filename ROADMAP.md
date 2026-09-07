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

Se ejecuta entera con `python -m src.nba.pipeline`, que deja los dos modelos en `models/`,
la tabla `customer_id → acción → valor esperado` en `predictions/nba_actions.parquet` y los
informes en `reports/nba/`.

- [x] Target de propensión: compra en categoría en próximos 7 días / churn en 4 semanas
      · `src/nba/targets.py` · **no se usa `customers.churn_label` como target**: está
      definido respecto al final del dataset (60 días sin comprar hasta 2025-12-31), así
      que en un corte de abril sería adivinar algo de ocho meses después, y como *feature*
      sería fuga. El churn se construye por corte de forma observacional; `churn_label`
      queda para una comprobación de cordura, que da **83,4 % de coincidencia** ·
      verificado por `tests/test_nba.py`
- [x] Modelo (LightGBM o logística) + validación temporal (no aleatoria, para no filtrar
      futuro)
      · `src/nba/propensity.py` · dos binarios · cuatro cortes de entrenamiento
      (abr–ago 2025), validación en 2025-09-15, test en 2025-11-01 · las features de cada
      corte sólo miran compras anteriores a él, fijado por un test que añade una compra
      posterior y comprueba que ninguna feature se mueve
- [x] Catálogo de acciones y coste/margen asociado por acción
      · `src/nba/config.py` · `ninguna_accion`, `recomendar_producto`,
      `enviar_cupon_categoria` · margen bruto **por departamento** (18 % Frescos → 35 %
      Droguería/Higiene) · el coste se parte en dos: `send_cost` se paga siempre,
      `discount` sólo si el cliente compra · valor facial del cupón anclado a la media real
      de los cupones de `promotions` (2,54 €)
- [x] Política de valor esperado (Tarea 3b)
      · `src/nba/policy.py` · todo se mide **incremental sobre no actuar**, así
      `ninguna_accion` vale 0 por construcción y una acción sólo gana si su efecto paga su
      coste · actúa sobre el **75,6 %** de los clientes, no sobre todos
- [x] Evaluación: AUC/PR-AUC del modelo, uplift de valor esperado de la política frente a
      "no actuar" y "actuar siempre"
      · **Churn 4 semanas: AUC = 0,8531 · PR-AUC = 0,8556** (tasa base 0,4759) ·
      **Compra en categoría 7 días: AUC = 0,7634 · PR-AUC = 0,2190** (tasa base 0,0619) ·
      política **+4.012 €** frente a "no actuar", **+3.082 €** frente a la mejor
      alternativa trivial · `reports/nba/metrics.md`

### Lo que no es medible con este dataset (hallazgo de la fase)

`P(conversión | acción)` **no es identificable aquí**. El generador aplica su
`PROMO_UPLIFT = 3.0` al reparto de cuota *dentro* de una categoría (qué SKU se elige), no a
la probabilidad de comprar la categoría ni a la de volver: el tratamiento nunca varía. Las
propensiones sí se miden; el efecto de cada acción es un **supuesto declarado** en
`config.py`, y el informe lo barre en vez de esconderlo.

Barrer el supuesto cambió la lectura del resultado. El barrido obvio —el uplift de
conversión del cupón— resultó ser el parámetro equivocado: pasar de 1,00 a 2,00 mueve el
total de 3.938 € a 4.721 €. La razón es económica: con un cupón de 2,54 € persiguiendo un
margen esperado de ~1,4 €, **el descuento es mayor que el margen**, así que por cross-sell
el cupón destruye valor haga lo que haga la conversión. Todo lo que aporta viene de la
retención. El barrido que importa es el de `churn_reduction`:

| `churn_reduction` | Valor de la política | Cupones repartidos |
| ---: | ---: | ---: |
| **0,00** (no retiene a nadie) | **1.210 €** | 0 % |
| 0,05 | 1.734 € | 41,7 % |
| **0,10** (el supuesto) | **4.012 €** | 64,8 % |
| 0,20 | 8.862 € | 71,7 % |

La conclusión robusta es la primera fila: **incluso suponiendo que el cupón no retenga a
nadie, la política sigue ganando** (1.210 € frente a 930 € de "recomendar siempre" y 0 € de
"no actuar"), y en ese escenario deja de repartir cupones por completo. Lo que depende del
supuesto es el tamaño del premio, no el signo.

- [ ] **Deuda: el "modelo de churn" es en realidad un modelo de inactividad a 4 semanas.**
      Su tasa base es 0,4759 porque la cadencia media de visita ronda los 24 días, así que
      no comprar en 4 semanas le pasa a media base sin ser abandono. Las features
      dominantes (`n_baskets_90d`, `avg_days_between_baskets`, `recency_days`) confirman
      que buena parte de lo que acierta es frecuencia de compra. Un target honesto de churn
      pediría una ventana más larga o condicionar por la cadencia propia de cada cliente.

## Fase 5 — Empaquetado y storytelling

- [~] README principal con arquitectura, resultados y cómo reproducir
      · `README.md` cubre ya las Fases 0-4 (generación, lógica inyectada, ETL, features,
      EDA, recomendador con su NDCG@5 y el diagnóstico SKU/categoría, NBA con sus AUC y el
      barrido de sensibilidad, tests y notas de entorno), y cierra con un resumen fase a
      fase. Falta añadir los resultados de la Fase 6 (demo).
- [x] Diagrama ER (Mermaid) del modelo relacional de las 7 tablas, con sus claves y
      relaciones, incluido en el README
      · sección "Modelo relacional" del `README.md`; las 14 relaciones están además
      verificadas empíricamente en `notebooks/01_eda.ipynb` (0 claves huérfanas)
- [ ] Resumen de impacto de negocio (medio folio): NDCG@5 y uplift de NBA traducidos a
      impacto estimado (ej. cross-sell extra en €/mes)
- [ ] Notebook o informe con los hallazgos de negocio (estilo "vistazo al análisis" del
      otro proyecto)
- [ ] Tests de las funciones de Tarea 1 y Tarea 2
- [ ] (Opcional) MLflow para registrar experimentos del recomendador y de propensión —
      aporta un ángulo de MLOps/BI que no está en los otros dos proyectos

## Fase 6a — Preparación visual del catálogo

Paso previo a la app, se ejecuta una sola vez. Es la única parte del proyecto que necesita
internet — el resultado se cachea en `assets/` y a partir de ahí todo vuelve a ser local.

- [ ] `.env` con `PEXELS_API_KEY` (en `.gitignore`, nunca comiteado) + `.env.example` sin
      valores reales, comiteado como documentación de qué variable hace falta
- [ ] Analizar `products` (department, category, brand) y decidir si `category` ya es
      suficientemente granular o hace falta una columna `visual_group` nueva — ni tan
      amplia como el departamento ni tan específica como el SKU (p.ej. `leche_entera`,
      `yogur_griego`, `salmón`, no `Lácteos` ni un `product_id` concreto)
- [ ] CSV `visual_group, search_term` (término de búsqueda en inglés, que es donde Pexels
      tiene mejor cobertura, aunque el resto del proyecto esté en español)
- [ ] Revisar recuento de productos por `visual_group` y fusionar los grupos demasiado
      pequeños
- [ ] Descargar vía la API de Pexels (`PEXELS_API_KEY` en variable de entorno, nunca
      hardcodeada) una imagen representativa por `visual_group` — priorizando fondo limpio
      o blanco, aspecto ecommerce/supermercado, sin personas, sin composiciones complejas,
      sin fotos de cocina o restaurante
- [ ] Guardar las imágenes en `assets/` (`assets/leche_entera.jpg`, etc.)
- [ ] CSV final `product_id, product_name, visual_group, image_path` — `product_name` se
      deriva de department/category/brand (el dataset no tiene un nombre de producto
      propio; no hace falta tocar `DATA_SPEC.md` ni el generador para esto)
- [ ] Comitear `assets/` y los CSV de mapeo — no se regeneran en cada ejecución, se tratan
      como un fixture cacheado (los resultados de búsqueda de Pexels no son reproducibles
      por semilla)

## Fase 6b — Demo web interactiva

Es ahora el entregable central del proyecto: la pieza que hace tangibles el recomendador
(Fase 3) y el NBA (Fase 4) para cualquiera que la abra, sin tener que leer métricas.
Prioridad sobre lo que quede pendiente de la Fase 5. Se construye en tres versiones
incrementales — cada una cabe en una sesión de la suscripción Pro sin arriesgar quedarse a
medias. No pasar a la siguiente versión sin haber cerrado y verificado la anterior.

### V1 — Carrito, sin recomendaciones

- [ ] Comprobar que `streamlit`, `python-dotenv` y `requests` están en `requirements.txt`
      (no estaban en el `requirements.txt` original de la Fase 0) e instalarlos; verificar
      que `streamlit run` arranca antes de seguir
- [ ] Instalar la skill `developing-with-streamlit` (repo `streamlit/agent-skills`) en
      `.claude/skills/`
- [ ] Tema propio en `.streamlit/config.toml` (colores, fuente) en vez del tema por defecto
- [ ] Selector de cliente: existente (`customer_id` real) o "nuevo simulado" (sin historial)
- [ ] Cargar un carrito existente de un cliente (una cesta real de `basket_items` de test) o
      simular uno nuevo eligiendo productos manualmente
- [ ] Catálogo de productos visual: tarjetas con la foto real de su `visual_group` (Fase
      6a, vía `image_path`), nombre, categoría y precio
- [ ] La cesta (cargada o simulada) se muestra con el mismo estilo de tarjeta que el
      catálogo
- [ ] Sin llamadas al recomendador ni al NBA todavía — solo navegación de catálogo y cesta

### V2 — Recomendaciones en vivo

- [ ] Integrar el pipeline de candidatos + ranking (Fase 3, ya entrenado): al cambiar la
      cesta, se recalcula y muestra el top-5 de recomendaciones
- [ ] Las recomendaciones se muestran como tarjetas con foto (mismo estilo que el catálogo)
      y un motivo breve cuando se pueda derivar de las features del ranker (ej. "porque te
      toca reponerlo", "co-compra habitual con lo que llevas")
- [ ] Verificar que los 4 perfiles de cliente funcionan correctamente (nuevo sin cesta,
      nuevo con cesta, recurrente sin cesta, recurrente con cesta)
- [ ] Sigue sin NBA

### V3 — Next Best Action

- [ ] Cargar los modelos de propensión (Fase 4) y calcular la acción recomendada para el
      cliente activo
- [ ] Banner de Next Best Action visualmente destacado (color/icono), no solo texto plano
- [ ] La demo completa (V1+V2+V3) solo hace inferencia sobre los modelos ya guardados en
      `models/` y las imágenes ya descargadas en `assets/` — no reentrena nada ni vuelve a
      llamar a Pexels
- [ ] README corto de la demo: cómo lanzarla en local

## Fuera de alcance (por ahora)

- Dashboard en Power BI (Tarea 4 de `CHALLENGE.md`) — el foco pasa a la demo de la Fase 6;
  se retoma más adelante si el proyecto da más de sí
- Servir el modelo en producción/cloud como API pública (la demo de la Fase 6 es solo local)
- Multi-país (se queda en España para esta primera versión)
- Optimización de precios/promociones (posible fase futura si el proyecto da más de sí)

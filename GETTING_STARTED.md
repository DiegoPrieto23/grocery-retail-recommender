# GETTING_STARTED — cómo arrancar el proyecto con Claude Code

## 1. Por qué esta estructura

Claude Code lee automáticamente, al inicio de cada sesión, cualquier `CLAUDE.md` que esté en
el directorio donde lo lanzas (o en directorios por encima). Los demás documentos
(`CHALLENGE.md`, `DATA_SPEC.md`, `ROADMAP.md`) **no** se cargan solos — Claude los lee bajo
demanda con sus herramientas de fichero, cuando el propio `CLAUDE.md` o tu prompt le dicen
que lo haga. Por eso el `CLAUDE.md` que ya tienes empieza indicando "lee también
CHALLENGE.md, DATA_SPEC.md y ROADMAP.md antes de generar código": así se aseguran de entrar
en contexto sin tener que pegarlos tú a mano cada vez.

## 2. Carpeta inicial

Antes de abrir Claude Code, deja el repo así (los 4 `.md` ya los tienes; el resto son
carpetas vacías que Claude Code rellenará en la Fase 0 del `ROADMAP.md`):

```
grocery-retail-recommender/
├── CLAUDE.md
├── CHALLENGE.md
├── DATA_SPEC.md
├── ROADMAP.md
├── data_generation/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── etl/
│   ├── recommender/
│   └── nba/
├── notebooks/
├── models/
├── predictions/
└── tests/
```

Comandos para dejarlo listo en PowerShell (ajustados a tu ruta y a que los 4 `.md` están
en Descargas):

```powershell
$base    = "C:\Users\Diego Prieto\Documents\Claude Code"
$project = Join-Path $base "grocery-retail-recommender"

$subfolders = @(
    "data_generation",
    "data\raw",
    "data\processed",
    "src\etl",
    "src\recommender",
    "src\nba",
    "notebooks",
    "models",
    "predictions",
    "tests"
)

foreach ($folder in $subfolders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $project $folder) | Out-Null
}

$downloads = "$env:USERPROFILE\Downloads"
$mdFiles   = @("CLAUDE.md", "CHALLENGE.md", "DATA_SPEC.md", "ROADMAP.md")

foreach ($file in $mdFiles) {
    Move-Item -Path (Join-Path $downloads $file) -Destination $project
}

Set-Location $project
git init
git add .
git commit -m "docs: brief, esquema de datos y roadmap iniciales"
```

Todo va en variables (`$base`, `$project`, `$downloads`) precisamente porque la ruta tiene
espacios (`Diego Prieto`, `Claude Code`) — así no hace falta ir entrecomillando cada línea a
mano.

### Alternativa en bash (Git Bash / WSL)

Si prefieres bash en vez de PowerShell, primero averigua cuál tienes — la ruta se escribe
distinto en cada uno. En tu terminal bash:

```bash
uname -a
```

- Si la salida menciona `Microsoft` o `WSL` → tienes **WSL**, y el disco `C:` vive en `/mnt/c/...`
- Si empieza por `MINGW64` o `MSYS` → tienes **Git Bash**, y el disco `C:` vive en `/c/...`

**Git Bash:**

```bash
base="/c/Users/Diego Prieto/Documents/Claude Code"
project="$base/grocery-retail-recommender"

mkdir -p "$project"/{data_generation,data/raw,data/processed,src/etl,src/recommender,src/nba,notebooks,models,predictions,tests}

downloads="/c/Users/Diego Prieto/Downloads"
mv "$downloads/CLAUDE.md" "$downloads/CHALLENGE.md" "$downloads/DATA_SPEC.md" "$downloads/ROADMAP.md" "$project/"

cd "$project"
git init
git add .
git commit -m "docs: brief, esquema de datos y roadmap iniciales"
```

**WSL:**

```bash
base="/mnt/c/Users/Diego Prieto/Documents/Claude Code"
project="$base/grocery-retail-recommender"

mkdir -p "$project"/{data_generation,data/raw,data/processed,src/etl,src/recommender,src/nba,notebooks,models,predictions,tests}

downloads="/mnt/c/Users/Diego Prieto/Downloads"
mv "$downloads/CLAUDE.md" "$downloads/CHALLENGE.md" "$downloads/DATA_SPEC.md" "$downloads/ROADMAP.md" "$project/"

cd "$project"
git init
git add .
git commit -m "docs: brief, esquema de datos y roadmap iniciales"
```

Si vas a trabajar con Claude Code dentro de WSL, instálalo también dentro de WSL (es un
entorno Linux aparte del Windows nativo). Si dudas entre todo esto, la opción de PowerShell
de arriba es la más directa: es la shell que ya trae Windows 11 por defecto, sin instalar ni
comprobar nada más.

No hace falta crear `README.md` a mano: es una de las tareas de la Fase 5 del `ROADMAP.md`,
y tenerlo vacío desde el principio solo invita a que quede desactualizado.

## 3. Prompt inicial para Claude Code

Lánzalo con `claude` dentro de `grocery-retail-recommender/` y pega esto como primer mensaje:

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos.

Quiero que ejecutemos la Fase 0 y la Fase 1 del ROADMAP.md:

1. Fase 0: monta el esqueleto del proyecto (requirements.txt con pyspark, pandas, numpy,
   faker, scikit-learn, lightgbm; constraints.txt con versiones fijas; un workflow de CI
   mínimo que instale dependencias y corra los tests).

2. Fase 1: implementa data_generation/generate_dataset.py siguiendo DATA_SPEC.md al pie de
   la letra — las 7 tablas, con seed=42, los ciclos de reposición, los 10 pares de afinidad
   de cesta y la estacionalidad con sus multiplicadores exactos, el uplift de promoción, el
   churn progresivo y los problemas de calidad deliberados.

No implementes nada de ETL, recomendador ni NBA todavía — eso es la Fase 2 en adelante y lo
abordamos en otra sesión.

Antes de dar la Fase 1 por terminada, verifica que dos ejecuciones del generador con la
misma seed producen el mismo hash de fichero, y déjame un resumen de qué tablas y volúmenes
ha generado.

Si algo de DATA_SPEC.md es ambiguo o te faltan datos para decidir un detalle concreto,
pregúntame antes de asumir.
```

Este prompt deja fuera del alcance el ETL/recomendador/NBA a propósito — es mucha lógica de
negocio para una sola sesión, y conviene revisar el dataset generado (Fase 1) antes de
construir nada encima.

## 4. Prompts para las siguientes sesiones

Cada sesión nueva de Claude Code no recuerda la anterior por defecto, así que cada prompt
empieza igual: relee los 4 documentos y repasa qué hay ya hecho en el repo antes de avanzar.
Van en orden — no pegues el de la Fase 3 si la Fase 2 no está terminada y verificada.

### Estado: qué prompts ya se han ejecutado

Los prompts de abajo se conservan como registro de cómo se construyó el proyecto, pero la
mayoría ya están gastados. El detalle fase a fase, con su verificación, está en
`ROADMAP.md`; este es el resumen para saber cuál toca:

| Fase | Estado | Nota |
| --- | --- | --- |
| 0 · Setup | ✅ | |
| 1 · Generador | ✅ | |
| 2 · ETL y features | ✅ | |
| 3 · Recomendador | ✅ | queda una deuda anotada (fidelidad de SKU en el generador) |
| 4 · Next Best Action | ✅ | queda una deuda anotada (el churn es inactividad a 4 semanas) |
| 5 · Empaquetado | 🟡 **a medias** | faltan impacto de negocio, informe de hallazgos y los resultados de la Fase 6 en el README |
| 6a · Preparación visual | ✅ | 60 `visual_group`, 60 fotos cacheadas en `assets/` |
| 6b · Demo (V1+V2+V3) | 🟡 **casi** | la app funciona entera; quedan 2 cosas, ver abajo |

**Lo único pendiente de la Fase 6b** (los prompts V1/V2/V3 de abajo ya no aplican tal cual:
las tres versiones aterrizaron juntas en el commit `55116b3`):

1. Sembrar la cesta con una cesta real de `basket_items` de test para el cliente elegido
   — hoy solo se construye a mano desde el catálogo.
2. El README de la demo, y el `tests/test_demo.py` que los docstrings ya citan pero que no
   existe.

### Prompt para retomar la Fase 6b

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa el estado actual del repo: la demo de la Fase 6b ya funciona entera (catálogo, cesta,
recomendaciones del ranker y banner de Next Best Action), así que NO la reescribas.

Quedan dos cosas abiertas en la Fase 6b del ROADMAP.md:

1. Permitir sembrar la cesta con una cesta real del cliente elegido, tomada de basket_items
   en la ventana de test (la misma que usa el split del recomendador, para no meter fuga).
   Debe convivir con la construcción manual que ya existe, no sustituirla: elegir cliente
   recurrente → poder cargar una de sus cestas reales → y desde ahí seguir añadiendo o
   quitando productos a mano.
2. Un README corto de la demo: cómo lanzarla en local, qué artefactos necesita
   (data/serving/, models/, predictions/, assets/) y qué ejecutar si falta alguno.

Añade además el tests/test_demo.py que los docstrings de streamlit_app.py y
src/demo/__init__.py ya citan pero que no existe: debe cubrir el buscador, el escaparate por
departamento y el motivo de una recomendación, sin levantar la app.

Al terminar, marca los checkboxes que cierres en ROADMAP.md y confirma que la app sigue
arrancando y que la batería de tests sigue en verde.

Si algo es ambiguo, pregúntame antes de asumir.
```

### Fase 2 — ETL y feature engineering

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa el estado actual del repo (qué hay en data_generation/ y data/raw/ de la Fase 1).

Vamos a ejecutar la Fase 2 del ROADMAP.md:

1. Limpieza de las 7 tablas generadas en la Fase 1: duplicados, nulos, categorías
   inconsistentes, cantidades negativas y outliers de total_amount — documenta qué se
   corrige y por qué.
2. Implementa el Data Trust Score (Tarea 1 del CHALLENGE.md) sobre el dataset limpio.
3. Notebook de EDA (notebooks/): resuelve cada pregunta de negocio de la Tarea 1 con
   Spark SQL (vistas temporales + spark.sql) y una visualización por pregunta
   (matplotlib/seaborn/plotly) — no solo una tabla de números.
4. Calcula RFM por cliente.
5. Implementa la función due_for_repurchase por cliente-categoría (Tarea 2 del
   CHALLENGE.md), usando typical_repurchase_days de products.
6. Construye la tabla de afinidad de cesta (co-ocurrencia o FP-Growth) a partir de
   basket_items, que servirá de candidato para el recomendador en la Fase 3.

Todo esto en PySpark, en src/etl/, con tests para las funciones de las Tareas 1 y 2.

No implementes nada del recomendador ni del NBA todavía — eso es la Fase 3 en adelante.

Al terminar, marca en ROADMAP.md los checkboxes de la Fase 2 que hayas completado, y déjame
un resumen de qué decisiones de limpieza tomaste y qué pinta tiene la tabla de afinidad
resultante (¿se ven los 10 pares de DATA_SPEC.md con lift alto?).

Si algo de DATA_SPEC.md o CHALLENGE.md es ambiguo, pregúntame antes de asumir.
```

### Fase 3 — Recomendador de cesta

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa el estado actual del repo (ETL y features de la Fase 2, en particular la tabla de
afinidad de cesta y el RFM).

Vamos a ejecutar la Fase 3 del ROADMAP.md — el recomendador de cesta (Tarea 3a del
CHALLENGE.md), con la arquitectura de dos etapas que describe CHALLENGE.md:

1. Split train/test por basket_id (nunca por fila de basket_items, para no filtrar datos).
2. Generación de candidatos, cada uno como una fuente independiente:
   - Popularidad/estacionalidad (fallback)
   - Co-compra (tabla de afinidad de la Fase 2)
   - ALS (Spark MLlib) sobre customer_id × product_id
3. Feature engineering para el ranker: score de cada fuente de candidatos,
   recency/frequency del cliente con esa categoría/producto, si está en promoción,
   popularidad reciente, y señal de sesión si la hay (session_events).
4. Entrena un LightGBM con objetivo de ranking (LambdaRank) sobre esas features para
   reordenar los candidatos y quedarte con el top-5.
5. Verifica que la lógica cubre los 4 perfiles de cliente del CHALLENGE.md (nuevo sin
   cesta, nuevo con cesta, recurrente sin cesta, recurrente con cesta) — lo que cambia
   entre perfiles es qué fuentes de candidatos tienen señal, no el ranker.
6. Evaluación con NDCG@5 y Recall@5 sobre el test.

Todo en src/recommender/, con un notebook o script que muestre el resultado para al menos
un ejemplo de cada uno de los 4 perfiles.

No implementes el NBA todavía — eso es la Fase 4.

Al terminar, marca los checkboxes de la Fase 3 en ROADMAP.md y dime qué NDCG@5/Recall@5 ha
salido, y si algún perfil de cliente rinde claramente peor que los otros.

Si algo es ambiguo, pregúntame antes de asumir.
```

### Fase 4 — Next Best Action

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa el estado actual del repo (recomendador de la Fase 3, y las features de RFM/churn de
la Fase 2).

Vamos a ejecutar la Fase 4 del ROADMAP.md — Next Best Action (Tarea 3b del CHALLENGE.md):

1. Define el target de propensión: compra en categoría en los próximos 7 días, y/o churn en
   las próximas 4 semanas (usa customers.churn_label y la caída progresiva de
   frecuencia/ticket de DATA_SPEC.md).
2. Split temporal (no aleatorio): entrena con un periodo pasado, evalúa con uno posterior.
3. Modelo de propensión (LightGBM o logística) en src/nba/.
4. Define el catálogo de acciones con coste y margen esperado (recomendar_producto,
   enviar_cupón_categoría, ninguna_acción).
5. Implementa la política de valor esperado:
   acción* = argmax_a (P(conversión|a) × margen_esperado(a) − coste(a)).
6. Evalúa el modelo con AUC/PR-AUC, y la política comparándola contra "no actuar siempre" y
   "actuar siempre".

Al terminar, marca los checkboxes de la Fase 4 en ROADMAP.md y dime qué AUC ha salido y qué
uplift de valor esperado tiene la política frente a las dos alternativas triviales.

Si algo es ambiguo (por ejemplo, qué margen o coste asignar a cada acción si no está en
DATA_SPEC.md), pregúntame antes de asumir un número arbitrario.
```

### Fase 5 — Empaquetado y storytelling

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa todo el trabajo de las Fases 1 a 4.

Vamos a ejecutar la Fase 5 del ROADMAP.md — empaquetado. El dashboard de Power BI que
menciona la Tarea 4 de CHALLENGE.md queda fuera de alcance por ahora (ver "Fuera de
alcance" en ROADMAP.md) — no lo implementes aunque CHALLENGE.md todavía lo describa; el
foco del proyecto pasa a la demo de la Fase 6.

1. README.md principal: qué es el proyecto, arquitectura (diagrama en texto o Mermaid está
   bien), cómo reproducirlo de principio a fin, y un resumen de resultados del recomendador
   y del NBA.
2. Diagrama ER (Mermaid) del modelo relacional de las 7 tablas (customers, products,
   promotions, baskets, basket_items, sessions, session_events) con sus claves y
   relaciones, incluido en el README.
3. Resumen de impacto de negocio (medio folio, en el README o en IMPACT.md aparte): traduce
   el NDCG@5 y el uplift del NBA a impacto estimado (ej. cross-sell extra en €/mes sobre el
   volumen de cestas simulado).
4. Notebook o informe con los hallazgos de negocio más interesantes (estacionalidad,
   afinidad de cesta, qué perfiles de cliente responden mejor al NBA) — con el mismo
   espíritu narrativo del informe de sports-rental-analytics.
5. Revisa que existan tests para las funciones clave de las Tareas 1 y 2, y añade los que
   falten.
6. (Opcional, si te sobra tiempo) Integra MLflow para registrar los experimentos del
   recomendador y del modelo de propensión.

Al terminar, marca los checkboxes de la Fase 5 en ROADMAP.md (el dashboard de Power BI no
cuenta como pendiente: está fuera de alcance, no a medias).

Antes de darla por cerrada, dime qué quedó sin cubrir de la Tarea 4 del CHALLENGE.md más
allá de Power BI, y por qué.
```

### Fase 6a — Preparación visual del catálogo

✅ **Ya ejecutada** (commit `b810bd5`). Se conserva como registro.

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa el esquema real de la tabla products (data/processed/ o donde la dejó la Fase 2).

Vamos a ejecutar la Fase 6a del ROADMAP.md — preparación visual del catálogo. Es un paso
previo a la demo, se ejecuta una sola vez y es la única parte del proyecto que necesita
internet.

1. Analiza las columnas de products (department, category, brand) y decide si category ya
   es suficientemente granular para agrupar visualmente los productos, o si hace falta una
   columna nueva visual_group. No uses el departamento (muy amplio) ni el product_id o la
   marca (muy específico) — el objetivo son grupos reutilizables tipo "leche_entera",
   "yogur_griego", "salmón".
2. Genera un CSV visual_group, search_term con el término de búsqueda en inglés para cada
   grupo (Pexels tiene mejor cobertura en inglés aunque el resto del proyecto esté en
   español).
3. Cuenta productos por visual_group y fusiona los grupos demasiado pequeños; documenta qué
   fusionaste y por qué.
4. Usa la API de Pexels para buscar y descargar una imagen representativa por
   visual_group. Prioriza imágenes de producto sobre fondo limpio o blanco, con aspecto de
   ecommerce/supermercado — evita personas, composiciones complejas o fotografías de
   cocinas y restaurantes en la query de búsqueda.
5. Guarda las imágenes en assets/ (assets/leche_entera.jpg, etc.).
6. Genera el CSV final product_id, product_name, visual_group, image_path. Como el dataset
   no tiene un nombre de producto propio, deriva product_name de department/category/brand
   — no hace falta tocar DATA_SPEC.md ni el generador para esto.
7. Comitea assets/ y los CSV de mapeo. No se regeneran en cada ejecución del pipeline: son
   un fixture cacheado, porque los resultados de Pexels no son reproducibles por semilla.

La PEXELS_API_KEY ya está en un .env en la raíz del proyecto. Cárgala con python-dotenv
(añádelo a requirements.txt si no está) — nunca la imprimas por pantalla, la metas en un
commit, un log o una celda de notebook, ni la hardcodees en el código. Antes de nada,
comprueba que .env está en .gitignore; si no lo está, añádelo tú antes de continuar (y
confirma que .env.example sí queda comiteado, como documentación de qué variable hace
falta, sin el valor real).

Al terminar, marca los checkboxes de la Fase 6a en ROADMAP.md y déjame un resumen: cuántos
visual_group salieron, cuáles fusionaste, y si algún grupo se quedó sin imagen válida.
```

### Fase 6b — V1: Carrito, sin recomendaciones

✅ **Ya ejecutada** (commit `55116b3`), salvo sembrar la cesta desde `basket_items`.

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa el estado actual del repo (README de la Fase 5, y el CSV de imágenes de la Fase 6a
en assets/).

Antes de nada, comprueba que streamlit, python-dotenv y requests están en
requirements.txt (el requirements.txt original de la Fase 0 no los incluía) e instálalos
si falta alguno. Verifica que streamlit run funciona (aunque sea sobre un app.py vacío)
antes de seguir — si da un error tipo "command not found" (127), es casi seguro que
streamlit no está instalado en el entorno activo.

Si no tienes ya instalada la skill developing-with-streamlit (repo streamlit/agent-skills)
en .claude/skills/, instálala ahora — la vamos a necesitar para el theming y estilizado.

Vamos a ejecutar la V1 de la Fase 6b del ROADMAP.md: una primera versión de la demo SIN
recomendaciones ni NBA todavía, solo catálogo y cesta. Es la primera de tres versiones
incrementales — hazlo así de acotado a propósito, no adelantes trabajo de la V2 o la V3.

1. Define un tema propio en .streamlit/config.toml (colores, fuente) en vez de dejar el
   tema por defecto de Streamlit.
2. App en Streamlit, en local, que:
   - Deja elegir un customer_id existente (cliente recurrente) o simular "cliente nuevo"
     (sin historial).
   - Si es un cliente existente, permite cargar una cesta real suya de basket_items de
     test como punto de partida.
   - Si es un cliente nuevo (o si se prefiere empezar de cero), permite construir una
     cesta manualmente eligiendo productos del catálogo.
   - Muestra el catálogo/resultado de búsqueda de productos como tarjetas visuales: cada
     producto lleva la foto real de su visual_group (columna image_path del CSV de la Fase
     6a), nombre, categoría y precio.
   - La cesta (cargada o construida a mano) se muestra con el mismo estilo de tarjeta, no
     como una tabla.
3. Todavía NO llames al recomendador ni al NBA — eso es la V2 y la V3. Esta versión es solo
   navegación de catálogo y gestión de cesta.
4. Añade un README corto dentro de la carpeta de la demo explicando cómo lanzarla en local
   (streamlit run ...).

Al terminar, marca los checkboxes de la V1 en ROADMAP.md y déjame confirmación de que
streamlit run arranca sin errores antes de que sigamos con la V2 en otra sesión.

Si algo es ambiguo, pregúntame antes de asumir.
```

### Fase 6b — V2: Recomendaciones en vivo

✅ **Ya ejecutada** (commit `55116b3`). Los 4 perfiles quedaron verificados.

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa el estado actual del repo — en particular la app de la V1 de la Fase 6b (catálogo y
cesta ya funcionando) y el ranker entrenado en models/ de la Fase 3.

Vamos a ejecutar la V2 de la Fase 6b del ROADMAP.md: añadir recomendaciones en vivo sobre
la app de la V1. No reescribas ni rehagas lo de la V1 — amplíalo.

1. Integra el pipeline de candidatos + ranking (Fase 3, ya entrenado): carga el modelo una
   vez al arrancar la app (no en cada interacción), y recalcula el top-5 cada vez que
   cambie la cesta.
2. Muestra las recomendaciones como tarjetas con el mismo estilo visual que el catálogo
   (foto real del visual_group), y añade un motivo breve por recomendación cuando se pueda
   derivar de las features del ranker (ej. "porque te toca reponerlo", "co-compra habitual
   con lo que llevas").
3. Verifica explícitamente que los 4 perfiles de cliente funcionan: nuevo sin cesta, nuevo
   con cesta, recurrente sin cesta, recurrente con cesta — prueba los cuatro casos y
   dime qué fuentes de candidatos tiene señal en cada uno.
4. Todavía NO implementes el NBA — eso es la V3.

Al terminar, marca los checkboxes de la V2 en ROADMAP.md.

Si algo es ambiguo, pregúntame antes de asumir.
```

### Fase 6b — V3: Next Best Action

✅ **Ya ejecutada** (commit `55116b3`), salvo el README de la demo.

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa el estado actual del repo — en particular la app de la V1+V2 de la Fase 6b (catálogo,
cesta y recomendaciones ya funcionando) y los modelos de propensión en models/ de la Fase 4.

Vamos a ejecutar la V3 de la Fase 6b del ROADMAP.md: añadir el Next Best Action sobre la
app de la V1+V2. No reescribas lo anterior — amplíalo.

1. Carga los modelos de propensión (Fase 4) una vez al arrancar la app, y calcula la acción
   recomendada (Tarea 3b del CHALLENGE.md) para el cliente activo.
2. Muestra un banner con esa acción, visualmente destacado (color/icono), no como texto
   plano perdido en la página.
3. Comprueba que la demo completa (V1+V2+V3) solo hace inferencia: carga los artefactos ya
   entrenados de models/ y las imágenes ya descargadas en assets/, sin reentrenar nada ni
   volver a llamar a la API de Pexels en caliente.
4. Actualiza el README de la demo si hace falta, con el flujo completo.

No despliegues nada en la nube ni montes una API pública — sigue fuera de alcance del
proyecto, esto es solo para local.

Al terminar, marca los checkboxes de la V3 en ROADMAP.md — con esto, la Fase 6b queda
cerrada del todo.

Si algo es ambiguo, pregúntame antes de asumir.
```

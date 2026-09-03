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

### Fase 6 — Demo web interactiva

```text
Antes de escribir nada, lee CLAUDE.md, CHALLENGE.md, DATA_SPEC.md y ROADMAP.md completos, y
revisa el estado actual del repo (modelos guardados en models/ de las Fases 3 y 4, y el
README de la Fase 5).

Vamos a ejecutar la Fase 6 del ROADMAP.md — demo web interactiva (Tarea 4 del CHALLENGE.md).
Es ahora el entregable central del proyecto: tiene que ser gráfica e intuitiva, no una
lista de texto con botones.

1. App en Streamlit, en local, que:
   - Deja elegir un customer_id existente (cliente recurrente) o simular "cliente nuevo"
     (sin historial), para cubrir los 4 perfiles del recomendador.
   - Muestra el catálogo/resultado de búsqueda de productos como tarjetas visuales: cada
     producto lleva un icono representativo de su departamento (son datos sintéticos, no
     hay fotos reales — usa un emoji/icono fijo por departamento, no busques imágenes
     externas), nombre, categoría y precio.
   - Simula una cesta: buscar/añadir productos uno a uno; la cesta se muestra con el mismo
     estilo de tarjeta, no como una tabla.
   - Tras cada cambio en la cesta, llama al pipeline de candidatos + ranking (Fase 3, ya
     entrenado) y muestra el top-5 de recomendaciones actualizado, también como tarjetas
     con icono, y un motivo breve por recomendación cuando se pueda derivar de las features
     del ranker (ej. "porque te toca reponerlo", "co-compra habitual con lo que llevas").
   - Muestra un banner con la Next Best Action (Fase 4, ya entrenada) para el cliente
     simulado (ej. "cupón 10% en Detergente"), destacado visualmente (color/icono), no como
     texto plano perdido en la página.
2. La demo solo hace inferencia: carga los artefactos ya entrenados de models/, no
   reentrena nada en caliente.
3. Añade un README corto dentro de la carpeta de la demo explicando cómo lanzarla en local
   (streamlit run ...).

No despliegues nada en la nube ni montes una API pública — sigue fuera de alcance del
proyecto, esto es solo para local. Tampoco busques imágenes de producto reales ni conectes
a ningún servicio externo de imágenes — los iconos por departamento son suficientes.

Al terminar, marca los checkboxes de la Fase 6 en ROADMAP.md.

Si algo es ambiguo, pregúntame antes de asumir.
```

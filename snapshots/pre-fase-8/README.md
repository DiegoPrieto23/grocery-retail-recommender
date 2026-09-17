# Snapshot `pre-fase-8`

Foto del proyecto justo antes de regenerar el dataset en la Fase 8 (misiones de compra,
cola larga del tamaño de cesta y sustitución: puntos A5, M1 y M2 de
`docs/diagnostico-fase7.md`). Congelado el 2026-09-17 sobre el commit `0a49c2e`, con el
mismo criterio que la Fase 7: guardar el "antes" para poder comparar sin fiarse de la
memoria.

## Qué hay aquí

| Ruta | En git | Qué es |
| --- | :---: | --- |
| `README.md` | sí | Este fichero |
| `MANIFEST.json` | sí | sha256 de cada fichero del snapshot, el commit y los hashes de las 7 tablas del dataset |
| `reports/` | sí | Copia de `reports/` de ese momento (ETL, recomendador con su diagnóstico, NBA, impacto e insights) y de `IMPACT.md` |
| `reports/etl/verify_dataset.json` | sí | `python -m data_generation.verify_dataset --json` sobre el dataset de entonces, con las secciones nuevas de la Fase 8 (tamaño, co-ocurrencia, varias referencias, marca blanca) |
| `data/raw/` | no | Dataset de la Fase 7a (7 CSV, 147 MB) |
| `data/processed/` | no | Salida del ETL de la Fase 2 |
| `data/oracle/` | no | Oráculo del recomendador (formato anterior a la Fase 8) |
| `data/serving/` | no | Bundle de la demo |
| `models/` | no | LambdaRank y los dos modelos del NBA |
| `predictions/` | no | Predicciones de test, incluidas las referencias congeladas de antes de A1 (`pre_a1/`) y de A4 (`pre_a4/`) |

Lo que no va en git pesa unos 280 MB y se regenera; si se pierde, `MANIFEST.json` dice qué
había y permite comprobar una copia.

Las cifras que usan los informes para comparar están además en tres ficheros junto a los
informes vivos, que es donde las buscan los pipelines:

- `reports/recommender/baseline_pre_fase8.json`: `metrics.json` y `diagnostics.json` del
  recomendador (baselines y techo teórico incluidos);
- `reports/nba/baseline_pre_fase8.json`: `metrics.json` del NBA;
- `reports/impact/baseline_pre_fase8.json`: `impact.json`.

Los tres guardan `dataset_sha256`, los hashes de las 7 tablas de `data/raw`. Los
pipelines solo pintan una comparación "antes/después" de los puntos A1, A2 y A4 cuando el
snapshot se midió sobre el dataset en uso; con el de la Fase 8, esas comparaciones viven
aquí, en `reports/recommender/metrics.md`.

## Cifras de partida

| | Fase 7a (este snapshot) |
| --- | ---: |
| Cestas / líneas | 600.174 / 3.103.685 |
| Líneas por cesta: media, CV, p99, máximo | 5,08 · 0,42 · 11 · 19 |
| Pares de categorías con lift > 1,5 (crudo / controlado por tamaño) | 40 / — |
| Líneas por categoría en una cesta | siempre 1 |
| `cat_hit_rate@5` del LambdaRank / techo | 0,7181 / 0,7816 (91,9 %) |
| `sku_hit_rate@5` del LambdaRank / techo | 0,5416 / 0,5920 (91,5 %) |
| NDCG@5 graduada | 0,2148 |

El lift controlado de la tabla se definió después de congelar el snapshot; su valor sobre
este dataset está en `reports/etl/verify_dataset.json` y en la comparación de
`ROADMAP.md` (Fase 8).

## Cómo restaurarlo

```bash
cp -r snapshots/pre-fase-8/data/raw data/            # y lo mismo con processed, oracle, serving
cp -r snapshots/pre-fase-8/models snapshots/pre-fase-8/predictions .
git checkout 0a49c2e -- data_generation src          # el código que los produjo
```

Para comprobar que la copia es la buena:

```bash
python -c "import hashlib, json, pathlib; m = json.load(open('snapshots/pre-fase-8/MANIFEST.json')); \
bad = [f for f, h in m['files'].items() if hashlib.sha256((pathlib.Path('snapshots/pre-fase-8') / f).read_bytes()).hexdigest() != h]; \
print('OK' if not bad else bad)"
```

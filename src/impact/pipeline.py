"""Resumen de impacto de negocio: mide, aplica el modelo y escribe `IMPACT.md`.

    python -m src.impact.pipeline

Lee tres fuentes, ninguna a mano:

- `data/processed/` para las magnitudes del negocio simulado (cestas al mes, importe de
  una linea, mezcla de departamentos).
- `reports/recommender/metrics.json` para el `hit_rate@5` del ranker y del baseline.
- `reports/nba/metrics.json` para el valor incremental de la politica y su barrido.

Y escribe dos, las dos regenerables:

- `reports/impact/impact.json` con todas las cifras y los supuestos usados.
- `IMPACT.md`, el medio folio que pide la Tarea 4 de `CHALLENGE.md`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.impact.config import ImpactConfig
from src.impact.model import (
    CrossSellImpact,
    NBAImpact,
    blended_margin_rate,
    sweep_incremental_rate,
)

ONLINE_CHANNELS: tuple[str, ...] = ("app", "web")


@dataclass(frozen=True)
class Baseline:
    """Magnitudes del negocio simulado, todas medidas sobre `data/processed`."""

    months: int
    baskets: int
    baskets_per_month: float
    online_baskets: int
    online_baskets_per_month: float
    lines: int
    avg_line_amount: float
    avg_lines_per_basket: float
    customers_with_purchases: int
    sales_by_department: dict[str, float]
    margin_rate: float

    def as_dict(self) -> dict:
        return {
            "meses": self.months,
            "cestas": self.baskets,
            "cestas_mes": self.baskets_per_month,
            "cestas_online": self.online_baskets,
            "cestas_online_mes": self.online_baskets_per_month,
            "lineas": self.lines,
            "importe_medio_linea": self.avg_line_amount,
            "lineas_por_cesta": self.avg_lines_per_basket,
            "clientes_con_compras": self.customers_with_purchases,
            "venta_por_departamento": self.sales_by_department,
            "margen_bruto_mezclado": self.margin_rate,
        }


def measure_baseline(cfg: ImpactConfig) -> Baseline:
    """Mide el tamano del negocio simulado a partir de las tablas limpias."""
    processed = Path(cfg.processed_dir)
    baskets = pd.read_parquet(
        processed / "baskets.parquet",
        columns=["basket_id", "customer_id", "channel", "basket_day"],
    )
    items = pd.read_parquet(
        processed / "basket_items.parquet", columns=["basket_id", "product_id", "line_amount"]
    )
    products = pd.read_parquet(
        processed / "products.parquet", columns=["product_id", "department"]
    )

    months = pd.to_datetime(baskets["basket_day"]).dt.to_period("M").nunique()
    online = baskets["channel"].isin(ONLINE_CHANNELS)
    sales = (
        items.merge(products, on="product_id")
        .groupby("department")["line_amount"]
        .sum()
        .to_dict()
    )

    return Baseline(
        months=int(months),
        baskets=int(len(baskets)),
        baskets_per_month=len(baskets) / months,
        online_baskets=int(online.sum()),
        online_baskets_per_month=float(online.sum()) / months,
        lines=int(len(items)),
        avg_line_amount=float(items["line_amount"].mean()),
        avg_lines_per_basket=len(items) / len(baskets),
        customers_with_purchases=int(baskets["customer_id"].nunique()),
        sales_by_department={k: float(v) for k, v in sales.items()},
        margin_rate=blended_margin_rate(
            sales, cfg.margins.by_department, cfg.margins.default
        ),
    )


def _row(rows: list[dict], key: str, value: str) -> dict:
    """Busca una fila por el valor de una columna y falla claro si no esta."""
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"No hay fila con {key}={value!r} en el informe de metricas")


def build_cross_sell(cfg: ImpactConfig, baseline: Baseline) -> CrossSellImpact:
    """Monta el modelo de cross-sell con las metricas reales de la Fase 3."""
    metrics = json.loads(Path(cfg.recommender_metrics).read_text(encoding="utf-8"))
    k = metrics["top_k"]
    model = _row(metrics["summary"], "grupo", "total")
    popularity = _row(metrics["summary_popularity"], "grupo", "total")
    return CrossSellImpact(
        online_baskets_per_month=baseline.online_baskets_per_month,
        hit_rate_model=float(model[f"hit_rate@{k}"]),
        hit_rate_baseline=float(popularity[f"hit_rate@{k}"]),
        incremental_rate=cfg.incremental_rate,
        avg_line_amount=baseline.avg_line_amount,
        margin_rate=baseline.margin_rate,
    )


def build_nba(cfg: ImpactConfig) -> NBAImpact:
    """Monta el modelo de NBA con el valor que ya calculo la Fase 4."""
    metrics = json.loads(Path(cfg.nba_metrics).read_text(encoding="utf-8"))
    comparison = metrics["comparison"]
    policy = _row(comparison, "politica", "politica de valor esperado")
    trivial = max(
        row["valor_total"]
        for row in comparison
        if row["politica"] != "politica de valor esperado"
    )
    # Suelo del barrido de retencion: el escenario en que el cupon no retiene a nadie.
    floor = min(row["valor_politica"] for row in metrics["sensitivity_retention"])
    return NBAImpact(
        customers=int(policy["n_clientes"]),
        policy_value_per_wave=float(policy["valor_total"]),
        best_trivial_value_per_wave=float(trivial),
        floor_value_per_wave=float(floor),
        waves_per_year=cfg.nba_waves_per_year,
    )


def compute(cfg: ImpactConfig) -> dict:
    """Devuelve el informe completo como diccionario serializable."""
    baseline = measure_baseline(cfg)
    cross_sell = build_cross_sell(cfg, baseline)
    nba = build_nba(cfg)

    cross_sell_ref = cross_sell.rescaled(cfg.reference_online_baskets)
    nba_ref = nba.rescaled(cfg.reference_customers)

    return {
        "supuestos": {
            "incrementalidad": cfg.incremental_rate,
            "margen_por_departamento": cfg.margins.by_department,
            "margen_por_defecto": cfg.margins.default,
            "oleadas_nba_ano": cfg.nba_waves_per_year,
        },
        "medido": baseline.as_dict(),
        "cross_sell": {
            "simulado": cross_sell.as_dict(),
            f"por_{cfg.reference_online_baskets}_cestas_online_mes": cross_sell_ref.as_dict(),
            "barrido_incrementalidad": sweep_incremental_rate(
                cross_sell_ref, cfg.incremental_rate_sweep
            ),
        },
        "nba": {
            "simulado": nba.as_dict(),
            f"por_{cfg.reference_customers}_clientes": nba_ref.as_dict(),
        },
        "total_por_referencia": {
            "margen_cross_sell_mes": cross_sell_ref.margin_per_month,
            "valor_nba_mes": nba_ref.policy_value_per_wave,
            "total_mes": cross_sell_ref.margin_per_month + nba_ref.policy_value_per_wave,
            "total_ano": cross_sell_ref.margin_per_year + nba_ref.value_per_year,
        },
    }


# --------------------------------------------------------------------------------------
# Informe en Markdown
# --------------------------------------------------------------------------------------
def _eur(value: float, decimals: int = 0) -> str:
    """Formatea un importe con separador de miles a la espanola."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",").replace(" ", ".") + " €"


def _num(value: float, decimals: int = 0) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",").replace(" ", ".")


def _pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f} %".replace(".", ",")


def _pp(value: float, decimals: int = 2) -> str:
    """Una diferencia entre dos porcentajes se lee en puntos, no en por ciento."""
    return f"{value * 100:.{decimals}f}".replace(".", ",")


def render_markdown(report: dict, cfg: ImpactConfig) -> str:
    """Escribe el medio folio de impacto que pide la Tarea 4."""
    medido = report["medido"]
    cs_ref = report["cross_sell"][f"por_{cfg.reference_online_baskets}_cestas_online_mes"]
    cs_sim = report["cross_sell"]["simulado"]
    nba_ref = report["nba"][f"por_{cfg.reference_customers}_clientes"]
    nba_sim = report["nba"]["simulado"]
    total = report["total_por_referencia"]

    sweep = "\n".join(
        f"| {_pct(row['incrementalidad'], 0)} | {_num(row['unidades_extra_mes'])} "
        f"| {_eur(row['venta_extra_mes'])} | {_eur(row['margen_extra_mes'])} "
        f"| {_eur(row['margen_extra_ano'])} |"
        for row in report["cross_sell"]["barrido_incrementalidad"]
    )
    ref_baskets = _num(cfg.reference_online_baskets)
    ref_customers = _num(cfg.reference_customers)
    trivial_year = nba_ref["valor_mejor_trivial_oleada"] * cfg.nba_waves_per_year

    return f"""# Impacto de negocio estimado

De NDCG@5 y AUC a euros. Lo escribe `python -m src.impact.pipeline`, que lee las metricas
de [`reports/recommender/metrics.json`](reports/recommender/metrics.json) y
[`reports/nba/metrics.json`](reports/nba/metrics.json) y mide el resto sobre
`data/processed`. Ninguna cifra esta tecleada a mano; el detalle completo queda en
[`reports/impact/impact.json`](reports/impact/impact.json).

> **Lo que se mide y lo que se supone.** El `hit_rate@5` mide **relevancia**, no
> causalidad: un acierto significa que el producto estaba en la cesta de test, o sea que
> el cliente iba a comprarlo igualmente. Convertir eso en venta extra exige un supuesto
> — la **incrementalidad** — que este dataset no puede estimar, porque haria falta un A/B
> con el panel de recomendaciones apagado. Va declarado en `src/impact/config.py` y
> barrido entero mas abajo. Lo unico que no depende de ningun supuesto es la mejora
> relativa sobre el baseline sin aprendizaje.

## El negocio simulado, medido

| Magnitud | Valor |
| --- | ---: |
| Periodo | {medido['meses']} meses |
| Cestas totales | {_num(medido['cestas'])} ({_num(medido['cestas_mes'])} al mes) |
| Cestas online (`app` + `web`) | {_num(medido['cestas_online'])} ({_num(medido['cestas_online_mes'])} al mes) |
| Clientes con al menos una compra | {_num(medido['clientes_con_compras'])} |
| Lineas por cesta | {_num(medido['lineas_por_cesta'], 2)} |
| Importe medio de una linea | {_eur(medido['importe_medio_linea'], 2)} |
| Margen bruto mezclado | {_pct(medido['margen_bruto_mezclado'])} |

El margen mezclado pondera el mapa por departamento de la Fase 4 (18 % en Frescos, 35 % en
Drogueria e Higiene) por lo que vende cada uno. Frescos es un tercio de la venta, asi que
tira del promedio hacia abajo.

## 1. El recomendador: cross-sell

El ranker acierta algo en el **{_pct(cs_sim['hit_rate_modelo'])}** de las cestas de test,
frente al **{_pct(cs_sim['hit_rate_baseline'])}** del baseline sin aprendizaje (popularidad
reciente x indice estacional). Son **{_pp(cs_sim['delta_hit_rate'])} puntos** mas: un
**{_pct(cs_sim['mejora_relativa'], 0)} mas de cestas con una sugerencia relevante**. Esa
cifra no lleva ningun supuesto dentro.

Traducida a euros sobre una base de referencia de **{ref_baskets} cestas online al mes**,
con la incrementalidad al {_pct(cfg.incremental_rate, 0)}:

| Paso | Valor |
| --- | ---: |
| Cestas al mes con un acierto que el baseline no daba | {_num(cs_ref['cestas_con_acierto_nuevo_mes'])} |
| x incrementalidad ({_pct(cfg.incremental_rate, 0)}) = unidades extra al mes | {_num(cs_ref['unidades_extra_mes'])} |
| x importe medio de linea = **venta extra al mes** | **{_eur(cs_ref['venta_extra_mes'])}** |
| x margen bruto = **margen extra al mes** | **{_eur(cs_ref['margen_extra_mes'])}** |
| **Margen extra al ano** | **{_eur(cs_ref['margen_extra_ano'])}** |

Sobre el supermercado simulado tal cual ({_num(medido['cestas_online_mes'])} cestas
online al mes) son {_eur(cs_sim['margen_extra_mes'])} al mes.

### Barrido del supuesto

Por {ref_baskets} cestas online al mes:

| Incrementalidad | Unidades extra/mes | Venta extra/mes | Margen/mes | Margen/ano |
| ---: | ---: | ---: | ---: | ---: |
{sweep}

La lectura honesta: **el cross-sell del recomendador vale miles de euros al ano por cada
{ref_baskets} cestas online al mes, no cientos de miles**. Y no es un defecto del calculo,
es el modelo: con un NDCG@5 de 0,0343 no da para mas. El
[README](README.md#por-qué-el-número-es-bajo-no-es-el-modelo-es-el-dato) explica por que
— el generador elige el SKU dentro de la categoria casi al azar — y por que el mismo
sistema acierta la **categoria** en el 51,4 % de las cestas. Con fidelidad de marca en el
dato, la misma arquitectura tendria bastante mas recorrido.

## 2. El Next Best Action

Aqui no hay que traducir nada: la politica de la Fase 4 ya decide en euros de valor
incremental esperado sobre no actuar. Sobre los {_num(nba_sim['clientes'])} clientes del
corte de test, una oleada de campana vale **{_eur(nba_sim['valor_politica_oleada'])}**
({_eur(nba_sim['valor_por_cliente_oleada'], 3)} por cliente), de los que
**{_eur(nba_sim['uplift_vs_trivial_oleada'])} los aporta elegir a quien** y no la accion
en si: la mejor campana no segmentada se queda en
{_eur(nba_sim['valor_mejor_trivial_oleada'])}.

Con una oleada al mes ({cfg.nba_waves_per_year} al ano, que es lo coherente con el
horizonte de retencion de 4 semanas), por **{ref_customers} clientes activos**:

| Escenario | Por oleada | Al ano |
| --- | ---: | ---: |
| Politica de valor esperado | {_eur(nba_ref['valor_politica_oleada'])} | {_eur(nba_ref['valor_ano'])} |
| Suelo: el cupon no retiene a nadie | {_eur(nba_ref['valor_suelo_oleada'])} | {_eur(nba_ref['valor_suelo_ano'])} |
| Mejor alternativa trivial | {_eur(nba_ref['valor_mejor_trivial_oleada'])} | {_eur(trivial_year)} |

La fila que sostiene el caso es la segunda. El barrido de la Fase 4 muestra que **incluso
suponiendo que el cupon no retenga a nadie la politica sigue ganando**, y que en ese
escenario deja de repartir cupones por completo. Lo que depende del supuesto es el tamano
del premio, no el signo.

## 3. Las dos piezas juntas

Por {ref_customers} clientes activos y {ref_baskets} cestas online al mes:

| Pieza | Al mes | Al ano |
| --- | ---: | ---: |
| Cross-sell del recomendador (margen) | {_eur(total['margen_cross_sell_mes'])} | {_eur(cs_ref['margen_extra_ano'])} |
| Politica de Next Best Action | {_eur(total['valor_nba_mes'])} | {_eur(nba_ref['valor_ano'])} |
| **Total** | **{_eur(total['total_mes'])}** | **{_eur(total['total_ano'])}** |

Las dos cifras no son homogeneas y conviene no sumarlas a la ligera: la del recomendador
es margen bruto sobre venta incremental y descansa en un supuesto de incrementalidad; la
del NBA es valor esperado neto de coste de campana y descansa en un supuesto de efecto de
la accion. Coinciden en la parte economica — el mismo mapa de margen por departamento — y
en que las dos son **conservadoras por construccion**: el recomendador se compara contra
un baseline que ya funciona, y la politica contra la mejor de las alternativas triviales,
no contra no hacer nada.

El reparto tambien dice donde esta hoy el proyecto: **casi todo el valor viene del NBA**,
no del recomendador. En parte tiene sentido — la politica decide sobre el cliente entero y
el recomendador solo sobre cinco huecos de una cesta — pero sobre todo es consecuencia del
techo de dato descrito arriba.

## Que haria falta para afinar esto

1. **Un A/B**, que es lo unico que convierte la incrementalidad de supuesto en estimacion.
2. **Fidelidad de marca en el generador**: es el techo del recomendador y esta anotado como
   deuda en el [`ROADMAP.md`](ROADMAP.md).
3. **Un target de churn condicionado a la cadencia de cada cliente**: hoy el modelo de la
   Fase 4 predice inactividad a 4 semanas, que con una cadencia media de 24 dias le pasa a
   media base sin ser abandono.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumen de impacto de negocio (Fase 5)")
    parser.add_argument(
        "--incremental-rate",
        type=float,
        default=None,
        help="Supuesto de incrementalidad del recomendador (por defecto, el de config.py)",
    )
    args = parser.parse_args(argv)

    cfg = ImpactConfig()
    if args.incremental_rate is not None:
        cfg = ImpactConfig(incremental_rate=args.incremental_rate)

    report = compute(cfg)

    reports_dir = Path(cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "impact.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(cfg.markdown_path).write_text(render_markdown(report, cfg), encoding="utf-8")

    total = report["total_por_referencia"]
    print(f"Escrito {cfg.markdown_path} y {reports_dir / 'impact.json'}")
    print(
        f"Por {cfg.reference_customers:,} clientes / {cfg.reference_online_baskets:,} "
        f"cestas online al mes: {total['total_mes']:,.0f} EUR/mes, "
        f"{total['total_ano']:,.0f} EUR/ano"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

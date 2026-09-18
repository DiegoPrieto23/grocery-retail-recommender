"""Verifica la capa comun de necesidad de categoria entre el NBA y el recomendador (M7).

    python -m src.nba.verify_category_need

`CLAUDE.md` no deja citar una cifra que no recalcule un script. Las que sostienen el punto
M7 -- que la politica elegia la categoria equivocada, y cuanto -- se recalculan aqui, sobre
los artefactos de `data/processed` y la tabla `predictions/nba_actions.parquet`.

## Que se comprueba, y por que estas tres cosas

1. **La curva de necesidad.** Que el `overdue_ratio` de la Tarea 2 -- el mismo que usa el
   recomendador para decidir si una categoria toca -- tambien predice la etiqueta del NBA,
   que es otra: comprar la categoria en los 7 dias siguientes al corte. Si no lo hiciera,
   compartir la capa no tendria sentido y el punto M7 se cerraria con un "no compensa".

2. **A que categoria apunta cada accion.** Con la tabla de acciones ya escrita, cuanta
   necesidad tenian de verdad las categorias elegidas. Es el diagnostico que motivo el
   cambio: sin el factor de relevancia, el `argmax` sobre categorias no maximizaba lo que
   le sirve al cliente sino lo que minimiza la fuga de descuento, y acababa eligiendo la
   categoria que el cliente casi seguro **no** iba a comprar.

3. **El contraste.** Lo mismo medido sobre la tabla de acciones **anterior** al cambio,
   congelada en `reports/nba/category_need_pre_m7.json`. Dice cuanto se movio el reparto.
   No se recalcula la politica vieja: se compara contra la foto congelada, que es el
   criterio que ya usan `baseline_pre_a1.json` y compania.

No reentrena nada: lee `p_purchase` y `p_churn` de la tabla ya escrita por
`src.nba.pipeline`. Por eso corre en segundos y no en minutos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.nba.config import NBAConfig
from src.recommender.formulas import (
    PANDAS_OPS,
    expected_repurchase_days,
    household_factor,
    mean_gap_days,
    overdue_ratio,
)

# Bandas del ciclo de reposicion. Se corta fino cerca de 1 porque es donde esta el codo:
# antes de 1 la categoria no toca, despues empieza la cola de categorias abandonadas.
NEED_BANDS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf)


def category_state(processed: Path, cutoff: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    """Estado del ciclo de cada par `(cliente, categoria)` en el corte, y que paso despues.

    Replica en pandas lo que `nba.features.category_features` hace en Spark, con las
    **mismas** funciones de `recommender.formulas`: si las dos dieran cosas distintas, la
    capa no seria comun y este verificador no valdria de nada.
    """
    baskets = pd.read_parquet(
        processed / "baskets.parquet", columns=["basket_id", "customer_id", "basket_day"]
    )
    items = pd.read_parquet(
        processed / "basket_items.parquet", columns=["basket_id", "product_id"]
    )
    products = pd.read_parquet(
        processed / "products.parquet",
        columns=["product_id", "category", "typical_repurchase_days"],
    )
    customers = pd.read_parquet(
        processed / "customers.parquet", columns=["customer_id", "household_size_est"]
    )

    baskets["basket_day"] = pd.to_datetime(baskets["basket_day"])
    baskets = baskets[baskets["customer_id"].notna()]
    lines = items.merge(baskets, on="basket_id").merge(
        products[["product_id", "category"]], on="product_id"
    )

    past = lines[lines["basket_day"] < cutoff]
    horizon = cutoff + pd.Timedelta(days=NBAConfig().category_horizon_days)
    future = lines[(lines["basket_day"] >= cutoff) & (lines["basket_day"] < horizon)]
    future = future[["customer_id", "category"]].drop_duplicates().assign(bought=1)

    grouped = past.groupby(["customer_id", "category"])["basket_day"]
    state = pd.DataFrame(
        {
            "last": grouped.max(),
            "first": grouped.min(),
            "n_purchase_days": grouped.nunique(),
        }
    ).reset_index()

    typical = products.groupby("category")["typical_repurchase_days"].median().rename("typical")
    state = state.merge(typical, on="category").merge(customers, on="customer_id", how="left")

    state["days_since"] = (cutoff - state["last"]).dt.days.astype(float)
    span = (state["last"] - state["first"]).dt.days.astype(float)
    factor = household_factor(state["household_size_est"], PANDAS_OPS)
    gap = mean_gap_days(state["n_purchase_days"], span, PANDAS_OPS)
    expected = expected_repurchase_days(
        state["n_purchase_days"], gap, state["typical"], factor, PANDAS_OPS
    )
    state["cat_overdue_ratio"] = overdue_ratio(state["days_since"], expected)

    # El mismo lookback que acota los candidatos de la politica: sin el, la cola de
    # categorias abandonadas hace anos domina la foto y no es lo que se esta midiendo.
    state = state[state["days_since"] <= lookback_days]
    state = state.merge(future, on=["customer_id", "category"], how="left")
    state["bought"] = state["bought"].fillna(0.0)
    state["band"] = pd.cut(state["cat_overdue_ratio"], NEED_BANDS, right=False)
    return state


def need_curve(state: pd.DataFrame) -> pd.DataFrame:
    """Tasa real de compra a 7 dias por banda de `overdue_ratio`."""
    out = (
        state.groupby("band", observed=True)
        .agg(n_pares=("bought", "size"), tasa_real=("bought", "mean"))
        .reset_index()
    )
    out["band"] = out["band"].astype(str)
    return out


def targeting(actions: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """A que necesidad de categoria apunta cada accion, y que paso de verdad."""
    acted = actions[actions["action"] != "ninguna_accion"]
    joined = acted.merge(
        state[["customer_id", "category", "cat_overdue_ratio", "bought"]],
        on=["customer_id", "category"],
        how="inner",
    )
    return (
        joined.groupby("action")
        .agg(
            n=("bought", "size"),
            overdue_mediano=("cat_overdue_ratio", "median"),
            pct_no_vencida=("cat_overdue_ratio", lambda s: float((s < 1.0).mean())),
            pct_recien_repuesta=("cat_overdue_ratio", lambda s: float((s < 0.5).mean())),
            tasa_real_7d=("bought", "mean"),
        )
        .reset_index()
    )


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}".replace(".", ",")


def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def render(report: dict) -> str:
    """Informe en Markdown, con las tres secciones que justifican el punto M7."""
    lines = [
        "# Capa comun de necesidad de categoria (punto M7)",
        "",
        "> Lo recalcula `python -m src.nba.verify_category_need`. Ninguna cifra de este",
        "> fichero esta escrita a mano.",
        "",
        f"Corte de test: **{report['cutoff']}** · "
        f"horizonte de la etiqueta: {report['horizon_days']} dias · "
        f"pares con compra en los ultimos {report['lookback_days']} dias: "
        f"**{_fmt_int(report['n_pairs'])}**",
        "",
        "## 1. El ciclo de reposicion predice tambien la etiqueta del NBA",
        "",
        "`overdue_ratio` es la senal con la que el recomendador decide si una categoria",
        "toca. La etiqueta del NBA es otra cosa -- comprar la categoria en los 7 dias",
        "siguientes --, y aun asi la ordena: por eso la capa se puede compartir.",
        "",
        "| banda de `overdue_ratio` | pares | tasa real de compra a 7 d |",
        "| --- | ---: | ---: |",
    ]
    for row in report["need_curve"]:
        lines.append(
            f"| {row['band']} | {_fmt_int(row['n_pares'])} | {_fmt_pct(row['tasa_real'])} |"
        )
    lines += [
        "",
        f"La tasa sube del {_fmt_pct(report['need_curve'][0]['tasa_real'])} en la primera",
        f"banda al {_fmt_pct(report['peak_rate'])} en `{report['peak_band']}` y vuelve a",
        f"bajar al {_fmt_pct(report['need_curve'][-1]['tasa_real'])} en la cola. Por eso",
        "`category_need_weight` se satura en 1 en vez de seguir creciendo: pasada la",
        "banda del ciclo, un ratio alto es una categoria abandonada, no una necesidad.",
        "",
        "## 2. A que categoria apunta cada accion",
        "",
        "| accion | n | `overdue` mediano | % no vencida | % recien repuesta | tasa real 7 d |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["targeting"]:
        lines.append(
            f"| `{row['action']}` | {_fmt_int(row['n'])} | {row['overdue_mediano']:.2f} | "
            f"{_fmt_pct(row['pct_no_vencida'])} | {_fmt_pct(row['pct_recien_repuesta'])} | "
            f"{_fmt_pct(row['tasa_real_7d'])} |"
        )
    lines += [
        "",
        f"Referencia del pool completo de candidatos: el {_fmt_pct(report['pool_not_due'])}",
        f"de los pares no esta vencido y el {_fmt_pct(report['pool_fresh'])} esta recien",
        f"repuesto, con una tasa real del {_fmt_pct(report['pool_rate'])}. Una accion que",
        "apunte por encima de esas cifras esta eligiendo peor que el azar del pool.",
        "",
    ]

    if report.get("before"):
        lines += [
            "## 3. Antes y despues del factor de relevancia",
            "",
            "El antes sale de `reports/nba/category_need_pre_m7.json`, congelado con la",
            "tabla de acciones que escribia la politica sin el factor.",
            "",
            "| accion | % no vencida antes | ahora | tasa real antes | ahora |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        antes = {r["action"]: r for r in report["before"]}
        # La accion de categoria se llamaba `recomendar_producto` antes del punto M7.
        alias = {"recomendar_categoria": "recomendar_producto"}
        for row in report["targeting"]:
            old = antes.get(row["action"]) or antes.get(alias.get(row["action"], ""))
            if old is None:
                continue
            lines.append(
                f"| `{row['action']}` | {_fmt_pct(old['pct_no_vencida'])} | "
                f"**{_fmt_pct(row['pct_no_vencida'])}** | "
                f"{_fmt_pct(old['tasa_real_7d'])} | **{_fmt_pct(row['tasa_real_7d'])}** |"
            )
        lines.append("")

    return "\n".join(lines)


def _frozen_before(reports: Path) -> list[dict] | None:
    """El `targeting` de la foto anterior al punto M7, si esta congelada."""
    path = reports / "category_need_pre_m7.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("targeting")


def compute(cfg: NBAConfig, processed: Path, predictions: Path, reports: Path | None = None) -> dict:
    cutoff = pd.Timestamp(cfg.test_cutoff)
    state = category_state(processed, cutoff, cfg.category_lookback_days)
    actions = pd.read_parquet(predictions / "nba_actions.parquet")

    curve = need_curve(state)
    peak = curve.loc[curve["tasa_real"].idxmax()]
    return {
        "cutoff": str(cfg.test_cutoff),
        "horizon_days": cfg.category_horizon_days,
        "lookback_days": cfg.category_lookback_days,
        "n_pairs": int(len(state)),
        "need_curve": curve.to_dict("records"),
        "peak_band": str(peak["band"]),
        "peak_rate": float(peak["tasa_real"]),
        "targeting": targeting(actions, state).to_dict("records"),
        "pool_not_due": float((state["cat_overdue_ratio"] < 1.0).mean()),
        "pool_fresh": float((state["cat_overdue_ratio"] < 0.5).mean()),
        "pool_rate": float(state["bought"].mean()),
        "before": _frozen_before(reports) if reports is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument("--reports", type=Path, default=Path("reports/nba"))
    parser.add_argument("--no-write", action="store_true", help="No escribir informes.")
    args = parser.parse_args(argv)

    report = compute(NBAConfig(), args.processed, args.predictions, args.reports)
    markdown = render(report)
    print(markdown)

    if not args.no_write:
        args.reports.mkdir(parents=True, exist_ok=True)
        (args.reports / "category_need.md").write_text(markdown, encoding="utf-8")
        (args.reports / "category_need.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

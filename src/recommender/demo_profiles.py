"""Un ejemplo real de cada uno de los cuatro perfiles de `CHALLENGE.md`.

    python -m src.recommender.demo_profiles

No entrena ni recalcula nada: lee lo que dejo `python -m src.recommender.pipeline` en
`predictions/` y lo traduce a algo legible -- quien es el cliente, que llevaba ya en el
carrito, que le recomendo el sistema, que fuente propuso cada producto y si acerto.

Lo que se ve aqui es la tesis de la Tarea 3a: **el ranker es el mismo para los cuatro
perfiles; lo que cambia es que fuentes tienen algo que decir**. En el perfil 1 (cliente
nuevo, carrito vacio) solo hay popularidad; en el 4 (recurrente a media compra) coinciden
historial, co-compra, ALS y sesion, y el ranker arbitra entre ellas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.recommender.candidates import SOURCE_NAMES
from src.recommender.splits import PROFILE_LABELS

# Como se lee cada bandera de fuente en la salida.
SOURCE_LABELS: dict[str, str] = {
    "pop": "popularidad/estacionalidad",
    "aff": "co-compra (producto)",
    "cataff": "co-compra (categoria)",
    "hist": "historial + recompra",
    "als": "ALS",
}

_REQUIRED = (
    "recommendations_test.parquet",
    "recommender_test_queries.parquet",
    "recommender_test_context.parquet",
    "recommender_per_query.parquet",
)


def load_artifacts(predictions_dir: Path, processed_dir: Path) -> dict[str, pd.DataFrame]:
    """Lee las salidas del pipeline y el catalogo, o explica que falta ejecutar."""
    missing = [name for name in _REQUIRED if not (predictions_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Faltan salidas de la Fase 3 en {predictions_dir}: {', '.join(missing)}.\n"
            "Ejecuta antes:  python -m src.recommender.pipeline"
        )
    return {
        "recommendations": pd.read_parquet(predictions_dir / "recommendations_test.parquet"),
        "queries": pd.read_parquet(predictions_dir / "recommender_test_queries.parquet"),
        "context": pd.read_parquet(predictions_dir / "recommender_test_context.parquet"),
        "per_query": pd.read_parquet(predictions_dir / "recommender_per_query.parquet"),
        "products": pd.read_parquet(processed_dir / "products.parquet"),
        "customers": pd.read_parquet(processed_dir / "customers.parquet"),
    }


def pick_examples(per_query: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Una cesta por perfil, para ver la mecanica del sistema.

    Se elige la **mediana entre las cestas del perfil que aciertan al menos un producto**,
    no la mejor de todas y no la mediana global. La mediana global de cualquier perfil
    puntua 0 -- solo una de cada ocho cestas recibe algun acierto --, y una tabla de cinco
    fallos no ensena nada sobre como se combinan las fuentes. Coger la mejor seria el
    extremo contrario: un escaparate.

    El sesgo queda declarado en el informe junto al `hit_rate` real de cada perfil, que es
    la cifra honesta; las metricas agregadas viven en `reports/recommender/metrics.md`.
    """
    merged = per_query.merge(queries.drop(columns=["profile", "n_target"]), on="basket_id")
    hit_rate = merged.groupby("profile")["hit"].mean()

    picks = []
    for profile in sorted(merged["profile"].unique()):
        subset = merged.loc[merged["profile"] == profile]
        with_hits = subset.loc[subset["n_hits"] > 0]
        pool = with_hits if not with_hits.empty else subset
        pool = pool.sort_values(["ndcg", "basket_id"], kind="stable")
        row = pool.iloc[len(pool) // 2].copy()
        row["profile_hit_rate"] = hit_rate.loc[profile]
        row["profile_n_queries"] = int(len(subset))
        picks.append(row)
    return pd.DataFrame(picks).reset_index(drop=True)


def _describe(products: pd.DataFrame, product_ids: list[str]) -> list[str]:
    catalog = products.set_index("product_id")
    out = []
    for pid in product_ids:
        if pid in catalog.index:
            row = catalog.loc[pid]
            out.append(f"{row['category']} - {row['brand']} ({pid})")
        else:
            out.append(pid)
    return out


def _sources_of(row: pd.Series) -> str:
    active = [SOURCE_LABELS[s] for s in SOURCE_NAMES if row.get(f"src_{s}", 0) > 0]
    extra = []
    if row.get("sess_viewed", 0) > 0:
        extra.append("visto en la sesion")
    if row.get("is_on_promo", 0) > 0:
        extra.append("en promocion")
    return ", ".join(active + extra) if active or extra else "-"


def render_example(example: pd.Series, art: dict[str, pd.DataFrame]) -> str:
    """Dibuja un caso completo en Markdown."""
    basket_id = example["basket_id"]
    profile = int(example["profile"])
    products = art["products"]

    context = art["context"].loc[art["context"]["basket_id"] == basket_id]
    cart = _describe(products, context.loc[context["role"] == "prefix", "product_id"].tolist())
    target = context.loc[context["role"] == "target", "product_id"].tolist()

    recs = art["recommendations"]
    recs = recs.loc[recs["basket_id"] == basket_id].sort_values("rank")

    customer = example.get("customer_id")
    if pd.isna(customer):
        who = "compra anonima (sin `customer_id`)"
    else:
        master = art["customers"].set_index("customer_id")
        if customer in master.index:
            row = master.loc[customer]
            who = (
                f"`{customer}` - {row['loyalty_tier']}, hogar de "
                f"{row['household_size_est']}, {row['city']}"
            )
        else:
            who = f"`{customer}`"

    lines = [
        f"### {PROFILE_LABELS[profile].replace('-', '·', 1)}",
        "",
        f"_Caso mediano entre las cestas de este perfil que aciertan algo. En el conjunto"
        f" del perfil acierta al menos un producto el"
        f" {example['profile_hit_rate']:.1%} de las"
        f" {int(example['profile_n_queries']):,} cestas._",
        "",
        f"- **Cesta**: `{basket_id}`, canal `{example['channel']}`, {example['basket_day']}",
        f"- **Cliente**: {who}",
        f"- **Ya en el carrito** ({len(cart)}): " + ("; ".join(cart) if cart else "_vacio_"),
        f"- **Falta por anadir** ({len(target)}): " + "; ".join(_describe(products, target)),
        f"- **Resultado**: NDCG@5 = {example['ndcg']:.3f}, Recall@5 = {example['recall']:.3f}"
        f" ({int(example['n_hits'])} de {int(example['n_target'])})",
        "",
        "| # | Recomendacion | Fuentes que lo propusieron | ¿Acierto? |",
        "| ---: | --- | --- | :---: |",
    ]
    for _, rec in recs.iterrows():
        name = _describe(products, [rec["product_id"]])[0]
        lines.append(
            f"| {int(rec['rank'])} | {name} | {_sources_of(rec)} | "
            f"{'SI' if rec['label'] == 1 else 'no'} |"
        )
    return "\n".join(lines)


def build_report(art: dict[str, pd.DataFrame]) -> str:
    """Informe completo: los cuatro casos mas la lectura de que fuente cubre cada perfil."""
    examples = pick_examples(art["per_query"], art["queries"])

    recs = art["recommendations"]
    coverage = (
        recs.groupby("profile")[[f"src_{s}" for s in SOURCE_NAMES]]
        .mean()
        .rename(columns={f"src_{s}": SOURCE_LABELS[s] for s in SOURCE_NAMES})
    )
    coverage.index = [PROFILE_LABELS[int(i)] for i in coverage.index]

    header = [
        "# Los cuatro perfiles de cliente, con un caso real de cada uno",
        "",
        "Generado por `python -m src.recommender.demo_profiles` sobre las cestas de test de",
        "la Fase 3. Sirve para ver **la mecanica**: que fuentes se activan en cada perfil y",
        "como el ranker las arbitra. Las metricas agregadas -- las cifras honestas -- estan en",
        "`reports/recommender/metrics.md`; cada caso lleva ademas el `hit_rate` real de su",
        "perfil al lado, para no confundir un ejemplo con un resultado.",
        "",
        "## De donde sale cada recomendacion",
        "",
        "Proporcion del top-5 que propuso cada fuente, por perfil. Es la tesis de la Tarea 3a",
        "en una tabla: el ranker es el mismo en las cuatro filas, lo que cambia es quien tiene",
        "algo que decir.",
        "",
        "| Perfil | " + " | ".join(coverage.columns) + " |",
        "| --- | " + " | ".join("---:" for _ in coverage.columns) + " |",
    ]
    for name, row in coverage.iterrows():
        header.append(f"| {name} | " + " | ".join(f"{v:.0%}" for v in row) + " |")

    body = [render_example(example, art) for _, example in examples.iterrows()]
    return "\n".join(header) + "\n\n---\n\n" + "\n\n---\n\n".join(body) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports", type=Path, default=Path("reports/recommender"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    art = load_artifacts(args.predictions, args.processed)
    report = build_report(art)

    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "profiles_demo.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nInforme escrito en {out}")


if __name__ == "__main__":
    main()

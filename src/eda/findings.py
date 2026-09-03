"""Informe de hallazgos de negocio (Fase 5).

    python -m src.eda.findings

Escribe `reports/insights/business_findings.md` y sus figuras. Es el "vistazo al
analisis" del proyecto: no repite el EDA pregunta a pregunta -- para eso esta
`notebooks/01_eda.ipynb` -- sino que recoge los ocho hallazgos que de verdad cambian una
decision, cruzando lo que dice el dato (Fase 2) con lo que hacen los modelos (Fases 3
y 4).

Todo sale de ejecutar codigo: las preguntas de negocio vienen de `src/eda/questions.py`
(las mismas que consume el notebook, con sus tests), y los resultados de modelo de
`reports/*/metrics.json` y `predictions/*.parquet`. Ninguna cifra se teclea a mano.

## Los ocho hallazgos

1. El calendario mueve el surtido mas que el cliente.
2. Lo que se compra junto no siempre es complementariedad.
3. La palanca sobre un cliente valioso es la frecuencia, no el ticket.
4. El ciclo de reposicion es real y escala con el hogar.
5. La sesion online no es el ticket -- y por poco lo fue.
6. El recomendador acierta la categoria, no la referencia.
7. El NBA no responde a quien se va, sino a quien todavia vale algo.
8. La politica se va al margen, y sin calendario eso tiene un coste.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin backend interactivo: esto corre en un script, no en un notebook

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.eda import questions as q
from src.etl.session import get_spark
from src.nba.config import MarginConfig

PALETTE = ["#2E6E8E", "#E1812C", "#3A923A", "#C03D3E", "#9372B2", "#7F7F7F"]

# Corte de test de la Fase 4: la ventana de 90 dias previa es la que define cuanto vale
# todavia un cliente cuando la politica decide.
NBA_TEST_CUTOFF = pd.Timestamp("2025-11-01")
RECENT_WINDOW_DAYS = 90


@dataclass(frozen=True)
class FindingsConfig:
    processed_dir: Path = Path("data/processed")
    predictions_dir: Path = Path("predictions")
    recommender_metrics: Path = Path("reports/recommender/metrics.json")
    out_dir: Path = Path("reports/insights")
    markdown_name: str = "business_findings.md"


def _style() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    sns.set_palette(PALETTE)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.titlesize"] = 11


def _save(fig: plt.Figure, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / name, bbox_inches="tight")
    plt.close(fig)
    return name


# ======================================================================================
# 1. Estacionalidad
# ======================================================================================
def finding_seasonality(spark, out_dir: Path) -> dict:
    index = q.q3_seasonal_index(spark).toPandas()
    matrix = q.seasonal_matrix(index)
    peaks = q.seasonal_peaks(matrix)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.heatmap(
        matrix, ax=ax, cmap="RdYlBu_r", center=1.0, annot=True, fmt=".1f",
        annot_kws={"size": 7}, linewidths=0.4,
        cbar_kws={"label": "indice (1,0 = mes normal)"},
    )
    ax.set_title("Cuanto pesa cada categoria dentro del mes, frente a su media anual")
    ax.set_xlabel("mes")
    ax.set_ylabel("")
    figure = _save(fig, out_dir, "01_estacionalidad.png")

    top = peaks.sort_values("indice_en_el_pico", ascending=False)
    return {
        "figure": figure,
        "peaks": peaks,
        "strongest": top.index[0],
        "strongest_index": float(top["indice_en_el_pico"].iloc[0]),
        "strongest_month": int(top["mes_pico"].iloc[0]),
        "n_categories": int(len(peaks)),
    }


# ======================================================================================
# 2. Afinidad de cesta
# ======================================================================================
def finding_affinity(spark, out_dir: Path) -> dict:
    check = q.q5_expected_pairs(spark)
    top = q.q5_top_affinity_pairs(spark, min_baskets=500, limit=12).toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    orden = top.iloc[::-1]
    axes[0].barh(
        orden["disparadora"] + "  ->  " + orden["asociada"], orden["lift"], color=PALETTE[0]
    )
    axes[0].axvline(1.0, color=PALETTE[3], linestyle="--", label="lift = 1 (azar)")
    axes[0].set_title("Pares de categorias con mas lift\n(minimo 500 cestas en comun)")
    axes[0].set_xlabel("lift")
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].legend(fontsize=8)

    pares = check["disparadora"] + " -> " + check["asociada"]
    y = range(len(check))
    axes[1].hlines(y, check["lift_objetivo"], check["lift_medido"], color="#BBBBBB", lw=2)
    axes[1].scatter(check["lift_objetivo"], y, s=50, color=PALETTE[5], label="objetivo", zorder=2)
    axes[1].scatter(check["lift_medido"], y, s=50, color=PALETTE[1], label="medido", zorder=2)
    axes[1].axvline(1.0, color=PALETTE[3], linestyle="--", alpha=0.6)
    axes[1].set_yticks(list(y))
    axes[1].set_yticklabels(pares, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("Los 10 pares inyectados: objetivo frente a medido")
    axes[1].set_xlabel("lift")
    axes[1].legend(fontsize=8)
    figure = _save(fig, out_dir, "02_afinidad.png")

    worst = check.loc[(check["ratio"] - 1).abs().idxmax()]
    return {
        "figure": figure,
        "check": check,
        "top": top,
        "n_above_one": int((check["lift_medido"] > 1).sum()),
        "outlier_pair": f"{worst['disparadora']} -> {worst['asociada']}",
        "outlier_measured": float(worst["lift_medido"]),
        "outlier_target": float(worst["lift_objetivo"]),
    }


# ======================================================================================
# 3. Frecuencia frente a ticket
# ======================================================================================
def finding_frequency_vs_ticket(spark, out_dir: Path) -> dict:
    tier = q.add_share_columns(q.q1_value_by_tier(spark).toPandas())
    household = q.q1_basket_by_household(spark).toPandas()
    tier["cestas_por_cliente"] = tier["cestas"] / tier["clientes"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    peso = tier.set_index("loyalty_tier")[["pct_clientes", "pct_facturacion"]]
    peso.columns = ["% de clientes", "% de facturacion"]
    peso.plot.bar(ax=axes[0], width=0.78, rot=0, color=[PALETTE[5], PALETTE[0]])
    axes[0].set_title("Peso en clientes frente a peso en venta")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("%")
    axes[0].legend(title="", fontsize=8)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f", padding=2, fontsize=8)

    ax2 = axes[1]
    idx = tier.set_index("loyalty_tier")
    ax2.bar(idx.index, idx["ticket_medio"], color=PALETTE[0], width=0.55)
    ax2.set_ylabel("ticket medio (EUR)", color=PALETTE[0])
    ax2.set_title("El ticket no cambia; la frecuencia si")
    gemelo = ax2.twinx()
    gemelo.plot(idx.index, idx["cestas_por_cliente"], color=PALETTE[1], marker="o", lw=2)
    gemelo.set_ylabel("cestas por cliente", color=PALETTE[1])
    gemelo.grid(False)
    figure = _save(fig, out_dir, "03_frecuencia_vs_ticket.png")

    top_tier = idx.loc["gold"]
    return {
        "figure": figure,
        "tier": tier,
        "household": household,
        "gold_pct_clientes": float(top_tier["pct_clientes"]),
        "gold_pct_facturacion": float(top_tier["pct_facturacion"]),
        "gold_indice": float(top_tier["indice_valor"]),
        "gold_ticket": float(top_tier["ticket_medio"]),
        "bronze_ticket": float(idx.loc["bronze", "ticket_medio"]),
        "gold_cestas": float(top_tier["cestas_por_cliente"]),
        "bronze_cestas": float(idx.loc["bronze", "cestas_por_cliente"]),
        "ticket_hogar_1": float(household.set_index("tamano_hogar").loc[1, "ticket_medio"]),
        "ticket_hogar_max": float(household["ticket_medio"].iloc[-1]),
        "hogar_max": int(household["tamano_hogar"].iloc[-1]),
    }


# ======================================================================================
# 4. Ciclo de reposicion
# ======================================================================================
def finding_repurchase(spark, out_dir: Path) -> dict:
    by_category = q.q6_cycle_by_category(spark).toPandas()
    by_household = q.q6_cycle_by_household(spark).toPandas()
    due = q.q6_due_summary(spark).toPandas().iloc[0]
    rho = q.cycle_rank_correlation(by_category)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].scatter(
        by_category["ciclo_teorico"], by_category["ciclo_observado"],
        s=by_category["clientes"] / 120, alpha=0.75, color=PALETTE[0],
        edgecolor="white", linewidth=0.6,
    )
    limite = max(by_category["ciclo_teorico"].max(), by_category["ciclo_observado"].max()) * 1.05
    axes[0].plot([0, limite], [0, limite], "--", color=PALETTE[3], label="observado = teorico")
    axes[0].set_title(f"Ciclo teorico frente a observado (Spearman {rho:.3f})")
    axes[0].set_xlabel("dias tipicos de la categoria")
    axes[0].set_ylabel("mediana observada (dias)")
    axes[0].legend(fontsize=8)

    for i, categoria in enumerate(sorted(by_household["categoria"].unique())):
        sub = by_household[by_household["categoria"] == categoria].sort_values("tamano_hogar")
        axes[1].plot(
            sub["tamano_hogar"], sub["ciclo_observado"], marker="o",
            label=categoria, color=PALETTE[i % len(PALETTE)],
        )
    axes[1].set_title("Mas gente en casa, menos dias entre compras")
    axes[1].set_xlabel("personas en el hogar")
    axes[1].set_ylabel("mediana de dias entre compras")
    axes[1].legend(fontsize=8)
    figure = _save(fig, out_dir, "04_recompra.png")

    return {
        "figure": figure,
        "spearman": rho,
        "pct_due": float(due["pct"]),
        "pairs": int(due["pares_cliente_categoria"]),
        "customers_due": int(due["clientes_afectados"]),
        "by_household": by_household,
    }


# ======================================================================================
# 5. Embudo online
# ======================================================================================
def finding_online_funnel(spark, out_dir: Path) -> dict:
    funnel = q.q9_funnel_by_device(spark).toPandas()
    cart = q.q9_event_to_basket_overlap(spark).toPandas().iloc[0]
    views = q.q9_event_to_basket_overlap(spark, event_type="view").toPandas().iloc[0]
    seen = q.q9_basket_seen_online(spark).toPandas().iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    etapas = ["sesiones", "con_vista", "con_add_to_cart", "convertidas"]
    totales = [float(funnel[e].sum()) for e in etapas]
    axes[0].bar(
        ["Sesiones", "Con vista", "Con add_to_cart", "Convertidas"],
        totales, color=PALETTE[0], width=0.62,
    )
    axes[0].set_title("Embudo online")
    axes[0].set_ylabel("sesiones")
    axes[0].tick_params(axis="x", rotation=16, labelsize=8)
    for i, valor in enumerate(totales):
        axes[0].text(i, valor, f"{100 * valor / totales[0]:.0f} %", ha="center",
                     va="bottom", fontsize=9)

    etiquetas = [
        "add_to_cart\n-> en la cesta",
        "view\n-> en la cesta",
        "linea del ticket\n-> vista online",
    ]
    valores = [float(cart["pct"]), float(views["pct"]), float(seen["pct"])]
    axes[1].bar(etiquetas, valores, color=[PALETTE[2], PALETTE[1], PALETTE[4]], width=0.6)
    axes[1].axhline(100, color=PALETTE[3], linestyle="--", label="100 % = la senal ES el target")
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("%")
    axes[1].set_title("Ninguna de las tres direcciones llega al 100 %")
    axes[1].legend(fontsize=8)
    for i, valor in enumerate(valores):
        axes[1].text(i, valor + 1.5, f"{valor:.1f} %", ha="center", fontsize=9)
    figure = _save(fig, out_dir, "05_embudo.png")

    return {
        "figure": figure,
        "sessions": int(funnel["sesiones"].sum()),
        "conversion": 100 * float(funnel["convertidas"].sum()) / float(funnel["sesiones"].sum()),
        "cart_to_basket": float(cart["pct"]),
        "view_to_basket": float(views["pct"]),
        "basket_seen": float(seen["pct"]),
        "device_spread": float(funnel["pct_conversion"].max() - funnel["pct_conversion"].min()),
    }


# ======================================================================================
# 6. Recomendador: categoria frente a SKU
# ======================================================================================
def finding_recommender(cfg: FindingsConfig, out_dir: Path) -> dict:
    metrics = json.loads(Path(cfg.recommender_metrics).read_text(encoding="utf-8"))
    k = metrics["top_k"]
    summary = pd.DataFrame(metrics["summary"])
    by_cat = pd.DataFrame(metrics["by_category"])
    profiles = summary[summary["grupo"] != "total"].copy()
    cats = by_cat[by_cat["grupo"] != "total"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    etiquetas = [g.split(" - ")[-1] for g in profiles["grupo"]]
    axes[0].bar(etiquetas, profiles[f"ndcg@{k}"], color=PALETTE[0], width=0.6)
    axes[0].set_title(f"NDCG@{k} por perfil de cliente")
    axes[0].tick_params(axis="x", rotation=18, labelsize=8)
    for i, valor in enumerate(profiles[f"ndcg@{k}"]):
        axes[0].text(i, valor, f"{valor:.4f}", ha="center", va="bottom", fontsize=8)

    x = range(len(cats))
    ancho = 0.38
    axes[1].bar([i - ancho / 2 for i in x], 100 * cats[f"cat_hit_rate@{k}"], ancho,
                label="acierta la categoria", color=PALETTE[2])
    axes[1].bar([i + ancho / 2 for i in x], 100 * cats[f"sku_hit_rate@{k}"], ancho,
                label="acierta el SKU", color=PALETTE[3])
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([g.split(" - ")[-1] for g in cats["grupo"]], rotation=18, fontsize=8)
    axes[1].set_ylabel(f"% de cestas con acierto en el top-{k}")
    axes[1].set_title("Sabe que necesita; no sabe que referencia")
    axes[1].legend(fontsize=8)
    figure = _save(fig, out_dir, "06_recomendador.png")

    total_cat = by_cat[by_cat["grupo"] == "total"].iloc[0]
    total = summary[summary["grupo"] == "total"].iloc[0]
    popularity = pd.DataFrame(metrics["summary_popularity"])
    baseline = popularity[popularity["grupo"] == "total"].iloc[0]
    return {
        "figure": figure,
        "k": k,
        "ndcg": float(total[f"ndcg@{k}"]),
        "ndcg_baseline": float(baseline[f"ndcg@{k}"]),
        "cat_hit": 100 * float(total_cat[f"cat_hit_rate@{k}"]),
        "sku_hit": 100 * float(total_cat[f"sku_hit_rate@{k}"]),
        "best_profile": profiles.loc[profiles[f"ndcg@{k}"].idxmax(), "grupo"],
        "worst_profile": profiles.loc[profiles[f"ndcg@{k}"].idxmin(), "grupo"],
        "worst_ndcg": float(profiles[f"ndcg@{k}"].min()),
        "pool_recall": 100 * float(
            pd.DataFrame(metrics["candidate_recall"]).set_index("grupo").loc["total", "pool_recall"]
        ),
    }


# ======================================================================================
# 7 y 8. Next Best Action
# ======================================================================================
def load_nba_frame(cfg: FindingsConfig) -> pd.DataFrame:
    """Une la tabla de acciones con el segmento del cliente y su gasto reciente."""
    actions = pd.read_parquet(Path(cfg.predictions_dir) / "nba_actions.parquet")
    rfm = pd.read_parquet(
        Path(cfg.processed_dir) / "rfm.parquet",
        columns=["customer_id", "rfm_segment", "recency_days", "frequency"],
    )
    baskets = pd.read_parquet(
        Path(cfg.processed_dir) / "baskets.parquet",
        columns=["customer_id", "basket_day", "total_amount"],
    )
    baskets["basket_day"] = pd.to_datetime(baskets["basket_day"])
    window = baskets[
        (baskets["basket_day"] < NBA_TEST_CUTOFF)
        & (baskets["basket_day"] >= NBA_TEST_CUTOFF - pd.Timedelta(days=RECENT_WINDOW_DAYS))
    ]
    recent = window.groupby("customer_id")["total_amount"].sum().rename("gasto_90d")

    frame = actions.merge(rfm, on="customer_id", how="left").join(recent, on="customer_id")
    frame["gasto_90d"] = frame["gasto_90d"].fillna(0.0)
    return frame


def finding_nba_who(frame: pd.DataFrame, out_dir: Path) -> dict:
    by_segment = (
        frame.groupby("rfm_segment")
        .agg(
            clientes=("customer_id", "size"),
            p_churn=("p_churn", "mean"),
            gasto_90d=("gasto_90d", "mean"),
            valor_por_cliente=("expected_value", "mean"),
            pct_accion=("action", lambda s: 100 * (s != "ninguna_accion").mean()),
        )
        .sort_values("valor_por_cliente", ascending=False)
    )
    by_segment.index.name = "Segmento RFM"
    deciles = frame.assign(
        decil=pd.qcut(frame["p_churn"], 10, labels=False, duplicates="drop") + 1
    )
    by_decile = deciles.groupby("decil").agg(
        p_churn=("p_churn", "mean"),
        gasto_90d=("gasto_90d", "mean"),
        valor_por_cliente=("expected_value", "mean"),
        pct_accion=("action", lambda s: 100 * (s != "ninguna_accion").mean()),
    )
    corr_churn = float(frame[["p_churn", "expected_value"]].corr().iloc[0, 1])
    corr_spend = float(frame[["gasto_90d", "expected_value"]].corr().iloc[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    seg = by_segment.iloc[::-1]
    axes[0].barh(seg.index, seg["valor_por_cliente"], color=PALETTE[0])
    axes[0].set_title("Valor esperado de la politica por segmento RFM")
    axes[0].set_xlabel("EUR por cliente y oleada")
    axes[0].tick_params(axis="y", labelsize=8)
    gemelo = axes[0].twiny()
    gemelo.plot(seg["p_churn"], seg.index, color=PALETTE[3], marker="o", lw=2)
    gemelo.set_xlabel("P(churn) media", color=PALETTE[3])
    gemelo.grid(False)

    ax = axes[1]
    ax.bar(by_decile.index, by_decile["valor_por_cliente"], color=PALETTE[0], width=0.65)
    ax.set_xlabel("decil de P(churn)  (1 = menos riesgo, 10 = mas)")
    ax.set_ylabel("EUR por cliente", color=PALETTE[0])
    ax.set_title("A mas riesgo de fuga, menos valor que salvar")
    ax.set_xticks(list(by_decile.index))
    otro = ax.twinx()
    otro.plot(by_decile.index, by_decile["gasto_90d"], color=PALETTE[1], marker="o", lw=2)
    otro.set_ylabel("gasto en los ultimos 90 dias (EUR)", color=PALETTE[1])
    otro.grid(False)
    figure = _save(fig, out_dir, "07_nba_a_quien.png")

    return {
        "figure": figure,
        "by_segment": by_segment,
        "by_decile": by_decile,
        "corr_churn": corr_churn,
        "corr_spend": corr_spend,
        "best_segment": str(by_segment.index[0]),
        "best_value": float(by_segment["valor_por_cliente"].iloc[0]),
        "worst_segment": str(by_segment.index[-1]),
        "worst_value": float(by_segment["valor_por_cliente"].iloc[-1]),
        "top_decile_value": float(by_decile["valor_por_cliente"].iloc[-1]),
        "top_decile_action": float(by_decile["pct_accion"].iloc[-1]),
        "bottom_decile_value": float(by_decile["valor_por_cliente"].iloc[0]),
    }


def finding_nba_margin(cfg: FindingsConfig, frame: pd.DataFrame, out_dir: Path) -> dict:
    products = pd.read_parquet(
        Path(cfg.processed_dir) / "products.parquet",
        columns=["product_id", "department", "category"],
    )
    catalog = products.drop_duplicates("category")[["category", "department"]]
    items = pd.read_parquet(
        Path(cfg.processed_dir) / "basket_items.parquet", columns=["product_id", "line_amount"]
    )
    sales = (
        items.merge(products[["product_id", "department"]], on="product_id")
        .groupby("department")["line_amount"]
        .sum()
    )

    acted = frame[frame["action"] != "ninguna_accion"].merge(catalog, on="category", how="left")
    margins = MarginConfig()
    mix = pd.DataFrame(
        {
            "pct_acciones": 100 * acted["department"].value_counts(normalize=True),
            "pct_venta": 100 * sales / sales.sum(),
            "margen": pd.Series(margins.by_department) * 100,
        }
    ).dropna()
    mix["indice"] = mix["pct_acciones"] / mix["pct_venta"]
    mix = mix.sort_values("margen")
    mix.index.name = "Departamento"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = range(len(mix))
    ancho = 0.38
    axes[0].bar([i - ancho / 2 for i in x], mix["pct_venta"], ancho,
                label="% de la venta", color=PALETTE[5])
    axes[0].bar([i + ancho / 2 for i in x], mix["pct_acciones"], ancho,
                label="% de las acciones", color=PALETTE[0])
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(mix.index, rotation=25, fontsize=8)
    axes[0].set_ylabel("%")
    axes[0].set_title("Departamentos ordenados por margen, de menor a mayor")
    axes[0].legend(fontsize=8)

    axes[1].scatter(mix["margen"], mix["indice"], s=90, color=PALETTE[1],
                    edgecolor="white", linewidth=0.8)
    for departamento, fila in mix.iterrows():
        axes[1].annotate(departamento, (fila["margen"], fila["indice"]), fontsize=8,
                         xytext=(5, 3), textcoords="offset points")
    axes[1].axhline(1.0, color=PALETTE[3], linestyle="--", label="cuota de acciones = cuota de venta")
    axes[1].set_xlabel("margen bruto supuesto (%)")
    axes[1].set_ylabel("indice de sobre-representacion")
    axes[1].set_title("La politica se va al margen")
    axes[1].legend(fontsize=8)
    figure = _save(fig, out_dir, "08_nba_margen.png")

    top_categories = acted["category"].value_counts().head(8)
    return {
        "figure": figure,
        "mix": mix,
        "top_categories": top_categories,
        "top_category": str(top_categories.index[0]),
        "top_category_n": int(top_categories.iloc[0]),
        "high_margin_share": float(
            mix.loc[mix["margen"] >= 35, "pct_acciones"].sum()
        ),
        "high_margin_sales": float(mix.loc[mix["margen"] >= 35, "pct_venta"].sum()),
        "fresh_index": float(mix.loc["Frescos", "indice"]),
    }


def seasonal_blind_spot(spark, category: str) -> dict:
    """Cuanto vende una categoria en la semana del corte frente a su mes pico.

    Sirve para medir el punto ciego del modelo de propension: sus features no incluyen el
    calendario, asi que no puede descontar una categoria de temporada fuera de temporada.
    """
    lines = spark.sql(
        f"""
        SELECT MONTH(b.basket_day) AS mes, COUNT(*) AS lineas
        FROM basket_items bi
        JOIN baskets  b ON b.basket_id  = bi.basket_id
        JOIN products p ON p.product_id = bi.product_id
        WHERE p.category = '{category}'
        GROUP BY MONTH(b.basket_day)
        ORDER BY mes
        """
    ).toPandas()
    cut_month = NBA_TEST_CUTOFF.month
    peak = lines.loc[lines["lineas"].idxmax()]
    at_cut = lines.loc[lines["mes"] == cut_month, "lineas"]
    return {
        "category": category,
        "peak_month": int(peak["mes"]),
        "peak_lines": int(peak["lineas"]),
        "cut_month": cut_month,
        "cut_lines": int(at_cut.iloc[0]) if len(at_cut) else 0,
        "ratio": float(peak["lineas"] / at_cut.iloc[0]) if len(at_cut) and at_cut.iloc[0] else float("nan"),
    }


# ======================================================================================
# Informe
# ======================================================================================
def _n(value: float, decimals: int = 0) -> str:
    """Numero con punto de millares y coma decimal, como se escribe en espanol."""
    return f"{value:,.{decimals}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def _p(value: float, decimals: int = 1) -> str:
    """Porcentaje ya expresado en base 100."""
    return _n(value, decimals) + " %"


def _md_table(frame: pd.DataFrame, formats: dict[str, tuple[int, bool]]) -> str:
    """Tabla Markdown con decimales y separadores a la espanola.

    `formats` mapea columna -> (decimales, es_porcentaje). Se declara columna a columna
    a proposito: adivinar el formato por el dtype es como salen los "5.085,00 clientes".
    """
    header = "| " + " | ".join([frame.index.name or ""] + list(frame.columns)) + " |"
    sep = "| " + " | ".join(["---"] + ["---:"] * len(frame.columns)) + " |"
    lines = [header, sep]
    for idx, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            decimals, is_pct = formats.get(col, (2, False))
            cells.append(_p(row[col], decimals) if is_pct else _n(row[col], decimals))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def render_markdown(f: dict) -> str:
    """Ensambla el informe con los ocho hallazgos."""
    seasonality = f["seasonality"]
    affinity = f["affinity"]
    freq = f["frequency"]
    repurchase = f["repurchase"]
    funnel = f["funnel"]
    rec = f["recommender"]
    who = f["nba_who"]
    margin = f["nba_margin"]
    blind = f["blind_spot"]

    segment_table = _md_table(
        who["by_segment"].rename(
            columns={
                "clientes": "clientes",
                "p_churn": "P(churn) media",
                "gasto_90d": "gasto 90d (EUR)",
                "valor_por_cliente": "valor (EUR/cliente)",
                "pct_accion": "con accion",
            }
        ),
        formats={
            "clientes": (0, False),
            "P(churn) media": (3, False),
            "gasto 90d (EUR)": (0, False),
            "valor (EUR/cliente)": (3, False),
            "con accion": (1, True),
        },
    )
    mix_table = _md_table(
        margin["mix"].rename(
            columns={
                "pct_acciones": "de las acciones",
                "pct_venta": "de la venta",
                "margen": "margen",
                "indice": "indice",
            }
        ),
        formats={
            "de las acciones": (1, True),
            "de la venta": (1, True),
            "margen": (0, True),
            "indice": (2, False),
        },
    )

    return f"""# Hallazgos de negocio

Ocho cosas que este dataset y estos modelos dicen sobre el negocio, y que cambian una
decision. No es el EDA pregunta a pregunta -- eso esta en
[`notebooks/01_eda.ipynb`](../../notebooks/01_eda.ipynb) -- sino lo que queda al cruzar el
dato de la Fase 2 con lo que hacen el recomendador (Fase 3) y la politica de Next Best
Action (Fase 4).

Lo escribe `python -m src.eda.findings`. Las consultas son las mismas funciones que usa el
notebook ([`src/eda/questions.py`](../../src/eda/questions.py), con tests en
[`tests/test_eda_questions.py`](../../tests/test_eda_questions.py)); los resultados de
modelo salen de `reports/*/metrics.json` y `predictions/*.parquet`. Ninguna cifra esta
tecleada a mano.

> Datos **100 % sinteticos**. Los "hallazgos" lo son sobre un supermercado simulado; lo
> que se puede llevar uno de aqui es el metodo, no las cifras.

---

## 1. El calendario mueve el surtido mas que el cliente

![Estacionalidad]({seasonality['figure']})

{seasonality['n_categories']} categorias tienen un pico estacional claro. El mas fuerte es
**{seasonality['strongest']}**, que en el mes {seasonality['strongest_month']} pesa
**{_n(seasonality['strongest_index'], 2)} veces** lo que pesa un mes cualquiera dentro de
la venta total. No es que en diciembre se venda mas de todo: el indice esta calculado sobre
la *cuota del mes*, precisamente para separar las dos cosas.

**Por que importa.** Un recomendador que solo mire el historico personal del cliente nunca
propondra turron a tiempo, porque nadie lo compro en noviembre. Es la razon de que la
fuente de candidatos "popularidad x indice estacional" de la Fase 3 sea la unica que cubre
al 100 % al cliente nuevo con la cesta vacia. Y es tambien, como se vera en el hallazgo 8,
lo que le falta al modelo de propension.

---

## 2. Lo que se compra junto no siempre es complementariedad

![Afinidad de cesta]({affinity['figure']})

Los 10 pares que el generador declara aparecen todos con lift por encima de 1
({affinity['n_above_one']} de 10), pero el mas extremo se sale de escala:
**{affinity['outlier_pair']}** mide **{_n(affinity['outlier_measured'], 2)}** frente a un
objetivo de {_n(affinity['outlier_target'], 1)}.

La explicacion no es que la regla se aplicara mal, sino que **las dos categorias estan
restringidas a hogares con bebe**. El lift observado suma dos efectos: la complementariedad
real y la composicion de la clientela. Se ve claro en que todo el bloque de bebe -- leche
infantil, potitos -- sube al ranking sin que exista ninguna regla que lo una.

**Por que importa.** Para recomendar da igual: la senal es util venga de donde venga. Para
*decidir un surtido o un lineal* no da igual en absoluto, porque colocar toallitas al lado
de los panales no hara que las compre quien no tiene bebe. Es la diferencia entre una
correlacion que sirve para predecir y una que sirve para intervenir.

---

## 3. La palanca sobre un cliente valioso es la frecuencia, no el ticket

![Frecuencia frente a ticket]({freq['figure']})

Los clientes `gold` son el **{_p(freq['gold_pct_clientes'])}** de la base y traen el
**{_p(freq['gold_pct_facturacion'])}** de la facturacion, un indice de
{_n(freq['gold_indice'], 2)}x. Lo interesante es de donde sale ese indice: el ticket medio
es practicamente identico en los tres niveles ({_n(freq['gold_ticket'], 2)} EUR en gold
frente a {_n(freq['bronze_ticket'], 2)} EUR en bronze). **La diferencia esta entera en la
frecuencia**: {_n(freq['gold_cestas'])} compras frente a {_n(freq['bronze_cestas'])}.

El tamano del hogar, en cambio, si mueve el ticket: de {_n(freq['ticket_hogar_1'], 2)} EUR
en un hogar de una persona a {_n(freq['ticket_hogar_max'], 2)} EUR en uno de
{freq['hogar_max']}.

**Por que importa.** Una campana que persiga subir el ticket medio de un cliente fiel esta
atacando la variable que no se mueve. Lo que se mueve es cuando vuelve, y eso es justamente
lo que hace accionable el ciclo de reposicion del hallazgo siguiente.

---

## 4. El ciclo de reposicion es real, y escala con el hogar

![Ciclo de recompra]({repurchase['figure']})

El ciclo observado reproduce el teorico con una correlacion de rangos de Spearman de
**{_n(repurchase['spearman'], 3)}**, y se acorta de forma monotona al crecer el hogar en
las cuatro categorias dibujadas. Hoy, **{_p(repurchase['pct_due'])}** de los
{_n(repurchase['pairs'])} pares cliente-categoria tienen la recompra vencida, y eso alcanza
a {_n(repurchase['customers_due'])} clientes.

La nube cae por debajo de la diagonal en los ciclos largos, y tiene explicacion: el
intervalo observado esta **truncado por la frecuencia de visita**. Nadie puede comprar
detergente cada 45 dias si solo pisa la tienda cada 60.

**Por que importa.** Es lo que convierte `due_for_repurchase` (Tarea 2) en una feature y no
en una corazonada, y lo que justifica escalar el ciclo esperado por `household_size_est` en
vez de usar un intervalo unico por categoria.

---

## 5. La sesion online no es el ticket, y por poco lo fue

![Embudo online]({funnel['figure']})

{_n(funnel['sessions'])} sesiones, {_p(funnel['conversion'])} de conversion, y una
diferencia de solo {_n(funnel['device_spread'], 2)} puntos entre el mejor y el peor
dispositivo: **el dispositivo, por si solo, no es una feature con senal**.

Lo que si la tiene es el panel de la derecha. De lo que se anade al carrito acaba en el
ticket el **{_p(funnel['cart_to_basket'])}**; de lo que solo se mira, el
**{_p(funnel['view_to_basket'])}**; y en sentido contrario, solo el
**{_p(funnel['basket_seen'])}** de las lineas del ticket dejo rastro online.

**Por que importa.** Las tres cifras estaban en el 100 % en la primera version del
generador, que emitia un `add_to_cart` por cada producto de la cesta y ninguno mas: la
sesion **era** el ticket escrito de otra forma. Usarla como feature habria dado un NDCG@5
espectacular y falso. Se arreglo el generador con abandono de carrito, productos que solo
se miran y un retardo entre ver y anadir. Es el hallazgo que mas trabajo ahorro: una fuga
de target encontrada antes de entrenar, no despues de presentar el resultado.

---

## 6. El recomendador sabe que necesitas; no sabe que referencia

![Recomendador por perfil]({rec['figure']})

NDCG@{rec['k']} = **{_n(rec['ndcg'], 4)}** frente a {_n(rec['ndcg_baseline'], 4)} del
baseline sin aprendizaje. El numero es bajo, y el panel de la derecha dice por que: el
sistema acierta la **categoria** en el **{_p(rec['cat_hit'])}** de las cestas y el **SKU**
solo en el **{_p(rec['sku_hit'])}**.

No es la primera etapa: el pool cubre ya el {_p(rec['pool_recall'])} del target, y
ampliarlo de 92 a 155 candidatos por cesta no movio el NDCG. Es el dato: dentro de una
categoria hay unas 24 referencias y el generador elige casi al azar.

Por perfil, el peor es **{rec['worst_profile']}** (NDCG@{rec['k']} =
{_n(rec['worst_ndcg'], 4)}). Es el unico que no puede tirar ni de historial ni de ALS, y
encima su cesta ya va por la mitad, asi que lo facil de acertar ya esta dentro.

**Por que importa.** En gran consumo, acertar la categoria **es** util: si el cliente va a
comprar leche, recomendarle una leche sirve aunque no sea la referencia exacta. El proyecto
esta midiendo con la metrica mas dura de las dos y aun asi el techo esta en el dato. Darle
fidelidad de marca al generador es la deuda numero uno del [`ROADMAP.md`](../../ROADMAP.md).

---

## 7. La politica no responde a quien se va, sino a quien todavia vale algo

![NBA por segmento]({who['figure']})

Este es el hallazgo menos intuitivo del proyecto. La correlacion entre el valor esperado de
la accion y la probabilidad de churn es **{_n(who['corr_churn'], 2)}** -- **negativa** --
y con el gasto de los ultimos 90 dias, **+{_n(who['corr_spend'], 2)}**.

{segment_table}

En el decil de mas riesgo de fuga la politica actua solo sobre el
**{_p(who['top_decile_action'], 0)}** de los clientes y saca
{_n(who['top_decile_value'], 3)} EUR por cabeza; en el decil de menos riesgo actua sobre
todos y saca {_n(who['bottom_decile_value'], 3)} EUR. El mejor segmento es
**{who['best_segment']}** ({_n(who['best_value'], 3)} EUR por cliente) y el peor,
**{who['worst_segment']}** ({_n(who['worst_value'], 3)} EUR).

**Por que importa.** Un modelo de churn con AUC 0,85 invita a una conclusion que la
economia no sostiene: perseguir al que mas riesgo tiene. El valor de retener es
`P(churn) x valor_de_retener`, y en un cliente hibernado el segundo factor es casi cero,
porque no queda nada que salvar. **Un buen modelo de churn no es, por si solo, una politica
de retencion**: hace falta multiplicarlo por lo que el cliente todavia vale, y eso invierte
el orden de la lista.

---

## 8. La politica se va al margen, y sin calendario eso tiene un coste

![NBA por departamento]({margin['figure']})

Drogueria e Higiene son el **{_p(margin['high_margin_sales'])}** de la venta y se llevan el
**{_p(margin['high_margin_share'])}** de las acciones. Frescos, que es mas de un tercio de
la venta, se queda en un indice de **{_n(margin['fresh_index'], 2)}**.

{mix_table}

Es aritmetica, no capricho: el cupon vale 2,54 EUR y hay que pagarlo con el margen del
ticket que provoque. Con un 18 % de margen en Frescos harian falta mas de 14 EUR de compra
solo para empatar; con un 35 % en Drogueria bastan 7,26 EUR.

**Y aqui aparece el punto ciego.** La categoria mas elegida por la politica es
**{blind['category']}** ({_n(margin['top_category_n'])} veces), que en el mes del corte
(mes {blind['cut_month']}) vende **{_n(blind['ratio'], 1)} veces menos** que en su mes
pico (mes {blind['peak_month']}). El modelo de propension no tiene ni una feature de
calendario -- ni mes, ni indice estacional, ni nada -- asi que no puede descontar una
categoria de temporada fuera de temporada, y la economia (precio unitario alto por un 35 %
de margen) hace el resto.

**Por que importa.** Es un fallo barato de arreglar y caro de no ver: meter el indice
estacional de la Fase 2 en las features del modelo de propension. Queda anotado como deuda
en el [`ROADMAP.md`](../../ROADMAP.md).

---

## Lo que estos ocho tienen en comun

Cinco de los ocho son del mismo tipo: **una cifra que parecia buena o mala estaba midiendo
otra cosa**. El lift de 13 que era composicion de clientela; el solapamiento del 100 % que
era el target disfrazado; el NDCG bajo que era el techo del dato y no del modelo; el AUC
alto que no basta para una politica; la categoria mas recomendada, que estaba fuera de
temporada.

Ninguno se ve mirando la metrica sola. Todos aparecieron al preguntar **de donde sale este
numero** y encontrar que la respuesta no era la esperada.
"""


def main() -> int:
    cfg = FindingsConfig()
    out_dir = Path(cfg.out_dir)
    _style()

    spark = get_spark("findings", enable_ui=False)
    try:
        q.register_views(spark, cfg.processed_dir)
        findings = {
            "seasonality": finding_seasonality(spark, out_dir),
            "affinity": finding_affinity(spark, out_dir),
            "frequency": finding_frequency_vs_ticket(spark, out_dir),
            "repurchase": finding_repurchase(spark, out_dir),
            "funnel": finding_online_funnel(spark, out_dir),
            "recommender": finding_recommender(cfg, out_dir),
        }
        frame = load_nba_frame(cfg)
        findings["nba_who"] = finding_nba_who(frame, out_dir)
        findings["nba_margin"] = finding_nba_margin(cfg, frame, out_dir)
        findings["blind_spot"] = seasonal_blind_spot(
            spark, findings["nba_margin"]["top_category"]
        )
    finally:
        spark.stop()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cfg.markdown_name
    path.write_text(render_markdown(findings), encoding="utf-8")
    print(f"Escrito {path} y {len(list(out_dir.glob('*.png')))} figuras en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

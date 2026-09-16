"""Evaluacion del recomendador: NDCG@5, Recall@5, Precision@5 y F1@5 sobre las cestas ocultas.

## Las metricas

Para una query con `T` productos que hay que adivinar y una lista de `k` recomendaciones:

    DCG@k   = suma_{i=1..k} acierto_i / log2(i + 1)
    IDCG@k  = suma_{i=1..min(k, T)} 1 / log2(i + 1)
    NDCG@k  = DCG@k / IDCG@k
    Recall@k = aciertos en el top-k / T
    Precision@k = aciertos en el top-k / k
    F1@k = 2 x Precision@k x Recall@k / (Precision@k + Recall@k)

**NDCG@5** premia acertar arriba: el mismo acierto vale el doble en el primer hueco que en
el quinto. **Recall@5** mide cobertura y no mira el orden. **Precision@5** mide que parte
de los cinco huecos se aprovecha; su denominador es siempre `k`, porque el sistema siempre
ensena cinco productos aunque el pool tuviera menos.

El F1@5 se reporta de dos formas, porque no dan lo mismo:

- `f1@5`: la media armonica de la Precision@5 media y el Recall@5 medio. Es la definicion
  del `ROADMAP.md` (Fase 7c) y la cifra de cabecera.
- `f1@5_por_cesta`: el F1 de cada cesta, `2 x aciertos / (k + T)`, promediado. Es como
  puntuaba la competicion de Kaggle "Instacart Market Basket Analysis" (mean F1 por
  pedido), asi que es la variante que se usa al lado de ese benchmark.

Dos decisiones que cambian el numero y conviene tener escritas:

- El denominador de Recall@5 es el numero **real** de productos que faltaban en la cesta,
  no los que llegaron al pool de candidatos. Si la primera etapa no propuso un producto,
  eso cuenta como fallo: es el techo real del sistema, no un detalle de implementacion.
- Con `T > 5` el Recall@5 no puede llegar a 1 (cinco huecos no cubren siete productos).
  Por eso se reporta tambien `hit_rate@5` -- la fraccion de queries con al menos un acierto
  --, que no tiene ese techo y se lee mejor en negocio: "en cuantas cestas acertamos algo".

Todo se desglosa por los cuatro perfiles de `CHALLENGE.md`, que es donde se ve si el
sistema aguanta el cold-start o solo funciona con clientes conocidos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommender.splits import PROFILE_LABELS

# Descuento posicional 1 / log2(i+1), precalculado para los primeros puestos.
_DISCOUNT = 1.0 / np.log2(np.arange(2, 64))


def _idcg(n_target: np.ndarray, k: int) -> np.ndarray:
    """DCG del orden perfecto: aciertos en los `min(k, T)` primeros huecos."""
    ideal = np.cumsum(_DISCOUNT[:k])
    return ideal[np.clip(n_target, 1, k) - 1]


def top_k_predictions(scored: pd.DataFrame, *, k: int, score_col: str = "score") -> pd.DataFrame:
    """Se queda con los `k` mejores candidatos de cada query, con su posicion.

    El desempate por `product_id` no es cosmetico: sin el, dos ejecuciones con el mismo
    modelo podrian devolver listas distintas cuando hay scores empatados.
    """
    ordered = scored.sort_values(
        ["basket_id", score_col, "product_id"], ascending=[True, False, True], kind="stable"
    )
    ordered = ordered.assign(rank=ordered.groupby("basket_id").cumcount() + 1)
    return ordered.loc[ordered["rank"] <= k].reset_index(drop=True)


def per_query_metrics(
    top_k: pd.DataFrame, queries: pd.DataFrame, *, k: int
) -> pd.DataFrame:
    """NDCG@k, Recall@k, Precision@k, F1@k y acierto/no acierto de cada query.

    Args:
        top_k: Salida de `top_k_predictions`, con `label` y `rank`.
        queries: Una fila por query con `basket_id`, `profile` y `n_target`.
        k: Longitud de la lista.

    Returns:
        Una fila por query de `queries`. Las que no recibieron ni un candidato aparecen
        con metricas 0, que es lo que les corresponde.
    """
    hits = top_k.loc[top_k["label"] == 1]
    dcg = (
        hits.assign(gain=_DISCOUNT[hits["rank"].to_numpy() - 1])
        .groupby("basket_id")["gain"]
        .sum()
    )
    n_hits = hits.groupby("basket_id").size()

    out = queries[["basket_id", "profile", "n_target"]].copy()
    out["dcg"] = out["basket_id"].map(dcg).fillna(0.0)
    out["n_hits"] = out["basket_id"].map(n_hits).fillna(0).astype(int)
    out["ndcg"] = out["dcg"] / _idcg(out["n_target"].to_numpy(), k)
    out["recall"] = out["n_hits"] / out["n_target"]
    out["precision"] = out["n_hits"] / k
    # Con P = h/k y R = h/T, la media armonica se simplifica a 2h / (k + T), que ademas
    # vale 0 sin division por cero cuando no hay aciertos.
    out["f1"] = 2 * out["n_hits"] / (k + out["n_target"])
    out["hit"] = (out["n_hits"] > 0).astype(float)
    return out


def harmonic_f1(precision: float, recall: float) -> float:
    """Media armonica de dos medias ya agregadas; 0 si las dos son 0."""
    total = precision + recall
    return 0.0 if total == 0 else 2 * precision * recall / total


def summarise(per_query: pd.DataFrame, *, k: int, label: str = "total") -> pd.DataFrame:
    """Agrega las metricas por perfil y en total, con el numero de queries de cada uno."""

    def block(frame: pd.DataFrame, name: str) -> dict:
        precision = float(frame["precision"].mean())
        recall = float(frame["recall"].mean())
        return {
            "grupo": name,
            "n_queries": int(len(frame)),
            f"ndcg@{k}": float(frame["ndcg"].mean()),
            f"recall@{k}": recall,
            f"precision@{k}": precision,
            f"f1@{k}": harmonic_f1(precision, recall),
            f"f1@{k}_por_cesta": float(frame["f1"].mean()),
            f"hit_rate@{k}": float(frame["hit"].mean()),
            "n_target_medio": float(frame["n_target"].mean()),
        }

    rows = [block(per_query, label)]
    for profile in sorted(per_query["profile"].unique()):
        subset = per_query.loc[per_query["profile"] == profile]
        rows.append(block(subset, PROFILE_LABELS.get(int(profile), str(profile))))
    return pd.DataFrame(rows)


def candidate_recall(scored: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Techo de la primera etapa: que parte del target llego siquiera al pool.

    Separa los dos fallos posibles del sistema. Si el pool ya no contiene el producto, el
    ranker no puede hacer nada; si lo contiene y no lo sube al top-5, el fallo es del
    ranker. Sin esta cifra no se sabe cual de las dos etapas hay que tocar.
    """
    in_pool = scored.loc[scored["label"] == 1].groupby("basket_id").size()
    out = queries[["basket_id", "profile", "n_target"]].copy()
    out["n_in_pool"] = out["basket_id"].map(in_pool).fillna(0).astype(int)
    out["pool_recall"] = out["n_in_pool"] / out["n_target"]
    out["pool_size"] = out["basket_id"].map(scored.groupby("basket_id").size()).fillna(0).astype(int)

    rows = [
        {
            "grupo": "total",
            "n_queries": int(len(out)),
            "pool_recall": float(out["pool_recall"].mean()),
            "pool_size_medio": float(out["pool_size"].mean()),
        }
    ]
    for profile in sorted(out["profile"].unique()):
        subset = out.loc[out["profile"] == profile]
        rows.append(
            {
                "grupo": PROFILE_LABELS.get(int(profile), str(profile)),
                "n_queries": int(len(subset)),
                "pool_recall": float(subset["pool_recall"].mean()),
                "pool_size_medio": float(subset["pool_size"].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    scored: pd.DataFrame, queries: pd.DataFrame, *, k: int, score_col: str = "score"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atajo: top-k, metricas por query y resumen por perfil.

    Returns:
        `(resumen, por_query)`.
    """
    top = top_k_predictions(scored, k=k, score_col=score_col)
    per_query = per_query_metrics(top, queries, k=k)
    return summarise(per_query, k=k), per_query


def category_metrics(
    top_k: pd.DataFrame,
    target: pd.DataFrame,
    product_category: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    k: int,
) -> pd.DataFrame:
    """Las mismas listas, evaluadas a nivel de **categoria** en vez de de SKU.

    Separa dos fallos que la metrica de SKU confunde en un solo numero:

    - *"no se que necesita este cliente"* -- ni siquiera acierta la categoria;
    - *"se que necesita leche, pero no cual de las referencias de leche"*.

    En gran consumo el segundo es un problema distinto y mucho mas benigno: si el cliente
    va a comprar leche, cualquier leche recomendada es una recomendacion util aunque no
    sea la referencia exacta que acabo eligiendo. Sin este desglose no se sabe cual de los
    dos limita al sistema.

    Args:
        top_k: Salida de `top_k_predictions`.
        target: Pares `(basket_id, product_id)` que habia que adivinar.
        product_category: Catalogo con `product_id` y `category`.
        queries: Una fila por query, con `basket_id` y `profile`.
        k: Longitud de la lista.

    Returns:
        Resumen por perfil con `hit_rate` y `precision` de categoria y de SKU.
    """
    catalog = product_category[["product_id", "category"]]
    target_cats = (
        target.merge(catalog, on="product_id").groupby("basket_id")["category"].apply(set)
    )

    recs = top_k.merge(catalog, on="product_id")
    empty: set[str] = set()
    recs["cat_ok"] = [
        cat in target_cats.get(basket, empty)
        for basket, cat in zip(recs["basket_id"], recs["category"])
    ]

    per_query = recs.groupby("basket_id").agg(
        cat_hits=("cat_ok", "sum"), sku_hits=("label", "sum")
    )
    out = queries[["basket_id", "profile"]].copy()
    out["cat_hits"] = out["basket_id"].map(per_query["cat_hits"]).fillna(0)
    out["sku_hits"] = out["basket_id"].map(per_query["sku_hits"]).fillna(0)

    def block(frame: pd.DataFrame, name: str) -> dict:
        return {
            "grupo": name,
            "n_queries": int(len(frame)),
            f"cat_hit_rate@{k}": float((frame["cat_hits"] > 0).mean()),
            f"cat_precision@{k}": float((frame["cat_hits"] / k).mean()),
            f"sku_hit_rate@{k}": float((frame["sku_hits"] > 0).mean()),
            f"sku_precision@{k}": float((frame["sku_hits"] / k).mean()),
        }

    rows = [block(out, "total")]
    for profile in sorted(out["profile"].unique()):
        subset = out.loc[out["profile"] == profile]
        rows.append(block(subset, PROFILE_LABELS.get(int(profile), str(profile))))
    return pd.DataFrame(rows)


def popularity_baseline(scored: pd.DataFrame) -> pd.Series:
    """Baseline sin aprendizaje: ordenar por popularidad reciente x estacionalidad.

    Es el suelo contra el que hay que medir el ranker. Un LambdaRank que no lo supere con
    holgura no esta aportando nada que no estuviera ya en un `ORDER BY ventas DESC`.
    """
    return (
        scored["prod_pop_recent"].fillna(0.0) * scored["prod_seasonal_index"].fillna(0.0)
    ).astype(float)

"""Re-ranking final del top-k: diversidad por categoria y exclusion del carrito (punto A2).

El LambdaRank puntua cada candidato por separado, asi que nada le impide poner dos leches
en el top-5, o una leche cuando ya hay leche en el carrito. Con una linea por categoria en
cada cesta (regla del generador), esos huecos casi nunca aciertan. Este modulo aplica,
despues del score y antes de cortar, las reglas de `config.RerankConfig`:

- **Cuota por categoria**: solo las `max_per_category` primeras referencias de cada
  categoria (en orden de score) son admisibles.
- **Exclusion del carrito**: los candidatos con `cat_in_cart = 1` no son admisibles.

Los no admisibles no se borran: se relegan detras de todos los admisibles, conservando su
orden. Es un MMR con penalizacion infinita, determinista y vectorizado, y garantiza que la
lista tiene `k` productos mientras el pool los tenga.

Vive aparte, sin PySpark, porque lo usan la evaluacion (`evaluate.top_k_predictions`) y
la demo (`serving.rank_queries`): tiene que ser literalmente la misma funcion para que el
test de paridad siga valiendo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommender.config import RerankConfig

# Columnas que necesita el re-ranking, ademas del score. Las dos son features del ranker,
# asi que estan en cualquier matriz puntuada.
CATEGORY_COLUMN = "category_idx"
CART_COLUMN = "cat_in_cart"


def order_candidates(
    scored: pd.DataFrame,
    *,
    score_col: str = "score",
    rerank: RerankConfig | None = None,
) -> pd.DataFrame:
    """Ordena los candidatos de cada query y anade `rank` (1..n).

    Sin reglas, el orden es score descendente con desempate por `product_id`. Con reglas,
    primero van los admisibles y despues los relegados, cada bloque en ese mismo orden.
    """
    keys, ascending = ["basket_id", score_col, "product_id"], [True, False, True]
    ordered = scored.sort_values(keys, ascending=ascending, kind="stable")

    if rerank is not None and rerank.active:
        demoted = np.zeros(len(ordered), dtype=bool)
        if rerank.exclude_cart_categories:
            demoted |= ordered[CART_COLUMN].to_numpy() > 0
        if rerank.max_per_category is not None:
            seen = ordered.groupby(["basket_id", CATEGORY_COLUMN], sort=False).cumcount()
            demoted |= seen.to_numpy() >= rerank.max_per_category
        ordered = ordered.assign(_demoted=demoted).sort_values(
            ["basket_id", "_demoted", score_col, "product_id"],
            ascending=[True, True, False, True],
            kind="stable",
        ).drop(columns="_demoted")

    return ordered.assign(rank=ordered.groupby("basket_id").cumcount() + 1)


def top_k(
    scored: pd.DataFrame,
    *,
    k: int,
    score_col: str = "score",
    rerank: RerankConfig | None = None,
) -> pd.DataFrame:
    """Los `k` primeros de cada query segun `order_candidates`."""
    ordered = order_candidates(scored, score_col=score_col, rerank=rerank)
    return ordered.loc[ordered["rank"] <= k].reset_index(drop=True)

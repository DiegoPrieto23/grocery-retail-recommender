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

Al final del modulo esta la bateria de **baselines independientes del pool** (punto A3 de
`docs/diagnostico-fase7.md`), que la lanza `verify_recommender_diagnostics.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.recommender import rerank as rr
from src.recommender.config import RerankConfig
from src.recommender.splits import PROFILE_LABELS

# Descuento posicional 1 / log2(i+1), precalculado para los primeros puestos.
_DISCOUNT = 1.0 / np.log2(np.arange(2, 64))


def _idcg(n_target: np.ndarray, k: int) -> np.ndarray:
    """DCG del orden perfecto: aciertos en los `min(k, T)` primeros huecos."""
    ideal = np.cumsum(_DISCOUNT[:k])
    return ideal[np.clip(n_target, 1, k) - 1]


def top_k_predictions(
    scored: pd.DataFrame,
    *,
    k: int,
    score_col: str = "score",
    rerank: RerankConfig | None = None,
) -> pd.DataFrame:
    """Se queda con los `k` mejores candidatos de cada query, con su posicion.

    El desempate por `product_id` no es cosmetico: sin el, dos ejecuciones con el mismo
    modelo podrian devolver listas distintas cuando hay scores empatados.

    Con `rerank`, antes de cortar se aplican las reglas de diversidad y de carrito
    (`rerank.py`); `scored` necesita entonces `category_idx` y `cat_in_cart`. Es la misma
    funcion que usa la demo (`serving.rank_queries`).
    """
    return rr.top_k(scored, k=k, score_col=score_col, rerank=rerank)


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
    scored: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    k: int,
    score_col: str = "score",
    rerank: RerankConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atajo: top-k, metricas por query y resumen por perfil.

    Returns:
        `(resumen, por_query)`.
    """
    top = top_k_predictions(scored, k=k, score_col=score_col, rerank=rerank)
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


WITH_CART_GROUP = "con carrito (perfiles 2 y 4)"


def wasted_slot_metrics(
    top_k: pd.DataFrame,
    prefix: pd.DataFrame,
    product_category: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    k: int,
) -> pd.DataFrame:
    """Huecos "regalados" del top-k: los que por construccion casi no pueden acertar.

    Con una linea por categoria en cada cesta (regla del generador), un hueco se regala si
    su categoria:

    - ya esta en el prefijo del ticket (`en_carrito`): nunca puede acertar;
    - ya aparecio mas arriba en la misma lista (`repetida`): de las dos, como mucho una
      acierta. Se cuenta a partir de la segunda aparicion, y solo si no es ya `en_carrito`,
      para que las dos columnas sumen `regalados`.

    Se mide contra el **prefijo** (lo que acabo en el ticket), no contra el carrito de
    sesion completo: es donde el fallo esta garantizado. Punto A2 del diagnostico.

    Returns:
        Por grupo (total, perfiles y "con carrito"): porcentaje de huecos en cada caso y
        porcentaje de listas con al menos un hueco regalado.
    """
    catalog = product_category[["product_id", "category"]]
    recs = (
        top_k[["basket_id", "product_id", "rank"]]
        .merge(catalog, on="product_id")
        .sort_values(["basket_id", "rank"], kind="stable")
    )
    cart_cats = (
        prefix[["basket_id", "product_id"]]
        .merge(catalog, on="product_id")[["basket_id", "category"]]
        .drop_duplicates()
        .assign(_in_cart=1)
    )
    recs = recs.merge(cart_cats, on=["basket_id", "category"], how="left")
    in_cart = recs["_in_cart"].notna().to_numpy()
    repeated = recs.duplicated(["basket_id", "category"]).to_numpy() & ~in_cart
    recs = recs.assign(in_cart=in_cart, repeated=repeated, wasted=in_cart | repeated)

    per_query = recs.groupby("basket_id").agg(
        slots=("rank", "size"),
        in_cart=("in_cart", "sum"),
        repeated=("repeated", "sum"),
        wasted=("wasted", "sum"),
    )
    out = queries[["basket_id", "profile"]].merge(
        per_query, left_on="basket_id", right_index=True, how="left"
    ).fillna({"slots": 0, "in_cart": 0, "repeated": 0, "wasted": 0})

    def block(frame: pd.DataFrame, name: str) -> dict:
        slots = max(float(frame["slots"].sum()), 1.0)
        return {
            "grupo": name,
            f"huecos_en_carrito@{k}": float(frame["in_cart"].sum()) / slots,
            f"huecos_repetidos@{k}": float(frame["repeated"].sum()) / slots,
            f"huecos_regalados@{k}": float(frame["wasted"].sum()) / slots,
            f"listas_con_regalo@{k}": float((frame["wasted"] > 0).mean()),
        }

    rows = [block(out, "total")]
    for profile in sorted(out["profile"].unique()):
        subset = out.loc[out["profile"] == profile]
        rows.append(block(subset, PROFILE_LABELS.get(int(profile), str(profile))))
    rows.append(block(out.loc[out["profile"].isin([2, 4])], WITH_CART_GROUP))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Huecos segun la recencia de su categoria (punto A1 del diagnostico)
# --------------------------------------------------------------------------------------
# Umbrales de "categoria recien repuesta", en dias desde la ultima compra del cliente. El
# generador castiga con fuerza volver a comprar justo despues de reponer
# (`ratio^1.8`, minimo 0,03), asi que esos huecos casi nunca aciertan.
RECENT_DAYS: tuple[int, ...] = (7, 14)


def recency_slots(
    top_k: pd.DataFrame,
    queries: pd.DataFrame,
    target: pd.DataFrame,
    category_last_day: pd.DataFrame,
    product_category: pd.DataFrame,
) -> pd.DataFrame:
    """Un hueco por fila, con si acierta (SKU y categoria) y cuanto hace que se compro.

    `category_last_day` (`basket_id`, `category`, `last_day`) es la ultima compra real del
    cliente de la query en cada categoria antes del dia de la cesta: lo que el cliente
    sabe, lo viera o no el modelo. `days_since` queda nulo si nunca la habia comprado.
    """
    catalog = product_category[["product_id", "category"]]
    target_cats = (
        target.merge(catalog, on="product_id")[["basket_id", "category"]]
        .drop_duplicates()
        .assign(cat_hit=1.0)
    )
    slots = (
        top_k[["basket_id", "product_id", "label"]]
        .merge(catalog, on="product_id")
        .merge(queries[["basket_id", "basket_day"]], on="basket_id")
        .merge(target_cats, on=["basket_id", "category"], how="left")
        .merge(
            category_last_day[["basket_id", "category", "last_day"]],
            on=["basket_id", "category"],
            how="left",
        )
    )
    slots["cat_hit"] = slots["cat_hit"].fillna(0.0)
    slots["days_since"] = (
        pd.to_datetime(slots["basket_day"]) - pd.to_datetime(slots["last_day"])
    ).dt.days
    return slots


def _recency_groups(slots: pd.DataFrame, window_start) -> dict[str, np.ndarray]:
    days = slots["days_since"]
    in_window = (pd.to_datetime(slots["last_day"]) >= pd.Timestamp(window_start)).to_numpy()
    short, long = RECENT_DAYS
    return {
        f"<= {short} dias": (days <= short).to_numpy(),
        f"<= {long} dias": (days <= long).to_numpy(),
        f"{short + 1}-{long} dias": ((days > short) & (days <= long)).to_numpy(),
        "comprada en la ventana, antes de la cesta": in_window,
        "sin compra en la ventana": ~in_window,
        "total": np.ones(len(slots), dtype=bool),
    }


def recency_slot_metrics(slots: pd.DataFrame, *, window_start) -> pd.DataFrame:
    """Que parte del top-k cae en categorias recien compradas, y cuanto acierta ahi.

    Args:
        slots: Salida de `recency_slots`.
        window_start: Inicio de la ventana de test (para "comprada en la ventana").

    Returns:
        Una fila por grupo de recencia con la proporcion de huecos y la precision de SKU
        y de categoria de esos huecos.
    """
    total = max(len(slots), 1)
    rows = []
    for name, mask in _recency_groups(slots, window_start).items():
        sub = slots.loc[mask]
        rows.append(
            {
                "grupo": name,
                "huecos": int(len(sub)),
                "proporcion_huecos": len(sub) / total,
                "sku_precision": float(sub["label"].mean()) if len(sub) else float("nan"),
                "cat_precision": float(sub["cat_hit"].mean()) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def recency_query_comparison(
    systems: dict[str, pd.DataFrame], reference: str, *, k: int
) -> pd.DataFrame:
    """Sistemas comparados en las queries donde `reference` gastaba huecos recientes.

    Para cada umbral de `RECENT_DAYS` se toman las queries en las que la lista de
    `reference` tenia al menos un hueco en una categoria comprada hace `<= d` dias, y en
    ellas se mide cada sistema: que parte de sus huecos sigue cayendo en esas categorias
    y cuanto acierta la lista entera.

    Args:
        systems: Nombre -> salida de `recency_slots`, sobre las mismas queries.
        reference: El sistema que define el subconjunto (el modelo de antes).
        k: Longitud de la lista.
    """
    ref = systems[reference]
    rows = []
    for d in RECENT_DAYS:
        chosen = pd.Index(ref.loc[ref["days_since"] <= d, "basket_id"].unique())
        for name, slots in systems.items():
            sub = slots.loc[slots["basket_id"].isin(chosen)]
            per_query = (
                sub.groupby("basket_id")
                .agg(sku=("label", "sum"), cat=("cat_hit", "sum"))
                .reindex(chosen, fill_value=0.0)
            )
            rows.append(
                {
                    "umbral_dias": d,
                    "sistema": name,
                    "n_queries": int(len(chosen)),
                    f"huecos_recientes@{k}": float((sub["days_since"] <= d).sum())
                    / max(len(chosen) * k, 1),
                    f"sku_precision@{k}": float((per_query["sku"] / k).mean()),
                    f"cat_precision@{k}": float((per_query["cat"] / k).mean()),
                    f"sku_hit_rate@{k}": float((per_query["sku"] > 0).mean()),
                    f"cat_hit_rate@{k}": float((per_query["cat"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def popularity_baseline(scored: pd.DataFrame) -> pd.Series:
    """Baseline sin aprendizaje: ordenar por popularidad reciente x estacionalidad.

    Es el suelo contra el que hay que medir el ranker. Un LambdaRank que no lo supere con
    holgura no esta aportando nada que no estuviera ya en un `ORDER BY ventas DESC`.
    """
    return (
        scored["prod_pop_recent"].fillna(0.0) * scored["prod_seasonal_index"].fillna(0.0)
    ).astype(float)


# --------------------------------------------------------------------------------------
# Baselines independientes del pool (punto A3 del diagnostico)
# --------------------------------------------------------------------------------------
# `popularity_baseline` reordena el mismo pool que el ranker, asi que solo aisla lo que
# aporta el ranker frente a la primera etapa. Los baselines de abajo construyen su top-k
# desde cero, con el mismo historial que ve el sistema y sobre las mismas queries: la
# popularidad y las reglas, congeladas en `test_start` como las fuentes del ranker; el
# historial personal, as-of el dia de cada cesta como las features (punto A1). Contestan otra pregunta: si un sistema trivial, sin ningun
# aprendizaje, recomienda igual o mejor.
#
# Reglas comunes, para que las cifras sean comparables entre si:
#
# - Ninguno recomienda una categoria que ya esta en el carrito. Por construccion del
#   generador (una linea por categoria) esos huecos no pueden acertar. El LambdaRank lo
#   hace con `RerankConfig.exclude_cart_categories` (punto A2).
# - Los baselines de categoria eligen 5 categorias distintas y una referencia en cada
#   una; los de producto pueden repetir categoria.
# - Cuando el criterio principal no da para cinco (cliente nuevo, pocas reglas...), se
#   rellena con la popularidad de categoria (o de SKU en los de producto). Los empates se
#   deshacen por esa popularidad y despues por identificador, asi que el resultado es
#   determinista.

BASELINE_LABELS: dict[str, str] = {
    "random": "Aleatorio",
    "global_popularity": "Popularidad global (SKU)",
    "category_popularity": "Popularidad de categoria + referencia lider",
    "personal_frequency": "Frecuencia personal por categoria + referencia favorita",
    "personal_due": "Frecuencia personal x due_for_repurchase + referencia favorita",
    "repeat_favorite": "Repetir las referencias favoritas del cliente",
    "association_rules": "Reglas de asociacion de categoria + referencia lider",
}

_NS_PER_DAY = 86_400_000_000_000


@dataclass(frozen=True)
class BaselineInputs:
    """Lo que necesitan los baselines, ya en pandas y recortado al pasado de la ventana.

    Attributes:
        queries: `basket_id`, `customer_id` (nulo si es anonimo) y `basket_day`.
        prefix: `(basket_id, product_id)` de lo que ya lleva el carrito.
        target: `(basket_id, product_id)` de lo que habia que adivinar.
        products: Catalogo con `product_id` y `category`.
        product_popularity: `product_id`, `n_baskets` en la ventana reciente.
        category_popularity: `category`, `n_baskets` en la ventana reciente.
        customer_products: `basket_id`, `product_id`, `n_baskets`: el historial del
            cliente de cada query al dia de su cesta (`history.asof_history`).
        customer_categories: `basket_id`, `category`, `n_purchase_days`,
            `last_purchase_date` y `expected_repurchase_days`, tambien as-of.
        category_rules: `antecedent`, `consequent`, `confidence` y `lift` entre
            categorias (de `src.etl.affinity.cooccurrence_affinity`).
    """

    queries: pd.DataFrame
    prefix: pd.DataFrame
    target: pd.DataFrame
    products: pd.DataFrame
    product_popularity: pd.DataFrame
    category_popularity: pd.DataFrame
    customer_products: pd.DataFrame
    customer_categories: pd.DataFrame
    category_rules: pd.DataFrame


class _Grid:
    """Indices densos query x categoria y query x producto que comparten los baselines."""

    def __init__(self, data: BaselineInputs) -> None:
        self.data = data
        self.basket_ids = data.queries["basket_id"].to_numpy()
        self.row = pd.Series(np.arange(len(self.basket_ids)), index=self.basket_ids)

        catalog = data.products[["product_id", "category"]].sort_values("product_id")
        self.product_ids = catalog["product_id"].to_numpy()
        self.categories = np.array(sorted(catalog["category"].unique()), dtype=object)
        self.prod_pos = pd.Series(np.arange(len(self.product_ids)), index=self.product_ids)
        self.cat_pos = pd.Series(np.arange(len(self.categories)), index=self.categories)
        self.product_cat = self.cat_pos.loc[catalog["category"]].to_numpy()

        self.prefix_cats = np.zeros((len(self.basket_ids), len(self.categories)), dtype=bool)
        pref = data.prefix.merge(catalog, on="product_id")
        self.prefix_cats[
            self.row.loc[pref["basket_id"]].to_numpy(),
            self.cat_pos.loc[pref["category"]].to_numpy(),
        ] = True

        self.product_pop = (
            data.product_popularity.set_index("product_id")["n_baskets"]
            .reindex(self.product_ids)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        self.category_pop = (
            data.category_popularity.set_index("category")["n_baskets"]
            .reindex(self.categories)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        # Referencia lider de cada categoria: la mas vendida en la ventana reciente.
        leader = (
            catalog.assign(pop=self.product_pop)
            .sort_values(["category", "pop", "product_id"], ascending=[True, False, True])
            .drop_duplicates("category")
            .set_index("category")["product_id"]
        )
        self.category_leader = leader.reindex(self.categories).to_numpy()

    def query_matrix(
        self, frame: pd.DataFrame, key: str, positions: pd.Series, value: str, fill: float = 0.0
    ) -> np.ndarray:
        """Pasa una tabla `(basket_id, key, value)` a matriz query x `key`."""
        out = np.full((len(self.basket_ids), len(positions)), fill, dtype=float)
        joined = frame[["basket_id", key, value]]
        joined = joined.loc[
            joined["basket_id"].isin(self.row.index) & joined[key].isin(positions.index)
        ]
        out[
            self.row.loc[joined["basket_id"]].to_numpy(),
            positions.loc[joined[key]].to_numpy(),
        ] = joined[value].to_numpy(dtype=float)
        return out

    def favorite_reference(self) -> np.ndarray:
        """Referencia favorita del cliente en cada categoria; la lider si no tiene."""
        cp = self.data.customer_products.merge(
            self.data.products[["product_id", "category"]], on="product_id"
        )
        cp["prod_pos"] = self.prod_pos.loc[cp["product_id"]].to_numpy()
        cp["pop"] = self.product_pop[cp["prod_pos"].to_numpy()]
        fav = cp.sort_values(
            ["basket_id", "category", "n_baskets", "pop", "product_id"],
            ascending=[True, True, False, False, True],
        ).drop_duplicates(["basket_id", "category"])
        pos = self.query_matrix(fav, "category", self.cat_pos, "prod_pos", fill=-1.0)
        pos = pos.astype(int)
        leader_pos = self.prod_pos.loc[self.category_leader].to_numpy()
        return self.product_ids[np.where(pos >= 0, pos, leader_pos[None, :])]

    def to_top_k(self, item_ids: np.ndarray, k: int) -> pd.DataFrame:
        """Matriz (Q, k) de `product_id` (None = hueco vacio) a un top-k con `label`."""
        frame = pd.DataFrame(
            {
                "basket_id": np.repeat(self.basket_ids, k),
                "product_id": item_ids.reshape(-1),
                "rank": np.tile(np.arange(1, k + 1), len(self.basket_ids)),
            }
        ).dropna(subset=["product_id"])
        truth = self.data.target[["basket_id", "product_id"]].drop_duplicates().assign(label=1)
        frame = frame.merge(truth, on=["basket_id", "product_id"], how="left")
        frame["label"] = frame["label"].fillna(0).astype(int)
        return frame.sort_values(["basket_id", "rank"]).reset_index(drop=True)


def _rank_rows(
    primary: np.ndarray, secondary: np.ndarray, allowed: np.ndarray, k: int
) -> np.ndarray:
    """Las `k` mejores columnas por fila: `primary` desc, `secondary` desc, columna asc.

    Las columnas no permitidas nunca entran; si una fila no tiene `k`, el hueco es -1.
    """
    shape = allowed.shape
    keys = (
        np.broadcast_to(np.arange(shape[1]), shape),
        -np.broadcast_to(secondary, shape),
        -np.broadcast_to(primary, shape),
        ~allowed,
    )
    # `np.lexsort` ordena por la ultima clave primero.
    order = np.lexsort(keys, axis=1)[:, :k]
    return np.where(np.take_along_axis(allowed, order, axis=1), order, -1)


def _category_lists(
    grid: _Grid, primary: np.ndarray, reference: np.ndarray, k: int
) -> pd.DataFrame:
    """Top-k de categorias distintas fuera del carrito, concretadas en una referencia."""
    lists = _rank_rows(primary, grid.category_pop, ~grid.prefix_cats, k)
    ref = np.broadcast_to(reference, grid.prefix_cats.shape)
    picked = np.take_along_axis(ref, np.maximum(lists, 0), axis=1).astype(object)
    picked[lists < 0] = None
    return grid.to_top_k(picked, k)


def _product_lists(
    grid: _Grid, primary: np.ndarray, secondary: np.ndarray, k: int
) -> pd.DataFrame:
    """Top-k de productos cuya categoria no esta ya en el carrito."""
    allowed = ~grid.prefix_cats[:, grid.product_cat]
    lists = _rank_rows(primary, secondary, allowed, k)
    picked = grid.product_ids[np.maximum(lists, 0)].astype(object)
    picked[lists < 0] = None
    return grid.to_top_k(picked, k)


def baseline_random(grid: _Grid, k: int, *, seed: int) -> pd.DataFrame:
    """Cinco productos al azar (semilla fija) de categorias que no estan en el carrito."""
    noise = np.random.default_rng(seed).random((len(grid.basket_ids), len(grid.product_ids)))
    return _product_lists(grid, noise, np.zeros(len(grid.product_ids)), k)


def baseline_global_popularity(grid: _Grid, k: int) -> pd.DataFrame:
    """Los SKU mas vendidos en la ventana reciente: el `ORDER BY ventas DESC`."""
    return _product_lists(grid, grid.product_pop, np.zeros(len(grid.product_ids)), k)


def baseline_category_popularity(grid: _Grid, k: int) -> pd.DataFrame:
    """Las categorias mas vendidas, cada una con su referencia lider."""
    return _category_lists(grid, np.zeros(len(grid.categories)), grid.category_leader, k)


def _personal_category_frequency(grid: _Grid) -> np.ndarray:
    return grid.query_matrix(
        grid.data.customer_categories, "category", grid.cat_pos, "n_purchase_days"
    )


def baseline_personal_frequency(grid: _Grid, k: int, favorite: np.ndarray) -> pd.DataFrame:
    """Las categorias que el cliente compra en mas dias distintos, con su referencia favorita."""
    return _category_lists(grid, _personal_category_frequency(grid), favorite, k)


def baseline_personal_due(grid: _Grid, k: int, favorite: np.ndarray) -> pd.DataFrame:
    """Frecuencia personal, duplicada en las categorias a las que ya les toca reponer.

    `due` se evalua el dia de la cesta (`dias desde la ultima compra / intervalo esperado
    >= 1`), con el historial as-of de ese dia: el mismo que ven las features del ranker
    (punto A1), para que la comparacion sea justa.
    """
    cc = grid.data.customer_categories.assign(
        last_day=lambda d: pd.to_datetime(d["last_purchase_date"]).astype("int64") // _NS_PER_DAY
    )
    last = grid.query_matrix(cc, "category", grid.cat_pos, "last_day", fill=np.nan)
    expected = grid.query_matrix(
        cc, "category", grid.cat_pos, "expected_repurchase_days", fill=np.nan
    )
    day = pd.to_datetime(grid.data.queries["basket_day"]).astype("int64").to_numpy()
    with np.errstate(invalid="ignore"):
        ratio = (day[:, None] // _NS_PER_DAY - last) / expected
    due = np.nan_to_num(ratio, nan=0.0) >= 1.0
    primary = _personal_category_frequency(grid) * (1.0 + due)
    return _category_lists(grid, primary, favorite, k)


def baseline_repeat_favorite(grid: _Grid, k: int) -> pd.DataFrame:
    """Los SKU que el cliente mas ha comprado; los mas vendidos si no le llegan."""
    primary = grid.query_matrix(
        grid.data.customer_products, "product_id", grid.prod_pos, "n_baskets"
    )
    return _product_lists(grid, primary, grid.product_pop, k)


def baseline_association_rules(
    grid: _Grid, k: int, *, min_confidence: float, min_lift: float = 1.0
) -> pd.DataFrame:
    """Reglas categoria -> categoria disparadas por el carrito (tipo Apriori).

    Cada categoria candidata puntua con la regla mas confiable cuyo antecedente esta en
    el carrito, siempre que `confidence >= min_confidence` y `lift > min_lift`. Sin
    carrito, o sin reglas que pasen el umbral, rellena la popularidad de categoria.
    """
    rules = grid.data.category_rules
    rules = rules.loc[
        (rules["confidence"] >= min_confidence)
        & (rules["lift"] > min_lift)
        & rules["antecedent"].isin(grid.cat_pos.index)
        & rules["consequent"].isin(grid.cat_pos.index)
    ]
    n = len(grid.categories)
    conf = np.zeros((n, n))
    conf[
        grid.cat_pos.loc[rules["antecedent"]].to_numpy(),
        grid.cat_pos.loc[rules["consequent"]].to_numpy(),
    ] = rules["confidence"].to_numpy()
    # max_{a en carrito} conf(a -> c), por bloques para no materializar Q x C x C.
    primary = np.zeros(grid.prefix_cats.shape)
    step = 2_000
    for lo in range(0, len(primary), step):
        block = grid.prefix_cats[lo : lo + step, :, None] * conf[None, :, :]
        primary[lo : lo + step] = block.max(axis=1)
    return _category_lists(grid, primary, grid.category_leader, k)


def run_baselines(
    data: BaselineInputs, *, k: int, seed: int, min_confidence: float
) -> dict[str, pd.DataFrame]:
    """Top-k de cada baseline, con `label`, en el orden de `BASELINE_LABELS`."""
    grid = _Grid(data)
    favorite = grid.favorite_reference()
    return {
        "random": baseline_random(grid, k, seed=seed),
        "global_popularity": baseline_global_popularity(grid, k),
        "category_popularity": baseline_category_popularity(grid, k),
        "personal_frequency": baseline_personal_frequency(grid, k, favorite),
        "personal_due": baseline_personal_due(grid, k, favorite),
        "repeat_favorite": baseline_repeat_favorite(grid, k),
        "association_rules": baseline_association_rules(grid, k, min_confidence=min_confidence),
    }


def system_summary(
    top_k: pd.DataFrame,
    queries: pd.DataFrame,
    target: pd.DataFrame,
    product_category: pd.DataFrame,
    *,
    k: int,
    prefix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Metricas de SKU y de categoria de un top-k, en total y por perfil, en una tabla.

    Con `prefix`, anade las columnas de huecos regalados (`wasted_slot_metrics`).
    """
    sku = summarise(per_query_metrics(top_k, queries, k=k), k=k)
    cat = category_metrics(top_k, target, product_category, queries, k=k)
    out = sku.merge(cat.drop(columns="n_queries"), on="grupo")
    if prefix is not None:
        wasted = wasted_slot_metrics(top_k, prefix, product_category, queries, k=k)
        out = out.merge(wasted, on="grupo", how="left")
    return out

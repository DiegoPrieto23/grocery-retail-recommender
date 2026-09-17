"""Oraculo bayesiano: el techo teorico del acierto de categoria (punto A6 del diagnostico).

El generador conoce, para cada cesta, todo lo que decide que categorias entran
(`data_generation/export_oracle.py`). Un sistema que conociera esas probabilidades -- y
nada mas del futuro -- es el mejor recomendador posible de categorias: su acierto es el
techo contra el que tiene sentido medir el LambdaRank y los baselines. Lo que queda por
debajo de ese techo es entropia del propio proceso generativo, no un fallo del modelo.

## El proceso que hay que invertir (Fase 8)

Para cada cesta, el generador:

1. parte de un peso por categoria `w` (afinidad x estacionalidad x ciclo de reposicion);
2. sortea una **mision** `m` con la probabilidad de esa visita (`mission_prior`) y
   multiplica `w` por su perfil;
3. sortea el **numero de categorias distintas** `k = 1 + BN(media_m x f, r_m)`, con tope;
4. saca `k` categorias **una a una, sin reemplazo**, con probabilidad proporcional al
   peso; tras cada una, la fila de la categoria en la **matriz de interaccion** multiplica
   el resto (complementos hacia arriba, sustitutos hacia abajo).

Despues, `splits.build_query_items` reparte las lineas entre prefijo y target en un orden
que no tiene relacion con el del sorteo. El oraculo conoce el prefijo `S` (en
categorias) y cuantas categorias distintas tiene la cesta, `n`, igual que antes de la
Fase 8 conocia el tamano del target. No sabe la mision.

Con interacciones en las dos direcciones y cadenas (un sustituto que es a la vez
disparador de un complemento), la carrera de relojes exponenciales de la Fase 7 ya no
vale. Se usa **muestreo secuencial por importancia** (SIS), que replica el sorteo paso a
paso:

- la mision de cada muestra se estratifica entre las misiones posibles de la cesta
  (peso `prior_m x misiones posibles`), y la muestra arrastra `P(k = n | m)`;
- en cada paso, con `u` categorias del prefijo aun por salir y `r` pasos por dar, la
  propuesta elige "del prefijo" con probabilidad `u / r` y "de fuera" con `1 - u / r`, y
  dentro de cada grupo en proporcion al peso. Asi toda muestra acaba con el prefijo
  entero dentro de sus `n` categorias;
- el peso de la muestra multiplica, en cada paso, `P(real) / P(propuesta)`: `W_S / (W
  pS)` si sale del prefijo y `W_N / (W (1 - pS))` si sale de fuera.

El promedio de esos pesos es `P(S dentro de la cesta, n | cesta)`, y normalizados por
query dan `P(categoria en el resto | S, n)` sin aproximaciones: con infinitas muestras
es la probabilidad exacta del generador. Si en alguna query todas las muestras pesaran
0, se cuenta en `MonteCarloResult.n_fallback` (no deberia pasar: ningun factor del
proceso es 0 fuera de los gates, y el prefijo nunca cae en un gate cerrado).

## Las listas del oraculo

Las dos listas se eligen con la mitad par de las muestras y su valor esperado se mide con
la impar, para no premiar al oraculo con su propio ruido (las misiones se estratifican
por parejas de muestras, asi que las dos mitades ven las mismas misiones):

- **Oraculo de categoria:** las 5 categorias fuera del carrito con mas `P(categoria en el
  resto)`, cada una con su referencia mas probable. Es la lista optima en valor esperado
  salvo el ruido de estimacion, asi que el techo es una cota inferior muy ajustada del
  maximo alcanzable.
- **Oraculo de SKU:** las 5 categorias con mas `P(categoria en el resto) x P(mejor
  referencia en la cesta | categoria)`, con esa referencia. La probabilidad de la
  referencia cuenta la segunda referencia de las categorias de exploracion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WEIGHTS_FILENAME = "category_weights.parquet"
MISSIONS_FILENAME = "basket_missions.parquet"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class GenerativeProcess:
    """La parte del proceso del generador que no depende de la cesta."""

    mission_names: tuple[str, ...]
    profiles: np.ndarray  # (M, C) multiplicador de cada categoria en cada mision
    size_mean: np.ndarray  # (M,) media de categorias ademas de la primera
    size_dispersion: np.ndarray  # (M,) parametro r de la binomial negativa
    interaction: np.ndarray  # (C, C) fila = categoria que entra, columna = afectada
    max_categories: int

    @classmethod
    def from_manifest(cls, manifest: dict) -> "GenerativeProcess":
        missions = manifest["missions"]
        return cls(
            mission_names=tuple(m["name"] for m in missions),
            profiles=np.asarray(manifest["mission_profiles"], dtype=np.float64),
            size_mean=np.array([m["size_mean"] for m in missions], dtype=np.float64),
            size_dispersion=np.array([m["size_dispersion"] for m in missions], dtype=np.float64),
            interaction=np.asarray(manifest["interaction"], dtype=np.float64),
            max_categories=int(manifest["max_categories_per_basket"]),
        )

    def size_log_prob(
        self, n: np.ndarray, size_factor: np.ndarray, n_positive: np.ndarray
    ) -> np.ndarray:
        """`log P(la cesta tiene n categorias | mision)`, (Q, M).

        `k = 1 + BN(media x f, r)` con tope; si el tope o las categorias con peso > 0
        (`n_positive`) se alcanzan, la cesta se queda en ese numero para cualquier
        `k` mayor.
        """
        mean = self.size_mean[None, :] * size_factor[:, None]
        r = np.broadcast_to(self.size_dispersion[None, :], mean.shape)
        p = r / (r + np.maximum(mean, 1e-12))
        extra = (n - 1)[:, None].astype(np.float64)
        saturated = (n >= np.minimum(self.max_categories, n_positive))[:, None]
        with np.errstate(divide="ignore"):
            exact = stats.nbinom.logpmf(extra, r, p)
            tail = stats.nbinom.logsf(extra - 1, r, p)  # P(extra >= n - 1)
        return np.where(saturated, tail, exact)


@dataclass(frozen=True)
class OracleInputs:
    """Todo lo que usa el oraculo, alineado por query (filas) y categoria (columnas)."""

    basket_ids: np.ndarray  # (Q,)
    categories: list[str]  # (C,)
    weights: np.ndarray  # (Q, C) peso antes de aplicar la mision
    mission_prior: np.ndarray  # (Q, M) probabilidad de cada mision en esa visita
    size_factor: np.ndarray  # (Q,) factor de tamano (hogar x rampa de churn)
    best_product: np.ndarray  # (Q, C) referencia mas probable de cada categoria
    best_prob: np.ndarray  # (Q, C) P(esa referencia en la cesta | la categoria sale)
    prefix_mask: np.ndarray  # (Q, C) categorias ya en el carrito
    target_mask: np.ndarray  # (Q, C) categorias con alguna linea en el target
    process: GenerativeProcess

    @property
    def rest_mask(self) -> np.ndarray:
        """Categorias que faltaban de verdad: en el target y no en el carrito."""
        return self.target_mask & ~self.prefix_mask

    @property
    def n_target(self) -> np.ndarray:
        """Categorias distintas que faltaban (una categoria con una linea en cada lado no
        cuenta: ya estaba en el carrito)."""
        return self.rest_mask.sum(axis=1)

    @property
    def n_categories(self) -> np.ndarray:
        """Categorias distintas de la cesta entera."""
        return (self.prefix_mask | self.target_mask).sum(axis=1)


def load_inputs(
    oracle_dir: Path,
    queries: pd.DataFrame,
    context: pd.DataFrame,
    product_category: pd.DataFrame,
) -> OracleInputs:
    """Cruza lo exportado por el generador con las queries de test.

    Raises:
        FileNotFoundError: Si no se ha ejecutado `python -m data_generation.export_oracle`.
        ValueError: Si falta alguna query en el oraculo o el prefijo/target cae en una
            categoria con peso 0 (el oraculo no corresponderia al dataset).
    """
    oracle_dir = Path(oracle_dir)
    manifest = json.loads((oracle_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if "missions" not in manifest:
        raise ValueError(
            f"{oracle_dir / MANIFEST_FILENAME} es de un oraculo anterior a la Fase 8; "
            "volver a ejecutar `python -m data_generation.export_oracle`"
        )
    process = GenerativeProcess.from_manifest(manifest)
    categories: list[str] = manifest["categories"]
    pos = {name: i for i, name in enumerate(categories)}

    basket_ids = queries["basket_id"].to_numpy()
    row = {b: i for i, b in enumerate(basket_ids)}
    raw = pd.read_parquet(oracle_dir / WEIGHTS_FILENAME)
    raw = raw.loc[raw["basket_id"].isin(row)]
    missing = set(basket_ids) - set(raw["basket_id"])
    if missing:
        raise ValueError(
            f"{len(missing)} queries de test no estan en el oraculo "
            f"(p. ej. {sorted(missing)[:3]}); revisar --from-date del exportador"
        )

    q, c = len(basket_ids), len(categories)
    r = raw["basket_id"].map(row).to_numpy()
    k = raw["category"].map(pos).to_numpy()
    weights = np.zeros((q, c))
    weights[r, k] = raw["weight"].to_numpy()
    best_product = np.full((q, c), None, dtype=object)
    best_product[r, k] = raw["best_product_id"].to_numpy()
    best_prob = np.zeros((q, c))
    best_prob[r, k] = raw["best_product_prob"].to_numpy()

    missions = pd.read_parquet(oracle_dir / MISSIONS_FILENAME).set_index("basket_id")
    missions = missions.loc[basket_ids]
    mission_prior = missions[[f"mission_{m}" for m in process.mission_names]].to_numpy()
    size_factor = missions["size_factor"].to_numpy(dtype=np.float64)

    def mask(role: str) -> np.ndarray:
        rows = context.loc[context["role"] == role].merge(
            product_category[["product_id", "category"]], on="product_id"
        )
        out = np.zeros((q, c), dtype=bool)
        out[rows["basket_id"].map(row).to_numpy(), rows["category"].map(pos).to_numpy()] = True
        return out

    prefix_mask, target_mask = mask("prefix"), mask("target")
    impossible = ((prefix_mask | target_mask) & (weights <= 0)).any(axis=1)
    if impossible.any():
        raise ValueError(
            f"{int(impossible.sum())} queries compran categorias con peso 0 en el oraculo: "
            "no corresponde a este dataset"
        )
    return OracleInputs(
        basket_ids=basket_ids,
        categories=categories,
        weights=weights,
        mission_prior=mission_prior,
        size_factor=size_factor,
        best_product=best_product,
        best_prob=best_prob,
        prefix_mask=prefix_mask,
        target_mask=target_mask,
        process=process,
    )


def top_k_categories(scores: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """Las `k` columnas permitidas de mas puntuacion por fila (desempate por posicion).

    Devuelve -1 en los huecos que no se pueden llenar con puntuacion > 0.
    """
    masked = np.where(allowed, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    valid = np.take_along_axis(masked, order, axis=1) > 0
    return np.where(valid, order, -1)


def _stratified_missions(prior: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Mision de cada muestra, repartida por igual entre las misiones posibles de la query.

    Las muestras `2i` y `2i + 1` comparten mision, para que las mitades par e impar vean
    el mismo reparto. Devuelve `(mision, log del peso de la propuesta)`.
    """
    q = prior.shape[0]
    mission = np.empty((q, n_samples), dtype=np.int64)
    pair = np.arange(n_samples) // 2
    for i in range(q):
        active = np.flatnonzero(prior[i] > 0)
        mission[i] = active[pair % active.size]
    n_active = (prior > 0).sum(axis=1)
    with np.errstate(divide="ignore"):
        log_prior = np.log(np.take_along_axis(prior, mission, axis=1))
    return mission, log_prior + np.log(n_active)[:, None]


def simulate(
    weights: np.ndarray,
    mission_prior: np.ndarray,
    size_factor: np.ndarray,
    prefix_mask: np.ndarray,
    n_categories: np.ndarray,
    process: GenerativeProcess,
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cestas simuladas condicionadas al prefijo, con su peso de importancia (ver modulo).

    Args:
        weights: (Q, C) pesos antes de la mision.
        mission_prior: (Q, M) probabilidad de cada mision.
        size_factor: (Q,) factor de tamano.
        prefix_mask: (Q, C) categorias del prefijo.
        n_categories: (Q,) categorias distintas de la cesta.
        process: Constantes del proceso.
        rng: Generador de las muestras.
        n_samples: Muestras por query (par).

    Returns:
        `(in_rest, log_weight)`: (Q, N, C) si cada categoria cae en el resto y (Q, N) el
        logaritmo del peso de importancia.
    """
    q, c = weights.shape
    n = n_categories.astype(np.int64)
    mission, log_w = _stratified_missions(mission_prior, n_samples)
    size_lp = process.size_log_prob(n, size_factor, (weights > 0).sum(axis=1))
    log_w = log_w + np.take_along_axis(size_lp, mission, axis=1)

    w = (weights[:, None, :] * process.profiles[mission]).astype(np.float64)
    pending = np.broadcast_to(prefix_mask[:, None, :], w.shape).copy()
    in_rest = np.zeros(w.shape, dtype=bool)
    qi = np.arange(q)[:, None]
    si = np.arange(n_samples)[None, :]
    for step in range(int(n.max(initial=0))):
        live = np.broadcast_to((step < n)[:, None], (q, n_samples))
        remaining = np.maximum(n - step, 1)[:, None].astype(np.float64)
        u = pending.sum(axis=2)
        total = w.sum(axis=2)
        w_prefix = np.where(pending, w, 0.0).sum(axis=2)
        w_out = total - w_prefix
        p_prefix = np.where(w_out > 0, u / remaining, 1.0)
        from_prefix = rng.random((q, n_samples)) < p_prefix

        pool = np.where(from_prefix[:, :, None], pending, ~pending) & (w > 0)
        cum = np.cumsum(np.where(pool, w, 0.0), axis=2)
        draw = rng.random((q, n_samples)) * cum[:, :, -1]
        pick = np.minimum((cum < draw[:, :, None]).sum(axis=2), c - 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            step_w = np.where(
                from_prefix,
                np.log(w_prefix) - np.log(total) - np.log(p_prefix),
                np.log(w_out) - np.log(total) - np.log1p(-p_prefix),
            )
        dead = cum[:, :, -1] <= 0
        step_w = np.where(dead, -np.inf, step_w)
        log_w = np.where(live, log_w + step_w, log_w)

        was_pending = pending[qi, si, pick]
        in_rest[qi, si, pick] |= live & ~was_pending
        pending[qi, si, pick] &= ~live
        w[qi, si, pick] = np.where(live, 0.0, w[qi, si, pick])
        w *= np.where(live[:, :, None], process.interaction[pick], 1.0)
    log_w = np.where(pending.any(axis=2), -np.inf, log_w)
    return in_rest, np.nan_to_num(log_w, nan=-np.inf)


def _normalise(log_weight: np.ndarray) -> np.ndarray:
    """Pesos autonormalizados por query (cada fila suma 1)."""
    top = log_weight.max(axis=1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    w = np.exp(log_weight - top)
    total = w.sum(axis=1, keepdims=True)
    return np.divide(w, total, out=np.zeros_like(w), where=total > 0)


def _list_metrics(
    lists: np.ndarray, in_rest: np.ndarray, sku_prob: np.ndarray, w: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Metricas esperadas de una lista de categorias (con su mejor SKU) sobre las muestras."""
    valid = lists >= 0
    safe = np.where(valid, lists, 0)
    idx = np.broadcast_to(safe[:, None, :], (*in_rest.shape[:2], safe.shape[1]))
    hits = np.take_along_axis(in_rest, idx, axis=2) & valid[:, None, :]
    p = np.take_along_axis(sku_prob, safe, axis=1) * valid  # (Q, k)
    cat_hit = (w * hits.any(axis=2)).sum(axis=1)
    cat_prec = (w * hits.sum(axis=2)).sum(axis=1) / k
    miss_all = np.prod(np.where(hits, 1.0 - p[:, None, :], 1.0), axis=2)
    sku_hit = (w * (1.0 - miss_all)).sum(axis=1)
    sku_prec = (w * (hits * p[:, None, :]).sum(axis=2)).sum(axis=1) / k
    return cat_hit, cat_prec, sku_hit, sku_prec


@dataclass(frozen=True)
class MonteCarloResult:
    """Listas del oraculo y sus valores esperados por query."""

    cat_lists: np.ndarray  # (Q, k) categorias del oraculo de categoria
    sku_lists: np.ndarray  # (Q, k) categorias del oraculo de SKU
    cat_hit: np.ndarray  # P(la lista de categoria acierta alguna)
    cat_precision: np.ndarray  # E[aciertos de categoria] / k
    sku_hit: np.ndarray  # P(la lista de SKU acierta algun SKU)
    sku_precision: np.ndarray  # E[aciertos de SKU] / k
    ess: np.ndarray  # tamano efectivo de muestra por query (mitad de evaluacion)
    n_fallback: int  # queries sin ninguna muestra valida
    inclusion: np.ndarray  # (Q, C) P(categoria en el resto), con todas las muestras


def _effective_size(w: np.ndarray) -> np.ndarray:
    return np.divide(1.0, (w**2).sum(axis=1), out=np.zeros(w.shape[0]), where=(w**2).sum(axis=1) > 0)


def monte_carlo(
    inputs: OracleInputs,
    *,
    k: int,
    n_samples: int = 1_000,
    seed: int = 42,
    chunk_size: int = 64,
) -> MonteCarloResult:
    """Listas del oraculo y su valor esperado por query (ver docstring del modulo).

    Las queries se procesan agrupadas por numero de categorias, para que cada bloque dure
    tantos pasos como su cesta mas grande y no como la mayor de todas.
    """
    if n_samples < 2 or n_samples % 2:
        raise ValueError("n_samples debe ser par y >= 2")
    rng = np.random.default_rng(seed)
    q, n_cat = inputs.weights.shape
    allowed = ~inputs.prefix_mask
    n_categories = inputs.n_categories
    cat_lists = np.full((q, k), -1, dtype=int)
    sku_lists = np.full((q, k), -1, dtype=int)
    inclusion_all = np.zeros((q, n_cat))
    out = {name: np.zeros(q) for name in ("cat_hit", "cat_prec", "sku_hit", "sku_prec", "ess")}
    n_fallback = 0

    order = np.argsort(n_categories, kind="stable")
    for lo in range(0, q, chunk_size):
        idx = order[lo : lo + chunk_size]
        in_rest, log_w = simulate(
            inputs.weights[idx],
            inputs.mission_prior[idx],
            inputs.size_factor[idx],
            inputs.prefix_mask[idx],
            n_categories[idx],
            inputs.process,
            rng,
            n_samples,
        )
        n_fallback += int(np.isneginf(log_w).all(axis=1).sum())
        w_all = _normalise(log_w)
        w_even = _normalise(log_w[:, 0::2])
        w_odd = _normalise(log_w[:, 1::2])
        rest_odd = in_rest[:, 1::2, :]
        best = inputs.best_prob[idx]

        inclusion_all[idx] = (w_all[:, :, None] * in_rest).sum(axis=1)
        inclusion = (w_even[:, :, None] * in_rest[:, 0::2, :]).sum(axis=1)
        cat_lists[idx] = top_k_categories(inclusion, allowed[idx], k)
        sku_lists[idx] = top_k_categories(inclusion * best, allowed[idx], k)
        out["cat_hit"][idx], out["cat_prec"][idx], _, _ = _list_metrics(
            cat_lists[idx], rest_odd, best, w_odd, k
        )
        _, _, out["sku_hit"][idx], out["sku_prec"][idx] = _list_metrics(
            sku_lists[idx], rest_odd, best, w_odd, k
        )
        out["ess"][idx] = _effective_size(w_odd)

    return MonteCarloResult(
        cat_lists=cat_lists,
        sku_lists=sku_lists,
        cat_hit=out["cat_hit"],
        cat_precision=out["cat_prec"],
        sku_hit=out["sku_hit"],
        sku_precision=out["sku_prec"],
        ess=out["ess"],
        n_fallback=n_fallback,
        inclusion=inclusion_all,
    )


def lists_to_top_k(inputs: OracleInputs, lists: np.ndarray, target: pd.DataFrame) -> pd.DataFrame:
    """Convierte listas de categorias en un top-k evaluable (`basket_id`, `product_id`, `rank`, `label`).

    Cada categoria se concreta en su referencia mas probable. `label` es 1 si esa
    referencia esta en el target real, igual que en el resto de sistemas.
    """
    q, k = lists.shape
    rows = np.repeat(np.arange(q), k)
    cols = lists.ravel()
    keep = cols >= 0
    frame = pd.DataFrame(
        {
            "basket_id": inputs.basket_ids[rows[keep]],
            "product_id": inputs.best_product[rows[keep], cols[keep]],
            "rank": np.tile(np.arange(1, k + 1), q)[keep],
        }
    )
    truth = set(zip(target["basket_id"], target["product_id"]))
    frame["label"] = [int(pair in truth) for pair in zip(frame["basket_id"], frame["product_id"])]
    return frame

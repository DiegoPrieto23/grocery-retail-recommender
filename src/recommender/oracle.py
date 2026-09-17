"""Oraculo bayesiano: el techo teorico del acierto de categoria (punto A6 del diagnostico).

El generador conoce, para cada cesta, el peso con el que sale cada categoria antes del
primer sorteo (`data_generation/export_oracle.py`). Un sistema que conociera esos pesos
-- y nada mas del futuro -- es el mejor recomendador posible de categorias: su acierto es
el techo contra el que tiene sentido medir el LambdaRank y los baselines. Lo que queda por
debajo de ese techo es entropia del propio proceso generativo, no un fallo del modelo.

## El proceso que hay que invertir

El generador saca las categorias de una cesta **una a una, sin reemplazo**, con
probabilidad proporcional a su peso. Al salir una disparadora de `AFFINITY_PAIRS`, su
asociada multiplica el peso por el lift. Despues, `splits.build_query_items` reparte las
categorias sorteadas entre prefijo y target en un orden que no tiene relacion con el del
sorteo (permutacion aleatoria o hash).

Ese sorteo secuencial es exactamente una **carrera de relojes exponenciales**: cada
categoria `c` tiene un reloj `T_c ~ Exp(w_c)` y la cesta de `n` lineas son los `n`
relojes que suenan antes. El lift tambien encaja: cuando suena la disparadora, el reloj
restante de la asociada pasa a correr `lift` veces mas rapido (la exponencial no tiene
memoria), y el orden de llegada sigue siendo el del sorteo con pesos actualizados.

Dado el prefijo `S` y el tamano del target `T`, el resto `R` son las `T` primeras
categorias de fuera de `S`, y la unica condicion es que **todo `S` haya sonado antes que
la categoria `T+1` de fuera** (instante `t`). Por eso basta con sortear los relojes de
fuera de `S` y ponderar cada muestra por la probabilidad de esa condicion:

- una categoria del prefijo que no es asociada de nada: `1 - exp(-w_i t)`;
- una asociada del prefijo cuya disparadora sono en `tau`: `1 - exp(-H_i)`, con
  `H_i = w_i min(tau, t) + lift w_i max(0, t - tau)`;
- una disparadora del prefijo: su reloj si se sortea (su instante mueve a su asociada),
  y la muestra cuenta solo si sono antes de `t`.

Es muestreo por importancia autonormalizado y **no tiene aproximaciones**: con infinitas
muestras da la probabilidad exacta del generador. Si en alguna query ninguna muestra
cumple la condicion (disparadora del prefijo con muy poco peso), se usa para ella el peso
sin el indicador y se cuenta en `MonteCarloResult.n_fallback`.

## Las listas del oraculo

- **Oraculo de categoria:** las 5 categorias de mas peso fuera del carrito, con el lift
  de las disparadoras del carrito ya aplicado (`adjusted_weights`), cada una con su
  referencia mas probable. Fuera de los pares de afinidad, ordenar por peso es ordenar
  por probabilidad de estar en el resto (con `t` fijo, `1 - exp(-w t)` crece con `w`),
  asi que la lista es la optima salvo en esos 10 pares. Se elige asi, y no con las
  probabilidades simuladas, porque es determinista: ordenar por estimaciones de Monte
  Carlo desordena las categorias casi empatadas y la lista sale peor. Su valor esperado
  si se calcula con la simulacion exacta. El techo resultante es, por tanto, una cota
  inferior muy ajustada del maximo alcanzable.
- **Oraculo de SKU:** las 5 categorias con mas `P(categoria en el resto) x P(mejor
  referencia | categoria)`, con esa referencia. Aqui no hay orden determinista posible
  (la probabilidad de inclusion no es lineal en el peso), asi que las probabilidades se
  estiman con la mitad par de las muestras y el valor esperado se mide con la impar,
  para no premiar al oraculo con su propio ruido.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS_FILENAME = "category_weights.parquet"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class OracleInputs:
    """Todo lo que usa el oraculo, alineado por query (filas) y categoria (columnas)."""

    basket_ids: np.ndarray  # (Q,)
    categories: list[str]  # (C,)
    weights: np.ndarray  # (Q, C) peso antes del primer sorteo
    best_product: np.ndarray  # (Q, C) referencia mas probable de cada categoria
    best_prob: np.ndarray  # (Q, C) su probabilidad si la categoria sale
    prefix_mask: np.ndarray  # (Q, C) categorias ya en el carrito
    target_mask: np.ndarray  # (Q, C) categorias que realmente faltaban
    pairs: tuple[tuple[int, int, float], ...]  # (disparadora, asociada, lift aplicado)

    @property
    def n_target(self) -> np.ndarray:
        return self.target_mask.sum(axis=1)


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
    weights_path = Path(oracle_dir) / WEIGHTS_FILENAME
    manifest = json.loads((Path(oracle_dir) / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    categories: list[str] = manifest["categories"]
    pos = {name: i for i, name in enumerate(categories)}
    pairs = tuple(
        (pos[p["trigger"]], pos[p["associated"]], float(p["applied_lift"]))
        for p in manifest["affinity_pairs"]
    )
    triggers = {t for t, _, _ in pairs}
    if triggers & {a for _, a, _ in pairs}:
        # La simulacion aplica los lifts en un solo paso; con cadenas haria falta ordenar.
        raise ValueError("Una categoria no puede ser disparadora y asociada a la vez")

    basket_ids = queries["basket_id"].to_numpy()
    row = {b: i for i, b in enumerate(basket_ids)}
    raw = pd.read_parquet(weights_path)
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
        best_product=best_product,
        best_prob=best_prob,
        prefix_mask=prefix_mask,
        target_mask=target_mask,
        pairs=pairs,
    )


def adjusted_weights(
    weights: np.ndarray, prefix_mask: np.ndarray, pairs: tuple[tuple[int, int, float], ...]
) -> np.ndarray:
    """Pesos del resto de la cesta: el prefijo a 0 y el lift de sus disparadoras aplicado."""
    out = np.where(prefix_mask, 0.0, weights)
    for trigger, associated, lift in pairs:
        out[:, associated] *= np.where(prefix_mask[:, trigger], lift, 1.0)
    return out


def top_k_categories(scores: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """Las `k` columnas permitidas de mas puntuacion por fila (desempate por posicion).

    Devuelve -1 en los huecos que no se pueden llenar con puntuacion > 0.
    """
    masked = np.where(allowed, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    valid = np.take_along_axis(masked, order, axis=1) > 0
    return np.where(valid, order, -1)


def race(
    weights: np.ndarray,
    prefix_mask: np.ndarray,
    n_target: np.ndarray,
    pairs: tuple[tuple[int, int, float], ...],
    exp_draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simula el resto de cada cesta con relojes exponenciales, condicionado al prefijo.

    Args:
        weights: (Q, C) pesos antes del primer sorteo.
        prefix_mask: (Q, C) categorias del prefijo.
        n_target: (Q,) categorias que faltaban.
        pairs: pares de afinidad `(disparadora, asociada, lift)`.
        exp_draws: (Q, N, C) variables Exp(1).

    Returns:
        `(in_rest, log_weight, log_weight_fallback)`: (Q, N, C) si cada categoria cae en
        el resto, y (Q, N) el logaritmo del peso de importancia exacto y el del peso sin
        el indicador de las disparadoras del prefijo (ver docstring del modulo).
    """
    with np.errstate(divide="ignore"):
        times = exp_draws / weights[:, None, :]  # peso 0 -> reloj infinito
    for trigger, associated, lift in pairs:
        tau = times[:, :, trigger]
        t_assoc = times[:, :, associated]
        # inf - inf da NaN, pero solo en ramas que el `where` descarta.
        with np.errstate(invalid="ignore"):
            sped_up = tau + (t_assoc - tau) / lift
        times[:, :, associated] = np.where(t_assoc > tau, sped_up, t_assoc)

    outside = np.where(prefix_mask[:, None, :], np.inf, times)
    n_cat = times.shape[2]
    kth = np.minimum(n_target, n_cat - 1)
    ordered = np.sort(outside, axis=2)
    threshold = np.take_along_axis(
        ordered, np.broadcast_to(kth[:, None, None], (*times.shape[:2], 1)), axis=2
    )[:, :, 0]
    threshold = np.where((n_target >= n_cat)[:, None], np.inf, threshold)
    in_rest = outside < threshold[:, :, None]

    # Riesgo acumulado de cada categoria del prefijo hasta `t`.
    t = threshold[:, :, None]
    hazard = weights[:, None, :] * t
    for trigger, associated, lift in pairs:
        tau = times[:, :, trigger]
        th = threshold
        w_a = weights[:, associated][:, None]
        with np.errstate(invalid="ignore"):
            piecewise = w_a * np.minimum(tau, th) + lift * w_a * np.maximum(0.0, th - tau)
        hazard[:, :, associated] = np.where(np.isinf(th), np.inf, piecewise)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_cdf = np.log(-np.expm1(-hazard))
    in_prefix = prefix_mask[:, None, :]
    log_cdf = np.where(in_prefix, log_cdf, 0.0)
    fallback = log_cdf.sum(axis=2)

    is_trigger = np.zeros(n_cat, dtype=bool)
    is_trigger[[t for t, _, _ in pairs]] = True
    early = times < t
    exact_terms = np.where(
        in_prefix & is_trigger, np.where(early, 0.0, -np.inf), log_cdf
    )
    return in_rest, exact_terms.sum(axis=2), fallback


def _normalise(log_weight: np.ndarray) -> np.ndarray:
    """Pesos autonormalizados por query (cada fila suma 1)."""
    top = log_weight.max(axis=1, keepdims=True)
    w = np.exp(log_weight - top)
    return w / w.sum(axis=1, keepdims=True)


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
    ess: np.ndarray  # tamano efectivo de muestra por query (todas las muestras)
    n_fallback: int  # queries en las que no valio el peso exacto


def _effective_size(w: np.ndarray) -> np.ndarray:
    return 1.0 / (w**2).sum(axis=1)


def monte_carlo(
    inputs: OracleInputs,
    *,
    k: int,
    n_samples: int = 1_000,
    seed: int = 42,
    chunk_size: int = 100,
) -> MonteCarloResult:
    """Listas del oraculo y su valor esperado por query (ver docstring del modulo)."""
    if n_samples < 2 or n_samples % 2:
        raise ValueError("n_samples debe ser par y >= 2")
    rng = np.random.default_rng(seed)
    q, n_cat = inputs.weights.shape
    allowed = ~inputs.prefix_mask
    n_target = inputs.n_target
    cat_lists = top_k_categories(
        adjusted_weights(inputs.weights, inputs.prefix_mask, inputs.pairs), allowed, k
    )
    sku_lists = np.full((q, k), -1, dtype=int)
    out = {name: np.zeros(q) for name in ("cat_hit", "cat_prec", "sku_hit", "sku_prec", "ess")}
    n_fallback = 0
    halves = {"all": slice(None), "even": slice(0, None, 2), "odd": slice(1, None, 2)}

    for lo in range(0, q, chunk_size):
        sl = slice(lo, min(lo + chunk_size, q))
        draws = rng.standard_exponential((sl.stop - sl.start, n_samples, n_cat))
        in_rest, log_w, log_w_fb = race(
            inputs.weights[sl], inputs.prefix_mask[sl], n_target[sl], inputs.pairs, draws
        )
        w = {}
        for name, part in halves.items():
            exact = log_w[:, part]
            degenerate = np.isneginf(exact).all(axis=1)
            if name == "all":
                n_fallback += int(degenerate.sum())
            w[name] = _normalise(np.where(degenerate[:, None], log_w_fb[:, part], exact))
        best = inputs.best_prob[sl]

        out["cat_hit"][sl], out["cat_prec"][sl], _, _ = _list_metrics(
            cat_lists[sl], in_rest, best, w["all"], k
        )
        out["ess"][sl] = _effective_size(w["all"])

        # Oraculo de SKU: elegir con las muestras pares, medir con las impares.
        inclusion = (w["even"][:, :, None] * in_rest[:, 0::2, :]).sum(axis=1)
        sku_lists[sl] = top_k_categories(inclusion * best, allowed[sl], k)
        _, _, out["sku_hit"][sl], out["sku_prec"][sl] = _list_metrics(
            sku_lists[sl], in_rest[:, 1::2, :], best, w["odd"], k
        )

    return MonteCarloResult(
        cat_lists=cat_lists,
        sku_lists=sku_lists,
        cat_hit=out["cat_hit"],
        cat_precision=out["cat_prec"],
        sku_hit=out["sku_hit"],
        sku_precision=out["sku_prec"],
        ess=out["ess"],
        n_fallback=n_fallback,
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

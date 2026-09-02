"""Politica de valor esperado: de dos probabilidades a una accion por cliente.

    accion* = argmax_a ( P(conversion | a) x margen_esperado(a) - coste(a) )

La formula es la de `CHALLENGE.md`. Lo que sigue es como se instancia cada termino sin
hacer trampa.

## Todo se mide en incremental, no en absoluto

La tentacion es puntuar cada accion por el valor que genera y quedarse con la mayor. Eso
premia a la accion cara solo por ir asociada a clientes que iban a comprar de todas
formas. Aqui cada accion se puntua por lo que anade **sobre no hacer nada**:

    V(a)       = P(compra | a) x (margen_bruto - descuento(a)) - coste_envio(a)
    V(ninguna) = P(compra)     x  margen_bruto
    delta(a)   = V(a) - V(ninguna)

Asi `ninguna_accion` vale 0 por construccion, y una accion solo gana si su efecto paga su
coste. Un cupon de 2,54 EUR necesita mover bastante la aguja en un cliente con margen
esperado de 1,4 EUR: la mayoria de las veces no lo paga, y eso es precisamente lo que la
politica tiene que descubrir.

## Los dos costes no son el mismo coste

`coste_envio` se paga siempre que se ejecuta la accion. El `descuento` del cupon solo se
paga **si el cliente compra**, porque es valor facial que se descuenta del ticket. Por eso
entra restando dentro del parentesis, multiplicado por la probabilidad, y no fuera como un
coste fijo. Meterlo fuera penalizaria de mas a los clientes con baja propension y la
politica se volveria trivialmente conservadora.

## La retencion entra aparte

Actuar sobre un cliente que se esta yendo tiene un segundo efecto: puede que no se vaya.

    delta_retencion(a) = reduccion_churn(a) x P(churn) x valor_de_retener

donde `valor_de_retener` son `retention_weeks` semanas de compra futura al margen del
cliente. Ese termino no depende de la categoria, asi que solo es coherente porque la
politica elige **una** accion por cliente.

## El limite honesto

`P(compra | a)` y `reduccion_churn(a)` son supuestos declarados en `config.py`, no
estimaciones: este dataset no permite identificarlos (ver la cabecera de ese modulo). Lo
que **si** es merito del modelo es el reparto: con el uplift fijado, toda la diferencia
entre la politica y "actuar siempre" viene de acertar a quien. Por eso `sensitivity`
recorre el uplift del cupon y muestra donde deja de compensar.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.nba.config import Action, NBAConfig, PolicyConfig

# Columnas que describen a un candidato antes de evaluar acciones.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "category",
    "department",
    "p_purchase",
    "expected_spend",
    "p_churn",
    "retention_value",
)


def expected_spend(
    pdf: pd.DataFrame, cfg: PolicyConfig, *, fallback: float
) -> pd.Series:
    """Cuanto se gasta el cliente en esa categoria cuando la compra.

    Se usa su propia media historica (`cat_spend / cat_n_purchase_days`) y, cuando no hay
    historial suficiente, la media global de gasto por cesta-categoria. El techo evita que
    un outlier de ticket se lleve todo el presupuesto de campana.
    """
    per_visit = pdf["cat_spend"] / pdf["cat_n_purchase_days"].replace(0, np.nan)
    return per_visit.fillna(fallback).clip(upper=cfg.max_expected_spend)


def blended_margin(pdf: pd.DataFrame, cfg: PolicyConfig) -> pd.Series:
    """Margen medio del cliente, ponderado por su propia mezcla de departamentos.

    Un cliente que solo compra fresco no vale lo mismo que uno que llena el carro de
    droguería, y el valor de retenerlo tiene que reflejarlo.
    """
    rate = pdf["department"].map(cfg.margins.by_department).fillna(cfg.margins.default)
    weighted = (pdf["cat_spend"] * rate).groupby(pdf["customer_id"]).sum()
    total = pdf["cat_spend"].groupby(pdf["customer_id"]).sum()
    return (weighted / total.replace(0, np.nan)).fillna(cfg.margins.default)


def build_candidates(
    scored: pd.DataFrame,
    churn: pd.DataFrame,
    cfg: NBAConfig,
    *,
    fallback_spend: float,
) -> pd.DataFrame:
    """Une propension de categoria, propension de churn y economia en una sola tabla.

    Args:
        scored: Una fila por `(customer_id, category)` con `p_purchase`, `department`,
            `cat_spend` y `cat_n_purchase_days`.
        churn: Una fila por `customer_id` con `p_churn` y `spend_90d`.
        cfg: Configuracion completa.
        fallback_spend: Gasto medio por cesta-categoria del dataset, para los pares sin
            historial suficiente.

    Returns:
        Una fila por candidato con todo lo necesario para valorar cualquier accion.
    """
    pol = cfg.policy
    out = scored.copy()
    out["expected_spend"] = expected_spend(out, pol, fallback=fallback_spend)
    out["margin_rate"] = (
        out["department"].map(pol.margins.by_department).fillna(pol.margins.default)
    )
    out["gross_margin"] = out["expected_spend"] * out["margin_rate"]

    per_customer = blended_margin(out, pol).rename("blended_margin").reset_index()
    churn = churn.merge(per_customer, on="customer_id", how="left")
    churn["blended_margin"] = churn["blended_margin"].fillna(pol.margins.default)
    # Gasto semanal reciente x semanas que se dan por salvadas x margen del cliente.
    weekly = churn["spend_90d"] * (7.0 / 90.0)
    churn["retention_value"] = pol.retention_weeks * weekly * churn["blended_margin"]

    return out.merge(
        churn[["customer_id", "p_churn", "retention_value"]], on="customer_id", how="left"
    ).fillna({"p_churn": 0.0, "retention_value": 0.0})


def with_coupon(
    cfg: NBAConfig,
    *,
    conversion_uplift: float | None = None,
    churn_reduction: float | None = None,
) -> NBAConfig:
    """Copia de la configuracion con los supuestos del cupon sustituidos.

    Es el gancho de los barridos de sensibilidad: en vez de hacer viajar un override por
    cada funcion de la politica, se cambia la accion en la configuracion y todo lo demas
    sigue leyendo de donde siempre.
    """
    actions = tuple(
        replace(
            a,
            conversion_uplift=(
                a.conversion_uplift if conversion_uplift is None else conversion_uplift
            ),
            churn_reduction=(a.churn_reduction if churn_reduction is None else churn_reduction),
        )
        if a.name == "enviar_cupon_categoria"
        else a
        for a in cfg.actions
    )
    return replace(cfg, actions=actions)


def action_value(candidates: pd.DataFrame, action: Action) -> pd.Series:
    """Valor incremental de una accion sobre no hacer nada, en euros por cliente."""
    if (
        action.conversion_uplift == 1.0
        and action.send_cost == 0.0
        and action.discount == 0.0
        and action.churn_reduction == 0.0
    ):
        return pd.Series(0.0, index=candidates.index)

    uplift = action.conversion_uplift
    p0 = candidates["p_purchase"].to_numpy()
    margin = candidates["gross_margin"].to_numpy()
    # La probabilidad no puede pasar de 1 por mucho que se multiplique.
    p1 = np.minimum(p0 * uplift, 1.0)

    v_action = p1 * (margin - action.discount) - action.send_cost
    v_nothing = p0 * margin
    cross_sell = v_action - v_nothing

    retention = (
        action.churn_reduction
        * candidates["p_churn"].to_numpy()
        * candidates["retention_value"].to_numpy()
    )
    return pd.Series(cross_sell + retention, index=candidates.index)


def score_actions(candidates: pd.DataFrame, cfg: NBAConfig) -> pd.DataFrame:
    """Valor incremental de cada accion sobre cada candidato, en formato ancho."""
    out = candidates[list(CANDIDATE_COLUMNS)].copy()
    for action in cfg.actions:
        out[f"ev_{action.name}"] = action_value(candidates, action)
    return out


def decide(candidates: pd.DataFrame, cfg: NBAConfig) -> pd.DataFrame:
    """La tabla de la Tarea 3b: `customer_id` -> accion recomendada -> valor esperado.

    Se elige el maximo sobre el producto cartesiano de categorias y acciones. Si ninguna
    combinacion supera el cero, gana `ninguna_accion`, que es el resultado correcto y no
    un fallo: para la mayoria de los clientes lo mas rentable es no gastar nada.
    """
    scored = score_actions(candidates, cfg)
    value_cols = [f"ev_{a.name}" for a in cfg.actions]

    long = scored.melt(
        id_vars=["customer_id", "category", "p_purchase", "p_churn"],
        value_vars=value_cols,
        var_name="action",
        value_name="expected_value",
    )
    long["action"] = long["action"].str.removeprefix("ev_")

    idx = long.groupby("customer_id")["expected_value"].idxmax()
    best = long.loc[idx].reset_index(drop=True)

    # `ninguna_accion` no tiene categoria: se limpia para que la tabla no sugiera que la
    # hubo. Y si el maximo es <= 0, se fuerza igualmente la accion nula.
    inactive = best["expected_value"] <= 0
    best.loc[inactive, "action"] = "ninguna_accion"
    best.loc[inactive, "expected_value"] = 0.0
    best.loc[best["action"] == "ninguna_accion", "category"] = None

    action_ids = {a.name: a.action_id for a in cfg.actions}
    best["action_id"] = best["action"].map(action_ids).astype("int32")
    return best[
        ["customer_id", "action_id", "action", "category", "p_purchase", "p_churn", "expected_value"]
    ]


def cover_all(actions: pd.DataFrame, customer_ids: pd.Series) -> pd.DataFrame:
    """Completa la tabla con los clientes que no tenian ningun candidato.

    Un cliente activo que no compra ninguna categoria desde hace `category_lookback_days`
    no genera pares, y sin pares la politica no puede elegir categoria. Dejarlo fuera de la
    tabla seria mentir por omision: la Tarea 3b pide una accion **por cliente**, y "no hay
    nada que ofrecerle que merezca la pena" es una respuesta, no una ausencia. Se le asigna
    `ninguna_accion` de forma explicita.
    """
    missing = pd.Index(customer_ids.unique()).difference(pd.Index(actions["customer_id"]))
    if missing.empty:
        return actions
    filler = pd.DataFrame(
        {
            "customer_id": missing,
            "action_id": 0,
            "action": "ninguna_accion",
            "category": None,
            "p_purchase": np.nan,
            "p_churn": np.nan,
            "expected_value": 0.0,
        }
    )
    return pd.concat([actions, filler], ignore_index=True)


def never(candidates: pd.DataFrame) -> pd.DataFrame:
    """Baseline "no actuar siempre". Vale 0 por definicion de valor incremental."""
    out = candidates[["customer_id"]].drop_duplicates().reset_index(drop=True)
    out["action_id"] = 0
    out["action"] = "ninguna_accion"
    out["category"] = None
    out["p_purchase"] = np.nan
    out["p_churn"] = np.nan
    out["expected_value"] = 0.0
    return out


def always(candidates: pd.DataFrame, cfg: NBAConfig, action_name: str) -> pd.DataFrame:
    """Baseline "actuar siempre": la misma accion a todo el mundo.

    Se aplica sobre la categoria de mayor propension de cada cliente, que es lo que haria
    una campana no segmentada razonable -- no sobre una categoria al azar, que seria un
    hombre de paja demasiado facil de batir.
    """
    action = next(a for a in cfg.actions if a.name == action_name)
    values = action_value(candidates, action)
    frame = candidates[["customer_id", "category", "p_purchase", "p_churn"]].copy()
    frame["expected_value"] = values
    idx = frame.groupby("customer_id")["p_purchase"].idxmax()
    best = frame.loc[idx].reset_index(drop=True)
    best["action"] = action.name
    best["action_id"] = action.action_id
    return best[
        ["customer_id", "action_id", "action", "category", "p_purchase", "p_churn", "expected_value"]
    ]


def compare(
    candidates: pd.DataFrame, cfg: NBAConfig, *, all_customers: pd.Series | None = None
) -> pd.DataFrame:
    """La politica frente a las alternativas triviales, en euros de valor incremental.

    Args:
        candidates: Salida de `build_candidates`.
        cfg: Configuracion completa.
        all_customers: Universo sobre el que comparar. Si se pasa, los clientes sin ningun
            candidato entran en las cuatro filas con valor 0 -- no pueden recibir accion en
            ninguna de las politicas, asi que no alteran la comparacion, pero mantienen el
            denominador honesto: el `pct_accion` se lee sobre el maestro y no sobre el
            subconjunto comodo de los que si tenian algo que ofrecer.
    """
    rows = []

    def _row(name: str, table: pd.DataFrame) -> dict:
        if all_customers is not None:
            table = cover_all(table, all_customers)
        acted = table["action"] != "ninguna_accion"
        return {
            "politica": name,
            "n_clientes": int(len(table)),
            "n_acciones": int(acted.sum()),
            "pct_accion": float(acted.mean()),
            "valor_total": float(table["expected_value"].sum()),
            "valor_por_cliente": float(table["expected_value"].mean()),
        }

    rows.append(_row("no actuar siempre", never(candidates)))
    for action in cfg.actions:
        if action.name == "ninguna_accion":
            continue
        rows.append(_row(f"actuar siempre: {action.name}", always(candidates, cfg, action.name)))
    rows.append(_row("politica de valor esperado", decide(candidates, cfg)))

    out = pd.DataFrame(rows)
    baseline = out.loc[out["politica"] == "no actuar siempre", "valor_total"].iloc[0]
    out["uplift_vs_no_actuar"] = out["valor_total"] - baseline
    return out


def _sweep_row(
    candidates: pd.DataFrame,
    cfg: NBAConfig,
    *,
    all_customers: pd.Series | None,
) -> dict:
    """Una fila de barrido: valor de la politica y del cupon a todos, con su reparto."""
    policy = decide(candidates, cfg)
    coupon_always = always(candidates, cfg, "enviar_cupon_categoria")
    if all_customers is not None:
        policy = cover_all(policy, all_customers)
        coupon_always = cover_all(coupon_always, all_customers)
    return {
        "valor_politica": float(policy["expected_value"].sum()),
        "valor_cupon_a_todos": float(coupon_always["expected_value"].sum()),
        "pct_accion": float((policy["action"] != "ninguna_accion").mean()),
        "pct_cupon": float((policy["action"] == "enviar_cupon_categoria").mean()),
    }


def sensitivity(
    candidates: pd.DataFrame, cfg: NBAConfig, *, all_customers: pd.Series | None = None
) -> pd.DataFrame:
    """Barrido del **uplift de conversion** supuesto del cupon.

    Dice cuanto del resultado depende de creerse que un cupon hace comprar mas.
    """
    rows = []
    for uplift in cfg.policy.sensitivity_uplifts:
        variant = with_coupon(cfg, conversion_uplift=uplift)
        rows.append(
            {"uplift_cupon": uplift, **_sweep_row(candidates, variant, all_customers=all_customers)}
        )
    return pd.DataFrame(rows)


def sensitivity_retention(
    candidates: pd.DataFrame, cfg: NBAConfig, *, all_customers: pd.Series | None = None
) -> pd.DataFrame:
    """Barrido de la **reduccion de churn** supuesta del cupon.

    Este es el barrido que de verdad importa, y merece la pena explicar por que.

    Con el cupon anclado a su valor real (2,54 EUR) y un margen esperado por categoria de
    alrededor de 1,4 EUR, el descuento **es mayor que el margen que persigue**: por el lado
    del cross-sell el cupon destruye valor pase lo que pase con la conversion. Todo lo que
    aporta viene del otro termino, el de retencion. Por eso barrer solo `conversion_uplift`
    -- que es lo primero que uno piensa -- deja la conclusion sin auditar: hay que mover
    `churn_reduction`, que es el supuesto del que cuelga el resultado.

    Con `churn_reduction = 0` la politica se queda con lo que valga el cross-sell puro, que
    para el cupon es negativo: si esa fila sigue dando un valor apreciable, viene de
    `recomendar_producto` y no del cupon.
    """
    rows = []
    for reduction in cfg.policy.sensitivity_churn_reductions:
        variant = with_coupon(cfg, churn_reduction=reduction)
        rows.append(
            {
                "reduccion_churn": reduction,
                **_sweep_row(candidates, variant, all_customers=all_customers),
            }
        )
    return pd.DataFrame(rows)

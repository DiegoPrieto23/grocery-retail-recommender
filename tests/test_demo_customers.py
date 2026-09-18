"""A quien ofrece la demo como cliente (`src/demo/customers.py`).

El selector ofrecia los 60 clientes con mas historial, y eso resultaba ser la cola extrema:
todos con mas de 400 cestas, casi todos con el mismo riesgo de fuga y la misma accion del
NBA. La demo presume de que las recomendaciones se adaptan a quien compra y luego ofrecia
60 clientes intercambiables; ademas, dos de ellos no tenian ninguna cesta de test, asi que
al elegirlos media demo no funcionaba.

Los tests van sobre tablas escritas a mano, como en el resto de la bateria: la propiedad
que se fija es la **regla**, no las cifras del dataset de hoy.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.demo import customers as cu


def _stats(**por_cliente: tuple[float, float, float]) -> pd.DataFrame:
    """`customer_stats` a mano: `cliente=(cestas, ticket, referencias)`."""
    return pd.DataFrame(
        [
            {
                "customer_id": cid,
                "cust_frequency": n,
                "cust_avg_ticket": ticket,
                "cust_n_products": refs,
            }
            for cid, (n, ticket, refs) in por_cliente.items()
        ]
    )


def _actions(**churn: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"customer_id": cid, "p_churn": p} for cid, p in churn.items()]
    )


def _pool_basico() -> pd.DataFrame:
    """Cuatro clientes que cubren los tres escenarios y el caso sin cesta de test."""
    stats = _stats(
        C1=(100, 50.0, 300),  # compra mucho, sin riesgo -> Fiel
        C2=(3, 20.0, 12),  # compra poco -> Ocasional
        C3=(50, 40.0, 150),  # riesgo alto -> En riesgo
        C4=(80, 45.0, 200),  # sin cestas de test -> fuera
    )
    test = pd.Series({"C1": 5, "C2": 2, "C3": 3, "C4": 0})
    acciones = _actions(C1=0.05, C2=0.10, C3=0.90, C4=0.05)
    return cu.build_pool(stats, test, acciones)


# --------------------------------------------------------------------------------------
# Quien entra
# --------------------------------------------------------------------------------------
def test_solo_entran_clientes_con_cestas_de_test() -> None:
    """Es la condicion que hace que las dos mitades de la demo funcionen.

    Sin cesta de test no se puede cargar una compra real ni contrastar el top-5 contra lo
    que el cliente compro. Antes se colaban, y solo se descubria despues de elegir.
    """
    pool = _pool_basico()
    assert set(pool["customer_id"]) == {"C1", "C2", "C3"}
    assert (pool["n_test"] > 0).all()


def test_un_cliente_sin_fila_en_el_nba_no_se_queda_fuera() -> None:
    """No tener accion calculada no deberia expulsarte del selector."""
    stats = _stats(C1=(10, 30.0, 50))
    pool = cu.build_pool(stats, pd.Series({"C1": 2}), _actions())
    assert list(pool["customer_id"]) == ["C1"]
    assert pool["p_churn"].iloc[0] == 0.0


def test_la_ficha_trae_lo_que_el_selector_necesita() -> None:
    pool = _pool_basico()
    assert list(pool.columns) == list(cu.COLUMNAS)
    fila = pool.set_index("customer_id").loc["C1"]
    assert fila["n_baskets"] == 100
    assert fila["avg_ticket"] == 50.0
    assert fila["n_test"] == 5


# --------------------------------------------------------------------------------------
# Los escenarios
# --------------------------------------------------------------------------------------
def test_cada_escenario_recoge_a_quien_le_toca() -> None:
    pool = _pool_basico().set_index("customer_id")
    assert pool.loc["C1", "escenario"] == "Fiel"
    assert pool.loc["C2", "escenario"] == "Ocasional"
    assert pool.loc["C3", "escenario"] == "En riesgo"


def test_el_riesgo_gana_a_la_fidelidad() -> None:
    """Un cliente que compra mucho **y** se esta yendo es el caso mas interesante.

    La prioridad esta declarada en `clasificar`; sin ella, dependeria del orden en que se
    evaluan las reglas, que es una forma silenciosa de que cambie sola.
    """
    # Diez clientes para que los cuantiles tengan de donde separar: A y B son los dos que
    # mas compran, y solo se diferencian en el riesgo de fuga.
    stats = _stats(
        A=(100, 50.0, 300),
        B=(99, 50.0, 300),
        **{f"C{i}": (i, 20.0, 20) for i in range(1, 9)},
    )
    test = pd.Series({"A": 3, "B": 3, **{f"C{i}": 3 for i in range(1, 9)}})
    churn = _actions(A=0.95, B=0.02, **{f"C{i}": 0.02 for i in range(1, 9)})
    pool = cu.build_pool(stats, test, churn).set_index("customer_id")

    assert pool.loc["A", "escenario"] == "En riesgo"  # aunque sea el que mas compra
    assert pool.loc["B", "escenario"] == "Fiel"


def test_quien_no_encaja_en_ninguno_no_se_inventa_una_etiqueta() -> None:
    """La mayoria de la base esta en el medio; forzarla a un escenario seria mentir."""
    stats = _stats(**{f"C{i}": (i * 10, 30.0, 100) for i in range(1, 11)})
    test = pd.Series({f"C{i}": 2 for i in range(1, 11)})
    pool = cu.build_pool(stats, test, _actions(**{f"C{i}": 0.1 for i in range(1, 11)}))
    assert (pool["escenario"] == "").any()


def test_los_umbrales_salen_de_la_poblacion_y_no_estan_escritos_a_mano() -> None:
    """La misma forma de distribucion, multiplicada por 100, da la misma clasificacion.

    Si hubiera un numero fijo dentro (por ejemplo "mas de 50 cestas es fiel"), escalar la
    base lo romperia y este test lo caza.
    """
    base = {f"C{i}": (i, 30.0, 100) for i in range(1, 21)}
    escalado = {f"C{i}": (i * 100, 30.0, 100) for i in range(1, 21)}
    test = pd.Series({f"C{i}": 2 for i in range(1, 21)})
    churn = _actions(**{f"C{i}": 0.1 for i in range(1, 21)})

    uno = cu.build_pool(_stats(**base), test, churn).set_index("customer_id")["escenario"]
    dos = cu.build_pool(_stats(**escalado), test, churn).set_index("customer_id")[
        "escenario"
    ]
    pd.testing.assert_series_equal(uno, dos)


# --------------------------------------------------------------------------------------
# A quien se acaba ofreciendo
# --------------------------------------------------------------------------------------
def test_pick_filtra_por_escenario() -> None:
    pool = _pool_basico()
    assert cu.pick(pool, "Fiel") == ["C1"]
    assert cu.pick(pool, "Ocasional") == ["C2"]
    assert cu.pick(pool, "En riesgo") == ["C3"]


def test_todos_no_filtra() -> None:
    pool = _pool_basico()
    assert set(cu.pick(pool, cu.TODOS)) == {"C1", "C2", "C3"}


def test_la_seleccion_cubre_el_rango_y_no_la_cabeza() -> None:
    """El bug original: coger los `n` primeros de una lista ordenada por frecuencia.

    Con 100 candidatos y 5 huecos, tiene que salir gente de todo el rango, no los 5 mas
    frecuentes. Es lo que hacia que los 60 del selector fueran el mismo cliente.
    """
    stats = _stats(**{f"C{i:03d}": (i, 30.0, 100) for i in range(1, 101)})
    test = pd.Series({f"C{i:03d}": 2 for i in range(1, 101)})
    pool = cu.build_pool(stats, test, _actions(**{f"C{i:03d}": 0.1 for i in range(1, 101)}))

    elegidos = cu.pick(pool, cu.TODOS, n=5)
    cestas = pool.set_index("customer_id").loc[elegidos, "n_baskets"]
    assert len(elegidos) == 5
    # La cabeza pura daria 100, 99, 98, 97, 96. Repartido, el minimo baja mucho.
    assert cestas.min() <= 20
    assert cestas.max() >= 90


def test_pick_es_deterministico() -> None:
    pool = _pool_basico()
    assert cu.pick(pool, cu.TODOS) == cu.pick(pool, cu.TODOS)


def test_pick_no_se_rompe_si_el_escenario_esta_vacio() -> None:
    stats = _stats(C1=(10, 30.0, 50))
    pool = cu.build_pool(stats, pd.Series({"C1": 2}), _actions(C1=0.1))
    assert cu.pick(pool, "En riesgo") == [] or len(cu.pick(pool, "En riesgo")) <= 1


def test_pick_devuelve_como_mucho_n() -> None:
    stats = _stats(**{f"C{i:03d}": (i, 30.0, 100) for i in range(1, 101)})
    test = pd.Series({f"C{i:03d}": 2 for i in range(1, 101)})
    pool = cu.build_pool(stats, test, _actions(**{f"C{i:03d}": 0.1 for i in range(1, 101)}))
    assert len(cu.pick(pool, cu.TODOS, n=7)) == 7


# --------------------------------------------------------------------------------------
# Como se describe
# --------------------------------------------------------------------------------------
def test_la_etiqueta_dice_quien_es_y_si_hay_cesta_que_cargar() -> None:
    """`n_test` va en la etiqueta a proposito: decide si la mejor parte de la demo esta
    disponible, y antes solo se sabia despues de elegir."""
    fila = pd.Series(
        {"customer_id": "C1", "n_baskets": 52, "avg_ticket": 44.1, "n_test": 11}
    )
    texto = cu.label(fila)
    assert "C1" in texto
    assert "52 cestas" in texto
    assert "44 €" in texto
    assert "11 de test" in texto


@pytest.mark.parametrize(
    ("p_churn", "esperado"),
    [(0.01, "bajo"), (0.50, "medio"), (0.99, "alto")],
)
def test_el_riesgo_se_dice_en_relacion_a_la_base(p_churn: float, esperado: str) -> None:
    """Un 30 % no significa lo mismo si la mediana de la base es el 5 % que si es el 29 %."""
    pool = pd.DataFrame({"p_churn": [0.0, 0.1, 0.3, 0.5, 0.8, 0.95]})
    nivel, color = cu.riesgo_texto(p_churn, pool)
    assert nivel == esperado
    assert color in {"green", "orange", "red"}


def test_funciona_sin_la_tabla_del_nba() -> None:
    """La demo contempla que falte `predictions/nba_actions.parquet`.

    En ese caso llega un DataFrame vacio, sin columnas. Sin la guarda, el selector entero
    se caia por un `KeyError` antes de pintar nada.
    """
    stats = _stats(C1=(10, 30.0, 50), C2=(3, 20.0, 12))
    pool = cu.build_pool(stats, pd.Series({"C1": 2, "C2": 1}), pd.DataFrame())
    assert len(pool) == 2
    assert (pool["p_churn"] == 0.0).all()
    assert (pool["escenario"] != "En riesgo").all()
    assert cu.pick(pool, cu.TODOS)

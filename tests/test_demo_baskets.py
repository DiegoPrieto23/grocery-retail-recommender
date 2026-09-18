"""Carga de cestas reales en la demo (Fase 6b).

Lo que estos tests protegen es una propiedad concreta: que lo que la demo siembra sea **el
carrito en el instante del corte**, y que lo que ensena como "lo que compro despues" sea
**el target**, exactamente como los definio el split de la Fase 3. Si eso se rompe, la demo
seguiria pintando tarjetas pero estaria ensenando un escenario que nunca se evaluo, y el
contraste con el top-5 seria mentira.

Ojo con la trampa que estos tests ya cazaron una vez: el carrito **no** es el prefijo. Trae
ademas las lineas abandonadas (220 en 171 de las 18.000 cestas), asi que `len(cart)` puede
pasarse de `prefix_size` y `len(cart) + len(target)` puede pasarse de `n_items`.

Se saltan enteros si no esta el bundle de serving, igual que `test_serving_parity.py`, para
que la CI de un repo recien clonado (donde `data/` no se versiona) siga en verde.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.demo.baskets import (
    basket_label,
    build_basket,
    customer_baskets,
    hits,
    load_carts,
    load_queries,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARITY_DIR = PROJECT_ROOT / "data" / "serving" / "parity"
BASKET_ITEMS = PROJECT_ROOT / "data" / "processed" / "basket_items.parquet"

needs_bundle = pytest.mark.skipif(
    not (PARITY_DIR / "queries.parquet").is_file()
    or not (PARITY_DIR / "cart.parquet").is_file()
    or not BASKET_ITEMS.is_file(),
    reason=(
        "falta el bundle de serving o las tablas de la Fase 2; "
        "genera con `python -m src.serving.export_bundle`"
    ),
)

# Cuantas cestas se reconstruyen en los tests que recorren varias. Cada una lee su fila de
# `basket_items`, asi que unas pocas decenas bastan para cubrir los casos y siguen siendo
# rapidas.
N_MUESTRA = 40

pytestmark = needs_bundle


@pytest.fixture(scope="module")
def queries() -> pd.DataFrame:
    return load_queries()


@pytest.fixture(scope="module")
def carts() -> pd.DataFrame:
    return load_carts()


@pytest.fixture(scope="module")
def muestra(queries: pd.DataFrame) -> pd.DataFrame:
    """Cestas variadas y deterministas: las primeras por `basket_id`."""
    return queries.sort_values("basket_id").head(N_MUESTRA).reset_index(drop=True)


def test_el_carrito_contiene_el_prefijo_del_split(
    muestra: pd.DataFrame, carts: pd.DataFrame
) -> None:
    """El carrito sembrado tiene al menos `prefix_size` productos, y a veces mas.

    Los de mas son lineas abandonadas: estaban en el carrito en el corte y no llegaron al
    ticket. Son 220 en 171 de las 18.000 cestas. Se siembran igual, porque es lo que el
    ranker tenia delante cuando excluyo candidatos.
    """
    for i in range(len(muestra)):
        row = muestra.iloc[i]
        basket = build_basket(row, carts)
        assert len(basket.cart) >= int(row["prefix_size"]), basket.basket_id
        assert basket.n_bought_in_cart == int(row["prefix_size"]), basket.basket_id


def test_el_target_es_lo_que_falta_por_anadir(
    muestra: pd.DataFrame, carts: pd.DataFrame
) -> None:
    """Lo que se ensena como "compro despues" cuadra con el `n_target` del split."""
    for i in range(len(muestra)):
        row = muestra.iloc[i]
        basket = build_basket(row, carts)
        assert len(basket.target) == int(row["n_target"]), basket.basket_id


def test_carrito_y_target_no_se_solapan(muestra: pd.DataFrame, carts: pd.DataFrame) -> None:
    """Un producto no puede estar a la vez en el carrito y en lo que queda por adivinar.

    Es la condicion que hace honesto el contraste: si un producto ya sembrado apareciera
    en el target, contaria como acierto sin que el ranker hubiera acertado nada.
    """
    for i in range(len(muestra)):
        basket = build_basket(muestra.iloc[i], carts)
        assert not (set(basket.cart) & set(basket.target)), basket.basket_id


def test_lo_abandonado_no_esta_en_el_ticket(
    muestra: pd.DataFrame, carts: pd.DataFrame
) -> None:
    """Lo abandonado no cuenta como acierto ni como compra: no esta en el target."""
    for i in range(len(muestra)):
        basket = build_basket(muestra.iloc[i], carts)
        assert not (set(basket.abandoned) & set(basket.target)), basket.basket_id
        assert set(basket.abandoned) <= set(basket.cart), basket.basket_id


def test_las_lineas_compradas_suman_el_ticket(
    muestra: pd.DataFrame, carts: pd.DataFrame
) -> None:
    """Lo comprado del carrito mas el target son las lineas del ticket.

    Ojo con la version ingenua de esta suma: `len(cart) + len(target)` se pasa en las
    cestas con abandono, y por eso `n_items` viene de la query y no de sumar.
    """
    for i in range(len(muestra)):
        basket = build_basket(muestra.iloc[i], carts)
        assert basket.n_bought_in_cart + len(basket.target) == basket.n_items


def test_el_perfil_concuerda_con_tener_carrito(
    muestra: pd.DataFrame, carts: pd.DataFrame
) -> None:
    """`has_cart` y el perfil del split cuentan lo mismo.

    La app deriva el perfil de si el carrito tiene algo, asi que si estos dos se separaran
    el badge de la demo diria un perfil distinto del que se midio.
    """
    for i in range(len(muestra)):
        basket = build_basket(muestra.iloc[i], carts)
        con_carrito = basket.profile in (2, 4)
        assert basket.has_cart == con_carrito, basket.basket_id


def test_las_cestas_de_un_cliente_son_suyas_y_van_de_nueva_a_vieja(
    queries: pd.DataFrame,
) -> None:
    customer_id = queries["customer_id"].iloc[0]
    mine = customer_baskets(queries, customer_id)
    assert not mine.empty
    assert (mine["customer_id"] == customer_id).all()
    dias = mine["basket_day"].tolist()
    assert dias == sorted(dias, reverse=True)


def test_un_cliente_sin_cestas_devuelve_vacio_sin_reventar(queries: pd.DataFrame) -> None:
    """2 de los 60 clientes que ofrece el selector no tienen ninguna cesta de test."""
    mine = customer_baskets(queries, "NO_EXISTE")
    assert mine.empty


def test_la_etiqueta_distingue_carrito_vacio_de_carrito_con_cosas(
    queries: pd.DataFrame,
) -> None:
    vacia = queries.loc[queries["prefix_size"] == 0].iloc[0]
    llena = queries.loc[queries["prefix_size"] > 0].iloc[0]
    assert "carrito vacío" in basket_label(vacia)
    assert "en el carrito" in basket_label(llena)


def test_la_etiqueta_concuerda_en_singular(queries: pd.DataFrame) -> None:
    de_una = queries.loc[queries["n_items"] == 1]
    if de_una.empty:
        pytest.skip("no hay cestas de una sola linea en la muestra de test")
    assert "1 línea " in basket_label(de_una.iloc[0])


def test_los_aciertos_son_la_interseccion() -> None:
    assert hits(["A", "B", "C"], ["C", "D"]) == {"C"}
    assert hits(["A"], []) == set()
    assert hits([], ["A"]) == set()


# --------------------------------------------------------------------------------------
# Que cestas ofrece la demo
# --------------------------------------------------------------------------------------
def _query(basket_id: str, n_items: int, prefix_size: int) -> dict:
    return {
        "customer_id": "C1",
        "basket_id": basket_id,
        "basket_day": pd.Timestamp("2025-11-10").date(),
        "channel": "web",
        "n_items": n_items,
        "prefix_size": prefix_size,
        "profile": 4 if prefix_size else 3,
    }


def test_solo_se_ofrecen_cestas_con_contexto_y_con_algo_que_adivinar() -> None:
    """El split deja la mitad de las queries con el carrito vacio (punto M3).

    Eso es correcto para medir, pero al elegir una al azar en la demo salia carrito vacio
    el 52,6 % de las veces, y ese no es el caso que la demo quiere ensenar: sin nada en el
    carrito no hay contexto que contrastar.
    """
    from src.demo.baskets import demo_baskets

    queries = pd.DataFrame(
        [
            _query("vale", n_items=8, prefix_size=4),  # 4 dentro, 4 por adivinar
            _query("justo", n_items=4, prefix_size=2),  # 2 y 2: el limite
            _query("vacio", n_items=6, prefix_size=0),  # perfil 3
            _query("poco_carrito", n_items=6, prefix_size=1),
            _query("poco_target", n_items=4, prefix_size=3),
        ]
    )
    assert set(demo_baskets(queries)["basket_id"]) == {"vale", "justo"}


def test_el_filtro_no_toca_las_cestas_sobre_las_que_se_mide() -> None:
    """Decide que se **ofrece** en un desplegable, no sobre que se evalua el modelo."""
    from src.demo.baskets import demo_baskets

    queries = pd.DataFrame([_query("a", 8, 4), _query("b", 6, 0)])
    antes = queries.copy()
    demo_baskets(queries)
    pd.testing.assert_frame_equal(queries, antes)


@needs_bundle
def test_sobre_el_dataset_real_quedan_cestas_de_sobra() -> None:
    """Si el filtro dejase el selector casi vacio, el remedio seria peor."""
    from src.demo.baskets import demo_baskets, load_queries

    todas = load_queries()
    ofrecidas = demo_baskets(todas)
    assert len(ofrecidas) > 3_000, len(ofrecidas)
    assert ofrecidas["customer_id"].nunique() > 2_000
    assert (ofrecidas["prefix_size"] >= 2).all()
    assert ((ofrecidas["n_items"] - ofrecidas["prefix_size"]) >= 2).all()

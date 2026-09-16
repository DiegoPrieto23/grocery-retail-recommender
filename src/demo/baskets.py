"""Cestas reales de la ventana de test, para sembrar el carrito de la demo.

Por que el corte y no la cesta entera
-------------------------------------
`src/recommender/splits.py` parte cada cesta de test en dos:

    posiciones 1..prefix_size  ->  `prefix`, lo que el cliente ya llevaba en el corte
    posiciones prefix_size+1.. ->  `target`, lo que le faltaba por anadir

El `target` es justo lo que el sistema tiene que adivinar, y es contra lo que se midio el
NDCG@5 de la Fase 3. Sembrar el carrito con la cesta **completa** dejaria al recomendador
sin nada que predecir y ensenaria un escenario que nunca se evaluo. Asi que se carga el
carrito tal y como estaba en el corte, y el target se guarda aparte para comparar
despues.

De donde salen los datos
------------------------
De `data/serving/parity/`, que es el mismo material con el que `tests/test_serving_parity.py`
comprueba que la inferencia sin Spark da identico resultado que el pipeline offline:

- `queries.parquet` -> una fila por cesta de test (18.000), con `basket_day`, `channel`,
  `prefix_size`, `n_target` y `profile` ya resueltos.
- `cart.parquet` -> lo que habia en el carrito en el instante del corte.

**No se recalcula el corte aqui.** `prefix_size` sale de un hash estable del `basket_id` y
el orden de las lineas depende de `add_ts` cuando la cesta tiene sesion detras; rehacer eso
en pandas se arriesgaria a divergir en silencio del split que se evaluo.

El carrito no es el prefijo: es el prefijo **mas lo abandonado**
----------------------------------------------------------------
`cart.parquet` tiene 20.748 filas y la suma de `prefix_size` es 20.528. La diferencia son
**220 lineas abandonadas** repartidas en 171 cestas (0,95 %): productos que el cliente
anadio al carrito y luego quito, asi que nunca llegaron al ticket. El generador los produce
a proposito desde que se arreglo la contaminacion de la señal de sesion (Fase 3), y el
recomendador los excluye del pool de candidatos porque en ese instante estaban en el
carrito.

Para la demo eso es lo correcto: se siembra el carrito con `cart.parquet`, que es
exactamente lo que el ranker tenia delante. Pero obliga a no confundir tres cosas:

    carrito  = lo que habia en el corte      (`cart`, lo que se siembra)
    ticket   = lo que acabo comprando        (`basket_items`)
    target   = ticket - carrito              (`target`, lo que hay que adivinar)
    abandono = carrito - ticket              (`abandoned`, ni se compro ni se adivina)

Por eso `len(cart) + len(target)` **no** tiene por que ser el numero de lineas de la cesta:
el numero bueno es el `n_items` que trae la propia query.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARITY_DIR = PROJECT_ROOT / "data" / "serving" / "parity"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Canales que acepta el selector de la app. El orden es el de `streamlit_app.py`.
CHANNELS: tuple[str, ...] = ("app", "web", "store")


@dataclass(frozen=True)
class RealBasket:
    """Una cesta real de test, separada en lo que se siembra y lo que se adivina."""

    basket_id: str
    customer_id: str
    basket_day: date
    channel: str
    profile: int
    n_items: int
    """Lineas del ticket. Viene de la query, no se deduce sumando (ver docstring)."""
    cart: list[str]
    """Lo que habia en el carrito en el corte. Es lo que se siembra en la demo."""
    target: list[str]
    """Lo que anadio despues y acabo en el ticket. Es lo que hay que acertar."""
    abandoned: list[str]
    """Lo que estaba en el carrito y no llego al ticket. Ni se compro ni se adivina."""

    @property
    def has_cart(self) -> bool:
        """`False` en las cestas de perfil 3: el corte cayo en 0 y el carrito va vacio.

        No es un fallo de carga, es la mitad del reparto que hace `splits.py`, y la app
        tiene que decirlo en vez de aparentar que la cesta no tenia nada.
        """
        return bool(self.cart)

    @property
    def n_bought_in_cart(self) -> int:
        """Cuantas de las del carrito acabaron en el ticket (el `prefix_size` del split)."""
        return len(self.cart) - len(self.abandoned)


def _missing(path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"No encuentro {path.relative_to(PROJECT_ROOT)}. Ejecuta antes "
        "`python -m src.serving.export_bundle` (necesita las salidas de la Fase 3)."
    )


def load_queries(parity_dir: Path | None = None) -> pd.DataFrame:
    """Las cestas de test, una fila por cesta.

    Se queda con las columnas que la demo usa y normaliza `basket_day` a `date`, que es
    lo que espera el `st.date_input` de la app (el parquet lo trae como texto).
    """
    directory = parity_dir or PARITY_DIR
    path = directory / "queries.parquet"
    if not path.is_file():
        raise _missing(path)
    queries = pd.read_parquet(
        path,
        columns=[
            "customer_id",
            "basket_id",
            "basket_day",
            "channel",
            "n_items",
            "prefix_size",
            "n_target",
            "profile",
        ],
    )
    queries["basket_day"] = pd.to_datetime(queries["basket_day"]).dt.date
    return queries


def load_carts(parity_dir: Path | None = None) -> pd.DataFrame:
    """Lo que habia en el carrito en el corte de cada cesta de test."""
    directory = parity_dir or PARITY_DIR
    path = directory / "cart.parquet"
    if not path.is_file():
        raise _missing(path)
    return pd.read_parquet(path, columns=["basket_id", "product_id"])


def customer_baskets(queries: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    """Cestas de test de un cliente, de la mas reciente a la mas antigua.

    Devuelve un frame vacio si el cliente no tiene ninguna, que le pasa a 2 de los 60
    clientes que ofrece el selector: tienen mucho historial pero ninguna cesta cayo en la
    muestra de 18.000 queries de test.
    """
    mine = queries.loc[queries["customer_id"] == customer_id]
    return mine.sort_values("basket_day", ascending=False).reset_index(drop=True)


def basket_label(row: pd.Series) -> str:
    """Texto de una cesta en el selector: fecha, canal y de que se compone.

    Se dice `prefix_size` de `n_items` y no solo el total, porque es lo que explica que
    una cesta de 7 lineas siembre el carrito con 3.
    """
    day: date = row["basket_day"]
    prefix = int(row["prefix_size"])
    total = int(row["n_items"])
    carrito = f"{prefix} en el carrito" if prefix else "carrito vacío"
    lineas = "línea" if total == 1 else "líneas"
    return (
        f"{day:%d/%m/%Y} · {row['channel']} · {total} {lineas} "
        f"({carrito}) · perfil {int(row['profile'])}"
    )


def basket_items(
    basket_id: str, processed_dir: Path | None = None, engine: str | None = None
) -> pd.DataFrame:
    """Las lineas reales de una cesta, leyendo solo esa cesta del parquet.

    `basket_items` son millones de filas y la demo necesita cinco. El filtro va en la
    lectura (`filters=`, que pyarrow empuja hasta el row group) y no en un `DataFrame`
    ya cargado en memoria.
    """
    directory = processed_dir or PROCESSED_DIR
    path = directory / "basket_items.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"No encuentro {path.relative_to(PROJECT_ROOT)}. Ejecuta antes la Fase 2 "
            "(`python -m src.etl.run_etl`)."
        )
    return pd.read_parquet(
        path,
        columns=["basket_id", "product_id", "quantity"],
        filters=[("basket_id", "==", basket_id)],
        engine=engine or "auto",
    )


def build_basket(
    row: pd.Series,
    carts: pd.DataFrame,
    processed_dir: Path | None = None,
) -> RealBasket:
    """Monta la cesta real de una fila de `queries`: carrito, target y abandono.

    Las dos restas se hacen recorriendo la lista original y no operando con `set`, para que
    dos ejecuciones de la demo ensenen los productos en el mismo orden.
    """
    basket_id = str(row["basket_id"])
    cart = carts.loc[carts["basket_id"] == basket_id, "product_id"].tolist()
    en_carrito = set(cart)
    lineas = basket_items(basket_id, processed_dir)["product_id"].tolist()
    en_ticket = set(lineas)
    return RealBasket(
        basket_id=basket_id,
        customer_id=str(row["customer_id"]),
        basket_day=row["basket_day"],
        channel=str(row["channel"]),
        profile=int(row["profile"]),
        n_items=int(row["n_items"]),
        cart=cart,
        target=[p for p in lineas if p not in en_carrito],
        abandoned=[p for p in cart if p not in en_ticket],
    )


def hits(recommended: list[str], target: list[str]) -> set[str]:
    """Los productos recomendados que el cliente acabo comprando de verdad."""
    return set(recommended) & set(target)

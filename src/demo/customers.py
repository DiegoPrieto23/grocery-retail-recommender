"""A quien ofrece la demo como cliente, y como se describe.

El selector de la demo pedia elegir un cliente por su `customer_id`, de una lista de los 60
con mas historial. Dos problemas, y el segundo es el grave:

1. `C007693` no le dice nada a nadie.
2. **Los 60 eran casi el mismo cliente.** Ordenar por frecuencia descendente sobre 18.729
   clientes devuelve la cola extrema: todos con mas de 400 cestas, todos con `p_churn` del
   6 %, y a 46 de los 60 les tocaba la misma accion del NBA. La demo presume de que las
   recomendaciones se adaptan a quien compra, y luego ofrecia 60 clientes intercambiables.
   Ademas, 2 de esos 60 no tenian ninguna cesta en la ventana de test, asi que al elegirlos
   la mitad de la demo no funcionaba y no habia forma de saberlo antes de hacer clic.

Medido sobre los 8.811 clientes con alguna cesta de test, que es la poblacion util:

| | Los 60 de antes | Poblacion con cesta de test |
| --- | --- | --- |
| Cestas historicas | todos 400+ | p10 7 · mediana 30 · p90 85 |
| Ticket medio | 24-71 EUR | p10 20 · mediana 36 · p90 62 |
| `p_churn` | casi todos 6 % | p10 8 % · mediana 29 % · p90 56 % |

## De donde sale cada dato, y por que importa

Todo lo que se ensena aqui es **point-in-time correcto** respecto al 1 de noviembre:

- frecuencia, ticket y referencias salen de `customer_stats` del bundle, ajustado con lo
  anterior al inicio de la ventana servida;
- `p_churn` sale de `predictions/nba_actions.parquet`, resuelto en el corte del 1-nov.

**No se usan los segmentos de `data/processed/rfm.parquet`**, que serian mucho mas legibles
("Campeones", "En riesgo"), porque se calculan sobre el dataset entero, hasta el 31 de
diciembre. Etiquetar a un cliente "en riesgo" con datos de diciembre mientras la demo
simula una decision del 1 de noviembre es el mismo anacronismo que el punto M7 quito del
banner del NBA. No tiene sentido arreglarlo alli y reintroducirlo aqui.

## Los umbrales no estan escritos a mano

Cada escenario se define por cuantiles de la propia poblacion elegible, no por numeros
fijos. Si el generador cambia y la base se mueve, los escenarios se mueven con ella en vez
de quedarse describiendo un dataset que ya no existe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Cuantos clientes se ofrecen por escenario. Suficientes para que haya donde elegir, pocos
# para que el desplegable siga siendo un desplegable y no una lista infinita.
N_POR_ESCENARIO = 20

TODOS = "Todos"

# Cuantil por encima del cual se considera que un cliente compra mucho, y por debajo del
# cual compra poco.
Q_FRECUENCIA_ALTA = 0.75
Q_FRECUENCIA_BAJA = 0.25
# Y por encima del cual su riesgo de fuga destaca sobre el resto de la base.
Q_CHURN_ALTO = 0.75

COLUMNAS: tuple[str, ...] = (
    "customer_id",
    "n_baskets",
    "avg_ticket",
    "n_products",
    "n_test",
    "p_churn",
    "escenario",
)


@dataclass(frozen=True)
class Escenario:
    """Un caso de uso que la demo sabe ensenar."""

    nombre: str
    ayuda: str


ESCENARIOS: tuple[Escenario, ...] = (
    Escenario("Fiel", "Compra mucho y no da senales de irse: el historial manda."),
    Escenario("Ocasional", "Compra poco, asi que hay menos historial del que tirar."),
    Escenario("En riesgo", "Su probabilidad de dejar de comprar esta entre las mas altas."),
    Escenario(TODOS, "Una muestra que cubre todo el rango, sin filtrar."),
)

NOMBRES: tuple[str, ...] = tuple(e.nombre for e in ESCENARIOS)


def clasificar(pool: pd.DataFrame) -> pd.Series:
    """Asigna a cada cliente **un** escenario, por prioridad.

    Un cliente puede cumplir dos reglas a la vez -- comprar mucho y estar en riesgo es
    justo el caso mas interesante para el negocio --, asi que el orden importa y se declara
    aqui en vez de dejarlo al azar del orden de las columnas:

    1. **En riesgo** gana siempre. Es el escenario que mueve la politica del NBA, y un
       cliente fiel que ademas se esta yendo es mejor ejemplo que uno cualquiera.
    2. **Fiel** exige las dos cosas: comprar mucho y *no* estar en riesgo.
    3. **Ocasional** es el extremo contrario de la frecuencia.

    Quien no cae en ninguno se queda sin escenario y solo aparece bajo "Todos". No es un
    descarte: es la mayoria de la base, y forzarla a una etiqueta seria inventarse una.

    ## Una regla no se aplica si no separa a nadie

    Un cuantil sobre una distribucion plana no distingue: si todos los clientes tienen el
    mismo `p_churn`, el percentil 75 vale lo mismo que el minimo y `p_churn >= umbral` es
    cierto para **todos**, de modo que la base entera quedaria etiquetada "En riesgo" --
    incluido alguien con un 0 % de probabilidad de fuga. Lo mismo con la frecuencia.

    Por eso cada regla exige antes que sus cuantiles se separen de verdad. Con los datos
    reales siempre lo hacen; el caso degenerado aparece con muestras pequenas, que es justo
    lo que corre el *smoke* de CI con `--scale 0.02`.
    """
    alta = pool["n_baskets"].quantile(Q_FRECUENCIA_ALTA)
    baja = pool["n_baskets"].quantile(Q_FRECUENCIA_BAJA)
    riesgo = pool["p_churn"].quantile(Q_CHURN_ALTO)

    # Cada regla exige dos cosas: llegar al umbral **y** dejar a alguien por debajo. Lo
    # segundo es lo que neutraliza el empate: si media base esta clavada en el mismo valor,
    # ese valor es a la vez el percentil y el minimo, y sin la segunda condicion todos
    # entrarian. Con un solo cliente distinto del resto, en cambio, el percentil se queda
    # abajo y es la comparacion estricta la que lo rescata.
    churn = pool["p_churn"]
    cestas = pool["n_baskets"]

    en_riesgo = (churn >= riesgo) & (churn > churn.min())
    compra_mucho = (cestas >= alta) & (cestas > cestas.min())
    compra_poco = (cestas <= baja) & (cestas < cestas.max())

    escenario = pd.Series("", index=pool.index, dtype="object")
    escenario[compra_poco] = "Ocasional"
    escenario[compra_mucho & ~en_riesgo] = "Fiel"
    # El riesgo se aplica el ultimo porque gana a los anteriores.
    escenario[en_riesgo] = "En riesgo"
    return escenario


def build_pool(
    stats: pd.DataFrame,
    test_baskets: pd.Series,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Los clientes que la demo puede ofrecer, con su ficha y su escenario.

    Args:
        stats: `customer_stats` del bundle (`cust_frequency`, `cust_avg_ticket`,
            `cust_n_products`), ajustado con el pasado de la ventana servida.
        test_baskets: Cestas de test por cliente, indexado por `customer_id`.
        actions: `nba_actions`, con `p_churn` en el corte del NBA.

    Returns:
        Una fila por cliente **elegible**, con las columnas de `COLUMNAS`.

    Solo entran los clientes con al menos una cesta de test. Es la condicion que hace que
    las dos mitades de la demo funcionen: sin cesta de test no se puede cargar una compra
    real ni contrastar el top-5 contra lo que el cliente compro.
    """
    pool = stats.set_index("customer_id") if "customer_id" in stats.columns else stats
    pool = pool.rename(
        columns={
            "cust_frequency": "n_baskets",
            "cust_avg_ticket": "avg_ticket",
            "cust_n_products": "n_products",
        }
    )[["n_baskets", "avg_ticket", "n_products"]]

    pool = pool.join(test_baskets.rename("n_test"), how="inner")
    pool = pool[pool["n_test"] > 0]

    # La demo funciona sin la tabla del NBA (lo contempla `streamlit_app.get_nba`), y en
    # ese caso llega un DataFrame vacio, sin columnas. Sin esta guarda el selector entero
    # se caia por un `KeyError` en cuanto faltaba `predictions/nba_actions.parquet`.
    if "p_churn" in actions.columns:
        churn = (
            actions.set_index("customer_id")
            if "customer_id" in actions.columns
            else actions
        )
        pool = pool.join(churn["p_churn"], how="left")
    else:
        pool["p_churn"] = np.nan
    # Un cliente sin fila en el NBA no puede clasificarse por riesgo; se le deja fuera de
    # "En riesgo" con un 0, que es lo que la politica asume para quien no tiene candidatos.
    pool["p_churn"] = pool["p_churn"].fillna(0.0)

    pool["escenario"] = clasificar(pool)
    return pool.reset_index().rename(columns={"index": "customer_id"})[list(COLUMNAS)]


def _reparto(n_disponibles: int, n: int) -> np.ndarray:
    """Indices repartidos por todo el rango, no los `n` primeros.

    Tomar la cabeza de una lista ordenada por frecuencia es lo que dejaba 60 clientes
    identicos. Repartir por el rango da variedad dentro del propio escenario: el "Fiel" mas
    fiel y el que apenas pasa el corte son casos distintos y los dos merecen estar.
    """
    if n_disponibles <= n:
        return np.arange(n_disponibles)
    return np.linspace(0, n_disponibles - 1, n).round().astype(int)


def pick(pool: pd.DataFrame, escenario: str, *, n: int = N_POR_ESCENARIO) -> list[str]:
    """Los `customer_id` que se ofrecen para un escenario.

    Deterministico: ordena por frecuencia y reparte por el rango. Dos ejecuciones con el
    mismo bundle dan la misma lista, que es lo que `CLAUDE.md` pide del proyecto entero.
    """
    sub = pool if escenario == TODOS else pool[pool["escenario"] == escenario]
    if sub.empty:
        return []
    sub = sub.sort_values(["n_baskets", "customer_id"], ascending=[False, True])
    return sub["customer_id"].iloc[_reparto(len(sub), n)].tolist()


def label(fila: pd.Series) -> str:
    """La etiqueta del desplegable: quien es, en una linea.

    Lleva `n_test` a proposito. Es lo que decide si la parte mas interesante de la demo
    --cargar una compra real y ver si el modelo la acerto-- va a estar disponible, y antes
    solo se descubria despues de elegir.
    """
    ticket = f"{fila['avg_ticket']:.0f}".replace(".", ",")
    return (
        f"{fila['customer_id']} · {int(fila['n_baskets'])} cestas · "
        f"{ticket} € · {int(fila['n_test'])} de test"
    )


def riesgo_texto(p_churn: float, pool: pd.DataFrame) -> tuple[str, str]:
    """Riesgo de fuga en palabras y el color con que pintarlo.

    Se dice en relacion a la propia base y no con umbrales absolutos: un 30 % de
    probabilidad de fuga no significa lo mismo en un negocio donde la mediana es el 5 % que
    en uno donde es el 29 %, que es el caso de este dataset.
    """
    alto = pool["p_churn"].quantile(Q_CHURN_ALTO)
    medio = pool["p_churn"].median()
    if p_churn >= alto:
        return "alto", "red"
    if p_churn >= medio:
        return "medio", "orange"
    return "bajo", "green"

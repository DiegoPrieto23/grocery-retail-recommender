"""Demo de la Fase 6b: cesta en curso, recomendaciones en vivo y Next Best Action.

    .venv/Scripts/streamlit run streamlit_app.py     # Windows
    .venv/bin/streamlit run streamlit_app.py         # macOS / Linux

Solo hace inferencia: lee los modelos de `models/`, el bundle de `data/serving/` y las
fotos de `assets/`. No reentrena nada ni vuelve a llamar a Pexels.

Este fichero dibuja; la logica vive en `src/serving/` (recomendador) y `src/demo/`
(catalogo, buscador y motivos), que es lo que `CLAUDE.md` pide y lo que permite testear el
porque de una recomendacion sin levantar la app.

El orden del script no es casual: titulo y barra lateral primero, carga del bundle
despues. Son ~3,8 M de filas y unos segundos de espera la primera vez; si la carga fuera
lo primero, quien abre la demo miraria una pagina en blanco sin saber si arranco.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from src.demo import customers
from src.demo.baskets import (
    RealBasket,
    basket_label,
    build_basket,
    customer_baskets,
    load_carts,
    load_queries,
    load_reference_hit_rates,
    score_hits,
)
from src.demo.catalog import (
    Product,
    action_product,
    browse,
    cart_total,
    describe_action,
    explain,
    get_products,
    grid_columns,
    load_catalog,
    promo_badge,
    search,
)
from src.serving.recommend import load_bundle, recommend

PROJECT_ROOT = Path(__file__).resolve().parent
NBA_ACTIONS = PROJECT_ROOT / "predictions" / "nba_actions.parquet"

# Departamento que se abre al entrar: es el que mas categorias tiene despues de Despensa
# y el mas reconocible de un supermercado, asi que el escaparate inicial sale variado.
DEFAULT_DEPARTMENT = "Frescos"

# Ultimo dia con datos del generador. Mas alla no hay promociones vigentes ni popularidad
# reciente que mover, asi que el selector de fecha no deja salir de aqui.
DATASET_END = dt.date(2025, 12, 31)

# Canal de una cesta construida a mano. No hay selector para cambiarlo, y no es un olvido.
#
# `channel` es una feature legitima del ranker (`channel_idx`) y sigue viajando en la
# query, pero en la demo no puede significar lo que parece: el canal solo tiene efecto de
# verdad a traves del embudo de sesion, y una cesta inventada no tiene sesion, asi que esas
# features van nulas se elija lo que se elija. Medido sobre 80 combinaciones de cliente y
# carrito, el top-5 era identico en los tres canales en 69; de las 11 restantes, 6 eran el
# mismo top-5 reordenado. `channel_idx` es la feature 43 de 60 por ganancia (0,07 %), y las
# diferencias no siguen ningun patron de negocio -- `app` y `web` se separan tan a menudo
# como `store` de `app`.
#
# Un control que casi nunca cambia nada ensena algo falso: quien lo mueva y no vea reaccion
# concluira que el canal no importa, cuando lo que pasa es que aqui no puede importar. Se
# quita el control, no la feature.
DEFAULT_CHANNEL = "app"

# Cuantas recomendaciones se pintan. Es el `k` de todas las metricas del proyecto.
TOP_K = 5

# Lado de la miniatura del carrito, en pixeles. Tiene que dejar reconocer el producto sin
# forzar la vista; a 56 px se veian demasiado pequenas al lado de las recomendaciones.
# Siguen siendo bastante menores que las fotos del top-5, que es lo que se quiere: lo que ya
# esta en el carrito se identifica, lo que hay que decidir se mira.
THUMBNAIL_PX = 80

# Y cuantas se calculan: las que sobran no se ensenan, solo sirven para localizar la
# categoria objetivo del NBA dentro del ranking (`describe_action`). El re-ranking deja
# como mucho una referencia por categoria, asi que 40 puestos cubren 40 categorias.
NBA_LOOKUP_K = 40

st.set_page_config(
    page_title="Supermercado — recomendador y NBA",
    page_icon=":material/shopping_cart:",
    layout="wide",
)

# La **unica** regla de CSS de la demo, y va con explicacion porque el criterio del proyecto
# (y el de la guia oficial de Streamlit) es no inyectar CSS: el tema de
# `.streamlit/config.toml` se aplica a todos los elementos y sobrevive a una actualizacion,
# mientras que un selector apunta a nombres internos que pueden cambiar.
#
# La excepcion aqui es que el hueco por encima del titulo no es un token del tema: Streamlit
# reserva unos 6 rem para su barra de herramientas, que en esta demo esta vacia, y no hay
# ninguna opcion de configuracion que lo toque. Se apunta a `data-testid`, que es mas
# estable que una clase generada, y se cambia una sola propiedad.
st.html(
    "<style>[data-testid='stMainBlockContainer']{padding-top:2.5rem;}</style>",
)


# --------------------------------------------------------------------------------------
# Carga (cacheada: el bundle son ~3,8 M de filas y tarda unos segundos)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el recomendador…")
def get_bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)
def get_catalog() -> pd.DataFrame:
    return load_catalog()


@st.cache_data(show_spinner=False)
def get_nba() -> pd.DataFrame:
    if not NBA_ACTIONS.is_file():
        return pd.DataFrame()
    return pd.read_parquet(NBA_ACTIONS).set_index("customer_id")


@st.cache_data(show_spinner=False)
def get_queries() -> pd.DataFrame:
    """Las 18.000 cestas reales de la ventana de test."""
    return load_queries()


@st.cache_data(show_spinner=False)
def get_carts() -> pd.DataFrame:
    """Lo que habia en el carrito en el corte de cada una de esas cestas."""
    return load_carts()


@st.cache_data(show_spinner=False)
def get_reference_hit_rates() -> dict[str, float] | None:
    """Acierto medio en test (Fase 7c), leido del informe y no copiado a mano."""
    return load_reference_hit_rates()


@st.cache_data(show_spinner=False)
def get_customer_pool() -> pd.DataFrame:
    """Los clientes que el selector puede ofrecer, con su ficha y su escenario.

    La logica vive en `src/demo/customers.py`; aqui solo se cachea, porque recorre los
    18.729 clientes y las 18.000 cestas de test.
    """
    return customers.build_pool(
        get_bundle().customer_stats,
        get_queries().groupby("customer_id")["basket_id"].nunique(),
        pd.read_parquet(NBA_ACTIONS) if NBA_ACTIONS.is_file() else pd.DataFrame(),
    )


# --------------------------------------------------------------------------------------
# Estado
# --------------------------------------------------------------------------------------
if "cart" not in st.session_state:
    st.session_state.cart = []

# La cesta real cargada, si la hay. Se guarda entera (no solo lo que se siembra) porque
# su `target` es lo que permite contrastar el top-5 con lo que el cliente compro.
if "loaded_basket" not in st.session_state:
    st.session_state.loaded_basket = None


def add_to_cart(product_id: str) -> None:
    if product_id not in st.session_state.cart:
        st.session_state.cart.append(product_id)


def remove_from_cart(product_id: str) -> None:
    st.session_state.cart = [p for p in st.session_state.cart if p != product_id]


def clear_cart() -> None:
    st.session_state.cart = []
    st.session_state.loaded_basket = None


def load_real_basket(basket: RealBasket) -> None:
    """Siembra el carrito con una cesta real y alinea su contexto.

    Se llama desde un `on_click`, que corre **antes** del rerun: por eso aqui si se puede
    escribir sobre la clave de `date_input`. Alinear el dia no es cosmetico — mueve la
    estacionalidad y las promociones vigentes, asi que sin el el recomendador veria un
    contexto que nunca existio.

    El canal ya no tiene selector (ver `DEFAULT_CHANNEL`), pero se sigue alineando: en una
    cesta real es el unico caso en que el valor significa algo, porque es el que esa compra
    tuvo de verdad.
    """
    st.session_state.cart = list(basket.cart)
    st.session_state.loaded_basket = basket
    st.session_state.basket_day = basket.basket_day
    st.session_state.channel = basket.channel


# --------------------------------------------------------------------------------------
# Piezas de interfaz
# --------------------------------------------------------------------------------------
def product_card(
    product: Product,
    *,
    key_prefix: str,
    badge: tuple[str, str] | None = None,
    promo: str | None = None,
    action: str | None = None,
) -> None:
    """Una tarjeta con la foto real del `visual_group`, nombre, categoria y precio.

    Dos `height="stretch"` anidados: el de fuera iguala la altura de todas las tarjetas
    de la fila (los nombres van de una a cuatro lineas y, sin esto, la rejilla queda
    dentada) y el de dentro se come el hueco sobrante, de modo que el boton queda pegado
    al fondo y todos los botones de la fila caen a la misma altura.
    """
    with st.container(border=True, height="stretch"):
        image = product.image
        if image:
            st.image(image, width="stretch")
        else:
            # Un grupo sin foto valida no rompe la tarjeta: se dice y punto.
            st.caption(":material/image_not_supported: sin foto")
        with st.container(height="stretch"):
            st.markdown(f"**{product.name}**")
            st.markdown(f"{product.price:.2f} €".replace(".", ","))
            st.caption(product.category)
            if badge:
                st.badge(badge[0], color=badge[1])
            if promo:
                st.badge(promo, icon=":material/sell:", color="orange")
        if action == "add":
            st.button(
                "Añadir",
                icon=":material/add_shopping_cart:",
                key=f"{key_prefix}_{product.product_id}",
                on_click=add_to_cart,
                args=(product.product_id,),
                width="stretch",
            )
        elif action == "remove":
            st.button(
                "Quitar",
                icon=":material/close:",
                key=f"{key_prefix}_{product.product_id}",
                on_click=remove_from_cart,
                args=(product.product_id,),
                width="stretch",
            )


def cart_list(products: list[Product]) -> None:
    """El carrito como lista compacta, no como rejilla de tarjetas.

    Una tarjeta con foto grande necesita ancho, y el carrito vive en una columna estrecha
    para que las recomendaciones se lleven el espacio. Intentar meter ahi la misma rejilla
    era lo que apelotonaba las dos cosas.

    Una lista con miniatura es ademas como se ensena un carrito en cualquier tienda: lo que
    importa de lo que ya has metido es que esta y cuanto cuesta, no volver a mirar la foto
    a tamano escaparate. Las fotos grandes se reservan para lo que hay que decidir, que son
    las recomendaciones y el catalogo.
    """
    for product in products:
        # Columnas y no `container(horizontal=True)`: con el contenedor horizontal, un
        # nombre largo empujaba el boton a la linea de abajo y las filas quedaban
        # desiguales, unas con el boton al lado y otras debajo. Con columnas, la miniatura,
        # el texto y el boton caen siempre en el mismo sitio.
        thumb, texto, quitar = st.columns([1, 3, 1], vertical_alignment="center")
        if product.image:
            thumb.image(product.image, width=THUMBNAIL_PX)
        texto.markdown(
            f"**{product.name}**  \n{product.price:.2f} €".replace(".", ",")
        )
        # Sin etiqueta, solo el icono: "Quitar" escrito se llevaba un tercio del ancho de
        # la columna, que es justo lo que le faltaba al nombre y a la miniatura. La `x` en
        # una fila de carrito se entiende sin leerla, y el `help` la nombra para quien
        # navegue con lector de pantalla.
        quitar.button(
            "",
            icon=":material/close:",
            key=f"cart_{product.product_id}",
            on_click=remove_from_cart,
            args=(product.product_id,),
            help=f"Quitar {product.name} del carrito",
        )


def product_grid(
    products: list[Product],
    *,
    key_prefix: str,
    columns: int = 5,
    badges: dict[str, tuple[str, str]] | None = None,
    promos: dict[str, str] | None = None,
    action: str | None = None,
) -> None:
    """Rejilla de tarjetas, con el ancho repartido entre las que de verdad hay.

    `columns` es el **maximo**, no una constante. Antes se creaban siempre 5 columnas
    aunque hubiera 2 productos, asi que las tarjetas salian al 20 % de ancho con el 60 %
    de la fila vacia. No era un caso raro: el **74,8 %** de las cestas reales que la demo
    puede cargar traen entre 1 y 4 lineas en el carrito.

    El suelo de `MIN_GRID_COLUMNS` evita el extremo contrario, una unica tarjeta ocupando
    el ancho entero con una foto enorme. Se calcula sobre el total y no por fila, para que
    todas las filas de una misma rejilla tengan tarjetas del mismo tamano.
    """
    badges = badges or {}
    promos = promos or {}
    if not products:
        return

    n_columns = grid_columns(len(products), maximum=columns)
    for start in range(0, len(products), n_columns):
        row = products[start : start + n_columns]
        for column, product in zip(st.columns(n_columns), row):
            with column:
                product_card(
                    product,
                    key_prefix=key_prefix,
                    badge=badges.get(product.product_id),
                    promo=promos.get(product.product_id),
                    action=action,
                )


def nba_banner(
    customer_id: str | None,
    recommendations: pd.DataFrame,
    category_of: dict[str, str],
) -> None:
    """Banner de la proxima mejor accion, destacado y con sus dos probabilidades.

    Recibe las recomendaciones ya calculadas para poder ensenar **que referencia** de la
    categoria objetivo propone el recomendador: la politica elige la categoria y el ranker
    la referencia (punto M7 de `docs/diagnostico-fase7.md`).
    """
    if customer_id is None:
        with st.container(border=True):
            st.markdown("### :material/person_add: Cliente nuevo")
            st.caption(
                "El Next Best Action decide sobre historial (propensión de compra y de "
                "abandono). Un cliente que aún no tiene ninguno no entra en la política: "
                "aquí no hay acción que decidir."
            )
        return

    nba = get_nba()
    if nba.empty or customer_id not in nba.index:
        with st.container(border=True):
            st.caption(":material/info: Sin acción calculada para este cliente.")
        return

    info = describe_action(nba.loc[customer_id])
    with st.container(border=True):
        head, value = st.columns([3, 2], vertical_alignment="center")
        with head:
            # `####` y no `###`: el banner es contexto de la cesta, no el titulo de la
            # pagina, y a tamano de h3 pesaba mas que el propio recomendador.
            st.markdown(f"#### {info['icon']} {info['title']}")
            if info["is_action"] and info["category"]:
                # La política elige la categoría; la referencia la pone el recomendador.
                product_id = action_product(recommendations, info["category"], category_of)
                if product_id is not None:
                    producto = get_products(catalog, [product_id])[0]
                    st.markdown(
                        f"**{info['category']}** · el recomendador propone "
                        f"{producto.name} ({producto.price:.2f} €)".replace(".", ",", 1)
                    )
                else:
                    st.markdown(
                        f"**{info['category']}** · el recomendador no coloca ninguna "
                        "referencia de esa categoría en esta cesta"
                    )
            elif not info["is_action"]:
                st.markdown(
                    "El valor esperado de actuar no compensa su coste: la política "
                    "prefiere no gastar el impacto."
                )

            # Las dos insignias en una fila: el valor de la accion y el corte con el que se
            # decidio. La fecha de corte iba antes en un parrafo de tres lineas al pie del
            # banner, que era lo que mas altura le costaba. Sigue estando --el punto M7
            # pide declararla, porque el NBA se resuelve una vez y se ensena junto a cestas
            # posteriores-- pero como insignia, con la explicacion en su `help`.
            with st.container(horizontal=True, gap="small"):
                st.badge(
                    f"valor esperado {info['expected_value']:.2f} €".replace(".", ","),
                    color=info["color"],
                )
                if info["cutoff"] is not None:
                    desfase = ""
                    if basket_day != info["cutoff"]:
                        dias = (basket_day - info["cutoff"]).days
                        desfase = (
                            f" La cesta que estás viendo es {dias:+d} días respecto a él."
                        )
                    st.badge(
                        f"corte {info['cutoff']:%d/%m/%Y}",
                        icon=":material/event:",
                        color="gray",
                        help=(
                            "La próxima mejor acción se decidió con el historial anterior "
                            "a esa fecha y no se recalcula al mover el día de la cesta: el "
                            "recomendador sí razona con el día seleccionado, esta acción "
                            "no." + desfase
                        ),
                    )
        with value:
            # En fila y no apiladas: apiladas, las dos tarjetas hacian el banner el doble
            # de alto que su propio contenido.
            compra, churn = st.columns(2)
            compra.metric(
                "Compra 7 d",
                f"{info['p_purchase']:.1%}".replace(".", ","),
                border=True,
                help="Probabilidad de que compre la categoría objetivo en 7 días.",
            )
            churn.metric(
                "Churn 4 sem",
                f"{info['p_churn']:.1%}".replace(".", ","),
                border=True,
                help="Probabilidad de que no vuelva a comprar en 4 semanas.",
            )


# --------------------------------------------------------------------------------------
# Cabecera (antes de cargar nada pesado: la pagina no se queda en blanco)
# --------------------------------------------------------------------------------------
st.title("Supermercado online")
# Los textos de la interfaz no mencionan fases ni tareas del proyecto: quien abre la demo
# no tiene por que saber que hubo una Fase 7c. Lo que si se dice es como funciona el
# sistema, que es informacion util, y eso se queda.
# Acotado: a pantalla completa la linea se iba a ~150 caracteres y se leia mal.
st.caption(
    "Recomendador de cesta en vivo y próxima mejor acción, sobre un catálogo de 496 "
    "referencias. Candidatos con ALS, co-compra y popularidad; el orden lo pone un "
    "LambdaRank. Solo inferencia: nada se reentrena aquí.",
    width=820,
)


# --------------------------------------------------------------------------------------
# Barra lateral: quien compra y cuando
# --------------------------------------------------------------------------------------
bundle = get_bundle()
catalog = get_catalog()

with st.sidebar:
    st.header("Cliente", icon=":material/person:")
    tipo = st.segmented_control(
        "Tipo de cliente",
        ["Recurrente", "Nuevo"],
        default="Recurrente",
        help="Un cliente nuevo no tiene historial: cambia qué fuentes tienen señal.",
    )

    customer_id: str | None = None
    if tipo == "Recurrente":
        pool = get_customer_pool()
        escenario = st.segmented_control(
            "Qué caso quieres ver",
            customers.NOMBRES,
            default=customers.TODOS,
            help="Filtra la lista por el tipo de cliente, para no elegir a ciegas un ID.",
        )
        candidatos = customers.pick(pool, escenario or customers.TODOS)
        fichas = pool.set_index("customer_id").loc[candidatos]

        customer_id = st.selectbox(
            "Cliente",
            candidatos,
            format_func=lambda cid: customers.label(
                pd.Series({**fichas.loc[cid].to_dict(), "customer_id": cid})
            ),
            help=(
                "Todos tienen alguna cesta en la ventana de test, así que siempre se "
                "puede cargar una compra real suya."
            ),
        )
        ficha = fichas.loc[customer_id]
        st.caption(
            f":material/receipt_long: {int(ficha['n_baskets'])} cestas · "
            f"{int(ficha['n_products'])} referencias distintas · "
            f"ticket medio {ficha['avg_ticket']:.2f} €".replace(".", ",")
        )
        nivel, color = customers.riesgo_texto(float(ficha["p_churn"]), pool)
        st.badge(
            f"riesgo de fuga {nivel} ({ficha['p_churn']:.0%})".replace(".", ","),
            color=color,
            icon=":material/warning:",
        )
        st.caption(
            f"Su historial hasta el {bundle.window_start:%d/%m/%Y}: cestas cerradas y "
            "referencias distintas compradas, no lo que lleva ahora en el carrito."
        )

        # --- Sembrar el carrito con una cesta real de test ---
        st.subheader("Cargar una cesta real", icon=":material/history:")
        suyas = customer_baskets(get_queries(), customer_id)
        if suyas.empty:
            st.caption(
                ":material/info: Este cliente no tiene ninguna cesta en la muestra de "
                "test, así que aquí solo se puede construir la cesta a mano."
            )
        else:
            elegida = st.selectbox(
                "Cesta de test",
                range(len(suyas)),
                format_func=lambda i: basket_label(suyas.iloc[i]),
                help=(
                    "Cestas suyas posteriores al corte. Se carga lo que ya llevaba en el "
                    "carrito; lo que añadió después es lo que el recomendador debe acertar."
                ),
            )
            real = build_basket(suyas.iloc[elegida], get_carts())
            st.button(
                "Cargar esta cesta",
                icon=":material/shopping_basket:",
                on_click=load_real_basket,
                args=(real,),
                width="stretch",
                type="primary",
            )
            if not real.has_cart:
                st.caption(
                    ":material/info: Esta cesta corta en 0: el carrito se queda vacío a "
                    "propósito (perfil 3). Es la mitad del reparto que hace el split, no "
                    "un fallo de carga."
                )
    else:
        st.caption(
            ":material/person_add: Sin historial: solo popularidad y co-compra tienen señal."
        )

    st.header("Contexto", icon=":material/tune:")
    # Las dos con `key`: al cargar una cesta real, `load_real_basket` escribe sobre ellas
    # para que el dia y el canal sean los que esa compra tuvo de verdad. El valor inicial
    # se siembra en el estado y **no** se pasa por `value=`/`default=`: hacer las dos
    # cosas a la vez es lo que dispara el aviso de Streamlit por valor duplicado.
    st.session_state.setdefault("basket_day", bundle.window_start)
    st.session_state.setdefault("channel", DEFAULT_CHANNEL)
    basket_day = st.date_input(
        "Día de la compra",
        min_value=bundle.window_start,
        max_value=DATASET_END,
        format="DD/MM/YYYY",
        help="Mueve la estacionalidad y las promociones vigentes.",
        key="basket_day",
    )
    # El canal no se elige: es una feature del ranker, pero en la demo no hay nada que
    # elegir de verdad (ver `DEFAULT_CHANNEL`). Cuando se carga una cesta real, toma el
    # canal que esa compra tuvo, que es el unico valor con significado.
    channel = st.session_state.channel

    if st.session_state.cart:
        st.button(
            "Vaciar cesta",
            icon=":material/delete_sweep:",
            on_click=clear_cart,
            width="stretch",
        )

    st.caption(
        f"Fuentes ajustadas con el historial anterior al {bundle.window_start:%d/%m/%Y}. "
        "La demo solo hace inferencia."
    )


# --------------------------------------------------------------------------------------
# Perfil activo y Next Best Action
# --------------------------------------------------------------------------------------
# Si se cambia de cliente, la cesta real del anterior deja de tener sentido: se
# atribuiria a quien no la compro y el contraste con el `target` seria falso. Se descarta
# solo en ese caso; una cesta construida a mano (sin `loaded_basket`) se respeta.
_loaded = st.session_state.loaded_basket
if _loaded is not None and _loaded.customer_id != customer_id:
    clear_cart()

cart_ids: list[str] = st.session_state.cart
is_known = customer_id is not None
profile = (3 if is_known else 1) + (1 if cart_ids else 0)
PROFILE_TEXT = {
    1: "Cliente nuevo, cesta vacía — solo popularidad y estacionalidad",
    2: "Cliente nuevo, con cesta — entra la co-compra",
    3: "Cliente recurrente, cesta vacía — historial y reposición",
    4: "Cliente recurrente, con cesta — las tres fuentes a la vez",
}

st.badge(f"Perfil {profile}: {PROFILE_TEXT[profile]}", icon=":material/account_circle:")

# Se calcula aqui, antes del banner, porque el banner del NBA ensena que referencia de su
# categoria objetivo propone el recomendador. Se piden mas de 5 a proposito: el top-5 que
# se pinta abajo es `head(5)` de esta misma lista -- `rerank.order_candidates` ordena sin
# mirar `k`, asi que ampliarla no cambia ni un puesto-- y los de mas abajo solo sirven para
# buscar la categoria del NBA, que rara vez esta entre los cinco primeros.
category_of: dict[str, str] = catalog.set_index("product_id")["category"].to_dict()
with st.spinner("Calculando…"):
    ranked = recommend(
        bundle,
        customer_id=customer_id,
        cart=cart_ids,
        basket_day=basket_day,
        channel=channel,
        top_k=NBA_LOOKUP_K,
    )

nba_banner(customer_id, ranked, category_of)


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Cesta y recomendaciones, lado a lado
# --------------------------------------------------------------------------------------
# La demo dice que las recomendaciones se recalculan con cada cambio del carrito, y
# apilarlas debajo lo desmentia: con una cesta real de 15 lineas, las recomendaciones
# caian fuera de pantalla justo cuando mas interesa ver como cambian. En dos columnas
# la relacion causa-efecto se ve sin desplazarse.
#
# El reparto es 1:2,2 y no 1:1 porque el carrito es contexto y las recomendaciones son
# el asunto: cinco tarjetas en una fila necesitan el ancho (con 1:2 los distintivos de
# las tarjetas ya se cortaban), y dos del carrito no.
loaded: RealBasket | None = st.session_state.loaded_basket

cesta_col, recom_col = st.columns([1, 2.2], gap="medium")

with cesta_col:
    st.subheader("Tu cesta", icon=":material/shopping_cart:")

    if loaded is not None:
        tocada = list(cart_ids) != list(loaded.cart)
        st.caption(
            f":material/history: Cesta real **{loaded.basket_id}** del "
            f"{loaded.basket_day:%d/%m/%Y} ({loaded.channel}) · se cargó lo que el cliente ya "
            f"llevaba en el carrito en el instante del corte "
            f"({loaded.n_bought_in_cart} de las {loaded.n_items} líneas del ticket)"
            + (" · **modificada a mano desde entonces**" if tocada else "")
        )
        if loaded.abandoned:
            # No es un detalle decorativo: son lineas que el ranker vio en el carrito y por
            # eso excluyo de los candidatos, pero que no cuentan como acierto.
            n = len(loaded.abandoned)
            frase = (
                "1 producto del carrito no llegó al ticket (lo abandonó). Se carga igual"
                if n == 1
                else f"{n} productos del carrito no llegaron al ticket (los abandonó). "
                "Se cargan igual"
            )
            st.caption(
                f":material/remove_shopping_cart: {frase}, porque es lo que el recomendador "
                "tenía delante, pero no cuentan como acierto."
            )

    if not cart_ids:
        st.caption(
            "La cesta está vacía. Busca o navega el catálogo de abajo y añade productos: "
            "las recomendaciones se recalculan con cada cambio."
        )
    else:
        st.caption(
            f"{len(cart_ids)} productos · "
            f"**{cart_total(catalog, cart_ids):.2f} €**".replace(".", ",")
        )
        cart_list(get_products(catalog, cart_ids))


    # --------------------------------------------------------------------------------------
    # Recomendaciones
    # --------------------------------------------------------------------------------------

with recom_col:
    st.subheader("Te recomendamos", icon=":material/auto_awesome:")
    recommendations = ranked.head(TOP_K)

    if recommendations.empty:
        st.caption("No hay candidatos para esta combinación.")
    else:
        top5 = recommendations["product_id"].tolist()
        badges = {row["product_id"]: explain(row) for _, row in recommendations.iterrows()}
        promos = {
            row["product_id"]: label
            for _, row in recommendations.iterrows()
            if (label := promo_badge(row))
        }

        # Sobre una cesta real se puede decir algo que en una inventada no: si el cliente
        # acabo comprando lo que se le recomendo. El acierto pisa al motivo en la insignia,
        # porque es el dato mas fuerte de la tarjeta. Se distinguen dos niveles: el producto
        # exacto y "otra referencia de la misma categoria", que tambien es una recomendacion
        # util en gran consumo y que el SKU exacto solo no deja ver.
        score = None
        if loaded is not None and loaded.target:
            score = score_hits(top5, loaded.target, category_of)
            for product_id in score.category_only:
                badges[product_id] = ("acertó categoría", "yellow")
            for product_id in score.exact:
                badges[product_id] = ("sí lo compró", "green")

        product_grid(
            get_products(catalog, top5),
            key_prefix="rec",
            badges=badges,
            promos=promos,
            action="add",
        )
        st.caption(
            "El motivo de cada tarjeta sale de las features con las que el ranker ordenó, "
            "no de una explicación escrita a posteriori."
        )

        if score is not None:
            st.markdown(score.describe())
            st.caption(
                f"El cliente añadió {len(loaded.target)} líneas después del corte, de "
                f"{len(score.target_categories)} categorías distintas. «Acertar la categoría» "
                "es recomendar un producto de una categoría que sí compró, aunque fuera otra "
                "referencia; es la misma definición que usa el informe del recomendador."
            )
            reference = get_reference_hit_rates()
            if reference:
                def es(value: float, fmt: str) -> str:
                    return format(value, fmt).replace(".", ",")

                st.caption(
                    ":material/query_stats: En las cestas de test del recomendador, de media "
                    f"{es(reference['cat_per_basket'], '.2f')} de 5 recomendaciones aciertan "
                    f"la categoría y {es(reference['sku_per_basket'], '.2f')} el producto "
                    f"exacto; el {es(reference['cat_hit_rate'], '.1%')} de las cestas tiene al "
                    f"menos un acierto de categoría y el {es(reference['sku_hit_rate'], '.1%')} "
                    "al menos uno exacto. Una cesta suelta puede quedar por encima o por debajo."
                )

            # En el target se marca lo mismo desde el otro lado: que linea se acerto tal cual
            # y cual solo por categoria (se recomendo otra referencia de la suya).
            recommended_categories = {category_of.get(p) for p in score.category} - {None}
            target_badges = {
                p: ("acertó categoría", "yellow")
                for p in loaded.target
                if category_of.get(p) in recommended_categories
            }
            target_badges.update({p: ("acertada", "green") for p in score.exact})
            with st.expander("Ver lo que compró realmente después del corte"):
                st.caption(
                    "Es el `target` del split: las líneas que el cliente añadió tras el "
                    "instante del corte. El ranker no las ha visto."
                )
                product_grid(
                    get_products(catalog, loaded.target),
                    key_prefix="target",
                    columns=5,
                    badges=target_badges,
                )



# --------------------------------------------------------------------------------------
# Catalogo: buscador y navegacion por departamento
# --------------------------------------------------------------------------------------
# Con una cesta real cargada, la app esta haciendo otra cosa: no se esta construyendo una
# compra, se esta revisando si el modelo acerto lo que ese cliente compro de verdad. Anadir
# productos a mano no ayuda a eso — rompe el contraste contra el ticket real, que es justo
# lo que hace interesante el modo. Asi que el catalogo se repliega, no se quita: sigue
# estando para quien quiera ver que pasa al tocar la cesta, pero deja de ocupar media
# pantalla pidiendo ser usado.
revisando_cesta_real = loaded is not None

if revisando_cesta_real:
    contenedor = st.expander(
        "Añadir productos a mano", icon=":material/add_shopping_cart:", expanded=False
    )
    contenedor.caption(
        "Estás revisando una cesta real. Si añades productos, el carrito deja de ser el "
        "que el cliente tenía en el corte y la comparación con lo que compró de verdad "
        "deja de valer."
    )
else:
    st.subheader("Catálogo", icon=":material/storefront:")
    contenedor = st.container()

with contenedor:
    query = st.text_input(
        "Buscar productos",
        placeholder="Buscar por nombre, categoría o marca…",
        icon=":material/search:",
        label_visibility="collapsed",
    )

    LIMIT = 15

    if query:
        # Buscar manda sobre navegar: el resultado sale de todo el catalogo, no del
        # departamento que estuviera abierto. Los filtros se esconden mientras tanto para que
        # no haya duda de sobre que se ha buscado.
        found = search(catalog, query, exclude=cart_ids, limit=LIMIT)
        st.caption(f"Resultados en todo el catálogo, del más barato al más caro (máx. {LIMIT}).")
    else:
        departments = sorted(catalog["department"].unique())
        department = st.segmented_control(
            "Departamento",
            departments,
            default=DEFAULT_DEPARTMENT,
            label_visibility="collapsed",
        )
        category = None
        if department:
            categories = sorted(
                catalog.loc[catalog["department"] == department, "category"].unique()
            )
            category = st.pills("Categoría", categories, label_visibility="collapsed")
        found = (
            browse(catalog, department, category, exclude=cart_ids, limit=LIMIT)
            if department
            else []
        )
        if department and not category:
            st.caption(
                "Una referencia por categoría. Elige una categoría para ver todas sus marcas."
            )

    if not found:
        st.caption("Ningún producto casa con esa búsqueda.")
    else:
        product_grid(get_products(catalog, found), key_prefix="cat", action="add")

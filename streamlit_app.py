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
    browse,
    cart_total,
    describe_action,
    explain,
    get_products,
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

st.set_page_config(
    page_title="Supermercado — recomendador y NBA",
    page_icon=":material/shopping_cart:",
    layout="wide",
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
def sample_customers(n: int = 60) -> list[str]:
    """Unos cuantos clientes con historial, para el selector.

    Se ordenan por numero de cestas: un cliente con mucho historial ensena mejor los
    perfiles 3 y 4, que son los que tienen senal personal.
    """
    stats = get_bundle().customer_stats.sort_values("cust_frequency", ascending=False)
    return stats["customer_id"].head(n).tolist()


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
    escribir sobre las claves de `date_input` y `segmented_control`. Alinear dia y canal
    no es cosmetico — mueven la estacionalidad, las promociones vigentes y la señal de
    sesion, asi que sin ellos el recomendador veria un contexto que nunca existio.
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


def product_grid(
    products: list[Product],
    *,
    key_prefix: str,
    columns: int = 5,
    badges: dict[str, tuple[str, str]] | None = None,
    promos: dict[str, str] | None = None,
    action: str | None = None,
) -> None:
    badges = badges or {}
    promos = promos or {}
    for start in range(0, len(products), columns):
        row = products[start : start + columns]
        for column, product in zip(st.columns(columns), row):
            with column:
                product_card(
                    product,
                    key_prefix=key_prefix,
                    badge=badges.get(product.product_id),
                    promo=promos.get(product.product_id),
                    action=action,
                )


def nba_banner(customer_id: str | None) -> None:
    """Banner de la proxima mejor accion, destacado y con sus dos probabilidades."""
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
        head, value = st.columns([3, 1], vertical_alignment="center")
        with head:
            st.markdown(f"### {info['icon']} {info['title']}")
            if info["is_action"] and info["category"]:
                st.markdown(f"Categoría objetivo: **{info['category']}**")
            elif not info["is_action"]:
                st.caption(
                    "El valor esperado de actuar no compensa su coste: la política "
                    "prefiere no gastar el impacto."
                )
            st.badge(
                f"valor esperado {info['expected_value']:.2f} €".replace(".", ","),
                color=info["color"],
            )
        with value:
            st.metric("Compra 7 d", f"{info['p_purchase']:.1%}".replace(".", ","))
            st.metric("Churn 4 sem", f"{info['p_churn']:.1%}".replace(".", ","))


# --------------------------------------------------------------------------------------
# Cabecera (antes de cargar nada pesado: la pagina no se queda en blanco)
# --------------------------------------------------------------------------------------
st.title("Supermercado online")
st.caption(
    "Cesta en curso, recomendaciones del sistema de dos etapas (ALS + co-compra + "
    "popularidad, reordenado con LambdaRank) y próxima mejor acción. Todo es inferencia "
    "sobre los modelos ya entrenados de las Fases 3 y 4, reentrenados en las 7c y 7d "
    "sobre el catálogo de 496 referencias."
)


# --------------------------------------------------------------------------------------
# Barra lateral: quien compra y cuando
# --------------------------------------------------------------------------------------
bundle = get_bundle()
catalog = get_catalog()

with st.sidebar:
    st.header("Cliente")
    tipo = st.segmented_control(
        "Tipo de cliente",
        ["Recurrente", "Nuevo"],
        default="Recurrente",
        help="Un cliente nuevo no tiene historial: cambia qué fuentes tienen señal.",
    )

    customer_id: str | None = None
    if tipo == "Recurrente":
        customer_id = st.selectbox(
            "Cliente",
            sample_customers(),
            help="Clientes con más historial, para que se note la señal personal.",
        )
        ficha = bundle.customer_stats.set_index("customer_id").loc[customer_id]
        st.caption(
            f":material/receipt_long: {int(ficha['cust_frequency'])} cestas · "
            f"{int(ficha['cust_n_products'])} referencias distintas · "
            f"ticket medio {ficha['cust_avg_ticket']:.2f} €".replace(".", ",")
        )
        st.caption(
            f"Su historial hasta el {bundle.window_start:%d/%m/%Y}: cestas cerradas y "
            "referencias distintas compradas, no lo que lleva ahora en el carrito."
        )

        # --- Sembrar el carrito con una cesta real de test ---
        st.subheader("Cargar una cesta real")
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

    st.header("Contexto")
    # Las dos con `key`: al cargar una cesta real, `load_real_basket` escribe sobre ellas
    # para que el dia y el canal sean los que esa compra tuvo de verdad. El valor inicial
    # se siembra en el estado y **no** se pasa por `value=`/`default=`: hacer las dos
    # cosas a la vez es lo que dispara el aviso de Streamlit por valor duplicado.
    st.session_state.setdefault("basket_day", bundle.window_start)
    st.session_state.setdefault("channel", "app")
    basket_day = st.date_input(
        "Día de la compra",
        min_value=bundle.window_start,
        max_value=DATASET_END,
        format="DD/MM/YYYY",
        help="Mueve la estacionalidad y las promociones vigentes.",
        key="basket_day",
    )
    channel = st.segmented_control("Canal", ["app", "web", "store"], key="channel")

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
nba_banner(customer_id)


# --------------------------------------------------------------------------------------
# Cesta
# --------------------------------------------------------------------------------------
st.subheader("Tu cesta")

loaded: RealBasket | None = st.session_state.loaded_basket
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
    st.markdown(
        f"**{len(cart_ids)} productos · "
        f"{cart_total(catalog, cart_ids):.2f} €**".replace(".", ",")
    )
    product_grid(get_products(catalog, cart_ids), key_prefix="cart", action="remove")


# --------------------------------------------------------------------------------------
# Recomendaciones
# --------------------------------------------------------------------------------------
st.subheader("Te recomendamos")
with st.spinner("Calculando…"):
    recommendations = recommend(
        bundle,
        customer_id=customer_id,
        cart=cart_ids,
        basket_day=basket_day,
        channel=channel,
    )

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
        category_of = catalog.set_index("product_id")["category"].to_dict()
        score = score_hits(top5, loaded.target, category_of)
        for product_id in score.category_only:
            badges[product_id] = ("categoría acertada", "yellow")
        for product_id in score.exact:
            badges[product_id] = ("lo compró de verdad", "green")

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
            p: ("categoría acertada", "yellow")
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
st.subheader("Catálogo")
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

"""La demo arranca y ensena lo que dice ensenar (Fase 6b).

Es el unico test que **ejecuta `streamlit_app.py`**. El resto de la bateria cubre la
logica de `src/demo/` y `src/serving/`, que es donde vive el porque de cada recomendacion,
pero nada tocaba el fichero que la dibuja: un error de importacion, un mal uso de la API de
Streamlit o una clave de `session_state` duplicada solo aparecian cuando una persona abria
la app. Eso ya paso al menos una vez (ver el comentario sobre `value=`/`default=` en la
barra lateral de `streamlit_app.py`).

Se usa `streamlit.testing.v1.AppTest`, que corre el script en el mismo proceso y da acceso
al arbol de widgets y a `session_state`. No hace falta navegador ni Playwright.

## Coste

La primera ejecucion carga el bundle (~3,8 M de filas) y tarda unos 13 segundos; las demas
van en torno a 1, porque `st.cache_resource` sobrevive entre instancias de `AppTest` dentro
del mismo proceso. Por eso el arranque se comparte en una fixture de modulo en vez de
repetirse en cada test.

## Que se fija aqui, y que no

Se fijan **propiedades estructurales y decisiones tomadas**: que la app no lance, que el
contexto de una cesta real se alinee, que el banner del NBA declare su fecha de corte. No
se fija la redaccion de los textos, que cambia sin que eso sea una regresion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "streamlit_app.py"
SERVING = PROJECT_ROOT / "data" / "serving" / "metadata.json"
RANKER = PROJECT_ROOT / "models" / "recommender_ranker_lgbm.txt"
NBA_ACTIONS = PROJECT_ROOT / "predictions" / "nba_actions.parquet"
CATALOG = PROJECT_ROOT / "assets" / "product_catalog.csv"

needs_demo = pytest.mark.skipif(
    not SERVING.is_file()
    or not RANKER.is_file()
    or not NBA_ACTIONS.is_file()
    or not CATALOG.is_file(),
    reason=(
        "faltan los artefactos de la demo (bundle, ranker o tabla del NBA); "
        "genera con `python -m src.pipeline all`"
    ),
)

pytestmark = needs_demo

# Margen amplio a proposito: la primera ejecucion carga el bundle entero y en una maquina
# lenta o en la CI puede pasar de largo del minuto por defecto de `AppTest`.
TIMEOUT = 900

# El icono con el que la demo marca la insignia de la fecha de corte del NBA. Sirve de
# ancla para distinguirla del resto: hay otros textos con esa misma fecha (ver el test).
BANNER_MARKER = ":material/event:"


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=TIMEOUT)


@pytest.fixture(scope="module")
def arrancada():
    """La app recien abierta, sin tocar nada."""
    at = _app()
    at.run()
    return at


@pytest.fixture(scope="module")
def con_cesta_real():
    """La app despues de pulsar "Cargar esta cesta".

    Va en su propia instancia y no encadenada a `arrancada`: compartir una `AppTest` entre
    tests que la mutan los haria depender del orden en que pytest los ejecute.
    """
    at = _app()
    at.run()
    boton = next(b for b in at.button if "Cargar" in b.label)
    boton.click().run()
    return at


# --------------------------------------------------------------------------------------
# Que arranque
# --------------------------------------------------------------------------------------
def test_la_app_arranca_sin_excepciones(arrancada) -> None:
    """El test que justifica el fichero: nada mas ejecuta `streamlit_app.py`."""
    assert not arrancada.exception, [e.value for e in arrancada.exception]


def test_la_app_no_deja_avisos_de_streamlit(arrancada) -> None:
    """Un `st.warning` aqui seria la app diciendo que le falta algo para funcionar."""
    assert not [w.value for w in arrancada.warning]


def test_pinta_recomendaciones(arrancada) -> None:
    """Si el bundle esta, el top-5 tiene que salir; una demo vacia no es una demo."""
    assert len(arrancada.markdown) > 10
    assert any("Te recomendamos" in s.value for s in arrancada.subheader)


# --------------------------------------------------------------------------------------
# El selector de canal, que se quito a proposito
# --------------------------------------------------------------------------------------
def test_la_barra_lateral_no_ofrece_elegir_canal(arrancada) -> None:
    """Se quito porque no cambiaba casi nada y ensenaba algo falso.

    El canal solo tiene efecto real a traves del embudo de sesion, y una cesta construida a
    mano no tiene sesion. Medido en su dia: el top-5 era identico en los tres canales en 69
    de 80 combinaciones. El razonamiento completo esta en `DEFAULT_CHANNEL`.

    Este test no impide reponerlo: impide reponerlo **sin querer**.
    """
    etiquetas = [c.label for c in arrancada.sidebar.segmented_control]
    etiquetas += [c.label for c in arrancada.sidebar.selectbox]
    etiquetas += [c.label for c in arrancada.sidebar.radio]
    assert "Canal" not in etiquetas


def test_el_canal_sigue_llegando_al_modelo(arrancada) -> None:
    """Se quito el control, **no** la feature: `channel_idx` sigue en el ranker.

    Si alguien limpiara tambien la variable, la matriz de features dejaria de cuadrar con
    la de Spark y el test de paridad se caeria mucho mas lejos de la causa.
    """
    from src.recommender.schema import FEATURE_COLUMNS

    assert arrancada.session_state["channel"]
    assert "channel_idx" in FEATURE_COLUMNS


# --------------------------------------------------------------------------------------
# Cargar una cesta real
# --------------------------------------------------------------------------------------
def test_cargar_una_cesta_real_no_rompe_nada(con_cesta_real) -> None:
    assert not con_cesta_real.exception, [e.value for e in con_cesta_real.exception]


def test_cargar_una_cesta_real_alinea_dia_y_canal(con_cesta_real) -> None:
    """Alinear el contexto no es cosmetico.

    El dia mueve la estacionalidad y las promociones vigentes; el canal es una feature del
    ranker. Sin alinearlos, la demo ensenaria el top-5 de un contexto que nunca existio y
    lo contrastaria con lo que el cliente compro de verdad, que es una comparacion falsa.
    """
    cesta = con_cesta_real.session_state["loaded_basket"]
    assert cesta is not None
    assert con_cesta_real.session_state["basket_day"] == cesta.basket_day
    assert con_cesta_real.session_state["channel"] == cesta.channel


def test_el_carrito_se_siembra_con_lo_que_habia_en_el_corte(con_cesta_real) -> None:
    cesta = con_cesta_real.session_state["loaded_basket"]
    assert con_cesta_real.session_state["cart"] == list(cesta.cart)


# --------------------------------------------------------------------------------------
# El banner del NBA (punto M7)
# --------------------------------------------------------------------------------------
def test_el_banner_del_nba_declara_su_fecha_de_corte(arrancada) -> None:
    """El NBA se resuelve una vez, en un corte fijo, y se ensena junto a cestas
    posteriores. Callarse la fecha invitaba a leer el banner como si fuera de hoy.

    Se comprueba que la fecha que se pinta es la que trae la tabla, no una constante
    escrita en la demo: ese era justamente el modo de fallo que el punto M7 arreglo.

    **Hay que buscar la fecha en el banner, no en cualquier leyenda.** La primera version
    de este test hacia `any(fecha in c for c in captions)` y pasaba aunque el banner
    pintase una fecha inventada, porque otras dos leyendas llevan esa misma fecha por
    coincidencia: el inicio de la ventana servida es tambien el 1 de noviembre. Lo caza
    una mutacion, no la lectura.
    """
    import pandas as pd

    from src.nba.config import NBAConfig

    corte = pd.read_parquet(NBA_ACTIONS)["cutoff_date"].iloc[0]
    assert pd.Timestamp(corte).date() == NBAConfig().test_cutoff

    # Se pinta como insignia (`st.badge`), que AppTest expone como markdown. Antes era
    # un parrafo al pie del banner; se compacto porque ocupaba tres lineas de alto,
    # pero la fecha sigue declarada y la explicacion entera vive en el `help` de la
    # insignia.
    marcas = [m.value for m in arrancada.markdown if BANNER_MARKER in m.value]
    assert len(marcas) == 1, f"esperaba una sola marca de fecha del NBA: {marcas}"
    assert pd.Timestamp(corte).strftime("%d/%m/%Y") in marcas[0], marcas[0]


def test_el_banner_del_nba_ensena_las_dos_probabilidades(arrancada) -> None:
    etiquetas = {m.label for m in arrancada.metric}
    assert {"Compra 7 d", "Churn 4 sem"} <= etiquetas


# --------------------------------------------------------------------------------------
# El catalogo se repliega al revisar una cesta real
# --------------------------------------------------------------------------------------
def test_con_cesta_a_mano_el_catalogo_esta_abierto(arrancada) -> None:
    """Sin cesta real, la app sirve para construir una compra: el catalogo es el centro."""
    assert "Catálogo" in [s.value for s in arrancada.subheader]
    assert not any("revisando una cesta real" in c.value for c in arrancada.caption)


def test_con_cesta_real_el_catalogo_se_repliega(con_cesta_real) -> None:
    """Con una cesta real cargada la app hace otra cosa: revisar si el modelo acerto.

    Anadir productos a mano rompe el contraste contra el ticket real, que es justo lo que
    hace interesante ese modo. El catalogo se repliega detras de un desplegable -- sigue
    disponible, pero deja de pedir ser usado.
    """
    assert "Catálogo" not in [s.value for s in con_cesta_real.subheader]
    assert any("revisando una cesta real" in c.value for c in con_cesta_real.caption)


def test_el_catalogo_sigue_alcanzable_con_una_cesta_real(con_cesta_real) -> None:
    """Replegado no es eliminado: el buscador tiene que seguir ahi."""
    assert con_cesta_real.text_input


# --------------------------------------------------------------------------------------
# La interfaz no habla de fases ni de tareas del proyecto
# --------------------------------------------------------------------------------------
def test_la_interfaz_no_menciona_fases_ni_tareas(arrancada) -> None:
    """Quien abre la demo no tiene por que saber que hubo una Fase 7c.

    Solo se mira el texto visible; los docstrings del codigo si citan fases y puntos del
    diagnostico, que es donde esa referencia sirve de algo.
    """
    import re

    visible = [
        e.value
        for lista in (
            arrancada.caption,
            arrancada.markdown,
            arrancada.subheader,
            arrancada.title,
            arrancada.warning,
            arrancada.info,
        )
        for e in lista
    ]
    visible += [c.label for c in arrancada.sidebar.selectbox]
    visible += [c.label for c in arrancada.sidebar.segmented_control]

    patron = re.compile(r"Fases?\s*\d|Tareas?\s*\d", re.IGNORECASE)
    culpables = [t for t in visible if patron.search(t)]
    assert not culpables, culpables


# --------------------------------------------------------------------------------------
# El selector de cliente
# --------------------------------------------------------------------------------------
def test_el_selector_filtra_por_escenario(arrancada) -> None:
    """Antes se elegia un `customer_id` a ciegas de una lista de 60 clientes clonicos."""
    from src.demo import customers

    filtro = [
        c for c in arrancada.sidebar.segmented_control if c.label == "Qué caso quieres ver"
    ]
    assert filtro, [c.label for c in arrancada.sidebar.segmented_control]
    assert list(filtro[0].options) == list(customers.NOMBRES)


def test_las_opciones_de_cliente_se_leen_como_una_ficha(arrancada) -> None:
    """La etiqueta trae cestas, ticket y cuantas cestas de test tiene.

    Lo ultimo decide si la mejor parte de la demo va a estar disponible, y antes solo se
    descubria despues de elegir.
    """
    selector = [c for c in arrancada.sidebar.selectbox if c.label == "Cliente"][0]
    assert selector.options
    for opcion in selector.options:
        assert "cestas" in opcion
        assert "test" in opcion
        # Y tiene que caber: el desplegable recorta por el final, que es donde va el dato
        # que solo esta aqui (ver `test_la_etiqueta_cabe_en_la_barra_lateral`).
        assert len(opcion) <= 32, opcion


def test_todos_los_clientes_ofrecidos_tienen_cesta_de_test(arrancada) -> None:
    """Dos de los 60 de antes no tenian ninguna, y al elegirlos media demo no funcionaba."""
    from src.demo.baskets import customer_baskets, load_queries

    queries = load_queries()
    selector = [c for c in arrancada.sidebar.selectbox if c.label == "Cliente"][0]
    # La etiqueta empieza por el `customer_id`.
    for opcion in selector.options:
        cid = opcion.split(" ")[0]
        assert not customer_baskets(queries, cid).empty, cid


def test_cambiar_de_escenario_cambia_a_quien_se_ofrece() -> None:
    """Cada escenario tiene que traer gente distinta; si no, el filtro es decorativo."""
    listas = {}
    for escenario in ("Fiel", "Ocasional", "En riesgo"):
        at = _app()
        at.run()
        filtro = [
            c for c in at.sidebar.segmented_control if c.label == "Qué caso quieres ver"
        ][0]
        filtro.set_value(escenario).run()
        assert not at.exception, [e.value for e in at.exception]
        selector = [c for c in at.sidebar.selectbox if c.label == "Cliente"][0]
        listas[escenario] = {o.split(" ")[0] for o in selector.options}

    assert not listas["Fiel"] & listas["Ocasional"]
    assert not listas["Fiel"] & listas["En riesgo"]

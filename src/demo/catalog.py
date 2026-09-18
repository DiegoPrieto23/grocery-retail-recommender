"""Lo que la demo necesita para pintar un producto: nombre, foto, precio y el porque.

La logica de negocio vive aqui y no en `streamlit_app.py`, que se limita a invocarla y
dibujar (misma regla que `CLAUDE.md` fija para los notebooks). Asi el motivo de una
recomendacion y el buscador del catalogo se pueden probar sin levantar la app; la guarda
`check_catalog_matches` la prueba `tests/test_demo_hits.py`.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_CSV = PROJECT_ROOT / "assets" / "product_catalog.csv"

# Etiquetas legibles de las cinco fuentes de candidatos. Cortas a proposito: se pintan como
# insignia dentro de una tarjeta estrecha y un texto largo se corta a media palabra. El
# limite medido es de unos 17 caracteres con la rejilla de 5 tarjetas; "lo compras a menudo"
# (19) salia como "lo compras a menu...".
SOURCE_LABELS: dict[str, str] = {
    "src_hist": "lo compras mucho",
    "src_aff": "va con tu cesta",
    "src_cataff": "pega con tu cesta",
    "src_als": "clientes como tú",
    "src_pop": "top ventas ahora",
}

# Color de la insignia de cada fuente, con los semanticos que define `.streamlit/config.toml`.
SOURCE_COLORS: dict[str, str] = {
    "src_hist": "green",
    "src_aff": "violet",
    "src_cataff": "violet",
    "src_als": "blue",
    "src_pop": "gray",
}

# Orden en que se prefiere el motivo cuando varias fuentes proponen el mismo producto:
# de lo mas personal y explicable a lo mas generico.
SOURCE_PRIORITY: tuple[str, ...] = ("src_hist", "src_aff", "src_cataff", "src_als", "src_pop")


@dataclass(frozen=True)
class Product:
    """Un producto listo para pintar como tarjeta."""

    product_id: str
    name: str
    category: str
    department: str
    brand: str
    price: float
    image_path: str

    @property
    def image(self) -> str | None:
        """Ruta absoluta de la foto, o `None` si el grupo se quedo sin imagen."""
        if not self.image_path:
            return None
        path = PROJECT_ROOT / self.image_path
        return str(path) if path.is_file() else None


def _normalize(text: pd.Series | str) -> pd.Series | str:
    """Minusculas y sin tildes, para que "cafe" encuentre "Café".

    El catalogo esta en espanol y quien busca no escribe los acentos.
    """
    if isinstance(text, str):
        stripped = unicodedata.normalize("NFKD", text)
        return "".join(c for c in stripped if not unicodedata.combining(c)).lower()
    return (
        text.fillna("")
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
    )


def check_catalog_matches(products: pd.DataFrame, visual: pd.DataFrame) -> None:
    """Falla si el mapeo de la Fase 6a no es el del catalogo actual.

    Los `product_id` son correlativos (`P00001`, `P00002`...), asi que un
    `product_catalog.csv` de un dataset anterior **sigue cruzando**: tras la Fase 7a el
    CSV viejo de 1.500 filas cubria los 496 ids nuevos, y la demo pintaba nombres y fotos
    de otra categoria sin ningun error. Por eso no basta con que cada producto tenga fila:
    los dos conjuntos de ids tienen que ser iguales, y cada `product_name` tiene que
    empezar por la categoria real del producto (asi lo compone la Fase 6a).
    """
    ids_products, ids_visual = set(products["product_id"]), set(visual["product_id"])
    stale = ids_products != ids_visual
    if not stale:
        names = products[["product_id", "category"]].merge(
            visual[["product_id", "product_name"]], on="product_id"
        )
        stale = not all(
            str(name).startswith(str(category))
            for name, category in zip(names["product_name"], names["category"])
        )
    if stale:
        raise ValueError(
            f"{CATALOG_CSV.relative_to(PROJECT_ROOT)} no corresponde al catálogo actual "
            f"({len(ids_visual)} filas frente a {len(ids_products)} productos, o con "
            "categorías que no casan). Regenéralo sin llamar a Pexels: "
            "`python -m src.catalog.build_assets --offline`."
        )


def load_catalog(processed_dir: Path | None = None) -> pd.DataFrame:
    """Cruza `products` con el mapeo visual de la Fase 6a.

    Lee `data/processed/products.parquet` y no la tabla del bundle de serving: aquella
    esta indexada para el modelo y ya no conserva `department` como texto, que es
    justamente lo que el navegador de la demo necesita para agrupar.

    Anade `search_text`, la concatenacion normalizada de nombre, categoria, marca y
    departamento. Se precalcula una vez al cargar porque el buscador la recorre en cada
    pulsacion de tecla.

    Falla con un mensaje util si `assets/` no esta: sin fotos la demo no tiene sentido,
    y el arreglo (`python -m src.catalog.build_assets`) no es adivinable.
    """
    if not CATALOG_CSV.is_file():
        raise FileNotFoundError(
            f"No encuentro {CATALOG_CSV.relative_to(PROJECT_ROOT)}. "
            "Ejecuta antes la Fase 6a: `python -m src.catalog.build_assets`."
        )
    directory = processed_dir or (PROJECT_ROOT / "data" / "processed")
    products = pd.read_parquet(directory / "products.parquet")
    visual = pd.read_csv(CATALOG_CSV)
    check_catalog_matches(products, visual)
    merged = products.merge(visual, on="product_id", how="left")
    merged["product_name"] = merged["product_name"].fillna(merged["category"])
    merged["image_path"] = merged["image_path"].fillna("")
    merged["brand"] = merged["brand"].fillna("Marca blanca")
    merged["search_text"] = _normalize(
        merged["product_name"]
        + " "
        + merged["category"]
        + " "
        + merged["brand"]
        + " "
        + merged["department"]
    )
    return merged


def _cheapest_first(frame: pd.DataFrame, exclude: list[str] | None, limit: int) -> list[str]:
    """Del mas barato al mas caro, quitando lo que ya esta en la cesta.

    El orden por precio es una decision de escaparate, no del modelo: dentro de una
    categoria las 8 referencias solo se distinguen por marca y precio, asi que
    ordenarlas por precio da una lista estable y con sentido para quien compra. El
    ranking del modelo se ve en "Te recomendamos", que es donde toca.
    """
    if exclude:
        frame = frame.loc[~frame["product_id"].isin(exclude)]
    return frame.sort_values(["unit_price", "product_id"])["product_id"].head(limit).tolist()


def search(
    catalog: pd.DataFrame,
    query: str,
    *,
    exclude: list[str] | None = None,
    limit: int = 15,
) -> list[str]:
    """`product_id` que casan con el texto, en todo el catalogo.

    Todas las palabras tienen que aparecer (AND, no OR): buscar "leche desnatada" no
    devuelve todas las leches mas todo lo desnatado.
    """
    frame = catalog
    for word in str(query).split():
        frame = frame.loc[frame["search_text"].str.contains(_normalize(word), regex=False)]
    return _cheapest_first(frame, exclude, limit)


def browse(
    catalog: pd.DataFrame,
    department: str,
    category: str | None = None,
    *,
    exclude: list[str] | None = None,
    limit: int = 15,
) -> list[str]:
    """Escaparate de un departamento, o el detalle de una de sus categorias.

    Sin categoria elegida devuelve **un producto por categoria**, no los mas baratos del
    departamento. La razon es visual: la foto de la Fase 6a es una por `visual_group`
    (= categoria), asi que una rejilla de los 15 productos mas baratos de "Bebé" sale
    entera de potitos y con la misma foto quince veces. Un producto por categoria enseña
    el departamento de verdad, y las pastillas de categoria son las que bajan al detalle.
    """
    frame = catalog.loc[catalog["department"] == department]
    if category:
        return _cheapest_first(frame.loc[frame["category"] == category], exclude, limit)
    if exclude:
        frame = frame.loc[~frame["product_id"].isin(exclude)]
    escaparate = (
        frame.sort_values(["category", "unit_price", "product_id"])
        .groupby("category", as_index=False)
        .head(1)
    )
    return escaparate["product_id"].head(limit).tolist()


def to_product(row: pd.Series) -> Product:
    return Product(
        product_id=row["product_id"],
        name=row["product_name"],
        category=row["category"],
        department=row["department"],
        brand=row.get("brand", "") or "",
        price=float(row["unit_price"]),
        image_path=row.get("image_path", "") or "",
    )


def get_products(catalog: pd.DataFrame, product_ids: list[str]) -> list[Product]:
    """Productos en el orden pedido (el del ranking), no en el del catalogo."""
    indexed = catalog.set_index("product_id")
    products = []
    for product_id in product_ids:
        if product_id not in indexed.index:
            continue
        row = indexed.loc[product_id].to_dict()
        row["product_id"] = product_id
        products.append(to_product(pd.Series(row)))
    return products


def cart_total(catalog: pd.DataFrame, product_ids: list[str]) -> float:
    """Importe de la cesta. Un `product_id` que no este en el catalogo no suma."""
    return float(catalog.loc[catalog["product_id"].isin(product_ids), "unit_price"].sum())


def explain(row: pd.Series) -> tuple[str, str]:
    """Motivo de una recomendacion, derivado de las features del ranker.

    Devuelve `(texto, color)` para pintarlo como badge. El motivo no es una
    racionalizacion a posteriori: sale de las mismas columnas que el ranker uso para
    ordenar, asi que si dice "te toca reponerlo" es porque `cat_due` estaba a 1.

    La reposicion manda sobre la fuente: es lo mas concreto que se le puede decir a un
    cliente, y es la senal que el generador inyecta a proposito (`DATA_SPEC.md`).
    """
    if float(row.get("cat_due", 0) or 0) >= 1.0:
        ratio = row.get("cat_overdue_ratio")
        if pd.notna(ratio) and ratio >= 1.5:
            return "toca reponerlo ya", "green"
        return "te toca reponerlo", "green"

    for source in SOURCE_PRIORITY:
        if float(row.get(source, 0) or 0) >= 1.0:
            return SOURCE_LABELS[source], SOURCE_COLORS[source]

    return "sugerencia del modelo", "gray"


def promo_badge(row: pd.Series) -> str | None:
    """Texto de la insignia de promocion, si el producto esta de oferta ese dia.

    No se ensena el importe: `discount_value` significa una cosa distinta segun
    `promo_type` (0,5 en un 2x1, una fraccion en un `discount_pct`, euros en un `coupon`,
    ver `DATA_SPEC.md`). Al ranker le vale esa mezcla como feature; a un cliente no se le
    puede ensenar "-0,50" y dejar que adivine si son euros o la mitad del segundo.
    """
    if float(row.get("is_on_promo", 0) or 0) >= 1.0:
        return "en promoción"
    return None


ACTION_LABELS: dict[str, tuple[str, str, str]] = {
    # accion -> (titulo, icono material, color del badge)
    "recomendar_categoria": (
        "Destacar categoría",
        ":material/recommend:",
        "blue",
    ),
    "enviar_cupon_categoria": (
        "Enviar cupón de categoría",
        ":material/local_activity:",
        "orange",
    ),
    "ninguna_accion": (
        "No actuar",
        ":material/do_not_disturb_on:",
        "gray",
    ),
}


def describe_action(action: pd.Series) -> dict[str, object]:
    """Traduce una fila de `nba_actions` a algo que se pueda leer en un banner.

    `cutoff` es la fecha del corte con el que se decidio la accion. Viaja en la propia
    tabla desde el punto M7: el NBA se resuelve una vez, en un corte fijo, y la demo
    ensena cestas de fechas posteriores, asi que callarse la fecha invitaba a leer el
    banner como si fuera de hoy.
    """
    name = str(action["action"])
    title, icon, color = ACTION_LABELS.get(name, (name, ":material/help:", "gray"))
    cutoff = action.get("cutoff_date")
    return {
        "title": title,
        "icon": icon,
        "color": color,
        "category": action.get("category"),
        "p_purchase": float(action["p_purchase"]),
        "p_churn": float(action["p_churn"]),
        "expected_value": float(action["expected_value"]),
        "is_action": name != "ninguna_accion",
        "cutoff": None if cutoff is None or pd.isna(cutoff) else pd.Timestamp(cutoff).date(),
    }


def action_product(
    recommendations: pd.DataFrame,
    category: str | None,
    category_of: dict[str, str],
) -> str | None:
    """La referencia que el recomendador pone primero dentro de la categoria del NBA.

    Es el punto de contacto entre las dos tareas (punto M7). La politica de la Tarea 3b
    decide **la categoria** y con que uplift supuesto; el ranker de la Tarea 3a decide
    **que referencia** de esa categoria se ensena, con las features del cliente y del
    carrito de ese momento. Cada uno responde a la pregunta que sabe responder, y por eso
    la accion se llama `recomendar_categoria` y no `recomendar_producto`.

    Devuelve `None` si la categoria no aparece en la lista: con el re-ranking activo hay
    como mucho una referencia por categoria, y las que no tocan se quedan fuera. Es un
    resultado legitimo y el banner lo dice, en vez de rellenar con lo que sea.
    """
    if category is None or recommendations.empty:
        return None
    for product_id in recommendations["product_id"]:
        if category_of.get(product_id) == category:
            return str(product_id)
    return None


# Columnas minimas de una rejilla de productos: con menos, una tarjeta suelta se comeria
# el ancho entero y su foto quedaria desproporcionada.
MIN_GRID_COLUMNS = 3


def grid_columns(n_products: int, *, maximum: int, minimum: int = MIN_GRID_COLUMNS) -> int:
    """Cuantas columnas usar para pintar `n_products` tarjetas.

    `maximum` es un **tope**, no una constante. La demo creaba siempre 5 columnas aunque
    hubiera 2 productos, asi que las tarjetas salian al 20 % de ancho con el 60 % de la
    fila vacio. No era un caso raro: el **74,8 %** de las cestas reales que se pueden
    cargar traen entre 1 y 4 lineas en el carrito.

    El suelo evita el extremo contrario, una unica tarjeta a pantalla completa. Y se
    calcula sobre el total y no por fila, para que todas las filas de una misma rejilla
    tengan tarjetas del mismo tamano: si la ultima fila se estirase, la rejilla quedaria
    dentada.
    """
    if n_products <= 0:
        return maximum
    return max(min(maximum, max(n_products, minimum)), 1)

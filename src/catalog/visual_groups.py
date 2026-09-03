"""Agrupacion visual del catalogo: `category` -> `visual_group` -> termino de busqueda.

Por que `category` y no otra columna
------------------------------------
`products` solo tiene tres columnas descriptivas: `department` (8 valores), `category`
(62) y `brand` (144 razones sociales generadas por Faker, sin significado visual).

- `department` es demasiado amplio: una sola foto para "Frescos" tendria que representar
  a la vez leche, pan, fruta, carne y pescado.
- `brand` y `product_id` son demasiado especificos: 1.500 fotos distintas, y ademas las
  marcas son razones sociales sinteticas ("Familia Mir S.A."), no tienen aspecto propio.
- `category` cae justo en el nivel que pide el reto: "Leche", "Yogur", "Pescado blanco"
  son el tipo de grupo reutilizable del ejemplo (`leche_entera`, `yogur_griego`,
  `salmon`). 1.500 productos en 62 categorias: 24 productos por grupo de media.

Por eso `visual_group` **no** parte la categoria en trozos mas finos: el dataset no tiene
ningun atributo (variedad, formato, sabor) con el que hacerlo, y dividir "Leche" en
"leche entera / desnatada" seria inventarse un dato que el generador nunca escribio.

Aun asi `visual_group` es una columna propia y no un alias de `category`, por dos motivos:

1. Es un slug ASCII en snake_case, usable como nombre de fichero (`assets/leche.jpg`) y
   estable frente a la grafia de la categoria (`category_raw` trae "LECHE", "leche"...).
2. Dos categorias pequenas se fusionan con una hermana visualmente identica (ver `MERGES`),
   asi que el mapa es 62 -> 60, no 1:1.

Criterio de fusion
------------------
Se fusiona una categoria solo si cumple las tres condiciones:

1. tiene menos de `MIN_GROUP_SIZE` productos,
2. existe una categoria hermana cuya foto describe con honestidad *todos* los productos
   de la pequena (mismo objeto fisico, no solo "temas parecidos"),
3. las dos estan en el mismo departamento.

La condicion (2) es la que manda: el objetivo de la fusion no es ahorrar descargas, es
evitar dos grupos que acabarian con la misma foto. Por eso quedan grupos pequenos sin
fusionar (turron, torrijas, cava, marisco, protector solar): son pequenos, pero ninguna
hermana los representa sin mentir. El detalle esta en `docs/VISUAL_CATALOG.md`.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Umbral por debajo del cual una categoria se considera candidata a fusion (ver modulo).
MIN_GROUP_SIZE = 15

# category -> (visual_group, search_term en ingles).
#
# El termino de busqueda va en ingles a proposito: Pexels indexa sobre todo en ingles y
# una busqueda en espanol devuelve mucho menos stock, aunque el resto del proyecto este
# en espanol. Todos los terminos empujan hacia foto de producto sobre fondo limpio
# ("white background", "isolated") y evitan a proposito palabras que traigan personas,
# cocinas o restaurantes.
CATEGORY_TO_GROUP: dict[str, tuple[str, str]] = {
    # --- Frescos ---
    "Leche": ("leche", "milk bottle isolated white background"),
    "Pan": ("pan", "loaf of bread isolated white background"),
    "Fruta": ("fruta", "fresh fruit assortment white background"),
    "Verdura": ("verdura", "fresh vegetables isolated white background"),
    "Huevos": ("huevos", "eggs carton isolated white background"),
    "Yogur": ("yogur", "greek yogurt bowl with spoon"),
    "Carne de pollo": ("carne_pollo", "raw chicken breast white background"),
    "Embutido y fiambre": ("embutido", "sliced cured ham charcuterie white background"),
    "Queso": ("queso", "cheese wedge isolated white background"),
    "Carne de ternera": ("carne_ternera", "raw beef steak white background"),
    "Pescado blanco": ("pescado_blanco", "raw white fish fillet white background"),
    "Bacalao": ("pescado_blanco", ""),  # fusionada, ver MERGES
    "Torrijas y bolleria de Cuaresma": (
        "torrijas",
        "sweet pastry bun sugar white background",
    ),
    # --- Bebidas ---
    "Agua": ("agua", "bottled water plastic bottle white background"),
    "Refrescos": ("refrescos", "aluminium soft drink cans stack"),
    "Cerveza": ("cerveza", "beer bottles isolated white background"),
    "Vino": ("vino", "red wine bottle isolated white background"),
    "Zumos": ("zumos", "orange juice bottle white background"),
    "Cava y espumosos": ("cava", "sparkling wine bottle and glass"),
    # --- Despensa ---
    "Snacks y aperitivos": ("snacks", "potato chips snack bag white background"),
    "Pasta": ("pasta", "dry spaghetti pasta white background"),
    "Cafe": ("cafe", "coffee beans package white background"),
    "Chocolate y huevos de Pascua": ("chocolate", "dark chocolate bar squares broken"),
    "Galletas": ("galletas", "biscuits cookies isolated white background"),
    "Arroz": ("arroz", "uncooked white rice grains in bowl"),
    "Cereales": ("cereales", "breakfast cereal box white background"),
    "Salsa de tomate": ("salsa_tomate", "tomato sauce jar white background"),
    "Conservas de pescado": ("conservas_pescado", "canned tuna tin white background"),
    "Aceite de oliva": ("aceite_oliva", "olive oil bottle white background"),
    "Legumbres": ("legumbres", "dried beans lentils white background"),
    "Azucar y edulcorante": ("azucar", "white sugar cubes in bowl"),
    "Sopas y caldos": ("sopas", "vegetable soup can tin label"),
    "Palomitas de microondas": ("palomitas", "popcorn bowl isolated white background"),
    "Sal y especias": ("especias", "spice jars isolated white background"),
    "Harina": ("harina", "flour bag baking white background"),
    "Turron y mazapan": ("turron", "nougat almond bar sweet white background"),
    # --- Drogueria ---
    "Detergente": ("detergente", "laundry detergent bottle white background"),
    "Lavavajillas": ("lavavajillas", "dishwasher detergent tablets white background"),
    "Lejia y limpiadores": ("limpiadores_hogar", "cleaning spray bottles white background"),
    "Limpiacristales": ("limpiadores_hogar", ""),  # fusionada, ver MERGES
    "Suavizante": ("suavizante", "fabric softener blue bottle laundry care"),
    "Bolsas de basura": ("bolsas_basura", "roll of garbage bags plastic isolated"),
    "Protector solar": ("protector_solar", "sunscreen bottle isolated white background"),
    # --- Higiene ---
    "Papel higienico": ("papel_higienico", "toilet paper rolls white background"),
    "Gel de ducha": ("gel_ducha", "body wash shower gel bathroom bottle"),
    "Champu": ("champu", "shampoo bottle isolated white background"),
    "Pasta de dientes": ("pasta_dientes", "toothpaste tube white background"),
    "Desodorante": ("desodorante", "deodorant spray can cosmetic product"),
    "Acondicionador": ("acondicionador", "hair conditioner bottle white background"),
    "Higiene femenina": ("higiene_femenina", "menstrual pads tampons box product"),
    # --- Bebe ---
    "Panales": ("panales", "disposable diapers stack folded"),
    "Toallitas humedas": ("toallitas", "wet wipes tissue pack"),
    "Potitos": ("potitos", "jars of baby food puree row"),
    "Leche infantil": ("leche_infantil", "baby formula powder tin white background"),
    # --- Mascotas ---
    "Comida para perro": ("comida_perro", "dry dog food kibble pile"),
    "Comida para gato": ("comida_gato", "cat food kibble bowl white background"),
    "Arena para gato": ("arena_gato", "cat litter granules pellets"),
    # --- Congelados ---
    "Pizza congelada": ("pizza", "frozen pizza isolated white background"),
    "Precocinados congelados": ("precocinados", "frozen ready meal package white background"),
    "Verduras congeladas": ("verduras_congeladas", "frozen peas vegetables white background"),
    "Helados": ("helados", "ice cream tub isolated white background"),
    "Marisco": ("marisco", "raw prawns shrimp white background"),
}

# Fusiones aplicadas: (categoria absorbida, grupo destino, motivo).
MERGES: list[tuple[str, str, str]] = [
    (
        "Bacalao",
        "pescado_blanco",
        "El bacalao es un pescado blanco: la misma foto de lomo crudo describe con "
        "honestidad las dos categorias. Solo tiene 10 productos y comparte departamento "
        "(Frescos) con 'Pescado blanco' (22). Fusionar la foto no pierde la "
        "estacionalidad de Cuaresma, que vive en `category`, no en la imagen.",
    ),
    (
        "Limpiacristales",
        "limpiadores_hogar",
        "Limpiacristales (15) y 'Lejia y limpiadores' (21) son el mismo objeto en la "
        "estanteria: botella o spray de limpiador domestico. Por separado acabarian con "
        "dos fotos practicamente iguales.",
    ),
]

_LEGAL_FORM = re.compile(r"\s+S\.(?:[A-Za-z]{1,3}\.?)+\s*$")
_PARTNERS = re.compile(r"\s+(?:y\s+asociados|&\s*Asociados|e\s+Hijos)\s*$", re.IGNORECASE)


def slugify(text: str) -> str:
    """Pasa un texto a slug ASCII en snake_case, apto como nombre de fichero."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", ascii_text.lower())).strip("_")


def clean_brand(brand: str) -> str:
    """Quita la forma juridica de la razon social para que quepa en una tarjeta.

    Las marcas del generador son nombres de empresa de Faker ("Familia Mir S.A.",
    "Comercial Bru y asociados S.L.L."). Para el nombre de producto de la demo interesa
    la parte reconocible: "Familia Mir", "Comercial Bru".
    """
    cleaned = _LEGAL_FORM.sub("", brand.strip())
    cleaned = _PARTNERS.sub("", cleaned)
    return cleaned.strip() or brand.strip()


def build_product_name(category: str, brand: str, pack_size: int) -> str:
    """Nombre comercial derivado de category + brand + pack_size.

    El dataset no trae un nombre de producto propio (ver `DATA_SPEC.md`), asi que la demo
    lo compone: la categoria hace de descriptor, la marca de identificador y el formato
    distingue los packs. No se toca ni el esquema ni el generador.
    """
    name = f"{category} {clean_brand(brand)}".strip()
    if pack_size and pack_size > 1:
        name = f"{name} - Pack {pack_size}"
    return name


def assign_visual_groups(products: pd.DataFrame) -> pd.DataFrame:
    """Anade `visual_group` a products. Falla si aparece una categoria sin mapear.

    Fallar en vez de asignar un grupo por defecto es deliberado: si el generador anade una
    categoria nueva, hay que decidir su foto a mano, no dejarla caer en un cajon de sastre.
    """
    unknown = sorted(set(products["category"]) - set(CATEGORY_TO_GROUP))
    if unknown:
        raise ValueError(
            f"Categorias sin `visual_group` asignado: {unknown}. "
            "Anadelas a CATEGORY_TO_GROUP con su termino de busqueda en ingles."
        )
    out = products.copy()
    out["visual_group"] = out["category"].map(lambda c: CATEGORY_TO_GROUP[c][0])
    return out


def build_group_table(products: pd.DataFrame) -> pd.DataFrame:
    """Una fila por `visual_group`: termino de busqueda, recuento y categorias que agrupa.

    El termino de busqueda de un grupo fusionado es el de la categoria destino (las
    absorbidas lo llevan vacio en `CATEGORY_TO_GROUP`).
    """
    with_groups = assign_visual_groups(products)
    terms = {group: term for group, term in CATEGORY_TO_GROUP.values() if term}
    rows = (
        with_groups.groupby("visual_group")
        .agg(
            n_products=("product_id", "size"),
            categories=("category", lambda s: " + ".join(sorted(set(s)))),
            departments=("department", lambda s: " + ".join(sorted(set(s)))),
        )
        .reset_index()
    )
    rows["search_term"] = rows["visual_group"].map(terms)
    missing = rows.loc[rows["search_term"].isna(), "visual_group"].tolist()
    if missing:
        raise ValueError(f"Grupos sin termino de busqueda: {missing}")
    return rows[
        ["visual_group", "search_term", "n_products", "categories", "departments"]
    ].sort_values("visual_group", ignore_index=True)


def build_product_catalog(
    products: pd.DataFrame, image_paths: dict[str, str]
) -> pd.DataFrame:
    """CSV final: product_id, product_name, visual_group, image_path.

    `image_paths` mapea grupo -> ruta relativa de su foto; un grupo sin foto valida deja
    `image_path` vacio, para que la demo pueda distinguirlo y tirar de un marcador.
    """
    with_groups = assign_visual_groups(products)
    catalog = pd.DataFrame(
        {
            "product_id": with_groups["product_id"],
            "product_name": [
                build_product_name(category, brand, pack)
                for category, brand, pack in zip(
                    with_groups["category"],
                    with_groups["brand"],
                    with_groups["pack_size"],
                )
            ],
            "visual_group": with_groups["visual_group"],
        }
    )
    catalog["image_path"] = catalog["visual_group"].map(image_paths).fillna("")
    return _deduplicate_names(catalog).sort_values("product_id", ignore_index=True)


def _deduplicate_names(catalog: pd.DataFrame) -> pd.DataFrame:
    """Desempata nombres repetidos con un ordinal.

    Una misma categoria, marca y formato puede tocar a varios SKU con precios distintos.
    En un lineal real serian variedades; aqui basta con numerarlas para que la demo no
    muestre dos tarjetas con el mismo titulo.
    """
    out = catalog.copy()
    ordinal = out.groupby("product_name").cumcount() + 1
    totals = out["product_name"].map(out["product_name"].value_counts())
    needs_suffix = totals > 1
    out.loc[needs_suffix, "product_name"] = (
        out.loc[needs_suffix, "product_name"]
        + " ("
        + ordinal[needs_suffix].astype(str)
        + ")"
    )
    return out

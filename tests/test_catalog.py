"""Tests de la Fase 6a: agrupacion visual del catalogo y ficheros de `assets/`.

Se dividen en dos bloques:

- los que solo dependen del codigo (`visual_groups`, `pexels`, `image_hash`), que corren
  siempre y sin red;
- los que comprueban el fixture ya descargado en `assets/`, que se saltan si todavia no
  se ha ejecutado `python -m src.catalog.build_assets`.

Ninguno llama a la API de Pexels: la Fase 6a se ejecuta una sola vez a mano y su
resultado se comitea.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd
import pytest

from src.catalog.image_hash import dhash, hamming_distance, is_duplicate
from src.catalog.pexels import score_photo
from src.catalog.visual_groups import (
    CATEGORY_TO_GROUP,
    MERGES,
    assign_visual_groups,
    build_group_table,
    build_product_catalog,
    build_product_name,
    clean_brand,
    slugify,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
GROUPS_CSV = ASSETS_DIR / "visual_groups.csv"
CATALOG_CSV = ASSETS_DIR / "product_catalog.csv"
CREDITS_CSV = ASSETS_DIR / "image_credits.csv"
PROCESSED_PRODUCTS = PROJECT_ROOT / "data" / "processed" / "products.parquet"

needs_assets = pytest.mark.skipif(
    not CATALOG_CSV.exists(),
    reason="assets/ no construido; lanza `python -m src.catalog.build_assets`",
)


@pytest.fixture(scope="module")
def products() -> pd.DataFrame:
    """Catalogo de productos de un dataset de prueba, ya con el esquema limpio."""
    return pd.DataFrame(
        [
            ("P00001", "Frescos", "Leche", "Familia Mir S.A.", 1),
            ("P00002", "Frescos", "Leche", "Familia Mir S.A.", 1),
            ("P00003", "Frescos", "Leche", "Comercial Bru y asociados S.L.L.", 6),
            ("P00004", "Frescos", "Bacalao", "Sin marca", 1),
            ("P00005", "Frescos", "Pescado blanco", "Fabrica KNI S.Coop.", 2),
            ("P00006", "Drogueria", "Limpiacristales", "Banco BID S.L.", 1),
            ("P00007", "Drogueria", "Lejia y limpiadores", "Banco BID S.L.", 1),
        ],
        columns=["product_id", "department", "category", "brand", "pack_size"],
    )


# --------------------------------------------------------------------------------------
# Agrupacion visual
# --------------------------------------------------------------------------------------
def test_todas_las_categorias_del_dataset_tienen_grupo(dataset_dir: Path) -> None:
    """El mapa cubre el catalogo que escribe el generador, sin categorias sueltas."""
    real = pd.read_csv(dataset_dir / "products.csv")
    # El generador inyecta grafias sucias a proposito -- mayusculas, espacios de sobra al
    # principio, al final y en medio --; el ETL las normaliza, asi que se compara contra
    # la forma canonica de la categoria.
    categorias = {
        re.sub(r"\s+", " ", c).strip().title() for c in real["category"].dropna().unique()
    }
    conocidas = {c.title() for c in CATEGORY_TO_GROUP}
    assert categorias <= conocidas, f"sin `visual_group`: {sorted(categorias - conocidas)}"


def test_el_grupo_no_es_ni_el_departamento_ni_el_sku(products: pd.DataFrame) -> None:
    """El grano visual esta entre los dos extremos que el reto descarta."""
    grupos = {group for group, _ in CATEGORY_TO_GROUP.values()}
    assert len(grupos) == 60
    # Mas grupos que departamentos (8) y muchos menos que productos (496 desde la 7a).
    assert 8 < len(grupos) < 496


def test_cada_grupo_tiene_un_unico_termino_de_busqueda() -> None:
    """Dos categorias fusionadas comparten grupo, y por tanto termino."""
    terminos: dict[str, set[str]] = {}
    for group, term in CATEGORY_TO_GROUP.values():
        if term:
            terminos.setdefault(group, set()).add(term)
    repetidos = {g: t for g, t in terminos.items() if len(t) > 1}
    assert not repetidos, f"grupos con varios terminos: {repetidos}"


def test_los_terminos_de_busqueda_estan_en_ingles_y_son_especificos() -> None:
    """Ni vacios ni de una sola palabra: Pexels necesita contexto para acertar."""
    for _, term in CATEGORY_TO_GROUP.values():
        if term:
            assert len(term.split()) >= 3, f"termino demasiado corto: {term!r}"


def test_las_fusiones_declaradas_son_las_que_aplica_el_mapa() -> None:
    """`MERGES` documenta exactamente las categorias que ceden su grupo a otra."""
    absorbidas = {c for c, (_, term) in CATEGORY_TO_GROUP.items() if not term}
    assert absorbidas == {categoria for categoria, _, _ in MERGES}
    for categoria, destino, motivo in MERGES:
        assert CATEGORY_TO_GROUP[categoria][0] == destino
        assert len(motivo) > 40, "cada fusion tiene que venir con su motivo"


def test_build_group_table_cuenta_los_productos_fusionados(products: pd.DataFrame) -> None:
    grupos = build_group_table(products).set_index("visual_group")
    # Bacalao (1) + Pescado blanco (1) caen en el mismo grupo.
    assert grupos.loc["pescado_blanco", "n_products"] == 2
    assert "Bacalao" in grupos.loc["pescado_blanco", "categories"]
    assert grupos.loc["limpiadores_hogar", "n_products"] == 2
    assert grupos.loc["leche", "n_products"] == 3


def test_categoria_desconocida_falla_en_vez_de_caer_en_un_cajon_de_sastre() -> None:
    inventada = pd.DataFrame(
        [("P99999", "Frescos", "Ambrosia", "Sin marca", 1)],
        columns=["product_id", "department", "category", "brand", "pack_size"],
    )
    with pytest.raises(ValueError, match="Ambrosia"):
        build_group_table(inventada)


# --------------------------------------------------------------------------------------
# Nombre de producto
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("bruto", "esperado"),
    [
        ("Familia Mir S.A.", "Familia Mir"),
        ("Comercial Bru y asociados S.L.L.", "Comercial Bru"),
        ("Banco Segovia & Asociados S.L.N.E", "Banco Segovia"),
        ("Fabrica KNI S.Coop.", "Fabrica KNI"),
        ("Sin marca", "Sin marca"),
    ],
)
def test_clean_brand_quita_la_forma_juridica(bruto: str, esperado: str) -> None:
    assert clean_brand(bruto) == esperado


def test_build_product_name_solo_menciona_el_pack_si_hay_pack() -> None:
    assert build_product_name("Leche", "Familia Mir S.A.", 1) == "Leche Familia Mir"
    assert build_product_name("Leche", "Familia Mir S.A.", 6) == "Leche Familia Mir - Pack 6"


def test_el_catalogo_no_repite_nombres(products: pd.DataFrame) -> None:
    """P00001 y P00002 comparten categoria, marca y formato: se desempatan."""
    catalogo = build_product_catalog(products, {})
    assert catalogo["product_name"].is_unique
    assert catalogo.loc[0, "product_name"] == "Leche Familia Mir (1)"
    assert catalogo.loc[1, "product_name"] == "Leche Familia Mir (2)"


def test_el_catalogo_deja_image_path_vacio_si_el_grupo_no_tiene_foto(
    products: pd.DataFrame,
) -> None:
    catalogo = build_product_catalog(products, {"leche": "assets/leche.jpg"})
    con_foto = catalogo["image_path"] != ""
    assert con_foto.sum() == 3
    assert set(catalogo.loc[~con_foto, "visual_group"]) == {
        "pescado_blanco",
        "limpiadores_hogar",
    }


def test_slugify_da_nombres_de_fichero_ascii() -> None:
    assert slugify("Turrón y mazapán") == "turron_y_mazapan"
    assert slugify("Lejía / limpiadores") == "lejia_limpiadores"


# --------------------------------------------------------------------------------------
# Filtro de fotos y huella perceptual
# --------------------------------------------------------------------------------------
def _photo(alt: str, avg_color: str = "#FFFFFF", width: int = 800, height: int = 800):
    return {"alt": alt, "avg_color": avg_color, "width": width, "height": height}


def test_score_photo_descarta_personas() -> None:
    assert score_photo(_photo("A woman holding a bottle of milk")) is None
    assert score_photo(_photo("Close-up of a baby's feet on a white background")) is None
    assert score_photo(_photo("Crop anonymous female showing sanitary pad")) is None


def test_score_photo_acepta_bebe_si_no_sale_el_bebe() -> None:
    """"baby" es la palabra clave de cuatro grupos legitimos, no un descarte."""
    assert score_photo(_photo("Stack of baby diapers on white background")) is not None


def test_score_photo_prefiere_fondo_claro_y_producto_solo() -> None:
    limpia = score_photo(_photo("Olive oil bottle isolated on white background"))
    oscura = score_photo(_photo("Olive oil bottle isolated on white background", "#101010"))
    escena = score_photo(_photo("Olive oil bottle on a restaurant table with bread"))
    assert limpia > oscura
    assert limpia > escena


def test_score_photo_penaliza_los_envases_de_mockup() -> None:
    real = score_photo(_photo("Soup can tin with label"))
    mockup = score_photo(_photo("Blank mockup can template"))
    assert real > mockup


def test_dhash_reconoce_la_misma_imagen_recomprimida(tmp_path: Path) -> None:
    from PIL import Image

    original = Image.new("RGB", (400, 400), "white")
    for x in range(0, 400, 40):
        for y in range(0, 400, 40):
            original.paste(Image.new("RGB", (20, 20), (x % 256, y % 256, 128)), (x, y))

    def _bytes(image: "Image.Image", quality: int) -> bytes:
        path = tmp_path / f"tmp_{quality}.jpg"
        image.save(path, quality=quality)
        return path.read_bytes()

    igual = hamming_distance(dhash(_bytes(original, 95)), dhash(_bytes(original, 40)))
    distinta = hamming_distance(
        dhash(_bytes(original, 95)),
        dhash(_bytes(original.rotate(90), 95)),
    )
    assert igual <= 2
    assert distinta > igual


def test_is_duplicate_senala_el_grupo_gemelo() -> None:
    conocidas = {"detergente": 0b1010, "leche": 0b1111_1111_0000}
    assert is_duplicate(0b1010, conocidas) == "detergente"
    assert is_duplicate(0b1010 ^ ((1 << 40) - 1), conocidas) is None


# --------------------------------------------------------------------------------------
# El fixture ya construido en assets/
# --------------------------------------------------------------------------------------
@needs_assets
def test_el_csv_de_grupos_tiene_las_dos_columnas_pedidas() -> None:
    with GROUPS_CSV.open(encoding="utf-8") as handle:
        cabecera = next(csv.reader(handle))
    assert cabecera == ["visual_group", "search_term"]


@needs_assets
def test_el_csv_final_tiene_las_cuatro_columnas_pedidas() -> None:
    catalogo = pd.read_csv(CATALOG_CSV)
    assert list(catalogo.columns) == [
        "product_id",
        "product_name",
        "visual_group",
        "image_path",
    ]
    assert catalogo["product_id"].is_unique
    assert catalogo["product_name"].is_unique


@needs_assets
@pytest.mark.skipif(
    not PROCESSED_PRODUCTS.is_file(),
    reason="falta data/processed/products.parquet; lanza `python -m src.etl.run_etl`",
)
def test_el_csv_final_corresponde_al_catalogo_actual() -> None:
    """Una fila por producto de `products` actual, con el grupo que le toca hoy.

    Este es el test que habria cazado el desfase de la Fase 7a: el CSV de 1.500 filas
    seguia cruzando con los 496 `product_id` nuevos, pero con otra categoria detras.
    """
    products = pd.read_parquet(PROCESSED_PRODUCTS)
    catalogo = pd.read_csv(CATALOG_CSV)
    assert set(catalogo["product_id"]) == set(products["product_id"])

    esperado = assign_visual_groups(products).set_index("product_id")["visual_group"]
    actual = catalogo.set_index("product_id")["visual_group"]
    distintos = (actual != esperado.reindex(actual.index)).sum()
    assert distintos == 0, f"{distintos} productos con un `visual_group` que no les toca"
    assert (catalogo["image_path"] != "").all()
    assert catalogo["image_path"].notna().all()


@needs_assets
def test_todos_los_productos_apuntan_a_una_foto_que_existe() -> None:
    catalogo = pd.read_csv(CATALOG_CSV)
    for ruta in catalogo["image_path"].dropna().unique():
        assert (PROJECT_ROOT / ruta).is_file(), f"falta {ruta}"


@needs_assets
def test_cada_grupo_visual_tiene_su_foto() -> None:
    grupos = pd.read_csv(GROUPS_CSV)
    sin_foto = [g for g in grupos["visual_group"] if not (ASSETS_DIR / f"{g}.jpg").is_file()]
    assert not sin_foto, f"grupos sin imagen: {sin_foto}"


@needs_assets
def test_las_fotos_no_se_repiten_entre_grupos() -> None:
    """Ni el mismo id de Pexels ni una imagen visualmente igual en dos grupos."""
    creditos = pd.read_csv(CREDITS_CSV)
    assert creditos["pexels_photo_id"].is_unique

    huellas = {p.stem: dhash(p.read_bytes()) for p in sorted(ASSETS_DIR.glob("*.jpg"))}
    for grupo, huella in huellas.items():
        otras = {g: h for g, h in huellas.items() if g != grupo}
        gemelo = is_duplicate(huella, otras)
        assert gemelo is None, f"{grupo} y {gemelo} llevan la misma imagen"


@needs_assets
def test_las_fotos_pesan_lo_que_pesa_un_fixture_comiteado() -> None:
    """Van a git: si una se cuela a tamano original, el repo engorda sin motivo."""
    for foto in ASSETS_DIR.glob("*.jpg"):
        assert foto.stat().st_size < 400_000, f"{foto.name} demasiado grande"


@needs_assets
def test_cada_foto_esta_acreditada() -> None:
    """La licencia de Pexels no exige atribucion, pero el proyecto la deja por escrito."""
    creditos = pd.read_csv(CREDITS_CSV)
    grupos = set(pd.read_csv(GROUPS_CSV)["visual_group"])
    assert set(creditos["visual_group"]) == grupos
    assert creditos["photographer"].notna().all()
    assert creditos["pexels_url"].str.startswith("https://www.pexels.com/").all()

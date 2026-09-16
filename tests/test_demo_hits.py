"""Acierto de la demo en dos niveles, y la guarda contra un catalogo visual obsoleto (Fase 7e).

Son funciones puras: no necesitan el bundle de serving ni `data/processed/`, asi que corren
siempre, tambien en la CI de un repo recien clonado.

Lo que protegen:

- que el numero de SKU exacto **no se pierda ni se maquille** al anadir el de categoria:
  un acierto exacto cuenta siempre, cuenta tambien como acierto de categoria, y la frase de
  la app lleva los dos numeros;
- que la definicion de "acertar la categoria" sea la de `category_metrics` del informe;
- que un `product_catalog.csv` de otro dataset no pase por bueno solo porque sus ids cruzan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.demo.baskets import load_reference_hit_rates, score_hits
from src.demo.catalog import check_catalog_matches

CATEGORIAS = {
    "L1": "Leche",
    "L2": "Leche",
    "L3": "Leche",
    "P1": "Pan",
    "P2": "Pan",
    "G1": "Comida para gato",
    "C1": "Cafe",
    "V1": "Vino",
}


def test_la_categoria_cuenta_otra_referencia_de_lo_comprado() -> None:
    # Compro L1 y P1. Se le recomendo L2 (otra leche), P1 (exacto), G1 y C1 (nada) y V1.
    score = score_hits(["L2", "P1", "G1", "C1", "V1"], ["L1", "P1"], CATEGORIAS)
    assert score.exact == {"P1"}
    assert score.category == {"L2", "P1"}
    assert score.category_only == {"L2"}
    assert score.target_categories == {"Leche", "Pan"}
    assert score.n_recommended == 5


def test_el_exacto_siempre_esta_dentro_de_la_categoria() -> None:
    score = score_hits(["L1", "L2", "P2"], ["L1", "L3", "P2"], CATEGORIAS)
    assert score.exact <= score.category
    assert score.exact == {"L1", "P2"}
    # L2 no es exacto pero es leche, igual que L1 y L3.
    assert score.category == {"L1", "L2", "P2"}


def test_dos_recomendaciones_de_la_misma_categoria_cuentan_las_dos() -> None:
    """Igual que en `category_metrics`: se puntua cada recomendacion, no cada categoria."""
    score = score_hits(["L1", "L2", "L3"], ["L1"], CATEGORIAS)
    assert len(score.category) == 3
    assert len(score.exact) == 1


def test_un_exacto_sin_categoria_en_el_mapa_no_se_pierde() -> None:
    """Si el catalogo fallara, se pierde contexto de categoria, nunca el numero de SKU."""
    score = score_hits(["X9", "L2"], ["X9", "L1"], {"L1": "Leche", "L2": "Leche"})
    assert score.exact == {"X9"}
    assert score.category == {"X9", "L2"}


def test_sin_aciertos() -> None:
    score = score_hits(["G1", "C1"], ["L1", "P1"], CATEGORIAS)
    assert score.category == frozenset()
    assert score.exact == frozenset()
    assert "**0 de 2**" in score.describe()
    assert "**0** acertaron el producto exacto" in score.describe()


def test_la_frase_lleva_los_dos_numeros() -> None:
    score = score_hits(["L2", "P1", "G1", "C1", "V1"], ["L1", "P1"], CATEGORIAS)
    frase = score.describe()
    assert "**2 de 5** acertaron la categoría" in frase
    assert "de esas, **1** era el producto exacto" in frase


def test_la_frase_concuerda_en_singular_y_plural() -> None:
    uno = score_hits(["L2", "G1"], ["L1"], CATEGORIAS).describe()
    assert "**1 de 2** acertó la categoría" in uno
    assert "**0** eran el producto exacto" in uno

    dos = score_hits(["L1", "P1"], ["L1", "P1"], CATEGORIAS).describe()
    assert "**2 de 2** acertaron" in dos
    assert "**2** eran el producto exacto" in dos


# --------------------------------------------------------------------------------------
# Cifras de referencia del informe
# --------------------------------------------------------------------------------------
def test_las_cifras_de_referencia_salen_del_informe(tmp_path: Path) -> None:
    informe = tmp_path / "metrics.json"
    informe.write_text(
        json.dumps(
            {
                "top_k": 5,
                "by_category": [
                    {"grupo": "perfil_1", "cat_hit_rate@5": 0.1},
                    {
                        "grupo": "total",
                        "cat_hit_rate@5": 0.6,
                        "cat_precision@5": 0.2,
                        "sku_hit_rate@5": 0.5,
                        "sku_precision@5": 0.1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    ref = load_reference_hit_rates(informe)
    assert ref == pytest.approx(
        {"cat_hit_rate": 0.6, "sku_hit_rate": 0.5, "cat_per_basket": 1.0, "sku_per_basket": 0.5}
    )


def test_sin_informe_no_hay_cifras(tmp_path: Path) -> None:
    assert load_reference_hit_rates(tmp_path / "no_existe.json") is None


# --------------------------------------------------------------------------------------
# Guarda contra un product_catalog.csv obsoleto
# --------------------------------------------------------------------------------------
PRODUCTOS = pd.DataFrame(
    {"product_id": ["P00001", "P00002"], "category": ["Leche", "Pan"]}
)


def test_el_catalogo_visual_actual_pasa() -> None:
    visual = pd.DataFrame(
        {"product_id": ["P00001", "P00002"], "product_name": ["Leche Mir", "Pan Bru"]}
    )
    check_catalog_matches(PRODUCTOS, visual)


def test_un_catalogo_visual_mas_largo_no_pasa_aunque_cubra_todos_los_ids() -> None:
    """El caso real de la Fase 7a: 1.500 filas viejas cubrian los 496 ids nuevos."""
    visual = pd.DataFrame(
        {
            "product_id": ["P00001", "P00002", "P00003"],
            "product_name": ["Leche Mir", "Pan Bru", "Pan Sol"],
        }
    )
    with pytest.raises(ValueError, match="--offline"):
        check_catalog_matches(PRODUCTOS, visual)


def test_un_catalogo_visual_con_categorias_cruzadas_no_pasa() -> None:
    visual = pd.DataFrame(
        {"product_id": ["P00001", "P00002"], "product_name": ["Leche Mir", "Leche Bru"]}
    )
    with pytest.raises(ValueError):
        check_catalog_matches(PRODUCTOS, visual)

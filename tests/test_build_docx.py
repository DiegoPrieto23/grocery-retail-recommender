"""Tests del generador del .docx de negocio (`docs/build_docx.py`, punto B3).

El `.docx` dejo de escribirse a mano: sale de `docs/como-funcionan-recomendador-y-nba.md`.
Lo que se comprueba aqui es la conversion, no el fichero: que el XML que produce este bien
formado y que no se cuelen marcadores de Markdown en el texto, que es exactamente como se
ve un fallo de este conversor cuando llega al documento (un `**negrita**` literal).

El documento entero se valida aparte, contra el esquema de OOXML, al ejecutar el script.
"""

from __future__ import annotations

import importlib.util
import re
import xml.dom.minidom
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUENTE = PROJECT_ROOT / "docs" / "como-funcionan-recomendador-y-nba.md"

_spec = importlib.util.spec_from_file_location(
    "build_docx", PROJECT_ROOT / "docs" / "build_docx.py"
)
build_docx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_docx)


def _xml(markdown: str) -> str:
    """El cuerpo convertido, envuelto para poder parsearlo."""
    return build_docx.PLANTILLA.format(cuerpo=build_docx.convertir(markdown))


def _texto(xml: str) -> str:
    return re.sub(r"<[^>]+>", "", xml)


def test_el_documento_real_produce_xml_bien_formado() -> None:
    """La comprobacion que importa: la fuente de verdad, convertida entera."""
    xml.dom.minidom.parseString(_xml(FUENTE.read_text(encoding="utf-8")))


def test_no_se_cuelan_marcadores_de_markdown_en_el_texto() -> None:
    """Un `**` o un backtick en el documento es un fallo del conversor, no del Markdown."""
    visible = _texto(_xml(FUENTE.read_text(encoding="utf-8")))
    assert "**" not in visible
    assert "`" not in visible
    assert "](" not in visible  # los enlaces se quedan solo con su texto


@pytest.mark.parametrize(
    ("markdown", "esperado"),
    [
        ("# Título", "Heading1"),
        ("## Sección", "Heading2"),
        ("### Apartado", "Heading3"),
        ("- una viñeta", "ListParagraph"),
    ],
)
def test_cada_marcador_usa_el_estilo_que_le_toca(markdown: str, esperado: str) -> None:
    assert f'w:val="{esperado}"' in _xml(markdown)


def test_la_negrita_y_la_cursiva_llegan_como_formato() -> None:
    assert "<w:b/>" in _xml("texto en **negrita** dentro")
    assert "<w:i/>" in _xml("texto en *cursiva* dentro")


def test_el_code_dentro_de_negrita_no_pierde_ninguno_de_los_dos() -> None:
    """El caso que fallaba: `**texto con `code` dentro**` sacaba las comillas al papel."""
    got = _xml("**la fuente es `build_docx.py` y punto**")
    assert "`" not in _texto(got)
    assert "Consolas" in got
    assert got.count("<w:b/>") >= 2  # el code de dentro sigue en negrita


def test_una_tabla_de_markdown_sale_como_tabla_de_word() -> None:
    got = _xml("| a | b |\n| --- | ---: |\n| 1 | 2 |\n")
    assert "<w:tbl>" in got
    assert got.count("<w:tr>") == 2  # cabecera y una fila
    assert 'w:fill="D9E2F3"' in got  # la cabecera va sombreada
    xml.dom.minidom.parseString(got)


def test_una_fila_incompleta_no_rompe_la_tabla() -> None:
    got = _xml("| a | b | c |\n| --- | --- | --- |\n| solo una |\n")
    assert got.count("<w:tc>") == 6  # 3 de cabecera + 3 de la fila, rellenada
    xml.dom.minidom.parseString(got)


def test_las_lineas_de_un_parrafo_se_juntan() -> None:
    """El Markdown va a 100 columnas; esos saltos no significan nada en Word."""
    got = _xml("una frase que sigue\nen la linea de abajo\n")
    assert got.count("<w:p>") == 1
    assert "una frase que sigue en la linea de abajo" in _texto(got)


def test_una_cita_se_sangra_y_lleva_barra() -> None:
    got = _xml("> primera linea\n> de la cita\n")
    assert "<w:pBdr>" in got
    assert "<w:ind" in got


def test_el_orden_de_ppr_respeta_el_esquema() -> None:
    """Word rechaza el fichero si `ind` va antes que `spacing`. Paso una vez."""
    got = _xml("> una cita cualquiera\n")
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", got, re.S).group(0)
    assert ppr.index("<w:spacing") < ppr.index("<w:ind")


def test_los_caracteres_especiales_se_escapan() -> None:
    got = _xml("margen < coste & cupón > 0")
    assert "&lt;" in got and "&amp;" in got and "&gt;" in got
    xml.dom.minidom.parseString(got)

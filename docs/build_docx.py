"""Genera el .docx de negocio a partir del Markdown (punto B3 del diagnostico).

    python docs/build_docx.py

`docs/Como funcionan el recomendador y el Next Best Action.docx` existia como binario
escrito a mano. Un binario no se puede comparar en un *diff*, asi que sus cifras
envejecieron en silencio durante tres fases: cuando se revisó decia que el `NDCG@5` era
0,0343 y que el sistema acertaba el SKU en el 11,8 %, cuando los reales ya eran 0,3015 y
61,6 %.

El diagnostico daba dos salidas -- pasarlo a Markdown, o documentar de donde se genera --
y aqui se hacen **las dos**: la fuente es
`docs/como-funcionan-recomendador-y-nba.md`, que si se revisa en un *diff*, y el `.docx`
pasa a ser un artefacto que sale de ella con este script. Deja de poder desfasarse por su
cuenta: para que cambie, tiene que cambiar el Markdown.

## Por que no pandoc

El `.docx` original lo produjo pandoc (se le ve en los estilos: `Title`, `Heading1-6`,
`Strong`, `ListParagraph`, `Hyperlink`, `FootnoteReference`). Pero pandoc no es una
dependencia del proyecto y obligaria a instalar un binario externo solo para esto. Como el
Markdown que hay que convertir usa un subconjunto pequeno y conocido --encabezados,
parrafos, vinetas, tablas, citas, negrita y `code`--, sale mas barato escribir la
conversion aqui, con `zipfile` de la libreria estandar y nada mas.

## Como conserva el aspecto

No se construye un `.docx` desde cero: se **reutiliza el original**. Se descomprime, se
sustituye solo `word/document.xml` y se vuelve a comprimir con el resto de partes intactas
(`styles.xml`, `numbering.xml`, las relaciones). Asi el documento sigue teniendo los mismos
estilos, la misma numeracion y el mismo aspecto que tenia, y lo unico que cambia es el
contenido.
"""

from __future__ import annotations

import html
import re
import shutil
import zipfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent
FUENTE = DOCS / "como-funcionan-recomendador-y-nba.md"
DESTINO = DOCS / "Como funcionan el recomendador y el Next Best Action.docx"

# Espaciado de cada nivel de encabezado, copiado del documento original.
ESPACIADO = {1: (100, 200), 2: (320, 140), 3: (240, 120), 4: (200, 100)}

# Ancho util de la pagina (A4 con margenes de 1440 dxa), en dxa.
ANCHO_TABLA = 9400

# La lista con vinetas del original usa este `numId` de `numbering.xml`.
NUM_ID = 2


# --------------------------------------------------------------------------------------
# Texto con formato dentro de un parrafo
# --------------------------------------------------------------------------------------
def _runs(texto: str, *, negrita: bool = False) -> str:
    """Convierte `**negrita**`, `*cursiva*`, `` `code` `` y `[texto](enlace)` en `<w:r>`.

    Los enlaces se quedan solo con su texto: un hipervinculo de verdad necesita una
    entrada en `document.xml.rels`, y en un documento que se lee impreso o en Word no
    aporta lo bastante como para justificar la complicacion.
    """
    texto = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texto)

    MONO = '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="19"/>'
    BOLD = "<w:b/><w:bCs/>"

    def _run(contenido: str, props: str) -> str:
        rpr = f'<w:rPr>{props}<w:color w:val="000000"/></w:rPr>'
        return f'<w:r>{rpr}<w:t xml:space="preserve">{html.escape(contenido)}</w:t></w:r>'

    def _con_codigo(contenido: str, base: str) -> list[str]:
        """Parte un tramo por sus `` `code` `` manteniendo el formato que ya trae.

        Hace falta porque el Markdown anida los dos: `**texto con `code` dentro**`. Sin
        esto, las comillas se verian tal cual en el documento.
        """
        salida = []
        for parte in re.split(r"(`[^`]+`)", contenido):
            if not parte:
                continue
            if parte.startswith("`") and parte.endswith("`"):
                salida.append(_run(parte[1:-1], base + MONO))
            else:
                salida.append(_run(parte, base))
        return salida

    partes: list[str] = []
    # Se trocea por los tres marcadores a la vez para no anidarlos mal. La negrita va
    # antes que la cursiva en la alternativa, porque `**x**` tambien casaria con `*x*`.
    for trozo in re.split(r"(\*\*.+?\*\*|\*[^*\s][^*]*\*|`[^`]+`)", texto):
        if not trozo:
            continue
        if trozo.startswith("**") and trozo.endswith("**"):
            partes += _con_codigo(trozo[2:-2], BOLD)
        elif trozo.startswith("`") and trozo.endswith("`"):
            partes.append(_run(trozo[1:-1], (BOLD if negrita else "") + MONO))
        elif trozo.startswith("*") and trozo.endswith("*"):
            partes += _con_codigo(trozo[1:-1], "<w:i/><w:iCs/>")
        else:
            partes += _con_codigo(trozo, BOLD if negrita else "")
    return "".join(partes)


def _parrafo(texto: str, *, estilo: str | None = None, extra: str = "") -> str:
    props = ""
    if estilo:
        props += f'<w:pStyle w:val="{estilo}"/>'
    props += extra
    ppr = f"<w:pPr>{props}</w:pPr>" if props else ""
    return f"<w:p>{ppr}{_runs(texto)}</w:p>"


def _encabezado(nivel: int, texto: str) -> str:
    antes, despues = ESPACIADO.get(nivel, (200, 100))
    return _parrafo(
        texto,
        estilo=f"Heading{min(nivel, 6)}",
        extra=f'<w:spacing w:after="{despues}" w:before="{antes}"/>',
    )


def _vineta(texto: str) -> str:
    return _parrafo(
        texto,
        estilo="ListParagraph",
        extra=(
            f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{NUM_ID}"/></w:numPr>'
            '<w:spacing w:after="60"/>'
        ),
    )


def _cita(lineas: list[str]) -> str:
    """Un bloque `>` se pinta sangrado y con una barra a la izquierda, como en Markdown.

    El orden de los hijos de `<w:pPr>` lo fija el esquema y Word rechaza el fichero si no
    se respeta: `pBdr` va antes que `spacing`, y `spacing` antes que `ind`.
    """
    borde = '<w:pBdr><w:left w:val="single" w:sz="18" w:space="8" w:color="B4C6E7"/></w:pBdr>'
    salida = []
    for i, linea in enumerate(lineas):
        espaciado = f'<w:spacing w:after="{60 if i < len(lineas) - 1 else 200}"/>'
        salida.append(_parrafo(linea, extra=borde + espaciado + '<w:ind w:left="284"/>'))
    return "".join(salida)


# --------------------------------------------------------------------------------------
# Tablas
# --------------------------------------------------------------------------------------
def _celdas(fila: str) -> list[str]:
    return [c.strip() for c in fila.strip().strip("|").split("|")]


def _tabla(filas: list[str]) -> str:
    cabecera = _celdas(filas[0])
    cuerpo = [_celdas(f) for f in filas[2:]]  # la fila 1 es el separador `| --- |`
    n = len(cabecera)

    # Reparto de anchos: la primera columna se lleva mas, que suele ser la etiqueta.
    if n == 1:
        anchos = [ANCHO_TABLA]
    else:
        primera = int(ANCHO_TABLA * (0.40 if n > 2 else 0.55))
        resto = (ANCHO_TABLA - primera) // (n - 1)
        anchos = [primera] + [resto] * (n - 2) + [ANCHO_TABLA - primera - resto * (n - 2)]

    borde = "".join(
        f'<w:{lado} w:val="single" w:color="auto" w:sz="4"/>'
        for lado in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    out = [
        f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{ANCHO_TABLA}"/>'
        f"<w:tblBorders>{borde}</w:tblBorders></w:tblPr><w:tblGrid>"
        + "".join(f'<w:gridCol w:w="{a}"/>' for a in anchos)
        + "</w:tblGrid>"
    ]

    def _fila(valores: list[str], *, encabezado: bool) -> str:
        tr = ["<w:tr>"]
        if encabezado:
            tr.append('<w:trPr><w:tblHeader/></w:trPr>')
        for valor, ancho in zip(valores, anchos):
            shd = '<w:shd w:fill="D9E2F3" w:val="clear"/>' if encabezado else ""
            tr.append(
                f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{ancho}"/>{shd}</w:tcPr>'
                f"<w:p>{_runs(valor, negrita=encabezado)}</w:p></w:tc>"
            )
        return "".join(tr) + "</w:tr>"

    out.append(_fila(cabecera, encabezado=True))
    for valores in cuerpo:
        # Una fila corta no debe romper la tabla.
        valores = (valores + [""] * n)[:n]
        out.append(_fila(valores, encabezado=False))
    out.append("</w:tbl>")
    # Word necesita un parrafo detras de una tabla o la siguiente se le pega.
    out.append('<w:p><w:pPr><w:spacing w:after="120"/></w:pPr></w:p>')
    return "".join(out)


# --------------------------------------------------------------------------------------
# Markdown -> cuerpo del documento
# --------------------------------------------------------------------------------------
def convertir(markdown: str) -> str:
    lineas = markdown.split("\n")
    cuerpo: list[str] = []
    i = 0
    while i < len(lineas):
        linea = lineas[i]
        pelada = linea.strip()

        if not pelada:
            i += 1
            continue

        if pelada.startswith("#"):
            nivel = len(pelada) - len(pelada.lstrip("#"))
            cuerpo.append(_encabezado(nivel, pelada.lstrip("#").strip()))
            i += 1

        elif pelada.startswith(">"):
            bloque = []
            while i < len(lineas) and lineas[i].strip().startswith(">"):
                bloque.append(lineas[i].strip().lstrip(">").strip())
                i += 1
            # Dentro de la cita, las lineas se unen en parrafos separados por `>` vacios.
            parrafos, actual = [], []
            for l in bloque:
                if l:
                    actual.append(l)
                elif actual:
                    parrafos.append(" ".join(actual))
                    actual = []
            if actual:
                parrafos.append(" ".join(actual))
            cuerpo.append(_cita(parrafos))

        elif pelada.startswith("|"):
            tabla = []
            while i < len(lineas) and lineas[i].strip().startswith("|"):
                tabla.append(lineas[i])
                i += 1
            if len(tabla) >= 2:
                cuerpo.append(_tabla(tabla))

        elif pelada.startswith("- "):
            # Una vineta puede seguir en las lineas de debajo, sin marcador.
            texto = [pelada[2:]]
            i += 1
            while i < len(lineas) and lineas[i].startswith("  ") and lineas[i].strip():
                texto.append(lineas[i].strip())
                i += 1
            cuerpo.append(_vineta(" ".join(texto)))

        elif pelada.startswith("```"):
            # Bloques de codigo: no hay ninguno hoy, pero si aparece uno no se pierde.
            i += 1
            codigo = []
            while i < len(lineas) and not lineas[i].strip().startswith("```"):
                codigo.append(lineas[i])
                i += 1
            i += 1
            for l in codigo:
                cuerpo.append(_parrafo(f"`{l}`" if l.strip() else " "))

        else:
            # Parrafo: se juntan las lineas seguidas, porque el Markdown va a 100 columnas
            # y en Word esos saltos serian arbitrarios.
            texto = []
            while i < len(lineas) and lineas[i].strip() and not re.match(
                r"^\s*(#|>|\||- |```)", lineas[i]
            ):
                texto.append(lineas[i].strip())
                i += 1
            cuerpo.append(_parrafo(" ".join(texto), extra='<w:spacing w:after="160"/>'))

    return "".join(cuerpo)


PLANTILLA = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    "<w:body>{cuerpo}"
    '<w:sectPr><w:pgSz w:w="11906" w:h="16838" w:orient="portrait"/>'
    '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
    'w:header="708" w:footer="708" w:gutter="0"/><w:pgNumType/>'
    '<w:docGrid w:linePitch="360"/></w:sectPr></w:body></w:document>'
)


def construir(fuente: Path = FUENTE, destino: Path = DESTINO) -> Path:
    """Reescribe `word/document.xml` del .docx dejando el resto de partes como estaban."""
    if not destino.is_file():
        raise SystemExit(
            f"No encuentro {destino}. Este script actualiza el .docx existente para "
            "conservar sus estilos; no construye uno desde cero."
        )

    documento = PLANTILLA.format(cuerpo=convertir(fuente.read_text(encoding="utf-8")))

    original = zipfile.ZipFile(destino)
    partes = {n: original.read(n) for n in original.namelist()}
    original.close()
    partes["word/document.xml"] = documento.encode("utf-8")

    # Se escribe aparte y se mueve al final: si algo falla, el .docx bueno sigue ahi.
    temporal = destino.with_suffix(".docx.tmp")
    with zipfile.ZipFile(temporal, "w", zipfile.ZIP_DEFLATED) as z:
        for nombre, datos in partes.items():
            z.writestr(nombre, datos)
    shutil.move(str(temporal), str(destino))
    return destino


def main() -> int:
    destino = construir()
    print(f"Escrito {destino} desde {FUENTE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

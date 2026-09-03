"""Huella perceptual (dHash) para no repetir la misma foto en dos grupos visuales.

El identificador de Pexels no basta para detectar repeticiones: hay fotos distintas de
la misma sesion (el mismo bote rosa fotografiado dos veces) que llegan con `id` distinto
y aspecto identico. En la demo eso se nota mucho, porque `detergente` y `suavizante` —
que el dataset relaciona con un lift de 3,5x — se acabarian mostrando juntas con la misma
imagen.

dHash compara cada pixel con su vecino de la derecha sobre una miniatura en gris: dos
fotos del mismo bodegon dan huellas casi iguales aunque cambien el recorte o la
compresion.
"""

from __future__ import annotations

import io

from PIL import Image

HASH_SIZE = 8
# Distancia de Hamming (sobre 64 bits) por debajo de la cual dos fotos se consideran la
# misma imagen. 8 deja pasar recortes y recompresiones, y corta las variantes de sesion.
DUPLICATE_THRESHOLD = 8


def dhash(image_bytes: bytes, size: int = HASH_SIZE) -> int:
    """Huella de 64 bits de una imagen en memoria."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        small = image.convert("L").resize((size + 1, size), Image.LANCZOS)
        pixels = list(small.getdata())

    bits = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            bits <<= 1
            bits |= int(pixels[offset + column] > pixels[offset + column + 1])
    return bits


def hamming_distance(left: int, right: int) -> int:
    """Numero de bits en que difieren dos huellas."""
    return bin(left ^ right).count("1")


def is_duplicate(candidate: int, known: dict[str, int]) -> str | None:
    """Devuelve el grupo cuya foto ya se parece a esta, o `None` si es nueva."""
    for group, fingerprint in known.items():
        if hamming_distance(candidate, fingerprint) <= DUPLICATE_THRESHOLD:
            return group
    return None

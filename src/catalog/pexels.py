"""Cliente minimo de la API de Pexels para la Fase 6a.

Es la unica parte del proyecto que sale a internet. Se ejecuta una sola vez: las fotos
quedan cacheadas en `assets/` y comiteadas, asi que ni la demo ni el pipeline vuelven a
llamar aqui.

La clave se lee de `PEXELS_API_KEY` (fichero `.env`, ignorado por git). Nunca se imprime
ni se registra: los mensajes de error hablan de la variable, jamas de su valor.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

API_URL = "https://api.pexels.com/v1/search"
TIMEOUT = 30
# Pexels da 200 peticiones/hora en el plan gratuito; 60 grupos caben de sobra, pero una
# pausa corta entre llamadas evita que un reintento masivo dispare un 429.
SLEEP_BETWEEN_CALLS = 0.35

# Palabras en el texto alternativo que descartan la foto: implican personas. El reto pide
# foto de producto, no de alguien usandolo.
PEOPLE_WORDS = frozenset(
    """woman women man men person people girl boy child children kid kids toddler
    infant newborn mother father parent family portrait model smiling hand hands
    holding chef cook cooking waiter barista shopper customer female females male
    males anonymous crop torso lying sitting standing wearing laughing cute adorable
    playing sleeping feet foot legs arms face fingers skin""".split()
)
# "baby" no entra en la lista: es la palabra clave de cuatro grupos legitimos (panales,
# potitos, toallitas, leche infantil), asi que descartar por ella dejaria esos cuatro sin
# foto. Lo que descarta una foto de bebe son las palabras de arriba: partes del cuerpo y
# posturas. Los alts de Pexels describen a las personas asi ("crop anonymous female
# showing...", "baby lying on back"), que es como se colaron las dos primeras tandas.

# Palabras que no descartan pero restan: escenas de cocina, restaurante o lineal de
# tienda, que se alejan del aspecto ecommerce sobre fondo limpio.
SCENE_WORDS = frozenset(
    """restaurant kitchen cafe bar dining table plate served meal recipe farm field
    garden market supermarket shelf store aisle picnic party christmas street urban
    graffiti outdoor sidewalk wall""".split()
)

# Palabras que suman: son las que describen justo el encuadre que buscamos.
STUDIO_WORDS = frozenset(
    """white background isolated studio closeup close-up product packaging container
    bottle jar can pack package""".split()
)

# Marcas de bodegon: el alt describe una escena montada con varios objetos ("milk bottle
# with stacked cookies"), no el producto solo. Restan, porque en una tarjeta de catalogo
# confunden sobre cual de los dos productos se vende.
COMPOSITION_WORDS = frozenset(
    """with alongside surrounded accompanied arrangement flatlay rustic cozy decorated
    festive aesthetic artistic vintage stacked composition""".split()
)

# Envases en blanco para diseñadores: salen limpisimos y con fondo blanco, asi que el
# resto de la puntuacion los adora, pero no muestran ningun producto reconocible.
BLANK_WORDS = frozenset("mockup blank plain unlabeled unbranded template".split())


@dataclass(frozen=True)
class PhotoChoice:
    """Una foto elegida, con lo necesario para descargarla y acreditarla."""

    photo_id: int
    download_url: str
    page_url: str
    photographer: str
    photographer_url: str
    alt: str
    score: float
    query_used: str


def load_api_key() -> str:
    """Lee `PEXELS_API_KEY` del entorno o de `.env`. No devuelve nunca el valor a un log."""
    load_dotenv()
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Falta PEXELS_API_KEY. Copia `.env.example` a `.env` y pon ahi tu clave de "
            "https://www.pexels.com/api/ . El fichero `.env` esta en `.gitignore`."
        )
    return key


def _brightness(avg_color: str | None) -> float:
    """Luminosidad 0-1 del color medio que devuelve Pexels (`#RRGGBB`).

    Una foto de producto sobre fondo blanco tiene un color medio muy claro; sirve de
    proxy barato para "fondo limpio" sin descargar la imagen.
    """
    if not avg_color or not avg_color.startswith("#") or len(avg_color) != 7:
        return 0.5
    r, g, b = (int(avg_color[i : i + 2], 16) for i in (1, 3, 5))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def score_photo(photo: dict) -> float | None:
    """Puntua una foto candidata. Devuelve `None` si hay que descartarla.

    No existe forma de pedirle a Pexels "sin personas", asi que el filtro se hace sobre el
    texto alternativo mas el color medio: es una heuristica, no una garantia, y por eso el
    resultado se revisa despues a ojo (las fotos quedan comiteadas y a la vista).
    """
    alt = (photo.get("alt") or "").lower()
    words = set(alt.replace("-", " ").replace(",", " ").split())
    if words & PEOPLE_WORDS:
        return None

    score = 0.0
    score += 3.0 * _brightness(photo.get("avg_color"))
    score += 1.2 * len(words & STUDIO_WORDS)
    score -= 1.0 * len(words & SCENE_WORDS)
    score -= 0.8 * len(words & COMPOSITION_WORDS)
    score -= 1.5 * len(words & BLANK_WORDS)

    width, height = photo.get("width") or 1, photo.get("height") or 1
    aspect = max(width, height) / min(width, height)
    score -= min(aspect - 1.0, 2.0)  # penaliza panoramicas: no encajan en una tarjeta

    return score


def _download_url(photo: dict) -> str:
    """URL de descarga recortada a cuadrado y comprimida.

    Pexels acepta parametros de transformacion sobre la URL original. Pedir 800x800
    deja ficheros de decenas de KB en vez de varios MB, que es lo que interesa para algo
    que se va a comitear.
    """
    original = photo["src"]["original"]
    return f"{original}?auto=compress&cs=tinysrgb&fit=crop&w=800&h=800"


def search_candidates(
    query: str,
    api_key: str,
    *,
    per_page: int = 24,
    limit: int = 10,
    exclude_ids: set[int] | None = None,
) -> list[PhotoChoice]:
    """Busca en Pexels y devuelve las candidatas validas, de mejor a peor puntuada.

    Devuelve una lista y no una sola foto porque la puntuacion se hace sobre el texto
    alternativo: hasta que no se descarga la imagen no se puede comprobar que no sea
    casi identica a la de otro grupo. Quien llama baja por la lista hasta dar con una
    que pase tambien esa comprobacion.

    Junta las candidatas de tres consultas en un solo pozo y elige la mejor del conjunto,
    en vez de quedarse con la primera consulta que devuelva algo:

    1. la consulta completa filtrando por `color=white` (el filtro nativo de Pexels que
       mas se acerca a "producto sobre fondo blanco"),
    2. la misma consulta sin filtro de color,
    3. una version corta (las tres primeras palabras), por si la larga no tiene stock.

    El filtro `color=white` por si solo tiende a sacar envases en blanco de mockup, asi
    que mezclarlo con la busqueda sin filtro da mejor material que usarlo en cascada.

    `exclude_ids` evita que dos grupos acaben con la misma foto: sin esto, "comida para
    perro" y "comida para gato" se llevaban el mismo cuenco de pienso.
    """
    excluded = exclude_ids or set()
    short_query = " ".join(query.split()[:3])
    attempts = [(query, "white"), (query, None), (short_query, None)]

    pool: dict[int, tuple[float, dict, str]] = {}
    for attempt_query, color in attempts:
        for photo in _search(attempt_query, api_key, per_page=per_page, color=color):
            photo_id = photo["id"]
            if photo_id in excluded or photo_id in pool:
                continue
            score = score_photo(photo)
            if score is None:
                continue
            label = attempt_query + (" [color=white]" if color else "")
            pool[photo_id] = (score, photo, label)

    ranked = sorted(pool.values(), key=lambda item: item[0], reverse=True)[:limit]
    return [
        PhotoChoice(
            photo_id=photo["id"],
            download_url=_download_url(photo),
            page_url=photo.get("url", ""),
            photographer=photo.get("photographer", ""),
            photographer_url=photo.get("photographer_url", ""),
            alt=photo.get("alt", ""),
            score=round(score, 3),
            query_used=label,
        )
        for score, photo, label in ranked
    ]


def _search(query: str, api_key: str, *, per_page: int, color: str | None) -> list[dict]:
    params: dict[str, object] = {"query": query, "per_page": per_page}
    if color:
        params["color"] = color
    response = requests.get(
        API_URL,
        headers={"Authorization": api_key},
        params=params,
        timeout=TIMEOUT,
    )
    time.sleep(SLEEP_BETWEEN_CALLS)
    if response.status_code == 401:
        raise RuntimeError(
            "Pexels ha rechazado la clave (401). Revisa el valor de PEXELS_API_KEY en `.env`."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "Pexels ha devuelto 429 (limite de peticiones). Espera a la siguiente hora y "
            "vuelve a lanzar: las fotos ya descargadas no se piden otra vez."
        )
    response.raise_for_status()
    return response.json().get("photos", [])


def fetch_photo_bytes(choice: PhotoChoice) -> bytes | None:
    """Descarga la foto y comprueba que es un JPEG de verdad. `None` si no lo es.

    No escribe nada: quien llama decide si la imagen vale (por ejemplo, si no es un
    duplicado visual de otro grupo) antes de dejarla en `assets/`.
    """
    response = requests.get(choice.download_url, timeout=TIMEOUT)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"\xff\xd8\xff") or len(content) < 5_000:
        return None
    return content


def save_photo(content: bytes, destination: Path) -> None:
    """Deja la imagen en disco pasando por un temporal.

    Asi un fallo a mitad de escritura no deja en `assets/` un fichero corrupto que la
    demo intentaria pintar.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    temporary.write_bytes(content)
    temporary.replace(destination)

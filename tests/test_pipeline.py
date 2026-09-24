"""Tests del orquestador (`src/pipeline.py`, punto B1 del diagnostico).

Lo que se comprueba aqui es el **grafo**, no los pasos: que las dependencias sean
coherentes, que el orden que sale de ellas sea ejecutable y que nadie pueda anadir un paso
que dependa de otro que va despues. Ejecutar los pasos de verdad es lo que hace
`python -m src.pipeline all --scale 0.02 --fast`, que tarda una hora y no cabe en `pytest`.

El valor de esto es que el orden dejo de estar escrito en el README: si vuelve a estar mal,
falla un test en vez de fallar una reproduccion tres minutos despues de empezarla.
"""

from __future__ import annotations

import pytest

from src import pipeline as pl


def test_todas_las_dependencias_existen() -> None:
    """Un `needs` con una errata seria un paso que nunca se arrastra."""
    for step in pl.STEPS:
        for need in step.needs:
            assert need in pl.BY_NAME, f"{step.name} depende de {need}, que no existe"


def test_ningun_paso_depende_de_uno_posterior() -> None:
    """`STEPS` esta en orden de ejecucion, y `resolve` lo respeta: tiene que ser topologico.

    Si alguien anade un paso en medio de la tupla y su dependencia queda detras, el plan
    saldria en un orden imposible sin que nada se queje. Aqui se queja.
    """
    position = {step.name: i for i, step in enumerate(pl.STEPS)}
    for step in pl.STEPS:
        for need in step.needs:
            assert position[need] < position[step.name], (
                f"{step.name} va antes que {need}, del que depende"
            )


def test_no_hay_ciclos() -> None:
    """Redundante con el test anterior mientras `STEPS` este ordenada, pero no depende de
    que lo este: comprueba la propiedad en si."""
    visitados: set[str] = set()

    def visit(name: str, camino: tuple[str, ...]) -> None:
        assert name not in camino, f"ciclo: {' -> '.join(camino + (name,))}"
        if name in visitados:
            return
        for need in pl.BY_NAME[name].needs:
            visit(need, camino + (name,))
        visitados.add(name)

    for step in pl.STEPS:
        visit(step.name, ())


def test_los_nombres_son_unicos() -> None:
    nombres = [s.name for s in pl.STEPS]
    assert len(nombres) == len(set(nombres))


def test_resolver_un_paso_arrastra_sus_dependencias() -> None:
    """Pedir el NBA tiene que traerse el generador y el ETL, y en ese orden."""
    plan = [s.name for s in pl.resolve(["nba"])]
    assert plan == ["generate", "etl", "nba"]


def test_resolver_con_only_no_arrastra_nada() -> None:
    assert [s.name for s in pl.resolve(["nba"], only=True)] == ["nba"]


def test_la_cadena_completa_los_incluye_a_todos() -> None:
    plan = pl.resolve([s.name for s in pl.STEPS])
    assert len(plan) == len(pl.STEPS)


def test_el_plan_siempre_pone_cada_dependencia_antes() -> None:
    """La garantia que de verdad importa, sobre cualquier subconjunto pedido."""
    for objetivo in pl.BY_NAME:
        plan = [s.name for s in pl.resolve([objetivo])]
        for i, name in enumerate(plan):
            for need in pl.BY_NAME[name].needs:
                assert need in plan[:i], f"{need} deberia ir antes que {name}"


def test_un_paso_desconocido_falla_con_un_mensaje_util() -> None:
    with pytest.raises(SystemExit) as err:
        pl.resolve(["no-existe"])
    assert "no-existe" in str(err.value)
    assert "generate" in str(err.value)  # enumera los que si hay


def test_el_dry_run_no_ejecuta_nada() -> None:
    """Un paso que explotaria si se ejecutase; con `--dry-run` no debe tocarse."""

    def boom(args: pl.RunArgs) -> None:
        raise AssertionError("no deberia ejecutarse")

    step = pl.Step(
        name="bomba", summary="", needs=(), produces=(), run=boom, minutes=1.0
    )
    assert pl.run([step], pl.RunArgs(), dry_run=True) == 0


def test_los_pasos_reciben_la_escala() -> None:
    visto: list[float] = []
    step = pl.Step(
        name="eco",
        summary="",
        needs=(),
        produces=(),
        run=lambda args: visto.append(args.scale),
        minutes=0.1,
    )
    pl.run([step], pl.RunArgs(scale=0.02))
    assert visto == [0.02]


def test_solo_los_pasos_que_generan_datos_aceptan_escala() -> None:
    """El resto procesa lo que encuentre en `data/`: pasarles `--scale` no significaria nada."""
    con_escala = {s.name for s in pl.STEPS if s.takes_scale}
    assert con_escala == {"generate", "export-oracle"}


def test_la_estimacion_baja_con_la_escala_pero_no_a_cero() -> None:
    caro = pl.BY_NAME["recommender"]
    assert pl._estimate(caro, 1.0) > pl._estimate(caro, 0.02) >= pl.STARTUP_MINUTES


def test_lo_que_sale_por_consola_es_ascii(capsys: pytest.CaptureFixture[str]) -> None:
    """La consola de Windows es cp1252 y no sabe escribir un `✓`.

    Este test existe porque paso: el orquestador ejecutaba el paso, lo terminaba bien y
    despues reventaba con `UnicodeEncodeError` al imprimir la marca de exito. Un fallo
    justo detras del trabajo util es el mas caro de diagnosticar, asi que el contrato es
    que nada de lo que imprime este modulo salga de ASCII.
    """
    step = pl.Step(
        name="eco", summary="prueba", needs=(), produces=(), run=lambda a: None, minutes=0.1
    )
    pl.run([step], pl.RunArgs())
    salida = capsys.readouterr().out
    pl._list_steps()
    salida += capsys.readouterr().out

    salida.encode("cp1252")  # lanzaria UnicodeEncodeError
    no_ascii = sorted({c for c in salida if ord(c) > 127})
    assert not no_ascii, f"caracteres fuera de ASCII en la salida: {no_ascii}"

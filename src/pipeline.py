"""Punto de entrada unico para reproducir el proyecto entero (punto B1 del diagnostico).

    python -m src.pipeline all                  # la cadena completa, en orden
    python -m src.pipeline all --scale 0.02     # la misma cadena en pequeno
    python -m src.pipeline recommender          # un paso y todo lo que necesita
    python -m src.pipeline nba --only           # un paso, dando por buenas sus entradas
    python -m src.pipeline list                 # que pasos hay, que producen y que tardan

Antes de esto, reproducir el proyecto eran once comandos en un orden que solo estaba
escrito en el README. Un orden escrito en prosa es un orden que se rompe: no hay nada que
falle si alguien lanza el NBA antes del ETL, solo un error raro tres minutos despues.

## Como esta montado

Cada paso es un `Step`: un nombre, de que otros pasos depende, que ficheros deja escritos
y como se invoca. Las dependencias son **datos**, no comentarios, asi que el orden de
ejecucion se deriva de ellas (orden topologico) en vez de mantenerse a mano. Anadir un
paso es anadir una entrada a `STEPS`.

Los pasos se invocan llamando al `main` de cada modulo **en el mismo proceso**, no con un
subproceso. Es mas rapido (una sola JVM de Spark por sesion, que tarda lo suyo en arrancar)
y hace que un fallo se propague como una excepcion de Python con su traza, en vez de como
un codigo de salida sin contexto.

## Lo que este modulo no hace

No es un motor de construccion: no compara marcas de tiempo ni decide que esta al dia. Si
un paso ya corrio, volver a lanzarlo lo repite. Con la semilla fija eso es inocuo -- sale
lo mismo-- pero cuesta tiempo, y por eso existe `--from` y `--only`. La alternativa,
saltarse pasos por la fecha de sus ficheros, es justo la clase de magia que hace que un
artefacto viejo sobreviva a un cambio de codigo sin que nadie se entere.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    """Un paso de la cadena.

    Args:
        name: Como se invoca desde la linea de comandos.
        needs: Pasos que tienen que haber corrido antes. Es la unica fuente de verdad del
            orden; nada mas en este modulo lo codifica.
        produces: Ficheros o directorios que el paso deja escritos. Se usan para avisar de
            lo que falta cuando se lanza un paso suelto con `--only`, no para decidir si
            hay que ejecutarlo.
        run: Que hacer. Recibe los argumentos ya resueltos y no devuelve nada.
        minutes: Duracion tipica a escala 1, para que `list` sea util y para estimar.
        takes_scale: Si su CLI acepta `--scale`. Solo los dos pasos que *generan* datos;
            los demas procesan lo que encuentren en `data/`, asi que su volumen ya viene
            dado y no hay nada que pasarles.
    """

    name: str
    summary: str
    needs: tuple[str, ...]
    produces: tuple[str, ...]
    run: Callable[["RunArgs"], None]
    minutes: float
    takes_scale: bool = False


@dataclass(frozen=True)
class RunArgs:
    """Lo que un paso necesita saber de la invocacion."""

    scale: float = 1.0
    # Con `--scale` pequeno no tiene sentido pedirle al oraculo la misma precision de
    # Monte Carlo que en la corrida grande: es el grueso de su tiempo y en una muestra el
    # intervalo es ancho de todas formas.
    fast: bool = False
    extra: tuple[str, ...] = ()


# --------------------------------------------------------------------------------------
# Los pasos
# --------------------------------------------------------------------------------------
def _generate(args: RunArgs) -> None:
    from data_generation.generate_dataset import main

    main(["--scale", str(args.scale)])


def _verify_dataset(args: RunArgs) -> None:
    from data_generation.verify_dataset import main

    main([])


def _etl(args: RunArgs) -> None:
    from src.etl.run_etl import main

    main([])


def _recommender(args: RunArgs) -> None:
    from src.recommender.pipeline import main

    main([])


def _demo_profiles(args: RunArgs) -> None:
    from src.recommender.demo_profiles import main

    main([])


def _export_bundle(args: RunArgs) -> None:
    from src.serving.export_bundle import main

    main([])


def _export_demo_bundle(args: RunArgs) -> None:
    from src.serving.export_demo_bundle import main

    main([])


def _export_oracle(args: RunArgs) -> None:
    from data_generation.export_oracle import main

    main(["--scale", str(args.scale)])


def _verify_recommender(args: RunArgs) -> None:
    from src.recommender.verify_recommender_diagnostics import main

    main(["--mc-samples", "200"] if args.fast else [])


def _nba(args: RunArgs) -> None:
    from src.nba.pipeline import main

    main([])


def _verify_category_need(args: RunArgs) -> None:
    from src.nba.verify_category_need import main

    main([])


def _impact(args: RunArgs) -> None:
    from src.impact.pipeline import main

    main([])


def _findings(args: RunArgs) -> None:
    from src.eda.findings import main

    main()


def _assets(args: RunArgs) -> None:
    from src.catalog.build_assets import main

    main()


STEPS: tuple[Step, ...] = (
    Step(
        name="generate",
        summary="Genera los 7 CSV sinteticos con la semilla fija",
        needs=(),
        produces=("data/raw/manifest.json",),
        run=_generate,
        minutes=3,
        takes_scale=True,
    ),
    Step(
        name="verify-dataset",
        summary="Comprueba que los patrones inyectados estan donde deben",
        needs=("generate",),
        produces=("reports/etl/verify_dataset.json",),
        run=_verify_dataset,
        minutes=1,
    ),
    Step(
        name="etl",
        summary="Limpia las 7 tablas y construye RFM, recompra y afinidades (Tareas 1-2)",
        needs=("generate",),
        produces=("data/processed/",),
        run=_etl,
        minutes=6,
    ),
    Step(
        name="recommender",
        summary="Entrena y evalua el recomendador de dos etapas (Tarea 3a)",
        needs=("etl",),
        produces=(
            "models/recommender_ranker_lgbm.txt",
            "predictions/recommendations_test.parquet",
            "reports/recommender/metrics.md",
        ),
        run=_recommender,
        minutes=33,
    ),
    Step(
        name="demo-profiles",
        summary="Un caso de cada uno de los 4 perfiles de la demo",
        needs=("recommender",),
        produces=("reports/recommender/demo_profiles.md",),
        run=_demo_profiles,
        minutes=1,
    ),
    Step(
        name="export-bundle",
        summary="Vuelca las fuentes ajustadas para servir sin Spark",
        needs=("recommender",),
        produces=("data/serving/metadata.json",),
        run=_export_bundle,
        minutes=2,
    ),
    Step(
        name="export-oracle",
        summary="Probabilidades reales de las cestas de test, para el techo teorico",
        needs=("generate",),
        produces=("data/oracle/",),
        run=_export_oracle,
        minutes=15,
        takes_scale=True,
    ),
    Step(
        name="verify-recommender",
        summary="Baselines, techo teorico e intervalos del recomendador",
        needs=("recommender", "export-oracle"),
        produces=("reports/recommender/diagnostics.md",),
        run=_verify_recommender,
        minutes=6,
    ),
    Step(
        name="nba",
        summary="Propension de compra y de churn, y politica de valor esperado (Tarea 3b)",
        needs=("etl",),
        produces=(
            "models/nba_purchase_lgbm.txt",
            "predictions/nba_actions.parquet",
            "reports/nba/metrics.md",
        ),
        run=_nba,
        minutes=7,
    ),
    Step(
        name="verify-category-need",
        summary="Capa comun de necesidad de categoria entre el NBA y el recomendador (M7)",
        needs=("nba",),
        produces=("reports/nba/category_need.md",),
        run=_verify_category_need,
        minutes=1,
    ),
    Step(
        name="impact",
        summary="Traduce las metricas de las dos tareas a euros (Fase 5)",
        needs=("recommender", "nba"),
        produces=("IMPACT.md", "reports/impact/impact.json"),
        run=_impact,
        minutes=0.1,
    ),
    Step(
        name="findings",
        summary="Informe de hallazgos de negocio y sus figuras",
        needs=("etl",),
        produces=("reports/insights/business_findings.md",),
        run=_findings,
        minutes=1,
    ),
    Step(
        name="assets",
        summary="Mapeo producto -> foto (offline: no llama a Pexels)",
        needs=("etl",),
        produces=("assets/product_catalog.csv",),
        run=_assets,
        minutes=0.2,
    ),
    Step(
        name="export-demo-bundle",
        summary="Recorta el bundle a los clientes que la demo ofrece (lo que se despliega)",
        needs=("export-bundle", "nba"),
        produces=("data/serving/demo/manifest.json",),
        run=_export_demo_bundle,
        minutes=0.5,
    ),
)

BY_NAME: dict[str, Step] = {s.name: s for s in STEPS}


# --------------------------------------------------------------------------------------
# Orden
# --------------------------------------------------------------------------------------
def resolve(targets: Sequence[str], *, only: bool = False) -> list[Step]:
    """Los pasos a ejecutar, en orden topologico.

    Con `only`, se ejecutan exactamente los pedidos (sus dependencias se dan por
    satisfechas). Sin `only`, se arrastran todas las dependencias transitivas.

    El orden final respeta el de `STEPS` entre pasos independientes, para que dos
    ejecuciones con los mismos argumentos hagan lo mismo en el mismo orden.
    """
    unknown = [t for t in targets if t not in BY_NAME]
    if unknown:
        raise SystemExit(
            f"Paso desconocido: {', '.join(unknown)}. "
            f"Los que hay: {', '.join(BY_NAME)}. Prueba `python -m src.pipeline list`."
        )

    if only:
        wanted = set(targets)
    else:
        wanted, stack = set(), list(targets)
        while stack:
            name = stack.pop()
            if name in wanted:
                continue
            wanted.add(name)
            stack.extend(BY_NAME[name].needs)

    return [s for s in STEPS if s.name in wanted]


def _plan_table(steps: Sequence[Step], scale: float) -> str:
    total = sum(_estimate(s, scale) for s in steps)
    width = max(len(s.name) for s in steps)
    plural = "paso" if len(steps) == 1 else "pasos"
    lines = [
        f"Plan: {len(steps)} {plural}, ~{total:.0f} min estimados (escala {scale:g})",
        "",
    ]
    for i, step in enumerate(steps, 1):
        lines.append(f"  {i:>2}. {step.name:<{width}}  ~{_estimate(step, scale):>5.1f} min")
    return "\n".join(lines)


# Lo que cuesta arrancar la JVM de Spark y leer el catalogo: no baja por reducir el
# volumen, asi que es el suelo de cualquier estimacion.
STARTUP_MINUTES = 0.3


def _estimate(step: Step, scale: float) -> float:
    """Minutos estimados a esa escala.

    Escalan **todos** los pasos, no solo los dos que aceptan `--scale`: los demas procesan
    lo que el generador dejo en `data/`, asi que un dataset al 2 % les da igual de poco
    trabajo. Lo que no baja es el arranque, de ahi el suelo.

    No es una prediccion fina: sirve para saber si da tiempo a un cafe o a comer.
    """
    return max(step.minutes * scale, STARTUP_MINUTES)


# --------------------------------------------------------------------------------------
# Ejecucion
# --------------------------------------------------------------------------------------
def run(steps: Sequence[Step], args: RunArgs, *, dry_run: bool = False) -> int:
    """Ejecuta los pasos en orden, cronometrando cada uno."""
    print(_plan_table(steps, args.scale), flush=True)
    if dry_run:
        return 0

    started = time.perf_counter()
    for i, step in enumerate(steps, 1):
        head = f"[{i}/{len(steps)}] {step.name}"
        # Solo ASCII en lo que sale por consola: la consola de Windows usa cp1252 por
        # defecto y un `✓` la hace reventar con `UnicodeEncodeError` *despues* de que el
        # paso haya terminado bien, que es la peor forma posible de fallar.
        print(f"\n{'=' * 78}\n{head} - {step.summary}\n{'=' * 78}", flush=True)
        t0 = time.perf_counter()
        step.run(args)
        print(f"\n{head} OK  {time.perf_counter() - t0:.1f}s", flush=True)

    total = time.perf_counter() - started
    print(f"\nCadena completa en {total / 60:.1f} min ({dt.datetime.now():%H:%M:%S})")
    _report_missing(steps)
    return 0


def _report_missing(steps: Sequence[Step]) -> None:
    """Avisa si algun paso no dejo lo que dice dejar.

    No hace fallar la ejecucion: hay pasos con salidas opcionales (el bundle de paridad,
    por ejemplo, se puede desactivar). Pero callarselo convertiria un fallo silencioso en
    un misterio media hora despues, cuando el paso siguiente no encuentre su entrada.
    """
    missing = [
        (step.name, path)
        for step in steps
        for path in step.produces
        if not Path(path).exists()
    ]
    if missing:
        print("\nAviso: estos artefactos no aparecieron donde se esperaban:")
        for name, path in missing:
            print(f"  {name}: {path}")


def _list_steps() -> int:
    width = max(len(s.name) for s in STEPS)
    print(f"{'paso':<{width}}  {'min':>5}  depende de")
    print(f"{'-' * width}  {'-' * 5}  {'-' * 30}")
    for step in STEPS:
        needs = ", ".join(step.needs) or "-"
        print(f"{step.name:<{width}}  {step.minutes:>5.1f}  {needs}")
        print(f"{'':<{width}}         {step.summary}")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pasos: " + ", ".join(BY_NAME) + ", all, list",
    )
    parser.add_argument(
        "steps",
        nargs="+",
        help="Pasos a ejecutar, `all` para la cadena completa o `list` para verlos.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Factor de volumen del generador (0.02 = muestra rapida de punta a punta).",
    )
    parser.add_argument(
        "--only",
        action="store_true",
        help="Ejecuta solo los pasos pedidos, sin arrastrar sus dependencias.",
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        help="Empieza en este paso de la cadena completa y sigue hasta el final.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Baja la precision de los pasos caros de Monte Carlo (util a escala reducida).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime el plan.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.steps == ["list"]:
        return _list_steps()

    targets = [s.name for s in STEPS] if args.steps == ["all"] else args.steps
    steps = resolve(targets, only=args.only)

    if args.from_step:
        if args.from_step not in BY_NAME:
            raise SystemExit(f"Paso desconocido en --from: {args.from_step}")
        names = [s.name for s in steps]
        if args.from_step not in names:
            raise SystemExit(f"{args.from_step} no esta en el plan: {', '.join(names)}")
        steps = steps[names.index(args.from_step) :]

    return run(
        steps,
        RunArgs(scale=args.scale, fast=args.fast),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

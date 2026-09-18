"""El bundle recortado que sirve la demo publicada (`src/serving/export_demo_bundle.py`).

El bundle completo son 826 MB en memoria y el 99 % es historial de clientes que el selector
nunca ofrece. La app publicada carga un recorte, y lo que estos tests protegen es la unica
propiedad que hace que ese recorte sea legitimo:

**la demo publicada tiene que ensenar exactamente lo mismo que la local.**

De ahi salen las tres cosas que se fijan aqui:

1. Los clientes exportados son **los que el selector ofrece**, calculados con las mismas
   funciones y no con una regla paralela que pueda desincronizarse.
2. Las tablas que deciden *quien* entra en el selector (`customer_stats`, las cestas de
   test) no se recortan: filtrarlas moveria los cuantiles que definen los escenarios y la
   lista de clientes ofrecidos cambiaria. Es circular, y el test lo comprueba de la unica
   forma que vale: exportando y volviendo a calcular la lista con lo exportado.
3. La resolucion completa-antes-que-recortada, que es lo que permite que en local manden
   las tablas enteras (y los tests de paridad sigan midiendo lo que median) sin que el
   despliegue necesite configurar nada.

Como en el resto de la bateria, las tablas van escritas a mano: lo que se fija es la regla,
no las cifras del dataset de hoy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.demo import customers as cu
from src.demo.baskets import basket_items
from src.serving import export_demo_bundle as edb
from src.serving.recommend import resolve_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVING_DIR = PROJECT_ROOT / "data" / "serving"


# --------------------------------------------------------------------------------------
# Un bundle diminuto, escrito a mano
# --------------------------------------------------------------------------------------
# Seis clientes que cubren los tres escenarios con margen para que los cuantiles separen:
# C1-C2 compran mucho, C5-C6 poco, y C3 tiene el riesgo de fuga mas alto.
CLIENTES = {
    "C1": (100, 50.0, 300, 0.05),
    "C2": (90, 48.0, 280, 0.04),
    "C3": (60, 40.0, 200, 0.95),
    "C4": (30, 35.0, 120, 0.10),
    "C5": (4, 22.0, 15, 0.08),
    "C6": (2, 18.0, 9, 0.06),
}


def _stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": cid,
                "cust_frequency": n,
                "cust_avg_ticket": ticket,
                "cust_n_products": refs,
            }
            for cid, (n, ticket, refs, _) in CLIENTES.items()
        ]
    )


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        [{"customer_id": cid, "p_churn": p} for cid, (*_, p) in CLIENTES.items()]
    )


def _queries() -> pd.DataFrame:
    """Dos cestas por cliente: una que la demo puede ofrecer y otra que no.

    La segunda tiene el carrito vacio, que es justo lo que `demo_baskets` descarta. Sirve
    para comprobar que el recorte de lineas no arrastra cestas que nunca se van a cargar.
    """
    filas = []
    for cid in CLIENTES:
        filas.append(
            {
                "customer_id": cid,
                "basket_id": f"B{cid}-ok",
                "basket_day": "2025-11-05",
                "channel": "app",
                "n_items": 8,
                "prefix_size": 4,
                "n_target": 4,
                "profile": 4,
            }
        )
        filas.append(
            {
                "customer_id": cid,
                "basket_id": f"B{cid}-vacia",
                "basket_day": "2025-11-06",
                "channel": "web",
                "n_items": 6,
                "prefix_size": 0,
                "n_target": 6,
                "profile": 3,
            }
        )
    return pd.DataFrame(filas)


def _escribe_bundle(serving_dir: Path, processed_dir: Path, predictions_dir: Path) -> None:
    """Deja en disco el bundle completo mas pequeno que el exportador sabe leer."""
    (serving_dir / "parity").mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    _stats().to_parquet(serving_dir / "customer_stats.parquet", index=False)
    _queries().to_parquet(serving_dir / "parity" / "queries.parquet", index=False)
    _actions().to_parquet(predictions_dir / "nba_actions.parquet", index=False)

    lineas, cestas, als, items = [], [], [], []
    for cid in CLIENTES:
        for sufijo in ("ok", "vacia"):
            basket_id = f"B{cid}-{sufijo}"
            cestas.append(
                {
                    "customer_id": cid,
                    "basket_id": basket_id,
                    "basket_day": "2025-11-05",
                    "total_amount": 40.0,
                }
            )
            for i in range(3):
                lineas.append(
                    {
                        "customer_id": cid,
                        "basket_id": basket_id,
                        "basket_day": "2025-11-05",
                        "product_id": f"P{i}",
                        "quantity": 1,
                    }
                )
                items.append(
                    {"basket_id": basket_id, "product_id": f"P{i}", "quantity": 1}
                )
        als.append(
            {"customer_id": cid, "product_id": "P0", "als_score": 0.5, "als_rank": 1}
        )

    pd.DataFrame(lineas).to_parquet(serving_dir / "customer_lines.parquet", index=False)
    pd.DataFrame(cestas).to_parquet(serving_dir / "customer_baskets.parquet", index=False)
    pd.DataFrame(als).to_parquet(serving_dir / "als_topn.parquet", index=False)
    pd.DataFrame(items).to_parquet(processed_dir / "basket_items.parquet", index=False)


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    serving, processed, predictions = (
        tmp_path / "serving",
        tmp_path / "processed",
        tmp_path / "predictions",
    )
    _escribe_bundle(serving, processed, predictions)
    return serving, processed, predictions


# --------------------------------------------------------------------------------------
# Quien entra en el recorte
# --------------------------------------------------------------------------------------
def test_los_clientes_exportados_son_los_que_ofrece_el_selector() -> None:
    """La lista no se calcula con una regla propia, sino con `build_pool` + `pick`.

    Es la garantia de que si manana cambia el criterio del selector, el recorte cambia con
    el. Una regla duplicada aqui se desincronizaria en silencio y el despliegue acabaria
    ofreciendo clientes cuyo historial no esta en el bundle.
    """
    stats, queries, actions = _stats(), _queries(), _actions()

    exportados = edb.demo_customer_ids(stats, queries, actions)

    pool = cu.build_pool(
        stats,
        edb.demo_baskets(queries).groupby("customer_id")["basket_id"].nunique(),
        actions,
    )
    ofrecidos = {cid for nombre in cu.NOMBRES for cid in cu.pick(pool, nombre)}
    assert set(exportados) == ofrecidos


def test_solo_se_exportan_las_cestas_que_la_demo_puede_cargar() -> None:
    """Las de carrito vacio no se ofrecen, asi que sus lineas no hacen falta."""
    ids = edb.demo_customer_ids(_stats(), _queries(), _actions())
    cestas = edb.demo_basket_ids(_queries(), ids)

    assert cestas, "el recorte no puede quedarse sin ninguna cesta"
    assert all(b.endswith("-ok") for b in cestas)
    assert all(b.removeprefix("B").removesuffix("-ok") in ids for b in cestas)


def test_el_recorte_no_cambia_a_quien_ofrece_el_selector(
    bundle: tuple[Path, Path, Path],
) -> None:
    """La propiedad central: exportar no puede mover la lista de clientes ofrecidos.

    Los escenarios se definen por cuantiles de la poblacion elegible, asi que si el recorte
    tocara `customer_stats` o las cestas de test, los umbrales se moverian y el selector de
    la app publicada ofreceria a otra gente. Por eso esas dos tablas se versionan enteras, y
    por eso este test recalcula la lista **despues** de exportar.
    """
    serving, processed, predictions = bundle
    manifest = edb.export(serving, processed, predictions)

    despues = edb.demo_customer_ids(
        pd.read_parquet(serving / "customer_stats.parquet"),
        pd.read_parquet(serving / "parity" / "queries.parquet"),
        pd.read_parquet(predictions / "nba_actions.parquet"),
    )
    assert despues == manifest["customer_ids"]


# --------------------------------------------------------------------------------------
# Que escribe
# --------------------------------------------------------------------------------------
def test_las_tablas_recortadas_solo_traen_a_esos_clientes(
    bundle: tuple[Path, Path, Path],
) -> None:
    serving, processed, predictions = bundle
    manifest = edb.export(serving, processed, predictions)
    esperados = set(manifest["customer_ids"])

    for name in edb.PER_CUSTOMER_TABLES:
        slim = pd.read_parquet(serving / edb.DEMO_SUBDIR / f"{name}.parquet")
        assert set(slim["customer_id"]) == esperados, name


def test_las_lineas_de_ticket_se_recortan_por_cesta(
    bundle: tuple[Path, Path, Path],
) -> None:
    serving, processed, predictions = bundle
    manifest = edb.export(serving, processed, predictions)

    items = pd.read_parquet(serving / edb.DEMO_SUBDIR / "basket_items.parquet")
    esperadas = set(
        edb.demo_basket_ids(
            pd.read_parquet(serving / "parity" / "queries.parquet"),
            list(manifest["customer_ids"]),  # type: ignore[arg-type]
        )
    )
    assert set(items["basket_id"]) == esperadas


def test_el_manifiesto_deja_constancia_de_lo_recortado(
    bundle: tuple[Path, Path, Path],
) -> None:
    """Sin el no hay forma de saber a que dataset corresponde lo que hay desplegado."""
    serving, processed, predictions = bundle
    manifest = edb.export(serving, processed, predictions)

    assert (serving / edb.DEMO_SUBDIR / "manifest.json").is_file()
    assert manifest["n_customers"] == len(manifest["customer_ids"])  # type: ignore[arg-type]
    for name in edb.PER_CUSTOMER_TABLES:
        detalle = manifest["tablas"][name]  # type: ignore[index]
        assert detalle["filas_recorte"] <= detalle["filas_completa"]


def test_sin_acciones_del_nba_falla_diciendo_que_falta(
    bundle: tuple[Path, Path, Path],
) -> None:
    """Sin `p_churn` el escenario "En riesgo" desaparece y el recorte dejaria fuera a los
    clientes mas interesantes. Mejor fallar que exportar un bundle silenciosamente pobre."""
    serving, processed, predictions = bundle
    (predictions / "nba_actions.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="nba_actions"):
        edb.export(serving, processed, predictions)


# --------------------------------------------------------------------------------------
# Como lo resuelve la app
# --------------------------------------------------------------------------------------
def test_manda_la_tabla_completa_cuando_estan_las_dos(tmp_path: Path) -> None:
    """En local no puede colarse el recorte: los tests de paridad recorren todo el dataset."""
    completa = tmp_path / "customer_lines.parquet"
    completa.write_bytes(b"completa")
    recortada = tmp_path / "demo" / "customer_lines.parquet"
    recortada.parent.mkdir()
    recortada.write_bytes(b"recortada")

    assert resolve_table(completa) == completa


def test_cae_al_recorte_cuando_falta_la_completa(tmp_path: Path) -> None:
    """Es el caso del despliegue, donde las tablas grandes no se versionan."""
    completa = tmp_path / "customer_lines.parquet"
    recortada = tmp_path / "demo" / "customer_lines.parquet"
    recortada.parent.mkdir()
    recortada.write_bytes(b"recortada")

    assert resolve_table(completa) == recortada


def test_sin_ninguna_de_las_dos_devuelve_la_ruta_original(tmp_path: Path) -> None:
    """Para que el error que ve el usuario nombre la tabla que deberia existir."""
    completa = tmp_path / "customer_lines.parquet"
    assert resolve_table(completa) == completa


def test_las_lineas_de_ticket_tambien_caen_al_recorte(
    bundle: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`basket_items` vive en `data/processed/`, que el despliegue no lleva entero."""
    serving, processed, predictions = bundle
    manifest = edb.export(serving, processed, predictions)
    basket_id = edb.demo_basket_ids(
        pd.read_parquet(serving / "parity" / "queries.parquet"),
        list(manifest["customer_ids"]),  # type: ignore[arg-type]
    )[0]

    monkeypatch.setattr(
        "src.demo.baskets.DEMO_BASKET_ITEMS",
        serving / edb.DEMO_SUBDIR / "basket_items.parquet",
    )
    # Un `data/processed/` que no existe, como en la app publicada.
    lineas = basket_items(basket_id, processed_dir=processed / "no-esta")

    assert not lineas.empty
    assert set(lineas["basket_id"]) == {basket_id}


# --------------------------------------------------------------------------------------
# Contra el bundle real, si esta
# --------------------------------------------------------------------------------------
needs_bundle = pytest.mark.skipif(
    not (SERVING_DIR / "customer_stats.parquet").is_file()
    or not (SERVING_DIR / edb.DEMO_SUBDIR / "manifest.json").is_file(),
    reason=(
        "falta el bundle de serving o su recorte; genera con "
        "`python -m src.serving.export_demo_bundle`"
    ),
)


@needs_bundle
def test_el_recorte_versionado_cubre_a_todos_los_clientes_ofrecidos() -> None:
    """La guarda contra un recorte obsoleto.

    Si se regenera el dataset y no se vuelve a exportar, el selector de la app publicada
    ofreceria clientes cuyo historial ya no esta en el bundle: se servirian como si fueran
    nuevos, sin que nadie lo notara. Este test lo caza en CI.
    """
    import json

    manifest = json.loads(
        (SERVING_DIR / edb.DEMO_SUBDIR / "manifest.json").read_text(encoding="utf-8")
    )
    ofrecidos = edb.demo_customer_ids(
        pd.read_parquet(SERVING_DIR / "customer_stats.parquet"),
        pd.read_parquet(SERVING_DIR / "parity" / "queries.parquet"),
        pd.read_parquet(PROJECT_ROOT / "predictions" / "nba_actions.parquet"),
    )
    assert set(ofrecidos) <= set(manifest["customer_ids"])

"""Tests del generador de dataset (Fase 1).

Cubren tres cosas distintas:

* que el **esquema** de las 7 tablas es el de `DATA_SPEC.md`,
* que los **patrones de negocio** que el generador dice inyectar estan realmente en el
  dato (afinidad, estacionalidad, promocion, reposicion, churn, calidad),
* que la generacion es **reproducible** byte a byte con la misma semilla.

Los umbrales son deliberadamente holgados: se ejecutan sobre una muestra pequena
(`TEST_SCALE`) y lo que se quiere detectar es que un patron desaparezca o se invierta,
no una variacion de unas decimas.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from data_generation import catalog as cat
from data_generation import verify_dataset as vd
from data_generation.generate_dataset import (
    TABLE_COLUMNS,
    TABLE_ORDER,
    GeneratorConfig,
    generate,
)
from tests.conftest import TEST_SCALE


# --------------------------------------------------------------------------------------
# Catalogo
# --------------------------------------------------------------------------------------
def test_catalogo_es_coherente() -> None:
    """Las tablas de afinidad y estacionalidad solo referencian categorias existentes."""
    cat.validate_catalog()


def test_departamentos_de_data_spec() -> None:
    """Los 8 departamentos de DATA_SPEC.md tienen surtido."""
    assert {c.department for c in cat.CATEGORIES} == set(cat.DEPARTMENTS)
    assert len(cat.DEPARTMENTS) == 8


def test_los_diez_pares_de_afinidad_de_data_spec() -> None:
    assert len(cat.AFFINITY_PAIRS) == 10


def test_surtido_de_ocho_referencias_por_categoria(dataset: dict) -> None:
    """Fase 7a: catalogo curado, mismo numero de referencias en todas las categorias."""
    productos = dataset["products"].copy()
    productos["category_clean"] = vd._clean_category(productos["category"])
    por_categoria = productos.groupby("category_clean").size()

    assert len(productos) == cat.PRODUCTS_PER_CATEGORY * len(cat.CATEGORIES)
    assert set(por_categoria) == {cat.PRODUCTS_PER_CATEGORY}
    assert len(por_categoria) == len(cat.CATEGORIES)


def test_fidelidad_de_marca_hace_que_el_cliente_repita_referencia(dataset_dir: Path) -> None:
    """Fase 7a: dentro de una categoria, el cliente vuelve a su referencia preferida.

    Es el patron que faltaba y que ponia el techo al recomendador de SKU (hallazgo de la
    Fase 3): antes, un cliente con tres o mas compras en una categoria se llevaba 0,85
    referencias distintas por compra --casi nunca repetia--. La banda de habito tiene que
    repetir claramente mas que la exploratoria, que es lo que distingue el cafe de la
    fruta.
    """
    tables = vd.load_tables(dataset_dir)
    fidelidad = vd.sku_loyalty(tables)

    assert fidelidad["referencias_distintas_por_compra"] < 0.70, "no hay repeticion de SKU"
    assert (
        fidelidad["referencias_distintas_por_compra__habito"]
        < fidelidad["referencias_distintas_por_compra__exploracion"]
    ), "el habito tiene que repetir mas que la exploracion"
    # La referencia favorita de una categoria de habito se lleva la mayoria de sus compras.
    assert fidelidad["cuota_de_la_referencia_favorita__habito"] > 0.60


def test_cada_categoria_declara_su_banda_de_lealtad() -> None:
    """La lealtad de toda categoria cae en una de las dos bandas de DATA_SPEC.md."""
    bandas = {c.name: cat.loyalty_band(c) for c in cat.CATEGORIES}
    assert set(bandas.values()) == {"habito", "exploracion"}
    for c in cat.CATEGORIES:
        banda = (
            cat.HABIT_LOYALTY_BAND
            if bandas[c.name] == "habito"
            else cat.EXPLORATORY_LOYALTY_BAND
        )
        assert banda[0] <= c.loyalty <= banda[1], c.name


# --------------------------------------------------------------------------------------
# Esquema
# --------------------------------------------------------------------------------------
def test_estan_las_siete_tablas(dataset: dict) -> None:
    assert set(dataset) == set(TABLE_ORDER)
    assert len(TABLE_ORDER) == 7


@pytest.mark.parametrize("table", TABLE_ORDER)
def test_columnas_segun_data_spec(dataset: dict, table: str) -> None:
    """Nombre y orden de columnas exactamente como en DATA_SPEC.md."""
    expected = [c for c, _, _ in TABLE_COLUMNS[table]]
    assert list(dataset[table].columns) == expected


def test_claves_primarias_unicas(dataset: dict) -> None:
    for table, pk in (
        ("customers", "customer_id"),
        ("products", "product_id"),
        ("promotions", "promotion_id"),
        ("baskets", "basket_id"),
        ("sessions", "session_id"),
    ):
        assert dataset[table][pk].is_unique, f"{table}.{pk} tiene duplicados"


def test_integridad_referencial(dataset_dir: Path) -> None:
    """Ninguna FK queda huerfana: la suciedad inyectada es de valores, no de claves."""
    tables = vd.load_tables(dataset_dir)
    for check, broken in vd.referential_integrity(tables).items():
        assert broken == 0, f"{check} = {broken}"


def test_periodo_simulado_respetado(dataset: dict) -> None:
    fechas = pd.to_datetime(dataset["baskets"]["basket_date"])
    assert fechas.min() >= pd.Timestamp(cat.PERIOD_START)
    assert fechas.max() < pd.Timestamp(cat.PERIOD_END) + pd.Timedelta(days=1)


def test_store_id_solo_en_canal_store(dataset: dict) -> None:
    b = dataset["baskets"]
    assert b.loc[b["channel"] == "store", "store_id"].notna().all()
    assert b.loc[b["channel"] != "store", "store_id"].isna().all()


def test_basket_id_de_sesion_solo_si_convierte(dataset: dict) -> None:
    s = dataset["sessions"]
    assert s.loc[s["converted"], "basket_id"].notna().all()
    assert s.loc[~s["converted"], "basket_id"].isna().all()


def test_tipos_de_evento(dataset: dict) -> None:
    assert set(dataset["session_events"]["event_type"]) <= {"view", "add_to_cart"}


def _eventos_de_sesiones_convertidas(dataset: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eventos de las sesiones que convirtieron, cruzados con el ticket resultante."""
    conv = dataset["sessions"]
    conv = conv.loc[conv["converted"], ["session_id", "basket_id"]]
    items = dataset["basket_items"][["basket_id", "product_id"]].drop_duplicates()
    eventos = dataset["session_events"].merge(conv, on="session_id")
    return eventos, items


def test_el_carrito_no_es_una_copia_del_ticket(dataset: dict) -> None:
    """La senal de sesion no puede coincidir con el target del recomendador (Fase 3).

    Si `add_to_cart` fuera exactamente el contenido del ticket, usarlo como *feature* del
    ranker filtraria el target. El generador introduce abandono de carrito a nivel de
    linea, vistas que no acaban en compra y lineas que nunca pasan por la web
    (`_generate_sessions`), asi que las tres probabilidades tienen que quedar por debajo
    de 1 con holgura -- y a la vez seguir siendo altas, o la senal no valdria para nada.
    """
    eventos, items = _eventos_de_sesiones_convertidas(dataset)
    marcados = eventos.merge(
        items.assign(en_la_cesta=True), on=["basket_id", "product_id"], how="left"
    )
    marcados["en_la_cesta"] = marcados["en_la_cesta"].notna()
    tasa = marcados.groupby("event_type")["en_la_cesta"].mean()

    assert 0.75 < tasa["add_to_cart"] < 0.97, "el carrito no puede predecir el ticket"
    assert 0.40 < tasa["view"] < tasa["add_to_cart"], "ver debe ser mas debil que anadir"

    # ... y al reves: hay lineas del ticket que nunca pasaron por la web.
    del_ticket = items[items["basket_id"].isin(set(eventos["basket_id"]))]
    vistos = eventos.loc[eventos["event_type"] == "view", ["basket_id", "product_id"]]
    cobertura = del_ticket.merge(
        vistos.drop_duplicates().assign(visto=True), on=["basket_id", "product_id"], how="left"
    )["visto"].notna().mean()
    assert 0.70 < cobertura < 0.95


def test_anadir_al_carrito_va_por_detras_de_ver(dataset: dict) -> None:
    """Sin ese retardo la senal de sesion seria inservible para el recomendador.

    En un corte temporal a mitad de sesion tiene que haber productos ya vistos y todavia
    no anadidos: son los candidatos que el ranker de la Fase 3 puede acertar.
    """
    eventos, _ = _eventos_de_sesiones_convertidas(dataset)
    primero = (
        eventos.groupby(["session_id", "product_id", "event_type"])["event_timestamp"]
        .min()
        .unstack("event_type")
        .dropna()
    )
    assert not primero.empty
    assert (primero["add_to_cart"] >= primero["view"]).all()
    assert (primero["add_to_cart"] > primero["view"]).mean() > 0.9


def test_promociones_con_ventana_valida(dataset: dict) -> None:
    p = dataset["promotions"]
    assert (pd.to_datetime(p["start_date"]) <= pd.to_datetime(p["end_date"])).all()
    assert set(p["promo_type"]) <= set(cat.PROMO_TYPES)


# --------------------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------------------
def test_dos_ejecuciones_producen_el_mismo_hash(tmp_path: Path) -> None:
    """Requisito de CLAUDE.md: misma semilla -> ficheros identicos byte a byte."""
    hashes = []
    for run in ("a", "b"):
        out = tmp_path / run
        generate(GeneratorConfig(out_dir=out, scale=TEST_SCALE))
        hashes.append(
            {
                name: hashlib.sha256((out / f"{name}.csv").read_bytes()).hexdigest()
                for name in TABLE_ORDER
            }
        )
    assert hashes[0] == hashes[1]


def test_semillas_distintas_producen_datos_distintos(tmp_path: Path) -> None:
    """Control negativo: si el hash coincidiera con cualquier semilla, no probaria nada."""
    a = generate(GeneratorConfig(out_dir=tmp_path / "a", scale=TEST_SCALE, seed=42))
    b = generate(GeneratorConfig(out_dir=tmp_path / "b", scale=TEST_SCALE, seed=7))
    assert not a["basket_items"].equals(b["basket_items"])


# --------------------------------------------------------------------------------------
# Patrones de negocio
# --------------------------------------------------------------------------------------
def test_afinidad_de_cesta_por_encima_de_lo_esperado_por_azar(dataset_dir: Path) -> None:
    """Cada par de DATA_SPEC.md tiene que co-ocurrir mas de lo que dictaria el azar.

    Se exige superar 1.35x en todos los pares y llegar al menos al 60 % del objetivo en
    la mediana; el valor exacto por par depende de cuanto margen de saturacion le queda a
    la categoria asociada (si ya esta en la mitad de las cestas, no puede subir 3x).
    """
    tables = vd.load_tables(dataset_dir)
    measured = vd.basket_affinity_lift(tables)

    ratios = []
    for trigger, associated, target in cat.AFFINITY_PAIRS:
        lift = measured[f"{trigger} -> {associated}"]
        assert lift > 1.35, f"{trigger} -> {associated}: lift {lift} no supera el azar"
        ratios.append(lift / target)
    assert sorted(ratios)[len(ratios) // 2] > 0.60


def test_estacionalidad_concentrada_en_los_meses_pico(dataset_dir: Path) -> None:
    """Toda categoria estacional vende mas en su mes pico que fuera de el."""
    tables = vd.load_tables(dataset_dir)
    measured = vd.seasonality_ratio(tables)
    for name, _months, target in cat.SEASONALITY:
        ratio = measured[name]
        assert ratio > 1.2, f"{name}: ratio pico/valle {ratio}"
        # El multiplicador de la spec es un techo: la cesta satura (una linea por
        # categoria), asi que el ratio observado queda por debajo del nominal.
        assert ratio <= target * 2.0, f"{name}: ratio {ratio} muy por encima de {target}"


def test_uplift_de_promocion(dataset_dir: Path) -> None:
    """Un producto en promocion se vende mas rapido que fuera de su ventana.

    El umbral es bajo a proposito por dos motivos. Uno, a `TEST_SCALE` hay un punado de
    promociones y la medida es ruidosa. Y dos, desde la Fase 7a el uplift compite con la
    fidelidad de marca: el multiplicador sigue siendo exactamente `PROMO_UPLIFT` sobre la
    parte no fiel de la categoria, pero el uplift observado a nivel de categoria baja,
    porque una promocion no mueve a quien ya tiene marca fija. La cifra a volumen completo
    la recalcula `python -m data_generation.verify_dataset`.
    """
    tables = vd.load_tables(dataset_dir)
    uplift = vd.promotion_uplift(tables)["promo_uplift_observado"]
    assert uplift > 1.2, f"uplift observado {uplift}"


def test_ciclos_de_reposicion_conservan_el_orden(dataset_dir: Path) -> None:
    """El intervalo observado no es el de la spec, pero si respeta su orden.

    La frecuencia de visita del cliente acota por abajo el intervalo observable, asi que
    lo que se comprueba es la correlacion de rangos: la leche se recompra antes que el
    detergente, y el detergente antes que el turron.
    """
    tables = vd.load_tables(dataset_dir)
    corr = vd.repurchase_cycles(tables)["correlacion_rangos_spec_vs_observado"]
    assert corr > 0.75, f"correlacion de rangos {corr}"


def test_churn_progresivo_no_es_un_corte_en_seco(dataset_dir: Path) -> None:
    """La frecuencia de compra decae antes del abandono, y mas que en los activos.

    Es la mitad robusta de la rampa: el generador reparte menos dias de compra segun se
    acerca el abandono, y eso se ve ya a `TEST_SCALE`. La otra mitad --el ticket-- es de
    segundo orden y necesita mas muestra: la comprueba
    `test_el_ticket_tambien_decae_antes_del_abandono`.
    """
    tables = vd.load_tables(dataset_dir)
    signal = vd.churn_signal(tables)
    assert signal["ratio_frecuencia_churn"] < signal["ratio_frecuencia_activo"]
    assert signal["ratio_frecuencia_churn"] < 0.90


@pytest.mark.slow
def test_el_ticket_tambien_decae_antes_del_abandono(dataset_dir_grande: Path) -> None:
    """El ticket medio encoge antes del abandono, no solo la frecuencia.

    Va sobre `dataset_dir_grande` a proposito: el estimador es una mediana de medianas
    sobre las dos o tres cestas que le quedan a un churner en la ventana, y con los ~170
    churners de `TEST_SCALE` se mueve varios puntos segun la semilla. Con ~700 el orden
    es estable. A volumen completo da 0,972 frente a 1,012 y lo recalcula
    `python -m data_generation.verify_dataset`.
    """
    signal = vd.churn_signal(vd.load_tables(dataset_dir_grande))
    assert signal["ratio_ticket_churn"] < signal["ratio_ticket_activo"]
    assert signal["ratio_ticket_churn"] < 1.0


def test_churn_label_es_la_definicion_de_data_spec(dataset: dict) -> None:
    """churn_label = sin compra en los ultimos CHURN_WINDOW_DAYS del periodo."""
    cutoff = pd.Timestamp(cat.PERIOD_END) - pd.Timedelta(days=cat.CHURN_WINDOW_DAYS - 1)
    b = dataset["baskets"].dropna(subset=["customer_id"]).copy()
    b["basket_date"] = pd.to_datetime(b["basket_date"])
    activos = set(b.loc[b["basket_date"] >= cutoff, "customer_id"])

    c = dataset["customers"]
    assert not c.loc[c["customer_id"].isin(activos), "churn_label"].any()
    assert c.loc[~c["customer_id"].isin(activos), "churn_label"].all()


# --------------------------------------------------------------------------------------
# Problemas de calidad deliberados
# --------------------------------------------------------------------------------------
def test_estan_los_problemas_de_calidad_deliberados(dataset_dir: Path) -> None:
    """El ETL de la Fase 2 tiene que tener algo que limpiar en cada frente."""
    tables = vd.load_tables(dataset_dir)
    q = vd.quality_issues(tables)

    assert 0.5 < q["lineas_duplicadas_pct"] < 4.0
    assert 0.1 < q["cantidad_negativa_pct"] < 2.0
    assert q["categorias_mal_escritas_pct"] > 1.0
    assert q["ciudad_nula_pct"] > 0.0
    assert q["cestas_anonimas_pct"] > 1.0
    assert q["outliers_importe_pct"] > 0.0
    assert q["alta_posterior_a_primera_compra"] > 0


def test_las_categorias_sucias_se_normalizan_a_las_del_catalogo(dataset_dir: Path) -> None:
    """Toda categoria ensuciada tiene que poder recuperarse: no se inventan valores."""
    tables = vd.load_tables(dataset_dir)
    assert tables["products"]["category_clean"].notna().all()


def test_las_compras_anonimas_no_son_un_error(dataset: dict) -> None:
    """Los nulos de baskets.customer_id son legitimos y llevan lineas de verdad."""
    b = dataset["baskets"]
    anon = b.loc[b["customer_id"].isna(), "basket_id"]
    assert len(anon) > 0
    items = dataset["basket_items"]
    assert items["basket_id"].isin(anon).any()


def test_el_diccionario_de_datos_se_autogenera(dataset_dir: Path) -> None:
    """Fase 1 del ROADMAP: diccionario de datos generado junto al dataset."""
    readme = (dataset_dir / "README.md").read_text(encoding="utf-8")
    for table in TABLE_ORDER:
        assert f"`{table}`" in readme
    assert (dataset_dir / "manifest.json").exists()

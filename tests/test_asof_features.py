"""Tests del punto A1: historial del cliente as-of el dia de cada cesta.

* **Corte**: una cesta ve las del mismo cliente con dia estrictamente anterior, tambien
  las de dentro de la ventana, y nunca las del mismo dia ni posteriores.
* **Paridad**: Spark (`history.asof_history`) y pandas (`serving.asof_history`) dan las
  mismas tablas sobre los mismos eventos.
* **Formulas compartidas**: el intervalo esperado as-of coincide con el de
  `repurchase_features` (Tarea 2) a la misma fecha, y la fuente `hist` deja de dar por
  vencida una categoria recien repuesta.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.recommender import formulas as fx
from src.recommender.config import CandidateConfig
from src.serving import recommend as srv
from tests.helpers import spark_df

PRODUCTS = pd.DataFrame(
    {
        "product_id": ["L1", "L2", "P1"],
        "category": ["leche", "leche", "pan"],
        "typical_repurchase_days": [7, 7, 3],
    }
)
CUSTOMERS = pd.DataFrame({"customer_id": ["C1", "C2"], "household_size_est": [2, None]})

# (basket_id, customer_id, dia, importe, [(producto, unidades)])
BASKETS = [
    ("H1", "C1", "2025-10-01", 10.0, [("L1", 2), ("P1", 1)]),
    ("H2", "C1", "2025-10-15", 20.0, [("L1", 1)]),
    ("H3", "C1", "2025-12-05", 5.0, [("L2", 1)]),  # dentro de la ventana de test
    ("H4", "C1", "2025-12-20", 7.0, [("P1", 1)]),  # mismo dia que Q1: invisible
    ("H5", "C2", "2025-12-01", 3.0, [("P1", 1)]),  # mismo dia que Q4: invisible
    ("A1", None, "2025-12-01", 4.0, [("L1", 1)]),  # anonima: no es historial de nadie
]
QUERIES = pd.DataFrame(
    {
        "basket_id": ["Q1", "Q2", "Q3", "Q4"],
        "customer_id": ["C1", "C1", None, "C2"],
        "basket_day": pd.to_datetime(["2025-12-20", "2025-11-01", "2025-12-20", "2025-12-01"]),
    }
)

LINES = pd.DataFrame(
    [
        {"basket_id": b, "product_id": p, "quantity": q}
        for b, _, _, _, items in BASKETS
        for p, q in items
    ]
)
HEADERS = pd.DataFrame(
    [
        {"basket_id": b, "customer_id": c, "basket_day": pd.Timestamp(d), "total_amount": t}
        for b, c, d, t, _ in BASKETS
    ]
)


def _bundle() -> srv.ServingBundle:
    identified = HEADERS.loc[HEADERS["customer_id"].notna()]
    lines = LINES.merge(identified[["basket_id", "customer_id", "basket_day"]], on="basket_id")
    lines_idx, baskets_idx = srv.index_customer_events(lines, identified, PRODUCTS)
    empty = pd.DataFrame()
    return srv.ServingBundle(
        popularity=empty,
        affinity_product=empty,
        affinity_category=empty,
        category_leaders=empty,
        customer_stats=empty,
        category_repurchase_days=PRODUCTS.groupby("category", as_index=False)[
            "typical_repurchase_days"
        ].first(),
        known_customers=empty,
        als_topn=empty,
        products=PRODUCTS,
        customers=CUSTOMERS,
        promotions=empty,
        booster=None,
        window_start=dt.date(2025, 11, 1),
        cfg=CandidateConfig(),
        lines_by_customer=lines_idx,
        baskets_by_customer=baskets_idx,
    )


@pytest.fixture(scope="module")
def pandas_history() -> srv.AsOfHistory:
    return srv.asof_history(QUERIES, _bundle())


def _row(frame: pd.DataFrame, **keys) -> pd.Series:
    mask = np.logical_and.reduce([frame[k] == v for k, v in keys.items()])
    rows = frame.loc[mask]
    assert len(rows) == 1, keys
    return rows.iloc[0]


# --------------------------------------------------------------------------------------
# Corte temporal
# --------------------------------------------------------------------------------------
def test_la_query_ve_las_compras_de_dentro_de_la_ventana(pandas_history) -> None:
    cust = _row(pandas_history.customer, basket_id="Q1")
    assert cust["cust_frequency"] == 3.0  # H1, H2 y H3; H4 es del mismo dia
    assert cust["cust_avg_ticket"] == pytest.approx(35.0 / 3)
    assert pd.Timestamp(cust["cust_last_day"]) == pd.Timestamp("2025-12-05")
    assert cust["cust_n_products"] == 3.0

    leche = _row(pandas_history.categories, basket_id="Q1", category="leche")
    assert pd.Timestamp(leche["cat_last_day"]) == pd.Timestamp("2025-12-05")
    assert leche["cat_n_purchase_days"] == 3.0


def test_nada_del_mismo_dia_ni_posterior(pandas_history) -> None:
    # Q1 (20-dic) no ve H4 (20-dic): el pan sigue con su unica compra de octubre.
    pan = _row(pandas_history.categories, basket_id="Q1", category="pan")
    assert pd.Timestamp(pan["cat_last_day"]) == pd.Timestamp("2025-10-01")
    # Q2 (1-nov) no ve H3 (5-dic).
    assert set(pandas_history.products.loc[
        pandas_history.products["basket_id"] == "Q2", "product_id"
    ]) == {"L1", "P1"}
    # Q4 (1-dic) no ve H5 (1-dic) y Q3 es anonima: ninguna tiene historial.
    for frame in (pandas_history.products, pandas_history.categories, pandas_history.customer):
        assert not frame["basket_id"].isin(["Q3", "Q4"]).any()


def test_agregados_cliente_producto(pandas_history) -> None:
    l1 = _row(pandas_history.products, basket_id="Q1", product_id="L1")
    assert (l1["hist_n_baskets"], l1["hist_units"]) == (2.0, 3.0)
    assert pd.Timestamp(l1["hist_last_day"]) == pd.Timestamp("2025-10-15")


def test_intervalo_esperado(pandas_history) -> None:
    # leche en Q1: dias 1-oct, 15-oct y 5-dic -> 65 dias en 2 huecos.
    leche = _row(pandas_history.categories, basket_id="Q1", category="leche")
    assert leche["cat_expected_days"] == pytest.approx(32.5)
    # pan con una sola compra: tipico 3 x hogar de 2 (1,45 - 0,125 x 2 = 1,2).
    pan = _row(pandas_history.categories, basket_id="Q1", category="pan")
    assert pan["cat_expected_days"] == pytest.approx(3.6)


# --------------------------------------------------------------------------------------
# Paridad Spark / pandas y coherencia con la Tarea 2
# --------------------------------------------------------------------------------------
def _spark_tables(spark):
    products = spark_df(
        spark,
        PRODUCTS.to_dict("records"),
        "product_id string, category string, typical_repurchase_days int",
    )
    customers = spark_df(
        spark, CUSTOMERS.to_dict("records"), "customer_id string, household_size_est int"
    )
    headers = HEADERS.assign(
        basket_day=HEADERS["basket_day"].dt.date,
        basket_date=HEADERS["basket_day"],
    )
    baskets = spark_df(
        spark,
        headers.to_dict("records"),
        "basket_id string, customer_id string, basket_day date, basket_date timestamp, "
        "total_amount double",
    )
    items = spark_df(
        spark, LINES.to_dict("records"), "basket_id string, product_id string, quantity int"
    )
    queries = spark_df(
        spark,
        QUERIES.assign(basket_day=QUERIES["basket_day"].dt.date).to_dict("records"),
        "basket_id string, customer_id string, basket_day date",
    )
    return products, customers, baskets, items, queries


def _normalise(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column.endswith("_day"):
            out[column] = pd.to_datetime(out[column])
    return out.sort_values(keys).reset_index(drop=True)[sorted(out.columns)]


def test_spark_y_pandas_dan_el_mismo_historial(spark, pandas_history) -> None:
    from src.recommender import history as hs

    products, customers, baskets, items, queries = _spark_tables(spark)
    got = hs.asof_history(queries, baskets, items, products, customers)

    for name, keys in (
        ("products", ["basket_id", "product_id"]),
        ("categories", ["basket_id", "category"]),
        ("customer", ["basket_id"]),
    ):
        spark_side = _normalise(getattr(got, name).toPandas(), keys)
        pandas_side = _normalise(getattr(pandas_history, name), keys)
        pd.testing.assert_frame_equal(
            spark_side, pandas_side, check_dtype=False, obj=name, rtol=0, atol=0
        )


def test_el_intervalo_as_of_es_el_de_la_tarea_2(spark) -> None:
    """A la misma fecha, `repurchase_features` y el as-of dan el mismo intervalo."""
    from src.etl.repurchase import repurchase_features
    from src.recommender import history as hs

    products, customers, baskets, items, queries = _spark_tables(spark)
    day = "2025-11-01"
    before = baskets.filter(f"basket_day < '{day}'")
    task2 = (
        repurchase_features(
            items.join(before.select("basket_id"), "basket_id"),
            before,
            products,
            customers,
            reference_date="2025-10-31",
        )
        .select("customer_id", "category", "expected_repurchase_days")
        .toPandas()
    )
    asof = hs.asof_history(
        queries.filter("basket_id = 'Q2'"), baskets, items, products, customers
    ).categories.toPandas()
    merged = asof.merge(task2.assign(basket_id="Q2"), on=["basket_id", "category"])
    assert len(merged) == 2
    np.testing.assert_allclose(merged["cat_expected_days"], merged["expected_repurchase_days"])


def test_la_fuente_hist_no_da_por_vencido_lo_recien_repuesto(pandas_history) -> None:
    """Con el historial congelado, la leche de Q1 pareceria vencida; as-of, no."""
    q1 = QUERIES.loc[QUERIES["basket_id"] == "Q1"]
    ranked = srv.candidates_personal(q1, _bundle(), pandas_history)
    # pan: 80 dias sin comprar con un ciclo de 3,6 -> vencido, 1 x 2 + 22,2.
    # leche: repuesta hace 15 dias con un ciclo de 32,5 -> no vencida, 2 + 0,46 y 1 + 0,46.
    assert list(ranked.sort_values("hist_rank")["product_id"]) == ["P1", "L1", "L2"]

    leche = _row(pandas_history.categories, basket_id="Q1", category="leche")
    ratio_asof = fx.overdue_ratio(15.0, leche["cat_expected_days"])  # 20-dic - 5-dic
    # Foto del 1-nov: ultima leche el 15-oct y ciclo de 14 dias -> 66 dias de retraso.
    ratio_frozen = fx.overdue_ratio(66.0, 14.0)
    assert ratio_asof < 1.0 < ratio_frozen


# --------------------------------------------------------------------------------------
# Formulas
# --------------------------------------------------------------------------------------
def test_formulas_con_nulos_como_spark() -> None:
    ops = fx.PANDAS_OPS
    n = pd.Series([1, 2, 3])
    gap = fx.mean_gap_days(n, pd.Series([0, 10, 30]), ops)
    assert np.isnan(gap[0]) and gap[1] == 10.0 and gap[2] == 15.0

    factor = fx.household_factor(pd.Series([None, 6.0, 1.0]), ops)
    np.testing.assert_allclose(factor, [1.325, 0.70, 1.325])

    expected = fx.expected_repurchase_days(n, gap, pd.Series([1.0, 7.0, 7.0]), factor, ops)
    # Una compra: tipico ajustado con suelo de 2 dias. Con dos o mas: el observado.
    np.testing.assert_allclose(expected, [2.0, 10.0, 15.0])

    due = fx.is_due(pd.Series([np.nan, 0.99, 1.0]), ops)
    assert due.tolist() == [0.0, 0.0, 1.0]

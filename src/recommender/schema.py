"""Nombres y esquemas del recomendador, sin dependencias pesadas.

Todo lo que hay aqui son constantes puras: los nombres de las fuentes de candidatos, el
orden de las columnas que ve LightGBM y las etiquetas de los cuatro perfiles.

Vive en su propio modulo, y no dentro de `candidates.py` o `features.py`, porque esos dos
importan PySpark en la primera linea. La demo de la Fase 6b sirve con pandas y sin Spark
(`src/serving/`), y necesita exactamente estas mismas constantes: si las copiase, el dia
que alguien anada una feature al ranker la demo seguiria presentando la lista antigua y
el modelo la leeria mal, en silencio.

`FEATURE_COLUMNS` es parte del contrato del modelo guardado: al predecir hay que
presentar exactamente estas columnas y en este orden.
"""

from __future__ import annotations

# Los cuatro perfiles de cliente de `CHALLENGE.md`.
PROFILE_LABELS: dict[int, str] = {
    1: "1 - nuevo, carrito vacio",
    2: "2 - nuevo, con articulos",
    3: "3 - recurrente, carrito vacio",
    4: "4 - recurrente, con articulos",
}

# Columnas de score que aporta cada fuente de candidatos.
SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "pop": ("pop_score", "pop_rank"),
    "aff": ("aff_lift_max", "aff_conf_sum", "aff_n_support"),
    "cataff": ("cataff_lift_max", "cataff_conf_max"),
    # El historial aporta solo su puesto: cuantas veces y cuando lo compro el cliente son
    # *features* que se cruzan con todos los candidatos, no solo con los que propuso esta
    # fuente.
    "hist": ("hist_rank",),
    "als": ("als_score", "als_rank"),
}
SOURCE_NAMES: tuple[str, ...] = tuple(SOURCE_COLUMNS)

SOURCE_FEATURES: tuple[str, ...] = tuple(
    [c for cols in SOURCE_COLUMNS.values() for c in cols]
    + [f"src_{s}" for s in SOURCE_NAMES]
    + ["n_sources"]
)

CUSTOMER_PRODUCT_FEATURES: tuple[str, ...] = (
    "hist_n_baskets",
    "hist_units",
    "hist_days_since",
    "hist_ever_bought",
)

CUSTOMER_CATEGORY_FEATURES: tuple[str, ...] = (
    "cat_n_purchase_days",
    "cat_days_since",
    "cat_expected_days",
    "cat_overdue_ratio",
    "cat_due",
)

PRODUCT_FEATURES: tuple[str, ...] = (
    "prod_pop_all",
    "prod_pop_recent",
    "prod_seasonal_index",
    "prod_month_rank",
    "unit_price",
    "pack_size",
    "typical_repurchase_days",
    "is_private_label",
    "is_perishable",
    "department_idx",
    "category_idx",
)

CONTEXT_FEATURES: tuple[str, ...] = (
    "prefix_size",
    "basket_month",
    "basket_dow",
    "channel_idx",
    "is_known_customer",
    "cust_frequency",
    "cust_recency_days",
    "cust_avg_ticket",
    "cust_n_products",
    "household_size_est",
    "loyalty_idx",
    "is_on_promo",
    "promo_discount",
)

SESSION_FEATURES: tuple[str, ...] = (
    "has_session",
    "sess_viewed",
    "sess_n_views",
    "sess_secs_since_view",
    "sess_n_events_before",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    SOURCE_FEATURES
    + CUSTOMER_PRODUCT_FEATURES
    + CUSTOMER_CATEGORY_FEATURES
    + PRODUCT_FEATURES
    + CONTEXT_FEATURES
    + SESSION_FEATURES
)

# Las que LightGBM debe tratar como categoricas y no como numeros ordenados.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "channel_idx",
    "loyalty_idx",
    "department_idx",
    "category_idx",
    "basket_month",
    "basket_dow",
)

# Dominios cerrados. Lo desconocido (y lo nulo: una cesta anonima no tiene `loyalty_tier`)
# recibe el codigo `len(values)`, no -1: LightGBM convierte los negativos de una feature
# categorica en NaN y avisa, mientras que con un codigo propio "sin dato" es una categoria
# mas y el arbol puede separarla.
CHANNELS: tuple[str, ...] = ("app", "web", "store")
LOYALTY: tuple[str, ...] = ("bronze", "silver", "gold")

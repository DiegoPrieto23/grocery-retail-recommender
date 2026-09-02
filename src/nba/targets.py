"""Construccion de las dos etiquetas de propension en un corte temporal.

Un **corte** parte el mundo en dos: lo anterior es lo unico que puede mirar una feature, y
lo posterior es lo unico que puede definir una etiqueta. Todas las funciones de este modulo
reciben la fecha de corte y no devuelven nunca una columna que mezcle los dos lados.

## Por que no se usa `customers.churn_label` como target

`DATA_SPEC.md` define `churn_label` como "no ha comprado en los ultimos 60 dias del periodo
simulado", es decir, respecto al **final del dataset** (2025-12-31). Es una etiqueta que
solo existe una vez y en un instante concreto: usarla como target en un corte de abril
seria pedirle al modelo que adivine algo definido ocho meses despues, y usarla como
*feature* seria fuga pura, porque se calcula con compras futuras.

Aqui el churn se construye de forma **observacional en cada corte**: un cliente activo
antes del corte que no compra en las 4 semanas siguientes. Eso si es aprendible y si se
puede evaluar en un periodo posterior. `churn_label` se reserva para una comprobacion de
cordura al final (`agreement_with_churn_label`): en el corte de test las dos definiciones
deben parecerse mucho, porque miran ventanas casi solapadas.
"""

from __future__ import annotations

import datetime as dt

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


def _day(value: dt.date) -> Column:
    return F.lit(value.isoformat()).cast("date")


def baskets_in_window(
    baskets: DataFrame, start: dt.date | None, end: dt.date | None
) -> DataFrame:
    """Cestas identificadas con `start <= basket_day < end`. Los bordes admiten None."""
    out = baskets.filter(F.col("customer_id").isNotNull())
    if start is not None:
        out = out.filter(F.col("basket_day") >= _day(start))
    if end is not None:
        out = out.filter(F.col("basket_day") < _day(end))
    return out


def category_lines(
    baskets: DataFrame, basket_items: DataFrame, products: DataFrame
) -> DataFrame:
    """Lineas de ticket con su categoria y departamento, ya unidas al cliente y al dia.

    Arrastra tambien `product_id`, `quantity` e `is_promo`: son las columnas con las que
    `features` calcula variedad de surtido, unidades y exposicion a promocion, y volver a
    unir `basket_items` mas tarde solo para eso costaria otro shuffle sobre 3,1 M de lineas.
    """
    return (
        basket_items.join(
            baskets.select("basket_id", "customer_id", "basket_day"), "basket_id", "inner"
        )
        .join(products.select("product_id", "category", "department"), "product_id", "inner")
        .select(
            "customer_id",
            "basket_day",
            "product_id",
            "category",
            "department",
            "quantity",
            "line_amount",
            "is_promo",
        )
    )


def active_customers(baskets: DataFrame, cutoff: dt.date) -> DataFrame:
    """Clientes con al menos una compra antes del corte.

    Es el universo sobre el que tiene sentido preguntarse si va a abandonar: de un cliente
    que nunca ha comprado no se puede decir que se este yendo.
    """
    return (
        baskets_in_window(baskets, None, cutoff)
        .select("customer_id")
        .distinct()
    )


def candidate_pairs(
    lines: DataFrame, cutoff: dt.date, *, lookback_days: int
) -> DataFrame:
    """Pares `(customer_id, category)` sobre los que se puede actuar en el corte.

    Se restringe a categorias compradas en los `lookback_days` previos. Sin ese filtro el
    grano seria 20.000 clientes x 62 categorias = 1,2 M de filas por corte, casi todas
    negativas y ninguna accionable: mandar un cupon de comida para gatos a quien no tiene
    gato no es una siguiente mejor accion, es ruido caro.
    """
    start = cutoff - dt.timedelta(days=lookback_days)
    return (
        baskets_in_window(lines, start, cutoff)
        .select("customer_id", "category")
        .distinct()
    )


def category_label(
    lines: DataFrame, cutoff: dt.date, *, horizon_days: int
) -> DataFrame:
    """Pares `(customer_id, category)` comprados en la ventana futura del corte."""
    end = cutoff + dt.timedelta(days=horizon_days)
    return (
        baskets_in_window(lines, cutoff, end)
        .select("customer_id", "category")
        .distinct()
        .withColumn("label", F.lit(1).cast("byte"))
    )


def churn_label(
    baskets: DataFrame, cutoff: dt.date, *, horizon_days: int
) -> DataFrame:
    """Por cliente activo: 1 si **no** compra en la ventana futura.

    El signo importa. La clase positiva es el abandono, que es el evento raro y el que
    tiene coste; asi el PR-AUC mide lo que interesa y no la facilidad de acertar que la
    mayoria sigue comprando.
    """
    end = cutoff + dt.timedelta(days=horizon_days)
    active = active_customers(baskets, cutoff)
    future = (
        baskets_in_window(baskets, cutoff, end)
        .select("customer_id")
        .distinct()
        .withColumn("_bought", F.lit(True))
    )
    return (
        active.join(future, "customer_id", "left")
        .withColumn("churn", F.when(F.col("_bought").isNotNull(), 0).otherwise(1).cast("byte"))
        .drop("_bought")
    )


def build_labels(
    baskets: DataFrame,
    lines: DataFrame,
    cutoff: dt.date,
    *,
    category_horizon_days: int,
    churn_horizon_days: int,
    lookback_days: int,
) -> tuple[DataFrame, DataFrame]:
    """Las dos tablas etiquetadas de un corte.

    Returns:
        `(category, churn)`. La primera tiene una fila por par candidato con `label` 0/1;
        la segunda una fila por cliente activo con `churn` 0/1. Ambas llevan `cutoff` como
        columna, para poder apilar varios cortes en una sola matriz de entrenamiento.
    """
    pairs = candidate_pairs(lines, cutoff, lookback_days=lookback_days)
    positives = category_label(lines, cutoff, horizon_days=category_horizon_days)
    category = (
        pairs.join(positives, ["customer_id", "category"], "left")
        .withColumn("label", F.coalesce(F.col("label"), F.lit(0).cast("byte")))
        .withColumn("cutoff", _day(cutoff))
    )
    churn = churn_label(baskets, cutoff, horizon_days=churn_horizon_days).withColumn(
        "cutoff", _day(cutoff)
    )
    return category, churn


def agreement_with_churn_label(
    churn: DataFrame, customers: DataFrame
) -> dict[str, float]:
    """Compara el churn observacional del corte con `customers.churn_label`.

    No es una metrica del modelo: es una comprobacion de que la etiqueta construida aqui
    describe el mismo fenomeno que la del generador. En el corte de test las dos ventanas
    casi coinciden, asi que un acuerdo alto confirma que no se ha invertido el signo ni se
    ha colado un desfase de fechas.
    """
    joined = churn.join(
        customers.select("customer_id", F.col("churn_label").cast("byte").alias("spec_label")),
        "customer_id",
        "inner",
    )
    agg = joined.agg(
        F.count("*").alias("n"),
        F.avg(F.col("churn").cast("double")).alias("rate_observed"),
        F.avg(F.col("spec_label").cast("double")).alias("rate_spec"),
        F.avg((F.col("churn") == F.col("spec_label")).cast("double")).alias("agreement"),
    ).collect()[0]
    return {
        "n": int(agg["n"]),
        "rate_observed": float(agg["rate_observed"]),
        "rate_spec": float(agg["rate_spec"]),
        "agreement": float(agg["agreement"]),
    }

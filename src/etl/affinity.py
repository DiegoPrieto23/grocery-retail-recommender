"""Afinidad de cesta: que se compra junto con que.

Es la tabla intermedia que en la Fase 3 alimenta la fuente de candidatos de co-compra del
recomendador, y a la vez la forma de verificar que los 10 pares de `DATA_SPEC.md` han
quedado realmente grabados en el dato.

Se calculan dos granos, porque sirven para cosas distintas:

- **Categoria x categoria**: pocos pares, mucha senal por par. Es el grano en el que estan
  escritas las reglas de negocio de `DATA_SPEC.md` (cerveza -> snacks) y el que se lee bien
  en el EDA.
- **Producto x producto**: el grano que necesita el recomendador para proponer un SKU
  concreto. Mas ruidoso, asi que exige un minimo de cestas de soporte.

## Metricas

Para un par ordenado (A -> B), sobre `N` cestas:

    support     = cestas con A y B  / N
    confidence  = cestas con A y B  / cestas con A     = P(B | A)
    lift        = confidence / (cestas con B / N)      = P(B|A) / P(B)
    jaccard     = cestas con A y B / cestas con A o B

`lift` es la metrica de `DATA_SPEC.md`: cuanto mas probable es B cuando ya hay A, frente a
lo que cabria esperar por azar. Un lift de 3.0 significa "tres veces mas probable".

El par se guarda **ordenado**: (A -> B) y (B -> A) comparten `support`, `lift` y `jaccard`
pero no `confidence`, y el recomendador consulta siempre por el antecedente.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Minimo de cestas en las que un par debe coincidir para que su lift sea creible. Con
# menos, el ratio lo domina el ruido: 2 coincidencias sobre 3 cestas dan un lift enorme.
DEFAULT_MIN_PAIR_BASKETS = 50


def basket_item_sets(
    basket_items: DataFrame,
    baskets: DataFrame | None = None,
    products: DataFrame | None = None,
    *,
    level: str = "category",
) -> DataFrame:
    """Reduce las lineas de ticket a pares `(basket_id, item)` unicos.

    Args:
        basket_items: Lineas limpias.
        baskets: Solo se usa para filtrar cestas, si se quiere; puede omitirse.
        products: Catalogo. Obligatorio si `level` es `"category"`.
        level: `"category"` o `"product"`.

    Returns:
        DataFrame con `basket_id` e `item`, sin repetidos.
    """
    if level not in {"category", "product"}:
        raise ValueError(f"level debe ser 'category' o 'product', no {level!r}")

    if level == "product":
        pairs = basket_items.select("basket_id", F.col("product_id").alias("item"))
    else:
        if products is None:
            raise ValueError("level='category' requiere pasar `products`")
        pairs = basket_items.select("basket_id", "product_id").join(
            F.broadcast(products.select("product_id", F.col("category").alias("item"))),
            "product_id",
        ).select("basket_id", "item")

    if baskets is not None:
        pairs = pairs.join(baskets.select("basket_id"), "basket_id")
    return pairs.distinct()


def cooccurrence_affinity(
    basket_items: DataFrame,
    products: DataFrame | None = None,
    *,
    level: str = "category",
    min_pair_baskets: int = DEFAULT_MIN_PAIR_BASKETS,
    top_n: int | None = None,
    baskets: DataFrame | None = None,
) -> DataFrame:
    """Tabla de afinidad por co-ocurrencia exacta.

    Args:
        basket_items: Lineas de ticket limpias.
        products: Catalogo, obligatorio con `level="category"`.
        level: Grano del calculo: `"category"` o `"product"`.
        min_pair_baskets: Cestas minimas en las que debe aparecer el par.
        top_n: Si se indica, deja solo los `top_n` mejores consecuentes por antecedente,
            ordenados por lift. Es lo que consume el generador de candidatos de la Fase 3.
        baskets: Opcional, para restringir el calculo a un subconjunto de cestas (por
            ejemplo, solo el split de train del recomendador).

    Returns:
        DataFrame con `antecedent`, `consequent`, `n_baskets_antecedent`,
        `n_baskets_consequent`, `n_baskets_both`, `support`, `confidence`, `lift` y
        `jaccard`, ordenado por lift descendente.
    """
    sets = basket_item_sets(basket_items, baskets, products, level=level).cache()

    total_baskets = sets.select("basket_id").distinct().count()
    if total_baskets == 0:
        raise ValueError("No hay cestas con las que calcular afinidad")

    singles = sets.groupBy("item").agg(F.count(F.lit(1)).cast("long").alias("n_baskets"))

    # Auto-join por cesta. `<` en vez de `!=` genera cada par una sola vez; el orden
    # inverso se anade despues con un union, que es la mitad de trabajo de shuffle.
    left = sets.select("basket_id", F.col("item").alias("a"))
    right = sets.select("basket_id", F.col("item").alias("b"))
    pair_counts = (
        left.join(right, "basket_id")
        .filter(F.col("a") < F.col("b"))
        .groupBy("a", "b")
        .agg(F.count(F.lit(1)).cast("long").alias("n_baskets_both"))
        .filter(F.col("n_baskets_both") >= min_pair_baskets)
    )

    both_directions = pair_counts.select(
        F.col("a").alias("antecedent"), F.col("b").alias("consequent"), "n_baskets_both"
    ).unionByName(
        pair_counts.select(
            F.col("b").alias("antecedent"), F.col("a").alias("consequent"), "n_baskets_both"
        )
    )

    n = F.lit(float(total_baskets))
    enriched = (
        both_directions.join(
            F.broadcast(
                singles.select(
                    F.col("item").alias("antecedent"),
                    F.col("n_baskets").alias("n_baskets_antecedent"),
                )
            ),
            "antecedent",
        )
        .join(
            F.broadcast(
                singles.select(
                    F.col("item").alias("consequent"),
                    F.col("n_baskets").alias("n_baskets_consequent"),
                )
            ),
            "consequent",
        )
        .withColumn("total_baskets", F.lit(total_baskets))
        .withColumn("support", F.round(F.col("n_baskets_both") / n, 6))
        .withColumn(
            "confidence", F.round(F.col("n_baskets_both") / F.col("n_baskets_antecedent"), 6)
        )
        .withColumn(
            "lift",
            F.round(
                (F.col("n_baskets_both") / F.col("n_baskets_antecedent"))
                / (F.col("n_baskets_consequent") / n),
                4,
            ),
        )
        .withColumn(
            "jaccard",
            F.round(
                F.col("n_baskets_both")
                / (
                    F.col("n_baskets_antecedent")
                    + F.col("n_baskets_consequent")
                    - F.col("n_baskets_both")
                ),
                6,
            ),
        )
    )

    if top_n is not None:
        ranked = Window.partitionBy("antecedent").orderBy(
            F.col("lift").desc(), F.col("n_baskets_both").desc(), F.col("consequent").asc()
        )
        enriched = (
            enriched.withColumn("rank", F.row_number().over(ranked))
            .filter(F.col("rank") <= top_n)
        )

    return enriched.select(
        "antecedent",
        "consequent",
        "total_baskets",
        "n_baskets_antecedent",
        "n_baskets_consequent",
        "n_baskets_both",
        "support",
        "confidence",
        "lift",
        "jaccard",
        *(["rank"] if top_n is not None else []),
    ).orderBy(F.col("lift").desc())


def fpgrowth_rules(
    basket_items: DataFrame,
    products: DataFrame | None = None,
    *,
    level: str = "category",
    min_support: float = 0.01,
    min_confidence: float = 0.10,
) -> tuple[DataFrame, DataFrame]:
    """Reglas de asociacion con FP-Growth (Spark MLlib), como contraste de la co-ocurrencia.

    FP-Growth encuentra conjuntos de cualquier tamano, no solo pares: sirve para ver si hay
    trios reales (`pasta + salsa + queso`) que la tabla de pares no captura. A cambio, poda
    por soporte global, asi que las categorias raras pero muy correlacionadas (turron, cava)
    desaparecen. Por eso el recomendador se apoya en la co-ocurrencia y esto es contraste.

    Returns:
        `(freq_itemsets, rules)`. `rules` trae `antecedent`, `consequent`, `confidence`,
        `lift` y `support`.
    """
    from pyspark.ml.fpm import FPGrowth

    sets = basket_item_sets(basket_items, None, products, level=level)
    transactions = sets.groupBy("basket_id").agg(F.collect_set("item").alias("items"))

    model = FPGrowth(
        itemsCol="items", minSupport=min_support, minConfidence=min_confidence
    ).fit(transactions)
    return model.freqItemsets, model.associationRules


def expected_pairs_report(
    affinity: DataFrame, expected: list[tuple[str, str, float]]
) -> DataFrame:
    """Compara el lift medido de una lista de pares con su objetivo.

    Sirve para verificar de un vistazo que los 10 pares de `DATA_SPEC.md` estan en el dato
    con la fuerza esperada, sin tener que leer la tabla entera.

    Args:
        affinity: Salida de `cooccurrence_affinity` a nivel de categoria.
        expected: Tripletas `(disparadora, asociada, lift objetivo)`.

    Returns:
        DataFrame con `antecedent`, `consequent`, `target_lift`, `lift`, `support`,
        `confidence` y `ratio_vs_target`, en el mismo orden en que se pasaron.
    """
    spark = affinity.sparkSession
    import pandas as pd

    wanted = spark.createDataFrame(
        pd.DataFrame(
            {
                "antecedent": [a for a, _, _ in expected],
                "consequent": [b for _, b, _ in expected],
                "target_lift": [float(t) for _, _, t in expected],
                "position": list(range(len(expected))),
            }
        )
    )
    return (
        wanted.join(
            affinity.select("antecedent", "consequent", "lift", "support", "confidence"),
            ["antecedent", "consequent"],
            "left",
        )
        .withColumn("ratio_vs_target", F.round(F.col("lift") / F.col("target_lift"), 3))
        .orderBy("position")
        .drop("position")
    )

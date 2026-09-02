"""Limpieza de las 7 tablas crudas (Fase 2 del ROADMAP).

Cada regla corrige uno de los problemas de calidad que `DATA_SPEC.md` inyecta a proposito
en la Fase 1. Las decisiones y su porque estan en `docs/CLEANING.md`; aqui se resumen las
tres que no son mecanicas:

1. **Corregir el signo ANTES de deduplicar.** El generador duplica lineas de ticket y,
   despues, invierte el signo de una muestra de `quantity`. Cuando la linea invertida es
   la copia de una duplicada, el par (+2, -2) ya no es un duplicado exacto y sobrevive a
   `dropDuplicates`. Deduplicando primero quedan 353 lineas fantasma; corrigiendo el signo
   primero, `(basket_id, product_id)` queda unico sin excepciones.

2. **`total_amount` se recalcula desde las lineas limpias**, como pide `DATA_SPEC.md`. Eso
   arregla de un golpe los tres motivos por los que la cabecera no cuadraba: duplicados,
   cantidades negativas y los outliers de importe inyectados.

3. **Los nulos de `baskets.customer_id` NO se tocan**: son compras anonimas legitimas, no
   un defecto. Descartarlas sesgaria a la baja la venta de tienda fisica.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# --------------------------------------------------------------------------------------
# Configuracion e informe
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CleaningConfig:
    """Parametros de la limpieza.

    Attributes:
        negative_quantity_policy: Que hacer con `quantity < 0`. `"abs"` corrige el signo
            (la evidencia dice que son ventas mal firmadas, ver `docs/CLEANING.md`);
            `"drop"` las elimina, tratandolas como devoluciones reales.
        missing_city_label: Etiqueta con la que se rellena `customers.city` nula.
        missing_brand_label: Etiqueta con la que se rellena `products.brand` nula.
        amount_outlier_ratio: Una cabecera cuyo importe supere este multiplo del importe
            recalculado se marca como outlier. El generador multiplica los outliers por
            20-60x, asi que cualquier corte entre 1.5 y 10 aisla exactamente los mismos.
        detailed_counts: Calcular las metricas del informe que exigen una pasada extra
            sobre `basket_items` (util para documentar, prescindible en tests).
    """

    negative_quantity_policy: str = "abs"
    missing_city_label: str = "Desconocida"
    missing_brand_label: str = "Sin marca"
    amount_outlier_ratio: float = 1.5
    detailed_counts: bool = True

    def __post_init__(self) -> None:
        if self.negative_quantity_policy not in {"abs", "drop"}:
            raise ValueError(
                "negative_quantity_policy debe ser 'abs' o 'drop', "
                f"no {self.negative_quantity_policy!r}"
            )
        if self.amount_outlier_ratio <= 1.0:
            raise ValueError("amount_outlier_ratio debe ser > 1.0")


@dataclass(frozen=True)
class CleaningStep:
    """Una regla de limpieza aplicada y su efecto medido."""

    table: str
    issue: str
    rule: str
    rows_affected: int
    rows_before: int
    rows_after: int

    @property
    def pct_affected(self) -> float:
        return 100.0 * self.rows_affected / self.rows_before if self.rows_before else 0.0


@dataclass
class CleaningReport:
    """Traza completa de la limpieza: que se toco, cuanto y con que regla."""

    steps: list[CleaningStep] = field(default_factory=list)
    category_fixes: dict[str, str] = field(default_factory=dict)

    def add(self, step: CleaningStep) -> None:
        self.steps.append(step)

    def rows_affected(self, table: str, issue: str) -> int:
        """Filas afectadas por una regla concreta. Pensado para los tests."""
        for step in self.steps:
            if step.table == table and step.issue == issue:
                return step.rows_affected
        raise KeyError(f"No hay paso {issue!r} para la tabla {table!r}")

    def to_markdown(self) -> str:
        """Informe en Markdown, listo para `reports/etl/`."""

        def n(value: int) -> str:
            """Separador de miles a la espanola, sin tocar el resto del texto."""
            return f"{value:,}".replace(",", ".")

        lines = [
            "| Tabla | Problema | Regla aplicada | Filas afectadas | % | Filas antes | Filas despues |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for s in self.steps:
            lines.append(
                f"| `{s.table}` | {s.issue} | {s.rule} | {n(s.rows_affected)} | "
                f"{s.pct_affected:.3f} % | {n(s.rows_before)} | {n(s.rows_after)} |"
            )
        if self.category_fixes:
            lines += [
                "",
                f"### Categorias normalizadas ({len(self.category_fixes)} grafias corregidas)",
                "",
                "| Grafia original | Canonica |",
                "| --- | --- |",
            ]
            for raw, canonical in sorted(self.category_fixes.items()):
                lines.append(f"| `{raw}` | `{canonical}` |")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------


def normalized_category(col: Column | str = "category") -> Column:
    """Clave de agrupacion de una categoria: sin mayusculas ni espacios sobrantes."""
    c = F.col(col) if isinstance(col, str) else col
    return F.lower(F.trim(F.regexp_replace(c, r"\s+", " ")))


def canonical_category_map(products: DataFrame) -> DataFrame:
    """Mapa `clave normalizada -> grafia canonica` deducido de los propios datos.

    La grafia canonica de cada categoria es la **mas frecuente** entre las variantes que
    comparten clave normalizada. Funciona porque el generador ensucia solo una minoria de
    las filas (~4 %): la escritura correcta siempre es mayoria. La alternativa (recapitalizar
    con `initcap`) destrozaria nombres como `Torrijas y bolleria de Cuaresma`, y mirar el
    catalogo del generador acoplaria el ETL a la Fase 1.

    Los empates se rompen prefiriendo la grafia que **parece** limpia, antes que el orden
    alfabetico: sin eso, un empate entre `Leche` y `LECHE` lo ganaria la segunda (las
    mayusculas van antes en ASCII) y entre `Salsa de tomate` y `Salsa  de  tomate` la
    segunda tambien (el espacio va antes que cualquier letra). Se penaliza, por ese orden,
    tener espacios sobrantes y estar toda en mayusculas o toda en minusculas.

    Returns:
        DataFrame con `category_key` y `category_canonical`.
    """
    counts = (
        products.select(normalized_category().alias("category_key"), "category")
        .groupBy("category_key", "category")
        .count()
    )

    tidy = F.trim(F.regexp_replace(F.col("category"), r"\s+", " "))
    spacing_penalty = F.when(F.col("category") != tidy, 1).otherwise(0)
    case_penalty = F.when(
        (F.col("category") == F.upper("category")) | (F.col("category") == F.lower("category")),
        1,
    ).otherwise(0)

    ranked = Window.partitionBy("category_key").orderBy(
        F.col("count").desc(), spacing_penalty.asc(), case_penalty.asc(), F.col("category").asc()
    )
    return (
        counts.withColumn("_rn", F.row_number().over(ranked))
        .filter(F.col("_rn") == 1)
        .select("category_key", F.col("category").alias("category_canonical"))
    )


def _dedupe_by_key(df: DataFrame, keys: list[str]) -> DataFrame:
    """Deja una fila por clave, eligiendo siempre la misma (orden por el resto de columnas)."""
    rest = [F.col(c).asc_nulls_first() for c in df.columns if c not in keys]
    window = Window.partitionBy(*keys).orderBy(*(rest or [F.lit(1)]))
    return (
        df.withColumn("_rn", F.row_number().over(window)).filter(F.col("_rn") == 1).drop("_rn")
    )


def _drop_duplicate_keys(
    df: DataFrame, keys: list[str], *, table: str, issue: str, rule: str, report: CleaningReport
) -> DataFrame:
    """Elimina claves repetidas solo si las hay, y lo anota en el informe."""
    before = df.count()
    distinct = df.select(*keys).distinct().count()
    dups = before - distinct
    out = _dedupe_by_key(df, keys) if dups else df
    report.add(CleaningStep(table, issue, rule, dups, before, before - dups))
    return out


# --------------------------------------------------------------------------------------
# Limpieza por tabla
# --------------------------------------------------------------------------------------


def clean_products(
    products: DataFrame, cfg: CleaningConfig, report: CleaningReport
) -> DataFrame:
    """Normaliza categorias y marca las marcas sin informar.

    `category` pasa a contener la grafia canonica y `category_raw` conserva la original,
    para poder auditar la correccion sin volver a `data/raw`.
    """
    out = _drop_duplicate_keys(
        products,
        ["product_id"],
        table="products",
        issue="product_id duplicado",
        rule="Se conserva una fila por PK",
        report=report,
    )

    mapping = canonical_category_map(out)
    joined = (
        out.withColumn("category_key", normalized_category())
        .join(F.broadcast(mapping), "category_key", "left")
        .drop("category_key")
    )

    n_before = joined.count()
    n_messy = joined.filter(F.col("category") != F.col("category_canonical")).count()
    # La traduccion exacta se guarda en el informe: es la evidencia de que la
    # normalizacion no ha inventado ninguna categoria nueva.
    report.category_fixes = {
        row["category"]: row["category_canonical"]
        for row in joined.filter(F.col("category") != F.col("category_canonical"))
        .select("category", "category_canonical")
        .distinct()
        .orderBy("category")
        .collect()
    }
    report.add(
        CleaningStep(
            "products",
            "category con mayusculas/espacios inconsistentes",
            "Se unifica a la grafia mas frecuente de su clave normalizada",
            n_messy,
            n_before,
            n_before,
        )
    )

    n_brand = joined.filter(F.col("brand").isNull()).count()
    report.add(
        CleaningStep(
            "products",
            "brand nula",
            f"Se rellena con '{cfg.missing_brand_label}' y se marca en brand_is_missing",
            n_brand,
            n_before,
            n_before,
        )
    )

    return joined.select(
        "product_id",
        "department",
        F.col("category_canonical").alias("category"),
        F.col("category").alias("category_raw"),
        F.coalesce("brand", F.lit(cfg.missing_brand_label)).alias("brand"),
        F.col("brand").isNull().alias("brand_is_missing"),
        "is_private_label",
        "is_perishable",
        "unit_price",
        "pack_size",
        "typical_repurchase_days",
    )


def clean_basket_items(
    basket_items: DataFrame, cfg: CleaningConfig, report: CleaningReport
) -> DataFrame:
    """Corrige el signo de `quantity` y elimina las lineas duplicadas.

    El orden es deliberado: primero el signo, despues el duplicado (ver el docstring del
    modulo). Anade `line_amount` e `is_promo`, que usan RFM, EDA y el recomendador.
    """
    before = basket_items.count()
    n_negative = basket_items.filter(F.col("quantity") < 0).count()

    if cfg.negative_quantity_policy == "abs":
        signed = basket_items.withColumn(
            "quantity_sign_corrected", F.col("quantity") < 0
        ).withColumn("quantity", F.abs(F.col("quantity")))
        rule = "Se corrige el signo: quantity = abs(quantity)"
    else:
        signed = basket_items.filter(F.col("quantity") >= 0).withColumn(
            "quantity_sign_corrected", F.lit(False)
        )
        rule = "Se descartan las lineas negativas (tratadas como devoluciones)"

    report.add(
        CleaningStep(
            "basket_items",
            "quantity negativa",
            rule,
            n_negative,
            before,
            signed.count() if cfg.negative_quantity_policy == "drop" else before,
        )
    )

    after_sign = before if cfg.negative_quantity_policy == "abs" else before - n_negative
    # Se agrupa en vez de usar `dropDuplicates` porque las copias de una misma linea
    # pueden diferir en `quantity_sign_corrected` (a una se le invirtio el signo y a la
    # otra no). `dropDuplicates` se quedaria con una cualquiera y el flag saldria distinto
    # en cada ejecucion; con `max` la linea recuerda que alguna copia venia mal firmada.
    deduped = signed.groupBy(
        "basket_id", "product_id", "quantity", "unit_price_paid", "promotion_id"
    ).agg(F.max("quantity_sign_corrected").alias("quantity_sign_corrected"))
    n_after = deduped.count()
    report.add(
        CleaningStep(
            "basket_items",
            "linea de ticket duplicada",
            "Se elimina la fila repetida entera, despues de corregir el signo",
            after_sign - n_after,
            after_sign,
            n_after,
        )
    )

    if cfg.detailed_counts:
        # Cuantos duplicados solo se ven tras corregir el signo: es la justificacion
        # medible del orden de las dos reglas.
        naive = basket_items.distinct().count()
        report.add(
            CleaningStep(
                "basket_items",
                "duplicados que solo afloran tras corregir el signo",
                "Diagnostico: (dedup tras signo) - (dedup sin corregir el signo)",
                (before - n_after) - (before - naive),
                before,
                n_after,
            )
        )

    n_zero = deduped.filter(F.col("quantity") <= 0).count()
    final = deduped.filter(F.col("quantity") > 0) if n_zero else deduped
    report.add(
        CleaningStep(
            "basket_items",
            "quantity nula o cero",
            "Se descarta: una linea sin unidades no es una venta",
            n_zero,
            n_after,
            n_after - n_zero,
        )
    )

    return final.select(
        "basket_id",
        "product_id",
        "quantity",
        "unit_price_paid",
        "promotion_id",
        F.round(F.col("quantity") * F.col("unit_price_paid"), 2).alias("line_amount"),
        F.col("promotion_id").isNotNull().alias("is_promo"),
        "quantity_sign_corrected",
    )


def clean_baskets(
    baskets: DataFrame, clean_items: DataFrame, cfg: CleaningConfig, report: CleaningReport
) -> DataFrame:
    """Recalcula `total_amount` desde las lineas limpias y marca los outliers.

    `total_amount_raw` conserva el importe original de la cabecera, y
    `total_amount_is_outlier` senala las cestas cuyo importe declarado desbordaba al real.
    """
    out = _drop_duplicate_keys(
        baskets,
        ["basket_id"],
        table="baskets",
        issue="basket_id duplicado",
        rule="Se conserva una fila por PK",
        report=report,
    )

    totals = clean_items.groupBy("basket_id").agg(
        F.round(F.sum("line_amount"), 2).alias("recalc_amount"),
        F.count(F.lit(1)).cast("int").alias("n_lines"),
        F.sum("quantity").cast("int").alias("n_units"),
    )
    joined = out.join(totals, "basket_id", "left").withColumn(
        "recalc_amount", F.coalesce("recalc_amount", F.lit(0.0))
    )

    n_before = joined.count()
    is_outlier = (F.col("recalc_amount") > 0) & (
        F.col("total_amount") > cfg.amount_outlier_ratio * F.col("recalc_amount")
    )
    n_changed = joined.filter(
        F.abs(F.col("total_amount") - F.col("recalc_amount")) > 0.01
    ).count()
    n_outlier = joined.filter(is_outlier).count()

    report.add(
        CleaningStep(
            "baskets",
            "total_amount no cuadra con las lineas",
            "Se recalcula como suma de line_amount de basket_items ya limpias",
            n_changed,
            n_before,
            n_before,
        )
    )
    report.add(
        CleaningStep(
            "baskets",
            "outlier de total_amount",
            f"Se marca en total_amount_is_outlier si la cabecera supera "
            f"{cfg.amount_outlier_ratio}x el importe recalculado",
            n_outlier,
            n_before,
            n_before,
        )
    )

    n_empty = joined.filter(F.col("n_lines").isNull()).count()
    report.add(
        CleaningStep(
            "baskets",
            "cesta sin lineas tras la limpieza",
            "Se conserva con importe 0: la cabecera sigue siendo una visita real",
            n_empty,
            n_before,
            n_before,
        )
    )

    return joined.select(
        "basket_id",
        "customer_id",
        "channel",
        "basket_date",
        F.to_date("basket_date").alias("basket_day"),
        "store_id",
        F.col("recalc_amount").alias("total_amount"),
        F.col("total_amount").alias("total_amount_raw"),
        is_outlier.alias("total_amount_is_outlier"),
        F.coalesce("n_lines", F.lit(0)).alias("n_lines"),
        F.coalesce("n_units", F.lit(0)).alias("n_units"),
        F.col("customer_id").isNull().alias("is_anonymous"),
    )


def clean_customers(
    customers: DataFrame, baskets: DataFrame, cfg: CleaningConfig, report: CleaningReport
) -> DataFrame:
    """Rellena `city` y corrige las altas posteriores a la primera compra.

    La correccion es `signup_date = min(signup_date, primera compra)`: una fecha de alta
    posterior a una compra del propio cliente es imposible, y la compra es el dato duro.
    """
    out = _drop_duplicate_keys(
        customers,
        ["customer_id"],
        table="customers",
        issue="customer_id duplicado",
        rule="Se conserva una fila por PK",
        report=report,
    )

    first_purchase = (
        baskets.filter(F.col("customer_id").isNotNull())
        .groupBy("customer_id")
        .agg(
            F.min(F.to_date("basket_date")).alias("first_purchase_date"),
            F.max(F.to_date("basket_date")).alias("last_purchase_date"),
        )
    )
    joined = out.join(first_purchase, "customer_id", "left")

    n_before = joined.count()
    bad_signup = F.col("first_purchase_date").isNotNull() & (
        F.col("signup_date") > F.col("first_purchase_date")
    )
    n_bad = joined.filter(bad_signup).count()
    n_city = joined.filter(F.col("city").isNull()).count()

    report.add(
        CleaningStep(
            "customers",
            "city nula",
            f"Se rellena con '{cfg.missing_city_label}' y se marca en city_is_missing",
            n_city,
            n_before,
            n_before,
        )
    )
    report.add(
        CleaningStep(
            "customers",
            "signup_date posterior a la primera compra",
            "Se corrige a la fecha de la primera compra observada",
            n_bad,
            n_before,
            n_before,
        )
    )

    return joined.select(
        "customer_id",
        F.when(bad_signup, F.col("first_purchase_date"))
        .otherwise(F.col("signup_date"))
        .alias("signup_date"),
        F.col("signup_date").alias("signup_date_raw"),
        bad_signup.alias("signup_date_corrected"),
        "country",
        F.coalesce("city", F.lit(cfg.missing_city_label)).alias("city"),
        F.col("city").isNull().alias("city_is_missing"),
        "household_size_est",
        "loyalty_tier",
        "preferred_channel",
        "churn_label",
        "first_purchase_date",
        "last_purchase_date",
    )


def clean_promotions(promotions: DataFrame, report: CleaningReport) -> DataFrame:
    """Deduplica y marca las ventanas de vigencia imposibles.

    Una promocion con `start_date > end_date` no se borra: `basket_items.promotion_id`
    apunta a ella y eliminarla romperia la integridad referencial. Se marca y ya.
    """
    out = _drop_duplicate_keys(
        promotions,
        ["promotion_id"],
        table="promotions",
        issue="promotion_id duplicado",
        rule="Se conserva una fila por PK",
        report=report,
    )

    n_before = out.count()
    invalid = F.col("start_date") > F.col("end_date")
    report.add(
        CleaningStep(
            "promotions",
            "ventana de vigencia invertida",
            "Se marca en date_range_invalid; no se borra para no romper las FK",
            out.filter(invalid).count(),
            n_before,
            n_before,
        )
    )
    return out.select(
        "promotion_id",
        "product_id",
        "promo_type",
        "discount_value",
        "start_date",
        "end_date",
        invalid.alias("date_range_invalid"),
        (F.datediff("end_date", "start_date") + 1).alias("duration_days"),
    )


def clean_sessions(sessions: DataFrame, report: CleaningReport) -> DataFrame:
    """Deduplica y alinea `converted` con la existencia de `basket_id`."""
    out = _drop_duplicate_keys(
        sessions,
        ["session_id"],
        table="sessions",
        issue="session_id duplicado",
        rule="Se conserva una fila por PK",
        report=report,
    )

    n_before = out.count()
    converted = F.col("basket_id").isNotNull()
    report.add(
        CleaningStep(
            "sessions",
            "converted incoherente con basket_id",
            "converted pasa a derivarse de basket_id IS NOT NULL",
            out.filter(F.col("converted") != converted).count(),
            n_before,
            n_before,
        )
    )
    return out.select(
        "session_id",
        "customer_id",
        "session_date",
        F.to_date("session_date").alias("session_day"),
        "device_type",
        converted.alias("converted"),
        "basket_id",
        F.col("customer_id").isNull().alias("is_anonymous"),
    )


def clean_session_events(session_events: DataFrame, report: CleaningReport) -> DataFrame:
    """Elimina eventos repetidos exactos (mismo producto, evento e instante)."""
    before = session_events.count()
    out = session_events.dropDuplicates()
    after = out.count()
    report.add(
        CleaningStep(
            "session_events",
            "evento duplicado exacto",
            "Se elimina la fila repetida entera",
            before - after,
            before,
            after,
        )
    )
    return out


# --------------------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------------------


def clean_all(
    raw: dict[str, DataFrame], cfg: CleaningConfig | None = None
) -> tuple[dict[str, DataFrame], CleaningReport]:
    """Limpia las 7 tablas y devuelve el resultado junto con la traza de lo corregido.

    El orden no es negociable: `basket_items` se limpia antes que `baskets` porque la
    cabecera se recalcula desde las lineas ya limpias, y `customers` despues de
    `baskets` porque necesita la primera compra para arreglar `signup_date`.
    """
    cfg = cfg or CleaningConfig()
    report = CleaningReport()

    products = clean_products(raw["products"], cfg, report)
    items = clean_basket_items(raw["basket_items"], cfg, report).cache()
    baskets = clean_baskets(raw["baskets"], items, cfg, report).cache()
    customers = clean_customers(raw["customers"], raw["baskets"], cfg, report)
    promotions = clean_promotions(raw["promotions"], report)
    sessions = clean_sessions(raw["sessions"], report)
    session_events = clean_session_events(raw["session_events"], report)

    return {
        "customers": customers,
        "products": products,
        "promotions": promotions,
        "baskets": baskets,
        "basket_items": items,
        "sessions": sessions,
        "session_events": session_events,
    }, report

"""Data Trust Score del dataset (Tarea 1 de `CHALLENGE.md`).

Una sola cifra 0-100 que resume si el dato es fiable, desglosada en cinco dimensiones
clasicas de calidad. Cada dimension agrupa varias comprobaciones; cada comprobacion es la
fraccion de filas que pasan una regla concreta:

| Dimension       | Que mide                                                             |
| ---------------- | ---------------------------------------------------------------------- |
| `completeness`   | Columnas obligatorias informadas (las nulables por diseno no penalizan) |
| `uniqueness`     | Claves primarias sin repetir y filas sin duplicar                      |
| `validity`       | Valores dentro de su dominio o rango                                   |
| `consistency`    | Coherencia entre columnas y entre tablas                               |
| `integrity`      | Claves ajenas que resuelven                                            |

## Como se agrega

Cada dimension se puntua mezclando a partes iguales dos lecturas de sus comprobaciones:

- **Tasa de filas** (`row_score`): media del % de filas que pasan cada regla. Responde a
  "cuanto dato esta bien".
- **Tasa de reglas** (`check_score`): % de reglas que pasan **sin una sola** fila mala.
  Responde a "cuantos problemas distintos hay".

Hacen falta las dos. Solo con la tasa de filas, un dataset con 1,5 % de lineas duplicadas
y tres problemas mas puntua 99,6 y saca una A: los defectos reales quedan diluidos entre
decenas de reglas que pasan de sobra. Solo con la tasa de reglas, un unico nulo pesaria lo
mismo que la mitad de la tabla corrupta. La media de ambas separa bien lo sucio de lo
limpio sin dramatizar un caso aislado.

La puntuacion global es la media ponderada de las dimensiones. Ponderar por numero de
filas seria enganoso: una FK rota en `promotions` (300 filas) es tan grave como en
`basket_items` (3 M).

Se ejecuta igual sobre el dataset crudo y sobre el limpio, y la diferencia entre ambos es
la medida de lo que ha aportado la limpieza. Ojo al leerla: tras la limpieza el score sube
en parte porque los defectos se han **corregido** y en parte porque se han **etiquetado**
(`city_is_missing`, `total_amount_is_outlier`...). Las columnas de marca mantienen el
defecto visible y auditable; el score mide aptitud para el uso, no ausencia de historia.

Uso:
    report = data_trust_score(tables)
    print(report.score, report.grade)
    print(report.to_markdown())
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from src.etl.cleaning import canonical_category_map, normalized_category

DIMENSIONS: tuple[str, ...] = (
    "completeness",
    "uniqueness",
    "validity",
    "consistency",
    "integrity",
)

# Las cinco dimensiones pesan igual: no hay razon de negocio para preferir una, y
# repartir a partes iguales evita discutir decimales sin fundamento.
DEFAULT_WEIGHTS: dict[str, float] = {d: 0.20 for d in DIMENSIONS}

# Mezcla "cuanto dato esta bien" con "cuantos problemas distintos hay" (ver el docstring
# del modulo). Deben sumar 1.
ROW_WEIGHT = 0.5
CHECK_WEIGHT = 0.5

# Columnas obligatorias por tabla. Las que admiten nulo por diseno se quedan fuera a
# proposito y estan justificadas aqui, que es donde alguien vendra a discutirlas.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    # `city` sí es obligatoria: un cliente sin ciudad es un hueco, no un caso legitimo.
    "customers": (
        "customer_id",
        "signup_date",
        "country",
        "city",
        "household_size_est",
        "loyalty_tier",
        "preferred_channel",
        "churn_label",
    ),
    "products": (
        "product_id",
        "department",
        "category",
        "brand",
        "is_private_label",
        "is_perishable",
        "unit_price",
        "pack_size",
        "typical_repurchase_days",
    ),
    "promotions": (
        "promotion_id",
        "product_id",
        "promo_type",
        "discount_value",
        "start_date",
        "end_date",
    ),
    # `customer_id` nulo = compra anonima legitima; `store_id` nulo = canal online.
    "baskets": ("basket_id", "channel", "basket_date", "total_amount"),
    # `promotion_id` nulo = linea sin promocion.
    "basket_items": ("basket_id", "product_id", "quantity", "unit_price_paid"),
    # `customer_id` nulo = visitante no identificado; `basket_id` nulo = no convirtio.
    "sessions": ("session_id", "session_date", "device_type", "converted"),
    "session_events": ("session_id", "product_id", "event_type", "event_timestamp"),
}

LOYALTY_TIERS = ("bronze", "silver", "gold")
CHANNELS = ("app", "web", "store")
DEVICE_TYPES = ("mobile", "desktop", "tablet")
PROMO_TYPES = ("2x1", "discount_pct", "coupon")
EVENT_TYPES = ("view", "add_to_cart")

GRADES: tuple[tuple[float, str], ...] = (
    (97.0, "A"),
    (93.0, "B"),
    (85.0, "C"),
    (75.0, "D"),
    (0.0, "E"),
)


# --------------------------------------------------------------------------------------
# Estructuras de salida
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustCheck:
    """Una comprobacion de calidad y su resultado."""

    table: str
    dimension: str
    name: str
    description: str
    rows_total: int
    rows_failed: int

    @property
    def pass_rate(self) -> float:
        """Fraccion de filas que pasan. Una tabla vacia se considera aprobada."""
        if self.rows_total == 0:
            return 1.0
        return (self.rows_total - self.rows_failed) / self.rows_total

    @property
    def passed(self) -> bool:
        """True si la regla no tiene ni una sola fila incumpliendo."""
        return self.rows_failed == 0

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "dimension": self.dimension,
            "name": self.name,
            "description": self.description,
            "rows_total": self.rows_total,
            "rows_failed": self.rows_failed,
            "pass_rate": round(self.pass_rate, 6),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class TrustReport:
    """Resultado completo del Data Trust Score."""

    checks: tuple[TrustCheck, ...]
    weights: dict[str, float]

    @property
    def dimension_detail(self) -> dict[str, dict[str, float]]:
        """Desglose por dimension: tasa de filas, tasa de reglas y puntuacion combinada."""
        out: dict[str, dict[str, float]] = {}
        for dim in DIMENSIONS:
            checks = [c for c in self.checks if c.dimension == dim]
            if not checks:
                out[dim] = {"row_score": 100.0, "check_score": 100.0, "score": 100.0, "checks": 0,
                            "checks_failing": 0}
                continue
            row_score = 100.0 * sum(c.pass_rate for c in checks) / len(checks)
            check_score = 100.0 * sum(1 for c in checks if c.passed) / len(checks)
            out[dim] = {
                "row_score": round(row_score, 2),
                "check_score": round(check_score, 2),
                "score": round(ROW_WEIGHT * row_score + CHECK_WEIGHT * check_score, 2),
                "checks": len(checks),
                "checks_failing": sum(1 for c in checks if not c.passed),
            }
        return out

    @property
    def dimension_scores(self) -> dict[str, float]:
        """Puntuacion 0-100 por dimension."""
        return {dim: detail["score"] for dim, detail in self.dimension_detail.items()}

    @property
    def score(self) -> float:
        """Data Trust Score global, 0-100."""
        scores = self.dimension_scores
        total_weight = sum(self.weights.get(d, 0.0) for d in DIMENSIONS)
        if total_weight == 0:
            return 0.0
        weighted = sum(scores[d] * self.weights.get(d, 0.0) for d in DIMENSIONS)
        return round(weighted / total_weight, 2)

    @property
    def grade(self) -> str:
        """Nota de la A (fiable) a la E (no usable sin limpiar)."""
        for threshold, grade in GRADES:
            if self.score >= threshold:
                return grade
        return GRADES[-1][1]

    def failing(self, min_failures: int = 1) -> tuple[TrustCheck, ...]:
        """Comprobaciones con al menos `min_failures` filas incumpliendo."""
        return tuple(c for c in self.checks if c.rows_failed >= min_failures)

    def check(self, table: str, name: str) -> TrustCheck:
        """Recupera una comprobacion concreta. Pensado para los tests."""
        for c in self.checks:
            if c.table == table and c.name == name:
                return c
        raise KeyError(f"No existe la comprobacion {name!r} para la tabla {table!r}")

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "grade": self.grade,
            "checks_total": len(self.checks),
            "checks_failing": sum(1 for c in self.checks if not c.passed),
            "dimension_scores": self.dimension_scores,
            "dimension_detail": self.dimension_detail,
            "weights": self.weights,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self, *, only_failing: bool = False) -> str:
        """Informe legible. Las comprobaciones se ordenan de peor a mejor."""
        detail = self.dimension_detail
        n_failing = sum(1 for c in self.checks if not c.passed)
        lines = [
            f"**Data Trust Score: {self.score:.2f} / 100 (nota {self.grade})** — "
            f"{len(self.checks) - n_failing} de {len(self.checks)} comprobaciones sin una "
            f"sola fila mala.",
            "",
            "| Dimension | Puntuacion | Tasa de filas | Tasa de reglas | Reglas KO |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for dim in DIMENSIONS:
            d = detail[dim]
            lines.append(
                f"| `{dim}` | {d['score']:.2f} | {d['row_score']:.2f} % | "
                f"{d['check_score']:.2f} % | {int(d['checks_failing'])} / {int(d['checks'])} |"
            )

        def n(value: int) -> str:
            """Separador de miles a la espanola, solo sobre el numero."""
            return f"{value:,}".replace(",", ".")

        shown = self.failing() if only_failing else self.checks
        lines += [
            "",
            "| Tabla | Dimension | Comprobacion | Filas KO | Total | % OK |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for c in sorted(shown, key=lambda c: (c.pass_rate, c.table, c.name)):
            lines.append(
                f"| `{c.table}` | {c.dimension} | {c.description} | "
                f"{n(c.rows_failed)} | {n(c.rows_total)} | {100 * c.pass_rate:.3f} % |"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Motor de comprobaciones
# --------------------------------------------------------------------------------------

# Una comprobacion a nivel de fila: (nombre, descripcion, condicion de FALLO).
_RowSpec = tuple[str, str, Column]


def _row_checks(
    df: DataFrame, table: str, dimension: str, specs: list[_RowSpec]
) -> list[TrustCheck]:
    """Evalua varias reglas de fila sobre el mismo DataFrame en una unica pasada."""
    if not specs:
        return []
    aggs = [F.count(F.lit(1)).alias("_total")]
    for i, (_, _, failure) in enumerate(specs):
        # `when` deja fuera los nulos de la condicion, que es lo que queremos: una regla
        # que no se puede evaluar no cuenta como fallo.
        aggs.append(F.sum(F.when(failure, F.lit(1)).otherwise(F.lit(0))).alias(f"_c{i}"))
    row = df.agg(*aggs).collect()[0]
    total = int(row["_total"])
    return [
        TrustCheck(table, dimension, name, desc, total, int(row[f"_c{i}"] or 0))
        for i, (name, desc, _) in enumerate(specs)
    ]


def _completeness_specs(table: str) -> list[_RowSpec]:
    return [
        (
            f"{col}_not_null",
            f"`{col}` informada",
            F.col(col).isNull(),
        )
        for col in REQUIRED_COLUMNS[table]
    ]


def _unique_check(
    df: DataFrame, table: str, keys: list[str], name: str, description: str
) -> TrustCheck:
    """Comprueba que una combinacion de columnas no se repite."""
    total = df.count()
    distinct = df.select(*keys).distinct().count()
    return TrustCheck(table, "uniqueness", name, description, total, total - distinct)


def _no_duplicate_rows_check(df: DataFrame, table: str) -> TrustCheck:
    total = df.count()
    return TrustCheck(
        table,
        "uniqueness",
        "no_duplicate_rows",
        "Sin filas duplicadas exactas",
        total,
        total - df.distinct().count(),
    )


def _fk_check(
    df: DataFrame,
    table: str,
    column: str,
    ref: DataFrame,
    ref_column: str,
) -> TrustCheck:
    """Cuenta valores no nulos de `column` que no existen en `ref.ref_column`."""
    present = df.filter(F.col(column).isNotNull()).select(F.col(column).alias("_v"))
    keys = ref.select(F.col(ref_column).alias("_k")).distinct()
    orphans = present.join(keys, F.col("_v") == F.col("_k"), "left_anti").count()
    return TrustCheck(
        table,
        "integrity",
        f"{column}_fk",
        f"`{column}` existe en `{ref_column}`",
        present.count(),
        orphans,
    )


def _isin_failure(column: str, allowed: tuple[str, ...]) -> Column:
    """Condicion de fallo: valor informado fuera del dominio permitido."""
    return F.col(column).isNotNull() & ~F.col(column).isin(list(allowed))


# --------------------------------------------------------------------------------------
# Comprobaciones por tabla
# --------------------------------------------------------------------------------------


def _customers_checks(tables: dict[str, DataFrame]) -> list[TrustCheck]:
    customers, baskets = tables["customers"], tables["baskets"]
    checks = _row_checks(customers, "customers", "completeness", _completeness_specs("customers"))
    checks += _row_checks(
        customers,
        "customers",
        "validity",
        [
            (
                "household_size_in_range",
                "`household_size_est` entre 1 y 6",
                F.col("household_size_est").isNotNull()
                & ~F.col("household_size_est").between(1, 6),
            ),
            (
                "loyalty_tier_domain",
                "`loyalty_tier` en bronze/silver/gold",
                _isin_failure("loyalty_tier", LOYALTY_TIERS),
            ),
            (
                "preferred_channel_domain",
                "`preferred_channel` en app/web/store",
                _isin_failure("preferred_channel", CHANNELS),
            ),
        ],
    )
    checks.append(_unique_check(customers, "customers", ["customer_id"], "customer_id_unique", "`customer_id` sin repetir"))

    # Consistencia con el historial de compra.
    max_date = baskets.agg(F.max(F.to_date("basket_date"))).collect()[0][0]
    first_purchase = (
        baskets.filter(F.col("customer_id").isNotNull())
        .groupBy("customer_id")
        .agg(F.min(F.to_date("basket_date")).alias("_first"))
    )
    joined = customers.join(first_purchase, "customer_id", "left")
    checks += _row_checks(
        joined,
        "customers",
        "consistency",
        [
            (
                "signup_before_first_purchase",
                "`signup_date` anterior o igual a la primera compra",
                F.col("_first").isNotNull() & (F.col("signup_date") > F.col("_first")),
            ),
            (
                "signup_within_period",
                "`signup_date` no posterior al fin del periodo observado",
                F.col("signup_date") > F.lit(max_date),
            ),
        ],
    )
    return checks


def _products_checks(tables: dict[str, DataFrame]) -> list[TrustCheck]:
    products = tables["products"]
    checks = _row_checks(products, "products", "completeness", _completeness_specs("products"))
    checks.append(
        _unique_check(products, "products", ["product_id"], "product_id_unique", "`product_id` sin repetir")
    )

    # `category` deberia venir ya en su grafia canonica: si difiere de la mayoritaria de
    # su clave normalizada, hay mayusculas o espacios de mas.
    canonical = canonical_category_map(products)
    with_canonical = products.withColumn("_key", normalized_category()).join(
        F.broadcast(canonical), F.col("_key") == F.col("category_key"), "left"
    )
    checks += _row_checks(
        with_canonical,
        "products",
        "validity",
        [
            (
                "category_canonical",
                "`category` sin mayusculas ni espacios inconsistentes",
                F.col("category") != F.col("category_canonical"),
            ),
        ],
    )
    checks += _row_checks(
        products,
        "products",
        "validity",
        [
            ("unit_price_positive", "`unit_price` mayor que 0", F.col("unit_price") <= 0),
            ("pack_size_positive", "`pack_size` mayor que 0", F.col("pack_size") <= 0),
            (
                "repurchase_days_positive",
                "`typical_repurchase_days` mayor que 0",
                F.col("typical_repurchase_days") <= 0,
            ),
        ],
    )
    return checks


def _promotions_checks(tables: dict[str, DataFrame]) -> list[TrustCheck]:
    promotions, products = tables["promotions"], tables["products"]
    checks = _row_checks(
        promotions, "promotions", "completeness", _completeness_specs("promotions")
    )
    checks.append(
        _unique_check(
            promotions, "promotions", ["promotion_id"], "promotion_id_unique", "`promotion_id` sin repetir"
        )
    )
    checks += _row_checks(
        promotions,
        "promotions",
        "validity",
        [
            (
                "promo_type_domain",
                "`promo_type` en 2x1/discount_pct/coupon",
                _isin_failure("promo_type", PROMO_TYPES),
            ),
            (
                "discount_value_positive",
                "`discount_value` mayor que 0",
                F.col("discount_value") <= 0,
            ),
            (
                "date_range_valid",
                "`start_date` anterior o igual a `end_date`",
                F.col("start_date") > F.col("end_date"),
            ),
        ],
    )
    checks.append(_fk_check(promotions, "promotions", "product_id", products, "product_id"))
    return checks


def _baskets_checks(tables: dict[str, DataFrame]) -> list[TrustCheck]:
    baskets, items, customers = tables["baskets"], tables["basket_items"], tables["customers"]
    checks = _row_checks(baskets, "baskets", "completeness", _completeness_specs("baskets"))
    checks.append(
        _unique_check(baskets, "baskets", ["basket_id"], "basket_id_unique", "`basket_id` sin repetir")
    )
    checks += _row_checks(
        baskets,
        "baskets",
        "validity",
        [
            ("channel_domain", "`channel` en app/web/store", _isin_failure("channel", CHANNELS)),
            ("total_amount_positive", "`total_amount` mayor que 0", F.col("total_amount") <= 0),
        ],
    )
    checks += _row_checks(
        baskets,
        "baskets",
        "consistency",
        [
            (
                "store_id_matches_channel",
                "`store_id` informado si y solo si el canal es `store`",
                (F.col("channel") == "store") != F.col("store_id").isNotNull(),
            ),
        ],
    )

    # La cabecera tiene que cuadrar con la suma de sus lineas. Es la comprobacion que
    # destapa a la vez los duplicados, las cantidades negativas y los outliers de importe.
    line_totals = items.groupBy("basket_id").agg(
        F.round(F.sum(F.col("quantity") * F.col("unit_price_paid")), 2).alias("_lines_total")
    )
    joined = baskets.join(line_totals, "basket_id", "left").withColumn(
        "_lines_total", F.coalesce("_lines_total", F.lit(0.0))
    )
    checks += _row_checks(
        joined,
        "baskets",
        "consistency",
        [
            (
                "total_amount_matches_items",
                "`total_amount` cuadra con la suma de sus lineas",
                F.abs(F.col("total_amount") - F.col("_lines_total")) > 0.01,
            ),
        ],
    )
    checks.append(_fk_check(baskets, "baskets", "customer_id", customers, "customer_id"))
    return checks


def _basket_items_checks(tables: dict[str, DataFrame]) -> list[TrustCheck]:
    items = tables["basket_items"]
    baskets, products, promotions = tables["baskets"], tables["products"], tables["promotions"]

    checks = _row_checks(
        items, "basket_items", "completeness", _completeness_specs("basket_items")
    )
    checks.append(_no_duplicate_rows_check(items, "basket_items"))
    checks.append(
        _unique_check(
            items,
            "basket_items",
            ["basket_id", "product_id"],
            "basket_product_unique",
            "Una sola linea por cesta y producto",
        )
    )
    checks += _row_checks(
        items,
        "basket_items",
        "validity",
        [
            ("quantity_positive", "`quantity` mayor que 0", F.col("quantity") <= 0),
            (
                "unit_price_paid_positive",
                "`unit_price_paid` mayor que 0",
                F.col("unit_price_paid") <= 0,
            ),
        ],
    )

    # Una linea en promocion tiene que caer dentro de la ventana de esa promocion.
    promo_lines = (
        items.filter(F.col("promotion_id").isNotNull())
        .join(baskets.select("basket_id", F.to_date("basket_date").alias("_day")), "basket_id")
        .join(
            F.broadcast(promotions.select("promotion_id", "start_date", "end_date")),
            "promotion_id",
        )
    )
    checks += _row_checks(
        promo_lines,
        "basket_items",
        "consistency",
        [
            (
                "promo_active_on_basket_date",
                "La promocion aplicada estaba vigente el dia de la cesta",
                ~F.col("_day").between(F.col("start_date"), F.col("end_date")),
            ),
        ],
    )

    checks.append(_fk_check(items, "basket_items", "basket_id", baskets, "basket_id"))
    checks.append(_fk_check(items, "basket_items", "product_id", products, "product_id"))
    checks.append(_fk_check(items, "basket_items", "promotion_id", promotions, "promotion_id"))
    return checks


def _sessions_checks(tables: dict[str, DataFrame]) -> list[TrustCheck]:
    sessions, baskets, customers = tables["sessions"], tables["baskets"], tables["customers"]
    checks = _row_checks(sessions, "sessions", "completeness", _completeness_specs("sessions"))
    checks.append(
        _unique_check(sessions, "sessions", ["session_id"], "session_id_unique", "`session_id` sin repetir")
    )
    checks += _row_checks(
        sessions,
        "sessions",
        "validity",
        [
            (
                "device_type_domain",
                "`device_type` en mobile/desktop/tablet",
                _isin_failure("device_type", DEVICE_TYPES),
            ),
        ],
    )
    checks += _row_checks(
        sessions,
        "sessions",
        "consistency",
        [
            (
                "converted_matches_basket",
                "`converted` coincide con tener `basket_id`",
                F.col("converted") != F.col("basket_id").isNotNull(),
            ),
        ],
    )
    checks.append(_fk_check(sessions, "sessions", "customer_id", customers, "customer_id"))
    checks.append(_fk_check(sessions, "sessions", "basket_id", baskets, "basket_id"))
    return checks


def _session_events_checks(tables: dict[str, DataFrame]) -> list[TrustCheck]:
    events, sessions, products = tables["session_events"], tables["sessions"], tables["products"]
    checks = _row_checks(
        events, "session_events", "completeness", _completeness_specs("session_events")
    )
    checks.append(_no_duplicate_rows_check(events, "session_events"))
    checks += _row_checks(
        events,
        "session_events",
        "validity",
        [
            (
                "event_type_domain",
                "`event_type` en view/add_to_cart",
                _isin_failure("event_type", EVENT_TYPES),
            ),
        ],
    )
    joined = events.join(sessions.select("session_id", "session_date"), "session_id", "left")
    checks += _row_checks(
        joined,
        "session_events",
        "consistency",
        [
            (
                "event_after_session_start",
                "El evento no es anterior al inicio de su sesion",
                F.col("session_date").isNotNull()
                & (F.col("event_timestamp") < F.col("session_date")),
            ),
        ],
    )
    checks.append(_fk_check(events, "session_events", "session_id", sessions, "session_id"))
    checks.append(_fk_check(events, "session_events", "product_id", products, "product_id"))
    return checks


# --------------------------------------------------------------------------------------
# API publica
# --------------------------------------------------------------------------------------

_TABLE_CHECKS = {
    "customers": _customers_checks,
    "products": _products_checks,
    "promotions": _promotions_checks,
    "baskets": _baskets_checks,
    "basket_items": _basket_items_checks,
    "sessions": _sessions_checks,
    "session_events": _session_events_checks,
}


def data_trust_score(
    tables: dict[str, DataFrame],
    *,
    weights: dict[str, float] | None = None,
    only_tables: tuple[str, ...] | None = None,
) -> TrustReport:
    """Calcula el Data Trust Score del dataset.

    Funciona igual sobre las tablas crudas de `data/raw` y sobre las limpias de la Fase 2:
    todas las reglas se escriben contra columnas que existen en ambas versiones.

    Args:
        tables: Las 7 tablas indexadas por nombre (`customers`, `products`, ...).
        weights: Peso de cada dimension. Por defecto todas al 20 %.
        only_tables: Limita el calculo a un subconjunto de tablas. Las reglas que cruzan
            tablas se saltan si les falta alguna.

    Returns:
        Un `TrustReport` con el detalle por comprobacion, por dimension y la nota global.

    Raises:
        KeyError: Si falta alguna de las 7 tablas.
    """
    missing = set(_TABLE_CHECKS) - set(tables)
    if missing:
        raise KeyError(f"Faltan tablas para el Data Trust Score: {sorted(missing)}")

    selected = only_tables or tuple(_TABLE_CHECKS)
    checks: list[TrustCheck] = []
    for name in selected:
        checks.extend(_TABLE_CHECKS[name](tables))

    return TrustReport(tuple(checks), dict(weights or DEFAULT_WEIGHTS))

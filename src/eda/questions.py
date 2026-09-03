"""Las nueve preguntas de negocio de la Tarea 1, resueltas con Spark SQL.

Cada pregunta del EDA es aqui una funcion que devuelve un DataFrame. El notebook
(`notebooks/01_eda.ipynb`) las llama y las pinta; el informe de hallazgos de la Fase 5
(`src/eda/findings.py`) las llama y las escribe. Ninguno de los dos calcula nada por su
cuenta, que es lo que pide `CLAUDE.md` ("la logica de negocio vive en `src/`, los
notebooks solo la invocan y visualizan").

Tenerlas como funciones tiene un segundo efecto, que es el que motivo la extraccion: se
pueden **testear**. `tests/test_eda_questions.py` monta un dataset diminuto donde cada
respuesta se calcula a mano y comprueba las nueve. Una consulta dentro de una celda de
notebook no se puede verificar mas que mirandola.

## Sobre que dato corren

Sobre `data/processed`, no sobre `data/raw`: son las tablas ya limpias que deja
`python -m src.etl.run_etl`, con sus columnas derivadas (`basket_day`, `line_amount`,
`n_lines`, `is_promo`...). `register_views` las registra como vistas temporales con el
nombre de la tabla, de modo que el SQL de cada funcion se puede copiar tal cual a
cualquier motor.

## Las nueve preguntas

| # | Pregunta | Alimenta a |
| --- | --- | --- |
| Q1 | Quien sostiene la facturacion | Segmentacion, NBA |
| Q2 | Que se vende y cuanta marca blanca hay | Catalogo del recomendador |
| Q3 | Que categorias son estacionales | Candidatos por estacionalidad |
| Q4 | Como es la cesta media por canal | Contexto de la cesta en curso |
| Q5 | Que se compra junto | Candidatos de co-compra (Fase 3) |
| Q6 | Cual es el ciclo de recompra | `due_for_repurchase` (Tarea 2) |
| Q7 | Que diferencia a quien abandona | Propension de churn (Fase 4) |
| Q8 | Funcionan las promociones | Feature del ranker |
| Q9 | Como es el embudo online | Senal de sesion |
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from src.etl.schemas import read_processed

# Vistas que necesita el conjunto de las nueve preguntas: las 7 tablas limpias y las 4 de
# features de la Fase 2.
REQUIRED_VIEWS: tuple[str, ...] = (
    "customers",
    "products",
    "promotions",
    "baskets",
    "basket_items",
    "sessions",
    "session_events",
    "rfm",
    "repurchase_features",
    "affinity_category",
    "affinity_product",
)

# Las 9 categorias con pico estacional declaradas en `DATA_SPEC.md`.
SEASONAL_CATEGORIES: tuple[str, ...] = (
    "Turron y mazapan",
    "Cava y espumosos",
    "Marisco",
    "Helados",
    "Protector solar",
    "Torrijas y bolleria de Cuaresma",
    "Bacalao",
    "Chocolate y huevos de Pascua",
    "Sopas y caldos",
)

# Los 10 pares de afinidad inyectados, con el lift que `DATA_SPEC.md` declara como
# objetivo. Se comparan contra lo medido en Q5.
EXPECTED_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("Cerveza", "Snacks y aperitivos", 3.0),
    ("Pasta", "Salsa de tomate", 2.5),
    ("Panales", "Toallitas humedas", 4.0),
    ("Cafe", "Azucar y edulcorante", 2.0),
    ("Pan", "Embutido y fiambre", 2.2),
    ("Cereales", "Leche", 2.8),
    ("Vino", "Queso", 2.5),
    ("Detergente", "Suavizante", 3.5),
    ("Champu", "Acondicionador", 3.0),
    ("Palomitas de microondas", "Refrescos", 2.0),
)

# Segmentos RFM que la Fase 4 considera accionables por riesgo de abandono.
AT_RISK_SEGMENTS: tuple[str, ...] = ("En riesgo", "Hibernando", "Necesitan atencion")

# Categorias de rotacion muy distinta con las que se ilustra el efecto del hogar en Q6.
HOUSEHOLD_CYCLE_CATEGORIES: tuple[str, ...] = (
    "Leche",
    "Pan",
    "Papel higienico",
    "Detergente",
)


def register_views(
    spark: SparkSession,
    processed_dir: str | Path = "data/processed",
    views: tuple[str, ...] = REQUIRED_VIEWS,
) -> dict[str, DataFrame]:
    """Registra las tablas procesadas como vistas temporales con su propio nombre.

    Args:
        spark: Sesion activa.
        processed_dir: Carpeta con la salida de `python -m src.etl.run_etl`.
        views: Tablas a registrar. Por defecto, las once que usan las nueve preguntas.

    Returns:
        Las tablas leidas, indexadas por nombre, por si el llamante las necesita como
        DataFrame ademas de como vista.
    """
    tables = read_processed(spark, views, processed_dir)
    for name, df in tables.items():
        df.createOrReplaceTempView(name)
    return tables


# ======================================================================================
# Q1 - Como se reparte la base de clientes y quien sostiene la facturacion
# ======================================================================================
def q1_value_by_tier(spark: SparkSession) -> DataFrame:
    """Clientes, cestas, facturacion y ticket medio por nivel de fidelidad.

    El `LEFT JOIN` es deliberado: un cliente sin ninguna compra tiene que seguir contando
    en el denominador de "cuantos clientes hay en este nivel".
    """
    return spark.sql(
        """
        SELECT
            c.loyalty_tier,
            COUNT(DISTINCT c.customer_id)              AS clientes,
            COUNT(b.basket_id)                         AS cestas,
            ROUND(SUM(COALESCE(b.total_amount, 0)), 2) AS facturacion,
            ROUND(AVG(b.total_amount), 2)              AS ticket_medio
        FROM customers c
        LEFT JOIN baskets b ON b.customer_id = c.customer_id
        GROUP BY c.loyalty_tier
        ORDER BY facturacion DESC
        """
    )


def q1_basket_by_household(spark: SparkSession) -> DataFrame:
    """Ticket medio y frecuencia por tamano de hogar."""
    return spark.sql(
        """
        SELECT
            c.household_size_est                                         AS tamano_hogar,
            COUNT(DISTINCT c.customer_id)                                AS clientes,
            ROUND(AVG(b.total_amount), 2)                                AS ticket_medio,
            ROUND(COUNT(b.basket_id) / COUNT(DISTINCT c.customer_id), 1) AS cestas_por_cliente
        FROM customers c
        LEFT JOIN baskets b ON b.customer_id = c.customer_id
        GROUP BY c.household_size_est
        ORDER BY tamano_hogar
        """
    )


def add_share_columns(
    pdf: pd.DataFrame, *, count_col: str = "clientes", amount_col: str = "facturacion"
) -> pd.DataFrame:
    """Anade el peso en clientes y en facturacion de cada fila, en porcentaje.

    Es la comparacion que responde a Q1: un nivel de fidelidad importa por lo que aporta,
    no por cuanta gente tiene. El cociente entre ambos pesos es el indice de valor.
    """
    out = pdf.copy()
    out["pct_clientes"] = 100 * out[count_col] / out[count_col].sum()
    out["pct_facturacion"] = 100 * out[amount_col] / out[amount_col].sum()
    out["indice_valor"] = (out["pct_facturacion"] / out["pct_clientes"]).round(2)
    return out


# ======================================================================================
# Q2 - Que se vende y que peso tiene la marca blanca
# ======================================================================================
def q2_sales_by_category(spark: SparkSession) -> DataFrame:
    """Cestas, unidades e importe por departamento y categoria."""
    return spark.sql(
        """
        SELECT
            p.department                 AS departamento,
            p.category                   AS categoria,
            COUNT(DISTINCT bi.basket_id) AS cestas,
            SUM(bi.quantity)             AS unidades,
            ROUND(SUM(bi.line_amount), 2) AS importe
        FROM basket_items bi
        JOIN products p ON p.product_id = bi.product_id
        GROUP BY p.department, p.category
        ORDER BY importe DESC
        """
    )


def q2_private_label_by_department(spark: SparkSession) -> DataFrame:
    """Cuota de marca blanca sobre el importe de cada departamento."""
    return spark.sql(
        """
        SELECT
            p.department                                                       AS departamento,
            ROUND(100 * SUM(CASE WHEN p.is_private_label THEN bi.line_amount ELSE 0 END)
                      / SUM(bi.line_amount), 2)                                AS pct_marca_blanca,
            ROUND(SUM(bi.line_amount), 2)                                      AS importe
        FROM basket_items bi
        JOIN products p ON p.product_id = bi.product_id
        GROUP BY p.department
        ORDER BY importe DESC
        """
    )


def categories_covering(
    pdf: pd.DataFrame, *, amount_col: str = "importe", threshold: float = 80.0
) -> int:
    """Cuantas categorias hacen falta para acumular `threshold` % del importe.

    Mide la concentracion del surtido, que es lo que le dice al recomendador cuanta cola
    larga tiene que cubrir. Espera el DataFrame ya ordenado de mayor a menor importe.
    """
    if not 0 < threshold <= 100:
        raise ValueError("threshold debe estar en (0, 100]")
    share = 100 * pdf[amount_col] / pdf[amount_col].sum()
    return int((share.cumsum() < threshold).sum() + 1)


def private_label_share(pdf: pd.DataFrame) -> float:
    """Cuota global de marca blanca, ponderando cada departamento por su importe."""
    return float(
        (pdf["pct_marca_blanca"] * pdf["importe"]).sum() / pdf["importe"].sum()
    )


# ======================================================================================
# Q3 - Como evoluciona la venta y que categorias son estacionales
# ======================================================================================
def q3_monthly_sales(spark: SparkSession) -> DataFrame:
    """Importe y numero de cestas por mes natural."""
    return spark.sql(
        """
        SELECT
            date_format(b.basket_day, 'yyyy-MM') AS mes,
            ROUND(SUM(b.total_amount), 2)        AS importe,
            COUNT(*)                             AS cestas
        FROM baskets b
        GROUP BY date_format(b.basket_day, 'yyyy-MM')
        ORDER BY mes
        """
    )


def q3_seasonal_index(spark: SparkSession) -> DataFrame:
    """Indice estacional por categoria y mes del ano.

    No se mira el importe absoluto sino la **cuota de la categoria sobre la venta del
    mes**, dividida por su cuota media anual. Asi "en diciembre se vende mas de todo" no
    se confunde con "en diciembre se vende mas turron": un indice de 8 significa que la
    categoria pesa ocho veces mas de lo habitual dentro de ese mes.
    """
    return spark.sql(
        """
        WITH linea AS (
            SELECT p.category AS categoria, MONTH(b.basket_day) AS mes, bi.line_amount
            FROM basket_items bi
            JOIN baskets  b ON b.basket_id  = bi.basket_id
            JOIN products p ON p.product_id = bi.product_id
        ),
        por_mes AS (
            SELECT categoria, mes, SUM(line_amount) AS importe
            FROM linea GROUP BY categoria, mes
        ),
        total_mes AS (
            SELECT mes, SUM(importe) AS importe_mes FROM por_mes GROUP BY mes
        ),
        cuota AS (
            SELECT p.categoria, p.mes, p.importe / t.importe_mes AS cuota
            FROM por_mes p JOIN total_mes t ON t.mes = p.mes
        )
        SELECT
            categoria,
            mes,
            ROUND(cuota / AVG(cuota) OVER (PARTITION BY categoria), 3) AS indice_estacional
        FROM cuota
        """
    )


def seasonal_matrix(
    pdf: pd.DataFrame, categories: tuple[str, ...] = SEASONAL_CATEGORIES
) -> pd.DataFrame:
    """Pivota el indice estacional a categoria x mes, en el orden declarado."""
    subset = pdf[pdf["categoria"].isin(categories)]
    return subset.pivot(
        index="categoria", columns="mes", values="indice_estacional"
    ).reindex(list(categories))


def seasonal_peaks(matrix: pd.DataFrame) -> pd.DataFrame:
    """Mes pico y valor del indice en ese mes, por categoria."""
    return (
        matrix.idxmax(axis=1)
        .rename("mes_pico")
        .to_frame()
        .join(matrix.max(axis=1).round(2).rename("indice_en_el_pico"))
    )


# ======================================================================================
# Q4 - Como es la cesta media y en que se diferencia por canal
# ======================================================================================
def q4_basket_by_channel(spark: SparkSession) -> DataFrame:
    """Tamano y composicion de la cesta por canal."""
    return spark.sql(
        """
        SELECT
            channel                                        AS canal,
            COUNT(*)                                       AS cestas,
            ROUND(AVG(n_lines), 2)                         AS lineas_medias,
            ROUND(AVG(n_units), 2)                         AS unidades_medias,
            ROUND(AVG(total_amount), 2)                    AS ticket_medio,
            ROUND(100 * AVG(CAST(is_anonymous AS INT)), 2) AS pct_anonimas
        FROM baskets
        GROUP BY channel
        ORDER BY cestas DESC
        """
    )


def q4_hour_profile(spark: SparkSession) -> DataFrame:
    """Reparto horario de las cestas de cada canal."""
    return spark.sql(
        """
        SELECT channel AS canal, HOUR(basket_date) AS hora, COUNT(*) AS cestas
        FROM baskets
        GROUP BY channel, HOUR(basket_date)
        ORDER BY canal, hora
        """
    )


def q4_lines_distribution(spark: SparkSession, *, max_lines: int = 15) -> DataFrame:
    """Distribucion del numero de lineas por cesta, hasta `max_lines`."""
    return spark.sql(
        f"""
        SELECT n_lines AS lineas, COUNT(*) AS cestas
        FROM baskets
        WHERE n_lines BETWEEN 1 AND {int(max_lines)}
        GROUP BY n_lines
        ORDER BY lineas
        """
    )


# ======================================================================================
# Q5 - Que categorias se compran juntas
# ======================================================================================
def q5_top_affinity_pairs(
    spark: SparkSession, *, min_baskets: int = 500, limit: int = 20
) -> DataFrame:
    """Pares de categorias con mas lift, exigiendo un soporte minimo.

    El filtro por `n_baskets_both` no es cosmetico: sin el, arriba del ranking solo
    aparecen pares rarisimos cuyo lift es enorme porque coincidieron tres veces.
    """
    return spark.sql(
        f"""
        SELECT
            antecedent              AS disparadora,
            consequent              AS asociada,
            n_baskets_both          AS cestas_juntas,
            ROUND(100 * support, 3) AS pct_soporte,
            ROUND(100 * confidence, 2) AS pct_confianza,
            lift
        FROM affinity_category
        WHERE n_baskets_both >= {int(min_baskets)}
        ORDER BY lift DESC
        LIMIT {int(limit)}
        """
    )


def q5_expected_pairs(
    spark: SparkSession, expected: tuple[tuple[str, str, float], ...] = EXPECTED_PAIRS
) -> pd.DataFrame:
    """Los 10 pares declarados en `DATA_SPEC.md`, con su lift objetivo y el medido.

    Devuelve pandas y no Spark a proposito: la lista de pares esperados es una constante
    de Python de diez filas, y subirla a Spark para hacer un join de diez filas no aporta
    nada.
    """
    measured = spark.sql(
        """
        SELECT antecedent AS disparadora, consequent AS asociada, lift AS lift_medido,
               ROUND(100 * confidence, 2) AS pct_confianza, n_baskets_both AS cestas_juntas
        FROM affinity_category
        """
    ).toPandas()
    target = pd.DataFrame(
        list(expected), columns=["disparadora", "asociada", "lift_objetivo"]
    )
    out = target.merge(measured, on=["disparadora", "asociada"], how="left")
    out["ratio"] = (out["lift_medido"] / out["lift_objetivo"]).round(2)
    return out


# ======================================================================================
# Q6 - Cual es el ciclo de recompra y como varia con el hogar
# ======================================================================================
def q6_cycle_by_category(spark: SparkSession, *, min_customers: int = 200) -> DataFrame:
    """Ciclo teorico frente a ciclo observado (mediana) por categoria.

    Es la validacion de la Tarea 2: si el observado no reprodujera el orden del teorico,
    `due_for_repurchase` estaria marcando ruido.
    """
    return spark.sql(
        f"""
        SELECT
            category                                                   AS categoria,
            typical_repurchase_days                                    AS ciclo_teorico,
            ROUND(percentile_approx(observed_repurchase_days, 0.5), 1) AS ciclo_observado,
            COUNT(*)                                                   AS clientes,
            ROUND(100 * AVG(CAST(due_for_repurchase AS INT)), 1)       AS pct_toca_reponer
        FROM repurchase_features
        WHERE observed_repurchase_days IS NOT NULL
        GROUP BY category, typical_repurchase_days
        HAVING COUNT(*) >= {int(min_customers)}
        ORDER BY ciclo_teorico
        """
    )


def q6_cycle_by_household(
    spark: SparkSession, categories: tuple[str, ...] = HOUSEHOLD_CYCLE_CATEGORIES
) -> DataFrame:
    """Ciclo observado por tamano de hogar en unas pocas categorias de referencia."""
    lista = ", ".join(f"'{c}'" for c in categories)
    return spark.sql(
        f"""
        SELECT
            c.household_size_est                                         AS tamano_hogar,
            r.category                                                   AS categoria,
            ROUND(percentile_approx(r.observed_repurchase_days, 0.5), 1) AS ciclo_observado
        FROM repurchase_features r
        JOIN customers c ON c.customer_id = r.customer_id
        WHERE r.observed_repurchase_days IS NOT NULL
          AND r.category IN ({lista})
        GROUP BY c.household_size_est, r.category
        ORDER BY categoria, tamano_hogar
        """
    )


def q6_due_summary(spark: SparkSession) -> DataFrame:
    """Cuantos pares cliente-categoria tienen la recompra vencida ahora mismo."""
    return spark.sql(
        """
        SELECT
            COUNT(*)                                                          AS pares_cliente_categoria,
            SUM(CAST(due_for_repurchase AS INT))                              AS con_recompra_vencida,
            ROUND(100 * AVG(CAST(due_for_repurchase AS INT)), 1)              AS pct,
            COUNT(DISTINCT CASE WHEN due_for_repurchase THEN customer_id END) AS clientes_afectados
        FROM repurchase_features
        """
    )


def cycle_rank_correlation(pdf: pd.DataFrame) -> float:
    """Correlacion de rangos (Spearman) entre ciclo teorico y observado.

    Se usa Spearman y no Pearson porque lo que importa es que el **orden** se conserve
    (la leche antes que el detergente), no que los dias coincidan: el ciclo observado
    esta truncado por la frecuencia de visita del cliente y no puede coincidir en
    magnitud con el teorico en las categorias largas.
    """
    return float(
        pdf[["ciclo_teorico", "ciclo_observado"]].corr(method="spearman").iloc[0, 1]
    )


# ======================================================================================
# Q7 - Que diferencia a quien abandona y cuanto valor esta en riesgo
# ======================================================================================
def q7_rfm_segments(spark: SparkSession) -> DataFrame:
    """Tamano, perfil RFM y tasa de abandono de cada segmento."""
    return spark.sql(
        """
        SELECT
            rfm_segment                                   AS segmento,
            COUNT(*)                                      AS clientes,
            ROUND(AVG(recency_days), 1)                   AS recencia_media,
            ROUND(AVG(frequency), 1)                      AS frecuencia_media,
            ROUND(AVG(monetary), 2)                       AS gasto_medio,
            ROUND(SUM(monetary), 2)                       AS gasto_total,
            ROUND(100 * AVG(CAST(churn_label AS INT)), 1) AS pct_churn
        FROM rfm
        GROUP BY rfm_segment
        ORDER BY gasto_total DESC
        """
    )


def q7_churn_profile(spark: SparkSession) -> DataFrame:
    """Perfil medio de quien abandona frente a quien sigue activo."""
    return spark.sql(
        """
        SELECT
            CASE WHEN churn_label THEN 'Abandonan' ELSE 'Activos' END AS estado,
            COUNT(*)                                                  AS clientes,
            ROUND(AVG(recency_days), 1)                               AS recencia_media,
            ROUND(AVG(frequency), 1)                                  AS frecuencia_media,
            ROUND(AVG(avg_ticket), 2)                                 AS ticket_medio,
            ROUND(AVG(avg_days_between_baskets), 1)                   AS dias_entre_cestas,
            ROUND(SUM(monetary), 2)                                   AS gasto_total
        FROM rfm
        WHERE has_purchases
        GROUP BY churn_label
        """
    )


def at_risk_share(
    pdf: pd.DataFrame, segments: tuple[str, ...] = AT_RISK_SEGMENTS
) -> dict[str, float]:
    """Cuantos clientes y cuanto gasto historico concentran los segmentos de riesgo."""
    at_risk = pdf[pdf["segmento"].isin(segments)]
    total_spend = float(pdf["gasto_total"].sum())
    return {
        "clientes": int(at_risk["clientes"].sum()),
        "pct_clientes": 100 * float(at_risk["clientes"].sum()) / float(pdf["clientes"].sum()),
        "gasto_total": float(at_risk["gasto_total"].sum()),
        "pct_gasto": 100 * float(at_risk["gasto_total"].sum()) / total_spend,
    }


# ======================================================================================
# Q8 - Funcionan las promociones
# ======================================================================================
def q8_promo_uplift(spark: SparkSession) -> DataFrame:
    """Unidades por dia del mismo producto, dentro y fuera de su ventana promocional.

    Se construye la rejilla completa `producto promocionado x dia con actividad` y se
    imputa cero a los dias sin venta. Sin esa rejilla el calculo premiaria a la promocion
    por partida doble: solo se contarian los dias en que hubo venta, que es justo lo que
    la promocion cambia.
    """
    return spark.sql(
        """
        WITH calendario AS (
            SELECT DISTINCT basket_day AS dia FROM baskets
        ),
        promocionados AS (
            SELECT DISTINCT product_id FROM promotions
        ),
        rejilla AS (
            SELECT p.product_id, c.dia
            FROM promocionados p CROSS JOIN calendario c
        ),
        dias_con_promo AS (
            SELECT DISTINCT pm.product_id, c.dia
            FROM promotions pm
            JOIN calendario c ON c.dia BETWEEN pm.start_date AND pm.end_date
        ),
        ventas AS (
            SELECT bi.product_id, b.basket_day AS dia, SUM(bi.quantity) AS unidades
            FROM basket_items bi
            JOIN baskets b ON b.basket_id = bi.basket_id
            GROUP BY bi.product_id, b.basket_day
        )
        SELECT
            CASE WHEN d.product_id IS NOT NULL THEN 'Con promocion' ELSE 'Sin promocion' END AS estado,
            COUNT(*)                                          AS dias_producto,
            ROUND(SUM(COALESCE(v.unidades, 0)) / COUNT(*), 4) AS unidades_por_dia
        FROM rejilla r
        LEFT JOIN dias_con_promo d ON d.product_id = r.product_id AND d.dia = r.dia
        LEFT JOIN ventas         v ON v.product_id = r.product_id AND v.dia = r.dia
        GROUP BY 1
        ORDER BY estado
        """
    )


def q8_promo_share_by_department(spark: SparkSession) -> DataFrame:
    """Que parte del importe de cada departamento se vende en promocion."""
    return spark.sql(
        """
        SELECT
            p.department                                                        AS departamento,
            ROUND(100 * SUM(CASE WHEN bi.is_promo THEN bi.line_amount ELSE 0 END)
                      / SUM(bi.line_amount), 2)                                 AS pct_venta_en_promo
        FROM basket_items bi
        JOIN products p ON p.product_id = bi.product_id
        GROUP BY p.department
        ORDER BY pct_venta_en_promo DESC
        """
    )


def q8_promo_by_type(spark: SparkSession) -> DataFrame:
    """Importe y unidades por linea de cada mecanica promocional."""
    return spark.sql(
        """
        SELECT
            pm.promo_type                   AS tipo,
            COUNT(DISTINCT pm.promotion_id) AS promociones,
            COUNT(*)                        AS lineas,
            ROUND(SUM(bi.line_amount), 2)   AS importe,
            ROUND(AVG(bi.quantity), 2)      AS unidades_por_linea
        FROM basket_items bi
        JOIN promotions pm ON pm.promotion_id = bi.promotion_id
        GROUP BY pm.promo_type
        ORDER BY importe DESC
        """
    )


def promo_uplift_ratio(pdf: pd.DataFrame) -> float:
    """Cuantas veces mas vende al dia un producto mientras esta en promocion."""
    rate = pdf.set_index("estado")["unidades_por_dia"]
    return float(rate["Con promocion"] / rate["Sin promocion"])


# ======================================================================================
# Q9 - Como es el embudo online
# ======================================================================================
def q9_funnel_by_device(spark: SparkSession) -> DataFrame:
    """Sesiones, vistas, anadidos al carrito y conversion por dispositivo."""
    return spark.sql(
        """
        WITH eventos AS (
            SELECT
                session_id,
                MAX(CASE WHEN event_type = 'view'        THEN 1 ELSE 0 END) AS vio,
                MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS anadio
            FROM session_events
            GROUP BY session_id
        )
        SELECT
            s.device_type                                AS dispositivo,
            COUNT(*)                                     AS sesiones,
            SUM(COALESCE(e.vio, 0))                      AS con_vista,
            SUM(COALESCE(e.anadio, 0))                   AS con_add_to_cart,
            SUM(CASE WHEN s.converted THEN 1 ELSE 0 END) AS convertidas,
            ROUND(100 * AVG(CAST(s.converted AS INT)), 2) AS pct_conversion
        FROM sessions s
        LEFT JOIN eventos e ON e.session_id = s.session_id
        GROUP BY s.device_type
        ORDER BY sesiones DESC
        """
    )


def q9_event_to_basket_overlap(
    spark: SparkSession, *, event_type: str = "add_to_cart"
) -> DataFrame:
    """De lo que se ve o se anade al carrito en una sesion, cuanto acaba en el ticket.

    Es la comprobacion que destapo la senal de sesion contaminada de la Fase 2: cuando
    esta cifra sale 100 %, el evento online **es** el ticket escrito de otra forma y
    usarlo como feature seria filtrar el target.
    """
    if event_type not in {"view", "add_to_cart"}:
        raise ValueError("event_type debe ser 'view' o 'add_to_cart'")
    return spark.sql(
        f"""
        WITH eventos AS (
            SELECT DISTINCT session_id, product_id
            FROM session_events
            WHERE event_type = '{event_type}'
        ),
        convertidas AS (
            SELECT session_id, basket_id FROM sessions WHERE converted
        )
        SELECT
            '{event_type}'                                                            AS evento,
            COUNT(*)                                                                  AS productos,
            SUM(CASE WHEN bi.product_id IS NOT NULL THEN 1 ELSE 0 END)                AS acabaron_en_la_cesta,
            ROUND(100 * AVG(CASE WHEN bi.product_id IS NOT NULL THEN 1 ELSE 0 END), 2) AS pct
        FROM eventos a
        JOIN convertidas c ON c.session_id = a.session_id
        LEFT JOIN basket_items bi
               ON bi.basket_id = c.basket_id AND bi.product_id = a.product_id
        """
    )


def q9_basket_seen_online(spark: SparkSession) -> DataFrame:
    """El sentido contrario: que parte de las lineas del ticket dejo rastro online.

    Complementa a `q9_event_to_basket_overlap`. Que las dos direcciones esten por debajo
    del 100 % es lo que hace la senal de sesion utilizable en vez de tautologica.
    """
    return spark.sql(
        """
        WITH convertidas AS (
            SELECT session_id, basket_id FROM sessions WHERE converted
        ),
        vistos AS (
            SELECT DISTINCT session_id, product_id FROM session_events
        )
        SELECT
            COUNT(*)                                                                  AS lineas_del_ticket,
            SUM(CASE WHEN v.product_id IS NOT NULL THEN 1 ELSE 0 END)                 AS con_rastro_online,
            ROUND(100 * AVG(CASE WHEN v.product_id IS NOT NULL THEN 1 ELSE 0 END), 2) AS pct
        FROM convertidas c
        JOIN basket_items bi ON bi.basket_id = c.basket_id
        LEFT JOIN vistos v ON v.session_id = c.session_id AND v.product_id = bi.product_id
        """
    )

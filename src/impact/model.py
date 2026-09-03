"""La aritmetica del resumen de impacto, en funciones puras.

Nada de esto lee ficheros ni escribe informes: recibe numeros y devuelve numeros, para
que `tests/test_impact.py` pueda comprobar cada paso a mano. `pipeline.py` es quien mide
las entradas y escribe la salida.

## El modelo de cross-sell (Fase 3)

El recomendador se evalua con `hit_rate@5`: la fraccion de cestas en las que al menos uno
de los cinco huecos contiene un producto que el cliente iba a comprar. Frente al baseline
sin aprendizaje -- ordenar por popularidad reciente x indice estacional -- la diferencia
es lo que aporta el ranker:

    cestas_con_acierto_nuevo = cestas_online_mes x (hit_rate_ranker - hit_rate_baseline)
    unidades_extra           = cestas_con_acierto_nuevo x incrementalidad
    venta_extra              = unidades_extra x importe_medio_de_linea
    margen_extra             = venta_extra x margen_bruto_mezclado

Es deliberadamente **conservador** por dos lados: cuenta una sola unidad por cesta
(aunque el top-5 acierte dos), y compara contra un baseline que ya es razonable, no
contra no recomendar nada.

## El modelo de NBA (Fase 4)

Aqui no hay que traducir nada: `policy.compare` ya devuelve euros de valor incremental
esperado sobre "no actuar". Lo unico que anade este modulo es la escala temporal --
cuantas oleadas de campana caben en un ano -- y la normalizacion por cliente.
"""

from __future__ import annotations

from dataclasses import dataclass

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class CrossSellImpact:
    """Impacto del recomendador sobre una base de cestas online al mes."""

    online_baskets_per_month: float
    hit_rate_model: float
    hit_rate_baseline: float
    incremental_rate: float
    avg_line_amount: float
    margin_rate: float

    @property
    def hit_rate_delta(self) -> float:
        """Puntos de `hit_rate@5` que anade el ranker sobre el baseline."""
        return self.hit_rate_model - self.hit_rate_baseline

    @property
    def relative_lift(self) -> float:
        """Mejora relativa sobre el baseline. Es la unica cifra sin supuestos."""
        if self.hit_rate_baseline == 0:
            raise ValueError("el baseline no puede tener hit_rate cero")
        return self.hit_rate_delta / self.hit_rate_baseline

    @property
    def baskets_with_new_hit(self) -> float:
        """Cestas al mes en las que el ranker pone algo relevante y el baseline no."""
        return self.online_baskets_per_month * self.hit_rate_delta

    @property
    def extra_units_per_month(self) -> float:
        return self.baskets_with_new_hit * self.incremental_rate

    @property
    def revenue_per_month(self) -> float:
        return self.extra_units_per_month * self.avg_line_amount

    @property
    def margin_per_month(self) -> float:
        return self.revenue_per_month * self.margin_rate

    @property
    def margin_per_year(self) -> float:
        return self.margin_per_month * MONTHS_PER_YEAR

    def rescaled(self, online_baskets_per_month: float) -> "CrossSellImpact":
        """El mismo modelo sobre otra base de cestas, para leerlo a escala reconocible."""
        return CrossSellImpact(
            online_baskets_per_month=online_baskets_per_month,
            hit_rate_model=self.hit_rate_model,
            hit_rate_baseline=self.hit_rate_baseline,
            incremental_rate=self.incremental_rate,
            avg_line_amount=self.avg_line_amount,
            margin_rate=self.margin_rate,
        )

    def with_incremental_rate(self, rate: float) -> "CrossSellImpact":
        """Copia con otro supuesto de incrementalidad. Es el gancho del barrido."""
        return CrossSellImpact(
            online_baskets_per_month=self.online_baskets_per_month,
            hit_rate_model=self.hit_rate_model,
            hit_rate_baseline=self.hit_rate_baseline,
            incremental_rate=rate,
            avg_line_amount=self.avg_line_amount,
            margin_rate=self.margin_rate,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "cestas_online_mes": self.online_baskets_per_month,
            "hit_rate_modelo": self.hit_rate_model,
            "hit_rate_baseline": self.hit_rate_baseline,
            "delta_hit_rate": self.hit_rate_delta,
            "mejora_relativa": self.relative_lift,
            "incrementalidad": self.incremental_rate,
            "cestas_con_acierto_nuevo_mes": self.baskets_with_new_hit,
            "unidades_extra_mes": self.extra_units_per_month,
            "venta_extra_mes": self.revenue_per_month,
            "margen_extra_mes": self.margin_per_month,
            "margen_extra_ano": self.margin_per_year,
        }


@dataclass(frozen=True)
class NBAImpact:
    """Impacto de la politica de Next Best Action sobre una base de clientes."""

    customers: int
    policy_value_per_wave: float
    best_trivial_value_per_wave: float
    floor_value_per_wave: float
    waves_per_year: int

    @property
    def value_per_customer_per_wave(self) -> float:
        if self.customers <= 0:
            raise ValueError("la base de clientes debe ser positiva")
        return self.policy_value_per_wave / self.customers

    @property
    def value_per_year(self) -> float:
        return self.policy_value_per_wave * self.waves_per_year

    @property
    def uplift_vs_trivial_per_wave(self) -> float:
        """Lo que aporta *elegir a quien* frente a la mejor campana no segmentada."""
        return self.policy_value_per_wave - self.best_trivial_value_per_wave

    @property
    def floor_per_year(self) -> float:
        """Valor anual en el escenario mas pesimista del barrido de la Fase 4."""
        return self.floor_value_per_wave * self.waves_per_year

    def rescaled(self, customers: int) -> "NBAImpact":
        """El mismo valor por cliente, proyectado a otra base."""
        factor = customers / self.customers
        return NBAImpact(
            customers=customers,
            policy_value_per_wave=self.policy_value_per_wave * factor,
            best_trivial_value_per_wave=self.best_trivial_value_per_wave * factor,
            floor_value_per_wave=self.floor_value_per_wave * factor,
            waves_per_year=self.waves_per_year,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "clientes": self.customers,
            "valor_politica_oleada": self.policy_value_per_wave,
            "valor_mejor_trivial_oleada": self.best_trivial_value_per_wave,
            "valor_suelo_oleada": self.floor_value_per_wave,
            "uplift_vs_trivial_oleada": self.uplift_vs_trivial_per_wave,
            "valor_por_cliente_oleada": self.value_per_customer_per_wave,
            "oleadas_ano": self.waves_per_year,
            "valor_ano": self.value_per_year,
            "valor_suelo_ano": self.floor_per_year,
        }


def blended_margin_rate(
    sales_by_department: dict[str, float], margins: dict[str, float], default: float
) -> float:
    """Margen bruto medio, ponderando cada departamento por lo que vende.

    Un margen unico para toda la tienda mentiria en las dos direcciones: el fresco es un
    tercio de la venta y deja un 18 %, la drogueria deja un 35 % y vende ocho veces menos.
    """
    total = sum(sales_by_department.values())
    if total <= 0:
        raise ValueError("no hay venta sobre la que ponderar el margen")
    weighted = sum(
        amount * margins.get(department, default)
        for department, amount in sales_by_department.items()
    )
    return weighted / total


def sweep_incremental_rate(
    impact: CrossSellImpact, rates: tuple[float, ...]
) -> list[dict[str, float]]:
    """Una fila por supuesto de incrementalidad, para publicar el barrido entero."""
    return [
        {
            "incrementalidad": rate,
            "unidades_extra_mes": variant.extra_units_per_month,
            "venta_extra_mes": variant.revenue_per_month,
            "margen_extra_mes": variant.margin_per_month,
            "margen_extra_ano": variant.margin_per_year,
        }
        for rate in rates
        for variant in [impact.with_incremental_rate(rate)]
    ]

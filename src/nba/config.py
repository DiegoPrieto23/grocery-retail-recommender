"""Parametros de la Fase 4: cortes temporales, catalogo de acciones y economia.

Este fichero concentra **todo lo que es un supuesto** y no un dato medido. Es deliberado:
la Tarea 3b mezcla dos cosas de naturaleza muy distinta, y confundirlas es la forma
habitual de contar una mentira con cifras.

- Lo que **se mide**: la propension. `P(compra en categoria en 7 dias)` y `P(churn en 4
  semanas)` se estiman del historico y se evaluan con AUC/PR-AUC sobre un periodo
  posterior. Eso es un resultado.
- Lo que **se supone**: el efecto de cada accion. `P(conversion | accion=a)` exige saber
  como cambia la conducta al actuar, y **este dataset no lo contiene**: el generador
  aplica `PROMO_UPLIFT = 3.0` al reparto de cuota *dentro* de una categoria (que SKU se
  elige), no a la probabilidad de comprar la categoria ni a la de volver. El tratamiento
  nunca varia, asi que el efecto causal no es identificable. Los multiplicadores de abajo
  son supuestos declarados, no estimaciones.

Que el uplift sea un supuesto **no vacia de contenido la politica**. Lo que el modelo
decide es *a quien* se actua: con un cupon que cuesta dinero, acertar el cliente es lo que
separa ganar de perder. Por eso el informe acompana la comparacion con un barrido de
sensibilidad (`policy.sensitivity`) que dice a partir de que uplift la politica deja de
ganar a las dos alternativas triviales. Si la conclusion solo se sostiene con un uplift
heroico, se ve.

## Los cortes

Un "corte" es una foto: se miran solo las compras **anteriores** a esa fecha para construir
las features, y se observa lo que pasa **despues** para etiquetar. Se usan varios cortes de
entrenamiento para tener mas de una foto por cliente, y uno de test muy posterior.

```
   entrenamiento                         validacion        test
   |    |    |    |                          |              |
 04-01 05-15 07-01 08-15                   09-15          11-01     (2025)
```

Ningun corte de entrenamiento observa etiquetas que caigan despues del corte de
validacion, y ninguno de los dos alcanza la ventana de test.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

SEED = 42

# --------------------------------------------------------------------------------------
# Cortes temporales
# --------------------------------------------------------------------------------------
TRAIN_CUTOFFS: tuple[dt.date, ...] = (
    dt.date(2025, 4, 1),
    dt.date(2025, 5, 15),
    dt.date(2025, 7, 1),
    dt.date(2025, 8, 15),
)
VALID_CUTOFF = dt.date(2025, 9, 15)
TEST_CUTOFF = dt.date(2025, 11, 1)

# Horizontes de las dos etiquetas, en dias.
CATEGORY_HORIZON_DAYS = 7
CHURN_HORIZON_DAYS = 28


@dataclass(frozen=True)
class MarginConfig:
    """Margen bruto por departamento.

    **Supuesto declarado** (decision de negocio, no dato del generador). Los ordenes de
    magnitud son los del gran consumo en Espana: el fresco es el gancho y deja poco, la
    droguería y la higiene son las que sostienen la cuenta, y la alimentacion envasada se
    queda en medio. La leche de la puerta de al lado no da 35 %; el gel de bano si.
    """

    by_department: dict[str, float] = field(
        default_factory=lambda: {
            "Frescos": 0.18,
            "Bebe": 0.20,
            "Congelados": 0.22,
            "Bebidas": 0.24,
            "Despensa": 0.25,
            "Mascotas": 0.30,
            "Drogueria": 0.35,
            "Higiene": 0.35,
        }
    )
    # Para un departamento que no este en el mapa (o venga nulo).
    default: float = 0.25

    def rate(self, department: str | None) -> float:
        return self.by_department.get(department or "", self.default)


@dataclass(frozen=True)
class Action:
    """Una accion del catalogo, con su coste y su efecto supuesto.

    El coste se parte en dos porque se comportan de forma distinta, y tratarlos igual
    falsea la cuenta:

    - `send_cost`: se paga **siempre** que se ejecuta la accion (el envio del push, el
      hueco de la newsletter). Es dinero perdido si el cliente no convierte.
    - `discount`: solo se paga **si el cliente compra**, porque es el valor facial del
      cupon que se descuenta del ticket. Por eso entra restando dentro del margen y no
      como un coste fijo.

    `conversion_uplift` y `churn_reduction` son los dos supuestos del modulo:
    multiplicadores sobre la probabilidad basal que estima el modelo de propension.
    """

    action_id: int
    name: str
    send_cost: float
    discount: float
    conversion_uplift: float
    churn_reduction: float
    # Si es False, la accion no necesita elegir una categoria (es la accion nula).
    needs_category: bool = True
    # Si es True, la retencion de la accion se pondera por la necesidad de categoria
    # (ver `PolicyConfig.retention_needs_relevance`). La accion nula no la necesita.
    retention_needs_relevance: bool = True


# El valor facial del cupon no es un numero inventado: es la media de los cupones que ya
# existen en `promotions` del propio dataset (`promo_type = 'coupon'`, media 2,54 EUR).
COUPON_FACE_VALUE = 2.54

ACTIONS: tuple[Action, ...] = (
    Action(
        action_id=0,
        name="ninguna_accion",
        send_cost=0.0,
        discount=0.0,
        conversion_uplift=1.0,
        churn_reduction=0.0,
        needs_category=False,
        retention_needs_relevance=False,
    ),
    Action(
        action_id=1,
        # Actua sobre una **categoria**, no sobre una referencia: la politica elige que
        # categoria destacar y con que uplift supuesto, y no mira el catalogo. Quien elige
        # el producto concreto dentro de esa categoria es el recomendador de la Tarea 3a,
        # en el momento de pintar el hueco (`src/demo/catalog.py`, `action_product`). El
        # nombre anterior, `recomendar_producto`, sugeria que esta accion resolvia las dos
        # cosas (punto M7 de `docs/diagnostico-fase7.md`).
        name="recomendar_categoria",
        # Un hueco de recomendacion en la app no tiene coste marginal real; se le pone un
        # valor simbolico para que la politica no la reparta gratis a todo el mundo.
        send_cost=0.01,
        discount=0.0,
        # Supuesto: una recomendacion bien puesta mueve poco, pero mueve.
        conversion_uplift=1.10,
        churn_reduction=0.02,
    ),
    Action(
        action_id=2,
        name="enviar_cupon_categoria",
        send_cost=0.05,
        discount=COUPON_FACE_VALUE,
        # Supuesto: el cupon mueve mas que la recomendacion, y ademas retiene.
        conversion_uplift=1.35,
        churn_reduction=0.10,
    ),
)

ACTION_BY_NAME: dict[str, Action] = {a.name: a for a in ACTIONS}


@dataclass(frozen=True)
class PolicyConfig:
    """Como se convierte una probabilidad en euros."""

    margins: MarginConfig = field(default_factory=MarginConfig)

    # Peso minimo de relevancia: una oferta de una categoria que el cliente acaba de
    # reponer no retiene a nadie, pero tampoco es exactamente inutil (sigue siendo un
    # gesto comercial). El peso va de este suelo a 1 segun la necesidad de categoria.
    # Ver `retention_needs_relevance` mas abajo.
    min_relevance: float = 0.15

    # Semanas de compra futura que se dan por salvadas al retener a un cliente. Con 4 se
    # mantiene la coherencia con el horizonte de churn de la Tarea 3b.
    retention_weeks: float = 4.0

    # Techo del gasto esperado por categoria, para que un outlier de ticket no dispare el
    # valor esperado de un solo cliente y se lleve todo el presupuesto.
    max_expected_spend: float = 60.0

    # Valores del barrido de sensibilidad sobre el uplift de conversion del cupon.
    sensitivity_uplifts: tuple[float, ...] = (1.00, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00)
    # Y sobre su reduccion de churn, que es el supuesto del que de verdad cuelga el
    # resultado: con un cupon de 2,54 EUR persiguiendo un margen de ~1,4 EUR, el
    # cross-sell del cupon es negativo por construccion y todo lo que aporta es retencion.
    sensitivity_churn_reductions: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20)


@dataclass(frozen=True)
class PropensityConfig:
    """Hiperparametros de los dos LightGBM binarios."""

    objective: str = "binary"
    metric: str = "auc"
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 600
    early_stopping_rounds: int = 50
    seed: int = SEED


@dataclass(frozen=True)
class NBAConfig:
    """Configuracion completa de la Fase 4."""

    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    predictions_dir: Path = Path("predictions")
    reports_dir: Path = Path("reports/nba")

    train_cutoffs: tuple[dt.date, ...] = TRAIN_CUTOFFS
    valid_cutoff: dt.date = VALID_CUTOFF
    test_cutoff: dt.date = TEST_CUTOFF

    category_horizon_days: int = CATEGORY_HORIZON_DAYS
    churn_horizon_days: int = CHURN_HORIZON_DAYS

    # Solo se consideran categorias que el cliente haya comprado en esta ventana previa al
    # corte. Es lo que acota el grano cliente x categoria a algo actuable: una categoria
    # que no toca desde hace un ano no es una siguiente mejor accion, es un desconocido.
    category_lookback_days: int = 180

    # Clientes por corte de entrenamiento. En test entran todos, porque la tabla de salida
    # de la Tarea 3b tiene que cubrir el maestro completo.
    n_train_customers: int = 10_000

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    propensity: PropensityConfig = field(default_factory=PropensityConfig)
    actions: tuple[Action, ...] = ACTIONS
    seed: int = SEED

    def __post_init__(self) -> None:
        last_train = max(self.train_cutoffs)
        if last_train >= self.valid_cutoff:
            raise ValueError("los cortes de entrenamiento deben ser anteriores al de validacion")
        if self.valid_cutoff >= self.test_cutoff:
            raise ValueError("el corte de validacion debe ser anterior al de test")


# Tablas de `data/processed` que necesita la Fase 4.
REQUIRED_TABLES: tuple[str, ...] = (
    "baskets",
    "basket_items",
    "products",
    "customers",
)

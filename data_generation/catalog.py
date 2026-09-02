"""Catalogo de negocio del supermercado sintetico.

Toda la parametrizacion de dominio (departamentos, categorias, ciclos de reposicion,
pares de afinidad de cesta y estacionalidad) vive aqui, separada de la mecanica de
generacion. Los valores de afinidad y estacionalidad son EXACTAMENTE los de
DATA_SPEC.md; si se cambian, hay que actualizar ese documento en el mismo commit.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Periodo simulado
# --------------------------------------------------------------------------------------
# Dos anos naturales completos: cada pico estacional (Navidad, verano, Cuaresma) aparece
# dos veces, que es lo minimo para que un modelo pueda aprender el patron.
PERIOD_START = "2024-01-01"
PERIOD_END = "2025-12-31"

# churn_label = 1 si el cliente no ha comprado en los ultimos CHURN_WINDOW_DAYS del
# periodo simulado (definicion de DATA_SPEC.md).
CHURN_WINDOW_DAYS = 60


# --------------------------------------------------------------------------------------
# Categorias
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Category:
    """Una subcategoria del surtido.

    Attributes:
        name: Nombre de la subcategoria (valor de products.category).
        department: Departamento al que cuelga (valor de products.department).
        repurchase_days: Intervalo tipico de recompra en dias (typical_repurchase_days).
        perishable: Si el producto es perecedero (is_perishable).
        price_lo: Extremo inferior del rango de unit_price.
        price_hi: Extremo superior del rango de unit_price.
        weight: Popularidad base de la categoria antes de estacionalidad y afinidad.
        gate: Restriccion de hogar; solo los clientes con ese atributo la compran.
            "baby" (hogar con bebe), "pet" (hogar con mascota), "alcohol" (consume
            alcohol) o None (sin restriccion).
    """

    name: str
    department: str
    repurchase_days: int
    perishable: bool
    price_lo: float
    price_hi: float
    weight: float
    gate: str | None = None


CATEGORIES: tuple[Category, ...] = (
    # --- Frescos ---
    Category("Leche", "Frescos", 6, True, 0.75, 1.60, 9.0),
    Category("Yogur", "Frescos", 8, True, 1.20, 3.50, 5.0),
    Category("Queso", "Frescos", 12, True, 2.00, 9.50, 4.5),
    Category("Huevos", "Frescos", 10, True, 1.80, 4.20, 5.0),
    Category("Pan", "Frescos", 4, True, 0.70, 2.80, 8.0),
    Category("Embutido y fiambre", "Frescos", 9, True, 1.80, 7.50, 4.5),
    Category("Fruta", "Frescos", 5, True, 0.90, 4.50, 7.5),
    Category("Verdura", "Frescos", 5, True, 0.80, 3.90, 7.0),
    Category("Carne de pollo", "Frescos", 8, True, 3.00, 8.50, 4.5),
    Category("Carne de ternera", "Frescos", 12, True, 5.50, 16.00, 3.0),
    Category("Pescado blanco", "Frescos", 11, True, 4.50, 14.00, 2.8),
    Category("Torrijas y bolleria de Cuaresma", "Frescos", 300, True, 2.50, 6.50, 0.35),
    Category("Bacalao", "Frescos", 90, True, 6.00, 18.00, 0.60),
    # --- Despensa ---
    Category("Pasta", "Despensa", 21, False, 0.75, 2.60, 5.0),
    Category("Salsa de tomate", "Despensa", 24, False, 0.80, 2.40, 3.5),
    Category("Arroz", "Despensa", 30, False, 1.00, 3.20, 3.5),
    Category("Legumbres", "Despensa", 30, False, 0.85, 2.90, 3.0),
    Category("Conservas de pescado", "Despensa", 25, False, 1.30, 5.50, 3.2),
    Category("Aceite de oliva", "Despensa", 40, False, 4.50, 12.50, 3.0),
    Category("Cafe", "Despensa", 25, False, 2.50, 8.50, 4.5),
    Category("Azucar y edulcorante", "Despensa", 45, False, 0.90, 3.10, 2.5),
    Category("Cereales", "Despensa", 22, False, 1.90, 4.80, 3.5),
    Category("Galletas", "Despensa", 15, False, 1.20, 3.80, 4.5),
    Category("Snacks y aperitivos", "Despensa", 12, False, 0.95, 3.40, 5.5),
    Category("Palomitas de microondas", "Despensa", 30, False, 1.10, 3.00, 1.8),
    Category("Harina", "Despensa", 60, False, 0.60, 1.90, 1.6),
    Category("Sal y especias", "Despensa", 90, False, 0.70, 3.50, 1.8),
    Category("Chocolate y huevos de Pascua", "Despensa", 14, False, 1.10, 4.50, 4.5),
    Category("Sopas y caldos", "Despensa", 20, False, 0.85, 3.20, 2.2),
    Category("Turron y mazapan", "Despensa", 330, False, 3.50, 14.00, 0.30),
    # --- Bebidas ---
    Category("Agua", "Bebidas", 10, False, 0.45, 3.50, 6.0),
    Category("Refrescos", "Bebidas", 12, False, 0.90, 4.20, 5.5),
    Category("Zumos", "Bebidas", 12, False, 1.10, 3.60, 3.5),
    Category("Cerveza", "Bebidas", 12, False, 1.20, 6.50, 5.0, "alcohol"),
    Category("Vino", "Bebidas", 18, False, 2.80, 15.00, 3.5, "alcohol"),
    Category("Cava y espumosos", "Bebidas", 120, False, 4.00, 18.00, 0.60, "alcohol"),
    # --- Drogueria ---
    Category("Detergente", "Drogueria", 45, False, 3.50, 12.00, 3.2),
    Category("Suavizante", "Drogueria", 50, False, 2.20, 7.50, 2.6),
    Category("Lavavajillas", "Drogueria", 40, False, 2.00, 8.00, 2.6),
    Category("Lejia y limpiadores", "Drogueria", 50, False, 1.10, 5.50, 2.6),
    Category("Limpiacristales", "Drogueria", 90, False, 1.60, 4.50, 1.3),
    Category("Bolsas de basura", "Drogueria", 45, False, 1.20, 4.00, 2.0),
    Category("Protector solar", "Drogueria", 200, False, 6.00, 18.00, 0.55),
    # --- Higiene ---
    Category("Papel higienico", "Higiene", 30, False, 2.50, 9.50, 4.5),
    Category("Champu", "Higiene", 55, False, 2.20, 8.50, 2.8),
    Category("Acondicionador", "Higiene", 60, False, 2.40, 8.00, 2.0),
    Category("Gel de ducha", "Higiene", 45, False, 1.80, 6.50, 3.0),
    Category("Pasta de dientes", "Higiene", 60, False, 1.50, 5.50, 2.6),
    Category("Desodorante", "Higiene", 70, False, 2.00, 6.50, 2.2),
    Category("Higiene femenina", "Higiene", 35, False, 1.80, 6.00, 2.0),
    # --- Bebe ---
    Category("Panales", "Bebe", 20, False, 6.50, 22.00, 6.0, "baby"),
    Category("Toallitas humedas", "Bebe", 18, False, 1.50, 5.50, 5.0, "baby"),
    Category("Leche infantil", "Bebe", 25, False, 9.00, 26.00, 3.0, "baby"),
    Category("Potitos", "Bebe", 12, False, 0.90, 2.60, 3.5, "baby"),
    # --- Mascotas ---
    Category("Comida para perro", "Mascotas", 25, False, 4.50, 26.00, 4.0, "pet"),
    Category("Comida para gato", "Mascotas", 22, False, 4.00, 22.00, 3.8, "pet"),
    Category("Arena para gato", "Mascotas", 35, False, 3.50, 12.00, 2.2, "pet"),
    # --- Congelados ---
    Category("Verduras congeladas", "Congelados", 25, False, 1.20, 4.20, 2.8),
    Category("Pizza congelada", "Congelados", 18, False, 2.20, 6.50, 3.2),
    Category("Precocinados congelados", "Congelados", 20, False, 2.50, 8.00, 2.8),
    Category("Helados", "Congelados", 14, False, 1.80, 7.50, 2.2),
    Category("Marisco", "Congelados", 60, False, 5.50, 24.00, 1.0),
)

DEPARTMENTS: tuple[str, ...] = (
    "Frescos",
    "Despensa",
    "Bebidas",
    "Drogueria",
    "Higiene",
    "Bebe",
    "Mascotas",
    "Congelados",
)


# --------------------------------------------------------------------------------------
# Afinidad de cesta (tabla de DATA_SPEC.md, seccion "Afinidad de cesta (detalle)")
# --------------------------------------------------------------------------------------
# Si ya hay un producto de la categoria disparadora en la cesta, la probabilidad de anadir
# un producto de la categoria asociada se multiplica por el lift indicado.
AFFINITY_PAIRS: tuple[tuple[str, str, float], ...] = (
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


# --------------------------------------------------------------------------------------
# Estacionalidad (tabla de DATA_SPEC.md, seccion "Estacionalidad (detalle)")
# --------------------------------------------------------------------------------------
# Multiplicador aplicado a la probabilidad base de la categoria durante los meses pico.
# Fuera de esos meses el multiplicador es 1.0.
SEASONALITY: tuple[tuple[str, tuple[int, ...], float], ...] = (
    ("Turron y mazapan", (12,), 8.0),
    ("Marisco", (12,), 3.0),
    ("Cava y espumosos", (12,), 5.0),
    ("Helados", (6, 7, 8), 4.0),
    ("Protector solar", (6, 7, 8), 6.0),
    ("Torrijas y bolleria de Cuaresma", (3, 4), 5.0),
    ("Bacalao", (3, 4), 3.0),
    ("Chocolate y huevos de Pascua", (3, 4), 3.0),
    ("Sopas y caldos", (11, 12, 1, 2), 1.8),
)


# La columna "Lift objetivo" de DATA_SPEC.md es el lift OBSERVADO que se quiere medir a
# nivel de cesta: P(asociada | disparadora) / P(asociada). El multiplicador que aplica el
# generador tiene que ser mayor que ese objetivo, porque solo afecta a los articulos que
# se anaden DESPUES de la categoria disparadora (de media, media cesta) y porque compite
# con el resto de pesos al normalizar. AFFINITY_CALIBRATION cubre esa dilucion:
#
#     multiplicador aplicado = 1 + (lift objetivo - 1) * AFFINITY_CALIBRATION
#
# El valor esta ajustado empiricamente para que el lift medido por
# `data_generation/verify_dataset.py` aterrice en el objetivo de la tabla.
AFFINITY_CALIBRATION = 3.8


def applied_affinity_lift(target_lift: float) -> float:
    """Multiplicador interno que hay que aplicar para observar `target_lift` en la cesta."""
    return 1.0 + (target_lift - 1.0) * AFFINITY_CALIBRATION


# --------------------------------------------------------------------------------------
# Otros parametros de dominio
# --------------------------------------------------------------------------------------
# Uplift de promocion: factor que multiplica la probabilidad de que un producto entre en
# la cesta mientras tiene una promocion activa (DATA_SPEC.md, "Uplift de promocion").
PROMO_UPLIFT = 3.0

# Prevalencia de los gates de hogar.
GATE_PREVALENCE = {"baby": 0.18, "pet": 0.30, "alcohol": 0.70}

LOYALTY_TIERS = ("bronze", "silver", "gold")
LOYALTY_TIER_PROBS = (0.55, 0.32, 0.13)

CHANNELS = ("app", "web", "store")
CHANNEL_PROBS = (0.30, 0.25, 0.45)

DEVICE_TYPES = ("mobile", "desktop", "tablet")
DEVICE_TYPE_PROBS = (0.62, 0.30, 0.08)

PROMO_TYPES = ("2x1", "discount_pct", "coupon")
PROMO_TYPE_PROBS = (0.25, 0.55, 0.20)

# Ciudades espanolas simuladas (peso ~ tamano relativo). No corresponden a ningun
# retailer real: solo dan variedad geografica al dataset.
CITIES: tuple[tuple[str, float], ...] = (
    ("Madrid", 18.0),
    ("Barcelona", 14.0),
    ("Valencia", 8.0),
    ("Sevilla", 7.0),
    ("Zaragoza", 4.5),
    ("Malaga", 4.5),
    ("Murcia", 3.5),
    ("Palma", 3.0),
    ("Las Palmas de Gran Canaria", 3.0),
    ("Bilbao", 3.0),
    ("Alicante", 2.8),
    ("Cordoba", 2.5),
    ("Valladolid", 2.4),
    ("Vigo", 2.4),
    ("Gijon", 2.0),
    ("A Coruna", 2.0),
    ("Granada", 2.0),
    ("Vitoria-Gasteiz", 1.9),
    ("Santa Cruz de Tenerife", 1.9),
    ("Pamplona", 1.8),
    ("Almeria", 1.7),
    ("Donostia-San Sebastian", 1.6),
    ("Santander", 1.5),
    ("Toledo", 1.5),
    ("Salamanca", 1.5),
    ("Badajoz", 1.5),
    ("Logrono", 1.3),
)

COUNTRY = "Espana"

# Marcas de distribuidor (marca blanca). El resto de marcas las genera Faker.
PRIVATE_LABEL_BRANDS: tuple[str, ...] = (
    "Marca Blanca Basico",
    "Marca Blanca Seleccion",
    "Marca Blanca Bio",
)


def category_index() -> dict[str, int]:
    """Devuelve el mapa nombre de categoria -> posicion en CATEGORIES."""
    return {c.name: i for i, c in enumerate(CATEGORIES)}


def validate_catalog() -> None:
    """Comprueba la coherencia interna del catalogo.

    Falla en import si una categoria referenciada por la tabla de afinidad o de
    estacionalidad de DATA_SPEC.md no existe en CATEGORIES, que es el error mas facil
    de cometer al tocar este fichero.
    """
    names = {c.name for c in CATEGORIES}
    if len(names) != len(CATEGORIES):
        raise ValueError("Hay nombres de categoria duplicados en CATEGORIES")

    departments = {c.department for c in CATEGORIES}
    missing_dep = departments - set(DEPARTMENTS)
    if missing_dep:
        raise ValueError(f"Departamentos fuera de DEPARTMENTS: {sorted(missing_dep)}")

    for trigger, associated, _lift in AFFINITY_PAIRS:
        for name in (trigger, associated):
            if name not in names:
                raise ValueError(f"AFFINITY_PAIRS referencia categoria inexistente: {name!r}")

    for name, months, _mult in SEASONALITY:
        if name not in names:
            raise ValueError(f"SEASONALITY referencia categoria inexistente: {name!r}")
        for month in months:
            if not 1 <= month <= 12:
                raise ValueError(f"Mes invalido {month} en SEASONALITY para {name!r}")


validate_catalog()

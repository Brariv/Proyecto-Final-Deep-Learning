"""Definición única de variables, constantes y catálogos usados en entrenamiento y en el servicio."""
from __future__ import annotations

# --- Variables numéricas, en el orden exacto de las columnas de X_num -------------------------
NUM_FEATURES = [
    "log_amt",            # signo * log(1 + |monto|)
    "is_refund",          # monto negativo (devolución)
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "err_bad_pin", "err_bad_cvv", "err_bad_card", "err_bad_exp", "err_bad_zip",
    "err_insufficient", "err_glitch",
    "is_foreign",         # comercio fuera de EE. UU.
    "away_from_home",     # comercio presencial en un estado distinto al del titular
    "log_dt_prev",        # log(1 + minutos desde la transacción anterior de la tarjeta)
    "log_dt_5",           # log(1 + minutos desde la quinta transacción anterior)
    "log_amt_ratio",      # log((|monto| + 1) / (promedio histórico de la tarjeta + 1))
    "log_prior_count",    # log(1 + transacciones previas de la tarjeta)
    "merchant_new",       # la tarjeta nunca había comprado en este comercio
    "mcc_new",            # la tarjeta nunca había comprado en esta categoría
    "state_changed",      # la ubicación cambió respecto a la transacción anterior
    "loc_new",            # la tarjeta nunca había comprado en esta ubicación (estado o país)
    "loc_share_prior",    # fracción de las compras previas de la tarjeta hechas en esta ubicación
    "online_share_prior", # fracción histórica de compras en línea de la tarjeta
    "log_cnt_24h",        # log(1 + transacciones de la tarjeta en las últimas 24 h)
    "log_sum_24h",        # log(1 + monto acumulado en las últimas 24 h)
    "log_credit_limit",
    "amt_over_limit",     # |monto| / límite de crédito (acotado a 5)
    "card_credit", "card_prepaid", "card_has_chip",
    "no_history",         # primera transacción de la tarjeta
]
NUM_INDEX = {name: i for i, name in enumerate(NUM_FEATURES)}
N_NUM = len(NUM_FEATURES)

# Variables binarias: no se estandarizan.
BINARY_FEATURES = {
    "is_refund", "err_bad_pin", "err_bad_cvv", "err_bad_card", "err_bad_exp", "err_bad_zip",
    "err_insufficient", "err_glitch", "is_foreign", "away_from_home", "merchant_new", "mcc_new",
    "state_changed", "loc_new", "card_credit", "card_prepaid", "card_has_chip", "no_history",
}

# --- Variables categóricas (van a embeddings), en el orden de las columnas de X_cat -----------
CAT_FEATURES = ["channel", "mcc", "loc"]
CHANNELS = ["Swipe Transaction", "Chip Transaction", "Online Transaction"]
CHANNEL_LABELS = {"Swipe Transaction": "Banda magnética", "Chip Transaction": "Chip",
                  "Online Transaction": "En línea"}
ONLINE_LOC = "ONLINE"
UNKNOWN_LOC = "UNKNOWN"
LOC_OTHER = 0   # índice reservado: ubicación desconocida o poco frecuente
MCC_OTHER = 0   # índice reservado: categoría no vista en entrenamiento

# Errores del dataset -> variable binaria.
ERROR_FLAGS = {
    "Bad PIN": "err_bad_pin",
    "Bad CVV": "err_bad_cvv",
    "Bad Card Number": "err_bad_card",
    "Bad Expiration": "err_bad_exp",
    "Bad Zipcode": "err_bad_zip",
    "Insufficient Balance": "err_insufficient",
    "Technical Glitch": "err_glitch",
}
ERROR_LABELS = {
    "Bad PIN": "PIN incorrecto",
    "Bad CVV": "CVV incorrecto",
    "Bad Card Number": "Número de tarjeta incorrecto",
    "Bad Expiration": "Fecha de vencimiento incorrecta",
    "Bad Zipcode": "Código postal incorrecto",
    "Insufficient Balance": "Saldo insuficiente",
    "Technical Glitch": "Falla técnica",
}

# --- Constantes de las variables de historial ------------------------------------------------
DT_CAP_MIN = 60 * 24 * 60     # 60 días en minutos: tope de "tiempo desde la anterior"
WINDOW_MIN = 24 * 60          # ventana de velocidad: 24 horas
WINDOW_MAX_TX = 64            # la ventana mira como máximo las 64 transacciones previas
CNT_CLIP = 50
AMT_OVER_LIMIT_CLIP = 5.0

# --- Decisiones del servicio -----------------------------------------------------------------
APPROVE, VERIFY, BLOCK = 0, 1, 2
DECISION_LABELS = {APPROVE: "APROBAR", VERIFY: "VERIFICAR", BLOCK: "BLOQUEAR"}

# --- Descripciones aproximadas de categorías de comercio (MCC) para la interfaz ---------------
MCC_NAMES = {
    4111: "Transporte local", 4112: "Tren de pasajeros", 4121: "Taxis y apps de transporte",
    4131: "Autobuses", 4214: "Mensajería y carga", 4411: "Cruceros", 4511: "Aerolíneas",
    4722: "Agencias de viaje", 4784: "Peajes", 4814: "Telecomunicaciones",
    4829: "Transferencias de dinero", 4899: "Cable y streaming", 4900: "Servicios públicos",
    5045: "Computadoras", 5094: "Joyería", 5192: "Libros y revistas", 5193: "Floristería",
    5211: "Materiales de construcción", 5251: "Ferretería", 5261: "Viveros",
    5300: "Clubes de compra al mayor", 5310: "Tiendas de descuento", 5311: "Tiendas por departamento",
    5411: "Supermercados", 5499: "Tiendas de conveniencia", 5533: "Repuestos de auto",
    5541: "Gasolineras", 5621: "Ropa de mujer", 5651: "Ropa familiar", 5655: "Ropa deportiva",
    5661: "Zapaterías", 5712: "Muebles", 5719: "Artículos para el hogar",
    5722: "Electrodomésticos", 5732: "Electrónica", 5733: "Instrumentos musicales",
    5812: "Restaurantes", 5813: "Bares", 5814: "Comida rápida", 5815: "Contenido digital",
    5816: "Videojuegos digitales", 5912: "Farmacias", 5921: "Licorerías",
    5932: "Antigüedades", 5941: "Artículos deportivos", 5942: "Librerías",
    5947: "Tiendas de regalos", 5970: "Artículos de arte", 5977: "Cosméticos",
    6300: "Seguros", 7011: "Hoteles", 7210: "Lavanderías", 7230: "Salones de belleza",
    7276: "Preparación de impuestos", 7349: "Servicios de limpieza", 7393: "Seguridad privada",
    7531: "Taller de carrocería", 7538: "Taller mecánico", 7542: "Autolavado",
    7549: "Grúas", 7801: "Apuestas en línea", 7802: "Carreras de caballos",
    7832: "Cines", 7922: "Boletos de espectáculos", 7995: "Casinos y apuestas",
    7996: "Parques de diversiones", 8011: "Médicos", 8021: "Dentistas",
    8041: "Quiroprácticos", 8043: "Ópticas", 8049: "Podólogos", 8062: "Hospitales",
    8099: "Servicios médicos", 8111: "Servicios legales", 8931: "Contabilidad",
    9402: "Servicio postal",
}


def mcc_name(mcc: int) -> str:
    """Descripción legible de un MCC (aproximada; solo para la interfaz)."""
    mcc = int(mcc)
    if mcc in MCC_NAMES:
        return MCC_NAMES[mcc]
    if 3000 <= mcc <= 3299:
        return "Aerolínea"
    if 3300 <= mcc <= 3499:
        return "Renta de autos"
    if 3500 <= mcc <= 3999:
        return "Hotel"
    return f"Categoría {mcc}"


def is_foreign_loc(loc: str) -> bool:
    """Los estados de EE. UU. son códigos de 2 letras; todo lo demás (salvo en línea) es extranjero."""
    return len(loc) != 2 and loc not in (ONLINE_LOC, UNKNOWN_LOC, "<otra>")

# FraudScan

Servicio de autorización de transacciones con tarjeta que usa deep learning para decidir, en milisegundos, si **aprobar**, **verificar** (pedir un código al titular) o **bloquear** una compra. Los umbrales de decisión se eligen por **valor de negocio en USD**, no por F1.

Proyecto final de CC3092 Deep Learning y Sistemas Inteligentes (UVG). Integrantes: Pedro Ávila (23089) y Brandon Rivera (23088).

## Qué hay aquí

| Carpeta | Contenido |
|---|---|
| `fraudscan/` | Paquete: preprocesamiento, variables, modelos, entrenamiento, modelo de valor, evaluación y servicio |
| `scripts/` | Pasos del pipeline, numerados en el orden en que se corren |
| `app/` | API (FastAPI) y página del demo |
| `artifacts/` | Modelos entrenados, umbrales y datos del demo (lo único que necesita el servicio) |
| `reports/` | Métricas, tabla de iteraciones y figuras |
| `tests/` | Pruebas del modelo de valor, de la paridad entrenamiento/servicio y de la API |
| `configs/default.yaml` | División temporal, hiperparámetros y **supuestos de costos** |

## Resultados

Perfil `full`, entrenado en CPU. Umbrales elegidos por valor en validación (2016–2017) y resultados medidos en prueba (2018–2019): 3,445,553 transacciones de 2,000 clientes, con 4,578 fraudes que suman 456,705 USD de pérdida posible.

| Iteración | Cambio | AP prueba | Recall | Precisión | Valor en prueba (USD) | IC 95 % | % del máximo | Valor neto / mes | Latencia p50 |
|---|---|---|---|---|---|---|---|---|---|
| 0 · Reglas fijas | Proceso actual: reglas escritas a mano | – | 0.454 | 0.019 | 211,966 | 167,280 a 262,438 | 46.4 % | 8,832 | – |
| 1 · Gradient boosting | Alternativa clásica, mismas variables y pesos que A | 0.175 | 0.410 | 0.124 | 192,009 | 167,311 a 220,391 | 42.0 % | 8,000 | 2.72 ms |
| 2 · Modelo A0 (MLP) | Pérdida estándar, con embedding del país o estado del comercio | 0.042 | 0.143 | 0.038 | 45,123 | 33,058 a 59,052 | 9.9 % | 1,880 | 0.04 ms |
| 3 · Modelo A1 (MLP) | Se quita la identidad de la ubicación | 0.359 | 0.708 | 0.139 | 302,915 | 266,882 a 339,039 | 66.3 % | 12,621 | 0.04 ms |
| 4 · Modelo A (MLP) | A1 con pérdida ponderada por costo | 0.240 | 0.521 | 0.143 | 297,910 | 263,165 a 334,651 | 65.2 % | 12,413 | 0.04 ms |
| 5 · **Modelo B (LSTM)** | LSTM sobre las últimas 16 transacciones | 0.564 | 0.764 | 0.221 | **364,848** | 324,053 a 407,618 | **79.9 %** | **15,202** | 0.27 ms |

El intervalo de confianza remuestrea clientes, no transacciones sueltas. La tabla completa, con las métricas de entrenamiento y validación, está en `reports/iteraciones.md`, y todos los números en `reports/metrics.json`.

**Qué dicen los números**

- **El modelo B aporta 2,789 USD más al mes que el modelo A**, con 0.23 ms más de latencia y menos de un centavo de dólar al mes de cómputo adicional. Su entrenamiento sí cuesta más: 26 minutos contra 81 segundos en la misma CPU.
- **El umbral de máximo F1 no es el de máximo valor.** Con el modelo B, el umbral que maximiza F1 da F1 0.435 y 181,710 USD; los umbrales elegidos por valor dan F1 0.342 y 364,848 USD. Conviene alertar más transacciones porque verificar a un cliente legítimo cuesta poco frente a dejar pasar un fraude.
- **Deriva.** El fraude de 2018–2019 ocurre casi todo en Italia, un país que hasta 2016 solo tenía compras legítimas. El modelo A0, que aprende el país del comercio, cae a 9.9 % del máximo en prueba aunque en validación parecía bueno. Al quitar la identidad de la ubicación y dejar señales genéricas (extranjero, ubicación nueva para la tarjeta), el valor sube a 66.3 %. El mismo efecto se ve dentro de validación: en 2017 el recall de A0 es 0.13 y el de A1 es 0.75.
- **La pérdida ponderada por costo no mejoró en prueba.** A supera a A1 en validación (86.6 % contra 84.8 % del máximo), pero en prueba quedan en 297,910 y 302,915 USD, una diferencia mucho menor que el intervalo de confianza.
- **El gradient boosting queda por debajo de las reglas en prueba** (42.0 % contra 46.4 %), aunque en validación llega a AP 0.795. Con las mismas variables generaliza peor a la campaña nueva que las redes.
- **Sensibilidad.** El supuesto que más mueve el valor es la efectividad de la verificación: si fuera 45 % en lugar de 90 %, el valor del modelo B bajaría a 315,602 USD volviendo a elegir umbrales. Los demás supuestos lo mueven menos de 7 % al cambiar ±50 %.
- **Verificación con datos reales (ULB).** Con la misma función de valor, el MLP obtiene 6,145 USD (69.4 % del máximo) y el gradient boosting 5,429 USD (61.3 %). Solo hay 75 fraudes en prueba, así que los intervalos se traslapan y la diferencia no es concluyente.

**Límites**

- El dataset principal es sintético y los costos son supuestos, así que los dólares son una estimación de orden de magnitud, no una promesa.
- La decisión de quitar la identidad de la ubicación se tomó después de ver la caída en prueba. Se sostiene con la validación de 2017, pero el conjunto de prueba ya no es del todo ciego para esa decisión.
- En 2017 el valor es bajo para todos los modelos (47 % del máximo para B) porque hay pocos fraudes ese año y los umbrales se eligen para 2016–2017 en conjunto.
- La pérdida de entrenamiento se mide sobre la muestra con 100 legítimas por fraude y la de validación sobre la proporción real, así que no son comparables entre sí en las curvas.
- El flujo del demo es una selección de 16 tarjetas con muchos casos de fraude para que se vean las ráfagas; los números representativos son los de la tabla.

## Datos

Los datos no están en el repositorio (pesan 2.4 GB). Descárgalos y descomprímelos así:

| Dataset | Uso | Dónde va |
|---|---|---|
| **IBM Credit Card Transactions** (Kaggle, `ealtman2019/credit-card-transactions`). Sintético: 24.4 millones de transacciones de 2,000 clientes y 6,139 tarjetas entre 1991 y 2020, con 29,757 fraudes (0.12 %) | Principal | `data/ibm/` (`credit_card_transactions-ibm_v2.csv`, `sd254_cards.csv`, `sd254_users.csv`) |
| **ULB Credit Card Fraud Detection** (Kaggle, `mlg-ulb/creditcardfraud`). Real: 284,807 transacciones de tarjetas europeas de dos días de 2013, con 492 fraudes | Verificación secundaria | `data/ulb/creditcard.csv` |

El dataset de IBM es **sintético** (lo genera una simulación), y eso limita qué tan representativo es de un emisor real. Se eligió como principal porque trae variables legibles y el historial completo de cada tarjeta, que es lo que permite el modelo secuencial y un demo con transacciones nuevas. ULB aporta el contraste con datos reales.

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

En Windows con GPU NVIDIA, instala antes PyTorch con CUDA con el comando que da <https://pytorch.org/get-started/locally/>. Los scripts detectan la GPU solos (`cuda`, luego `mps` en Mac y por último `cpu`).

## Reproducir los resultados

```bash
python scripts/run_all.py --profile full --with-ulb     # completo, pensado para GPU
python scripts/run_all.py                               # rápido, en CPU
```

O paso por paso (todos aceptan `--profile quick|full`):

| Paso | Qué hace |
|---|---|
| `01_prepare_data.py` | Lee el CSV por bloques y calcula las variables, usando solo información anterior a cada transacción. Necesita unos 6 GB de RAM y tarda unos 5 minutos |
| `02_train_baselines.py` | Iteración 0 (reglas fijas) e iteración 1 (gradient boosting) |
| `03_train_mlp.py` | Modelo A: MLP con pérdida estándar y con pérdida ponderada por costo |
| `04_train_lstm.py` | Modelo B: LSTM sobre las últimas 16 transacciones de la tarjeta |
| `05_evaluate.py` | Elige umbrales por valor en validación y reporta en prueba: métricas, valor, IC 95 %, sensibilidad, latencia y figuras |
| `06_export_demo.py` | Exporta las tarjetas y transacciones del demo |
| `ulb_benchmark.py` | Repite la comparación con el dataset real de ULB |

El perfil `quick` entrena con 15 transacciones legítimas por cada fraude y 4 épocas, y termina en unos 6 minutos en CPU. El perfil `full` usa 100 por fraude y 12 épocas; en una CPU de 2 núcleos tardó unos 40 minutos, de los cuales 26 son la LSTM. Los artefactos y reportes del repositorio vienen del perfil `full`.

## Demo

```bash
uvicorn app.main:app --reload
```

Abre <http://127.0.0.1:8000>. El demo no necesita el dataset: usa lo que hay en `artifacts/`.

- **Autorizar una transacción:** escribe cualquier compra para una de las tarjetas y el servicio responde con la decisión, el score, las razones y la latencia.
- **Flujo del periodo de prueba:** reproduce transacciones de 2019 que el modelo nunca vio y acumula, en dólares, el valor del modelo A, del modelo B y de no tener modelo.
- **API:** `POST /api/score` autoriza una transacción y `GET /api/stats` da las métricas operativas. La documentación interactiva está en `/docs`.

Con Docker:

```bash
docker build -t fraudscan .
docker run -p 8080:8080 fraudscan
```

## Cómo funciona

**División temporal.** Entrenamiento hasta 2015, validación 2016–2017 y prueba 2018–2019. La validación se usa para early stopping y para elegir umbrales; la prueba no se usa para decidir nada.

**Variables.** 33 numéricas y 3 categóricas por transacción: monto, hora, canal, categoría del comercio, ubicación, errores, y el historial de la tarjeta (tiempo desde la compra anterior, monto frente a lo habitual, compras en 24 horas, si el comercio o la ubicación son nuevos para la tarjeta, etc.). No se usan datos demográficos del titular (edad, género, ingreso). El mismo cálculo existe por lotes (`prepare.py`) y transacción por transacción (`online.py`), y una prueba verifica que coinciden.

**Modelos.**

- **Modelo A (MLP):** embeddings de canal, categoría y ubicación, más las variables numéricas, en una red densa de dos capas.
- **Modelo B (LSTM):** una LSTM lee las últimas 16 transacciones de la tarjeta y su estado final se combina con la transacción actual. En estos datos el fraude llega en ráfagas por tarjeta, y eso es lo que la secuencia captura.
- **Referencias:** reglas fijas (el proceso actual) y gradient boosting (la alternativa clásica).

Todos se entrenan desde cero. No hay LLM ni modelos preentrenados en ninguna parte.

**Pérdida alineada con el negocio.** Entropía cruzada binaria en la que cada fraude pesa en proporción al dinero que se pierde si pasa (monto más contracargo).

**Modelo de valor.** El valor se mide contra no tener modelo (aprobar todo):

```
valor =   Σ fraudes bloqueados   (monto + c_contracargo)
        + Σ fraudes verificados  p_otp · (monto + c_contracargo)
        − verificaciones · c_verificación
        − legítimas verificadas · c_fricción_verificar
        − legítimas bloqueadas  · c_fricción_bloquear
        − costo de cómputo
```

Los dos umbrales (verificar y bloquear) son los que maximizan ese valor en validación; la búsqueda es exacta. Los costos están en `configs/default.yaml` y **son supuestos**; `05_evaluate.py` mide qué pasa si están equivocados en ±20 % y ±50 %.

| Supuesto | Valor (USD) |
|---|---|
| Costo operativo de un contracargo, además del monto | 15.00 |
| Costo de una verificación (SMS y fracción de analista) | 0.25 |
| Fricción por verificar a un cliente legítimo | 1.00 |
| Fricción por bloquear una compra legítima | 10.00 |
| Fraudes que la verificación sí detiene | 90 % |
| Precio del cómputo | 0.000024 por vCPU-segundo |

## Pruebas

```bash
pytest -q
```

- `test_value.py`: los umbrales óptimos coinciden con una búsqueda exhaustiva.
- `test_online_parity.py`: el servicio calcula las mismas variables que el entrenamiento.
- `test_api.py`: el servicio responde de punta a punta y valida entradas incorrectas.

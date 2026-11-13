| Iteración | Cambio | AP train | AP val | AP test | Recall | Precisión | Valor en prueba (USD) | IC 95 % | % del máximo | Valor neto / mes | Entrenamiento | Latencia p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 · Reglas fijas | Proceso actual: reglas escritas a mano, sin modelo | – | – | – | 0.454 | 0.019 | 211,966 | 167,280 a 262,438 | 46.4 % | 8,832 | 0 s (cpu) | 0.00 ms |
| 1 · Gradient boosting | Alternativa clásica con las mismas variables y pesos que el modelo A | 0.971 | 0.795 | 0.175 | 0.410 | 0.124 | 192,009 | 167,311 a 220,391 | 42.0 % | 8,000 | 102 s (cpu) | 2.72 ms |
| 2 · Modelo A0 (MLP) | Red densa, pérdida estándar (BCE), con embedding del país o estado del comercio | 0.879 | 0.734 | 0.042 | 0.143 | 0.038 | 45,123 | 33,058 a 59,052 | 9.9 % | 1,880 | 92 s (cpu) | 0.04 ms |
| 3 · Modelo A1 (MLP) | Se quita la identidad de la ubicación; quedan las señales genéricas | 0.865 | 0.760 | 0.359 | 0.708 | 0.139 | 302,915 | 266,882 a 339,039 | 66.3 % | 12,621 | 80 s (cpu) | 0.04 ms |
| 4 · Modelo A (MLP) | Igual que A1, con pérdida ponderada por el costo del error | 0.825 | 0.714 | 0.240 | 0.521 | 0.143 | 297,910 | 263,165 a 334,651 | 65.2 % | 12,413 | 81 s (cpu) | 0.04 ms |
| 5 · Modelo B (LSTM) | LSTM sobre las últimas 16 transacciones de la tarjeta, pérdida por costo | 0.930 | 0.819 | 0.564 | 0.764 | 0.221 | 364,848 | 324,053 a 407,618 | 79.9 % | 15,202 | 1559 s (cpu) | 0.27 ms |

# Imagen del servicio de autorización (solo inferencia en CPU).
FROM python:3.12-slim
WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY fraudscan ./fraudscan
COPY app ./app
COPY artifacts ./artifacts
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

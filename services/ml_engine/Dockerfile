FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PROCESSED_TOPIC=processed_features
ENV ANOMALY_TOPIC=anomaly_scores

COPY services/ml_engine/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY services /app/services
COPY shared /app/shared

CMD ["sh", "-c", "python -m services.ml_engine.app.stream.feature_consumer --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} --in-topic user_activity --out-topic ${PROCESSED_TOPIC} & python -m services.ml_engine.app.stream.inference_consumer --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} --in-topic ${PROCESSED_TOPIC} --out-topic ${ANOMALY_TOPIC} --artifact-dir /models; wait"]

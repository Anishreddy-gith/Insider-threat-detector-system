FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/ /app/app/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV REDIS_URL=redis://redis:6379/0

ENV PROCESSED_TOPIC=processed_features
ENV ANOMALY_TOPIC=anomaly_scores

CMD ["sh", "-c", "python -m app.stream.feature_consumer --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} --in-topic user_activity --out-topic ${PROCESSED_TOPIC} & python -m app.stream.inference_consumer --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} --in-topic ${PROCESSED_TOPIC} --out-topic ${ANOMALY_TOPIC} --artifact-dir /models; wait"]

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV INPUT_PATH=/data/events.jsonl
ENV USER_ACTIVITY_TOPIC=user_activity

COPY services/ingestion/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY services /app/services
COPY shared /app/shared

CMD ["sh", "-c", "mkdir -p $(dirname ${INPUT_PATH}) && if [ ! -f ${INPUT_PATH} ]; then printf '%s\n' '{\"user_id\":\"user1\",\"timestamp\":\"2023-01-01T10:00:00\",\"activity_type\":\"logon\",\"metadata\":{}}' > ${INPUT_PATH}; fi; python -m services.ingestion.app.producer --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} --input ${INPUT_PATH} --topic ${USER_ACTIVITY_TOPIC}"]

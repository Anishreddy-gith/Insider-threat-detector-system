FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/ /app/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Set INPUT_PATH to a mounted CERT JSON/JSONL/CSV stream file.
ENV INPUT_PATH=/data/events.jsonl
ENV USER_ACTIVITY_TOPIC=user_activity

CMD ["sh", "-c", "python producer.py --bootstrap-servers ${KAFKA_BOOTSTRAP_SERVERS} --input ${INPUT_PATH} --topic ${USER_ACTIVITY_TOPIC}"]

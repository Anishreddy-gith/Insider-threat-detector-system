FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY services/api_gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt kafka-python PyJWT

COPY services /app/services
COPY shared /app/shared

EXPOSE 8000

CMD ["sh", "-c", "uvicorn services.api_gateway.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

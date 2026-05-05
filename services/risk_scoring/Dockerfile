FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY services/risk_scoring/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt kafka-python

COPY services /app/services
COPY shared /app/shared

CMD ["python", "-m", "services.risk_scoring.app.risk_consumer"]

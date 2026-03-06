# ─────────────────────────────────────────────────────────────────────────
# Ingestion Service – Collects and normalises raw telemetry
# ─────────────────────────────────────────────────────────────────────────
"""
ingestion-service: Real-time data collection from syslog, EDR, DLP, and
network-flow sources.  Normalises heterogeneous events into a canonical
EventEnvelope schema and publishes them to Kafka for downstream consumers.
"""

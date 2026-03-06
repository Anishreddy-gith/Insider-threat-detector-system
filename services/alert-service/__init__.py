"""
alert-service: Alert lifecycle management.

Consumes risk-score events from ``itds.risk-scores``, creates alerts
based on configurable thresholds, manages acknowledge / escalate /
resolve workflows, and persists alert state in PostgreSQL.
"""

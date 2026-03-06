"""
risk-scoring-service: Contextual risk aggregation & scoring.

Consumes anomaly events from ``itds.anomalies``, applies exponential
decay to historical scores, incorporates HR context, and publishes
updated risk assessments to ``itds.risk-scores``.
"""

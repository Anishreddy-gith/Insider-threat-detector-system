"""
ml-engine: Behavioural analytics and anomaly detection.

Consumes normalised events from Kafka, extracts feature vectors,
runs inference through LSTM Autoencoder + GraphSAGE GNN ensemble,
generates SHAP/LIME explanations, and publishes anomaly scores
back to Kafka for the risk-scoring-service.
"""

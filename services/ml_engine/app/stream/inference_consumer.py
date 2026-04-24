from __future__ import annotations

import argparse
import json
from typing import Any

from kafka import KafkaConsumer, KafkaProducer

from services.ml_engine.app.inference.infer import infer_one

TOPICS = {
    "user_activity": "user_activity",
    "processed_features": "processed_features",
    "anomaly_scores": "anomaly_scores",
    "risk_events": "risk_events",
}


def _json_deserializer(value: bytes) -> dict[str, Any]:
    obj = json.loads(value.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Kafka message must be a JSON object")
    return obj


def _json_serializer(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume processed_features and publish anomaly_scores")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--in-topic", default=TOPICS["processed_features"])
    parser.add_argument("--out-topic", default=TOPICS["anomaly_scores"])
    parser.add_argument("--group-id", default="inference-consumer")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--graph-cert-dir", default=None)
    parser.add_argument("--graph-behavior-path", default=None)
    parser.add_argument("--w-autoencoder", type=float, default=0.4)
    parser.add_argument("--w-isolation", type=float, default=0.4)
    parser.add_argument("--w-gnn", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    consumer = KafkaConsumer(
        args.in_topic,
        bootstrap_servers=args.bootstrap_servers,
        value_deserializer=_json_deserializer,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id=args.group_id,
    )
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=_json_serializer,
        linger_ms=10,
        acks="all",
    )

    try:
        for msg in consumer:
            payload = msg.value
            user_id = str(payload.get("user_id", "")).strip()
            features = payload.get("features", {})

            if not user_id or not isinstance(features, dict):
                continue

            inference_input = dict(features)
            inference_input["user_id"] = user_id

            result = infer_one(
                feature_input=inference_input,
                artifact_dir=args.artifact_dir,
                graph_cert_dir=args.graph_cert_dir,
                graph_behavior_path=args.graph_behavior_path,
                graph_target_user=user_id,
                w_autoencoder=args.w_autoencoder,
                w_isolation=args.w_isolation,
                w_gnn=args.w_gnn,
            )

            out_msg = {
                "user_id": user_id,
                "autoencoder_score": float(result.get("autoencoder_score", 0.0)),
                "isolation_score": float(result.get("isolation_score", 0.0)),
                "gnn_score": float(result.get("gnn_score", 0.0)),
                "final_score": float(result.get("final_score", 0.0)),
            }
            producer.send(args.out_topic, value=out_msg)
    finally:
        producer.flush()
        producer.close()
        consumer.close()


if __name__ == "__main__":
    main()


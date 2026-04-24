from __future__ import annotations

import argparse
import json
import os
from typing import Any

from kafka import KafkaConsumer, KafkaProducer
import redis

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


def _risk_level(score: float, high_threshold: float, medium_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume anomaly_scores and publish risk_events")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--in-topic", default=TOPICS["anomaly_scores"])
    parser.add_argument("--out-topic", default=TOPICS["risk_events"])
    parser.add_argument("--group-id", default="risk-consumer")
    parser.add_argument("--medium-threshold", type=float, default=0.5)
    parser.add_argument("--high-threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    redis_client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )
    redis_key_prefix = "risk:latest:"

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
            final_score = payload.get("final_score")

            if not user_id:
                continue
            try:
                score = float(final_score)
            except (TypeError, ValueError):
                continue

            out_msg = {
                "user_id": user_id,
                "risk_level": _risk_level(
                    score=score,
                    high_threshold=args.high_threshold,
                    medium_threshold=args.medium_threshold,
                ),
                "final_score": score,
            }
            producer.send(args.out_topic, value=out_msg)
            redis_client.set(
                f"{redis_key_prefix}{user_id}",
                json.dumps(out_msg, ensure_ascii=True, separators=(",", ":")),
            )
    finally:
        producer.flush()
        producer.close()
        consumer.close()


if __name__ == "__main__":
    main()

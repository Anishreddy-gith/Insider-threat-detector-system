from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

from services.ml_engine.app.cert_pipeline.feature_engineering import create_feature_dataframe

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


def _to_iso(ts: str) -> str:
    dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(dt):
        dt = datetime.now(timezone.utc)
    return dt.isoformat()


def _map_event_type(activity_type: str) -> str:
    t = str(activity_type).strip().lower()
    if t in {"logon", "login", "logoff", "logout"}:
        return "logon"
    if t in {"file", "file_access", "file_open", "file_copy", "file_write"}:
        return "file"
    if t in {"email", "email_send", "mail"}:
        return "email"
    if t in {"device", "usb", "connect", "disconnect"}:
        return "device"
    return t if t else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume user_activity and publish processed_features")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--in-topic", default=TOPICS["user_activity"])
    parser.add_argument("--out-topic", default=TOPICS["processed_features"])
    parser.add_argument("--group-id", default="feature-consumer")
    parser.add_argument("--history-size", type=int, default=500)
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

    history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    try:
        for msg in consumer:
            event = msg.value

            user_id = str(event.get("user_id", "")).strip()
            timestamp = _to_iso(str(event.get("timestamp", "")))
            activity_type = str(event.get("activity_type", "")).strip().lower()
            metadata = event.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"raw": metadata}

            if not user_id or not activity_type:
                continue

            row = {
                "user_id": user_id,
                "timestamp": timestamp,
                "event_type": _map_event_type(activity_type),
                "activity": activity_type,
            }
            row.update(metadata)

            bucket = history[user_id]
            bucket.append(row)
            if len(bucket) > args.history_size:
                del bucket[:-args.history_size]

            timeline = pd.DataFrame(bucket)
            features_df = create_feature_dataframe(timeline)
            if features_df.empty:
                continue

            latest = features_df.iloc[-1].to_dict()
            features_payload = {
                "login_frequency": float(latest["login_frequency"]),
                "after_hours_activity": float(latest["after_hours_activity"]),
                "file_access_deviation": float(latest["file_access_deviation"]),
                "email_anomalies": float(latest["email_anomalies"]),
                "session_duration": float(latest["session_duration"]),
            }
            out_msg = {
                "user_id": user_id,
                "features": features_payload,
            }

            producer.send(args.out_topic, value=out_msg)
    finally:
        producer.flush()
        producer.close()
        consumer.close()


if __name__ == "__main__":
    main()


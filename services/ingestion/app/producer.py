from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

from kafka import KafkaProducer

TOPICS = {
    "user_activity": "user_activity",
    "processed_features": "processed_features",
    "anomaly_scores": "anomaly_scores",
    "risk_events": "risk_events",
}


def _json_serializer(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _safe_json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": text}


def _normalize_event(raw: dict[str, Any]) -> dict[str, Any]:
    user_id = str(raw.get("user_id") or raw.get("user") or raw.get("userid") or "").strip()
    timestamp = str(raw.get("timestamp") or raw.get("date") or raw.get("time") or "").strip()
    activity_type = str(raw.get("activity_type") or raw.get("event_type") or raw.get("activity") or "").strip().lower()

    metadata = raw.get("metadata")
    if metadata is None:
        metadata = {
            k: v
            for k, v in raw.items()
            if k not in {"user_id", "user", "userid", "timestamp", "date", "time", "activity_type", "event_type", "activity", "metadata"}
        }
    metadata = _safe_json_obj(metadata)

    event = {
        "user_id": user_id,
        "timestamp": timestamp,
        "activity_type": activity_type,
        "metadata": metadata,
    }

    if not event["user_id"]:
        raise ValueError("Missing required field: user_id")
    if not event["timestamp"]:
        raise ValueError("Missing required field: timestamp")
    if not event["activity_type"]:
        raise ValueError("Missing required field: activity_type")

    return event


def _iter_json_stream(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return

    if text.startswith("["):
        payload = json.loads(text)
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            yield parsed


def _iter_csv_stream(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish CERT-style events to Kafka user_activity topic")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--input", required=True, help="Input .json/.jsonl/.csv file path")
    parser.add_argument("--topic", default=TOPICS["user_activity"])
    parser.add_argument("--sleep-ms", type=int, default=0, help="Delay between messages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=_json_serializer,
        linger_ms=10,
        acks="all",
    )

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        stream = _iter_csv_stream(input_path)
    else:
        stream = _iter_json_stream(input_path)

    sleep_s = max(args.sleep_ms, 0) / 1000.0

    try:
        for raw in stream:
            event = _normalize_event(raw)
            producer.send(args.topic, value=event)
            producer.flush()
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()

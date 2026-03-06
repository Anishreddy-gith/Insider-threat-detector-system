#!/usr/bin/env python3
"""
simulate_events.py – Generate 10 000 synthetic user events with embedded anomalies.

Usage
─────
    # Against a running ingestion-service
    python scripts/simulate_events.py --url http://localhost:8001

    # Dry-run (print to stdout as JSONL)
    python scripts/simulate_events.py --dry-run

What it generates
─────────────────
• 50 simulated users with realistic baseline behaviour
• 5 event types: login, logout, file_access, network_request, app_usage
• 10 000 events spread across a 30-day window
• ~5 % of events are *anomalous*, injected for 5 "insider" users:
    - 2 AM logins from unusual geolocations
    - 10× normal file-access volume in a single session
    - Downloading restricted/confidential files after hours
    - Massive data exfiltration via network requests
    - Abnormal application usage patterns (never-used apps)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

# ── Configuration ───────────────────────────────────────────────────

NUM_EVENTS       = 10_000
NUM_USERS        = 50
NUM_INSIDERS     = 5       # users that produce anomalous behaviour
WINDOW_DAYS      = 30      # events spread across this many days
ANOMALY_RATIO    = 0.05    # ~5 % anomalous events

DEPARTMENTS = [
    "Engineering", "Finance", "HR", "Legal",
    "Marketing", "Sales", "IT-Ops", "Executive",
]
APPS = [
    "slack", "outlook", "chrome", "vscode", "excel",
    "powerpoint", "teams", "zoom", "jira", "confluence",
]
NORMAL_GEO = ["US-East", "US-West", "EU-West", "EU-Central"]
ANOMALOUS_GEO = ["RU-Moscow", "CN-Shanghai", "KP-Pyongyang", "IR-Tehran"]

FILE_PATHS_NORMAL = [
    "/docs/quarterly_report.docx",
    "/shared/meeting_notes.pdf",
    "/projects/src/main.py",
    "/home/user/notes.txt",
    "/designs/mockup_v3.fig",
]
FILE_PATHS_SENSITIVE = [
    "/restricted/customer_pii_dump.csv",
    "/confidential/salary_data_2026.xlsx",
    "/restricted/merger_details.pdf",
    "/confidential/board_minutes_q1.docx",
    "/restricted/source_code_audit.tar.gz",
]


# ── User profiles ──────────────────────────────────────────────────

def _generate_users() -> list[dict]:
    users = []
    for i in range(NUM_USERS):
        users.append({
            "user_id": f"user-{i:04d}",
            "department": random.choice(DEPARTMENTS),
            "is_insider": i < NUM_INSIDERS,
            # Normal behavioural ranges (per day)
            "normal_logins_per_day": random.uniform(1, 4),
            "normal_files_per_day": random.uniform(2, 15),
            "normal_file_size": random.randint(1024, 500_000),   # bytes
            "normal_network_mb": random.uniform(5, 100),
            "normal_app_minutes": random.uniform(30, 480),
        })
    return users


# ── Event generators ───────────────────────────────────────────────

def _random_ts(base: datetime, day_offset: int, off_hours: bool = False) -> str:
    day = base + timedelta(days=day_offset)
    if off_hours:
        hour = random.choice([0, 1, 2, 3, 4, 23])
    else:
        hour = random.randint(8, 18)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return day.replace(hour=hour, minute=minute, second=second).isoformat()


def _login_event(user: dict, ts: str, anomalous: bool) -> dict:
    event: dict[str, Any] = {
        "event_type": "login",
        "user_id": user["user_id"],
        "timestamp": ts,
        "source_ip": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "auth_method": "password",
        "success": True,
    }
    if anomalous:
        event["geo_location"] = random.choice(ANOMALOUS_GEO)
        event["auth_method"] = random.choice(["password", "sso"])
        event["source_ip"] = f"185.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    else:
        event["geo_location"] = random.choice(NORMAL_GEO)
    return event


def _logout_event(user: dict, ts: str, anomalous: bool) -> dict:
    duration = random.uniform(300, 28800)  # 5 min – 8 hr
    if anomalous:
        duration = random.uniform(36000, 72000)  # 10–20 hours (suspiciously long)
    return {
        "event_type": "logout",
        "user_id": user["user_id"],
        "timestamp": ts,
        "source_ip": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "session_duration_seconds": round(duration, 1),
    }


def _file_access_event(user: dict, ts: str, anomalous: bool) -> dict:
    if anomalous:
        path = random.choice(FILE_PATHS_SENSITIVE)
        size = user["normal_file_size"] * random.randint(8, 15)  # 8–15× normal
        label = random.choice(["confidential", "restricted"])
        action = random.choice(["download", "copy"])
    else:
        path = random.choice(FILE_PATHS_NORMAL)
        size = int(user["normal_file_size"] * random.uniform(0.5, 1.5))
        label = "unclassified"
        action = random.choice(["read", "write"])
    return {
        "event_type": "file_access",
        "user_id": user["user_id"],
        "timestamp": ts,
        "file_path": path,
        "file_size_bytes": size,
        "action": action,
        "sensitivity_label": label,
        "source_device": f"workstation-{user['user_id']}",
    }


def _network_event(user: dict, ts: str, anomalous: bool) -> dict:
    if anomalous:
        # Massive exfiltration burst
        sent = int(user["normal_network_mb"] * 1_000_000 * random.uniform(8, 20))
        received = random.randint(1000, 50000)
        domain = random.choice([
            "mega.nz", "dropmefiles.com", "paste.ee", "transfer.sh",
        ])
        port = 443
    else:
        sent = int(user["normal_network_mb"] * 1_000_000 * random.uniform(0.01, 0.1) / WINDOW_DAYS)
        received = int(sent * random.uniform(0.5, 3))
        domain = random.choice([
            "github.com", "google.com", "office365.com", "slack.com",
        ])
        port = random.choice([80, 443, 8080])
    return {
        "event_type": "network_request",
        "user_id": user["user_id"],
        "timestamp": ts,
        "source_ip": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "destination_ip": f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "destination_port": port,
        "protocol": "TCP",
        "bytes_sent": sent,
        "bytes_received": received,
        "domain": domain,
        "is_encrypted": True,
    }


def _app_usage_event(user: dict, ts: str, anomalous: bool) -> dict:
    if anomalous:
        app = random.choice(["tor-browser", "wireshark", "mimikatz", "nmap"])
        duration = random.uniform(600, 7200)  # unusual apps for long durations
        category = "security-tool"
    else:
        app = random.choice(APPS)
        duration = random.uniform(60, user["normal_app_minutes"] * 60 / 5)
        category = "productivity"
    return {
        "event_type": "app_usage",
        "user_id": user["user_id"],
        "timestamp": ts,
        "application_name": app,
        "window_title": f"{app} - session",
        "duration_seconds": round(duration, 1),
        "foreground": random.random() > 0.2,
        "category": category,
    }


EVENT_GENERATORS = {
    "login": _login_event,
    "logout": _logout_event,
    "file_access": _file_access_event,
    "network_request": _network_event,
    "app_usage": _app_usage_event,
}

EVENT_TYPES = list(EVENT_GENERATORS.keys())
# Weight file_access and network heavier (more common in real telemetry)
EVENT_WEIGHTS = [0.15, 0.10, 0.30, 0.25, 0.20]


# ── Main generation loop ──────────────────────────────────────────

def generate_events() -> list[dict]:
    """Produce NUM_EVENTS events with ~ANOMALY_RATIO anomalous."""
    users = _generate_users()
    insiders = [u for u in users if u["is_insider"]]
    normals  = [u for u in users if not u["is_insider"]]

    base_date = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    events: list[dict] = []

    anomaly_count = int(NUM_EVENTS * ANOMALY_RATIO)
    normal_count  = NUM_EVENTS - anomaly_count

    # Normal events
    for _ in range(normal_count):
        user = random.choice(users)
        etype = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]
        day = random.randint(0, WINDOW_DAYS - 1)
        ts = _random_ts(base_date, day, off_hours=False)
        events.append(EVENT_GENERATORS[etype](user, ts, anomalous=False))

    # Anomalous events (clustered on insider users)
    for _ in range(anomaly_count):
        user = random.choice(insiders)
        etype = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]
        day = random.randint(0, WINDOW_DAYS - 1)
        # Many anomalies happen off-hours
        off_hours = random.random() < 0.7
        ts = _random_ts(base_date, day, off_hours=off_hours)
        events.append(EVENT_GENERATORS[etype](user, ts, anomalous=True))

    # Shuffle to interleave normal + anomalous chronologically
    events.sort(key=lambda e: e["timestamp"])
    return events


# ── Sender ─────────────────────────────────────────────────────────

async def send_events(events: list[dict], base_url: str, concurrency: int = 20):
    """POST events to /ingest/event with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    sent = 0
    errors = 0

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        async def _send(evt: dict):
            nonlocal sent, errors
            async with sem:
                try:
                    resp = await client.post("/ingest/event", json=evt)
                    if resp.status_code in (200, 202):
                        sent += 1
                    else:
                        errors += 1
                        if errors <= 5:
                            print(f"  ⚠  {resp.status_code}: {resp.text[:120]}")
                except httpx.HTTPError as exc:
                    errors += 1
                    if errors <= 5:
                        print(f"  ⚠  HTTP error: {exc}")

        tasks = [asyncio.create_task(_send(e)) for e in events]
        total = len(tasks)
        # Simple progress reporting
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % 1000 == 0 or done == total:
                print(f"  Progress: {done}/{total}  (sent={sent}, errors={errors})")

    print(f"\nDone.  sent={sent}  errors={errors}  total={total}")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate & send 10 000 synthetic insider-threat events"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8001",
        help="Ingestion service base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print events as JSONL to stdout instead of sending",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=NUM_EVENTS,
        help=f"Number of events to generate (default: {NUM_EVENTS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    global NUM_EVENTS
    NUM_EVENTS = args.count

    print(f"Generating {NUM_EVENTS} events ({NUM_USERS} users, "
          f"{NUM_INSIDERS} insiders, ~{ANOMALY_RATIO*100:.0f}% anomalous) …")
    events = generate_events()

    insider_events = sum(
        1 for e in events
        if e["user_id"].startswith("user-000") and int(e["user_id"][-1]) < NUM_INSIDERS
    )
    print(f"  Total: {len(events)}  |  From insiders: {insider_events}")

    if args.dry_run:
        for e in events:
            print(json.dumps(e, default=str))
    else:
        print(f"Sending to {args.url}/ingest/event …\n")
        asyncio.run(send_events(events, args.url))


if __name__ == "__main__":
    main()

# Real-Time Insider Threat Detection System using GNNs and Streaming ML

## Architecture

```text
Client/API
  -> API Gateway :8000
  -> Kafka user_activity
  -> ML Engine feature + inference consumers
  -> Kafka anomaly_scores
  -> Risk Scoring consumer
  -> Redis latest risk cache
  -> API Gateway /risk/{user_id}
```

## Features

- Real-time event ingestion
- Kafka streaming pipeline
- ML-based anomaly scoring
- GNN-ready ML engine structure
- Risk scoring with Redis-backed latest-risk lookup
- JWT-protected ingest and risk APIs
- Docker Compose deployment
- AWS EC2 deployment path

## Tech Stack

- Python 3.10
- FastAPI
- Kafka
- Redis
- scikit-learn
- PyTorch
- torch-geometric
- Docker
- Docker Compose
- AWS EC2

## System Design

```text
api_gateway
  POST /ingest
  GET  /risk/{user_id}
  GET  /health
  GET  /healthz
  GET  /metrics

ingestion
  Produces user_activity events to Kafka.

ml_engine
  Consumes user_activity.
  Publishes processed_features and anomaly_scores.

risk_scoring
  Consumes anomaly_scores.
  Publishes risk_events.
  Writes latest risk to Redis.

redis
  Stores risk:latest:{user_id}.

kafka
  Streams user_activity, processed_features, anomaly_scores, risk_events.
```

## Setup Instructions

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
set PYTHONPATH=.
pytest -v
```

## Deployment (Docker + AWS)

### Local Docker

```bash
docker-compose up --build
```

API:

```text
http://localhost:8000
```

Health:

```bash
curl http://localhost:8000/healthz
```

### AWS EC2

Launch EC2:

```bash
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t2.medium \
  --key-name <key_name> \
  --security-group-ids <security_group_id> \
  --subnet-id <subnet_id> \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=itds-prod}]'
```

SSH:

```bash
ssh -i <key.pem> ubuntu@<EC2-IP>
```

Install Docker:

```bash
sudo apt update
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER
newgrp docker
```

Clone and run:

```bash
git clone <your_repo>
cd <repo>
docker-compose up --build -d
```

Open inbound TCP port `8000` in the EC2 security group.

API:

```text
http://<EC2-IP>:8000
```

## API Usage

Create a token with payload signed by `JWT_SECRET=supersecretkey`, then call:

```bash
curl -X POST http://<EC2-IP>:8000/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "user_id": "user1",
    "timestamp": "2023-01-01T10:00:00",
    "activity_type": "logon",
    "metadata": {}
  }'
```

```bash
curl http://<EC2-IP>:8000/risk/user1 \
  -H "Authorization: Bearer <token>"
```

## Example Outputs

Ingest:

```json
{
  "status": "accepted",
  "topic": "user_activity"
}
```

Risk:

```json
{
  "user_id": "user1",
  "risk_level": "low",
  "final_score": 0.18
}
```

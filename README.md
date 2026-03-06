# ITDS – AI-Powered Insider Threat Detection System

A production-grade, full-stack behavioral analytics platform that detects insider threats using machine learning, graph neural networks, and explainable AI — while preserving privacy through differential privacy and federated learning.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          React Frontend                             │
│         (Dashboard · Alerts · Entity Profiles · XAI Views)          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTPS
┌──────────────────────────────▼──────────────────────────────────────┐
│                         API Gateway                                 │
│   (FastAPI · JWT Auth · Rate Limiting · RBAC · Audit Logging)       │
└──┬──────────────┬───────────────┬───────────────┬──────────────────┘
   │              │               │               │
   ▼              ▼               ▼               ▼
┌──────┐   ┌───────────┐  ┌───────────────┐  ┌──────────┐
│ Auth │   │   Data     │  │  Behavioral   │  │   Risk   │
│Service│  │ Ingestion  │  │  Analytics    │  │ Scoring  │
│      │   │            │  │   (ML/AI)     │  │          │
└──────┘   └─────┬──────┘  └──────┬────────┘  └────┬─────┘
                 │               │                  │
          ┌──────▼──────────────▼─────────────────▼──────┐
          │                Apache Kafka                    │
          │     (Event streaming · Decoupled pipeline)     │
          └───────────────────────────────────────────────┘
                 │               │                  │
          ┌──────▼───┐    ┌─────▼─────┐     ┌─────▼─────┐
          │PostgreSQL │    │   Redis   │     │Prometheus │
          │   16      │    │    7      │     │ + Grafana │
          └───────────┘    └───────────┘     └───────────┘
```

## Key Features

### ML / AI
- **Bidirectional LSTM Autoencoder** — learns normal behavioral patterns; detects anomalies via reconstruction error
- **Graph Neural Network (GraphSAGE)** — models entity relationships for lateral movement and collusion detection
- **Ensemble Scoring** — meta-learner combines model outputs with analyst feedback loop
- **Federated Learning** — train across organizational boundaries without sharing raw data (Flower framework)

### Privacy & Compliance
- **Differential Privacy** — OpenDP-backed Laplace/Gaussian mechanisms with per-user budget tracking
- **AES-256-GCM Encryption** — PII encrypted at rest; HKDF-derived pseudonymisation for analytics
- **GDPR/CCPA Ready** — data minimisation, right-to-erasure support, audit logging

### Explainability (XAI)
- **SHAP (DeepExplainer + KernelExplainer)** — per-prediction feature attributions
- **Natural Language Explanations** — auto-generated summaries of anomaly causes
- **Per-feature Reconstruction Error** — interpretable autoencoder diagnostics

### Security
- **JWT with Refresh Token Rotation** — family-based revocation detects token replay
- **RBAC** — admin, analyst, viewer, auditor roles with endpoint-level enforcement
- **Account Lockout** — configurable failed-attempt threshold with auto-unlock
- **NIST 800-63B Password Policy** — minimum 12 characters, no composition rules

### Infrastructure
- **Docker Compose** — one-command full-stack development environment
- **Kubernetes** — production manifests with HPA, Ingress, health checks
- **Prometheus + Grafana** — pre-configured dashboards for request rate, latency, ML inference, Kafka lag

---

## Project Structure

```
├── docker-compose.yml          # Full-stack orchestration
├── Makefile                    # Dev/build/test shortcuts
├── .env.example                # Environment variable template
│
├── shared/                     # Cross-service library
│   ├── models/events.py        # Kafka event schemas (EventEnvelope)
│   ├── utils/logging.py        # Structured logging (structlog)
│   ├── utils/crypto.py         # AES-256-GCM, pseudonymisation
│   └── constants.py            # Topics, risk levels, RBAC roles
│
├── services/
│   ├── api-gateway/            # Port 8000 – sole ingress point
│   ├── data-ingestion/         # Port 8001 – syslog, EDR, network collectors
│   ├── behavioral-analytics/   # Port 8002 – LSTM, GNN, ensemble, XAI
│   ├── risk-scoring/           # Port 8003 – exponential decay risk engine
│   ├── auth-service/           # Port 8004 – JWT, bcrypt, RBAC
│   └── frontend/               # Port 3000 (dev) / 80 (prod) – React SPA
│
├── infrastructure/
│   ├── k8s/                    # Kubernetes manifests
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml
│   │   ├── deployments.yaml
│   │   ├── ingress.yaml
│   │   └── hpa.yaml
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana/            # Provisioning + dashboards
│
└── tests/
    ├── conftest.py             # Shared fixtures
    ├── unit/                   # Fast, isolated tests
    └── integration/            # Service-level tests
```

---

## Quick Start

### Prerequisites
- Docker & Docker Compose v2
- Node.js 20+ (for frontend development)
- Python 3.12+ (for backend development)

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env with your secrets (JWT_SECRET_KEY, ENCRYPTION_KEY, etc.)
```

### 2. Start Everything

```bash
# Full stack (all services + Kafka + Postgres + Redis + Prometheus + Grafana)
make up

# Or with Docker Compose directly
docker compose up -d
```

### 3. Access the System

| Service              | URL                          |
|----------------------|------------------------------|
| Frontend (React)     | http://localhost:3000         |
| API Gateway (Swagger)| http://localhost:8000/docs    |
| Grafana              | http://localhost:3001         |
| Prometheus           | http://localhost:9090         |

### 4. Create Your First User

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@company.com",
    "username": "admin",
    "password": "SecurePass12345!",
    "full_name": "System Admin"
  }'
```

---

## Development

### Backend (any service)

```bash
cd services/api-gateway
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd services/frontend
npm install
npm run dev
```

### Run Tests

```bash
make test              # All tests
make test-unit         # Unit tests only
make lint              # Ruff + mypy
```

---

## ML Pipeline

### Training

The behavioral analytics service processes Kafka events through:

1. **Feature Extraction** → 45-dimensional vector (temporal, volumetric, behavioral, access, network, derived features)
2. **LSTM Autoencoder** → reconstruction error as anomaly signal
3. **GNN (GraphSAGE)** → relational anomaly detection across entity graph
4. **Ensemble** → meta-learner combines scores; falls back to weighted average before training
5. **XAI** → SHAP explanations generated for high-risk anomalies only (cost optimisation)

### Differential Privacy

All aggregate statistics (counts, feature means) are protected by calibrated noise:
- **Laplace mechanism** for counting queries
- **Gaussian mechanism** for feature aggregations
- Per-user privacy budget tracked via `(ε, δ)` accountant

### Federated Learning

Cross-organisational training via Flower:
- Gradient-level DP noise injection
- No raw data leaves the organisation
- Configurable noise multiplier and max gradient norm

---

## Risk Scoring

Risk scores follow an **exponential decay** model:

$$R(t) = R_0 \cdot e^{-\lambda \Delta t} + \sum_{i} w_i \cdot s_i$$

Where:
- $R_0$ = previous risk score
- $\lambda$ = decay rate (configurable, default 0.01)
- $\Delta t$ = hours since last update
- $w_i$ = severity weight for anomaly type $i$
- $s_i$ = anomaly score from ensemble

Alert deduplication uses Redis `SETNX` with configurable cooldown periods.

---

## Security Considerations

- **Never** commit `.env` files with real secrets
- In production, replace `JWT_SECRET_KEY` with a cryptographically random 256-bit key
- Use Sealed Secrets or HashiCorp Vault for Kubernetes secret management
- Enable TLS termination at the Ingress level (cert-manager config included)
- Review `BCRYPT_ROUNDS` setting — higher rounds = more secure but slower
- All API endpoints require authentication except `/healthz` and `/metrics`

---

## Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f infrastructure/k8s/namespace.yaml
kubectl apply -f infrastructure/k8s/configmap.yaml
kubectl apply -f infrastructure/k8s/secrets.yaml
kubectl apply -f infrastructure/k8s/deployments.yaml
kubectl apply -f infrastructure/k8s/ingress.yaml
kubectl apply -f infrastructure/k8s/hpa.yaml

# Or use Make
make k8s-deploy
```

---

## License

MIT

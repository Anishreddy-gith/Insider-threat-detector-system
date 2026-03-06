# ==========================================================
#  Makefile — Insider Threat Detection System
# ==========================================================
# Orchestrates common dev / CI / CD tasks across services.

.PHONY: help dev up down build test lint migrate clean

COMPOSE := docker compose --env-file .env

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Development ────────────────────────────────────────────
dev: ## Start all services in dev mode (with hot-reload)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up --build

up: ## Start all services (detached)
	$(COMPOSE) up -d --build

down: ## Stop all services
	$(COMPOSE) down

build: ## Build all images without cache
	$(COMPOSE) build --no-cache

logs: ## Tail logs for all services
	$(COMPOSE) logs -f

# ── Database ───────────────────────────────────────────────
migrate: ## Run Alembic migrations
	$(COMPOSE) exec api-gateway alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="add users table")
	$(COMPOSE) exec api-gateway alembic revision --autogenerate -m "$(MSG)"

# ── Testing ────────────────────────────────────────────────
test: ## Run all tests with pytest
	cd services/api-gateway && python -m pytest tests/ -v --tb=short
	cd services/behavioral-analytics && python -m pytest tests/ -v --tb=short
	cd services/risk-scoring && python -m pytest tests/ -v --tb=short
	cd services/auth-service && python -m pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	cd services/api-gateway && python -m pytest tests/ --cov=app --cov-report=html

# ── Linting ────────────────────────────────────────────────
lint: ## Lint all Python services
	ruff check services/ shared/ tests/
	mypy services/ shared/ --ignore-missing-imports

format: ## Auto-format code
	ruff format services/ shared/ tests/

# ── ML ─────────────────────────────────────────────────────
train-autoencoder: ## Train the LSTM autoencoder model
	python ml/training/train_autoencoder.py

train-gnn: ## Train the GNN model
	python ml/training/train_gnn.py

# ── Frontend ──────────────────────────────────────────────
frontend-dev: ## Start React dev server
	cd frontend && npm run dev

frontend-build: ## Build frontend for production
	cd frontend && npm run build

frontend-lint: ## Lint frontend code
	cd frontend && npm run lint

# ── Infrastructure ─────────────────────────────────────────
k8s-apply: ## Apply Kubernetes manifests
	kubectl apply -f infrastructure/kubernetes/namespace.yaml
	kubectl apply -f infrastructure/kubernetes/ -R

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -f infrastructure/kubernetes/ -R

# ── Cleanup ────────────────────────────────────────────────
clean: ## Remove containers, volumes, and build artifacts
	$(COMPOSE) down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true

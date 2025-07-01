.PHONY: help install install-dev test test-cov lint format clean docker-build docker-up docker-down

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code"
	@echo "  clean         Clean temporary files"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-up     Start Docker services"
	@echo "  docker-down   Stop Docker services"
	@echo "  run-dev       Run development server"
	@echo "  run-prod      Run production server"
	@echo "  db-upgrade    Run database migrations"
	@echo "  db-downgrade  Rollback database migrations"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev]
	pre-commit install

# Testing
test:
	pytest

test-cov:
	pytest --cov=app --cov-report=html --cov-report=term-missing

# Linting and formatting
lint:
	flake8 app tests
	black --check app tests
	isort --check-only app tests
	mypy app

format:
	black app tests
	isort app tests

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Docker
docker-build:
	docker build -f docker/Dockerfile -t workflow-engine:latest .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# Development
run-dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Database
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-revision:
	alembic revision --autogenerate -m "$(msg)"

# Setup development environment
setup-dev: install-dev
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file from .env.example"; fi
	@echo "Development environment setup complete!"

# CI/CD helpers
ci-test: install-dev lint test-cov

# Production deployment
deploy-prod: docker-build
	@echo "Deploying to production..."
	@echo "This would contain your production deployment steps"
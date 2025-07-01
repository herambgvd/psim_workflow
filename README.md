# Enterprise State Machine Workflow Engine

A robust, scalable workflow engine built with FastAPI, SQLAlchemy, and PostgreSQL for enterprise-grade workflow automation and state machine management.

## 🚀 Features

- **State Machine Engine**: Robust state machine implementation with support for complex workflows
- **Enterprise Ready**: Built for scale with proper logging, monitoring, and error handling
- **RESTful API**: Comprehensive REST API built with FastAPI
- **Database Driven**: PostgreSQL with SQLAlchemy ORM for reliable data persistence
- **Async Processing**: Built-in support for asynchronous task processing (Celery integration planned)
- **Monitoring**: Health checks, metrics, and structured logging
- **Docker Ready**: Containerized deployment with Docker and docker-compose

## 📋 Requirements

- Python 3.11+
- PostgreSQL 12+
- Redis 6+ (for future Celery integration)
- Docker & Docker Compose (optional)

## 🛠️ Installation

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd workflow-engine
   ```

2. **Set up the development environment**
   ```bash
   make setup-dev
   ```

3. **Start the development server**
   ```bash
   make run-dev
   ```

### Docker Setup

1. **Start all services with Docker Compose**
   ```bash
   make docker-up
   ```

2. **Access the application**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 🏗️ Project Structure

```
workflow_engine/
├── app/
│   ├── api/                    # API endpoints
│   ├── core/                   # Core configuration and utilities
│   ├── models/                 # Database models
│   ├── schemas/                # Pydantic schemas
│   ├── services/               # Business logic
│   ├── state_machine/          # State machine engine
│   └── utils/                  # Utility functions
├── docker/                     # Docker configuration
├── scripts/                    # Deployment and utility scripts
├── tests/                      # Test suite
└── requirements/               # Python dependencies
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run linting
make lint
```

## 🔧 Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and update the values:

```bash
cp .env.example .env
```

Key configuration options:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: Application secret key
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENVIRONMENT`: Application environment (development, staging, production)

## 📖 API Documentation

Once the application is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔍 Monitoring

### Health Checks

- **Basic Health**: `GET /health`
- **Detailed Health**: `GET /api/v1/health/detailed`
- **Database Health**: `GET /api/v1/health/database`
- **System Metrics**: `GET /api/v1/health/metrics`

### Logging

The application uses structured logging with the following features:

- JSON formatted logs for production
- Pretty console logs for development
- Request ID tracing
- Automatic log sanitization for sensitive data

## 🚢 Deployment

### Production Deployment

1. **Build Docker image**
   ```bash
   make docker-build
   ```

2. **Deploy with environment variables**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e DATABASE_URL=postgresql://user:pass@host:5432/db \
     -e SECRET_KEY=your-secret-key \
     -e ENVIRONMENT=production \
     workflow-engine:latest
   ```

### Database Migrations

```bash
# Upgrade to latest migration
make db-upgrade

# Create new migration
make db-revision msg="Add new feature"

# Rollback last migration
make db-downgrade
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
#!/bin/bash

# Enterprise Workflow Engine - Production Startup Script

set -e

echo "Starting Enterprise Workflow Engine in production mode..."

# Check environment variables
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL environment variable is required for production"
    exit 1
fi

if [ -z "$SECRET_KEY" ]; then
    echo "ERROR: SECRET_KEY environment variable is required for production"
    exit 1
fi

# Install production dependencies
pip install -r requirements/prod.txt

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the production server
echo "Starting production server with Gunicorn..."
exec gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --keepalive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile -
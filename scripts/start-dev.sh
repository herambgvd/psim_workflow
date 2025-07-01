#!/bin/bash

# Enterprise Workflow Engine - Development Startup Script

set -e

echo "Starting Enterprise Workflow Engine in development mode..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please review and update the .env file with your configuration."
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements/dev.txt

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the development server
echo "Starting development server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8004
#!/bin/bash

# Enterprise Workflow Engine - Database Initialization Script

set -e

echo "Initializing database for Enterprise Workflow Engine..."

# Default values
DB_HOST=${POSTGRES_SERVER:-localhost}
DB_PORT=${POSTGRES_PORT:-5433}
DB_NAME=${POSTGRES_DB:-psim_automation}
DB_USER=${POSTGRES_USER:-postgres}
DB_PASSWORD=${POSTGRES_PASSWORD:-Hanu@0542}

# Check if PostgreSQL is running
echo "Checking PostgreSQL connection..."
until pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; do
    echo "Waiting for PostgreSQL to be ready..."
    sleep 2
done

echo "PostgreSQL is ready!"

# Create database if it doesn't exist
echo "Creating database if it doesn't exist..."
PGPASSWORD=$DB_PASSWORD createdb -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME 2>/dev/null || echo "Database already exists"

# Run Alembic migrations
echo "Running database migrations..."
alembic upgrade head

echo "Database initialization completed successfully!"
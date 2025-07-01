"""
Enterprise State Machine Workflow Engine - API Dependencies

This module provides dependency functions for FastAPI endpoints
including database sessions, authentication, and common utilities.
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
import structlog

from app.core.database import get_db
from app.core.config import get_settings, Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_current_settings() -> Settings:
    """
    Dependency to get current application settings.

    Returns:
        Settings: Current application settings
    """
    return get_settings()


def get_database_session() -> Generator[Session, None, None]:
    """
    Dependency to get database session.

    Yields:
        Session: SQLAlchemy database session
    """
    yield from get_db()


def verify_api_health() -> bool:
    """
    Dependency to verify API health.

    Returns:
        bool: True if API is healthy

    Raises:
        HTTPException: If API is not healthy
    """
    from app.core.database import check_database_health

    if not check_database_health():
        logger.error("API health check failed - database unhealthy")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable - database unhealthy"
        )

    return True
"""
Enterprise State Machine Workflow Engine - Logging Configuration

This module provides structured logging configuration using structlog
for consistent, machine-readable logs across the application.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog
from structlog.types import EventDict, Processor
import uuid
from datetime import datetime

from app.core.config import settings


class RequestIDProcessor:
    """
    Structlog processor that adds request ID to log entries.

    This helps trace requests across different components and services.
    """

    def __init__(self):
        self._local = structlog.contextvars.get_contextvars()

    def __call__(self, logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        """Add request ID to event dictionary."""
        request_id = self._local.get("request_id")
        if request_id:
            event_dict["request_id"] = request_id
        return event_dict


class TimestampProcessor:
    """Custom timestamp processor with consistent formatting."""

    def __call__(self, logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        """Add formatted timestamp to event dictionary."""
        event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return event_dict


class SanitizingProcessor:
    """
    Processor that sanitizes sensitive information from logs.

    This helps prevent accidental logging of passwords, tokens, etc.
    """

    SENSITIVE_KEYS = {
        "password", "passwd", "secret", "token", "key", "authorization",
        "auth", "credential", "credentials", "api_key", "access_token",
        "refresh_token", "session_id", "cookie"
    }

    def __call__(self, logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        """Sanitize sensitive data from event dictionary."""
        return self._sanitize_dict(event_dict)

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary values."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, str) and any(
                    sensitive in key.lower() for sensitive in self.SENSITIVE_KEYS
            ):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        return sanitized


def configure_structlog() -> None:
    """
    Configure structlog with appropriate processors and formatting.

    This sets up structured logging with JSON output for production
    and pretty console output for development.
    """

    # Base processors for all environments
    processors: List[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        TimestampProcessor(),
        RequestIDProcessor(),
        SanitizingProcessor(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add environment-specific processors
    if settings.is_development and not settings.is_testing:
        # Pretty console output for development
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    else:
        # JSON output for production and testing
        processors.extend([
            structlog.processors.JSONRenderer()
        ])

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration dictionary for Python's logging module.

    Returns:
        Dict[str, Any]: Logging configuration dictionary
    """

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "format": "%(message)s",
            },
            "console": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "console" if settings.is_development else "json",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            # Application loggers
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False,
            },
            # SQLAlchemy loggers
            "sqlalchemy.engine": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "sqlalchemy.pool": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            # Celery loggers
            "celery": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "celery.worker": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "celery.task": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            # FastAPI/Uvicorn loggers
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            # Third-party loggers
            "httpx": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "redis": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
    }

    # Add file handler if log file is specified
    if settings.LOG_FILE:
        log_file_path = Path(settings.LOG_FILE)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "json",
            "filename": str(log_file_path),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
        }

        # Add file handler to all loggers
        for logger_config in config["loggers"].values():
            logger_config["handlers"].append("file")
        config["root"]["handlers"].append("file")

    return config


def setup_logging() -> None:
    """
    Initialize logging configuration for the application.

    This function should be called once during application startup
    to configure both Python's logging module and structlog.
    """

    # Configure Python's logging module
    logging_config = get_logging_config()
    logging.config.dictConfig(logging_config)

    # Configure structlog
    configure_structlog()

    # Get logger and log startup message
    logger = structlog.get_logger("app.logging")
    logger.info(
        "Logging configuration initialized",
        log_level=settings.LOG_LEVEL,
        log_format=settings.LOG_FORMAT,
        environment=settings.ENVIRONMENT
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structlog logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        structlog.BoundLogger: Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Something happened", user_id=123)
    """
    return structlog.get_logger(name)


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID in logging context.

    Args:
        request_id: Optional request ID. If not provided, generates a new UUID.

    Returns:
        str: The request ID that was set

    Example:
        # In middleware or request handler
        request_id = set_request_id()
        logger.info("Processing request")  # Will include request_id
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    return request_id


def clear_request_context() -> None:
    """Clear request-specific logging context."""
    structlog.contextvars.clear_contextvars()


def bind_context(**kwargs: Any) -> None:
    """
    Bind additional context to logging.

    Args:
        **kwargs: Key-value pairs to add to logging context

    Example:
        bind_context(user_id=123, workflow_id="wf-456")
        logger.info("User action")  # Will include user_id and workflow_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


class LoggingMiddleware:
    """
    ASGI middleware for request logging.

    This middleware automatically adds request IDs and logs
    request/response information.
    """

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("app.middleware.logging")

    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Set request ID
        request_id = set_request_id()

        # Log request start
        self.logger.info(
            "Request started",
            method=scope["method"],
            path=scope["path"],
            query_string=scope.get("query_string", b"").decode(),
        )

        try:
            await self.app(scope, receive, send)
            self.logger.info("Request completed successfully")
        except Exception as e:
            self.logger.error(
                "Request failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        finally:
            clear_request_context()


# Configure logging on module import
if not settings.is_testing:
    setup_logging()
"""
Enterprise State Machine Workflow Engine - FastAPI Application

This module contains the main FastAPI application setup with middleware,
error handlers, and route registration.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from app.core.config import settings
from app.core.database import create_tables, check_database_health, db_manager
from app.core.logging import get_logger, set_request_id, clear_request_context
from app.api.v1.api import api_router

# Configure logger
logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.

    Automatically logs request details, response status, and timing information
    for all API calls.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request and log details."""
        # Set request ID for tracing
        request_id = set_request_id()
        start_time = time.time()

        # Extract request information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Log request start
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log successful response
            logger.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
                request_id=request_id
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Calculate duration for failed requests
            duration = time.time() - start_time

            # Log error
            logger.error(
                "Request failed",
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(duration * 1000, 2),
                request_id=request_id
            )

            # Re-raise the exception
            raise

        finally:
            # Clean up request context
            clear_request_context()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to responses.

    Adds common security headers to protect against various attacks.
    """

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add HSTS header for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Enterprise Workflow Engine")

    try:
        # Create database tables
        create_tables()
        logger.info("Database tables created/verified")

        # Perform health checks
        if not check_database_health():
            logger.error("Database health check failed")
            raise RuntimeError("Database is not healthy")

        logger.info("Application startup completed successfully")

        yield

    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise

    finally:
        # Shutdown
        logger.info("Shutting down Enterprise Workflow Engine")

        # Close database connections
        db_manager.close()
        logger.info("Database connections closed")

        logger.info("Application shutdown completed")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """

    # Create FastAPI app with metadata
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.DEBUG else None,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan
    )

    # Configure CORS
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add security middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Add trusted host middleware for production
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure with actual allowed hosts
        )

    # Include API routes
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Add global exception handlers
    add_exception_handlers(app)

    return app


def add_exception_handlers(app: FastAPI) -> None:
    """
    Add global exception handlers to the FastAPI application.

    Args:
        app: FastAPI application instance
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions."""
        logger.warning(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "http_error",
                    "code": exc.status_code,
                    "message": exc.detail,
                    "timestamp": time.time()
                }
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
            request: Request,
            exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors."""
        logger.warning(
            "Request validation error",
            errors=exc.errors(),
            path=request.url.path
        )

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "validation_error",
                    "code": 422,
                    "message": "Request validation failed",
                    "details": exc.errors(),
                    "timestamp": time.time()
                }
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(
            request: Request,
            exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle Starlette HTTP exceptions."""
        logger.error(
            "Starlette HTTP exception",
            status_code=exc.status_code,
            detail=str(exc.detail),
            path=request.url.path
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "server_error",
                    "code": exc.status_code,
                    "message": str(exc.detail),
                    "timestamp": time.time()
                }
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.error(
            "Unexpected exception occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
            exc_info=True
        )

        # Don't expose internal errors in production
        if settings.is_production:
            message = "An unexpected error occurred"
        else:
            message = str(exc)

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "code": 500,
                    "message": message,
                    "timestamp": time.time()
                }
            }
        )


# Create the FastAPI application instance
app = create_application()


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing basic API information.

    Returns:
        Dict[str, Any]: API information and status
    """
    return {
        "message": "Enterprise State Machine Workflow Engine",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "docs_url": "/docs" if settings.DEBUG else None,
        "api_prefix": settings.API_V1_STR,
        "timestamp": time.time()
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        Dict[str, Any]: Health status of the application and its dependencies
    """
    start_time = time.time()

    # Check database health
    db_healthy = check_database_health()

    # Calculate response time
    response_time = round((time.time() - start_time) * 1000, 2)

    status = "healthy" if db_healthy else "unhealthy"
    status_code = 200 if db_healthy else 503

    health_data = {
        "status": status,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": time.time(),
        "response_time_ms": response_time,
        "checks": {
            "database": "healthy" if db_healthy else "unhealthy"
        }
    }

    logger.info("Health check performed", health_status=status, response_time_ms=response_time)

    return JSONResponse(
        status_code=status_code,
        content=health_data
    )


# Readiness probe endpoint
@app.get("/ready", tags=["Health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint for Kubernetes readiness probes.

    Returns:
        Dict[str, Any]: Readiness status of the application
    """
    # Perform basic checks to ensure the application is ready to serve requests
    db_healthy = check_database_health()

    if not db_healthy:
        raise HTTPException(
            status_code=503,
            detail="Application is not ready - database unavailable"
        )

    return {
        "status": "ready",
        "timestamp": time.time()
    }


# Liveness probe endpoint
@app.get("/live", tags=["Health"])
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check endpoint for Kubernetes liveness probes.

    Returns:
        Dict[str, Any]: Liveness status of the application
    """
    return {
        "status": "alive",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn

    # Run the application with uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD and settings.is_development,
        log_config=None,  # Use our custom logging configuration
        access_log=False,  # We handle access logging in our middleware
    )
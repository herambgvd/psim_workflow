"""
Enterprise State Machine Workflow Engine - FastAPI Application

Updated main FastAPI application with user management integration,
authentication middleware, and enhanced security features.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.api import api_router
from app.core.auth import get_user_service, AuthenticationError, AuthorizationError
from app.core.config import settings
from app.core.database import create_tables, check_database_health, db_manager
from app.core.logging import get_logger, set_request_id, clear_request_context

# Configure logger
logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses with user context.

    Automatically logs request details, response status, timing information,
    and user context for all API calls.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request and log details."""
        # Set request ID for tracing
        request_id = set_request_id()
        start_time = time.time()

        # Extract request information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Try to extract user info from Authorization header for logging
        user_info = {"user_id": None, "username": None}
        if settings.auth_enabled:
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                try:
                    token = auth_header.split(" ")[1]
                    user_service = get_user_service()
                    user_data = await user_service.validate_token(token)
                    user_info = {
                        "user_id": str(user_data.get("id")),
                        "username": user_data.get("username")
                    }
                except Exception:
                    # Don't fail the request if user extraction fails
                    pass

        # Log request start
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            user_id=user_info["user_id"],
            username=user_info["username"]
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
                request_id=request_id,
                user_id=user_info["user_id"],
                username=user_info["username"]
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
                request_id=request_id,
                user_id=user_info["user_id"],
                username=user_info["username"]
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

        # Add authentication-related headers
        if settings.auth_enabled:
            response.headers["X-Auth-Required"] = "true"
            response.headers["X-Auth-Service"] = settings.USER_SERVICE_URL
        else:
            response.headers["X-Auth-Required"] = "false"

        return response


class UserServiceHealthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to check user service health for authenticated endpoints.

    Provides circuit breaker functionality for user service calls.
    """

    def __init__(self, app):
        super().__init__(app)
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open = False

    async def dispatch(self, request: Request, call_next):
        """Check user service health for protected endpoints."""
        # Skip health check for public endpoints
        if not settings.auth_enabled or request.url.path in ["/", "/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Check if request has authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return await call_next(request)

        # Check circuit breaker
        if self.circuit_open:
            current_time = time.time()
            if current_time - self.last_failure_time > settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT:
                self.circuit_open = False
                self.failure_count = 0
                logger.info("User service circuit breaker reset")
            else:
                logger.warning("User service circuit breaker is open")
                raise HTTPException(
                    status_code=503,
                    detail="User service temporarily unavailable"
                )

        try:
            # Check user service health
            user_service = get_user_service()
            is_healthy = await user_service.health_check()

            if not is_healthy:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                    self.circuit_open = True
                    logger.error("User service circuit breaker opened")

                if settings.RBAC_STRICT_MODE:
                    raise HTTPException(
                        status_code=503,
                        detail="User service unavailable"
                    )
            else:
                # Reset failure count on successful health check
                self.failure_count = 0

            return await call_next(request)

        except Exception as e:
            if isinstance(e, HTTPException):
                raise

            logger.error("User service health check failed", error=str(e))

            if settings.RBAC_STRICT_MODE:
                raise HTTPException(
                    status_code=503,
                    detail="Authentication service error"
                )

            return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application
    including user service integration.
    """
    # Startup
    logger.info("Starting Enterprise Workflow Engine with User Management Integration")

    try:
        # Create database tables
        create_tables()
        logger.info("Database tables created/verified")

        # Perform health checks
        if not check_database_health():
            logger.error("Database health check failed")
            raise RuntimeError("Database is not healthy")

        # Check user service connectivity if authentication is enabled
        if settings.auth_enabled:
            try:
                user_service = get_user_service()
                user_service_healthy = await user_service.health_check()

                if user_service_healthy:
                    logger.info("User service connection verified")
                else:
                    logger.warning("User service is not responding")
                    if settings.RBAC_STRICT_MODE:
                        raise RuntimeError("User service is required but not available")

            except Exception as e:
                logger.error("Failed to connect to user service", error=str(e))
                if settings.RBAC_STRICT_MODE:
                    raise RuntimeError(f"User service connection failed: {str(e)}")

        logger.info(
            "Application startup completed successfully",
            auth_enabled=settings.auth_enabled,
            rbac_enabled=settings.rbac_enabled,
            user_service_url=settings.USER_SERVICE_URL if settings.auth_enabled else None
        )

        yield

    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise

    finally:
        # Shutdown
        logger.info("Shutting down Enterprise Workflow Engine")

        # Close user service connections
        if settings.auth_enabled:
            try:
                user_service = get_user_service()
                await user_service.close()
                logger.info("User service connections closed")
            except Exception as e:
                logger.error("Error closing user service connections", error=str(e))

        # Close database connections
        db_manager.close()
        logger.info("Database connections closed")

        logger.info("Application shutdown completed")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application with user management integration.

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

    # Add user service health middleware if authentication is enabled
    if settings.auth_enabled and settings.CIRCUIT_BREAKER_ENABLED:
        app.add_middleware(UserServiceHealthMiddleware)

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

    @app.exception_handler(AuthenticationError)
    async def authentication_exception_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
        """Handle authentication errors."""
        logger.warning(
            "Authentication error occurred",
            error=str(exc),
            path=request.url.path,
            client_ip=request.client.host if request.client else None
        )

        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "type": "authentication_error",
                    "code": 401,
                    "message": str(exc),
                    "timestamp": time.time()
                }
            }
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_exception_handler(request: Request, exc: AuthorizationError) -> JSONResponse:
        """Handle authorization errors."""
        logger.warning(
            "Authorization error occurred",
            error=str(exc),
            path=request.url.path,
            client_ip=request.client.host if request.client else None
        )

        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "type": "authorization_error",
                    "code": 403,
                    "message": str(exc),
                    "timestamp": time.time()
                }
            }
        )

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
        "message": "GVD Workflow Engine",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "docs_url": "/docs" if settings.DEBUG else None,
        "api_prefix": settings.API_V1_STR,
        "authentication": {
            "enabled": settings.auth_enabled,
            "user_service_url": settings.USER_SERVICE_URL if settings.auth_enabled else None
        },
        "authorization": {
            "enabled": settings.rbac_enabled,
            "strict_mode": settings.RBAC_STRICT_MODE
        },
        "timestamp": time.time()
    }


# Health check endpoint (public)
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

    # Check user service health if authentication is enabled
    user_service_healthy = True
    if settings.auth_enabled:
        try:
            user_service = get_user_service()
            user_service_healthy = await user_service.health_check()
        except Exception as e:
            logger.error("User service health check failed", error=str(e))
            user_service_healthy = False

    # Calculate response time
    response_time = round((time.time() - start_time) * 1000, 2)

    # Determine overall status
    if not db_healthy:
        status = "unhealthy"
        status_code = 503
    elif settings.auth_enabled and not user_service_healthy and settings.rbac_enabled:
        status = "degraded"
        status_code = 200  # Still functional but degraded
    else:
        status = "healthy"
        status_code = 200

    health_data = {
        "status": status,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": time.time(),
        "response_time_ms": response_time,
        "checks": {
            "database": "healthy" if db_healthy else "unhealthy",
            "user_service": "healthy" if user_service_healthy else "unhealthy" if settings.auth_enabled else "disabled"
        },
        "authentication": {
            "enabled": settings.auth_enabled,
            "user_service_healthy": user_service_healthy if settings.auth_enabled else None
        }
    }

    logger.info(
        "Health check performed",
        health_status=status,
        response_time_ms=response_time,
        db_healthy=db_healthy,
        user_service_healthy=user_service_healthy
    )

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

    # Check user service if authentication is required
    if settings.auth_enabled and settings.rbac_enabled:
        try:
            user_service = get_user_service()
            user_service_healthy = await user_service.health_check()

            if not user_service_healthy:
                raise HTTPException(
                    status_code=503,
                    detail="Application is not ready - user service unavailable"
                )
        except Exception as e:
            logger.error("User service readiness check failed", error=str(e))
            raise HTTPException(
                status_code=503,
                detail="Application is not ready - user service error"
            )

    return {
        "status": "ready",
        "timestamp": time.time(),
        "authentication_ready": not settings.auth_enabled or True  # Would check user service
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

"""
Enterprise State Machine Workflow Engine - Health Endpoints

Updated health check endpoints with authentication and user service integration.
"""

import time
import psutil
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import structlog

from app.api.deps import (
    get_database_session,
    get_current_settings,
    require_system_health,
    require_system_metrics,
    get_current_user_optional
)
from app.core.config import Settings
from app.core.database import check_database_health, get_database_info
from app.core.logging import get_logger
from app.core.auth import get_user_service

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", summary="Basic Health Check")
async def basic_health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns basic health status without detailed checks.
    Suitable for load balancer health checks.
    No authentication required.

    Returns:
        Dict[str, Any]: Basic health status
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "workflow-engine"
    }


@router.get("/detailed", summary="Detailed Health Check")
async def detailed_health_check(
        settings: Settings = Depends(get_current_settings),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_system_health)
) -> Dict[str, Any]:
    """
    Detailed health check with comprehensive system information.

    Checks all critical components and returns detailed status information.
    Requires system health viewing permission.

    Args:
        settings: Application settings
        db: Database session
        current_user: Authenticated user with health permissions

    Returns:
        Dict[str, Any]: Detailed health status and system information
    """
    start_time = time.time()

    # Initialize health status
    overall_status = "healthy"
    checks = {}

    # Database health check
    try:
        db_healthy = check_database_health()
        db_info = get_database_info()

        checks["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "info": db_info
        }

        if not db_healthy:
            overall_status = "unhealthy"

    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        checks["database"] = {
            "status": "error",
            "error": str(e)
        }
        overall_status = "unhealthy"

    # User service health check
    if settings.auth_enabled:
        try:
            user_service = get_user_service()
            user_service_healthy = await user_service.health_check()

            checks["user_service"] = {
                "status": "healthy" if user_service_healthy else "unhealthy",
                "url": settings.USER_SERVICE_URL,
                "enabled": settings.auth_enabled
            }

            if not user_service_healthy:
                if settings.rbac_enabled:
                    overall_status = "unhealthy"
                else:
                    overall_status = "degraded"

        except Exception as e:
            logger.error("User service health check failed", error=str(e))
            checks["user_service"] = {
                "status": "error",
                "error": str(e),
                "enabled": settings.auth_enabled
            }
            if settings.rbac_enabled:
                overall_status = "unhealthy"
    else:
        checks["user_service"] = {
            "status": "disabled",
            "enabled": False
        }

    # Memory usage check
    try:
        memory = psutil.virtual_memory()
        checks["memory"] = {
            "status": "healthy" if memory.percent < 90 else "warning",
            "usage_percent": memory.percent,
            "available_mb": round(memory.available / 1024 / 1024, 2),
            "total_mb": round(memory.total / 1024 / 1024, 2)
        }

        if memory.percent > 95:
            overall_status = "unhealthy"

    except Exception as e:
        logger.error("Memory check failed", error=str(e))
        checks["memory"] = {
            "status": "error",
            "error": str(e)
        }

    # CPU usage check
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        checks["cpu"] = {
            "status": "healthy" if cpu_percent < 90 else "warning",
            "usage_percent": cpu_percent,
            "core_count": psutil.cpu_count()
        }

        if cpu_percent > 95:
            overall_status = "unhealthy"

    except Exception as e:
        logger.error("CPU check failed", error=str(e))
        checks["cpu"] = {
            "status": "error",
            "error": str(e)
        }

    # Disk usage check
    try:
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100

        checks["disk"] = {
            "status": "healthy" if disk_percent < 90 else "warning",
            "usage_percent": round(disk_percent, 2),
            "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
            "total_gb": round(disk.total / 1024 / 1024 / 1024, 2)
        }

        if disk_percent > 95:
            overall_status = "unhealthy"

    except Exception as e:
        logger.error("Disk check failed", error=str(e))
        checks["disk"] = {
            "status": "error",
            "error": str(e)
        }

    # Application-specific checks
    checks["application"] = {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "debug_mode": settings.DEBUG,
        "auth_enabled": settings.auth_enabled,
        "rbac_enabled": settings.rbac_enabled
    }

    # Security checks
    checks["security"] = {
        "status": "healthy",
        "authentication": "enabled" if settings.auth_enabled else "disabled",
        "authorization": "enabled" if settings.rbac_enabled else "disabled",
        "user_service_connected": checks.get("user_service", {}).get("status") == "healthy"
    }

    # Calculate response time
    response_time = round((time.time() - start_time) * 1000, 2)

    health_data = {
        "status": overall_status,
        "timestamp": time.time(),
        "response_time_ms": response_time,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "uptime_seconds": time.time() - start_time,  # This would be actual uptime in real implementation
        "checked_by": current_user.get("username"),
        "checks": checks
    }

    # Return appropriate status code
    status_code = 200 if overall_status == "healthy" else 503

    logger.info(
        "Detailed health check performed",
        status=overall_status,
        response_time_ms=response_time,
        user_id=current_user.get("id")
    )

    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=health_data)

    return health_data


@router.get("/database", summary="Database Health Check")
async def database_health_check(
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_system_health)
) -> Dict[str, Any]:
    """
    Specific database health check endpoint.

    Tests database connectivity and returns connection information.
    Requires system health viewing permission.

    Args:
        db: Database session
        current_user: Authenticated user with health permissions

    Returns:
        Dict[str, Any]: Database health status and information
    """
    start_time = time.time()

    try:
        # Test database connectivity
        db_healthy = check_database_health()
        db_info = get_database_info()

        response_time = round((time.time() - start_time) * 1000, 2)

        if not db_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "unhealthy",
                    "message": "Database connectivity test failed",
                    "timestamp": time.time(),
                    "response_time_ms": response_time,
                    "checked_by": current_user.get("username")
                }
            )

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "database_info": db_info,
            "checked_by": current_user.get("username")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Database health check failed", error=str(e), user_id=current_user.get("id"))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "message": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/user-service", summary="User Service Health Check")
async def user_service_health_check(
        settings: Settings = Depends(get_current_settings),
        current_user: Dict[str, Any] = Depends(require_system_health)
) -> Dict[str, Any]:
    """
    User management service health check endpoint.

    Tests connectivity to the user management service.
    Requires system health viewing permission.

    Args:
        settings: Application settings
        current_user: Authenticated user with health permissions

    Returns:
        Dict[str, Any]: User service health status
    """
    start_time = time.time()

    if not settings.auth_enabled:
        return {
            "status": "disabled",
            "message": "Authentication is disabled",
            "timestamp": time.time(),
            "checked_by": current_user.get("username")
        }

    try:
        user_service = get_user_service()
        is_healthy = await user_service.health_check()

        response_time = round((time.time() - start_time) * 1000, 2)

        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "unhealthy",
                    "message": "User service connectivity test failed",
                    "service_url": settings.USER_SERVICE_URL,
                    "timestamp": time.time(),
                    "response_time_ms": response_time,
                    "checked_by": current_user.get("username")
                }
            )

        return {
            "status": "healthy",
            "service_url": settings.USER_SERVICE_URL,
            "timestamp": time.time(),
            "response_time_ms": response_time,
            "checked_by": current_user.get("username")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("User service health check failed", error=str(e), user_id=current_user.get("id"))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "message": str(e),
                "service_url": settings.USER_SERVICE_URL,
                "timestamp": time.time()
            }
        )


@router.get("/metrics", summary="Application Metrics")
async def application_metrics(
        current_user: Dict[str, Any] = Depends(require_system_metrics)
) -> Dict[str, Any]:
    """
    Application metrics endpoint.

    Returns various application and system metrics for monitoring.
    Requires system metrics viewing permission.

    Args:
        current_user: Authenticated user with metrics permissions

    Returns:
        Dict[str, Any]: Application metrics
    """
    try:
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage('/')

        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()

        metrics = {
            "timestamp": time.time(),
            "collected_by": current_user.get("username"),
            "system": {
                "memory": {
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "percent": memory.percent
                },
                "cpu": {
                    "percent": cpu_percent,
                    "core_count": psutil.cpu_count()
                },
                "disk": {
                    "total_bytes": disk.total,
                    "free_bytes": disk.free,
                    "used_bytes": disk.used,
                    "percent": round((disk.used / disk.total) * 100, 2)
                }
            },
            "process": {
                "memory": {
                    "rss_bytes": process_memory.rss,
                    "vms_bytes": process_memory.vms
                },
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads(),
                "connections": len(process.connections())
            }
        }

        return metrics

    except Exception as e:
        logger.error("Failed to collect metrics", error=str(e), user_id=current_user.get("id"))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect metrics: {str(e)}"
        )


@router.get("/dependencies", summary="Dependency Status")
async def dependency_status(
        settings: Settings = Depends(get_current_settings),
        current_user: Dict[str, Any] = Depends(require_system_health)
) -> Dict[str, Any]:
    """
    Check status of external dependencies.

    Returns the health status of all external dependencies
    like databases, user service, Redis, external APIs, etc.
    Requires system health viewing permission.

    Args:
        settings: Application settings
        current_user: Authenticated user with health permissions

    Returns:
        Dict[str, Any]: Status of all dependencies
    """
    dependencies = {}
    overall_status = "healthy"

    # Database dependency
    try:
        db_healthy = check_database_health()
        dependencies["database"] = {
            "name": "PostgreSQL",
            "status": "healthy" if db_healthy else "unhealthy",
            "type": "database",
            "critical": True
        }

        if not db_healthy:
            overall_status = "degraded"

    except Exception as e:
        dependencies["database"] = {
            "name": "PostgreSQL",
            "status": "error",
            "type": "database",
            "critical": True,
            "error": str(e)
        }
        overall_status = "unhealthy"

    # User service dependency
    if settings.auth_enabled:
        try:
            user_service = get_user_service()
            user_service_healthy = await user_service.health_check()

            dependencies["user_service"] = {
                "name": "User Management Service",
                "status": "healthy" if user_service_healthy else "unhealthy",
                "type": "authentication",
                "url": settings.USER_SERVICE_URL,
                "critical": settings.rbac_enabled
            }

            if not user_service_healthy and settings.rbac_enabled:
                overall_status = "degraded"

        except Exception as e:
            dependencies["user_service"] = {
                "name": "User Management Service",
                "status": "error",
                "type": "authentication",
                "url": settings.USER_SERVICE_URL,
                "critical": settings.rbac_enabled,
                "error": str(e)
            }
            if settings.rbac_enabled:
                overall_status = "unhealthy"
    else:
        dependencies["user_service"] = {
            "name": "User Management Service",
            "status": "disabled",
            "type": "authentication",
            "critical": False
        }

    # Redis dependency (for future use with Celery)
    dependencies["redis"] = {
        "name": "Redis",
        "status": "not_configured",  # Will be implemented in Phase 3
        "type": "cache",
        "critical": False
    }

    # Future dependencies can be added here
    # dependencies["external_api"] = {...}

    return {
        "status": overall_status,
        "timestamp": time.time(),
        "checked_by": current_user.get("username"),
        "dependencies": dependencies
    }


@router.get("/auth", summary="Authentication Status")
async def authentication_status(
        settings: Settings = Depends(get_current_settings),
        current_user: Dict[str, Any] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """
    Check authentication system status.

    Returns information about the authentication and authorization system.
    No special permissions required - shows public auth status.

    Args:
        settings: Application settings
        current_user: Optional authenticated user

    Returns:
        Dict[str, Any]: Authentication system status
    """
    auth_status = {
        "timestamp": time.time(),
        "authentication": {
            "enabled": settings.auth_enabled,
            "user_service_url": settings.USER_SERVICE_URL if settings.auth_enabled else None,
            "status": "unknown"
        },
        "authorization": {
            "enabled": settings.rbac_enabled,
            "strict_mode": settings.RBAC_STRICT_MODE,
            "status": "unknown"
        },
        "current_user": None
    }

    # Check user service connectivity if auth is enabled
    if settings.auth_enabled:
        try:
            user_service = get_user_service()
            is_healthy = await user_service.health_check()
            auth_status["authentication"]["status"] = "healthy" if is_healthy else "unhealthy"
            auth_status["authorization"]["status"] = "healthy" if is_healthy else "unhealthy"
        except Exception as e:
            auth_status["authentication"]["status"] = "error"
            auth_status["authorization"]["status"] = "error"
            auth_status["authentication"]["error"] = str(e)
    else:
        auth_status["authentication"]["status"] = "disabled"
        auth_status["authorization"]["status"] = "disabled"

    # Include current user info if authenticated
    if current_user:
        auth_status["current_user"] = {
            "id": str(current_user["id"]),
            "username": current_user.get("username"),
            "authenticated": True,
            "permissions_count": len(current_user.get("permissions", [])),
            "roles_count": len(current_user.get("roles", []))
        }
    else:
        auth_status["current_user"] = {
            "authenticated": False
        }

    return auth_status
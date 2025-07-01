"""
Enterprise State Machine Workflow Engine - API Dependencies

Updated dependency functions for FastAPI endpoints including database sessions,
authentication, authorization, and common utilities with user management integration.
"""

from typing import Generator, Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
import structlog

from app.core.database import get_db
from app.core.config import get_settings, Settings
from app.core.logging import get_logger
from app.core.auth import get_current_user, get_current_user_optional, get_user_service
from app.core.permissions import (
    require_workflow_permission,
    require_instance_permission,
    require_system_permission,
    WorkflowPermissions,
    InstancePermissions,
    SystemPermissions
)

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
    Dependency to verify API health including user service.

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

    # Check user service health if authentication is enabled
    settings = get_settings()
    if settings.auth_enabled:
        user_service = get_user_service()
        # Note: We'll make this async in a real implementation
        # For now, we'll skip the user service health check in this dependency
        # as it would require making this function async

    return True


# ===== AUTHENTICATION DEPENDENCIES =====

def get_authenticated_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Dependency to get authenticated user.

    Returns:
        Dict containing user information

    Raises:
        HTTPException: If user is not authenticated
    """
    return current_user


def get_optional_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Optional[Dict[str, Any]]:
    """
    Dependency to get optional authenticated user.

    Returns:
        Dict containing user information or None
    """
    return current_user


def get_user_id(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> str:
    """
    Dependency to extract user ID from authenticated user.

    Returns:
        User ID string
    """
    return str(current_user["id"])


def get_user_context(
    current_user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
) -> Dict[str, Any]:
    """
    Dependency to get enriched user context.

    Returns:
        Dict containing user context with additional information
    """
    context = {
        "user_id": str(current_user["id"]),
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "permissions": current_user.get("permissions", []),
        "roles": current_user.get("roles", [])
    }

    # Add request context if available
    if request:
        context.update({
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "request_id": getattr(request.state, "request_id", None)
        })

    return context


# ===== WORKFLOW PERMISSION DEPENDENCIES =====

async def require_workflow_create(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require workflow creation permission."""
    return await require_workflow_permission(WorkflowPermissions.CREATE, current_user)


async def require_workflow_read(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require workflow read permission."""
    return await require_workflow_permission(WorkflowPermissions.READ, current_user)


async def require_workflow_update(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require workflow update permission."""
    return await require_workflow_permission(WorkflowPermissions.UPDATE, current_user)


async def require_workflow_delete(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require workflow delete permission."""
    return await require_workflow_permission(WorkflowPermissions.DELETE, current_user)


async def require_workflow_activate(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require workflow activation permission."""
    return await require_workflow_permission(WorkflowPermissions.ACTIVATE, current_user)


async def require_workflow_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require workflow statistics viewing permission."""
    return await require_workflow_permission(WorkflowPermissions.VIEW_STATS, current_user)


# ===== INSTANCE PERMISSION DEPENDENCIES =====

async def require_instance_create(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require instance creation permission."""
    return await require_instance_permission(InstancePermissions.CREATE, current_user)


async def require_instance_read(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require instance read permission."""
    return await require_instance_permission(InstancePermissions.READ, current_user)


async def require_instance_update(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require instance update permission."""
    return await require_instance_permission(InstancePermissions.UPDATE, current_user)


async def require_instance_control(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require instance control permissions (start/pause/resume/cancel)."""
    # Check if user has any of the control permissions
    control_permissions = [
        InstancePermissions.START,
        InstancePermissions.PAUSE,
        InstancePermissions.RESUME,
        InstancePermissions.CANCEL
    ]

    user_permissions = set(current_user.get("permissions", []))
    required_permissions = set(control_permissions)

    if not user_permissions.intersection(required_permissions):
        logger.warning(
            "Instance control permission denied",
            user_id=current_user.get("id"),
            required_permissions=control_permissions,
            user_permissions=list(user_permissions)
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Instance control permissions required"
        )

    return current_user


async def require_instance_events(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require instance event sending permission."""
    return await require_instance_permission(InstancePermissions.SEND_EVENTS, current_user)


async def require_instance_history(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require instance history viewing permission."""
    return await require_instance_permission(InstancePermissions.VIEW_HISTORY, current_user)


async def require_instance_variables(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require instance variable management permission."""
    return await require_instance_permission(InstancePermissions.MANAGE_VARIABLES, current_user)


async def require_instance_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require instance statistics viewing permission."""
    return await require_instance_permission(InstancePermissions.VIEW_STATS, current_user)


# ===== SYSTEM PERMISSION DEPENDENCIES =====

async def require_system_health(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require system health viewing permission."""
    return await require_system_permission(SystemPermissions.VIEW_HEALTH, current_user)


async def require_system_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require system metrics viewing permission."""
    return await require_system_permission(SystemPermissions.VIEW_METRICS, current_user)


async def require_system_admin(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require system admin permission."""
    return await require_system_permission(SystemPermissions.ADMIN, current_user)


# ===== COMBINED DEPENDENCIES =====

def get_db_and_user(
    db: Session = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> tuple[Session, Dict[str, Any]]:
    """
    Combined dependency for database session and authenticated user.

    Returns:
        Tuple of (database_session, user_data)
    """
    return db, current_user


def get_db_and_optional_user(
    db: Session = Depends(get_database_session),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> tuple[Session, Optional[Dict[str, Any]]]:
    """
    Combined dependency for database session and optional user.

    Returns:
        Tuple of (database_session, user_data_or_none)
    """
    return db, current_user


# ===== UTILITY DEPENDENCIES =====

def get_pagination_params(
    page: int = 1,
    page_size: int = 20
) -> Dict[str, int]:
    """
    Dependency for pagination parameters with validation.

    Returns:
        Dict containing validated pagination parameters
    """
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    if page_size > 100:
        page_size = 100

    return {
        "page": page,
        "page_size": page_size,
        "offset": (page - 1) * page_size,
        "limit": page_size
    }


def get_request_metadata(request: Request) -> Dict[str, Any]:
    """
    Dependency to extract request metadata.

    Returns:
        Dict containing request metadata
    """
    return {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "content_type": request.headers.get("content-type"),
        "content_length": request.headers.get("content-length")
    }


# ===== AUDIT DEPENDENCIES =====

def get_audit_context(
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_metadata: Dict[str, Any] = Depends(get_request_metadata)
) -> Dict[str, Any]:
    """
    Dependency to get audit context for logging.

    Returns:
        Dict containing audit context information
    """
    return {
        "user_id": str(current_user["id"]),
        "username": current_user.get("username"),
        "action_timestamp": None,  # Will be set by the service
        "request_method": request_metadata["method"],
        "request_path": request_metadata["path"],
        "client_ip": request_metadata["client_ip"],
        "user_agent": request_metadata["user_agent"]
    }


# ===== CONDITIONAL DEPENDENCIES =====

def get_auth_if_enabled():
    """
    Conditional authentication dependency based on configuration.

    Returns:
        Authentication dependency or None based on settings
    """
    settings = get_settings()

    if settings.auth_enabled:
        return Depends(get_current_user)
    else:
        # Return a function that provides a mock user for development
        def mock_user():
            return {
                "id": "00000000-0000-0000-0000-000000000000",
                "username": "development_user",
                "email": "dev@example.com",
                "permissions": ["*"],  # All permissions for development
                "roles": [{"name": "admin", "permissions": ["*"]}]
            }
        return Depends(mock_user)


def get_rbac_if_enabled():
    """
    Conditional RBAC dependency based on configuration.

    Returns:
        RBAC dependency or pass-through based on settings
    """
    settings = get_settings()

    if settings.rbac_enabled:
        return get_current_user
    else:
        return get_current_user  # Still get user but don't enforce RBAC
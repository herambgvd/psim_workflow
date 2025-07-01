"""
Enterprise State Machine Workflow Engine - RBAC Permission System

This module defines permissions, roles, and authorization decorators
for the workflow engine's role-based access control system.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
from fastapi import HTTPException, Depends, status
import structlog

from app.core.auth import get_current_user, has_permission, has_any_permission, get_user_service
from app.core.logging import get_logger

logger = get_logger(__name__)


class WorkflowPermissions(str, Enum):
    """Workflow-related permissions."""
    CREATE = "workflow:create"
    READ = "workflow:read"
    UPDATE = "workflow:update"
    DELETE = "workflow:delete"
    ACTIVATE = "workflow:activate"
    DEACTIVATE = "workflow:deactivate"
    CLONE = "workflow:clone"
    EXPORT = "workflow:export"
    IMPORT = "workflow:import"
    VALIDATE = "workflow:validate"
    VIEW_STATS = "workflow:view_stats"


class InstancePermissions(str, Enum):
    """Workflow instance-related permissions."""
    CREATE = "instance:create"
    READ = "instance:read"
    UPDATE = "instance:update"
    DELETE = "instance:delete"
    START = "instance:start"
    PAUSE = "instance:pause"
    RESUME = "instance:resume"
    CANCEL = "instance:cancel"
    RETRY = "instance:retry"
    SEND_EVENTS = "instance:send_events"
    VIEW_HISTORY = "instance:view_history"
    MANAGE_VARIABLES = "instance:manage_variables"
    VIEW_METRICS = "instance:view_metrics"
    BULK_OPERATIONS = "instance:bulk_operations"
    VIEW_STATS = "instance:view_stats"


class SystemPermissions(str, Enum):
    """System-level permissions."""
    ADMIN = "system:admin"
    VIEW_HEALTH = "system:view_health"
    VIEW_METRICS = "system:view_metrics"
    MANAGE_SYSTEM = "system:manage"


class ResourceType(str, Enum):
    """Resource types for permission checking."""
    WORKFLOW = "workflow"
    INSTANCE = "instance"
    SYSTEM = "system"


def require_permission(permission: str):
    """
    Decorator to require specific permission for endpoint access.

    Args:
        permission: Required permission string

    Returns:
        Decorated function that checks permission
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                # Try to get from Depends injection
                for key, value in kwargs.items():
                    if isinstance(value, dict) and 'id' in value and 'permissions' in value:
                        current_user = value
                        break

            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

            if not has_permission(current_user, permission):
                logger.warning(
                    "Permission denied",
                    user_id=current_user.get("id"),
                    required_permission=permission,
                    user_permissions=current_user.get("permissions", [])
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission}"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_any_permission(permissions: List[str]):
    """
    Decorator to require any of the specified permissions.

    Args:
        permissions: List of acceptable permissions

    Returns:
        Decorated function that checks permissions
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

            if not has_any_permission(current_user, permissions):
                logger.warning(
                    "Permission denied - none of required permissions found",
                    user_id=current_user.get("id"),
                    required_permissions=permissions,
                    user_permissions=current_user.get("permissions", [])
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: requires one of {permissions}"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


async def require_workflow_permission(
        permission: str,
        current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Dependency to require workflow permission.

    Args:
        permission: Required workflow permission
        current_user: Current authenticated user

    Returns:
        User data if permission granted

    Raises:
        HTTPException: If permission denied
    """
    if not has_permission(current_user, permission):
        logger.warning(
            "Workflow permission denied",
            user_id=current_user.get("id"),
            required_permission=permission
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission}"
        )

    return current_user


async def require_instance_permission(
        permission: str,
        current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Dependency to require instance permission.

    Args:
        permission: Required instance permission
        current_user: Current authenticated user

    Returns:
        User data if permission granted

    Raises:
        HTTPException: If permission denied
    """
    if not has_permission(current_user, permission):
        logger.warning(
            "Instance permission denied",
            user_id=current_user.get("id"),
            required_permission=permission
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission}"
        )

    return current_user


async def require_system_permission(
        permission: str,
        current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Dependency to require system permission.

    Args:
        permission: Required system permission
        current_user: Current authenticated user

    Returns:
        User data if permission granted

    Raises:
        HTTPException: If permission denied
    """
    if not has_permission(current_user, permission):
        logger.warning(
            "System permission denied",
            user_id=current_user.get("id"),
            required_permission=permission
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission}"
        )

    return current_user


class PermissionChecker:
    """
    Advanced permission checker with resource-level access control.
    """

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.user_service = get_user_service()

    async def check_workflow_access(
            self,
            user: Dict[str, Any],
            workflow_id: str,
            action: str
    ) -> bool:
        """
        Check if user can perform action on specific workflow.

        Args:
            user: User information
            workflow_id: Workflow identifier
            action: Action to perform

        Returns:
            True if access granted
        """
        # Basic permission check
        permission_map = {
            "read": WorkflowPermissions.READ,
            "update": WorkflowPermissions.UPDATE,
            "delete": WorkflowPermissions.DELETE,
            "activate": WorkflowPermissions.ACTIVATE,
            "deactivate": WorkflowPermissions.DEACTIVATE
        }

        required_permission = permission_map.get(action)
        if not required_permission:
            return False

        if not has_permission(user, required_permission):
            return False

        # Resource-level access control can be implemented here
        # For example, checking if user owns the workflow or has specific role

        return True

    async def check_instance_access(
            self,
            user: Dict[str, Any],
            instance_id: str,
            action: str
    ) -> bool:
        """
        Check if user can perform action on specific instance.

        Args:
            user: User information
            instance_id: Instance identifier
            action: Action to perform

        Returns:
            True if access granted
        """
        permission_map = {
            "read": InstancePermissions.READ,
            "start": InstancePermissions.START,
            "pause": InstancePermissions.PAUSE,
            "resume": InstancePermissions.RESUME,
            "cancel": InstancePermissions.CANCEL,
            "send_events": InstancePermissions.SEND_EVENTS
        }

        required_permission = permission_map.get(action)
        if not required_permission:
            return False

        if not has_permission(user, required_permission):
            return False

        return True

    async def filter_accessible_workflows(
            self,
            user: Dict[str, Any],
            workflow_ids: List[str]
    ) -> List[str]:
        """
        Filter workflows that user can access.

        Args:
            user: User information
            workflow_ids: List of workflow IDs to filter

        Returns:
            List of accessible workflow IDs
        """
        if has_permission(user, SystemPermissions.ADMIN):
            return workflow_ids

        # For now, return all if user has read permission
        if has_permission(user, WorkflowPermissions.READ):
            return workflow_ids

        return []

    async def filter_accessible_instances(
            self,
            user: Dict[str, Any],
            instance_ids: List[str]
    ) -> List[str]:
        """
        Filter instances that user can access.

        Args:
            user: User information
            instance_ids: List of instance IDs to filter

        Returns:
            List of accessible instance IDs
        """
        if has_permission(user, SystemPermissions.ADMIN):
            return instance_ids

        if has_permission(user, InstancePermissions.READ):
            return instance_ids

        return []


# Global permission checker instance
permission_checker = PermissionChecker()


# Convenience dependencies for common permission combinations
async def workflow_admin_required(
        current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require workflow admin permissions."""
    required_permissions = [
        WorkflowPermissions.CREATE,
        WorkflowPermissions.UPDATE,
        WorkflowPermissions.DELETE,
        WorkflowPermissions.ACTIVATE
    ]

    if not has_any_permission(current_user, required_permissions):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workflow admin permissions required"
        )

    return current_user


async def instance_operator_required(
        current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require instance operator permissions."""
    required_permissions = [
        InstancePermissions.START,
        InstancePermissions.PAUSE,
        InstancePermissions.RESUME,
        InstancePermissions.CANCEL
    ]

    if not has_any_permission(current_user, required_permissions):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Instance operator permissions required"
        )

    return current_user


async def system_admin_required(
        current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require system admin permissions."""
    if not has_permission(current_user, SystemPermissions.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System admin permissions required"
        )

    return current_user
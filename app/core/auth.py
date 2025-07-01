"""
Enterprise State Machine Workflow Engine - Authentication Core

This module provides JWT authentication integration with the user management service
including token validation, user context extraction, and security dependencies.
"""

from functools import lru_cache
from typing import Dict, Optional, Any

import httpx
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Authentication related errors."""
    pass


class AuthorizationError(Exception):
    """Authorization related errors."""
    pass


class UserServiceClient:
    """
    Client for communicating with the user management service.

    Handles authentication validation, user data retrieval,
    and permission checking.
    """

    def __init__(self):
        self.base_url = settings.USER_SERVICE_URL.rstrip('/')
        self.timeout = settings.USER_SERVICE_TIMEOUT
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Create HTTP client with proper configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"WorkflowEngine/{settings.VERSION}"
            }
        )

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token with user management service.

        Args:
            token: JWT token to validate

        Returns:
            Dict containing user information

        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            self.logger.debug("Validating JWT token with user service")

            response = await self.client.get(
                f"{self.base_url}/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code == 401:
                raise AuthenticationError("Invalid or expired token")
            elif response.status_code == 403:
                raise AuthenticationError("Token does not have required permissions")
            elif response.status_code != 200:
                raise AuthenticationError(f"Authentication service error: {response.status_code}")

            user_data = response.json()

            self.logger.debug(
                "Token validation successful",
                user_id=user_data.get("id"),
                username=user_data.get("username")
            )

            return user_data

        except httpx.TimeoutException:
            self.logger.error("Timeout connecting to user management service")
            raise AuthenticationError("Authentication service unavailable")
        except httpx.RequestError as e:
            self.logger.error("Request error connecting to user service", error=str(e))
            raise AuthenticationError("Authentication service unavailable")
        except AuthenticationError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error during token validation", error=str(e))
            raise AuthenticationError("Authentication failed")

    async def get_user_permissions(self, user_id: str) -> list:
        """
        Get all permissions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of permission strings
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/rbac/me/permissions",
                headers={"X-User-ID": user_id}
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("permissions", [])
            else:
                self.logger.warning(
                    "Failed to get user permissions",
                    user_id=user_id,
                    status_code=response.status_code
                )
                return []

        except Exception as e:
            self.logger.error("Error getting user permissions", user_id=user_id, error=str(e))
            return []

    async def check_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if user has specific permission.

        Args:
            user_id: User identifier
            permission: Permission to check

        Returns:
            True if user has permission
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/rbac/me/check-permission/{permission}",
                headers={"X-User-ID": user_id}
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("has_permission", False)
            else:
                self.logger.warning(
                    "Permission check failed",
                    user_id=user_id,
                    permission=permission,
                    status_code=response.status_code
                )
                return False

        except Exception as e:
            self.logger.error(
                "Error checking permission",
                user_id=user_id,
                permission=permission,
                error=str(e)
            )
            return False

    async def get_user_roles(self, user_id: str) -> list:
        """
        Get all roles for a user.

        Args:
            user_id: User identifier

        Returns:
            List of role information
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/rbac/me/roles",
                headers={"X-User-ID": user_id}
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("roles", [])
            else:
                return []

        except Exception as e:
            self.logger.error("Error getting user roles", user_id=user_id, error=str(e))
            return []

    async def health_check(self) -> bool:
        """Check if user management service is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Global user service client
@lru_cache()
def get_user_service() -> UserServiceClient:
    """Get cached user service client instance."""
    return UserServiceClient()


async def get_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Dict containing user information

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_service = get_user_service()
        user_data = await user_service.validate_token(credentials.credentials)

        # Add additional context
        user_data["permissions"] = await user_service.get_user_permissions(
            str(user_data["id"])
        )
        user_data["roles"] = await user_service.get_user_roles(
            str(user_data["id"])
        )

        return user_data

    except AuthenticationError as e:
        logger.warning("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Unexpected authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


async def get_current_user_optional(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Optional authentication dependency.

    Returns user if authenticated, None otherwise.
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def get_user_id(current_user: Dict[str, Any]) -> str:
    """Extract user ID from user data."""
    return str(current_user["id"])


def get_username(current_user: Dict[str, Any]) -> str:
    """Extract username from user data."""
    return current_user.get("username", "unknown")


def has_permission(user: Dict[str, Any], permission: str) -> bool:
    """Check if user has specific permission."""
    permissions = user.get("permissions", [])
    return permission in permissions


def has_any_permission(user: Dict[str, Any], permissions: list) -> bool:
    """Check if user has any of the specified permissions."""
    user_permissions = set(user.get("permissions", []))
    required_permissions = set(permissions)
    return bool(user_permissions.intersection(required_permissions))


def has_role(user: Dict[str, Any], role_name: str) -> bool:
    """Check if user has specific role."""
    roles = user.get("roles", [])
    for role in roles:
        if isinstance(role, dict) and role.get("name") == role_name:
            return True
        elif isinstance(role, str) and role == role_name:
            return True
    return False

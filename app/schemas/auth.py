"""
Enterprise State Machine Workflow Engine - Authentication Schemas

Pydantic schemas for authentication, authorization, and user management integration.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class UserRole(BaseModel):
    """User role information."""

    id: UUID = Field(..., description="Role ID")
    name: str = Field(..., description="Role name")
    description: Optional[str] = Field(None, description="Role description")
    permissions: List[str] = Field(default_factory=list, description="Role permissions")

    model_config = ConfigDict(from_attributes=True)


class UserInfo(BaseModel):
    """User information schema."""

    id: UUID = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    is_active: bool = Field(True, description="User active status")
    is_verified: bool = Field(False, description="User verification status")
    roles: List[UserRole] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)

    @property
    def full_name(self) -> str:
        """Get user's full name."""
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        return " ".join(parts) if parts else self.username

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role."""
        return any(role.name == role_name for role in self.roles)


class AuthenticationStatus(str, Enum):
    """Authentication status enumeration."""
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    EXPIRED = "expired"
    INVALID = "invalid"
    SUSPENDED = "suspended"


class PermissionCheck(BaseModel):
    """Permission check request."""

    permission: str = Field(..., description="Permission to check")
    resource_id: Optional[str] = Field(None, description="Resource ID for resource-level checks")
    resource_type: Optional[str] = Field(None, description="Resource type")


class PermissionCheckResult(BaseModel):
    """Permission check result."""

    permission: str = Field(..., description="Checked permission")
    has_permission: bool = Field(..., description="Whether user has permission")
    reason: Optional[str] = Field(None, description="Reason for denial")
    checked_at: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")


class BulkPermissionCheck(BaseModel):
    """Bulk permission check request."""

    permissions: List[str] = Field(..., description="Permissions to check")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    resource_type: Optional[str] = Field(None, description="Resource type")


class BulkPermissionCheckResult(BaseModel):
    """Bulk permission check result."""

    results: Dict[str, bool] = Field(..., description="Permission check results")
    checked_at: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")


class AuthContext(BaseModel):
    """Authentication context for requests."""

    user: UserInfo = Field(..., description="Authenticated user")
    session_id: Optional[str] = Field(None, description="Session ID")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    authenticated_at: datetime = Field(default_factory=datetime.utcnow, description="Authentication timestamp")
    expires_at: Optional[datetime] = Field(None, description="Token expiration time")


class SecurityEvent(BaseModel):
    """Security event for audit logging."""

    event_type: str = Field(..., description="Event type")
    user_id: Optional[UUID] = Field(None, description="User ID")
    resource_type: Optional[str] = Field(None, description="Resource type")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    action: str = Field(..., description="Action performed")
    result: str = Field(..., description="Event result (success/failure)")
    ip_address: Optional[str] = Field(None, description="Client IP")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


class AccessLog(BaseModel):
    """Access log entry."""

    user_id: UUID = Field(..., description="User ID")
    endpoint: str = Field(..., description="API endpoint")
    method: str = Field(..., description="HTTP method")
    status_code: int = Field(..., description="Response status code")
    response_time: float = Field(..., description="Response time in seconds")
    ip_address: Optional[str] = Field(None, description="Client IP")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Access timestamp")


class ServiceHealthCheck(BaseModel):
    """User service health check result."""

    service_name: str = Field(..., description="Service name")
    status: str = Field(..., description="Health status")
    response_time: float = Field(..., description="Response time in seconds")
    version: Optional[str] = Field(None, description="Service version")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


# Error response schemas
class AuthError(BaseModel):
    """Authentication error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class PermissionError(BaseModel):
    """Permission error response."""

    error: str = Field("permission_denied", description="Error type")
    message: str = Field(..., description="Error message")
    required_permission: str = Field(..., description="Required permission")
    user_permissions: List[str] = Field(default_factory=list, description="User's current permissions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Request/Response schemas for user service integration
class UserServiceTokenValidation(BaseModel):
    """User service token validation response."""

    valid: bool = Field(..., description="Token validity")
    user: Optional[UserInfo] = Field(None, description="User information")
    expires_at: Optional[datetime] = Field(None, description="Token expiration")
    error: Optional[str] = Field(None, description="Validation error")


class UserServicePermissionResponse(BaseModel):
    """User service permission check response."""

    user_id: UUID = Field(..., description="User ID")
    permissions: List[str] = Field(..., description="User permissions")
    roles: List[UserRole] = Field(..., description="User roles")
    last_updated: datetime = Field(..., description="Last update timestamp")


class UserServiceRequest(BaseModel):
    """Base request to user service."""

    user_id: UUID = Field(..., description="User ID")
    action: str = Field(..., description="Action to perform")
    resource_type: Optional[str] = Field(None, description="Resource type")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# Audit and compliance schemas
class AuditEvent(BaseModel):
    """Audit event for compliance tracking."""

    event_id: UUID = Field(..., description="Event ID")
    user_id: Optional[UUID] = Field(None, description="User ID")
    event_type: str = Field(..., description="Event type")
    resource_type: str = Field(..., description="Resource type")
    resource_id: str = Field(..., description="Resource ID")
    action: str = Field(..., description="Action performed")
    result: str = Field(..., description="Action result")
    before_state: Optional[Dict[str, Any]] = Field(None, description="State before action")
    after_state: Optional[Dict[str, Any]] = Field(None, description="State after action")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    ip_address: Optional[str] = Field(None, description="Client IP")
    user_agent: Optional[str] = Field(None, description="Client user agent")


class ComplianceReport(BaseModel):
    """Compliance report for audit purposes."""

    report_id: UUID = Field(..., description="Report ID")
    report_type: str = Field(..., description="Report type")
    start_date: datetime = Field(..., description="Report start date")
    end_date: datetime = Field(..., description="Report end date")
    total_events: int = Field(..., description="Total events in period")
    user_activity: Dict[str, int] = Field(..., description="User activity summary")
    resource_access: Dict[str, int] = Field(..., description="Resource access summary")
    security_events: List[SecurityEvent] = Field(..., description="Security events")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Report generation time")
    generated_by: UUID = Field(..., description="Report generator user ID")

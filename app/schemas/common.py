"""
Enterprise State Machine Workflow Engine - Common Schemas

Common Pydantic schemas used across different modules.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class TimeStampedSchema(BaseModel):
    """Base schema with timestamp fields."""

    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class AuditSchema(BaseModel):
    """Base schema with audit fields."""

    created_by: Optional[UUID] = Field(None, description="ID of user who created the record")
    updated_by: Optional[UUID] = Field(None, description="ID of user who last updated the record")


class SoftDeleteSchema(BaseModel):
    """Base schema with soft delete fields."""

    is_deleted: bool = Field(False, description="Soft delete flag")
    deleted_at: Optional[datetime] = Field(None, description="Soft delete timestamp")


class BaseResponseSchema(TimeStampedSchema, AuditSchema, SoftDeleteSchema):
    """Base response schema with all common fields."""

    id: UUID = Field(..., description="Unique identifier")

    model_config = ConfigDict(from_attributes=True)


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=100, description="Number of items per page")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""

    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


class SortOrder(str, Enum):
    """Sort order enumeration."""
    ASC = "asc"
    DESC = "desc"


class SortParams(BaseModel):
    """Sorting parameters."""

    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.ASC, description="Sort order")


class FilterParams(BaseModel):
    """Base filtering parameters."""

    search: Optional[str] = Field(None, description="Search term")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
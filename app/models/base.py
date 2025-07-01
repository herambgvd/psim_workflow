"""
Enterprise State Machine Workflow Engine - Base Models

This module provides base model classes and common functionality
for all database models in the workflow engine.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, TypeVar, Generic
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Text,
    JSON,
    Integer,
    Enum as SQLEnum,
    Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import validates
from sqlalchemy.sql import func
import enum

from app.core.database import Base


ModelType = TypeVar("ModelType", bound="BaseModel")


class TimestampMixin:
    """
    Mixin class that provides created_at and updated_at timestamps.

    All models that inherit from this mixin will automatically
    have timestamp tracking for creation and updates.
    """

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="Timestamp when the record was created"
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True,
        doc="Timestamp when the record was last updated"
    )


class SoftDeleteMixin:
    """
    Mixin class that provides soft delete functionality.

    Records are not physically deleted but marked as deleted
    with a timestamp.
    """

    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Timestamp when the record was soft deleted"
    )

    is_deleted = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Flag indicating if the record is soft deleted"
    )

    def soft_delete(self) -> None:
        """Mark the record as soft deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()

    def restore(self) -> None:
        """Restore a soft deleted record."""
        self.is_deleted = False
        self.deleted_at = None


class AuditMixin:
    """
    Mixin class that provides audit trail functionality.

    Tracks who created and last modified the record.
    """

    created_by = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        doc="ID of the user who created the record"
    )

    updated_by = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        doc="ID of the user who last updated the record"
    )


class BaseModel(Base, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """
    Base model class for all database models.

    Provides common functionality including:
    - UUID primary key
    - Timestamp tracking
    - Soft delete capability
    - Audit trail
    - Common utility methods
    """

    __abstract__ = True

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        doc="Unique identifier for the record"
    )

    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def to_dict(self, exclude: Optional[set] = None) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.

        Args:
            exclude: Set of attribute names to exclude

        Returns:
            Dict[str, Any]: Dictionary representation of the model
        """
        exclude = exclude or set()
        result = {}

        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    value = str(value)
                elif isinstance(value, enum.Enum):
                    value = value.value
                result[column.name] = value

        return result

    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[set] = None) -> None:
        """
        Update model instance from dictionary.

        Args:
            data: Dictionary with field values
            exclude: Set of attribute names to exclude from update
        """
        exclude = exclude or {'id', 'created_at', 'created_by'}

        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"



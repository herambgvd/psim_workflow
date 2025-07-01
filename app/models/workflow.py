import enum
from typing import Any, Dict

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
from sqlalchemy.orm import validates
from sqlalchemy.sql import func

from app.models.base import BaseModel


# ===== WORKFLOW MODELS =====

class WorkflowStatus(enum.Enum):
    """Enumeration of workflow statuses."""
    DRAFT = "draft"  # Workflow is being designed
    ACTIVE = "active"  # Workflow is ready for use
    INACTIVE = "inactive"  # Workflow is temporarily disabled
    DEPRECATED = "deprecated"  # Workflow is outdated but kept for history
    ARCHIVED = "archived"  # Workflow is no longer in use


class WorkflowDefinition(BaseModel):
    """
    Model for workflow definitions.

    A workflow definition describes the structure, states, transitions,
    and business logic of a workflow that can be instantiated and executed.
    """

    __tablename__ = "workflow_definitions"

    # Basic workflow information
    name = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Human-readable name of the workflow"
    )

    description = Column(
        Text,
        nullable=True,
        doc="Detailed description of the workflow purpose and functionality"
    )

    version = Column(
        String(50),
        nullable=False,
        default="1.0.0",
        doc="Semantic version of the workflow definition"
    )

    status = Column(
        SQLEnum(WorkflowStatus),
        nullable=False,
        default=WorkflowStatus.DRAFT,
        index=True,
        doc="Current status of the workflow definition"
    )

    # Workflow configuration
    definition = Column(
        JSON,
        nullable=False,
        doc="JSON structure defining the workflow states, transitions, and logic"
    )

    input_schema = Column(
        JSON,
        nullable=True,
        doc="JSON schema defining expected input parameters for workflow instances"
    )

    output_schema = Column(
        JSON,
        nullable=True,
        doc="JSON schema defining expected output structure from workflow execution"
    )

    # Execution settings
    timeout_seconds = Column(
        Integer,
        nullable=True,
        doc="Maximum execution time for workflow instances in seconds"
    )

    max_retries = Column(
        Integer,
        nullable=False,
        default=3,
        doc="Maximum number of retry attempts for failed tasks"
    )

    retry_delay_seconds = Column(
        Integer,
        nullable=False,
        default=60,
        doc="Delay between retry attempts in seconds"
    )

    # Metadata and tags
    tags = Column(
        JSON,
        nullable=True,
        default=list,
        doc="List of tags for categorizing and searching workflows"
    )

    metadata = Column(
        JSON,
        nullable=True,
        default=dict,
        doc="Additional metadata for the workflow definition"
    )

    # Versioning and inheritance
    parent_workflow_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        doc="ID of the parent workflow this version is based on"
    )

    is_template = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="Flag indicating if this workflow can be used as a template"
    )

    # Usage statistics
    instance_count = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of workflow instances created from this definition"
    )

    success_count = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of successfully completed workflow instances"
    )

    failure_count = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of failed workflow instances"
    )

    # Indexes for performance
    __table_args__ = (
        Index('ix_workflow_name_version', 'name', 'version'),
        Index('ix_workflow_status_active', 'status', postgresql_where=status == WorkflowStatus.ACTIVE),
        Index('ix_workflow_tags', 'tags', postgresql_using='gin'),
    )

    @validates('name')
    def validate_name(self, key: str, name: str) -> str:
        """Validate workflow name."""
        if not name or len(name.strip()) == 0:
            raise ValueError("Workflow name cannot be empty")
        if len(name) > 255:
            raise ValueError("Workflow name cannot exceed 255 characters")
        return name.strip()

    @validates('version')
    def validate_version(self, key: str, version: str) -> str:
        """Validate semantic version format."""
        import re
        semver_pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'
        if not re.match(semver_pattern, version):
            raise ValueError("Version must follow semantic versioning format (e.g., 1.0.0)")
        return version

    @validates('definition')
    def validate_definition(self, key: str, definition: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow definition structure."""
        if not isinstance(definition, dict):
            raise ValueError("Workflow definition must be a dictionary")

        required_keys = ['states', 'transitions', 'initial_state']
        for required_key in required_keys:
            if required_key not in definition:
                raise ValueError(f"Workflow definition must include '{required_key}'")

        return definition

    def increment_instance_count(self) -> None:
        """Increment the instance count."""
        self.instance_count = (self.instance_count or 0) + 1

    def increment_success_count(self) -> None:
        """Increment the success count."""
        self.success_count = (self.success_count or 0) + 1

    def increment_failure_count(self) -> None:
        """Increment the failure count."""
        self.failure_count = (self.failure_count or 0) + 1

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        total = self.instance_count or 0
        if total == 0:
            return 0.0
        return (self.success_count or 0) / total * 100

    def get_state_names(self) -> list:
        """Get list of state names from the workflow definition."""
        return list(self.definition.get('states', {}).keys())

    def get_initial_state(self) -> str:
        """Get the initial state name."""
        return self.definition.get('initial_state', '')

    def is_valid_state(self, state_name: str) -> bool:
        """Check if a state name is valid for this workflow."""
        return state_name in self.get_state_names()

    def get_transitions_from_state(self, state_name: str) -> list:
        """Get all possible transitions from a given state."""
        transitions = self.definition.get('transitions', [])
        return [t for t in transitions if t.get('from_state') == state_name]

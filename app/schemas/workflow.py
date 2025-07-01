"""
Enterprise State Machine Workflow Engine - Workflow Schemas

Pydantic schemas for workflow definitions and related entities.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.common import BaseResponseSchema, PaginationParams, SortParams, FilterParams


class WorkflowStatusEnum(str, Enum):
    """Workflow status enumeration for API."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class StateDefinitionSchema(BaseModel):
    """Schema for state definition within workflow."""

    name: str = Field(..., min_length=1, max_length=255, description="State name")
    display_name: Optional[str] = Field(None, max_length=255, description="Human-readable display name")
    description: Optional[str] = Field(None, description="State description")
    state_type: str = Field("intermediate", description="Type of state")
    entry_actions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Entry actions")
    exit_actions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Exit actions")
    tasks: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Tasks to execute")
    timeout_seconds: Optional[int] = Field(None, gt=0, description="State timeout in seconds")
    conditions: Optional[Dict[str, Any]] = Field(None, description="State conditions")
    position: Optional[Dict[str, Any]] = Field(None, description="UI position")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate state name format."""
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError('State name must be a valid identifier')
        return v


class TransitionDefinitionSchema(BaseModel):
    """Schema for transition definition within workflow."""

    name: str = Field(..., min_length=1, max_length=255, description="Transition name")
    description: Optional[str] = Field(None, description="Transition description")
    from_state: str = Field(..., description="Source state name")
    to_state: str = Field(..., description="Target state name")
    transition_type: str = Field("automatic", description="Type of transition")
    conditions: Optional[List[Dict[str, Any]]] = Field(None, description="Transition conditions")
    trigger_event: Optional[str] = Field(None, description="Triggering event")
    actions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Transition actions")
    priority: int = Field(0, description="Transition priority")
    delay_seconds: Optional[int] = Field(None, ge=0, description="Transition delay")
    guard_expression: Optional[str] = Field(None, description="Guard expression")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class EventDefinitionSchema(BaseModel):
    """Schema for event definition within workflow."""

    name: str = Field(..., min_length=1, max_length=255, description="Event name")
    display_name: Optional[str] = Field(None, max_length=255, description="Human-readable display name")
    description: Optional[str] = Field(None, description="Event description")
    event_type: str = Field("system", description="Type of event")
    payload_schema: Optional[Dict[str, Any]] = Field(None, description="Event payload schema")
    handlers: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Event handlers")
    timeout_seconds: Optional[int] = Field(None, gt=0, description="Event timeout")
    is_repeatable: bool = Field(True, description="Whether event can repeat")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate event name format."""
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError('Event name must be a valid identifier')
        return v


class WorkflowDefinitionSchema(BaseModel):
    """Schema for complete workflow definition."""

    states: Dict[str, StateDefinitionSchema] = Field(..., description="Workflow states")
    transitions: List[TransitionDefinitionSchema] = Field(..., description="State transitions")
    events: Optional[List[EventDefinitionSchema]] = Field(default_factory=list, description="Workflow events")
    initial_state: str = Field(..., description="Initial state name")
    final_states: Optional[List[str]] = Field(default_factory=list, description="Final state names")
    variables: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Default variables")

    @model_validator(mode='after')
    def validate_workflow(self):
        """Validate workflow consistency."""
        states = set(self.states.keys())

        # Validate initial state exists
        if self.initial_state not in states:
            raise ValueError(f"Initial state '{self.initial_state}' not found in states")

        # Validate final states exist
        for final_state in self.final_states:
            if final_state not in states:
                raise ValueError(f"Final state '{final_state}' not found in states")

        # Validate transitions reference valid states
        for transition in self.transitions:
            if transition.from_state not in states:
                raise ValueError(
                    f"Transition '{transition.name}' references invalid from_state: {transition.from_state}")
            if transition.to_state not in states:
                raise ValueError(f"Transition '{transition.name}' references invalid to_state: {transition.to_state}")

        return self


class WorkflowCreateRequest(BaseModel):
    """Request schema for creating a workflow."""

    name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    version: str = Field("1.0.0", description="Workflow version")
    definition: WorkflowDefinitionSchema = Field(..., description="Workflow definition")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input validation schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output schema")
    timeout_seconds: Optional[int] = Field(None, gt=0, description="Workflow timeout")
    max_retries: int = Field(3, ge=0, description="Maximum retries")
    retry_delay_seconds: int = Field(60, ge=0, description="Retry delay")
    tags: Optional[List[str]] = Field(default_factory=list, description="Workflow tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    is_template: bool = Field(False, description="Whether this is a template")

    @field_validator('version')
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        import re
        semver_pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'
        if not re.match(semver_pattern, v):
            raise ValueError('Version must follow semantic versioning format (e.g., 1.0.0)')
        return v


class WorkflowUpdateRequest(BaseModel):
    """Request schema for updating a workflow."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    definition: Optional[WorkflowDefinitionSchema] = Field(None, description="Workflow definition")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input validation schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output schema")
    timeout_seconds: Optional[int] = Field(None, gt=0, description="Workflow timeout")
    max_retries: Optional[int] = Field(None, ge=0, description="Maximum retries")
    retry_delay_seconds: Optional[int] = Field(None, ge=0, description="Retry delay")
    tags: Optional[List[str]] = Field(None, description="Workflow tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    status: Optional[WorkflowStatusEnum] = Field(None, description="Workflow status")


class WorkflowResponse(BaseResponseSchema):
    """Response schema for workflow definition."""

    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    version: str = Field(..., description="Workflow version")
    status: WorkflowStatusEnum = Field(..., description="Workflow status")
    definition: WorkflowDefinitionSchema = Field(..., description="Workflow definition")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input validation schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output schema")
    timeout_seconds: Optional[int] = Field(None, description="Workflow timeout")
    max_retries: int = Field(..., description="Maximum retries")
    retry_delay_seconds: int = Field(..., description="Retry delay")
    tags: List[str] = Field(..., description="Workflow tags")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    parent_workflow_id: Optional[UUID] = Field(None, description="Parent workflow ID")
    is_template: bool = Field(..., description="Whether this is a template")
    instance_count: int = Field(..., description="Number of instances")
    success_count: int = Field(..., description="Number of successful instances")
    failure_count: int = Field(..., description="Number of failed instances")
    success_rate: float = Field(..., description="Success rate percentage")


class WorkflowSummaryResponse(BaseModel):
    """Summary response schema for workflow listing."""

    id: UUID = Field(..., description="Workflow ID")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    version: str = Field(..., description="Workflow version")
    status: WorkflowStatusEnum = Field(..., description="Workflow status")
    tags: List[str] = Field(..., description="Workflow tags")
    instance_count: int = Field(..., description="Number of instances")
    success_rate: float = Field(..., description="Success rate percentage")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


class WorkflowListParams(PaginationParams, SortParams, FilterParams):
    """Parameters for workflow listing."""

    status: Optional[WorkflowStatusEnum] = Field(None, description="Filter by status")
    is_template: Optional[bool] = Field(None, description="Filter by template flag")
    version: Optional[str] = Field(None, description="Filter by version")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")


class WorkflowValidationRequest(BaseModel):
    """Request schema for workflow validation."""

    definition: WorkflowDefinitionSchema = Field(..., description="Workflow definition to validate")
    strict: bool = Field(False, description="Enable strict validation")


class WorkflowValidationResponse(BaseModel):
    """Response schema for workflow validation."""

    is_valid: bool = Field(..., description="Whether the workflow is valid")
    errors: List[str] = Field(..., description="List of validation errors")
    warnings: List[str] = Field(..., description="List of validation warnings")
    suggestions: List[str] = Field(..., description="List of improvement suggestions")


class WorkflowStatsResponse(BaseModel):
    """Response schema for workflow statistics."""

    total_workflows: int = Field(..., description="Total number of workflows")
    active_workflows: int = Field(..., description="Number of active workflows")
    draft_workflows: int = Field(..., description="Number of draft workflows")
    total_instances: int = Field(..., description="Total workflow instances")
    running_instances: int = Field(..., description="Currently running instances")
    success_rate: float = Field(..., description="Overall success rate")
    avg_execution_time: Optional[float] = Field(None, description="Average execution time in seconds")


class WorkflowExportRequest(BaseModel):
    """Request schema for workflow export."""

    workflow_ids: List[UUID] = Field(..., description="List of workflow IDs to export")
    include_instances: bool = Field(False, description="Include instance data")
    format: str = Field("json", description="Export format")


class WorkflowImportRequest(BaseModel):
    """Request schema for workflow import."""

    workflows: List[WorkflowCreateRequest] = Field(..., description="List of workflows to import")
    overwrite_existing: bool = Field(False, description="Overwrite existing workflows")
    validate_only: bool = Field(False, description="Only validate, don't import")


class WorkflowImportResponse(BaseModel):
    """Response schema for workflow import."""

    imported_count: int = Field(..., description="Number of workflows imported")
    skipped_count: int = Field(..., description="Number of workflows skipped")
    error_count: int = Field(..., description="Number of workflows with errors")
    imported_workflows: List[UUID] = Field(..., description="IDs of imported workflows")
    errors: List[str] = Field(..., description="List of import errors")

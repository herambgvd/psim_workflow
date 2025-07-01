"""
Enterprise State Machine Workflow Engine - Instance Schemas

Pydantic schemas for workflow instances and execution management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

from app.schemas.common import BaseResponseSchema, PaginationParams, SortParams, FilterParams


class InstanceStatusEnum(str, Enum):
    """Instance status enumeration for API."""
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    TERMINATED = "terminated"


class PriorityEnum(str, Enum):
    """Priority enumeration for API."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionStatusEnum(str, Enum):
    """Execution status enumeration for API."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    WAITING = "waiting"


class HistoryEventTypeEnum(str, Enum):
    """History event type enumeration for API."""
    INSTANCE_CREATED = "instance_created"
    INSTANCE_STARTED = "instance_started"
    INSTANCE_PAUSED = "instance_paused"
    INSTANCE_RESUMED = "instance_resumed"
    INSTANCE_COMPLETED = "instance_completed"
    INSTANCE_FAILED = "instance_failed"
    INSTANCE_CANCELLED = "instance_cancelled"
    INSTANCE_TIMEOUT = "instance_timeout"
    STATE_ENTERED = "state_entered"
    STATE_EXITED = "state_exited"
    TRANSITION_EXECUTED = "transition_executed"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    EVENT_RECEIVED = "event_received"
    EVENT_PROCESSED = "event_processed"
    ERROR_OCCURRED = "error_occurred"
    RETRY_ATTEMPTED = "retry_attempted"
    VARIABLE_SET = "variable_set"
    CHECKPOINT_CREATED = "checkpoint_created"


class InstanceCreateRequest(BaseModel):
    """Request schema for creating a workflow instance."""

    workflow_definition_id: UUID = Field(..., description="ID of the workflow definition")
    name: Optional[str] = Field(None, max_length=255, description="Instance name")
    description: Optional[str] = Field(None, description="Instance description")
    input_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Input data")
    priority: PriorityEnum = Field(PriorityEnum.NORMAL, description="Execution priority")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled start time")
    timeout_at: Optional[datetime] = Field(None, description="Timeout time")
    max_retries: Optional[int] = Field(None, ge=0, description="Maximum retries")
    tags: Optional[List[str]] = Field(default_factory=list, description="Instance tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    external_id: Optional[str] = Field(None, max_length=255, description="External system ID")
    correlation_id: Optional[str] = Field(None, max_length=255, description="Correlation ID")
    context_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Runtime context")


class InstanceUpdateRequest(BaseModel):
    """Request schema for updating a workflow instance."""

    name: Optional[str] = Field(None, max_length=255, description="Instance name")
    description: Optional[str] = Field(None, description="Instance description")
    priority: Optional[PriorityEnum] = Field(None, description="Execution priority")
    tags: Optional[List[str]] = Field(None, description="Instance tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ExecutionContextResponse(BaseModel):
    """Response schema for execution context."""

    id: UUID = Field(..., description="Context ID")
    workflow_instance_id: UUID = Field(..., description="Workflow instance ID")
    current_state: str = Field(..., description="Current state")
    previous_state: Optional[str] = Field(None, description="Previous state")
    status: ExecutionStatusEnum = Field(..., description="Execution status")
    variables: Dict[str, Any] = Field(..., description="Runtime variables")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    last_activity_at: Optional[datetime] = Field(None, description="Last activity timestamp")
    error_message: Optional[str] = Field(None, description="Error message")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    retry_count: int = Field(..., description="Retry count")
    transition_count: int = Field(..., description="Transition count")
    task_count: int = Field(..., description="Task count")
    duration_seconds: Optional[int] = Field(None, description="Execution duration")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")

    model_config = ConfigDict(from_attributes=True)


class InstanceResponse(BaseResponseSchema):
    """Response schema for workflow instance."""

    workflow_definition_id: UUID = Field(..., description="Workflow definition ID")
    name: Optional[str] = Field(None, description="Instance name")
    description: Optional[str] = Field(None, description="Instance description")
    status: InstanceStatusEnum = Field(..., description="Instance status")
    priority: PriorityEnum = Field(..., description="Execution priority")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled start time")
    started_at: Optional[datetime] = Field(None, description="Actual start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    last_activity_at: Optional[datetime] = Field(None, description="Last activity time")
    timeout_at: Optional[datetime] = Field(None, description="Timeout time")
    max_retries: int = Field(..., description="Maximum retries")
    retry_count: int = Field(..., description="Current retry count")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
    context_data: Dict[str, Any] = Field(..., description="Runtime context")
    error_message: Optional[str] = Field(None, description="Error message")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    tags: List[str] = Field(..., description="Instance tags")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    parent_instance_id: Optional[UUID] = Field(None, description="Parent instance ID")
    external_id: Optional[str] = Field(None, description="External system ID")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    step_count: int = Field(..., description="Number of steps executed")
    duration_seconds: Optional[int] = Field(None, description="Execution duration")

    # Related data
    execution_context: Optional[ExecutionContextResponse] = Field(None, description="Execution context")
    workflow_definition: Optional[Dict[str, Any]] = Field(None, description="Workflow definition summary")


class InstanceSummaryResponse(BaseModel):
    """Summary response schema for instance listing."""

    id: UUID = Field(..., description="Instance ID")
    workflow_definition_id: UUID = Field(..., description="Workflow definition ID")
    name: Optional[str] = Field(None, description="Instance name")
    status: InstanceStatusEnum = Field(..., description="Instance status")
    priority: PriorityEnum = Field(..., description="Execution priority")
    current_state: Optional[str] = Field(None, description="Current state")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    duration_seconds: Optional[int] = Field(None, description="Execution duration")
    retry_count: int = Field(..., description="Retry count")
    tags: List[str] = Field(..., description="Instance tags")
    created_at: datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)


class InstanceListParams(PaginationParams, SortParams, FilterParams):
    """Parameters for instance listing."""

    workflow_definition_id: Optional[UUID] = Field(None, description="Filter by workflow definition")
    status: Optional[InstanceStatusEnum] = Field(None, description="Filter by status")
    priority: Optional[PriorityEnum] = Field(None, description="Filter by priority")
    started_after: Optional[datetime] = Field(None, description="Filter by start date")
    started_before: Optional[datetime] = Field(None, description="Filter by start date")
    completed_after: Optional[datetime] = Field(None, description="Filter by completion date")
    completed_before: Optional[datetime] = Field(None, description="Filter by completion date")
    external_id: Optional[str] = Field(None, description="Filter by external ID")
    correlation_id: Optional[str] = Field(None, description="Filter by correlation ID")


class InstanceControlRequest(BaseModel):
    """Request schema for instance control operations."""

    action: str = Field(..., description="Control action")
    reason: Optional[str] = Field(None, description="Reason for the action")
    force: bool = Field(False, description="Force the action")


class InstanceEventRequest(BaseModel):
    """Request schema for sending events to instances."""

    event_name: str = Field(..., description="Name of the event")
    event_type: str = Field("external", description="Type of event")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    source: str = Field("api", description="Event source")


class ExecutionHistoryResponse(BaseModel):
    """Response schema for execution history entry."""

    id: UUID = Field(..., description="History entry ID")
    workflow_instance_id: UUID = Field(..., description="Workflow instance ID")
    event_type: HistoryEventTypeEnum = Field(..., description="Event type")
    event_name: Optional[str] = Field(None, description="Event name")
    description: Optional[str] = Field(None, description="Event description")
    from_state: Optional[str] = Field(None, description="Source state")
    to_state: Optional[str] = Field(None, description="Target state")
    event_data: Optional[Dict[str, Any]] = Field(None, description="Event data")
    event_timestamp: datetime = Field(..., description="Event timestamp")
    duration_ms: Optional[int] = Field(None, description="Event duration")
    actor_type: Optional[str] = Field(None, description="Actor type")
    actor_id: Optional[str] = Field(None, description="Actor ID")
    execution_context: Optional[Dict[str, Any]] = Field(None, description="Execution context")
    error_message: Optional[str] = Field(None, description="Error message")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    sequence_number: int = Field(..., description="Sequence number")

    model_config = ConfigDict(from_attributes=True)


class InstanceStatsResponse(BaseModel):
    """Response schema for instance statistics."""

    total_instances: int = Field(..., description="Total instances")
    running_instances: int = Field(..., description="Running instances")
    completed_instances: int = Field(..., description="Completed instances")
    failed_instances: int = Field(..., description="Failed instances")
    success_rate: float = Field(..., description="Success rate percentage")
    avg_execution_time: Optional[float] = Field(None, description="Average execution time")
    avg_retry_count: float = Field(..., description="Average retry count")
    status_distribution: Dict[str, int] = Field(..., description="Status distribution")
    priority_distribution: Dict[str, int] = Field(..., description="Priority distribution")


class InstanceVariableRequest(BaseModel):
    """Request schema for setting instance variables."""

    variables: Dict[str, Any] = Field(..., description="Variables to set")


class InstanceVariableResponse(BaseModel):
    """Response schema for instance variables."""

    variables: Dict[str, Any] = Field(..., description="Current variables")
    updated_at: datetime = Field(..., description="Last update timestamp")


class BulkInstanceRequest(BaseModel):
    """Request schema for bulk instance operations."""

    instance_ids: List[UUID] = Field(..., description="List of instance IDs")
    action: str = Field(..., description="Bulk action to perform")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Action parameters")


class BulkInstanceResponse(BaseModel):
    """Response schema for bulk instance operations."""

    total_count: int = Field(..., description="Total instances processed")
    success_count: int = Field(..., description="Successfully processed instances")
    error_count: int = Field(..., description="Failed instances")
    results: List[Dict[str, Any]] = Field(..., description="Individual results")
    errors: List[str] = Field(..., description="Error messages")


class InstanceMetricsResponse(BaseModel):
    """Response schema for instance metrics."""

    instance_id: UUID = Field(..., description="Instance ID")
    execution_time: Optional[float] = Field(None, description="Total execution time")
    state_durations: Dict[str, float] = Field(..., description="Time spent in each state")
    transition_count: int = Field(..., description="Number of transitions")
    task_count: int = Field(..., description="Number of tasks executed")
    retry_count: int = Field(..., description="Number of retries")
    error_count: int = Field(..., description="Number of errors")
    checkpoint_count: int = Field(..., description="Number of checkpoints")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage metrics")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")


class InstanceRetryRequest(BaseModel):
    """Request schema for retrying failed instances."""

    reset_state: bool = Field(False, description="Reset to initial state")
    clear_errors: bool = Field(True, description="Clear previous errors")
    new_input_data: Optional[Dict[str, Any]] = Field(None, description="New input data")
    reason: Optional[str] = Field(None, description="Retry reason")
"""
Enterprise State Machine Workflow Engine - State Machine Models

This module defines the data models for state machine components including
states, transitions, events, and execution context.
"""

import enum
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    JSON,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Enum as SQLEnum,
    Index,
    UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from app.models.base import BaseModel


class StateType(enum.Enum):
    """Enumeration of state types in the workflow."""
    INITIAL = "initial"  # Starting state of the workflow
    INTERMEDIATE = "intermediate"  # Regular processing state
    FINAL = "final"  # End state (success)
    ERROR = "error"  # Error/failure state
    WAIT = "wait"  # Waiting for external input/event
    PARALLEL = "parallel"  # Parallel execution state
    CHOICE = "choice"  # Conditional branching state


class TransitionType(enum.Enum):
    """Enumeration of transition types."""
    AUTOMATIC = "automatic"  # Automatic transition based on completion
    CONDITIONAL = "conditional"  # Transition based on conditions
    MANUAL = "manual"  # Manual user-triggered transition
    EVENT = "event"  # Event-driven transition
    TIMEOUT = "timeout"  # Time-based transition


class EventType(enum.Enum):
    """Enumeration of event types."""
    USER_ACTION = "user_action"  # User-initiated event
    SYSTEM = "system"  # System-generated event
    EXTERNAL = "external"  # External system event
    TIMER = "timer"  # Time-based event
    ERROR = "error"  # Error event
    WEBHOOK = "webhook"  # Webhook-triggered event


class ExecutionStatus(enum.Enum):
    """Enumeration of execution statuses."""
    PENDING = "pending"  # Waiting to start
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed with error
    CANCELLED = "cancelled"  # Cancelled by user
    TIMEOUT = "timeout"  # Timed out
    WAITING = "waiting"  # Waiting for external input


class StateDefinition(BaseModel):
    """
    Model for state definitions within a workflow.

    Represents individual states in a state machine with their
    configuration, actions, and metadata.
    """

    __tablename__ = "state_definitions"

    # Reference to workflow
    workflow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflow_definitions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the workflow this state belongs to"
    )

    # State identification
    name = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Unique name of the state within the workflow"
    )

    display_name = Column(
        String(255),
        nullable=True,
        doc="Human-readable display name for the state"
    )

    description = Column(
        Text,
        nullable=True,
        doc="Detailed description of the state purpose and behavior"
    )

    # State configuration
    state_type = Column(
        SQLEnum(StateType),
        nullable=False,
        default=StateType.INTERMEDIATE,
        index=True,
        doc="Type of state (initial, intermediate, final, etc.)"
    )

    # Actions and tasks
    entry_actions = Column(
        JSON,
        nullable=True,
        default=list,
        doc="List of actions to execute when entering this state"
    )

    exit_actions = Column(
        JSON,
        nullable=True,
        default=list,
        doc="List of actions to execute when exiting this state"
    )

    tasks = Column(
        JSON,
        nullable=True,
        default=list,
        doc="List of tasks to execute while in this state"
    )

    # Timing and constraints
    timeout_seconds = Column(
        Integer,
        nullable=True,
        doc="Maximum time allowed in this state before timeout"
    )

    retry_config = Column(
        JSON,
        nullable=True,
        doc="Retry configuration for failed tasks in this state"
    )

    # Conditional logic
    conditions = Column(
        JSON,
        nullable=True,
        doc="Conditions that must be met to enter or remain in this state"
    )

    # UI and visualization
    position = Column(
        JSON,
        nullable=True,
        doc="Position coordinates for visual workflow designer"
    )

    style = Column(
        JSON,
        nullable=True,
        doc="Visual styling information for the state"
    )

    # Metadata
    metadata = Column(
        JSON,
        nullable=True,
        default=dict,
        doc="Additional metadata for the state"
    )

    # Relationships
    workflow = relationship(
        "WorkflowDefinition",
        back_populates="states"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint('workflow_id', 'name', name='uq_state_workflow_name'),
        Index('ix_state_type_workflow', 'state_type', 'workflow_id'),
    )

    @validates('name')
    def validate_name(self, key: str, name: str) -> str:
        """Validate state name."""
        if not name or len(name.strip()) == 0:
            raise ValueError("State name cannot be empty")

        # Check for valid identifier
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError("State name must be a valid identifier")

        return name.strip()

    def is_initial_state(self) -> bool:
        """Check if this is an initial state."""
        return self.state_type == StateType.INITIAL

    def is_final_state(self) -> bool:
        """Check if this is a final state."""
        return self.state_type in (StateType.FINAL, StateType.ERROR)

    def has_timeout(self) -> bool:
        """Check if this state has a timeout configured."""
        return self.timeout_seconds is not None and self.timeout_seconds > 0


class TransitionDefinition(BaseModel):
    """
    Model for transition definitions between states.

    Defines how and when the workflow can move from one state to another.
    """

    __tablename__ = "transition_definitions"

    # Reference to workflow
    workflow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflow_definitions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the workflow this transition belongs to"
    )

    # Transition identification
    name = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Unique name of the transition within the workflow"
    )

    description = Column(
        Text,
        nullable=True,
        doc="Description of the transition purpose and trigger conditions"
    )

    # State references
    from_state = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Name of the source state"
    )

    to_state = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Name of the target state"
    )

    # Transition configuration
    transition_type = Column(
        SQLEnum(TransitionType),
        nullable=False,
        default=TransitionType.AUTOMATIC,
        index=True,
        doc="Type of transition trigger"
    )

    # Conditions and triggers
    conditions = Column(
        JSON,
        nullable=True,
        doc="Conditions that must be met for the transition to fire"
    )

    trigger_event = Column(
        String(255),
        nullable=True,
        index=True,
        doc="Event that triggers this transition"
    )

    # Actions and side effects
    actions = Column(
        JSON,
        nullable=True,
        default=list,
        doc="Actions to execute during the transition"
    )

    # Priority and ordering
    priority = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Priority for transition evaluation (higher values = higher priority)"
    )

    # Timing
    delay_seconds = Column(
        Integer,
        nullable=True,
        doc="Delay before executing the transition"
    )

    # Guards and validation
    guard_expression = Column(
        Text,
        nullable=True,
        doc="Expression that must evaluate to true for transition to be allowed"
    )

    # Metadata
    metadata = Column(
        JSON,
        nullable=True,
        default=dict,
        doc="Additional metadata for the transition"
    )

    # Relationships
    workflow = relationship(
        "WorkflowDefinition",
        back_populates="transitions"
    )

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('workflow_id', 'name', name='uq_transition_workflow_name'),
        Index('ix_transition_from_state', 'workflow_id', 'from_state'),
        Index('ix_transition_to_state', 'workflow_id', 'to_state'),
        Index('ix_transition_trigger', 'trigger_event'),
    )

    @validates('from_state', 'to_state')
    def validate_states(self, key: str, state_name: str) -> str:
        """Validate state names."""
        if not state_name or len(state_name.strip()) == 0:
            raise ValueError(f"{key} cannot be empty")
        return state_name.strip()

    def is_conditional(self) -> bool:
        """Check if this is a conditional transition."""
        return self.transition_type == TransitionType.CONDITIONAL

    def has_delay(self) -> bool:
        """Check if this transition has a delay."""
        return self.delay_seconds is not None and self.delay_seconds > 0


class EventDefinition(BaseModel):
    """
    Model for event definitions in workflows.

    Events can trigger transitions or actions within the workflow execution.
    """

    __tablename__ = "event_definitions"

    # Reference to workflow
    workflow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflow_definitions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the workflow this event belongs to"
    )

    # Event identification
    name = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Unique name of the event within the workflow"
    )

    display_name = Column(
        String(255),
        nullable=True,
        doc="Human-readable display name for the event"
    )

    description = Column(
        Text,
        nullable=True,
        doc="Description of the event purpose and when it occurs"
    )

    # Event configuration
    event_type = Column(
        SQLEnum(EventType),
        nullable=False,
        default=EventType.SYSTEM,
        index=True,
        doc="Type of event (user_action, system, external, etc.)"
    )

    # Event schema and validation
    payload_schema = Column(
        JSON,
        nullable=True,
        doc="JSON schema defining the expected event payload structure"
    )

    # Event handling
    handlers = Column(
        JSON,
        nullable=True,
        default=list,
        doc="List of handlers to execute when this event occurs"
    )

    # Timing and lifecycle
    timeout_seconds = Column(
        Integer,
        nullable=True,
        doc="Maximum time to wait for this event before timeout"
    )

    is_repeatable = Column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether this event can occur multiple times in a workflow instance"
    )

    # Metadata
    metadata = Column(
        JSON,
        nullable=True,
        default=dict,
        doc="Additional metadata for the event"
    )

    # Relationships
    workflow = relationship(
        "WorkflowDefinition",
        back_populates="events"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint('workflow_id', 'name', name='uq_event_workflow_name'),
        Index('ix_event_type_workflow', 'event_type', 'workflow_id'),
    )

    @validates('name')
    def validate_name(self, key: str, name: str) -> str:
        """Validate event name."""
        if not name or len(name.strip()) == 0:
            raise ValueError("Event name cannot be empty")

        # Check for valid identifier
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', name):
            raise ValueError("Event name must be a valid identifier")

        return name.strip()

    def is_user_event(self) -> bool:
        """Check if this is a user-triggered event."""
        return self.event_type == EventType.USER_ACTION

    def is_external_event(self) -> bool:
        """Check if this is an external event."""
        return self.event_type == EventType.EXTERNAL

    def has_timeout(self) -> bool:
        """Check if this event has a timeout."""
        return self.timeout_seconds is not None and self.timeout_seconds > 0


class ExecutionContext(BaseModel):
    """
    Model for storing execution context and state during workflow execution.

    This stores the runtime state, variables, and execution history
    for workflow instances.
    """

    __tablename__ = "execution_contexts"

    # Reference to workflow instance
    workflow_instance_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflow_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the workflow instance this context belongs to"
    )

    # Current execution state
    current_state = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Current state of the workflow execution"
    )

    previous_state = Column(
        String(255),
        nullable=True,
        index=True,
        doc="Previous state before the current one"
    )

    # Execution status
    status = Column(
        SQLEnum(ExecutionStatus),
        nullable=False,
        default=ExecutionStatus.PENDING,
        index=True,
        doc="Current execution status"
    )

    # Runtime data
    variables = Column(
        JSON,
        nullable=True,
        default=dict,
        doc="Runtime variables and their current values"
    )

    input_data = Column(
        JSON,
        nullable=True,
        doc="Input data provided when the workflow was started"
    )

    output_data = Column(
        JSON,
        nullable=True,
        doc="Output data generated by the workflow execution"
    )

    # Execution tracking
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Timestamp when execution started"
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Timestamp when execution completed"
    )

    last_activity_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Timestamp of last execution activity"
    )

    # Error handling
    error_message = Column(
        Text,
        nullable=True,
        doc="Error message if execution failed"
    )

    error_details = Column(
        JSON,
        nullable=True,
        doc="Detailed error information including stack traces"
    )

    retry_count = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of retry attempts for the current state"
    )

    # Execution metrics
    transition_count = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Total number of state transitions"
    )

    task_count = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Total number of tasks executed"
    )

    duration_seconds = Column(
        Integer,
        nullable=True,
        doc="Total execution duration in seconds"
    )

    # Checkpointing and recovery
    checkpoint_data = Column(
        JSON,
        nullable=True,
        doc="Checkpoint data for execution recovery"
    )

    last_checkpoint_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp of last checkpoint"
    )

    # Relationships
    workflow_instance = relationship(
        "WorkflowInstance",
        back_populates="execution_context"
    )

    # Indexes for performance
    __table_args__ = (
        Index('ix_context_status_state', 'status', 'current_state'),
        Index('ix_context_activity', 'last_activity_at'),
        Index('ix_context_duration', 'started_at', 'completed_at'),
    )

    def start_execution(self, initial_state: str, input_data: Optional[Dict] = None) -> None:
        """Start workflow execution."""
        self.current_state = initial_state
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()
        if input_data:
            self.input_data = input_data
        self.variables = self.variables or {}

    def transition_to_state(self, new_state: str) -> None:
        """Transition to a new state."""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.transition_count += 1
        self.last_activity_at = datetime.utcnow()

    def complete_execution(self, output_data: Optional[Dict] = None) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()
        if output_data:
            self.output_data = output_data
        self._calculate_duration()

    def fail_execution(self, error_message: str, error_details: Optional[Dict] = None) -> None:
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.error_message = error_message
        self.error_details = error_details
        self.last_activity_at = datetime.utcnow()
        if not self.completed_at:
            self.completed_at = datetime.utcnow()
        self._calculate_duration()

    def cancel_execution(self) -> None:
        """Cancel workflow execution."""
        self.status = ExecutionStatus.CANCELLED
        self.last_activity_at = datetime.utcnow()
        if not self.completed_at:
            self.completed_at = datetime.utcnow()
        self._calculate_duration()

    def increment_retry_count(self) -> None:
        """Increment the retry count."""
        self.retry_count += 1
        self.last_activity_at = datetime.utcnow()

    def increment_task_count(self) -> None:
        """Increment the task count."""
        self.task_count += 1
        self.last_activity_at = datetime.utcnow()

    def create_checkpoint(self, checkpoint_data: Optional[Dict] = None) -> None:
        """Create a checkpoint for recovery."""
        self.checkpoint_data = checkpoint_data or {
            'current_state': self.current_state,
            'variables': self.variables,
            'transition_count': self.transition_count,
            'task_count': self.task_count
        }
        self.last_checkpoint_at = datetime.utcnow()

    def set_variable(self, key: str, value: Any) -> None:
        """Set a runtime variable."""
        if self.variables is None:
            self.variables = {}
        self.variables[key] = value
        self.last_activity_at = datetime.utcnow()

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a runtime variable value."""
        if self.variables is None:
            return default
        return self.variables.get(key, default)

    def _calculate_duration(self) -> None:
        """Calculate and set execution duration."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.duration_seconds = int(delta.total_seconds())

    @property
    def is_running(self) -> bool:
        """Check if execution is currently running."""
        return self.status == ExecutionStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED)

    @property
    def execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.started_at:
            end_time = self.completed_at or datetime.utcnow()
            return (end_time - self.started_at).total_seconds()
        return None


# Add relationships to WorkflowDefinition
# This extends the workflow model to include relationships with state machine components
def add_workflow_relationships():
    """Add relationships to WorkflowDefinition model."""
    from app.models.workflow import WorkflowDefinition

    # States relationship
    WorkflowDefinition.states = relationship(
        "StateDefinition",
        back_populates="workflow",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    # Transitions relationship
    WorkflowDefinition.transitions = relationship(
        "TransitionDefinition",
        back_populates="workflow",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    # Events relationship
    WorkflowDefinition.events = relationship(
        "EventDefinition",
        back_populates="workflow",
        cascade="all, delete-orphan",
        lazy="selectin"
    )


# Call this function to establish relationships
add_workflow_relationships()

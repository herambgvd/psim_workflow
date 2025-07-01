"""
Enterprise State Machine Workflow Engine - Core State Machine Engine

This module implements the core state machine engine that handles
workflow execution, state transitions, and event processing.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from app.core.logging import get_logger
from app.models.instance import WorkflowInstance
from app.models.state_machine import (
    TransitionDefinition,
    ExecutionContext,
    ExecutionStatus,
    TransitionType
)
from app.models.workflow import WorkflowDefinition

logger = get_logger(__name__)


class EngineError(Exception):
    """Base exception for state machine engine errors."""
    pass


class InvalidTransitionError(EngineError):
    """Raised when an invalid state transition is attempted."""
    pass


class StateNotFoundError(EngineError):
    """Raised when a referenced state is not found."""
    pass


class TransitionEvaluationError(EngineError):
    """Raised when transition condition evaluation fails."""
    pass


@dataclass
class TransitionResult:
    """Result of a state transition attempt."""
    success: bool
    from_state: str
    to_state: Optional[str]
    transition_name: Optional[str] = None
    error_message: Optional[str] = None
    executed_actions: List[str] = None

    def __post_init__(self):
        if self.executed_actions is None:
            self.executed_actions = []


@dataclass
class ExecutionEvent:
    """Represents an event that can trigger state transitions."""
    name: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    source: str = "system"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


class StateMachineEngine:
    """
    Core state machine engine for workflow execution.

    This engine handles the execution of workflow instances based on
    workflow definitions, managing state transitions, event processing,
    and execution context.
    """

    def __init__(self):
        """Initialize the state machine engine."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._running_instances: Dict[str, WorkflowInstance] = {}
        self._execution_contexts: Dict[str, ExecutionContext] = {}

    async def start_workflow_instance(
            self,
            workflow_definition: WorkflowDefinition,
            instance_id: str,
            input_data: Optional[Dict[str, Any]] = None
    ) -> ExecutionContext:
        """
        Start execution of a workflow instance.

        Args:
            workflow_definition: The workflow definition to execute
            instance_id: Unique identifier for the workflow instance
            input_data: Input data for the workflow execution

        Returns:
            ExecutionContext: The execution context for the started instance

        Raises:
            EngineError: If workflow cannot be started
        """
        self.logger.info(
            "Starting workflow instance",
            workflow_id=workflow_definition.id,
            instance_id=instance_id,
            workflow_name=workflow_definition.name
        )

        try:
            # Validate workflow definition
            self._validate_workflow_definition(workflow_definition)

            # Get initial state
            initial_state = workflow_definition.get_initial_state()
            if not initial_state:
                raise EngineError("Workflow definition has no initial state")

            # Create execution context
            execution_context = ExecutionContext(
                workflow_instance_id=instance_id,
                current_state=initial_state,
                status=ExecutionStatus.RUNNING,
                input_data=input_data or {},
                variables={}
            )

            # Initialize execution context
            execution_context.start_execution(initial_state, input_data)

            # Store execution context
            self._execution_contexts[instance_id] = execution_context

            # Execute entry actions for initial state
            await self._execute_state_entry_actions(
                workflow_definition,
                initial_state,
                execution_context
            )

            self.logger.info(
                "Workflow instance started successfully",
                instance_id=instance_id,
                initial_state=initial_state
            )

            return execution_context

        except Exception as e:
            self.logger.error(
                "Failed to start workflow instance",
                instance_id=instance_id,
                error=str(e),
                exc_info=True
            )
            raise EngineError(f"Failed to start workflow instance: {str(e)}")

    async def process_event(
            self,
            instance_id: str,
            event: ExecutionEvent,
            workflow_definition: WorkflowDefinition
    ) -> List[TransitionResult]:
        """
        Process an event for a workflow instance.

        Args:
            instance_id: Workflow instance identifier
            event: The event to process
            workflow_definition: The workflow definition

        Returns:
            List[TransitionResult]: Results of any triggered transitions
        """
        self.logger.info(
            "Processing event",
            instance_id=instance_id,
            event_name=event.name,
            event_type=event.event_type
        )

        execution_context = self._execution_contexts.get(instance_id)
        if not execution_context:
            raise EngineError(f"No execution context found for instance {instance_id}")

        if not execution_context.is_running:
            self.logger.warning(
                "Cannot process event for non-running instance",
                instance_id=instance_id,
                status=execution_context.status
            )
            return []

        # Find applicable transitions for current state
        applicable_transitions = self._find_applicable_transitions(
            workflow_definition,
            execution_context.current_state,
            event
        )

        results = []

        # Process transitions in priority order
        for transition in sorted(applicable_transitions, key=lambda t: t.priority, reverse=True):
            try:
                result = await self._attempt_transition(
                    workflow_definition,
                    execution_context,
                    transition,
                    event
                )

                results.append(result)

                # If transition was successful, break (only one transition per event)
                if result.success:
                    break

            except Exception as e:
                self.logger.error(
                    "Error during transition attempt",
                    instance_id=instance_id,
                    transition_name=transition.name,
                    error=str(e)
                )

                results.append(TransitionResult(
                    success=False,
                    from_state=execution_context.current_state,
                    to_state=None,
                    transition_name=transition.name,
                    error_message=str(e)
                ))

        return results

    async def execute_automatic_transitions(
            self,
            instance_id: str,
            workflow_definition: WorkflowDefinition
    ) -> List[TransitionResult]:
        """
        Execute automatic transitions for a workflow instance.

        Args:
            instance_id: Workflow instance identifier
            workflow_definition: The workflow definition

        Returns:
            List[TransitionResult]: Results of executed transitions
        """
        execution_context = self._execution_contexts.get(instance_id)
        if not execution_context or not execution_context.is_running:
            return []

        results = []
        max_transitions = 100  # Prevent infinite loops
        transition_count = 0

        while transition_count < max_transitions:
            # Find automatic transitions from current state
            automatic_transitions = [
                t for t in workflow_definition.get_transitions_from_state(execution_context.current_state)
                if t.get('transition_type') == TransitionType.AUTOMATIC.value
            ]

            if not automatic_transitions:
                break

            transition_executed = False

            for transition_data in automatic_transitions:
                # Create transition object (simplified)
                transition = TransitionDefinition(
                    workflow_id=workflow_definition.id,
                    name=transition_data['name'],
                    from_state=transition_data['from_state'],
                    to_state=transition_data['to_state'],
                    transition_type=TransitionType.AUTOMATIC,
                    conditions=transition_data.get('conditions'),
                    priority=transition_data.get('priority', 0)
                )

                try:
                    result = await self._attempt_transition(
                        workflow_definition,
                        execution_context,
                        transition
                    )

                    results.append(result)

                    if result.success:
                        transition_executed = True
                        transition_count += 1
                        break

                except Exception as e:
                    self.logger.error(
                        "Error during automatic transition",
                        instance_id=instance_id,
                        transition_name=transition.name,
                        error=str(e)
                    )

            if not transition_executed:
                break

        if transition_count >= max_transitions:
            self.logger.warning(
                "Maximum transition limit reached",
                instance_id=instance_id,
                max_transitions=max_transitions
            )

        return results

    def _validate_workflow_definition(self, workflow_definition: WorkflowDefinition) -> None:
        """
        Validate a workflow definition.

        Args:
            workflow_definition: The workflow definition to validate

        Raises:
            EngineError: If validation fails
        """
        definition = workflow_definition.definition

        # Check required fields
        required_fields = ['states', 'transitions', 'initial_state']
        for field in required_fields:
            if field not in definition:
                raise EngineError(f"Workflow definition missing required field: {field}")

        # Validate initial state exists
        initial_state = definition['initial_state']
        if initial_state not in definition['states']:
            raise EngineError(f"Initial state '{initial_state}' not found in states")

        # Validate all transitions reference valid states
        states = set(definition['states'].keys())
        for transition in definition.get('transitions', []):
            from_state = transition.get('from_state')
            to_state = transition.get('to_state')

            if from_state not in states:
                raise EngineError(f"Transition references invalid from_state: {from_state}")

            if to_state not in states:
                raise EngineError(f"Transition references invalid to_state: {to_state}")

    def _find_applicable_transitions(
            self,
            workflow_definition: WorkflowDefinition,
            current_state: str,
            event: Optional[ExecutionEvent] = None
    ) -> List[TransitionDefinition]:
        """
        Find transitions applicable for current state and event.

        Args:
            workflow_definition: The workflow definition
            current_state: Current state name
            event: Optional event that might trigger transitions

        Returns:
            List[TransitionDefinition]: Applicable transitions
        """
        applicable = []

        transitions = workflow_definition.get_transitions_from_state(current_state)

        for transition_data in transitions:
            # Check if transition is triggered by the event
            if event and transition_data.get('trigger_event'):
                if transition_data['trigger_event'] != event.name:
                    continue

            # Create transition object
            transition = TransitionDefinition(
                workflow_id=workflow_definition.id,
                name=transition_data['name'],
                from_state=transition_data['from_state'],
                to_state=transition_data['to_state'],
                transition_type=TransitionType(transition_data.get('transition_type', 'automatic')),
                conditions=transition_data.get('conditions'),
                trigger_event=transition_data.get('trigger_event'),
                priority=transition_data.get('priority', 0)
            )

            applicable.append(transition)

        return applicable

    async def _attempt_transition(
            self,
            workflow_definition: WorkflowDefinition,
            execution_context: ExecutionContext,
            transition: TransitionDefinition,
            event: Optional[ExecutionEvent] = None
    ) -> TransitionResult:
        """
        Attempt to execute a state transition.

        Args:
            workflow_definition: The workflow definition
            execution_context: Current execution context
            transition: The transition to attempt
            event: Optional triggering event

        Returns:
            TransitionResult: Result of the transition attempt
        """
        self.logger.debug(
            "Attempting transition",
            from_state=transition.from_state,
            to_state=transition.to_state,
            transition_name=transition.name
        )

        # Evaluate transition conditions
        if not await self._evaluate_transition_conditions(transition, execution_context, event):
            return TransitionResult(
                success=False,
                from_state=transition.from_state,
                to_state=transition.to_state,
                transition_name=transition.name,
                error_message="Transition conditions not met"
            )

        try:
            # Execute exit actions for current state
            await self._execute_state_exit_actions(
                workflow_definition,
                execution_context.current_state,
                execution_context
            )

            # Execute transition actions
            executed_actions = await self._execute_transition_actions(
                transition,
                execution_context,
                event
            )

            # Update execution context
            execution_context.transition_to_state(transition.to_state)

            # Execute entry actions for new state
            await self._execute_state_entry_actions(
                workflow_definition,
                transition.to_state,
                execution_context
            )

            self.logger.info(
                "Transition executed successfully",
                from_state=transition.from_state,
                to_state=transition.to_state,
                transition_name=transition.name
            )

            return TransitionResult(
                success=True,
                from_state=transition.from_state,
                to_state=transition.to_state,
                transition_name=transition.name,
                executed_actions=executed_actions
            )

        except Exception as e:
            self.logger.error(
                "Transition execution failed",
                transition_name=transition.name,
                error=str(e),
                exc_info=True
            )

            return TransitionResult(
                success=False,
                from_state=transition.from_state,
                to_state=transition.to_state,
                transition_name=transition.name,
                error_message=str(e)
            )

    async def _evaluate_transition_conditions(
            self,
            transition: TransitionDefinition,
            execution_context: ExecutionContext,
            event: Optional[ExecutionEvent] = None
    ) -> bool:
        """
        Evaluate whether transition conditions are met.

        Args:
            transition: The transition to evaluate
            execution_context: Current execution context
            event: Optional triggering event

        Returns:
            bool: True if conditions are met
        """
        if not transition.conditions:
            return True

        try:
            # Simple condition evaluation (can be extended with a proper expression engine)
            conditions = transition.conditions
            variables = execution_context.variables or {}

            # Add event data to evaluation context if available
            if event:
                variables['event'] = {
                    'name': event.name,
                    'type': event.event_type,
                    'payload': event.payload
                }

            # Basic condition evaluation
            for condition in conditions:
                condition_type = condition.get('type', 'expression')

                if condition_type == 'variable_equals':
                    var_name = condition['variable']
                    expected_value = condition['value']
                    actual_value = variables.get(var_name)

                    if actual_value != expected_value:
                        return False

                elif condition_type == 'variable_greater_than':
                    var_name = condition['variable']
                    threshold = condition['value']
                    actual_value = variables.get(var_name, 0)

                    if actual_value <= threshold:
                        return False

                # Add more condition types as needed

            return True

        except Exception as e:
            self.logger.error(
                "Error evaluating transition conditions",
                transition_name=transition.name,
                error=str(e)
            )
            return False

    async def _execute_state_entry_actions(
            self, workflow_definition: WorkflowDefinition,
            state_name: str, execution_context: ExecutionContext
    ) -> List[str]:
        """Execute entry actions for a state."""
        # Placeholder for state entry action execution
        # This will be expanded with actual action execution logic
        self.logger.debug(f"Executing entry actions for state: {state_name}")
        return []

    async def _execute_state_exit_actions(
            self, workflow_definition: WorkflowDefinition,
            state_name: str, execution_context: ExecutionContext) -> List[str]:
        """Execute exit actions for a state."""
        # Placeholder for state exit action execution
        self.logger.debug(f"Executing exit actions for state: {state_name}")
        return []

    async def _execute_transition_actions(self,
                                          transition: TransitionDefinition, execution_context: ExecutionContext,
                                          event: Optional[ExecutionEvent] = None) -> List[str]:
        """Execute actions during a transition."""
        # Placeholder for transition action execution
        self.logger.debug(f"Executing transition actions for: {transition.name}")
        return []

    def get_execution_context(self, instance_id: str) -> Optional[ExecutionContext]:
        """Get execution context for an instance."""
        return self._execution_contexts.get(instance_id)

    def stop_workflow_instance(self, instance_id: str) -> bool:
        """
        Stop a running workflow instance.

        Args:
            instance_id: Instance identifier

        Returns:
            bool: True if stopped successfully
        """
        execution_context = self._execution_contexts.get(instance_id)
        if execution_context:
            execution_context.cancel_execution()
            self._execution_contexts.pop(instance_id, None)
            self.logger.info("Workflow instance stopped", instance_id=instance_id)
            return True
        return False

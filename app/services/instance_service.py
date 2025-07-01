"""
Enterprise State Machine Workflow Engine - Instance Service

Business logic layer for workflow instance management including
execution control, monitoring, and lifecycle management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.instance import (
    WorkflowInstance,
    InstanceStatus,
    Priority,
    ExecutionHistory,
    HistoryEventType
)
from app.models.state_machine import ExecutionContext, ExecutionStatus
from app.models.workflow import WorkflowDefinition, WorkflowStatus
from app.schemas.common import PaginatedResponse
from app.schemas.instance import (
    InstanceCreateRequest,
    InstanceResponse,
    InstanceSummaryResponse,
    InstanceListParams,
    InstanceEventRequest,
    ExecutionHistoryResponse,
    InstanceStatsResponse
)
from app.state_machine.engine import StateMachineEngine, ExecutionEvent

logger = get_logger(__name__)


class InstanceServiceError(Exception):
    """Base exception for instance service errors."""
    pass


class InstanceNotFoundError(InstanceServiceError):
    """Raised when an instance is not found."""
    pass


class InstanceStateError(InstanceServiceError):
    """Raised when instance is in invalid state for operation."""
    pass


class InstanceService:
    """
    Service class for workflow instance management.

    Handles all business logic related to workflow instances including
    creation, execution control, monitoring, and lifecycle management.
    """

    def __init__(self):
        """Initialize the instance service."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.state_machine_engine = StateMachineEngine()

    async def create_instance(
            self,
            db: Session,
            instance_data: InstanceCreateRequest,
            created_by: Optional[UUID] = None
    ) -> InstanceResponse:
        """
        Create a new workflow instance.

        Args:
            db: Database session
            instance_data: Instance creation data
            created_by: ID of the user creating the instance

        Returns:
            InstanceResponse: Created instance

        Raises:
            InstanceServiceError: If creation fails
        """
        self.logger.info(
            "Creating new workflow instance",
            workflow_definition_id=instance_data.workflow_definition_id
        )

        try:
            # Validate workflow definition exists and is active
            workflow = db.query(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.id == instance_data.workflow_definition_id,
                    WorkflowDefinition.is_deleted == False
                )
            ).first()

            if not workflow:
                raise InstanceServiceError(
                    f"Workflow definition {instance_data.workflow_definition_id} not found"
                )

            if workflow.status not in (WorkflowStatus.ACTIVE, WorkflowStatus.DRAFT):
                raise InstanceServiceError(
                    f"Cannot create instance for workflow in status: {workflow.status.value}"
                )

            # Validate input data against schema if provided
            if workflow.input_schema and instance_data.input_data:
                # TODO: Implement JSON schema validation
                pass

            # Create workflow instance
            instance = WorkflowInstance(
                workflow_definition_id=instance_data.workflow_definition_id,
                name=instance_data.name,
                description=instance_data.description,
                status=InstanceStatus.CREATED,
                priority=Priority(instance_data.priority.value),
                scheduled_at=instance_data.scheduled_at,
                timeout_at=instance_data.timeout_at,
                max_retries=instance_data.max_retries or workflow.max_retries,
                input_data=instance_data.input_data,
                context_data=instance_data.context_data,
                tags=instance_data.tags,
                metadata=instance_data.metadata,
                external_id=instance_data.external_id,
                correlation_id=instance_data.correlation_id,
                created_by=created_by,
                updated_by=created_by
            )

            db.add(instance)
            db.flush()  # Get the ID without committing

            # Create execution context
            execution_context = ExecutionContext(
                workflow_instance_id=instance.id,
                current_state="",  # Will be set when started
                status=ExecutionStatus.PENDING,
                input_data=instance_data.input_data,
                variables={}
            )

            db.add(execution_context)

            # Create history entry
            history_entry = ExecutionHistory(
                workflow_instance_id=instance.id,
                event_type=HistoryEventType.INSTANCE_CREATED,
                description="Workflow instance created",
                event_data={
                    "workflow_definition_id": str(instance_data.workflow_definition_id),
                    "priority": instance_data.priority.value,
                    "input_data_size": len(str(instance_data.input_data)) if instance_data.input_data else 0
                },
                actor_type="user" if created_by else "system",
                actor_id=str(created_by) if created_by else None,
                sequence_number=1
            )

            db.add(history_entry)

            # Update workflow instance count
            workflow.increment_instance_count()

            db.commit()
            db.refresh(instance)

            self.logger.info(
                "Workflow instance created successfully",
                instance_id=instance.id,
                workflow_id=workflow.id
            )

            return await self._to_instance_response(db, instance)

        except InstanceServiceError:
            raise
        except IntegrityError as e:
            db.rollback()
            self.logger.error("Database integrity error during instance creation", error=str(e))
            raise InstanceServiceError("Failed to create instance due to data conflict")
        except Exception as e:
            db.rollback()
            self.logger.error("Unexpected error during instance creation", error=str(e), exc_info=True)
            raise InstanceServiceError(f"Failed to create instance: {str(e)}")

    async def get_instance_by_id(
            self,
            db: Session,
            instance_id: UUID,
            include_deleted: bool = False
    ) -> InstanceResponse:
        """
        Get an instance by ID.

        Args:
            db: Database session
            instance_id: Instance ID
            include_deleted: Whether to include soft-deleted instances

        Returns:
            InstanceResponse: Instance data

        Raises:
            InstanceNotFoundError: If instance is not found
        """
        query = db.query(WorkflowInstance).filter(WorkflowInstance.id == instance_id)

        if not include_deleted:
            query = query.filter(WorkflowInstance.is_deleted == False)

        instance = query.first()

        if not instance:
            raise InstanceNotFoundError(f"Instance with ID {instance_id} not found")

        return await self._to_instance_response(db, instance)

    async def get_instances(
            self,
            db: Session,
            params: InstanceListParams
    ) -> PaginatedResponse:
        """
        Get paginated list of instances.

        Args:
            db: Database session
            params: List parameters

        Returns:
            PaginatedResponse: Paginated instance list
        """
        # Build base query
        query = db.query(WorkflowInstance).filter(WorkflowInstance.is_deleted == False)

        # Apply filters
        if params.workflow_definition_id:
            query = query.filter(WorkflowInstance.workflow_definition_id == params.workflow_definition_id)

        if params.status:
            query = query.filter(WorkflowInstance.status == params.status.value)

        if params.priority:
            query = query.filter(WorkflowInstance.priority == params.priority.value)

        if params.started_after:
            query = query.filter(WorkflowInstance.started_at >= params.started_after)

        if params.started_before:
            query = query.filter(WorkflowInstance.started_at <= params.started_before)

        if params.completed_after:
            query = query.filter(WorkflowInstance.completed_at >= params.completed_after)

        if params.completed_before:
            query = query.filter(WorkflowInstance.completed_at <= params.completed_before)

        if params.external_id:
            query = query.filter(WorkflowInstance.external_id == params.external_id)

        if params.correlation_id:
            query = query.filter(WorkflowInstance.correlation_id == params.correlation_id)

        if params.search:
            search_term = f"%{params.search}%"
            query = query.filter(
                or_(
                    WorkflowInstance.name.ilike(search_term),
                    WorkflowInstance.description.ilike(search_term)
                )
            )

        if params.tags:
            query = query.filter(WorkflowInstance.tags.op('&&')(params.tags))

        # Apply sorting
        if params.sort_by:
            sort_column = getattr(WorkflowInstance, params.sort_by, None)
            if sort_column:
                if params.sort_order.value == "desc":
                    query = query.order_by(desc(sort_column))
                else:
                    query = query.order_by(asc(sort_column))
        else:
            query = query.order_by(desc(WorkflowInstance.created_at))

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (params.page - 1) * params.page_size
        instances = query.offset(offset).limit(params.page_size).all()

        # Convert to response format
        items = [await self._to_instance_summary_response(db, i) for i in instances]

        total_pages = (total + params.page_size - 1) // params.page_size

        return PaginatedResponse(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            total_pages=total_pages,
            has_next=params.page < total_pages,
            has_prev=params.page > 1
        )

    async def start_instance(
            self,
            db: Session,
            instance_id: UUID,
            started_by: Optional[UUID] = None
    ) -> InstanceResponse:
        """
        Start a workflow instance execution.

        Args:
            db: Database session
            instance_id: Instance ID
            started_by: ID of the user starting the instance

        Returns:
            InstanceResponse: Started instance

        Raises:
            InstanceNotFoundError: If instance is not found
            InstanceStateError: If instance cannot be started
        """
        self.logger.info("Starting workflow instance", instance_id=instance_id)

        try:
            instance = db.query(WorkflowInstance).filter(
                and_(
                    WorkflowInstance.id == instance_id,
                    WorkflowInstance.is_deleted == False
                )
            ).first()

            if not instance:
                raise InstanceNotFoundError(f"Instance with ID {instance_id} not found")

            if instance.status != InstanceStatus.CREATED:
                raise InstanceStateError(
                    f"Cannot start instance in status: {instance.status.value}"
                )

            # Get workflow definition
            workflow = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.id == instance.workflow_definition_id
            ).first()

            if not workflow:
                raise InstanceServiceError("Workflow definition not found")

            # Start instance
            instance.start_execution(instance.input_data)

            # Start state machine execution
            execution_context = await self.state_machine_engine.start_workflow_instance(
                workflow,
                str(instance.id),
                instance.input_data
            )

            # Update execution context in database
            db_execution_context = db.query(ExecutionContext).filter(
                ExecutionContext.workflow_instance_id == instance.id
            ).first()

            if db_execution_context:
                db_execution_context.current_state = execution_context.current_state
                db_execution_context.status = execution_context.status
                db_execution_context.started_at = execution_context.started_at
                db_execution_context.last_activity_at = execution_context.last_activity_at
                db_execution_context.variables = execution_context.variables

            # Create history entry
            await self._create_history_entry(
                db,
                instance.id,
                HistoryEventType.INSTANCE_STARTED,
                "Workflow instance started",
                {"initial_state": execution_context.current_state},
                started_by
            )

            db.commit()
            db.refresh(instance)

            self.logger.info("Workflow instance started successfully", instance_id=instance_id)

            return await self._to_instance_response(db, instance)

        except (InstanceNotFoundError, InstanceStateError):
            raise
        except Exception as e:
            db.rollback()
            self.logger.error("Error starting instance", instance_id=instance_id, error=str(e))
            raise InstanceServiceError(f"Failed to start instance: {str(e)}")

    async def pause_instance(
            self,
            db: Session,
            instance_id: UUID,
            paused_by: Optional[UUID] = None
    ) -> InstanceResponse:
        """
        Pause a running workflow instance.

        Args:
            db: Database session
            instance_id: Instance ID
            paused_by: ID of the user pausing the instance

        Returns:
            InstanceResponse: Paused instance
        """
        return await self._control_instance(
            db, instance_id, "pause", paused_by, InstanceStatus.PAUSED
        )

    async def resume_instance(
            self,
            db: Session,
            instance_id: UUID,
            resumed_by: Optional[UUID] = None
    ) -> InstanceResponse:
        """
        Resume a paused workflow instance.

        Args:
            db: Database session
            instance_id: Instance ID
            resumed_by: ID of the user resuming the instance

        Returns:
            InstanceResponse: Resumed instance
        """
        return await self._control_instance(
            db, instance_id, "resume", resumed_by, InstanceStatus.RUNNING
        )

    async def cancel_instance(
            self,
            db: Session,
            instance_id: UUID,
            cancelled_by: Optional[UUID] = None,
            reason: Optional[str] = None
    ) -> InstanceResponse:
        """
        Cancel a workflow instance.

        Args:
            db: Database session
            instance_id: Instance ID
            cancelled_by: ID of the user cancelling the instance
            reason: Cancellation reason

        Returns:
            InstanceResponse: Cancelled instance
        """
        return await self._control_instance(
            db, instance_id, "cancel", cancelled_by, InstanceStatus.CANCELLED, reason
        )

    async def send_event_to_instance(
            self,
            db: Session,
            instance_id: UUID,
            event_data: InstanceEventRequest,
            sent_by: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Send an event to a workflow instance.

        Args:
            db: Database session
            instance_id: Instance ID
            event_data: Event data
            sent_by: ID of the user sending the event

        Returns:
            Dict[str, Any]: Event processing result
        """
        self.logger.info(
            "Sending event to instance",
            instance_id=instance_id,
            event_name=event_data.event_name
        )

        try:
            instance = db.query(WorkflowInstance).filter(
                and_(
                    WorkflowInstance.id == instance_id,
                    WorkflowInstance.is_deleted == False
                )
            ).first()

            if not instance:
                raise InstanceNotFoundError(f"Instance with ID {instance_id} not found")

            if not instance.is_running:
                raise InstanceStateError(
                    f"Cannot send event to instance in status: {instance.status.value}"
                )

            # Get workflow definition
            workflow = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.id == instance.workflow_definition_id
            ).first()

            # Create execution event
            execution_event = ExecutionEvent(
                name=event_data.event_name,
                event_type=event_data.event_type,
                payload=event_data.payload,
                timestamp=datetime.utcnow(),
                source=event_data.source
            )

            # Process event through state machine
            transition_results = await self.state_machine_engine.process_event(
                str(instance.id),
                execution_event,
                workflow
            )

            # Update instance activity
            instance.last_activity_at = datetime.utcnow()

            # Create history entry
            await self._create_history_entry(
                db,
                instance.id,
                HistoryEventType.EVENT_RECEIVED,
                f"Event '{event_data.event_name}' received",
                {
                    "event_name": event_data.event_name,
                    "event_type": event_data.event_type,
                    "payload": event_data.payload,
                    "source": event_data.source
                },
                sent_by
            )

            db.commit()

            result = {
                "event_processed": True,
                "transitions_executed": len([r for r in transition_results if r.success]),
                "transition_results": [
                    {
                        "success": r.success,
                        "from_state": r.from_state,
                        "to_state": r.to_state,
                        "transition_name": r.transition_name,
                        "error_message": r.error_message
                    }
                    for r in transition_results
                ]
            }

            self.logger.info(
                "Event sent to instance successfully",
                instance_id=instance_id,
                event_name=event_data.event_name,
                transitions_executed=result["transitions_executed"]
            )

            return result

        except (InstanceNotFoundError, InstanceStateError):
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(
                "Error sending event to instance",
                instance_id=instance_id,
                event_name=event_data.event_name,
                error=str(e)
            )
            raise InstanceServiceError(f"Failed to send event: {str(e)}")

    async def get_instance_history(
            self,
            db: Session,
            instance_id: UUID,
            limit: int = 100,
            offset: int = 0
    ) -> List[ExecutionHistoryResponse]:
        """
        Get execution history for an instance.

        Args:
            db: Database session
            instance_id: Instance ID
            limit: Maximum number of entries
            offset: Offset for pagination

        Returns:
            List[ExecutionHistoryResponse]: History entries
        """
        history_entries = db.query(ExecutionHistory).filter(
            ExecutionHistory.workflow_instance_id == instance_id
        ).order_by(
            ExecutionHistory.sequence_number.desc()
        ).offset(offset).limit(limit).all()

        return [self._to_history_response(entry) for entry in history_entries]

    async def get_instance_stats(self, db: Session) -> InstanceStatsResponse:
        """
        Get instance statistics.

        Args:
            db: Database session

        Returns:
            InstanceStatsResponse: Instance statistics
        """
        try:
            # Basic instance counts
            total_instances = db.query(WorkflowInstance).count()

            running_instances = db.query(WorkflowInstance).filter(
                WorkflowInstance.status == InstanceStatus.RUNNING
            ).count()

            completed_instances = db.query(WorkflowInstance).filter(
                WorkflowInstance.status == InstanceStatus.COMPLETED
            ).count()

            failed_instances = db.query(WorkflowInstance).filter(
                WorkflowInstance.status == InstanceStatus.FAILED
            ).count()

            # Success rate
            success_rate = 0.0
            if total_instances > 0:
                success_rate = (completed_instances / total_instances) * 100

            # Average execution time
            avg_execution_time = db.query(
                func.avg(WorkflowInstance.duration_seconds)
            ).filter(
                WorkflowInstance.duration_seconds.is_not(None)
            ).scalar()

            # Average retry count
            avg_retry_count = db.query(
                func.avg(WorkflowInstance.retry_count)
            ).scalar() or 0.0

            # Status distribution
            status_counts = db.query(
                WorkflowInstance.status,
                func.count(WorkflowInstance.id)
            ).group_by(WorkflowInstance.status).all()

            status_distribution = {status.value: count for status, count in status_counts}

            # Priority distribution
            priority_counts = db.query(
                WorkflowInstance.priority,
                func.count(WorkflowInstance.id)
            ).group_by(WorkflowInstance.priority).all()

            priority_distribution = {priority.value: count for priority, count in priority_counts}

            return InstanceStatsResponse(
                total_instances=total_instances,
                running_instances=running_instances,
                completed_instances=completed_instances,
                failed_instances=failed_instances,
                success_rate=round(success_rate, 2),
                avg_execution_time=float(avg_execution_time) if avg_execution_time else None,
                avg_retry_count=round(avg_retry_count, 2),
                status_distribution=status_distribution,
                priority_distribution=priority_distribution
            )

        except Exception as e:
            self.logger.error("Error getting instance stats", error=str(e))
            raise InstanceServiceError(f"Failed to get instance statistics: {str(e)}")

    async def set_instance_variables(
            self,
            db: Session,
            instance_id: UUID,
            variables: Dict[str, Any],
            set_by: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Set variables for a workflow instance.

        Args:
            db: Database session
            instance_id: Instance ID
            variables: Variables to set
            set_by: ID of the user setting variables

        Returns:
            Dict[str, Any]: Updated variables
        """
        try:
            instance = db.query(WorkflowInstance).filter(
                and_(
                    WorkflowInstance.id == instance_id,
                    WorkflowInstance.is_deleted == False
                )
            ).first()

            if not instance:
                raise InstanceNotFoundError(f"Instance with ID {instance_id} not found")

            # Update context data
            if not instance.context_data:
                instance.context_data = {}

            instance.context_data.update(variables)
            instance.last_activity_at = datetime.utcnow()

            # Update execution context if exists
            execution_context = db.query(ExecutionContext).filter(
                ExecutionContext.workflow_instance_id == instance_id
            ).first()

            if execution_context:
                if not execution_context.variables:
                    execution_context.variables = {}
                execution_context.variables.update(variables)
                execution_context.last_activity_at = datetime.utcnow()

            # Create history entries for variable changes
            for key, value in variables.items():
                await self._create_history_entry(
                    db,
                    instance.id,
                    HistoryEventType.VARIABLE_SET,
                    f"Variable '{key}' set",
                    {"variable": key, "value": value},
                    set_by
                )

            db.commit()

            return instance.context_data

        except InstanceNotFoundError:
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(
                "Error setting instance variables",
                instance_id=instance_id,
                error=str(e)
            )
            raise InstanceServiceError(f"Failed to set variables: {str(e)}")

    async def _control_instance(
            self,
            db: Session,
            instance_id: UUID,
            action: str,
            actor_id: Optional[UUID],
            target_status: InstanceStatus,
            reason: Optional[str] = None
    ) -> InstanceResponse:
        """Internal method for instance control operations."""
        try:
            instance = db.query(WorkflowInstance).filter(
                and_(
                    WorkflowInstance.id == instance_id,
                    WorkflowInstance.is_deleted == False
                )
            ).first()

            if not instance:
                raise InstanceNotFoundError(f"Instance with ID {instance_id} not found")

            # Validate state transition
            valid_transitions = {
                InstanceStatus.PAUSED: [InstanceStatus.RUNNING],
                InstanceStatus.RUNNING: [InstanceStatus.PAUSED, InstanceStatus.CANCELLED],
                InstanceStatus.CANCELLED: [InstanceStatus.RUNNING, InstanceStatus.PAUSED, InstanceStatus.WAITING]
            }

            if target_status not in valid_transitions.get(instance.status, [target_status]):
                raise InstanceStateError(
                    f"Cannot {action} instance in status: {instance.status.value}"
                )

            # Update instance status
            if action == "pause":
                instance.pause_execution()
            elif action == "resume":
                instance.resume_execution()
            elif action == "cancel":
                instance.cancel_execution()

            # Update state machine
            if action == "cancel":
                self.state_machine_engine.stop_workflow_instance(str(instance.id))

            # Create history entry
            event_type_map = {
                "pause": HistoryEventType.INSTANCE_PAUSED,
                "resume": HistoryEventType.INSTANCE_RESUMED,
                "cancel": HistoryEventType.INSTANCE_CANCELLED
            }

            await self._create_history_entry(
                db,
                instance.id,
                event_type_map[action],
                f"Instance {action}ed",
                {"reason": reason} if reason else {},
                actor_id
            )

            db.commit()
            db.refresh(instance)

            return await self._to_instance_response(db, instance)

        except (InstanceNotFoundError, InstanceStateError):
            raise
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error {action}ing instance", instance_id=instance_id, error=str(e))
            raise InstanceServiceError(f"Failed to {action} instance: {str(e)}")

    async def _create_history_entry(
            self,
            db: Session,
            instance_id: UUID,
            event_type: HistoryEventType,
            description: str,
            event_data: Optional[Dict[str, Any]] = None,
            actor_id: Optional[UUID] = None
    ) -> None:
        """Create a history entry for an instance."""
        # Get next sequence number
        max_seq = db.query(func.max(ExecutionHistory.sequence_number)).filter(
            ExecutionHistory.workflow_instance_id == instance_id
        ).scalar() or 0

        history_entry = ExecutionHistory(
            workflow_instance_id=instance_id,
            event_type=event_type,
            description=description,
            event_data=event_data,
            actor_type="user" if actor_id else "system",
            actor_id=str(actor_id) if actor_id else None,
            sequence_number=max_seq + 1
        )

        db.add(history_entry)

    async def _to_instance_response(self, db: Session, instance: WorkflowInstance) -> InstanceResponse:
        """Convert instance model to response schema."""
        # Get execution context
        execution_context = db.query(ExecutionContext).filter(
            ExecutionContext.workflow_instance_id == instance.id
        ).first()

        # Get workflow definition summary
        workflow = db.query(WorkflowDefinition).filter(
            WorkflowDefinition.id == instance.workflow_definition_id
        ).first()

        workflow_summary = None
        if workflow:
            workflow_summary = {
                "id": str(workflow.id),
                "name": workflow.name,
                "version": workflow.version,
                "status": workflow.status.value
            }

        return InstanceResponse(
            id=instance.id,
            workflow_definition_id=instance.workflow_definition_id,
            name=instance.name,
            description=instance.description,
            status=instance.status.value,
            priority=instance.priority.value,
            scheduled_at=instance.scheduled_at,
            started_at=instance.started_at,
            completed_at=instance.completed_at,
            last_activity_at=instance.last_activity_at,
            timeout_at=instance.timeout_at,
            max_retries=instance.max_retries,
            retry_count=instance.retry_count,
            input_data=instance.input_data or {},
            output_data=instance.output_data,
            context_data=instance.context_data or {},
            error_message=instance.error_message,
            error_details=instance.error_details,
            tags=instance.tags or [],
            metadata=instance.metadata or {},
            parent_instance_id=instance.parent_instance_id,
            external_id=instance.external_id,
            correlation_id=instance.correlation_id,
            step_count=instance.step_count,
            duration_seconds=instance.duration_seconds,
            execution_context=self._to_execution_context_response(execution_context) if execution_context else None,
            workflow_definition=workflow_summary,
            created_at=instance.created_at,
            updated_at=instance.updated_at,
            created_by=instance.created_by,
            updated_by=instance.updated_by,
            is_deleted=instance.is_deleted,
            deleted_at=instance.deleted_at
        )

    async def _to_instance_summary_response(
            self,
            db: Session,
            instance: WorkflowInstance
    ) -> InstanceSummaryResponse:
        """Convert instance model to summary response schema."""
        # Get current state from execution context
        execution_context = db.query(ExecutionContext).filter(
            ExecutionContext.workflow_instance_id == instance.id
        ).first()

        current_state = execution_context.current_state if execution_context else None

        return InstanceSummaryResponse(
            id=instance.id,
            workflow_definition_id=instance.workflow_definition_id,
            name=instance.name,
            status=instance.status.value,
            priority=instance.priority.value,
            current_state=current_state,
            started_at=instance.started_at,
            completed_at=instance.completed_at,
            duration_seconds=instance.duration_seconds,
            retry_count=instance.retry_count,
            tags=instance.tags or [],
            created_at=instance.created_at
        )

    def _to_execution_context_response(self, context: ExecutionContext) -> Dict[str, Any]:
        """Convert execution context to response format."""
        return {
            "id": context.id,
            "workflow_instance_id": context.workflow_instance_id,
            "current_state": context.current_state,
            "previous_state": context.previous_state,
            "status": context.status.value,
            "variables": context.variables or {},
            "input_data": context.input_data,
            "output_data": context.output_data,
            "started_at": context.started_at,
            "completed_at": context.completed_at,
            "last_activity_at": context.last_activity_at,
            "error_message": context.error_message,
            "error_details": context.error_details,
            "retry_count": context.retry_count,
            "transition_count": context.transition_count,
            "task_count": context.task_count,
            "duration_seconds": context.duration_seconds,
            "created_at": context.created_at,
            "updated_at": context.updated_at
        }

    def _to_history_response(self, history: ExecutionHistory) -> ExecutionHistoryResponse:
        """Convert history model to response schema."""
        return ExecutionHistoryResponse(
            id=history.id,
            workflow_instance_id=history.workflow_instance_id,
            event_type=history.event_type.value,
            event_name=history.event_name,
            description=history.description,
            from_state=history.from_state,
            to_state=history.to_state,
            event_data=history.event_data,
            event_timestamp=history.event_timestamp,
            duration_ms=history.duration_ms,
            actor_type=history.actor_type,
            actor_id=history.actor_id,
            execution_context=history.execution_context,
            error_message=history.error_message,
            error_details=history.error_details,
            sequence_number=history.sequence_number
        )

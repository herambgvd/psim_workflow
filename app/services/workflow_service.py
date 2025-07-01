"""
Enterprise State Machine Workflow Engine - Workflow Service

Business logic layer for workflow definition management including
CRUD operations, validation, import/export, and statistics.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.exc import IntegrityError
import structlog

from app.models.workflow import WorkflowDefinition, WorkflowStatus
from app.models.instance import WorkflowInstance, InstanceStatus
from app.schemas.workflow import (
    WorkflowCreateRequest,
    WorkflowUpdateRequest,
    WorkflowResponse,
    WorkflowSummaryResponse,
    WorkflowListParams,
    WorkflowValidationResponse,
    WorkflowStatsResponse,
    WorkflowImportRequest,
    WorkflowImportResponse
)
from app.schemas.common import PaginatedResponse
from app.core.logging import get_logger


logger = get_logger(__name__)


class WorkflowServiceError(Exception):
    """Base exception for workflow service errors."""
    pass


class WorkflowNotFoundError(WorkflowServiceError):
    """Raised when a workflow is not found."""
    pass


class WorkflowValidationError(WorkflowServiceError):
    """Raised when workflow validation fails."""
    pass


class WorkflowService:
    """
    Service class for workflow definition management.

    Handles all business logic related to workflow definitions including
    creation, updates, validation, and lifecycle management.
    """

    def __init__(self):
        """Initialize the workflow service."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def create_workflow(
            self,
            db: Session,
            workflow_data: WorkflowCreateRequest,
            created_by: Optional[UUID] = None
    ) -> WorkflowResponse:
        """
        Create a new workflow definition.

        Args:
            db: Database session
            workflow_data: Workflow creation data
            created_by: ID of the user creating the workflow

        Returns:
            WorkflowResponse: Created workflow

        Raises:
            WorkflowValidationError: If validation fails
            WorkflowServiceError: If creation fails
        """
        self.logger.info(
            "Creating new workflow",
            workflow_name=workflow_data.name,
            version=workflow_data.version
        )

        try:
            # Validate workflow definition
            validation_result = await self.validate_workflow_definition(
                workflow_data.definition.model_dump()
            )

            if not validation_result.is_valid:
                raise WorkflowValidationError(
                    f"Workflow validation failed: {', '.join(validation_result.errors)}"
                )

            # Check for duplicate name/version combination
            existing = db.query(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.name == workflow_data.name,
                    WorkflowDefinition.version == workflow_data.version,
                    WorkflowDefinition.is_deleted == False
                )
            ).first()

            if existing:
                raise WorkflowServiceError(
                    f"Workflow with name '{workflow_data.name}' and version '{workflow_data.version}' already exists"
                )

            # Create workflow definition
            workflow = WorkflowDefinition(
                name=workflow_data.name,
                description=workflow_data.description,
                version=workflow_data.version,
                status=WorkflowStatus.DRAFT,
                definition=workflow_data.definition.model_dump(),
                input_schema=workflow_data.input_schema,
                output_schema=workflow_data.output_schema,
                timeout_seconds=workflow_data.timeout_seconds,
                max_retries=workflow_data.max_retries,
                retry_delay_seconds=workflow_data.retry_delay_seconds,
                tags=workflow_data.tags,
                metadata=workflow_data.metadata,
                is_template=workflow_data.is_template,
                created_by=created_by,
                updated_by=created_by
            )

            db.add(workflow)
            db.commit()
            db.refresh(workflow)

            self.logger.info(
                "Workflow created successfully",
                workflow_id=workflow.id,
                workflow_name=workflow.name
            )

            return self._to_workflow_response(workflow)

        except WorkflowValidationError:
            raise
        except IntegrityError as e:
            db.rollback()
            self.logger.error("Database integrity error during workflow creation", error=str(e))
            raise WorkflowServiceError("Failed to create workflow due to data conflict")
        except Exception as e:
            db.rollback()
            self.logger.error("Unexpected error during workflow creation", error=str(e), exc_info=True)
            raise WorkflowServiceError(f"Failed to create workflow: {str(e)}")

    async def get_workflow_by_id(
            self,
            db: Session,
            workflow_id: UUID,
            include_deleted: bool = False
    ) -> WorkflowResponse:
        """
        Get a workflow by ID.

        Args:
            db: Database session
            workflow_id: Workflow ID
            include_deleted: Whether to include soft-deleted workflows

        Returns:
            WorkflowResponse: Workflow data

        Raises:
            WorkflowNotFoundError: If workflow is not found
        """
        query = db.query(WorkflowDefinition).filter(WorkflowDefinition.id == workflow_id)

        if not include_deleted:
            query = query.filter(WorkflowDefinition.is_deleted == False)

        workflow = query.first()

        if not workflow:
            raise WorkflowNotFoundError(f"Workflow with ID {workflow_id} not found")

        return self._to_workflow_response(workflow)

    async def get_workflows(
            self,
            db: Session,
            params: WorkflowListParams
    ) -> PaginatedResponse:
        """
        Get paginated list of workflows.

        Args:
            db: Database session
            params: List parameters

        Returns:
            PaginatedResponse: Paginated workflow list
        """
        # Build base query
        query = db.query(WorkflowDefinition).filter(WorkflowDefinition.is_deleted == False)

        # Apply filters
        if params.status:
            query = query.filter(WorkflowDefinition.status == params.status.value)

        if params.is_template is not None:
            query = query.filter(WorkflowDefinition.is_template == params.is_template)

        if params.version:
            query = query.filter(WorkflowDefinition.version == params.version)

        if params.created_after:
            query = query.filter(WorkflowDefinition.created_at >= params.created_after)

        if params.created_before:
            query = query.filter(WorkflowDefinition.created_at <= params.created_before)

        if params.search:
            search_term = f"%{params.search}%"
            query = query.filter(
                or_(
                    WorkflowDefinition.name.ilike(search_term),
                    WorkflowDefinition.description.ilike(search_term)
                )
            )

        if params.tags:
            # PostgreSQL array overlap operator
            query = query.filter(WorkflowDefinition.tags.op('&&')(params.tags))

        # Apply sorting
        if params.sort_by:
            sort_column = getattr(WorkflowDefinition, params.sort_by, None)
            if sort_column:
                if params.sort_order.value == "desc":
                    query = query.order_by(desc(sort_column))
                else:
                    query = query.order_by(asc(sort_column))
        else:
            query = query.order_by(desc(WorkflowDefinition.updated_at))

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (params.page - 1) * params.page_size
        workflows = query.offset(offset).limit(params.page_size).all()

        # Convert to response format
        items = [self._to_workflow_summary_response(w) for w in workflows]

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

    async def update_workflow(
            self,
            db: Session,
            workflow_id: UUID,
            workflow_data: WorkflowUpdateRequest,
            updated_by: Optional[UUID] = None
    ) -> WorkflowResponse:
        """
        Update an existing workflow definition.

        Args:
            db: Database session
            workflow_id: Workflow ID
            workflow_data: Update data
            updated_by: ID of the user updating the workflow

        Returns:
            WorkflowResponse: Updated workflow

        Raises:
            WorkflowNotFoundError: If workflow is not found
            WorkflowValidationError: If validation fails
            WorkflowServiceError: If update fails
        """
        self.logger.info("Updating workflow", workflow_id=workflow_id)

        try:
            workflow = db.query(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.id == workflow_id,
                    WorkflowDefinition.is_deleted == False
                )
            ).first()

            if not workflow:
                raise WorkflowNotFoundError(f"Workflow with ID {workflow_id} not found")

            # Validate new definition if provided
            if workflow_data.definition:
                validation_result = await self.validate_workflow_definition(
                    workflow_data.definition.model_dump()
                )

                if not validation_result.is_valid:
                    raise WorkflowValidationError(
                        f"Workflow validation failed: {', '.join(validation_result.errors)}"
                    )

            # Update fields
            update_data = workflow_data.model_dump(exclude_unset=True)

            for field, value in update_data.items():
                if field == "definition" and value:
                    setattr(workflow, field, value.model_dump() if hasattr(value, 'model_dump') else value)
                elif field == "status" and value:
                    setattr(workflow, field, WorkflowStatus(value.value))
                elif hasattr(workflow, field):
                    setattr(workflow, field, value)

            workflow.updated_by = updated_by
            workflow.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(workflow)

            self.logger.info("Workflow updated successfully", workflow_id=workflow_id)

            return self._to_workflow_response(workflow)

        except (WorkflowNotFoundError, WorkflowValidationError):
            raise
        except Exception as e:
            db.rollback()
            self.logger.error("Error updating workflow", workflow_id=workflow_id, error=str(e))
            raise WorkflowServiceError(f"Failed to update workflow: {str(e)}")

    async def delete_workflow(
            self,
            db: Session,
            workflow_id: UUID,
            hard_delete: bool = False,
            deleted_by: Optional[UUID] = None
    ) -> bool:
        """
        Delete a workflow definition.

        Args:
            db: Database session
            workflow_id: Workflow ID
            hard_delete: Whether to perform hard delete
            deleted_by: ID of the user deleting the workflow

        Returns:
            bool: True if deleted successfully

        Raises:
            WorkflowNotFoundError: If workflow is not found
            WorkflowServiceError: If deletion fails
        """
        self.logger.info("Deleting workflow", workflow_id=workflow_id, hard_delete=hard_delete)

        try:
            workflow = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.id == workflow_id
            ).first()

            if not workflow:
                raise WorkflowNotFoundError(f"Workflow with ID {workflow_id} not found")

            # Check if workflow has running instances
            running_instances = db.query(WorkflowInstance).filter(
                and_(
                    WorkflowInstance.workflow_definition_id == workflow_id,
                    WorkflowInstance.status == InstanceStatus.RUNNING
                )
            ).count()

            if running_instances > 0:
                raise WorkflowServiceError(
                    f"Cannot delete workflow with {running_instances} running instances"
                )

            if hard_delete:
                db.delete(workflow)
            else:
                workflow.soft_delete()
                workflow.updated_by = deleted_by

            db.commit()

            self.logger.info("Workflow deleted successfully", workflow_id=workflow_id)
            return True

        except (WorkflowNotFoundError, WorkflowServiceError):
            raise
        except Exception as e:
            db.rollback()
            self.logger.error("Error deleting workflow", workflow_id=workflow_id, error=str(e))
            raise WorkflowServiceError(f"Failed to delete workflow: {str(e)}")

    async def validate_workflow_definition(
            self,
            definition: Dict[str, Any],
            strict: bool = False
    ) -> WorkflowValidationResponse:
        """
        Validate a workflow definition.

        Args:
            definition: Workflow definition to validate
            strict: Enable strict validation

        Returns:
            WorkflowValidationResponse: Validation result
        """
        errors = []
        warnings = []
        suggestions = []

        try:
            # Basic structure validation
            required_fields = ['states', 'transitions', 'initial_state']
            for field in required_fields:
                if field not in definition:
                    errors.append(f"Missing required field: {field}")

            if errors:
                return WorkflowValidationResponse(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    suggestions=suggestions
                )

            states = definition.get('states', {})
            transitions = definition.get('transitions', [])
            initial_state = definition.get('initial_state')

            # Validate states
            if not states:
                errors.append("At least one state is required")

            state_names = set(states.keys())

            # Validate initial state
            if initial_state and initial_state not in state_names:
                errors.append(f"Initial state '{initial_state}' not found in states")

            # Validate transitions
            for i, transition in enumerate(transitions):
                if 'from_state' not in transition:
                    errors.append(f"Transition {i}: missing 'from_state'")
                    continue

                if 'to_state' not in transition:
                    errors.append(f"Transition {i}: missing 'to_state'")
                    continue

                from_state = transition['from_state']
                to_state = transition['to_state']

                if from_state not in state_names:
                    errors.append(f"Transition {i}: invalid from_state '{from_state}'")

                if to_state not in state_names:
                    errors.append(f"Transition {i}: invalid to_state '{to_state}'")

            # Advanced validation for strict mode
            if strict:
                # Check for unreachable states
                reachable_states = {initial_state} if initial_state else set()

                for transition in transitions:
                    if transition.get('from_state') in reachable_states:
                        reachable_states.add(transition.get('to_state'))

                unreachable = state_names - reachable_states
                if unreachable:
                    warnings.extend([f"Unreachable state: {state}" for state in unreachable])

                # Check for states without outgoing transitions
                states_with_outgoing = {t.get('from_state') for t in transitions}
                final_states = state_names - states_with_outgoing

                if len(final_states) == 0:
                    warnings.append("No final states found - workflow may run indefinitely")

                # Suggest optimizations
                if len(state_names) > 50:
                    suggestions.append("Consider breaking down large workflows into sub-workflows")

                if len(transitions) > 100:
                    suggestions.append("Consider simplifying complex transition logic")

            is_valid = len(errors) == 0

            return WorkflowValidationResponse(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        except Exception as e:
            self.logger.error("Error during workflow validation", error=str(e))
            return WorkflowValidationResponse(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings,
                suggestions=suggestions
            )

    async def get_workflow_stats(self, db: Session) -> WorkflowStatsResponse:
        """
        Get workflow statistics.

        Args:
            db: Database session

        Returns:
            WorkflowStatsResponse: Workflow statistics
        """
        try:
            # Basic workflow counts
            total_workflows = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.is_deleted == False
            ).count()

            active_workflows = db.query(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.status == WorkflowStatus.ACTIVE,
                    WorkflowDefinition.is_deleted == False
                )
            ).count()

            draft_workflows = db.query(WorkflowDefinition).filter(
                and_(
                    WorkflowDefinition.status == WorkflowStatus.DRAFT,
                    WorkflowDefinition.is_deleted == False
                )
            ).count()

            # Instance statistics
            total_instances = db.query(WorkflowInstance).count()

            running_instances = db.query(WorkflowInstance).filter(
                WorkflowInstance.status == InstanceStatus.RUNNING
            ).count()

            # Success rate calculation
            completed_instances = db.query(WorkflowInstance).filter(
                WorkflowInstance.status == InstanceStatus.COMPLETED
            ).count()

            success_rate = 0.0
            if total_instances > 0:
                success_rate = (completed_instances / total_instances) * 100

            # Average execution time
            avg_execution_time = db.query(
                func.avg(WorkflowInstance.duration_seconds)
            ).filter(
                WorkflowInstance.duration_seconds.is_not(None)
            ).scalar()

            return WorkflowStatsResponse(
                total_workflows=total_workflows,
                active_workflows=active_workflows,
                draft_workflows=draft_workflows,
                total_instances=total_instances,
                running_instances=running_instances,
                success_rate=round(success_rate, 2),
                avg_execution_time=float(avg_execution_time) if avg_execution_time else None
            )

        except Exception as e:
            self.logger.error("Error getting workflow stats", error=str(e))
            raise WorkflowServiceError(f"Failed to get workflow statistics: {str(e)}")

    async def activate_workflow(
            self,
            db: Session,
            workflow_id: UUID,
            activated_by: Optional[UUID] = None
    ) -> WorkflowResponse:
        """
        Activate a workflow definition.

        Args:
            db: Database session
            workflow_id: Workflow ID
            activated_by: ID of the user activating the workflow

        Returns:
            WorkflowResponse: Updated workflow
        """
        return await self._update_workflow_status(
            db, workflow_id, WorkflowStatus.ACTIVE, activated_by
        )

    async def deactivate_workflow(
            self,
            db: Session,
            workflow_id: UUID,
            deactivated_by: Optional[UUID] = None
    ) -> WorkflowResponse:
        """
        Deactivate a workflow definition.

        Args:
            db: Database session
            workflow_id: Workflow ID
            deactivated_by: ID of the user deactivating the workflow

        Returns:
            WorkflowResponse: Updated workflow
        """
        return await self._update_workflow_status(
            db, workflow_id, WorkflowStatus.INACTIVE, deactivated_by
        )

    async def _update_workflow_status(
            self,
            db: Session,
            workflow_id: UUID,
            status: WorkflowStatus,
            updated_by: Optional[UUID] = None
    ) -> WorkflowResponse:
        """Update workflow status."""
        workflow = db.query(WorkflowDefinition).filter(
            and_(
                WorkflowDefinition.id == workflow_id,
                WorkflowDefinition.is_deleted == False
            )
        ).first()

        if not workflow:
            raise WorkflowNotFoundError(f"Workflow with ID {workflow_id} not found")

        workflow.status = status
        workflow.updated_by = updated_by
        workflow.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(workflow)

        return self._to_workflow_response(workflow)

    def _to_workflow_response(self, workflow: WorkflowDefinition) -> WorkflowResponse:
        """Convert workflow model to response schema."""
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            version=workflow.version,
            status=workflow.status.value,
            definition=workflow.definition,
            input_schema=workflow.input_schema,
            output_schema=workflow.output_schema,
            timeout_seconds=workflow.timeout_seconds,
            max_retries=workflow.max_retries,
            retry_delay_seconds=workflow.retry_delay_seconds,
            tags=workflow.tags or [],
            metadata=workflow.metadata or {},
            parent_workflow_id=workflow.parent_workflow_id,
            is_template=workflow.is_template,
            instance_count=workflow.instance_count,
            success_count=workflow.success_count,
            failure_count=workflow.failure_count,
            success_rate=workflow.success_rate,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            created_by=workflow.created_by,
            updated_by=workflow.updated_by,
            is_deleted=workflow.is_deleted,
            deleted_at=workflow.deleted_at
        )

    def _to_workflow_summary_response(self, workflow: WorkflowDefinition) -> WorkflowSummaryResponse:
        """Convert workflow model to summary response schema."""
        return WorkflowSummaryResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            version=workflow.version,
            status=workflow.status.value,
            tags=workflow.tags or [],
            instance_count=workflow.instance_count,
            success_rate=workflow.success_rate,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at
        )
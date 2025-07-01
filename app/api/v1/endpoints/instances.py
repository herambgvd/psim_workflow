"""
Enterprise State Machine Workflow Engine - Instance Endpoints

Complete REST API endpoints for workflow instance management with
authentication and RBAC integration - all endpoints implemented.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.orm import Session

from app.api.deps import (
    get_database_session,
    require_instance_create,
    require_instance_read,
    require_instance_update,
    require_instance_control,
    require_instance_events,
    require_instance_history,
    require_instance_variables,
    require_instance_stats,
    get_audit_context
)
from app.core.logging import get_logger
from app.schemas.common import PaginatedResponse
from app.schemas.instance import (
    InstanceCreateRequest,
    InstanceUpdateRequest,
    InstanceResponse,
    InstanceListParams,
    InstanceEventRequest,
    ExecutionHistoryResponse,
    InstanceStatsResponse,
    InstanceVariableRequest,
    InstanceVariableResponse,
    BulkInstanceRequest,
    BulkInstanceResponse,
    InstanceMetricsResponse,
    InstanceRetryRequest,
    InstanceStatusEnum,
    PriorityEnum
)
from app.services.instance_service import (
    InstanceService,
    InstanceServiceError,
    InstanceNotFoundError,
    InstanceStateError
)

logger = get_logger(__name__)
router = APIRouter()

# Initialize instance service
instance_service = InstanceService()


@router.post(
    "/",
    response_model=InstanceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Instance",
    description="Create a new workflow instance from a workflow definition."
)
async def create_instance(
        instance_data: InstanceCreateRequest = Body(..., description="Instance creation data"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_create)
) -> InstanceResponse:
    """
    Create a new workflow instance.

    Creates a workflow instance from an existing workflow definition.
    The instance will be in 'created' status and needs to be started explicitly.
    """
    try:
        logger.info(
            "Creating workflow instance via API",
            workflow_definition_id=instance_data.workflow_definition_id,
            user_id=current_user["id"]
        )

        created_by = UUID(str(current_user["id"]))

        instance = await instance_service.create_instance(
            db=db,
            instance_data=instance_data,
            created_by=created_by
        )

        logger.info(
            "Workflow instance created successfully via API",
            instance_id=instance.id,
            user_id=current_user["id"]
        )
        return instance

    except InstanceServiceError as e:
        logger.error(
            "Instance creation failed",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error creating instance",
            error=str(e),
            user_id=current_user["id"],
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating the instance"
        )


@router.get(
    "/",
    response_model=PaginatedResponse,
    summary="List Instances",
    description="Get a paginated list of workflow instances with filtering and sorting options."
)
async def list_instances(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        sort_by: Optional[str] = Query(None, description="Field to sort by"),
        sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
        search: Optional[str] = Query(None, description="Search term"),
        workflow_definition_id: Optional[UUID] = Query(None, description="Filter by workflow definition"),
        status: Optional[InstanceStatusEnum] = Query(None, description="Filter by status"),
        priority: Optional[PriorityEnum] = Query(None, description="Filter by priority"),
        external_id: Optional[str] = Query(None, description="Filter by external ID"),
        correlation_id: Optional[str] = Query(None, description="Filter by correlation ID"),
        tags: Optional[List[str]] = Query(None, description="Filter by tags"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_read)
) -> PaginatedResponse:
    """
    Retrieve a paginated list of workflow instances.

    Supports filtering by:
    - Workflow definition
    - Status, priority
    - External ID, correlation ID
    - Tags
    - Search terms in name/description

    Supports sorting by any instance field.
    """
    try:
        params = InstanceListParams(
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            search=search,
            workflow_definition_id=workflow_definition_id,
            status=status,
            priority=priority,
            external_id=external_id,
            correlation_id=correlation_id,
            tags=tags
        )

        instances = await instance_service.get_instances(
            db=db,
            params=params,
            user_id=str(current_user["id"])
        )

        logger.debug(
            "Instances listed successfully",
            total=instances.total,
            page=page,
            page_size=page_size,
            user_id=current_user["id"]
        )

        return instances

    except Exception as e:
        logger.error(
            "Error listing instances",
            error=str(e),
            user_id=current_user["id"],
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving instances"
        )


@router.get(
    "/{instance_id}",
    response_model=InstanceResponse,
    summary="Get Instance",
    description="Retrieve a specific workflow instance by ID."
)
async def get_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_read)
) -> InstanceResponse:
    """
    Retrieve a specific workflow instance by its ID.

    Returns the complete instance information including:
    - Current status and execution state
    - Input/output data
    - Execution context and variables
    - Error information (if any)
    - Execution history summary
    """
    try:
        instance = await instance_service.get_instance_by_id(
            db=db,
            instance_id=instance_id,
            user_id=str(current_user["id"])
        )

        logger.debug(
            "Instance retrieved successfully",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        return instance

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except Exception as e:
        logger.error(
            "Error retrieving instance",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the instance"
        )


@router.put(
    "/{instance_id}",
    response_model=InstanceResponse,
    summary="Update Instance",
    description="Update an existing workflow instance."
)
async def update_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        instance_data: InstanceUpdateRequest = Body(..., description="Instance update data"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_update)
) -> InstanceResponse:
    """
    Update an existing workflow instance.

    Allows updating certain instance properties such as:
    - Name and description
    - Priority
    - Tags and metadata

    Note: Core execution data cannot be modified through this endpoint.
    """
    try:
        logger.info(
            "Updating instance via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        updated_by = UUID(str(current_user["id"]))

        instance = await instance_service.update_instance(
            db=db,
            instance_id=instance_id,
            instance_data=instance_data,
            updated_by=updated_by,
            user_id=str(current_user["id"])
        )

        logger.info(
            "Instance update completed via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        return instance

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for update",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except Exception as e:
        logger.error(
            "Unexpected error updating instance",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating the instance"
        )


@router.delete(
    "/{instance_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Instance",
    description="Delete a workflow instance (soft delete by default)."
)
async def delete_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        hard_delete: bool = Query(False, description="Perform hard delete"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_update)  # Using update permission for delete
) -> None:
    """
    Delete a workflow instance.

    By default, performs a soft delete (marks as deleted but preserves data).
    Use hard_delete=true to permanently remove the instance.

    Note: Running instances should be stopped before deletion.
    """
    try:
        logger.info(
            "Deleting instance via API",
            instance_id=instance_id,
            hard_delete=hard_delete,
            user_id=current_user["id"]
        )

        deleted_by = UUID(str(current_user["id"]))

        await instance_service.delete_instance(
            db=db,
            instance_id=instance_id,
            hard_delete=hard_delete,
            deleted_by=deleted_by,
            user_id=str(current_user["id"])
        )

        logger.info(
            "Instance deleted successfully via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for deletion",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except InstanceStateError as e:
        logger.error(
            "Instance deletion failed due to state",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error deleting instance",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting the instance"
        )


@router.post(
    "/{instance_id}/start",
    response_model=InstanceResponse,
    summary="Start Instance",
    description="Start execution of a workflow instance."
)
async def start_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_control)
) -> InstanceResponse:
    """
    Start execution of a workflow instance.

    Begins the execution of a workflow instance that is in 'created' status.
    The instance will transition to 'running' and begin processing according
    to its workflow definition.
    """
    try:
        logger.info(
            "Starting instance via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        started_by = UUID(str(current_user["id"]))

        instance = await instance_service.start_instance(
            db=db,
            instance_id=instance_id,
            started_by=started_by
        )

        logger.info(
            "Instance started successfully via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        return instance

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for start",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except InstanceStateError as e:
        logger.warning(
            "Invalid instance state for start",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except InstanceServiceError as e:
        logger.error(
            "Instance start failed",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error starting instance",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while starting the instance"
        )


@router.post(
    "/{instance_id}/pause",
    response_model=InstanceResponse,
    summary="Pause Instance",
    description="Pause execution of a running workflow instance."
)
async def pause_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_control)
) -> InstanceResponse:
    """
    Pause execution of a running workflow instance.

    Suspends the execution of a running workflow instance.
    The instance can be resumed later from the same state.
    """
    try:
        logger.info(
            "Pausing instance via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        paused_by = UUID(str(current_user["id"]))

        instance = await instance_service.pause_instance(
            db=db,
            instance_id=instance_id,
            paused_by=paused_by
        )

        logger.info(
            "Instance paused successfully via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        return instance

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for pause",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except InstanceStateError as e:
        logger.warning(
            "Invalid instance state for pause",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error pausing instance",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while pausing the instance"
        )


@router.post(
    "/{instance_id}/resume",
    response_model=InstanceResponse,
    summary="Resume Instance",
    description="Resume execution of a paused workflow instance."
)
async def resume_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_control)
) -> InstanceResponse:
    """
    Resume execution of a paused workflow instance.

    Resumes the execution of a paused workflow instance from
    where it was paused.
    """
    try:
        logger.info(
            "Resuming instance via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        resumed_by = UUID(str(current_user["id"]))

        instance = await instance_service.resume_instance(
            db=db,
            instance_id=instance_id,
            resumed_by=resumed_by
        )

        logger.info(
            "Instance resumed successfully via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        return instance

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for resume",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except InstanceStateError as e:
        logger.warning(
            "Invalid instance state for resume",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error resuming instance",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while resuming the instance"
        )


@router.post(
    "/{instance_id}/cancel",
    response_model=InstanceResponse,
    summary="Cancel Instance",
    description="Cancel execution of a workflow instance."
)
async def cancel_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        reason: Optional[str] = Body(None, description="Cancellation reason"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_control)
) -> InstanceResponse:
    """
    Cancel execution of a workflow instance.

    Cancels the execution of a workflow instance. This operation
    is irreversible - the instance cannot be resumed after cancellation.
    """
    try:
        logger.info(
            "Cancelling instance via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        cancelled_by = UUID(str(current_user["id"]))

        instance = await instance_service.cancel_instance(
            db=db,
            instance_id=instance_id,
            cancelled_by=cancelled_by,
            reason=reason
        )

        logger.info(
            "Instance cancelled successfully via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        return instance

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for cancel",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except InstanceStateError as e:
        logger.warning(
            "Invalid instance state for cancel",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error cancelling instance",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while cancelling the instance"
        )


@router.post(
    "/{instance_id}/terminate",
    response_model=InstanceResponse,
    summary="Terminate Instance",
    description="Forcefully terminate a workflow instance."
)
async def terminate_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        reason: Optional[str] = Body(None, description="Termination reason"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_control)
) -> InstanceResponse:
    """
    Forcefully terminate a workflow instance.

    Immediately stops execution of a workflow instance regardless of state.
    This is a more aggressive action than cancel and should be used with caution.
    """
    try:
        logger.info(
            "Terminating instance via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        terminated_by = UUID(str(current_user["id"]))

        instance = await instance_service.terminate_instance(
            db=db,
            instance_id=instance_id,
            terminated_by=terminated_by,
            reason=reason
        )

        logger.info(
            "Instance terminated successfully via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        return instance

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for termination",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except Exception as e:
        logger.error(
            "Unexpected error terminating instance",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while terminating the instance"
        )


@router.post(
    "/{instance_id}/events",
    response_model=Dict[str, Any],
    summary="Send Event",
    description="Send an event to a workflow instance to trigger state transitions."
)
async def send_event_to_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        event_data: InstanceEventRequest = Body(..., description="Event data"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_events)
) -> Dict[str, Any]:
    """
    Send an event to a workflow instance.

    Sends an event to a running workflow instance that may trigger
    state transitions according to the workflow definition.
    """
    try:
        logger.info(
            "Sending event to instance via API",
            instance_id=instance_id,
            event_name=event_data.event_name,
            user_id=current_user["id"]
        )

        sent_by = UUID(str(current_user["id"]))

        result = await instance_service.send_event_to_instance(
            db=db,
            instance_id=instance_id,
            event_data=event_data,
            sent_by=sent_by
        )

        logger.info(
            "Event sent to instance successfully via API",
            instance_id=instance_id,
            event_name=event_data.event_name,
            user_id=current_user["id"]
        )

        return result

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for event",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except InstanceStateError as e:
        logger.warning(
            "Invalid instance state for event",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except InstanceServiceError as e:
        logger.error(
            "Event sending failed",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error sending event",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while sending the event"
        )


@router.get(
    "/{instance_id}/history",
    response_model=List[ExecutionHistoryResponse],
    summary="Get Instance History",
    description="Get execution history for a workflow instance."
)
async def get_instance_history(
        instance_id: UUID = Path(..., description="Instance ID"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of entries"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        event_type: Optional[str] = Query(None, description="Filter by event type"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_history)
) -> List[ExecutionHistoryResponse]:
    """
    Get execution history for a workflow instance.

    Returns a list of execution history entries showing all events
    and state changes that occurred during the instance execution.
    """
    try:
        history = await instance_service.get_instance_history(
            db=db,
            instance_id=instance_id,
            limit=limit,
            offset=offset,
            event_type=event_type,
            user_id=str(current_user["id"])
        )

        logger.debug(
            "Instance history retrieved successfully",
            instance_id=instance_id,
            entries_count=len(history),
            user_id=current_user["id"]
        )

        return history

    except Exception as e:
        logger.error(
            "Error retrieving instance history",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving instance history"
        )


@router.post(
    "/{instance_id}/variables",
    response_model=InstanceVariableResponse,
    summary="Set Instance Variables",
    description="Set runtime variables for a workflow instance."
)
async def set_instance_variables(
        instance_id: UUID = Path(..., description="Instance ID"),
        variable_data: InstanceVariableRequest = Body(..., description="Variables to set"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_variables)
) -> InstanceVariableResponse:
    """
    Set runtime variables for a workflow instance.

    Updates the runtime variables that can be used by the workflow
    during execution for decision making and data passing.
    """
    try:
        logger.info(
            "Setting instance variables via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        set_by = UUID(str(current_user["id"]))

        variables = await instance_service.set_instance_variables(
            db=db,
            instance_id=instance_id,
            variables=variable_data.variables,
            set_by=set_by
        )

        logger.info(
            "Instance variables set successfully via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        return InstanceVariableResponse(
            variables=variables,
            updated_at=datetime.utcnow()
        )

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for variable setting",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except InstanceServiceError as e:
        logger.error(
            "Variable setting failed",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error setting variables",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while setting variables"
        )


@router.get(
    "/{instance_id}/variables",
    response_model=InstanceVariableResponse,
    summary="Get Instance Variables",
    description="Get current runtime variables for a workflow instance."
)
async def get_instance_variables(
        instance_id: UUID = Path(..., description="Instance ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_variables)
) -> InstanceVariableResponse:
    """
    Get current runtime variables for a workflow instance.

    Returns all current runtime variables set for the instance.
    """
    try:
        instance = await instance_service.get_instance_by_id(
            db=db,
            instance_id=instance_id,
            user_id=str(current_user["id"])
        )

        logger.debug(
            "Instance variables retrieved successfully",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        return InstanceVariableResponse(
            variables=instance.context_data,
            updated_at=instance.updated_at
        )

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for variable retrieval",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except Exception as e:
        logger.error(
            "Error retrieving instance variables",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving instance variables"
        )


@router.delete(
    "/{instance_id}/variables/{variable_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Instance Variable",
    description="Delete a specific runtime variable from a workflow instance."
)
async def delete_instance_variable(
        instance_id: UUID = Path(..., description="Instance ID"),
        variable_name: str = Path(..., description="Variable name to delete"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_variables)
) -> None:
    """
    Delete a specific runtime variable from a workflow instance.

    Removes the specified variable from the instance's context data.
    """
    try:
        logger.info(
            "Deleting instance variable via API",
            instance_id=instance_id,
            variable_name=variable_name,
            user_id=current_user["id"]
        )

        deleted_by = UUID(str(current_user["id"]))

        await instance_service.delete_instance_variable(
            db=db,
            instance_id=instance_id,
            variable_name=variable_name,
            deleted_by=deleted_by
        )

        logger.info(
            "Instance variable deleted successfully via API",
            instance_id=instance_id,
            variable_name=variable_name,
            user_id=current_user["id"]
        )

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for variable deletion",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except Exception as e:
        logger.error(
            "Error deleting instance variable",
            instance_id=instance_id,
            variable_name=variable_name,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the variable"
        )


@router.get(
    "/stats",
    response_model=InstanceStatsResponse,
    summary="Get Instance Statistics",
    description="Get comprehensive statistics about workflow instances."
)
async def get_instance_statistics(
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_stats)
) -> InstanceStatsResponse:
    """
    Get comprehensive instance statistics.

    Returns statistics including:
    - Instance counts by status and priority
    - Success rates and performance metrics
    - Distribution analysis
    """
    try:
        logger.debug("Getting instance statistics via API", user_id=current_user["id"])

        stats = await instance_service.get_instance_stats(
            db=db,
            user_id=str(current_user["id"])
        )

        logger.debug("Instance statistics retrieved successfully", user_id=current_user["id"])
        return stats

    except Exception as e:
        logger.error(
            "Error getting instance statistics",
            error=str(e),
            user_id=current_user["id"],
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving instance statistics"
        )


@router.get(
    "/{instance_id}/metrics",
    response_model=InstanceMetricsResponse,
    summary="Get Instance Metrics",
    description="Get detailed execution metrics for a workflow instance."
)
async def get_instance_metrics(
        instance_id: UUID = Path(..., description="Instance ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_read)
) -> InstanceMetricsResponse:
    """
    Get detailed execution metrics for a workflow instance.

    Returns comprehensive metrics including execution times,
    state durations, resource usage, and performance data.
    """
    try:
        # Get basic instance data
        instance = await instance_service.get_instance_by_id(
            db=db,
            instance_id=instance_id,
            user_id=str(current_user["id"])
        )

        # Get detailed metrics
        metrics = await instance_service.get_instance_metrics(
            db=db,
            instance_id=instance_id,
            user_id=str(current_user["id"])
        )

        logger.debug(
            "Instance metrics retrieved successfully",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        return metrics

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for metrics",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except Exception as e:
        logger.error(
            "Error retrieving instance metrics",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving instance metrics"
        )


@router.post(
    "/{instance_id}/retry",
    response_model=InstanceResponse,
    summary="Retry Instance",
    description="Retry a failed workflow instance."
)
async def retry_instance(
        instance_id: UUID = Path(..., description="Instance ID"),
        retry_data: InstanceRetryRequest = Body(..., description="Retry configuration"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_control)
) -> InstanceResponse:
    """
    Retry a failed workflow instance.

    Attempts to restart a failed workflow instance, optionally
    resetting to the initial state or providing new input data.
    """
    try:
        logger.info(
            "Retrying instance via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )

        retried_by = UUID(str(current_user["id"]))

        instance = await instance_service.retry_instance(
            db=db,
            instance_id=instance_id,
            retry_data=retry_data,
            retried_by=retried_by
        )

        logger.info(
            "Instance retry completed via API",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        return instance

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for retry",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except InstanceStateError as e:
        logger.warning(
            "Invalid instance state for retry",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error retrying instance",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrying the instance"
        )


@router.post(
    "/bulk",
    response_model=BulkInstanceResponse,
    summary="Bulk Instance Operations",
    description="Perform bulk operations on multiple workflow instances."
)
async def bulk_instance_operations(
        bulk_request: BulkInstanceRequest = Body(..., description="Bulk operation request"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_control)
) -> BulkInstanceResponse:
    """
    Perform bulk operations on multiple workflow instances.

    Supports operations like:
    - Bulk cancel
    - Bulk pause/resume
    - Bulk variable updates
    """
    try:
        logger.info(
            "Performing bulk instance operation via API",
            action=bulk_request.action,
            instance_count=len(bulk_request.instance_ids),
            user_id=current_user["id"]
        )

        performed_by = UUID(str(current_user["id"]))

        response = await instance_service.bulk_instance_operations(
            db=db,
            bulk_request=bulk_request,
            performed_by=performed_by,
            user_id=str(current_user["id"])
        )

        logger.info(
            "Bulk instance operation completed via API",
            total=response.total_count,
            success=response.success_count,
            errors=response.error_count,
            user_id=current_user["id"]
        )

        return response

    except Exception as e:
        logger.error(
            "Error performing bulk operation",
            error=str(e),
            user_id=current_user["id"],
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while performing the bulk operation"
        )


@router.post(
    "/{instance_id}/checkpoint",
    response_model=Dict[str, Any],
    summary="Create Instance Checkpoint",
    description="Create a checkpoint for workflow instance recovery."
)
async def create_instance_checkpoint(
        instance_id: UUID = Path(..., description="Instance ID"),
        checkpoint_name: str = Body(..., description="Checkpoint name"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_control)
) -> Dict[str, Any]:
    """
    Create a checkpoint for workflow instance recovery.

    Creates a snapshot of the current instance state that can be used
    for recovery or rollback purposes.
    """
    try:
        logger.info(
            "Creating instance checkpoint via API",
            instance_id=instance_id,
            checkpoint_name=checkpoint_name,
            user_id=current_user["id"]
        )

        created_by = UUID(str(current_user["id"]))

        checkpoint = await instance_service.create_checkpoint(
            db=db,
            instance_id=instance_id,
            checkpoint_name=checkpoint_name,
            created_by=created_by
        )

        logger.info(
            "Instance checkpoint created successfully via API",
            instance_id=instance_id,
            checkpoint_name=checkpoint_name,
            user_id=current_user["id"]
        )

        return {
            "checkpoint_id": checkpoint["id"],
            "checkpoint_name": checkpoint_name,
            "created_at": checkpoint["created_at"],
            "instance_id": str(instance_id),
            "created_by": str(created_by)
        }

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for checkpoint creation",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except Exception as e:
        logger.error(
            "Error creating instance checkpoint",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while creating the checkpoint"
        )


@router.get(
    "/{instance_id}/checkpoints",
    response_model=List[Dict[str, Any]],
    summary="Get Instance Checkpoints",
    description="Get all checkpoints for a workflow instance."
)
async def get_instance_checkpoints(
        instance_id: UUID = Path(..., description="Instance ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_instance_read)
) -> List[Dict[str, Any]]:
    """
    Get all checkpoints for a workflow instance.

    Returns a list of all checkpoints created for the instance
    with their metadata and creation information.
    """
    try:
        checkpoints = await instance_service.get_instance_checkpoints(
            db=db,
            instance_id=instance_id,
            user_id=str(current_user["id"])
        )

        logger.debug(
            "Instance checkpoints retrieved successfully",
            instance_id=instance_id,
            checkpoint_count=len(checkpoints),
            user_id=current_user["id"]
        )

        return checkpoints

    except InstanceNotFoundError:
        logger.warning(
            "Instance not found for checkpoint retrieval",
            instance_id=instance_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance with ID {instance_id} not found"
        )
    except Exception as e:
        logger.error(
            "Error retrieving instance checkpoints",
            instance_id=instance_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving checkpoints"
        )
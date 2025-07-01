"""
Enterprise State Machine Workflow Engine - Workflow Endpoints

Updated REST API endpoints for workflow definition management with
authentication and RBAC integration.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.orm import Session

from app.api.deps import (
    get_database_session,
    require_workflow_create,
    require_workflow_read,
    require_workflow_update,
    require_workflow_delete,
    require_workflow_activate,
    require_workflow_stats,
    get_audit_context
)
from app.core.logging import get_logger
from app.schemas.common import PaginatedResponse
from app.schemas.workflow import (
    WorkflowCreateRequest,
    WorkflowUpdateRequest,
    WorkflowResponse,
    WorkflowListParams,
    WorkflowValidationRequest,
    WorkflowValidationResponse,
    WorkflowStatsResponse,
    WorkflowStatusEnum
)
from app.services.workflow_service import (
    WorkflowService,
    WorkflowServiceError,
    WorkflowNotFoundError,
    WorkflowValidationError
)

logger = get_logger(__name__)
router = APIRouter()

# Initialize workflow service
workflow_service = WorkflowService()


@router.post(
    "/",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Workflow",
    description="Create a new workflow definition with states, transitions, and configuration."
)
async def create_workflow(
        workflow_data: WorkflowCreateRequest = Body(..., description="Workflow creation data"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_create),
        audit_context: Dict[str, Any] = Depends(get_audit_context)
) -> WorkflowResponse:
    """
    Create a new workflow definition.

    Creates a complete workflow definition including:
    - States and their configurations
    - Transitions between states
    - Events and triggers
    - Input/output schemas
    - Retry policies and timeouts
    """
    try:
        logger.info(
            "Creating workflow via API",
            workflow_name=workflow_data.name,
            user_id=current_user["id"],
            username=current_user.get("username")
        )

        created_by = UUID(str(current_user["id"]))

        workflow = await workflow_service.create_workflow(
            db=db,
            workflow_data=workflow_data,
            created_by=created_by
        )

        logger.info(
            "Workflow created successfully via API",
            workflow_id=workflow.id,
            user_id=current_user["id"]
        )
        return workflow

    except WorkflowValidationError as e:
        logger.warning("Workflow validation failed", error=str(e), user_id=current_user["id"])
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Workflow validation failed: {str(e)}"
        )
    except WorkflowServiceError as e:
        logger.error("Workflow creation failed", error=str(e), user_id=current_user["id"])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error creating workflow",
            error=str(e),
            user_id=current_user["id"],
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating the workflow"
        )


@router.get(
    "/",
    response_model=PaginatedResponse,
    summary="List Workflows",
    description="Get a paginated list of workflow definitions with filtering and sorting options."
)
async def list_workflows(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        sort_by: Optional[str] = Query(None, description="Field to sort by"),
        sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
        search: Optional[str] = Query(None, description="Search term"),
        status: Optional[WorkflowStatusEnum] = Query(None, description="Filter by status"),
        is_template: Optional[bool] = Query(None, description="Filter by template flag"),
        tags: Optional[List[str]] = Query(None, description="Filter by tags"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_read)
) -> PaginatedResponse:
    """
    Retrieve a paginated list of workflow definitions.

    Supports filtering by:
    - Status (draft, active, inactive, etc.)
    - Template flag
    - Tags
    - Search terms in name/description

    Supports sorting by any workflow field.
    """
    try:
        params = WorkflowListParams(
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            search=search,
            status=status,
            is_template=is_template,
            tags=tags
        )

        workflows = await workflow_service.get_workflows(
            db=db,
            params=params,
            user_id=str(current_user["id"])  # Pass user ID for access filtering
        )

        logger.debug(
            "Workflows listed successfully",
            total=workflows.total,
            page=page,
            page_size=page_size,
            user_id=current_user["id"]
        )

        return workflows

    except Exception as e:
        logger.error(
            "Error listing workflows",
            error=str(e),
            user_id=current_user["id"],
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving workflows"
        )


@router.get(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    summary="Get Workflow",
    description="Retrieve a specific workflow definition by ID."
)
async def get_workflow(
        workflow_id: UUID = Path(..., description="Workflow ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_read)
) -> WorkflowResponse:
    """
    Retrieve a specific workflow definition by its ID.

    Returns the complete workflow definition including:
    - All states and their configurations
    - All transitions and conditions
    - Events and handlers
    - Execution statistics
    """
    try:
        workflow = await workflow_service.get_workflow_by_id(
            db=db,
            workflow_id=workflow_id,
            user_id=str(current_user["id"])
        )

        logger.debug(
            "Workflow retrieved successfully",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        return workflow

    except WorkflowNotFoundError:
        logger.warning(
            "Workflow not found",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID {workflow_id} not found"
        )
    except Exception as e:
        logger.error(
            "Error retrieving workflow",
            workflow_id=workflow_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the workflow"
        )


@router.put(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    summary="Update Workflow",
    description="Update an existing workflow definition."
)
async def update_workflow(
        workflow_id: UUID = Path(..., description="Workflow ID"),
        workflow_data: WorkflowUpdateRequest = Body(..., description="Workflow update data"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_update)
) -> WorkflowResponse:
    """
    Update an existing workflow definition.

    Allows partial updates of workflow properties including:
    - Name and description
    - Workflow definition (states, transitions)
    - Configuration settings
    - Status changes
    """
    try:
        logger.info(
            "Updating workflow via API",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )

        updated_by = UUID(str(current_user["id"]))

        workflow = await workflow_service.update_workflow(
            db=db,
            workflow_id=workflow_id,
            workflow_data=workflow_data,
            updated_by=updated_by,
            user_id=str(current_user["id"])
        )

        logger.info(
            "Workflow updated successfully via API",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        return workflow

    except WorkflowNotFoundError:
        logger.warning(
            "Workflow not found for update",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID {workflow_id} not found"
        )
    except WorkflowValidationError as e:
        logger.warning(
            "Workflow validation failed during update",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Workflow validation failed: {str(e)}"
        )
    except WorkflowServiceError as e:
        logger.error("Workflow update failed", error=str(e), user_id=current_user["id"])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error updating workflow",
            workflow_id=workflow_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating the workflow"
        )


@router.delete(
    "/{workflow_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Workflow",
    description="Delete a workflow definition (soft delete by default)."
)
async def delete_workflow(
        workflow_id: UUID = Path(..., description="Workflow ID"),
        hard_delete: bool = Query(False, description="Perform hard delete"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_delete)
) -> None:
    """
    Delete a workflow definition.

    By default, performs a soft delete (marks as deleted but preserves data).
    Use hard_delete=true to permanently remove the workflow.

    Note: Workflows with running instances cannot be deleted.
    """
    try:
        logger.info(
            "Deleting workflow via API",
            workflow_id=workflow_id,
            hard_delete=hard_delete,
            user_id=current_user["id"]
        )

        deleted_by = UUID(str(current_user["id"]))

        await workflow_service.delete_workflow(
            db=db,
            workflow_id=workflow_id,
            hard_delete=hard_delete,
            deleted_by=deleted_by,
            user_id=str(current_user["id"])
        )

        logger.info(
            "Workflow deleted successfully via API",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )

    except WorkflowNotFoundError:
        logger.warning(
            "Workflow not found for deletion",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID {workflow_id} not found"
        )
    except WorkflowServiceError as e:
        logger.error(
            "Workflow deletion failed",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error deleting workflow",
            workflow_id=workflow_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting the workflow"
        )


@router.post(
    "/validate",
    response_model=WorkflowValidationResponse,
    summary="Validate Workflow",
    description="Validate a workflow definition without creating it."
)
async def validate_workflow(
        validation_request: WorkflowValidationRequest = Body(..., description="Workflow validation request"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_read)  # Only need read permission for validation
) -> WorkflowValidationResponse:
    """
    Validate a workflow definition.

    Performs comprehensive validation including:
    - Required field validation
    - State and transition consistency
    - Reachability analysis (in strict mode)
    - Performance recommendations
    """
    try:
        logger.debug(
            "Validating workflow definition via API",
            user_id=current_user["id"]
        )

        validation_result = await workflow_service.validate_workflow_definition(
            definition=validation_request.definition.model_dump(),
            strict=validation_request.strict
        )

        logger.debug(
            "Workflow validation completed",
            is_valid=validation_result.is_valid,
            error_count=len(validation_result.errors),
            user_id=current_user["id"]
        )

        return validation_result

    except Exception as e:
        logger.error(
            "Error during workflow validation",
            error=str(e),
            user_id=current_user["id"],
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during workflow validation"
        )


@router.post(
    "/{workflow_id}/activate",
    response_model=WorkflowResponse,
    summary="Activate Workflow",
    description="Activate a workflow definition to make it available for execution."
)
async def activate_workflow(
        workflow_id: UUID = Path(..., description="Workflow ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_activate)
) -> WorkflowResponse:
    """
    Activate a workflow definition.

    Changes the workflow status to 'active', making it available for
    creating and executing workflow instances.
    """
    try:
        logger.info(
            "Activating workflow via API",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )

        activated_by = UUID(str(current_user["id"]))

        workflow = await workflow_service.activate_workflow(
            db=db,
            workflow_id=workflow_id,
            activated_by=activated_by,
            user_id=str(current_user["id"])
        )

        logger.info(
            "Workflow activated successfully via API",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        return workflow

    except WorkflowNotFoundError:
        logger.warning(
            "Workflow not found for activation",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID {workflow_id} not found"
        )
    except WorkflowServiceError as e:
        logger.error(
            "Workflow activation failed",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error activating workflow",
            workflow_id=workflow_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while activating the workflow"
        )


@router.post(
    "/{workflow_id}/deactivate",
    response_model=WorkflowResponse,
    summary="Deactivate Workflow",
    description="Deactivate a workflow definition to prevent new instances."
)
async def deactivate_workflow(
        workflow_id: UUID = Path(..., description="Workflow ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_activate)  # Same permission as activate
) -> WorkflowResponse:
    """
    Deactivate a workflow definition.

    Changes the workflow status to 'inactive', preventing the creation
    of new workflow instances while preserving existing ones.
    """
    try:
        logger.info(
            "Deactivating workflow via API",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )

        deactivated_by = UUID(str(current_user["id"]))

        workflow = await workflow_service.deactivate_workflow(
            db=db,
            workflow_id=workflow_id,
            deactivated_by=deactivated_by,
            user_id=str(current_user["id"])
        )

        logger.info(
            "Workflow deactivated successfully via API",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        return workflow

    except WorkflowNotFoundError:
        logger.warning(
            "Workflow not found for deactivation",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID {workflow_id} not found"
        )
    except WorkflowServiceError as e:
        logger.error(
            "Workflow deactivation failed",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error deactivating workflow",
            workflow_id=workflow_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deactivating the workflow"
        )


@router.get(
    "/{workflow_id}/instances",
    response_model=PaginatedResponse,
    summary="Get Workflow Instances",
    description="Get all instances for a specific workflow definition."
)
async def get_workflow_instances(
        workflow_id: UUID = Path(..., description="Workflow ID"),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        status: Optional[str] = Query(None, description="Filter by instance status"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_read)
) -> PaginatedResponse:
    """
    Get all instances for a specific workflow definition.

    Returns a paginated list of workflow instances created from
    the specified workflow definition.
    """
    try:
        # Import here to avoid circular import
        from app.services.instance_service import InstanceService
        from app.schemas.instance import InstanceListParams, InstanceStatusEnum

        instance_service = InstanceService()

        # Parse status if provided
        status_enum = None
        if status:
            try:
                status_enum = InstanceStatusEnum(status)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status}"
                )

        params = InstanceListParams(
            workflow_definition_id=workflow_id,
            page=page,
            page_size=page_size,
            status=status_enum
        )

        instances = await instance_service.get_instances(
            db=db,
            params=params,
            user_id=str(current_user["id"])
        )

        logger.debug(
            "Workflow instances retrieved successfully",
            workflow_id=workflow_id,
            total=instances.total,
            user_id=current_user["id"]
        )

        return instances

    except Exception as e:
        logger.error(
            "Error retrieving workflow instances",
            workflow_id=workflow_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving workflow instances"
        )


@router.get(
    "/stats",
    response_model=WorkflowStatsResponse,
    summary="Get Workflow Statistics",
    description="Get comprehensive statistics about workflows and their execution."
)
async def get_workflow_statistics(
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_stats)
) -> WorkflowStatsResponse:
    """
    Get comprehensive workflow statistics.

    Returns statistics including:
    - Total workflow counts by status
    - Instance execution statistics
    - Success rates and performance metrics
    """
    try:
        logger.debug("Getting workflow statistics via API", user_id=current_user["id"])

        stats = await workflow_service.get_workflow_stats(
            db=db,
            user_id=str(current_user["id"])
        )

        logger.debug("Workflow statistics retrieved successfully", user_id=current_user["id"])
        return stats

    except Exception as e:
        logger.error(
            "Error getting workflow statistics",
            error=str(e),
            user_id=current_user["id"],
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving workflow statistics"
        )


@router.get(
    "/{workflow_id}/definition",
    response_model=Dict[str, Any],
    summary="Get Workflow Definition",
    description="Get only the workflow definition (states, transitions, events)."
)
async def get_workflow_definition(
        workflow_id: UUID = Path(..., description="Workflow ID"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_read)
) -> Dict[str, Any]:
    """
    Get only the workflow definition structure.

    Returns just the definition part of the workflow (states, transitions, events)
    without metadata, statistics, or configuration.
    """
    try:
        workflow = await workflow_service.get_workflow_by_id(
            db=db,
            workflow_id=workflow_id,
            user_id=str(current_user["id"])
        )

        logger.debug(
            "Workflow definition retrieved successfully",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        return workflow.definition

    except WorkflowNotFoundError:
        logger.warning(
            "Workflow not found",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID {workflow_id} not found"
        )
    except Exception as e:
        logger.error(
            "Error retrieving workflow definition",
            workflow_id=workflow_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the workflow definition"
        )


@router.post(
    "/{workflow_id}/clone",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Clone Workflow",
    description="Create a copy of an existing workflow with a new version."
)
async def clone_workflow(
        workflow_id: UUID = Path(..., description="Workflow ID to clone"),
        new_name: Optional[str] = Query(None, description="New workflow name"),
        new_version: str = Query("1.0.0", description="New workflow version"),
        db: Session = Depends(get_database_session),
        current_user: Dict[str, Any] = Depends(require_workflow_create)
) -> WorkflowResponse:
    """
    Clone an existing workflow.

    Creates a copy of the specified workflow with a new version.
    Optionally allows changing the name.
    """
    try:
        logger.info(
            "Cloning workflow via API",
            workflow_id=workflow_id,
            new_version=new_version,
            user_id=current_user["id"]
        )

        # Get the original workflow
        original_workflow = await workflow_service.get_workflow_by_id(
            db=db,
            workflow_id=workflow_id,
            user_id=str(current_user["id"])
        )

        # Create clone request
        clone_data = WorkflowCreateRequest(
            name=new_name or f"{original_workflow.name} (Copy)",
            description=f"Clone of {original_workflow.name}",
            version=new_version,
            definition=original_workflow.definition,
            input_schema=original_workflow.input_schema,
            output_schema=original_workflow.output_schema,
            timeout_seconds=original_workflow.timeout_seconds,
            max_retries=original_workflow.max_retries,
            retry_delay_seconds=original_workflow.retry_delay_seconds,
            tags=original_workflow.tags,
            metadata={
                **original_workflow.metadata,
                "cloned_from": str(workflow_id),
                "cloned_at": datetime.utcnow().isoformat()
            },
            is_template=original_workflow.is_template
        )

        created_by = UUID(str(current_user["id"]))

        cloned_workflow = await workflow_service.create_workflow(
            db=db,
            workflow_data=clone_data,
            created_by=created_by
        )

        logger.info(
            "Workflow cloned successfully via API",
            original_id=workflow_id,
            cloned_id=cloned_workflow.id,
            user_id=current_user["id"]
        )

        return cloned_workflow

    except WorkflowNotFoundError:
        logger.warning(
            "Workflow not found for cloning",
            workflow_id=workflow_id,
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID {workflow_id} not found"
        )
    except WorkflowServiceError as e:
        logger.error(
            "Workflow cloning failed",
            error=str(e),
            user_id=current_user["id"]
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error cloning workflow",
            workflow_id=workflow_id,
            user_id=current_user["id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while cloning the workflow"
        )

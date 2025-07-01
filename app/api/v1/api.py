"""
Enterprise State Machine Workflow Engine - API Router

This module sets up the main API router and includes all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import health


# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health & Monitoring"]
)

# Note: Additional routers will be added as we implement more features
# api_router.include_router(
#     workflows.router,
#     prefix="/workflows",
#     tags=["Workflows"]
# )
# api_router.include_router(
#     instances.router,
#     prefix="/instances",
#     tags=["Workflow Instances"]
# )


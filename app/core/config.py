"""
Enterprise State Machine Workflow Engine - Core Configuration

Updated configuration with user management service integration settings.
"""

import secrets
from typing import Any, Dict, List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import (
    AnyHttpUrl,
    PostgresDsn,
    RedisDsn,
    field_validator,
    model_validator
)

from pydantic_core.core_schema import FieldValidationInfo

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class handles all configuration for the workflow engine,
    including database connections, security settings, user management integration,
    and external services.
    """
    # ===== APPLICATION SETTINGS =====
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "GVD Workflow Engine"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Automate your flow"

    # Environment configuration
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = True

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8004
    RELOAD: bool = True  # Auto-reload for development

    # ===== SECURITY SETTINGS =====
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days

    # Password hashing
    PWD_CONTEXT_SCHEMES: List[str] = ["bcrypt"]
    PWD_CONTEXT_DEPRECATED: str = "auto"

    # CORS settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Parse CORS origins from environment variable or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # ===== USER MANAGEMENT SERVICE INTEGRATION =====
    # User Service Configuration
    USER_SERVICE_URL: str = "http://localhost:8001"
    USER_SERVICE_TIMEOUT: int = 30
    USER_SERVICE_RETRY_ATTEMPTS: int = 3
    USER_SERVICE_RETRY_DELAY: float = 1.0

    # Authentication Configuration
    AUTH_ENABLED: bool = True
    AUTH_TOKEN_HEADER: str = "Authorization"
    AUTH_TOKEN_PREFIX: str = "Bearer"
    AUTH_CACHE_TTL: int = 300  # 5 minutes

    # RBAC Configuration
    RBAC_ENABLED: bool = True
    RBAC_STRICT_MODE: bool = False  # If True, deny access when permission check fails
    RBAC_CACHE_TTL: int = 300  # 5 minutes

    # Permission Definitions
    WORKFLOW_PERMISSIONS: Dict[str, str] = {
        "create": "workflow:create",
        "read": "workflow:read",
        "update": "workflow:update",
        "delete": "workflow:delete",
        "activate": "workflow:activate",
        "deactivate": "workflow:deactivate",
        "clone": "workflow:clone",
        "export": "workflow:export",
        "import": "workflow:import",
        "validate": "workflow:validate",
        "view_stats": "workflow:view_stats"
    }

    INSTANCE_PERMISSIONS: Dict[str, str] = {
        "create": "instance:create",
        "read": "instance:read",
        "update": "instance:update",
        "delete": "instance:delete",
        "start": "instance:start",
        "pause": "instance:pause",
        "resume": "instance:resume",
        "cancel": "instance:cancel",
        "retry": "instance:retry",
        "send_events": "instance:send_events",
        "view_history": "instance:view_history",
        "manage_variables": "instance:manage_variables",
        "view_metrics": "instance:view_metrics",
        "bulk_operations": "instance:bulk_operations",
        "view_stats": "instance:view_stats"
    }

    SYSTEM_PERMISSIONS: Dict[str, str] = {
        "admin": "system:admin",
        "view_health": "system:view_health",
        "view_metrics": "system:view_metrics",
        "manage_system": "system:manage"
    }

    # ===== DATABASE SETTINGS =====
    # PostgreSQL Database Configuration
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "Hanu%400542"
    POSTGRES_DB: str = "psim_automation"
    POSTGRES_PORT: str = "5433"

    # SQLAlchemy Database URL
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @model_validator(mode="after")
    def assemble_db_connection(self) -> "Settings":
        """Construct PostgreSQL database URL from components."""
        if not self.SQLALCHEMY_DATABASE_URI:
            self.SQLALCHEMY_DATABASE_URI = PostgresDsn.build(
                scheme="postgresql",
                username=self.POSTGRES_USER,
                password=self.POSTGRES_PASSWORD,
                host=self.POSTGRES_SERVER,
                port=int(self.POSTGRES_PORT),
                path=self.POSTGRES_DB,
            )
        return self

    # Database pool settings
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600  # 1 hour
    DB_POOL_PRE_PING: bool = True

    # ===== REDIS SETTINGS =====
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL: Optional[RedisDsn] = None

    @model_validator(mode="after")
    def assemble_redis_connection(self) -> "Settings":
        """Construct Redis URL from components."""
        if not self.REDIS_URL:
            auth_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
            self.REDIS_URL = RedisDsn(
                f"redis://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            )
        return self

    # Redis connection pool settings
    REDIS_POOL_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5

    # ===== CELERY SETTINGS =====
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""

    @model_validator(mode="after")
    def set_celery_urls(self) -> "Settings":
        """Set Celery broker and result backend URLs."""
        if not self.CELERY_BROKER_URL:
            self.CELERY_BROKER_URL = str(self.REDIS_URL)
        if not self.CELERY_RESULT_BACKEND:
            self.CELERY_RESULT_BACKEND = str(self.REDIS_URL)
        return self

    # Celery task settings
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_ENABLE_UTC: bool = True
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 30 * 60  # 30 minutes
    CELERY_TASK_SOFT_TIME_LIMIT: int = 25 * 60  # 25 minutes

    # ===== LOGGING SETTINGS =====
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or console
    LOG_FILE: Optional[str] = None
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"

    # Structured logging configuration
    STRUCTLOG_PROCESSORS: List[str] = [
        "structlog.stdlib.filter_by_level",
        "structlog.stdlib.add_logger_name",
        "structlog.stdlib.add_log_level",
        "structlog.stdlib.PositionalArgumentsFormatter",
        "structlog.processors.TimeStamper",
        "structlog.processors.StackInfoRenderer",
        "structlog.processors.format_exc_info",
    ]

    # ===== MONITORING SETTINGS =====
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    METRICS_PATH: str = "/metrics"

    # Health check settings
    HEALTH_CHECK_INTERVAL: int = 30  # seconds
    HEALTH_CHECK_TIMEOUT: int = 5  # seconds

    # ===== WORKFLOW ENGINE SETTINGS =====
    # Maximum number of concurrent workflow instances
    MAX_CONCURRENT_WORKFLOWS: int = 1000

    # Default timeouts (in seconds)
    DEFAULT_TASK_TIMEOUT: int = 300  # 5 minutes
    DEFAULT_WORKFLOW_TIMEOUT: int = 3600  # 1 hour

    # Retry settings
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: int = 60  # seconds

    # State machine configuration
    MAX_STATE_TRANSITIONS: int = 1000  # Prevent infinite loops
    STATE_PERSISTENCE_INTERVAL: int = 10  # seconds

    # ===== RATE LIMITING =====
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # ===== EMAIL SETTINGS (for notifications) =====
    EMAILS_ENABLED: bool = False
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None

    # ===== FIRST SUPERUSER =====
    FIRST_SUPERUSER: str = "admin@geniusvision.in"
    FIRST_SUPERUSER_PASSWORD: str = "Gvd@6001"

    # ===== TESTING SETTINGS =====
    TESTING: bool = False
    TEST_DATABASE_URL: Optional[str] = None

    # ===== SERVICE DISCOVERY & COMMUNICATION =====
    SERVICE_DISCOVERY_ENABLED: bool = False
    SERVICE_REGISTRY_URL: Optional[str] = None
    CIRCUIT_BREAKER_ENABLED: bool = True
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 60  # seconds

    # Request timeout and retry settings
    DEFAULT_HTTP_TIMEOUT: int = 30
    DEFAULT_HTTP_RETRIES: int = 3
    DEFAULT_HTTP_BACKOFF_FACTOR: float = 0.3

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.TESTING

    @property
    def auth_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return self.AUTH_ENABLED and not self.is_testing

    @property
    def rbac_enabled(self) -> bool:
        """Check if RBAC is enabled."""
        return self.RBAC_ENABLED and self.auth_enabled

    def get_database_url(self) -> str:
        """Get the appropriate database URL based on environment."""
        if self.is_testing and self.TEST_DATABASE_URL:
            return self.TEST_DATABASE_URL
        return str(self.SQLALCHEMY_DATABASE_URI)

    def get_permission(self, resource: str, action: str) -> Optional[str]:
        """
        Get permission string for resource and action.

        Args:
            resource: Resource type (workflow, instance, system)
            action: Action to perform

        Returns:
            Permission string or None if not found
        """
        permission_maps = {
            "workflow": self.WORKFLOW_PERMISSIONS,
            "instance": self.INSTANCE_PERMISSIONS,
            "system": self.SYSTEM_PERMISSIONS
        }

        resource_permissions = permission_maps.get(resource, {})
        return resource_permissions.get(action)

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency function to get settings instance.

    This function can be used as a FastAPI dependency to inject
    settings into route handlers and other functions.

    Returns:
        Settings: The global settings instance
    """
    return settings


# Export commonly used settings for convenience
DATABASE_URL = settings.get_database_url()
REDIS_URL = str(settings.REDIS_URL)
SECRET_KEY = settings.SECRET_KEY
API_V1_STR = settings.API_V1_STR
USER_SERVICE_URL = settings.USER_SERVICE_URL
"""
Enterprise State Machine Workflow Engine - Database Configuration

This module handles SQLAlchemy configuration, session management, and
database connection setup with proper connection pooling and error handling.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, event, Engine,MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from structlog import get_logger

from app.core.config import settings

# Configure logger
logger = get_logger(__name__)

# SQLAlchemy naming convention for constraints
# This ensures consistent naming across different databases
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

# Create metadata with naming convention
metadata = MetaData(naming_convention=NAMING_CONVENTION)

# Create declarative base
Base = declarative_base(metadata=metadata)


class DatabaseManager:
    """
    Database manager class that handles engine creation, session management,
    and connection health monitoring.
    """

    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database manager.

        Args:
            database_url: SQLAlchemy database URL
            echo: Whether to echo SQL statements (for debugging)
        """
        self.database_url = database_url
        self.echo = echo
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    def create_engine(self) -> Engine:
        """
        Create SQLAlchemy engine with optimized connection pooling.

        Returns:
            Engine: Configured SQLAlchemy engine
        """
        if self._engine is not None:
            return self._engine

        engine_kwargs = {
            "echo": self.echo,
            "future": True,  # Use SQLAlchemy 2.0 style
            "pool_pre_ping": settings.DB_POOL_PRE_PING,
            "pool_recycle": settings.DB_POOL_RECYCLE,
        }

        # Configure connection pooling based on environment
        if settings.is_testing:
            # Use in-memory SQLite for testing with StaticPool
            engine_kwargs.update({
                "poolclass": StaticPool,
                "connect_args": {
                    "check_same_thread": False,
                }
            })
        else:
            # Production/development settings with QueuePool
            engine_kwargs.update({
                "poolclass": QueuePool,
                "pool_size": settings.DB_POOL_SIZE,
                "max_overflow": settings.DB_MAX_OVERFLOW,
                "pool_timeout": settings.DB_POOL_TIMEOUT,
            })

        try:
            self._engine = create_engine(self.database_url, **engine_kwargs)

            # Add event listeners for connection management
            self._setup_engine_events(self._engine)

            logger.info(
                "Database engine created successfully",
                database_url=self.database_url,
                pool_size=engine_kwargs.get("pool_size"),
                max_overflow=engine_kwargs.get("max_overflow")
            )

            return self._engine

        except Exception as e:
            logger.error(
                "Failed to create database engine",
                error=str(e),
                database_url=self.database_url
            )
            raise

    def _setup_engine_events(self, engine: Engine) -> None:
        """
        Setup SQLAlchemy engine event listeners for monitoring and optimization.

        Args:
            engine: SQLAlchemy engine instance
        """

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance (if using SQLite)."""
            if "sqlite" in str(engine.url):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()

        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log when a connection is checked out from the pool."""
            logger.debug("Connection checked out from pool")

        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log when a connection is returned to the pool."""
            logger.debug("Connection returned to pool")

        @event.listens_for(engine, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidation."""
            logger.warning(
                "Connection invalidated",
                exception=str(exception) if exception else None
            )

    def create_session_factory(self) -> sessionmaker:
        """
        Create session factory for database sessions.

        Returns:
            sessionmaker: Configured session factory
        """
        if self._session_factory is not None:
            return self._session_factory

        engine = self.create_engine()

        self._session_factory = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

        logger.info("Database session factory created")
        return self._session_factory

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            Session: SQLAlchemy session instance
        """
        session_factory = self.create_session_factory()
        return session_factory()

    def health_check(self) -> bool:
        """
        Perform database health check.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            with self.get_session() as session:
                # Simple query to test connection
                session.execute(text("SELECT 1"))
                session.commit()
                return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

    def close(self) -> None:
        """Close database engine and cleanup resources."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database engine closed")


# Global database manager instance
db_manager = DatabaseManager(
    database_url=settings.get_database_url(),
    echo=settings.DEBUG and settings.is_development
)

# Convenience aliases
engine = db_manager.create_engine()
SessionLocal = db_manager.create_session_factory()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for getting database sessions.

    This function provides a database session that automatically
    handles rollback on exceptions and cleanup on completion.

    Yields:
        Session: SQLAlchemy session instance

    Example:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        logger.error("Database session error", error=str(e))
        session.rollback()
        raise
    except Exception as e:
        logger.error("Unexpected error in database session", error=str(e))
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions outside of FastAPI.

    This provides a database session that automatically handles
    commit/rollback and cleanup.

    Yields:
        Session: SQLAlchemy session instance

    Example:
        with get_db_context() as db:
            user = db.query(User).first()
            # Session is automatically committed and closed
    """
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        logger.error("Database context error", error=str(e))
        session.rollback()
        raise
    except Exception as e:
        logger.error("Unexpected error in database context", error=str(e))
        session.rollback()
        raise
    finally:
        session.close()


def create_tables() -> None:
    """
    Create all database tables.

    This function should be called during application startup
    to ensure all tables exist.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise


def drop_tables() -> None:
    """
    Drop all database tables.

    WARNING: This will delete all data! Use only for testing
    or complete database reset.
    """
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error("Failed to drop database tables", error=str(e))
        raise


def check_database_health() -> bool:
    """
    Check database connectivity and health.

    Returns:
        bool: True if database is healthy, False otherwise
    """
    return db_manager.health_check()


def get_database_info() -> dict:
    """
    Get database connection information.

    Returns:
        dict: Database connection information
    """
    return {
        "database_url": settings.database_url,
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_timeout": settings.DB_POOL_TIMEOUT,
        "pool_recycle": settings.DB_POOL_RECYCLE,
        "is_healthy": check_database_health()
    }
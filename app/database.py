from collections.abc import AsyncGenerator
from typing import cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


engine: AsyncEngine | None = None
session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_database(database_url: str) -> None:
    global engine
    global session_factory

    engine = create_async_engine(database_url, pool_pre_ping=True, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    from app.models import db  # noqa: F401

    async with engine.begin() as conn:
        if conn.engine.url.drivername.startswith("postgresql"):
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)


async def close_database() -> None:
    global engine
    global session_factory

    if engine is not None:
        await engine.dispose()
    engine = None
    session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if session_factory is None:
        raise RuntimeError("Database session factory is not initialized.")
    factory = cast(async_sessionmaker[AsyncSession], session_factory)
    async with factory() as session:
        yield session


async def check_database_health() -> bool:
    if session_factory is None:
        return False
    factory = cast(async_sessionmaker[AsyncSession], session_factory)
    try:
        async with factory() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar_one() == 1
    except Exception:
        return False

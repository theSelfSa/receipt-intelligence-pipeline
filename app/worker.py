from __future__ import annotations

import asyncio

from celery import Celery
from celery.schedules import crontab

import app.database as database
from app.config import get_settings
from app.services.error_analysis import ErrorAnalysisService
from app.services.pipeline import process_receipt_pipeline
from app.services.retraining import RetrainingService

settings = get_settings()

celery_app = Celery(
    "receipt_intelligence_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "nightly-error-analysis": {
            "task": "app.worker.run_error_analysis",
            "schedule": crontab(hour=2, minute=0),
        },
        "monthly-retraining-run": {
            "task": "app.worker.run_scheduled_retraining",
            "schedule": crontab(hour=3, minute=0, day_of_month=1),
        },
    },
)


@celery_app.task(name="app.worker.process_receipt")
def process_receipt_task(receipt_id: int) -> dict[str, str | int]:
    return asyncio.run(_process_receipt_task(receipt_id))


@celery_app.task(name="app.worker.run_error_analysis")
def run_error_analysis_task() -> dict[str, int]:
    return asyncio.run(_run_error_analysis_task())


@celery_app.task(name="app.worker.execute_retraining")
def execute_retraining_task(run_id: int) -> dict[str, str | int]:
    return asyncio.run(_execute_retraining_task(run_id))


@celery_app.task(name="app.worker.run_scheduled_retraining")
def run_scheduled_retraining_task() -> dict[str, str | int]:
    return asyncio.run(_run_scheduled_retraining_task())


async def _process_receipt_task(receipt_id: int) -> dict[str, str | int]:
    await database.init_database(settings.database_url)
    try:
        if database.session_factory is None:
            raise RuntimeError("Database session factory not initialized.")

        async with database.session_factory() as session:
            receipt = await process_receipt_pipeline(
                receipt_id=receipt_id,
                session=session,
                settings=settings,
            )
            return {
                "receipt_id": receipt.id,
                "status": receipt.processing_status,
            }
    finally:
        await database.close_database()


async def _run_error_analysis_task() -> dict[str, int]:
    await database.init_database(settings.database_url)
    try:
        if database.session_factory is None:
            raise RuntimeError("Database session factory not initialized.")

        analyzer = ErrorAnalysisService(settings)
        async with database.session_factory() as session:
            inserted = await analyzer.analyze_recent_corrections(session)
            return {"inserted_patterns": inserted}
    finally:
        await database.close_database()


async def _execute_retraining_task(run_id: int) -> dict[str, str | int]:
    await database.init_database(settings.database_url)
    try:
        if database.session_factory is None:
            raise RuntimeError("Database session factory not initialized.")

        retraining = RetrainingService()
        async with database.session_factory() as session:
            run = await retraining.execute_run(session, run_id=run_id)
            return {"run_id": run.id, "status": run.status or "unknown"}
    finally:
        await database.close_database()


async def _run_scheduled_retraining_task() -> dict[str, str | int]:
    await database.init_database(settings.database_url)
    try:
        if database.session_factory is None:
            raise RuntimeError("Database session factory not initialized.")

        retraining = RetrainingService()
        async with database.session_factory() as session:
            run = await retraining.create_run(session, trigger="scheduled")
            completed = await retraining.execute_run(session, run_id=run.id)
            return {"run_id": completed.id, "status": completed.status or "unknown"}
    finally:
        await database.close_database()

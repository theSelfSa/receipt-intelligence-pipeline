from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.api import (
    RetrainTriggerResponse,
    RetrainingRunResponse,
    RetrainingRunsListResponse,
)
from app.models.db import RetrainingRunDB
from app.services.retraining import RetrainingService
from app.worker import execute_retraining_task

router = APIRouter(prefix="/retrain", tags=["retrain"])


@router.post("/trigger", response_model=RetrainTriggerResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_retraining(
    trigger: str = Query(default="manual"),
    session: AsyncSession = Depends(get_session),
) -> RetrainTriggerResponse:
    service = RetrainingService()
    run = await service.create_run(session, trigger=trigger)
    task = execute_retraining_task.delay(run.id)
    return RetrainTriggerResponse(run_id=run.id, task_id=task.id, status="queued")


@router.get("/runs", response_model=RetrainingRunsListResponse)
async def list_retraining_runs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> RetrainingRunsListResponse:
    query = (
        select(RetrainingRunDB)
        .order_by(RetrainingRunDB.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    count_query = select(func.count()).select_from(RetrainingRunDB)

    rows = (await session.execute(query)).scalars().all()
    total = int((await session.execute(count_query)).scalar() or 0)
    items = [_to_run_response(row) for row in rows]
    return RetrainingRunsListResponse(items=items, total=total)


@router.get("/runs/{run_id}", response_model=RetrainingRunResponse)
async def get_retraining_run(
    run_id: int,
    session: AsyncSession = Depends(get_session),
) -> RetrainingRunResponse:
    run = await session.get(RetrainingRunDB, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Retraining run not found.")
    return _to_run_response(run)


def _to_run_response(run: RetrainingRunDB) -> RetrainingRunResponse:
    return RetrainingRunResponse(
        id=run.id,
        trigger=run.trigger,
        training_samples=run.training_samples,
        validation_accuracy=run.validation_accuracy,
        baseline_accuracy=run.baseline_accuracy,
        improvement=run.improvement,
        model_path=run.model_path,
        wandb_run_id=run.wandb_run_id,
        status=run.status,
        created_at=run.created_at,
    )

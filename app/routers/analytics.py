from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.api import (
    AccuracyCategoryPoint,
    AccuracyResponse,
    AnalyzeTriggerResponse,
    ConfidenceCalibrationPoint,
    ConfidenceCalibrationResponse,
    CostSummaryResponse,
    ErrorPatternResponse,
    ErrorPatternsListResponse,
)
from app.models.db import ErrorPatternDB, ReceiptDB, ReviewQueueDB
from app.worker import run_error_analysis_task

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/errors", response_model=ErrorPatternsListResponse)
async def list_error_patterns(session: AsyncSession = Depends(get_session)) -> ErrorPatternsListResponse:
    query = select(ErrorPatternDB).order_by(ErrorPatternDB.occurrence_count.desc(), ErrorPatternDB.created_at.desc())
    rows = (await session.execute(query)).scalars().all()
    items = [
        ErrorPatternResponse(
            id=row.id,
            error_type=row.error_type,
            description=row.description,
            occurrence_count=row.occurrence_count,
            suggested_prompt_fix=row.suggested_prompt_fix,
            effectiveness_score=row.effectiveness_score,
            created_at=row.created_at,
        )
        for row in rows
    ]
    return ErrorPatternsListResponse(items=items, total=len(items))


@router.post("/analyze", response_model=AnalyzeTriggerResponse)
async def trigger_error_analysis() -> AnalyzeTriggerResponse:
    task = run_error_analysis_task.delay()
    return AnalyzeTriggerResponse(task_id=task.id, status="queued")


@router.get("/accuracy", response_model=AccuracyResponse)
async def accuracy_breakdown(
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> AccuracyResponse:
    receipt_query = select(ReceiptDB.id, ReceiptDB.merchant_category)
    if start_date is not None:
        receipt_query = receipt_query.where(ReceiptDB.receipt_date >= start_date)
    if end_date is not None:
        receipt_query = receipt_query.where(ReceiptDB.receipt_date <= end_date)
    receipt_rows = (await session.execute(receipt_query)).all()

    receipt_ids = [int(row[0]) for row in receipt_rows]
    corrected_ids = await _corrected_receipt_ids(session, receipt_ids)

    grouped: dict[str | None, dict[str, int]] = {}
    for receipt_id, merchant_category in receipt_rows:
        key = merchant_category
        if key not in grouped:
            grouped[key] = {"receipts": 0, "corrected": 0}
        grouped[key]["receipts"] += 1
        if int(receipt_id) in corrected_ids:
            grouped[key]["corrected"] += 1

    points = [
        AccuracyCategoryPoint(
            merchant_category=category,
            receipts=values["receipts"],
            corrected=values["corrected"],
            estimated_accuracy=(
                1 - (values["corrected"] / values["receipts"])
                if values["receipts"] > 0
                else 0.0
            ),
        )
        for category, values in grouped.items()
    ]
    return AccuracyResponse(points=points)


@router.get("/confidence", response_model=ConfidenceCalibrationResponse)
async def confidence_calibration(
    session: AsyncSession = Depends(get_session),
) -> ConfidenceCalibrationResponse:
    receipt_query = select(ReceiptDB.id, ReceiptDB.confidence_level).where(ReceiptDB.confidence_level.is_not(None))
    receipt_rows = (await session.execute(receipt_query)).all()

    receipt_ids = [int(row[0]) for row in receipt_rows]
    corrected_ids = await _corrected_receipt_ids(session, receipt_ids)

    grouped: dict[str, dict[str, int]] = {}
    for receipt_id, confidence_level in receipt_rows:
        level = str(confidence_level)
        if level not in grouped:
            grouped[level] = {"receipts": 0, "corrected": 0}
        grouped[level]["receipts"] += 1
        if int(receipt_id) in corrected_ids:
            grouped[level]["corrected"] += 1

    points = [
        ConfidenceCalibrationPoint(
            confidence_level=level,
            receipts=values["receipts"],
            corrected=values["corrected"],
            estimated_accuracy=(
                1 - (values["corrected"] / values["receipts"])
                if values["receipts"] > 0
                else 0.0
            ),
        )
        for level, values in grouped.items()
    ]
    return ConfidenceCalibrationResponse(points=points)


@router.get("/cost", response_model=CostSummaryResponse)
async def cost_summary(session: AsyncSession = Depends(get_session)) -> CostSummaryResponse:
    rows = (
        await session.execute(
            select(ReceiptDB.raw_ocr_text).where(ReceiptDB.processing_status.in_(("completed", "partial_review", "review_required")))
        )
    ).all()

    receipts_processed = len(rows)
    estimated_tokens_total = sum(max(0, len((row[0] or "")) // 4) for row in rows)
    estimated_cost_total_usd = estimated_tokens_total * 0.000005
    estimated_cost_per_receipt_usd = (
        estimated_cost_total_usd / receipts_processed
        if receipts_processed > 0
        else 0.0
    )

    return CostSummaryResponse(
        receipts_processed=receipts_processed,
        estimated_tokens_total=estimated_tokens_total,
        estimated_cost_total_usd=estimated_cost_total_usd,
        estimated_cost_per_receipt_usd=estimated_cost_per_receipt_usd,
    )


async def _corrected_receipt_ids(session: AsyncSession, receipt_ids: list[int]) -> set[int]:
    if not receipt_ids:
        return set()
    rows = (
        await session.execute(
            select(ReviewQueueDB.receipt_id).where(
                ReviewQueueDB.status == "corrected",
                ReviewQueueDB.receipt_id.in_(receipt_ids),
            )
        )
    ).all()
    return {int(row[0]) for row in rows if row[0] is not None}

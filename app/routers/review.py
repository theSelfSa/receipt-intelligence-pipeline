import json
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.database import get_session
from app.models.api import (
    ReviewActionResponse,
    ReviewCorrectionRequest,
    ReviewDetailResponse,
    ReviewQueueItemResponse,
    ReviewQueueListResponse,
    ReviewStatsResponse,
)
from app.models.db import ErrorPatternDB, LineItemDB, ReceiptDB, RetrainingRunDB, ReviewQueueDB
from app.models.schemas import ConfidenceLevel, LineItem, Receipt
from app.worker import execute_retraining_task

router = APIRouter(prefix="/review", tags=["review"])


@router.get("/queue", response_model=ReviewQueueListResponse)
async def list_review_queue(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> ReviewQueueListResponse:
    query = (
        select(ReviewQueueDB)
        .where(ReviewQueueDB.status == "pending")
        .order_by(ReviewQueueDB.created_at.asc())
    )
    count_query = (
        select(func.count()).select_from(ReviewQueueDB).where(ReviewQueueDB.status == "pending")
    )
    query = query.offset((page - 1) * page_size).limit(page_size)

    rows = (await session.execute(query)).scalars().all()
    total = int((await session.execute(count_query)).scalar() or 0)
    items = [_to_review_item_response(row) for row in rows]
    return ReviewQueueListResponse(items=items, total=total)


@router.get("/{review_id:int}", response_model=ReviewDetailResponse)
async def get_review_item(
    review_id: int,
    session: AsyncSession = Depends(get_session),
) -> ReviewDetailResponse:
    review_query = (
        select(ReviewQueueDB)
        .where(ReviewQueueDB.id == review_id)
    )
    review = (await session.execute(review_query)).scalar_one_or_none()
    if review is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review item not found.")

    receipt: ReceiptDB | None = None
    if review.receipt_id is not None:
        receipt_query = (
            select(ReceiptDB)
            .where(ReceiptDB.id == review.receipt_id)
            .options(selectinload(ReceiptDB.line_items))
        )
        receipt = (await session.execute(receipt_query)).scalar_one_or_none()

    image_path = receipt.image_path if receipt is not None else None
    image_url = f"/receipts/{review.receipt_id}/image" if review.receipt_id is not None else None
    raw_ocr_text = receipt.raw_ocr_text if receipt is not None else None
    extracted_receipt = _build_receipt_payload(receipt) if receipt is not None else None

    return ReviewDetailResponse(
        review=_to_review_item_response(review),
        receipt_id=review.receipt_id,
        image_path=image_path,
        image_url=image_url,
        raw_ocr_text=raw_ocr_text,
        extracted_receipt=extracted_receipt,
    )


@router.post("/{review_id:int}/correct", response_model=ReviewActionResponse)
async def correct_review_item(
    review_id: int,
    payload: ReviewCorrectionRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> ReviewActionResponse:
    review = await session.get(ReviewQueueDB, review_id)
    if review is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review item not found.")
    if review.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Review item is already resolved.")
    if not payload.corrections:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one correction is required.")

    resolved_at = datetime.now(timezone.utc)
    review.corrected_value = json.dumps([c.model_dump() for c in payload.corrections])
    review.reviewer_notes = payload.reviewer_notes or payload.corrections[0].reviewer_notes
    review.error_type = payload.corrections[0].error_type or review.error_type
    review.field_name = payload.corrections[0].field_name

    review.status = "corrected"
    review.resolved_at = resolved_at

    for idx, correction in enumerate(payload.corrections):
        if idx > 0:
            session.add(
                ReviewQueueDB(
                    receipt_id=review.receipt_id,
                    field_name=correction.field_name,
                    extracted_value=None,
                    corrected_value=correction.corrected_value,
                    error_type=correction.error_type,
                    reviewer_notes=correction.reviewer_notes or payload.reviewer_notes,
                    status="corrected",
                    resolved_at=resolved_at,
                )
            )

    if review.receipt_id is not None:
        for correction in payload.corrections:
            await _apply_correction(
                session,
                review.receipt_id,
                correction.field_name,
                correction.corrected_value,
            )
        await _finalize_receipt_if_resolved(session, review.receipt_id)
    await _upsert_error_patterns_for_corrections(session, payload.corrections, review.receipt_id)
    threshold_run_id = await _maybe_create_threshold_retraining_run(
        session,
        threshold=settings.retraining_correction_threshold,
    )

    await session.commit()

    if threshold_run_id is not None:
        execute_retraining_task.delay(threshold_run_id)
    return ReviewActionResponse(review_id=review.id, status=review.status)


@router.post("/{review_id:int}/approve", response_model=ReviewActionResponse)
async def approve_review_item(
    review_id: int,
    session: AsyncSession = Depends(get_session),
) -> ReviewActionResponse:
    review = await session.get(ReviewQueueDB, review_id)
    if review is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review item not found.")
    if review.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Review item is already resolved.")

    review.status = "approved"
    review.resolved_at = datetime.now(timezone.utc)
    if review.receipt_id is not None:
        await _finalize_receipt_if_resolved(session, review.receipt_id)
    await session.commit()
    return ReviewActionResponse(review_id=review.id, status=review.status)


@router.post("/{review_id:int}/skip", response_model=ReviewActionResponse)
async def skip_review_item(
    review_id: int,
    session: AsyncSession = Depends(get_session),
) -> ReviewActionResponse:
    review = await session.get(ReviewQueueDB, review_id)
    if review is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review item not found.")
    if review.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Review item is already resolved.")

    review.status = "skipped"
    review.resolved_at = datetime.now(timezone.utc)
    if review.receipt_id is not None:
        await _finalize_receipt_if_resolved(session, review.receipt_id)
    await session.commit()
    return ReviewActionResponse(review_id=review.id, status=review.status)


@router.get("/stats", response_model=ReviewStatsResponse)
async def review_stats(session: AsyncSession = Depends(get_session)) -> ReviewStatsResponse:
    counts_query = select(ReviewQueueDB.status, func.count()).group_by(ReviewQueueDB.status)
    count_rows = (await session.execute(counts_query)).all()
    counts = {str(row[0]): int(row[1]) for row in count_rows}

    resolved_rows = (
        await session.execute(
            select(ReviewQueueDB.created_at, ReviewQueueDB.resolved_at)
            .where(ReviewQueueDB.resolved_at.is_not(None))
        )
    ).all()
    durations = [
        (resolved - created).total_seconds()
        for created, resolved in resolved_rows
        if created is not None and resolved is not None
    ]
    average_review_seconds = (sum(durations) / len(durations)) if durations else None

    corrected = counts.get("corrected", 0)
    resolved = corrected + counts.get("approved", 0) + counts.get("skipped", 0)
    correction_rate = (corrected / resolved) if resolved > 0 else 0.0

    return ReviewStatsResponse(
        pending=counts.get("pending", 0),
        approved=counts.get("approved", 0),
        corrected=counts.get("corrected", 0),
        skipped=counts.get("skipped", 0),
        correction_rate=correction_rate,
        average_review_seconds=average_review_seconds,
    )


async def _apply_correction(
    session: AsyncSession,
    receipt_id: int,
    field_name: str | None,
    corrected_value: str,
) -> None:
    receipt = await session.get(ReceiptDB, receipt_id)
    if receipt is None:
        return

    if field_name is None:
        receipt.processing_status = "completed"
        receipt.reviewed_at = datetime.now(timezone.utc)
        return

    if field_name.startswith("line_item:"):
        parsed = field_name.split(":")
        if len(parsed) != 3:
            return
        try:
            idx = int(parsed[1])
        except ValueError:
            return
        target_field = parsed[2]
        line_items_query = (
            select(LineItemDB)
            .where(LineItemDB.receipt_id == receipt_id)
            .order_by(LineItemDB.id.asc())
        )
        line_items = (await session.execute(line_items_query)).scalars().all()
        if idx < 0 or idx >= len(line_items):
            return
        line_item = line_items[idx]
        if target_field == "name":
            line_item.name = corrected_value
        elif target_field in {"total_price", "unit_price", "quantity", "confidence", "match_confidence"}:
            try:
                numeric = float(corrected_value)
            except ValueError:
                return
            setattr(line_item, target_field, numeric)
        elif target_field in {"category", "brand", "raw_text"}:
            setattr(line_item, target_field, corrected_value)
        return

    if field_name in {"merchant_name", "merchant_category", "currency"}:
        setattr(receipt, field_name, corrected_value)
    elif field_name in {"total", "total_amount"}:
        try:
            receipt.total_amount = float(corrected_value)
        except ValueError:
            return
    elif field_name in {"date", "receipt_date"}:
        try:
            receipt.receipt_date = date.fromisoformat(corrected_value)
        except ValueError:
            return


async def _finalize_receipt_if_resolved(session: AsyncSession, receipt_id: int) -> None:
    pending_count_query = (
        select(func.count())
        .select_from(ReviewQueueDB)
        .where(
            ReviewQueueDB.receipt_id == receipt_id,
            ReviewQueueDB.status == "pending",
        )
    )
    pending_count = int((await session.execute(pending_count_query)).scalar() or 0)
    if pending_count > 0:
        return

    receipt = await session.get(ReceiptDB, receipt_id)
    if receipt is None:
        return
    receipt.processing_status = "completed"
    receipt.reviewed_at = datetime.now(timezone.utc)


def _to_review_item_response(item: ReviewQueueDB) -> ReviewQueueItemResponse:
    return ReviewQueueItemResponse(
        id=item.id,
        receipt_id=item.receipt_id,
        field_name=item.field_name,
        extracted_value=item.extracted_value,
        corrected_value=item.corrected_value,
        error_type=item.error_type,
        reviewer_notes=item.reviewer_notes,
        status=item.status,
        created_at=item.created_at,
        resolved_at=item.resolved_at,
    )


def _build_receipt_payload(receipt_db: ReceiptDB | None) -> Receipt | None:
    if receipt_db is None:
        return None
    if receipt_db.receipt_date is None or receipt_db.total_amount is None:
        return None
    if receipt_db.confidence_level not in {level.value for level in ConfidenceLevel}:
        return None

    line_items: list[LineItem] = [
        LineItem(
            raw_text=item.raw_text,
            name=item.name,
            brand=item.brand,
            quantity=item.quantity,
            unit_price=item.unit_price if item.unit_price is not None else 0.0,
            total_price=item.total_price if item.total_price is not None else 0.0,
            category=item.category,
            confidence=item.confidence if item.confidence is not None else 0.0,
            canonical_product_id=item.canonical_product_id,
            match_confidence=item.match_confidence,
        )
        for item in receipt_db.line_items
    ]

    return Receipt(
        merchant_name=receipt_db.merchant_name or "",
        merchant_address=None,
        merchant_category=receipt_db.merchant_category,
        date=receipt_db.receipt_date,
        time=None,
        subtotal=None,
        tax=None,
        tip=None,
        total=receipt_db.total_amount,
        payment_method=None,
        line_items=line_items,
        overall_confidence=receipt_db.overall_confidence if receipt_db.overall_confidence is not None else 0.0,
        confidence_level=ConfidenceLevel(receipt_db.confidence_level),
        parse_warnings=[],
        raw_ocr_text=receipt_db.raw_ocr_text,
    )


async def _upsert_error_patterns_for_corrections(
    session: AsyncSession,
    corrections,
    receipt_id: int | None,
) -> None:
    for correction in corrections:
        error_type = correction.error_type or "manual_correction"
        count_query = (
            select(func.count())
            .select_from(ReviewQueueDB)
            .where(
                ReviewQueueDB.status == "corrected",
                ReviewQueueDB.error_type == error_type,
            )
        )
        corrected_count = int((await session.execute(count_query)).scalar() or 0)
        if corrected_count < 3:
            continue

        existing_query = select(ErrorPatternDB).where(ErrorPatternDB.error_type == error_type)
        existing = (await session.execute(existing_query)).scalars().first()
        if existing is None:
            session.add(
                ErrorPatternDB(
                    error_type=error_type,
                    description=f"Pattern observed from repeated {error_type} corrections.",
                    example_receipt_ids=[receipt_id] if receipt_id is not None else None,
                    occurrence_count=corrected_count,
                    suggested_prompt_fix="Add stricter extraction guidance for this error type.",
                )
            )
        else:
            existing.occurrence_count = max(existing.occurrence_count, corrected_count)
            if receipt_id is not None:
                prior_ids = existing.example_receipt_ids or []
                if receipt_id not in prior_ids:
                    existing.example_receipt_ids = (prior_ids + [receipt_id])[-10:]


async def _maybe_create_threshold_retraining_run(
    session: AsyncSession,
    threshold: int,
) -> int | None:
    latest_run_query = (
        select(RetrainingRunDB)
        .where(RetrainingRunDB.trigger == "threshold_reached")
        .order_by(RetrainingRunDB.created_at.desc())
        .limit(1)
    )
    latest_run = (await session.execute(latest_run_query)).scalars().first()

    corrected_query = select(func.count()).select_from(ReviewQueueDB).where(ReviewQueueDB.status == "corrected")
    if latest_run is not None:
        corrected_query = corrected_query.where(ReviewQueueDB.resolved_at > latest_run.created_at)
    corrected_count = int((await session.execute(corrected_query)).scalar() or 0)

    existing_queued_query = (
        select(func.count())
        .select_from(RetrainingRunDB)
        .where(
            RetrainingRunDB.trigger == "threshold_reached",
            RetrainingRunDB.status.in_(("queued", "running")),
        )
    )
    queued_count = int((await session.execute(existing_queued_query)).scalar() or 0)
    if corrected_count < threshold or queued_count > 0:
        return None

    run = RetrainingRunDB(
        trigger="threshold_reached",
        training_samples=corrected_count,
        status="queued",
    )
    session.add(run)
    await session.flush()
    return run.id

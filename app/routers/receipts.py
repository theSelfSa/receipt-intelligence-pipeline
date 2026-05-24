from datetime import date
from pathlib import Path

from celery import group
from celery.result import GroupResult
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.database import get_session
from app.models.api import (
    ReceiptBatchStatusResponse,
    ReceiptBatchUploadResponse,
    ReceiptDetailResponse,
    ReceiptListResponse,
    ReceiptOCRResponse,
    ReceiptStatsResponse,
    ReceiptSummary,
    ReceiptUploadResponse,
)
from app.models.db import ReceiptDB, ReviewQueueDB
from app.models.schemas import ConfidenceLevel, LineItem, Receipt
from app.utils.image import save_upload_file
from app.utils.logging import get_logger
from app.utils.metrics import upload_counter
from app.worker import celery_app, process_receipt_task

router = APIRouter(prefix="/receipts", tags=["receipts"])


@router.post("/upload", response_model=ReceiptUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_receipt(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> ReceiptUploadResponse:
    logger = get_logger(__name__)
    image_path, _ = await save_upload_file(file, settings.upload_dir, settings.max_upload_size_mb)

    receipt_db = ReceiptDB(
        image_path=str(image_path),
        raw_ocr_text="",
        processing_status="pending",
    )
    session.add(receipt_db)
    await session.commit()
    await session.refresh(receipt_db)

    task = process_receipt_task.delay(receipt_db.id)
    upload_counter.inc()

    logger.info(
        "receipt_upload_enqueued",
        receipt_id=receipt_db.id,
        task_id=task.id,
    )

    return ReceiptUploadResponse(
        receipt_id=receipt_db.id,
        processing_status=receipt_db.processing_status,
        task_id=task.id,
        message="Receipt accepted and queued for asynchronous processing.",
    )


@router.post("/batch", response_model=ReceiptBatchUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_receipts_batch(
    files: list[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> ReceiptBatchUploadResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file is required.")

    receipts: list[ReceiptDB] = []
    for file in files:
        image_path, _ = await save_upload_file(file, settings.upload_dir, settings.max_upload_size_mb)
        receipt = ReceiptDB(
            image_path=str(image_path),
            raw_ocr_text="",
            processing_status="pending",
        )
        session.add(receipt)
        receipts.append(receipt)

    await session.flush()
    receipt_ids = [receipt.id for receipt in receipts]
    await session.commit()

    signatures = [process_receipt_task.s(receipt_id) for receipt_id in receipt_ids]
    job = group(signatures).apply_async()
    job.save()

    task_ids = [result.id for result in job.results]
    return ReceiptBatchUploadResponse(
        job_id=job.id,
        receipt_ids=receipt_ids,
        task_ids=task_ids,
        submitted_count=len(receipt_ids),
    )


@router.get("/batch/{job_id}", response_model=ReceiptBatchStatusResponse)
async def get_batch_status(job_id: str) -> ReceiptBatchStatusResponse:
    group_result = GroupResult.restore(job_id, app=celery_app)
    if group_result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch job not found.")

    states = [result.state for result in group_result.results]
    completed = sum(1 for state in states if state == "SUCCESS")
    failed = sum(1 for state in states if state == "FAILURE")
    pending = len(states) - completed - failed

    return ReceiptBatchStatusResponse(
        job_id=job_id,
        total=len(states),
        completed=completed,
        failed=failed,
        pending=pending,
        task_states=states,
    )


@router.get("/{receipt_id:int}", response_model=ReceiptDetailResponse)
async def get_receipt(
    receipt_id: int,
    session: AsyncSession = Depends(get_session),
) -> ReceiptDetailResponse:
    stmt = (
        select(ReceiptDB)
        .where(ReceiptDB.id == receipt_id)
        .options(selectinload(ReceiptDB.line_items))
    )
    result = await session.execute(stmt)
    receipt_db = result.scalar_one_or_none()
    if receipt_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Receipt not found.")

    receipt = _build_receipt_payload(receipt_db)

    return ReceiptDetailResponse(
        id=receipt_db.id,
        image_path=receipt_db.image_path,
        processing_status=receipt_db.processing_status,
        ocr_provider=receipt_db.ocr_provider,
        overall_confidence=receipt_db.overall_confidence,
        confidence_level=(
            ConfidenceLevel(receipt_db.confidence_level)
            if receipt_db.confidence_level in {level.value for level in ConfidenceLevel}
            else None
        ),
        parse_warnings=receipt_db.parse_warnings if isinstance(receipt_db.parse_warnings, list) else [],
        created_at=receipt_db.created_at,
        processed_at=receipt_db.processed_at,
        receipt=receipt,
    )

@router.get("/{receipt_id:int}/image")
async def get_receipt_image(
    receipt_id: int,
    session: AsyncSession = Depends(get_session),
):
    receipt_db = await session.get(ReceiptDB, receipt_id)
    if receipt_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Receipt not found.")

    image_path = Path(receipt_db.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Receipt image not found.")

    return FileResponse(path=image_path)


@router.get("/{receipt_id:int}/ocr", response_model=ReceiptOCRResponse)
async def get_receipt_ocr(
    receipt_id: int,
    session: AsyncSession = Depends(get_session),
) -> ReceiptOCRResponse:
    receipt_db = await session.get(ReceiptDB, receipt_id)
    if receipt_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Receipt not found.")
    return ReceiptOCRResponse(receipt_id=receipt_db.id, raw_ocr_text=receipt_db.raw_ocr_text)


@router.get("/", response_model=ReceiptListResponse)
async def list_receipts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    status_filter: str | None = Query(default=None, alias="status"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> ReceiptListResponse:
    query = select(ReceiptDB)
    count_query = select(func.count()).select_from(ReceiptDB)

    if status_filter:
        query = query.where(ReceiptDB.processing_status == status_filter)
        count_query = count_query.where(ReceiptDB.processing_status == status_filter)
    if start_date is not None:
        query = query.where(ReceiptDB.receipt_date >= start_date)
        count_query = count_query.where(ReceiptDB.receipt_date >= start_date)
    if end_date is not None:
        query = query.where(ReceiptDB.receipt_date <= end_date)
        count_query = count_query.where(ReceiptDB.receipt_date <= end_date)

    query = query.order_by(ReceiptDB.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    rows_result = await session.execute(query)
    total_result = await session.execute(count_query)

    receipts = rows_result.scalars().all()
    total = int(total_result.scalar() or 0)
    items = [
        ReceiptSummary(
            id=receipt.id,
            processing_status=receipt.processing_status,
            merchant_name=receipt.merchant_name,
            receipt_date=receipt.receipt_date,
            total_amount=receipt.total_amount,
            confidence_level=receipt.confidence_level,
            created_at=receipt.created_at,
            processed_at=receipt.processed_at,
        )
        for receipt in receipts
    ]
    return ReceiptListResponse(items=items, page=page, page_size=page_size, total=total)


@router.get("/stats", response_model=ReceiptStatsResponse)
async def receipts_stats(session: AsyncSession = Depends(get_session)) -> ReceiptStatsResponse:
    total_result = await session.execute(select(func.count()).select_from(ReceiptDB))
    total = int(total_result.scalar() or 0)

    status_result = await session.execute(
        select(ReceiptDB.processing_status, func.count())
        .group_by(ReceiptDB.processing_status)
    )
    status_counts = {str(row[0]): int(row[1]) for row in status_result.all()}

    avg_confidence_result = await session.execute(select(func.avg(ReceiptDB.overall_confidence)))
    avg_confidence_raw = avg_confidence_result.scalar()
    avg_confidence = float(avg_confidence_raw) if avg_confidence_raw is not None else None

    processed_rows = (
        await session.execute(
            select(ReceiptDB.processed_at).where(ReceiptDB.processed_at.is_not(None))
        )
    ).all()
    processed_times = [row[0] for row in processed_rows if row[0] is not None]
    if not processed_times:
        throughput_receipts_per_day = None
    elif len(processed_times) == 1:
        throughput_receipts_per_day = 1.0
    else:
        earliest = min(processed_times)
        latest = max(processed_times)
        span_seconds = max(1.0, (latest - earliest).total_seconds())
        throughput_receipts_per_day = (len(processed_times) / span_seconds) * 86400

    review_counts_result = await session.execute(
        select(ReviewQueueDB.status, func.count()).group_by(ReviewQueueDB.status)
    )
    review_counts = {str(row[0]): int(row[1]) for row in review_counts_result.all()}
    corrected_reviews = review_counts.get("corrected", 0)
    resolved_reviews = corrected_reviews + review_counts.get("approved", 0) + review_counts.get("skipped", 0)
    estimated_accuracy = (
        1 - (corrected_reviews / resolved_reviews)
        if resolved_reviews > 0
        else None
    )

    raw_text_rows = (
        await session.execute(
            select(ReceiptDB.raw_ocr_text).where(ReceiptDB.raw_ocr_text.is_not(None))
        )
    ).all()
    estimated_tokens_total = sum(max(0, len((row[0] or "")) // 4) for row in raw_text_rows)
    estimated_cost_total_usd = estimated_tokens_total * 0.000005
    estimated_cost_per_receipt_usd = (
        estimated_cost_total_usd / total
        if total > 0
        else 0.0
    )

    return ReceiptStatsResponse(
        total_receipts=total,
        status_counts=status_counts,
        avg_confidence=avg_confidence,
        throughput_receipts_per_day=throughput_receipts_per_day,
        resolved_reviews=resolved_reviews,
        corrected_reviews=corrected_reviews,
        estimated_accuracy=estimated_accuracy,
        estimated_tokens_total=estimated_tokens_total,
        estimated_cost_total_usd=estimated_cost_total_usd,
        estimated_cost_per_receipt_usd=estimated_cost_per_receipt_usd,
    )


def _build_receipt_payload(receipt_db: ReceiptDB) -> Receipt | None:
    if receipt_db.processing_status in {"pending", "processing", "failed"}:
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

    if receipt_db.receipt_date is None or receipt_db.total_amount is None:
        return None
    if receipt_db.confidence_level not in {level.value for level in ConfidenceLevel}:
        return None

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

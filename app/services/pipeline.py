from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings
from app.models.db import LineItemDB, ReceiptDB, ReviewQueueDB
from app.models.schemas import ConfidenceLevel, Receipt
from app.services.confidence import compute_confidence
from app.services.extraction import ReceiptExtractionService
from app.services.matching import ProductMatchingService
from app.services.ocr import OCRService
from app.utils.image import preprocess_image_path


async def process_receipt_pipeline(
    *,
    receipt_id: int,
    session: AsyncSession,
    settings: Settings,
    ocr_service: OCRService | None = None,
    extraction_service: ReceiptExtractionService | None = None,
    matching_service: ProductMatchingService | None = None,
) -> ReceiptDB:
    ocr = ocr_service or OCRService(settings=settings)
    extractor = extraction_service or ReceiptExtractionService(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )
    matcher = matching_service or ProductMatchingService(settings)

    query = (
        select(ReceiptDB)
        .where(ReceiptDB.id == receipt_id)
        .options(selectinload(ReceiptDB.line_items))
    )
    result = await session.execute(query)
    receipt_db = result.scalar_one_or_none()
    if receipt_db is None:
        raise ValueError(f"Receipt {receipt_id} not found.")

    receipt_db.processing_status = "processing"
    await session.commit()

    try:
        preprocessed = preprocess_image_path(Path(receipt_db.image_path))
        ocr_result = await ocr.extract_text(preprocessed.processed_image)
        if not ocr_result.raw_text.strip():
            raise ValueError("OCR extraction returned empty text.")

        extracted_receipt: Receipt = await extractor.extract(ocr_result.raw_text)
        overall_confidence, confidence_level = compute_confidence(
            extracted_receipt,
            ocr_result.mean_ocr_confidence,
        )
        extracted_receipt.overall_confidence = overall_confidence
        extracted_receipt.confidence_level = confidence_level
        extracted_receipt.raw_ocr_text = ocr_result.raw_text

        receipt_db.raw_ocr_text = ocr_result.raw_text
        receipt_db.ocr_provider = ocr_result.ocr_provider
        receipt_db.overall_confidence = overall_confidence
        receipt_db.confidence_level = confidence_level.value
        receipt_db.merchant_name = extracted_receipt.merchant_name
        receipt_db.merchant_category = extracted_receipt.merchant_category
        receipt_db.receipt_date = extracted_receipt.date
        receipt_db.total_amount = extracted_receipt.total
        receipt_db.parse_warnings = [warning.value for warning in extracted_receipt.parse_warnings]
        receipt_db.processed_at = datetime.now(timezone.utc)
        receipt_db.processing_status = _processing_status_from_confidence(confidence_level)

        for existing_item in list(receipt_db.line_items):
            await session.delete(existing_item)
        await session.flush()

        for item in extracted_receipt.line_items:
            match_result = await matcher.match_line_item(item.name or item.raw_text, session)
            session.add(
                LineItemDB(
                    receipt_id=receipt_db.id,
                    raw_text=item.raw_text,
                    name=item.name,
                    brand=item.brand,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    total_price=item.total_price,
                    category=item.category,
                    confidence=item.confidence,
                    canonical_product_id=(
                        match_result.product.id
                        if match_result.product is not None
                        else item.canonical_product_id
                    ),
                    match_confidence=(
                        match_result.confidence
                        if match_result.method != "none"
                        else item.match_confidence
                    ),
                    match_method=match_result.method if match_result.method != "none" else None,
                )
            )

        await _refresh_review_queue(
            session=session,
            receipt_id=receipt_db.id,
            extracted_receipt=extracted_receipt,
            confidence_level=confidence_level,
        )
        await session.commit()
        await session.refresh(receipt_db)
        return receipt_db
    except Exception:
        receipt_db.processing_status = "failed"
        receipt_db.processed_at = datetime.now(timezone.utc)
        await session.commit()
        raise


async def _refresh_review_queue(
    *,
    session: AsyncSession,
    receipt_id: int,
    extracted_receipt: Receipt,
    confidence_level: ConfidenceLevel,
) -> None:
    existing_query = select(ReviewQueueDB).where(
        ReviewQueueDB.receipt_id == receipt_id,
        ReviewQueueDB.status == "pending",
    )
    existing_result = await session.execute(existing_query)
    for existing in existing_result.scalars().all():
        await session.delete(existing)
    await session.flush()

    if confidence_level == ConfidenceLevel.HIGH:
        return

    if confidence_level == ConfidenceLevel.LOW:
        session.add(
            ReviewQueueDB(
                receipt_id=receipt_id,
                field_name=None,
                extracted_value="Full receipt requires review",
                error_type="low_overall_confidence",
                status="pending",
            )
        )
        return

    for warning in extracted_receipt.parse_warnings:
        session.add(
            ReviewQueueDB(
                receipt_id=receipt_id,
                field_name=f"warning:{warning.value}",
                extracted_value=warning.value,
                error_type=warning.value,
                status="pending",
            )
        )

    for idx, item in enumerate(extracted_receipt.line_items):
        if item.confidence < 0.75:
            session.add(
                ReviewQueueDB(
                    receipt_id=receipt_id,
                    field_name=f"line_item:{idx}:name",
                    extracted_value=item.name,
                    error_type="low_item_confidence",
                    status="pending",
                )
            )


def _processing_status_from_confidence(level: ConfidenceLevel) -> str:
    if level == ConfidenceLevel.HIGH:
        return "completed"
    if level == ConfidenceLevel.MEDIUM:
        return "partial_review"
    return "review_required"

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.models.schemas import ConfidenceLevel, LineItem, ParseWarning, Receipt


SYSTEM_PROMPT = """
You are a receipt parsing expert focused on noisy OCR.
Return best-effort structured extraction with these rules:
- Never leave merchant_name blank. If unknown, use "Unknown Merchant".
- Never leave date blank. If missing, infer from text; otherwise use today's date.
- Never leave total blank. Infer from TOTAL/GRAND TOTAL/CASH lines; fallback to subtotal or line-item sum.
- Extract line items from product-like lines with prices.
- Exclude payment/change metadata lines from line items.
- For each line item, keep raw_text close to OCR text and set a realistic confidence.
- Preserve decimal values exactly when visible in OCR.
- Prefer factual extraction over guesses; use null only for optional fields when truly absent.
""".strip()


class _GroqLineItem(BaseModel):
    raw_text: str | None = None
    name: str | None = None
    brand: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    total_price: float | None = None
    category: str | None = None
    confidence: float | None = None
    canonical_product_id: int | None = None
    match_confidence: float | None = None


class _GroqReceipt(BaseModel):
    merchant_name: str | None = None
    merchant_address: str | None = None
    merchant_category: str | None = None
    date: str | None = None
    time: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    tip: float | None = None
    total: float | None = None
    payment_method: str | None = None
    line_items: list[_GroqLineItem] = Field(default_factory=list)
    overall_confidence: float | None = None
    confidence_level: str | None = None
    parse_warnings: list[str] = Field(default_factory=list)
    raw_ocr_text: str | None = None


class GroqReceiptExtractionService:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._model = model
        self._client = instructor.from_openai(
            AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        )

    async def extract(self, raw_text: str) -> Receipt:
        if not raw_text.strip():
            raise ValueError("OCR text is empty.")

        response = await self._client.chat.completions.create(
            model=self._model,
            response_model=_GroqReceipt,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Parse this receipt:\n\n{raw_text}"},
            ],
            temperature=0,
            max_retries=2,
        )
        return _normalize_receipt(response, raw_text)


def _normalize_receipt(parsed: _GroqReceipt, raw_text: str) -> Receipt:
    normalized_items = [_normalize_item(item) for item in parsed.line_items]
    subtotal = parsed.subtotal
    if subtotal is None and normalized_items:
        subtotal = round(sum(item.total_price for item in normalized_items), 2)

    total = parsed.total
    if total is None:
        if subtotal is not None:
            total = float(subtotal)
        else:
            total = round(sum(item.total_price for item in normalized_items), 2)

    overall_confidence = parsed.overall_confidence
    if overall_confidence is None:
        if normalized_items:
            overall_confidence = sum(item.confidence for item in normalized_items) / len(normalized_items)
        else:
            overall_confidence = 0.25
    overall_confidence = _clamp(overall_confidence)

    confidence_level = _normalize_confidence_level(parsed.confidence_level, overall_confidence)
    parse_warnings = _normalize_parse_warnings(parsed.parse_warnings)

    return Receipt(
        merchant_name=(parsed.merchant_name or "Unknown Merchant").strip() or "Unknown Merchant",
        merchant_address=parsed.merchant_address,
        merchant_category=parsed.merchant_category,
        date=_parse_date(parsed.date),
        time=_parse_time(parsed.time),
        subtotal=subtotal,
        tax=parsed.tax,
        tip=parsed.tip,
        total=float(total),
        payment_method=parsed.payment_method,
        line_items=normalized_items,
        overall_confidence=overall_confidence,
        confidence_level=confidence_level,
        parse_warnings=parse_warnings,
        raw_ocr_text=parsed.raw_ocr_text or raw_text,
    )


def _normalize_item(item: _GroqLineItem) -> LineItem:
    quantity = item.quantity if item.quantity is not None and item.quantity > 0 else 1.0
    total_price = item.total_price
    unit_price = item.unit_price

    if total_price is None and unit_price is not None:
        total_price = float(unit_price) * float(quantity)
    if unit_price is None and total_price is not None:
        unit_price = float(total_price) / float(quantity)
    if total_price is None:
        total_price = 0.0
    if unit_price is None:
        unit_price = float(total_price) / float(quantity) if quantity > 0 else 0.0

    name = (item.name or "").strip()
    raw = (item.raw_text or "").strip()
    if not name:
        name = raw or "Unknown Item"
    if not raw:
        raw = name

    return LineItem(
        raw_text=raw,
        name=name,
        brand=item.brand,
        quantity=float(quantity),
        unit_price=float(unit_price),
        total_price=float(total_price),
        category=item.category,
        confidence=_clamp(item.confidence if item.confidence is not None else 0.35),
        canonical_product_id=item.canonical_product_id,
        match_confidence=item.match_confidence,
    )


def _parse_date(value: str | None) -> date:
    if not value:
        return date.today()

    candidate = value.strip()
    if not candidate:
        return date.today()

    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%y",
        "%d/%m/%y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(candidate).date()
    except ValueError:
        return date.today()


def _parse_time(value: str | None) -> time | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    formats = ("%H:%M:%S", "%H:%M", "%I:%M %p")
    for fmt in formats:
        try:
            return datetime.strptime(candidate, fmt).time()
        except ValueError:
            continue
    try:
        return time.fromisoformat(candidate)
    except ValueError:
        return None


def _normalize_confidence_level(value: str | None, overall: float) -> ConfidenceLevel:
    if value:
        cleaned = value.strip().lower()
        if cleaned == ConfidenceLevel.HIGH.value:
            return ConfidenceLevel.HIGH
        if cleaned == ConfidenceLevel.MEDIUM.value:
            return ConfidenceLevel.MEDIUM
        if cleaned == ConfidenceLevel.LOW.value:
            return ConfidenceLevel.LOW

    if overall >= 0.85:
        return ConfidenceLevel.HIGH
    if overall >= 0.65:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _normalize_parse_warnings(values: list[str] | None) -> list[ParseWarning]:
    if not values:
        return []
    normalized: list[ParseWarning] = []
    known = {warning.value: warning for warning in ParseWarning}
    for value in values:
        key = value.strip().lower()
        warning = known.get(key)
        if warning is not None and warning not in normalized:
            normalized.append(warning)
    return normalized


def _clamp(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))

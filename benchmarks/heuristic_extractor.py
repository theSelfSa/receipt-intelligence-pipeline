from __future__ import annotations

import re
from datetime import date, datetime

from app.models.schemas import ConfidenceLevel, LineItem, Receipt


class HeuristicReceiptExtractor:
    async def extract(self, raw_text: str) -> Receipt:
        return parse_receipt_heuristic(raw_text)


def parse_receipt_heuristic(raw_text: str) -> Receipt:
    if not raw_text.strip():
        raise ValueError("OCR text is empty.")

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    merchant_name = _extract_merchant_name(lines)
    receipt_date = _extract_receipt_date(raw_text) or date.today()
    line_items = _extract_line_items(lines)

    subtotal = round(sum(item.total_price for item in line_items), 2) if line_items else None
    tax = _extract_keyword_amount(lines, ("tax", "vat", "gst"))
    tip = _extract_keyword_amount(lines, ("tip", "gratuity"))
    total = _extract_keyword_amount(lines, ("grand total", "amount due", "balance due", "total"))

    if total is None:
        amount_candidates = _extract_amount_candidates(raw_text)
        total = max(amount_candidates) if amount_candidates else 0.0

    if subtotal is None and total:
        subtotal = round(max(total - (tax or 0.0) - (tip or 0.0), 0.0), 2)

    line_item_confidence = (
        sum(item.confidence for item in line_items) / len(line_items) if line_items else 0.25
    )
    overall_confidence = min(
        0.78,
        max(
            0.35,
            (line_item_confidence * 0.7) + (0.15 if total > 0 else 0.0),
        ),
    )
    confidence_level = (
        ConfidenceLevel.MEDIUM if overall_confidence >= 0.65 else ConfidenceLevel.LOW
    )

    return Receipt(
        merchant_name=merchant_name,
        date=receipt_date,
        subtotal=subtotal,
        tax=tax,
        tip=tip,
        total=round(float(total), 2),
        line_items=line_items,
        overall_confidence=overall_confidence,
        confidence_level=confidence_level,
        raw_ocr_text=raw_text,
    )


def _extract_merchant_name(lines: list[str]) -> str:
    if not lines:
        return "Unknown Merchant"

    skip_keywords = {
        "tax",
        "total",
        "subtotal",
        "receipt",
        "invoice",
        "date",
        "time",
        "qty",
    }
    for line in lines[:8]:
        if len(line) < 2:
            continue
        lower = line.lower()
        if any(keyword in lower for keyword in skip_keywords):
            continue
        if _extract_amount_candidates(line):
            continue
        cleaned = re.sub(r"\s+", " ", line).strip(" -:_")
        if cleaned:
            return cleaned[:120]
    return lines[0][:120]


def _extract_line_items(lines: list[str]) -> list[LineItem]:
    items: list[LineItem] = []
    skip_keywords = {
        "subtotal",
        "total",
        "tax",
        "tip",
        "change",
        "cash",
        "visa",
        "mastercard",
        "amex",
        "debit",
        "credit",
        "amount due",
        "balance due",
    }

    for line in lines:
        lower = line.lower()
        if any(keyword in lower for keyword in skip_keywords):
            continue

        amounts = _extract_amount_candidates(line)
        if not amounts:
            continue

        total_price = round(amounts[-1], 2)
        if total_price <= 0:
            continue

        name = _clean_item_name(line)
        if len(name) < 2:
            continue

        quantity = _extract_quantity(line)
        unit_price = round(total_price / quantity, 2) if quantity > 0 else total_price
        items.append(
            LineItem(
                raw_text=line,
                name=name,
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price,
                confidence=0.45,
            )
        )
        if len(items) >= 80:
            break
    return items


def _clean_item_name(line: str) -> str:
    without_prices = re.sub(r"(?<!\d)\d{1,5}(?:[.,]\d{2})(?!\d)", " ", line)
    without_qty = re.sub(r"\b\d+(?:\.\d+)?\s*[x×]\b", " ", without_prices, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9\s\-\&/]", " ", without_qty)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:_")
    if not cleaned:
        return "Unknown Item"
    return cleaned[:100]


def _extract_quantity(line: str) -> float:
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*[x×]\b", line, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bx\s*(\d+(?:\.\d+)?)\b", line, flags=re.IGNORECASE)
    if not match:
        return 1.0
    try:
        quantity = float(match.group(1))
    except ValueError:
        return 1.0
    if quantity <= 0 or quantity > 100:
        return 1.0
    return quantity


def _extract_keyword_amount(lines: list[str], keywords: tuple[str, ...]) -> float | None:
    for line in reversed(lines):
        lower = line.lower()
        if any(keyword in lower for keyword in keywords):
            values = _extract_amount_candidates(line)
            if values:
                return max(values)
    return None


def _extract_amount_candidates(text: str) -> list[float]:
    matches = re.findall(r"(?<!\d)(\d{1,5}(?:[.,]\d{2}))(?!\d)", text)
    values: list[float] = []
    for match in matches:
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def _extract_receipt_date(raw_text: str) -> date | None:
    patterns = [
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    ]
    for pattern in patterns:
        for candidate in re.findall(pattern, raw_text):
            parsed = _parse_date(candidate)
            if parsed is not None:
                return parsed
    return None


def _parse_date(value: str) -> date | None:
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
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None

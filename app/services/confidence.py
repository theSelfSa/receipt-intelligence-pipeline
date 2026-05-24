from datetime import date

from app.models.schemas import ConfidenceLevel, ParseWarning, Receipt


def compute_confidence(receipt: Receipt, ocr_confidence: float) -> tuple[float, ConfidenceLevel]:
    scores = []

    ocr_score = max(0.0, min(1.0, ocr_confidence))
    scores.append(ocr_score)

    if ocr_score < 0.65:
        _append_warning_once(receipt, ParseWarning.LOW_OCR_QUALITY)

    if receipt.line_items:
        avg_item_confidence = sum(i.confidence for i in receipt.line_items) / len(receipt.line_items)
        scores.append(avg_item_confidence)
    else:
        _append_warning_once(receipt, ParseWarning.MISSING_LINE_ITEMS)
        scores.append(0.3)

    if receipt.line_items and receipt.subtotal is not None:
        computed_sum = sum(i.total_price for i in receipt.line_items)
        if abs(computed_sum - receipt.subtotal) > 0.10:
            scores.append(0.4)
            _append_warning_once(receipt, ParseWarning.TOTAL_MISMATCH)

    if receipt.date:
        days_diff = abs((date.today() - receipt.date).days)
        if days_diff > 1825:
            scores.append(0.5)

    overall = sum(scores) / len(scores) if scores else 0.0

    if overall >= 0.85:
        level = ConfidenceLevel.HIGH
    elif overall >= 0.65:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    return overall, level


def _append_warning_once(receipt: Receipt, warning: ParseWarning) -> None:
    if warning not in receipt.parse_warnings:
        receipt.parse_warnings.append(warning)

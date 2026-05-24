from datetime import date, timedelta

from app.models.schemas import ConfidenceLevel, ParseWarning
from app.services.confidence import compute_confidence


def test_compute_confidence_high_for_clean_receipt(sample_receipt) -> None:
    overall, level = compute_confidence(sample_receipt, ocr_confidence=0.95)

    assert overall >= 0.85
    assert level == ConfidenceLevel.HIGH


def test_compute_confidence_adds_total_mismatch_warning(sample_receipt) -> None:
    sample_receipt.subtotal = 7.50

    overall, level = compute_confidence(sample_receipt, ocr_confidence=0.9)

    assert ParseWarning.TOTAL_MISMATCH in sample_receipt.parse_warnings
    assert overall < 0.85
    assert level in {ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW}


def test_compute_confidence_penalizes_old_dates(sample_receipt) -> None:
    sample_receipt.date = date.today() - timedelta(days=365 * 7)

    overall, level = compute_confidence(sample_receipt, ocr_confidence=0.9)

    assert overall < 0.85
    assert level in {ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW}

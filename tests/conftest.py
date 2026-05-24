from datetime import date

import pytest

from app.models.schemas import ConfidenceLevel, LineItem, Receipt


@pytest.fixture
def sample_receipt() -> Receipt:
    return Receipt(
        merchant_name="Sample Store",
        merchant_address="123 Main St",
        merchant_category="grocery",
        date=date.today(),
        subtotal=5.98,
        tax=0.42,
        tip=0.0,
        total=6.40,
        payment_method="card",
        line_items=[
            LineItem(
                raw_text="MILK 2.99",
                name="Milk",
                quantity=1.0,
                unit_price=2.99,
                total_price=2.99,
                confidence=0.9,
            ),
            LineItem(
                raw_text="BREAD 2.99",
                name="Bread",
                quantity=1.0,
                unit_price=2.99,
                total_price=2.99,
                confidence=0.88,
            ),
        ],
        overall_confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH,
        parse_warnings=[],
        raw_ocr_text="MILK 2.99\nBREAD 2.99",
    )

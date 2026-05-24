from datetime import date

import pytest

from app.models.schemas import ConfidenceLevel, LineItem, Receipt
from app.services.extraction import ReceiptExtractionService


class _FakeCompletions:
    def __init__(self, response: Receipt):
        self.response = response
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.response


class _FakeChat:
    def __init__(self, response: Receipt):
        self.completions = _FakeCompletions(response)


class _FakeClient:
    def __init__(self, response: Receipt):
        self.chat = _FakeChat(response)


@pytest.mark.asyncio
async def test_extraction_returns_receipt_schema() -> None:
    expected = Receipt(
        merchant_name="Demo Market",
        merchant_address=None,
        merchant_category="grocery",
        date=date.today(),
        subtotal=10.0,
        tax=0.8,
        tip=None,
        total=10.8,
        payment_method="card",
        line_items=[
            LineItem(
                raw_text="APPLE 2 x 2.00",
                name="Apple",
                quantity=2.0,
                unit_price=2.00,
                total_price=4.00,
                confidence=0.92,
            )
        ],
        overall_confidence=0.9,
        confidence_level=ConfidenceLevel.HIGH,
        parse_warnings=[],
        raw_ocr_text="APPLE 2 x 2.00",
    )
    fake_client = _FakeClient(expected)
    service = ReceiptExtractionService(api_key="", model="gpt-4o", client=fake_client)

    result = await service.extract("APPLE 2 x 2.00")

    assert result == expected
    assert fake_client.chat.completions.last_kwargs is not None
    assert fake_client.chat.completions.last_kwargs["model"] == "gpt-4o"
    assert fake_client.chat.completions.last_kwargs["response_model"] is Receipt


@pytest.mark.asyncio
async def test_extraction_requires_openai_key_when_no_injected_client() -> None:
    service = ReceiptExtractionService(api_key="", model="gpt-4o")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await service.extract("receipt text")

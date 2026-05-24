import pytest
from PIL import Image

from app.services.ocr import OCRService


@pytest.mark.asyncio
async def test_ocr_service_builds_text_and_confidence(monkeypatch) -> None:
    fake_ocr_output = {
        "text": ["MILK", "2.99"],
        "conf": ["95", "90"],
        "page_num": [1, 1],
        "block_num": [1, 1],
        "par_num": [1, 1],
        "line_num": [1, 1],
    }

    def fake_image_to_data(*args, **kwargs):
        return fake_ocr_output

    def fake_image_to_string(*args, **kwargs):
        return "MILK 2.99"

    monkeypatch.setattr("app.services.ocr.pytesseract.image_to_data", fake_image_to_data)
    monkeypatch.setattr("app.services.ocr.pytesseract.image_to_string", fake_image_to_string)

    image = Image.new("L", (32, 32), "white")
    service = OCRService()
    result = await service.extract_text(image)

    assert result.raw_text == "MILK 2.99"
    assert len(result.word_confidences) == 2
    assert result.mean_ocr_confidence > 0.9

import asyncio
from collections import defaultdict
from io import BytesIO
import os
from pathlib import Path

import boto3
import pytesseract
from PIL import Image
from pydantic import BaseModel, Field

from app.config import Settings


class OCRResult(BaseModel):
    raw_text: str
    word_confidences: list[float] = Field(default_factory=list)
    mean_ocr_confidence: float = 0.0
    ocr_provider: str = "tesseract"


class OCRService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._configure_tesseract_binary()
        self._textract_client = self._build_textract_client(settings)

    async def extract_text(self, image: Image.Image) -> OCRResult:
        if self._textract_client is not None:
            textract_result = await self._extract_with_textract(image)
            if textract_result is not None:
                return textract_result
        return await self._extract_with_tesseract(image)

    async def _extract_with_textract(self, image: Image.Image) -> OCRResult | None:
        try:
            image_bytes = await asyncio.to_thread(_image_to_png_bytes, image)
            response = await asyncio.to_thread(
                self._textract_client.analyze_document,
                Document={"Bytes": image_bytes},
                FeatureTypes=["TABLES", "FORMS"],
            )
            result = _parse_textract_response(response)
            if result.raw_text.strip():
                return result
        except Exception:
            return None
        return None

    async def _extract_with_tesseract(self, image: Image.Image) -> OCRResult:
        data = await asyncio.to_thread(
            pytesseract.image_to_data,
            image,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )

        words = data.get("text", [])
        confidences = data.get("conf", [])
        page_nums = data.get("page_num", [])
        block_nums = data.get("block_num", [])
        par_nums = data.get("par_num", [])
        line_nums = data.get("line_num", [])

        grouped_lines: dict[tuple[int, int, int, int], list[str]] = defaultdict(list)
        word_confidences: list[float] = []

        for idx, raw_word in enumerate(words):
            word = str(raw_word).strip()
            if not word:
                continue

            key = (
                int(page_nums[idx]) if idx < len(page_nums) else 0,
                int(block_nums[idx]) if idx < len(block_nums) else 0,
                int(par_nums[idx]) if idx < len(par_nums) else 0,
                int(line_nums[idx]) if idx < len(line_nums) else 0,
            )
            grouped_lines[key].append(word)

            confidence = _to_confidence_ratio(confidences[idx] if idx < len(confidences) else -1)
            if confidence is not None:
                word_confidences.append(confidence)

        ordered_keys = sorted(grouped_lines.keys())
        lines = [" ".join(grouped_lines[key]) for key in ordered_keys]
        raw_text = "\n".join(lines).strip()

        if not raw_text:
            raw_text = (
                await asyncio.to_thread(
                    pytesseract.image_to_string,
                    image,
                    config="--psm 6",
                )
            ).strip()

        mean_ocr_confidence = (
            sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
        )

        return OCRResult(
            raw_text=raw_text,
            word_confidences=word_confidences,
            mean_ocr_confidence=mean_ocr_confidence,
            ocr_provider="tesseract",
        )

    @staticmethod
    def _build_textract_client(settings: Settings | None):
        if settings is None:
            return None
        try:
            return boto3.client(
                "textract",
                aws_access_key_id=settings.aws_access_key_id or None,
                aws_secret_access_key=settings.aws_secret_access_key or None,
                region_name=settings.aws_region,
            )
        except Exception:
            return None

    @staticmethod
    def _configure_tesseract_binary() -> None:
        candidates: list[str] = []
        env_path = os.getenv("TESSERACT_CMD")
        if env_path:
            candidates.append(env_path)

        candidates.extend(
            [
                "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
            ]
        )

        for candidate in candidates:
            if Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = str(Path(candidate))
                return


def _to_confidence_ratio(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if parsed < 0:
        return None
    return max(0.0, min(1.0, parsed / 100.0))


def _parse_textract_response(response: dict) -> OCRResult:
    blocks = response.get("Blocks", [])
    line_blocks = [block for block in blocks if block.get("BlockType") == "LINE" and block.get("Text")]
    word_blocks = [block for block in blocks if block.get("BlockType") == "WORD"]

    raw_text = "\n".join(str(block.get("Text", "")).strip() for block in line_blocks).strip()
    if not raw_text:
        raw_text = " ".join(str(block.get("Text", "")).strip() for block in word_blocks if block.get("Text")).strip()

    word_confidences: list[float] = []
    for block in word_blocks:
        conf = _to_confidence_ratio(block.get("Confidence"))
        if conf is not None:
            word_confidences.append(conf)

    mean_ocr_confidence = (
        sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
    )
    return OCRResult(
        raw_text=raw_text,
        word_confidences=word_confidences,
        mean_ocr_confidence=mean_ocr_confidence,
        ocr_provider="textract",
    )


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()

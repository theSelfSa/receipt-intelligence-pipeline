from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageEnhance, ImageStat, UnidentifiedImageError
import pypdfium2 as pdfium


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}


@dataclass
class PreprocessedImage:
    image_path: Path
    processed_image: Image.Image
    original_bytes: bytes | None = None


async def preprocess_upload(file: UploadFile, upload_dir: Path, max_upload_size_mb: int) -> PreprocessedImage:
    image_path, raw_bytes = await save_upload_file(file, upload_dir, max_upload_size_mb)
    return preprocess_image_bytes(raw_bytes, image_path)


async def save_upload_file(file: UploadFile, upload_dir: Path, max_upload_size_mb: int) -> tuple[Path, bytes]:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    max_size_bytes = max_upload_size_mb * 1024 * 1024
    if len(raw_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max size of {max_upload_size_mb}MB.",
        )

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {extension}. Supported: {sorted(ALLOWED_EXTENSIONS)}",
        )

    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / f"{uuid4().hex}{extension}"
    image_path.write_bytes(raw_bytes)
    return image_path, raw_bytes


def preprocess_image_path(image_path: Path) -> PreprocessedImage:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    raw_bytes = image_path.read_bytes()
    return preprocess_image_bytes(raw_bytes, image_path)


def preprocess_image_bytes(raw_bytes: bytes, image_path: Path) -> PreprocessedImage:
    if image_path.suffix.lower() == ".pdf":
        image = _render_first_pdf_page(raw_bytes)
    else:
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unable to parse image bytes.",
            ) from exc

    image = _resize_if_needed(image, max_dimension=4000)
    if _mean_pixel(image) < 128:
        image = ImageEnhance.Contrast(image).enhance(1.4)
    image = image.convert("L")

    return PreprocessedImage(
        image_path=image_path,
        processed_image=image,
        original_bytes=raw_bytes,
    )


def _resize_if_needed(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    largest_dimension = max(width, height)
    if largest_dimension <= max_dimension:
        return image

    ratio = max_dimension / largest_dimension
    new_size = (int(width * ratio), int(height * ratio))
    return image.resize(new_size)


def _mean_pixel(image: Image.Image) -> float:
    grayscale = image.convert("L")
    return ImageStat.Stat(grayscale).mean[0]


def _render_first_pdf_page(raw_bytes: bytes) -> Image.Image:
    try:
        document = pdfium.PdfDocument(raw_bytes)
        if len(document) == 0:
            raise ValueError("PDF has no pages.")
        page = document[0]
        rendered = page.render(scale=2.0)
        image = rendered.to_pil().convert("RGB")
        page.close()
        document.close()
        return image
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to render first page from PDF.",
        ) from exc

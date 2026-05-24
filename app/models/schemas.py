from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, time
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"      # > 0.85 — auto-accept
    MEDIUM = "medium"  # 0.65-0.85 — flag specific fields
    LOW = "low"        # < 0.65 — full human review


class LineItem(BaseModel):
    raw_text: str                          # original OCR text, verbatim
    name: str                              # cleaned product name
    brand: Optional[str] = None
    quantity: float
    unit_price: float
    total_price: float
    category: Optional[str] = None        # grocery, pharmacy, restaurant, fuel, etc.
    confidence: float = Field(ge=0, le=1) # LLM self-assessed confidence for this item
    canonical_product_id: Optional[int] = None
    match_confidence: Optional[float] = None


class ParseWarning(str, Enum):
    DATE_AMBIGUOUS = "date_format_ambiguous"
    TOTAL_MISMATCH = "total_does_not_match_items"
    MERCHANT_UNKNOWN = "merchant_not_in_database"
    LOW_OCR_QUALITY = "ocr_confidence_below_threshold"
    MISSING_LINE_ITEMS = "no_line_items_extracted"


class Receipt(BaseModel):
    merchant_name: str
    merchant_address: Optional[str] = None
    merchant_category: Optional[str] = None  # grocery, restaurant, pharmacy, etc.
    date: date
    time: Optional[time] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    tip: Optional[float] = None
    total: float
    payment_method: Optional[str] = None  # cash, card, etc.
    line_items: List[LineItem]
    overall_confidence: float = Field(ge=0, le=1)
    confidence_level: ConfidenceLevel
    parse_warnings: List[ParseWarning] = []
    raw_ocr_text: str                     # always store original OCR output

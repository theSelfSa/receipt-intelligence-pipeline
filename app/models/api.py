from datetime import date, datetime

from pydantic import BaseModel, Field

from app.models.schemas import ConfidenceLevel, Receipt


class ReceiptUploadResponse(BaseModel):
    receipt_id: int
    processing_status: str
    task_id: str | None = None
    message: str | None = None


class ReceiptBatchUploadResponse(BaseModel):
    job_id: str
    receipt_ids: list[int]
    task_ids: list[str]
    submitted_count: int


class ReceiptBatchStatusResponse(BaseModel):
    job_id: str
    total: int
    completed: int
    failed: int
    pending: int
    task_states: list[str]


class ReceiptSummary(BaseModel):
    id: int
    processing_status: str
    merchant_name: str | None = None
    receipt_date: date | None = None
    total_amount: float | None = None
    confidence_level: str | None = None
    created_at: datetime
    processed_at: datetime | None = None


class ReceiptListResponse(BaseModel):
    items: list[ReceiptSummary]
    page: int
    page_size: int
    total: int


class ReceiptStatsResponse(BaseModel):
    total_receipts: int
    status_counts: dict[str, int]
    avg_confidence: float | None = None
    throughput_receipts_per_day: float | None = None
    resolved_reviews: int = 0
    corrected_reviews: int = 0
    estimated_accuracy: float | None = None
    estimated_tokens_total: int = 0
    estimated_cost_total_usd: float = 0.0
    estimated_cost_per_receipt_usd: float = 0.0


class ReceiptOCRResponse(BaseModel):
    receipt_id: int
    raw_ocr_text: str


class ReceiptDetailResponse(BaseModel):
    id: int
    image_path: str
    processing_status: str
    ocr_provider: str | None = None
    overall_confidence: float | None = None
    confidence_level: ConfidenceLevel | None = None
    parse_warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    processed_at: datetime | None = None
    receipt: Receipt | None = None


class ReviewQueueItemResponse(BaseModel):
    id: int
    receipt_id: int | None
    field_name: str | None
    extracted_value: str | None
    corrected_value: str | None
    error_type: str | None
    reviewer_notes: str | None
    status: str
    created_at: datetime
    resolved_at: datetime | None = None


class ReviewQueueListResponse(BaseModel):
    items: list[ReviewQueueItemResponse]
    total: int

class ReviewFieldCorrection(BaseModel):
    field_name: str
    corrected_value: str
    error_type: str | None = None
    reviewer_notes: str | None = None


class ReviewCorrectionRequest(BaseModel):
    corrections: list[ReviewFieldCorrection]
    reviewer_notes: str | None = None


class ReviewActionResponse(BaseModel):
    review_id: int
    status: str


class ReviewDetailResponse(BaseModel):
    review: ReviewQueueItemResponse
    receipt_id: int | None = None
    image_path: str | None = None
    image_url: str | None = None
    raw_ocr_text: str | None = None
    extracted_receipt: Receipt | None = None


class ReviewStatsResponse(BaseModel):
    pending: int
    approved: int
    corrected: int
    skipped: int
    correction_rate: float
    average_review_seconds: float | None = None


class CatalogProductCreateRequest(BaseModel):
    name: str
    brand: str | None = None
    category: str
    subcategory: str | None = None
    upc: str | None = None


class CatalogProductResponse(BaseModel):
    id: int
    name: str
    brand: str | None = None
    category: str
    subcategory: str | None = None
    upc: str | None = None
    created_at: datetime
    updated_at: datetime


class CatalogProductsListResponse(BaseModel):
    items: list[CatalogProductResponse]
    page: int
    page_size: int
    total: int


class CatalogMatchRequest(BaseModel):
    raw_text: str


class CatalogMatchResult(BaseModel):
    method: str
    product_id: int | None = None
    name: str | None = None
    brand: str | None = None
    category: str | None = None
    confidence: float


class CatalogMatchResponse(BaseModel):
    query: str
    result: CatalogMatchResult


class CatalogEmbedResponse(BaseModel):
    updated_count: int
    model: str


class ErrorPatternResponse(BaseModel):
    id: int
    error_type: str
    description: str
    occurrence_count: int
    suggested_prompt_fix: str | None = None
    effectiveness_score: float | None = None
    created_at: datetime


class ErrorPatternsListResponse(BaseModel):
    items: list[ErrorPatternResponse]
    total: int


class AnalyzeTriggerResponse(BaseModel):
    task_id: str
    status: str


class AccuracyCategoryPoint(BaseModel):
    merchant_category: str | None = None
    receipts: int
    corrected: int
    estimated_accuracy: float


class AccuracyResponse(BaseModel):
    points: list[AccuracyCategoryPoint]


class ConfidenceCalibrationPoint(BaseModel):
    confidence_level: str
    receipts: int
    corrected: int
    estimated_accuracy: float


class ConfidenceCalibrationResponse(BaseModel):
    points: list[ConfidenceCalibrationPoint]


class CostSummaryResponse(BaseModel):
    receipts_processed: int
    estimated_tokens_total: int
    estimated_cost_total_usd: float
    estimated_cost_per_receipt_usd: float


class RetrainTriggerResponse(BaseModel):
    run_id: int
    task_id: str
    status: str


class RetrainingRunResponse(BaseModel):
    id: int
    trigger: str
    training_samples: int | None = None
    validation_accuracy: float | None = None
    baseline_accuracy: float | None = None
    improvement: float | None = None
    model_path: str | None = None
    wandb_run_id: str | None = None
    status: str | None = None
    created_at: datetime


class RetrainingRunsListResponse(BaseModel):
    items: list[RetrainingRunResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    database: str

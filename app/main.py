from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import get_settings
from app.database import check_database_health, close_database, init_database
from app.models.api import HealthResponse
from app.routers.analytics import router as analytics_router
from app.routers.catalog import router as catalog_router
from app.routers.receipts import router as receipts_router
from app.routers.retrain import router as retrain_router
from app.routers.review import router as review_router
from app.services.extraction import ReceiptExtractionService
from app.services.ocr import OCRService
from app.utils.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    await init_database(settings.database_url)

    app.state.ocr_service = OCRService(settings=settings)
    app.state.extraction_service = ReceiptExtractionService(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )

    logger.info("application_startup_complete", env=settings.app_env)
    try:
        yield
    finally:
        await close_database()
        logger.info("application_shutdown_complete")


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(receipts_router)
app.include_router(review_router)
app.include_router(catalog_router)
app.include_router(analytics_router)
app.include_router(retrain_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    db_ok = await check_database_health()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database="ok" if db_ok else "down",
    )


@app.get("/metrics", tags=["system"])
async def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

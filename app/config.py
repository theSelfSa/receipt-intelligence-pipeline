from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Receipt Intelligence Pipeline"
    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/receipts"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    openai_api_key: str = Field(default="")
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"

    upload_dir: Path = Path("uploads")
    max_upload_size_mb: int = 10
    low_ocr_threshold: float = 0.65
    medium_confidence_threshold: float = 0.65
    high_confidence_threshold: float = 0.85
    retraining_correction_threshold: int = 500
    analysis_min_corrections: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

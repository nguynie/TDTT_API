import os
from functools import lru_cache
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings and runtime configurations."""

    app_name: str = "Hugging Face Sentiment Analysis API"
    app_version: str = "1.0.0"
    app_description: str = (
        "A production-ready Web API built with FastAPI and Hugging Face Transformers "
        "providing real-time sentiment analysis using DistilBERT."
    )
    model_name: str = os.getenv(
        "MODEL_NAME", "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    device: str = os.getenv("DEVICE", "auto")
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "32"))
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "2000"))


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()

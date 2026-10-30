from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class RootResponse(BaseModel):
    """Response schema for the root endpoint (GET /)."""

    name: str = Field(..., description="API service name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API summary and purpose")
    model_name: str = Field(..., description="Hugging Face model identifier")
    model_url: str = Field(..., description="Link to model on Hugging Face Hub")
    endpoints: Dict[str, str] = Field(..., description="Map of available API endpoints")
    documentation: Dict[str, str] = Field(..., description="Links to interactive API docs")


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint (GET /health)."""

    status: str = Field(..., description="System operational status (healthy/degraded)")
    model_loaded: bool = Field(..., description="Whether the ML model is currently in memory")
    model_name: str = Field(..., description="Hugging Face model identifier")
    device: str = Field(..., description="Device used for inference (cpu or cuda)")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of check")


class PredictRequest(BaseModel):
    """Request payload for sentiment analysis (POST /predict)."""

    text: Optional[str] = Field(
        None,
        description="Single text string to analyze",
        examples=["FastAPI and Hugging Face are incredibly powerful and easy to use!"]
    )
    texts: Optional[List[str]] = Field(
        None,
        description="Optional list of text strings for batch analysis",
        examples=[
            [
                "The product quality exceeded all my expectations!",
                "The delivery was delayed and customer support was unhelpful."
            ]
        ]
    )

    @model_validator(mode="after")
    def validate_inputs(self) -> "PredictRequest":
        """Validate that either `text` or non-empty `texts` is supplied."""
        has_text = self.text is not None and bool(self.text.strip())
        has_texts = self.texts is not None and len(self.texts) > 0

        if not has_text and not has_texts:
            raise ValueError(
                "Either 'text' (non-empty string) or 'texts' (non-empty list of strings) must be provided."
            )

        if self.texts is not None:
            cleaned_texts = [t.strip() for t in self.texts if isinstance(t, str) and t.strip()]
            if not cleaned_texts:
                raise ValueError("The 'texts' list contains no valid non-empty strings.")
            self.texts = cleaned_texts

        if self.text is not None:
            self.text = self.text.strip()
            if not self.text:
                raise ValueError("'text' must not be empty or blank whitespace.")

        return self


class SentimentResult(BaseModel):
    """Prediction result for a single input text."""

    text: str = Field(..., description="Input text analyzed")
    label: str = Field(..., description="Predicted sentiment label (e.g. POSITIVE, NEGATIVE)")
    score: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")
    inference_time_ms: float = Field(..., description="Inference execution latency in milliseconds")


class PredictResponse(BaseModel):
    """Response schema for predictions (POST /predict)."""

    success: bool = Field(True, description="Indicates whether inference succeeded")
    model_name: str = Field(..., description="Hugging Face model identifier used")
    total_samples: int = Field(..., description="Number of items analyzed")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")
    results: List[SentimentResult] = Field(..., description="List of prediction results")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error category or type")
    detail: str = Field(..., description="Human-readable explanation of the error")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of error"
    )

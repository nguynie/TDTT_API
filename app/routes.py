import logging
import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.model import SentimentAnalyzer, get_model
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RootResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    response_model=RootResponse,
    summary="API Root Information",
    description="Provides an overview of the Web API service, active model, and available endpoints.",
    tags=["General"],
)
async def read_root(
    settings: Settings = Depends(get_settings),
) -> RootResponse:
    return RootResponse(
        name=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        model_name=settings.model_name,
        model_url=f"https://huggingface.co/{settings.model_name}",
        endpoints={
            "GET /": "Service introduction and metadata",
            "GET /health": "System and model readiness check",
            "POST /predict": "Execute sentiment inference on input text(s)",
        },
        documentation={
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Operational Status Check",
    description="Verifies the operational status of the service, whether the model is in memory, and the device used.",
    tags=["System"],
)
async def check_health(
    settings: Settings = Depends(get_settings),
    model: SentimentAnalyzer = Depends(get_model),
) -> HealthResponse:
    return HealthResponse(
        status="healthy" if model.is_loaded else "ready (model pending lazy load)",
        model_loaded=model.is_loaded,
        model_name=settings.model_name,
        device=model.device_str if model.is_loaded else "not loaded yet",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data or empty text"},
        422: {"description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal server or inference failure"},
    },
    summary="Perform Sentiment Analysis",
    description=(
        "Analyzes one or more sentences and predicts sentiment (POSITIVE or NEGATIVE) "
        "along with confidence scores using the Hugging Face transformer model."
    ),
    tags=["Inference"],
)
async def predict_sentiment(
    payload: PredictRequest,
    settings: Settings = Depends(get_settings),
    model: SentimentAnalyzer = Depends(get_model),
) -> PredictResponse:
    # Prepare list of texts to analyze
    if payload.texts:
        items_to_process = payload.texts
    elif payload.text:
        items_to_process = [payload.text]
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid text provided for prediction.",
        )

    # Batch size limit enforcement
    if len(items_to_process) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(items_to_process)} exceeds maximum allowed batch size of {settings.max_batch_size}.",
        )

    # Text length validation
    for text in items_to_process:
        if len(text) > settings.max_text_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input text exceeds maximum length of {settings.max_text_length} characters.",
            )

    start_total = time.perf_counter()
    try:
        results = model.predict(items_to_process)
    except Exception as exc:
        logger.error("Inference failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(exc)}",
        ) from exc

    total_time_ms = (time.perf_counter() - start_total) * 1000.0

    return PredictResponse(
        success=True,
        model_name=settings.model_name,
        total_samples=len(results),
        total_time_ms=round(total_time_ms, 2),
        results=results,
    )

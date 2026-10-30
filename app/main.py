import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.model import get_model
from app.routes import router

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle application startup and shutdown events."""
    logger.info("Initializing Hugging Face Sentiment Analysis API...")
    settings = get_settings()
    logger.info("Configuration: Model=%s, Device=%s", settings.model_name, settings.device)

    # Pre-load the Hugging Face model into memory on startup
    try:
        model = get_model()
        model.load()
        logger.info("Model pre-loaded successfully on startup.")
    except Exception as exc:
        logger.warning(
            "Model pre-loading failed or deferred: %s. Model will attempt lazy-load on first request.",
            exc,
        )

    yield

    logger.info("Shutting down Hugging Face Sentiment Analysis API...")


def create_app() -> FastAPI:
    """Factory function to build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom handler for validation errors
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        first_error = exc.errors()[0] if exc.errors() else {"msg": "Validation failed"}
        error_msg = str(first_error.get("msg", "Invalid request payload"))
        field = " -> ".join(str(loc) for loc in first_error.get("loc", []))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Validation Error",
                "detail": f"Field '{field}': {error_msg}" if field else error_msg,
                "errors": jsonable_encoder(exc.errors()),
            },
        )

    # Custom handler for standard HTTPExceptions
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": "HTTP Error",
                "detail": exc.detail,
            },
        )

    # Custom handler for uncaught unexpected exceptions
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred during processing.",
            },
        )

    # Mount routes
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

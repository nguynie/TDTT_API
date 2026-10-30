import logging
import time
from typing import List, Optional
import torch
from transformers import pipeline

from app.config import get_settings
from app.schemas import SentimentResult

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Wrapper around Hugging Face transformers pipeline for sentiment analysis."""

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.model_name
        self.pipeline = None
        self._device_str: str = "uninitialized"

    @property
    def is_loaded(self) -> bool:
        """Return True if model pipeline is loaded in memory."""
        return self.pipeline is not None

    @property
    def device_str(self) -> str:
        """Return device descriptor string."""
        return self._device_str

    def load(self) -> None:
        """Load the Hugging Face pipeline into CPU or GPU memory."""
        if self.is_loaded:
            logger.info("Model '%s' is already loaded.", self.model_name)
            return

        logger.info("Loading Hugging Face model: %s ...", self.model_name)

        # Device determination
        target_device = self.settings.device.lower()
        if target_device == "cpu":
            pipe_device = -1
            self._device_str = "cpu"
        elif target_device in ("cuda", "gpu") or (target_device == "auto" and torch.cuda.is_available()):
            pipe_device = 0
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda"
            self._device_str = f"cuda:0 ({gpu_name})"
        else:
            pipe_device = -1
            self._device_str = "cpu"

        logger.info("Target inference device: %s", self._device_str)

        # Initialize the pipeline
        self.pipeline = pipeline(
            task="sentiment-analysis",
            model=self.model_name,
            device=pipe_device,
        )
        logger.info("Model '%s' successfully loaded on %s", self.model_name, self._device_str)

    def predict(self, texts: List[str]) -> List[SentimentResult]:
        """Perform sentiment analysis inference on a list of text strings."""
        if not self.is_loaded:
            self.load()

        if not texts:
            return []

        results: List[SentimentResult] = []
        for text in texts:
            start_time = time.perf_counter()
            # truncation=True ensures sentences longer than model limit (512) are truncated cleanly
            output = self.pipeline(text, truncation=True, max_length=512)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            prediction = output[0] if isinstance(output, list) else output
            results.append(
                SentimentResult(
                    text=text,
                    label=prediction["label"],
                    score=round(float(prediction["score"]), 4),
                    inference_time_ms=round(elapsed_ms, 2),
                )
            )

        return results


# Global singleton instance
_model_instance: Optional[SentimentAnalyzer] = None


def get_model() -> SentimentAnalyzer:
    """Retrieve or initialize the singleton SentimentAnalyzer instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentAnalyzer()
    return _model_instance

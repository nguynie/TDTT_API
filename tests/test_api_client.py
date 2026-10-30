import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test GET / returns system metadata and available endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "model_name" in data
    assert "endpoints" in data
    assert "POST /predict" in data["endpoints"]


def test_health_endpoint():
    """Test GET /health returns healthy status and model information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "ready (model pending lazy load)")
    assert "model_name" in data
    assert "device" in data
    assert "timestamp" in data


def test_predict_single_positive():
    """Test POST /predict with an unambiguously positive sentence."""
    payload = {"text": "This API is blazing fast and the documentation is wonderful!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_samples"] == 1
    result = data["results"][0]
    assert result["label"] == "POSITIVE"
    assert result["score"] >= 0.8
    assert result["inference_time_ms"] > 0


def test_predict_single_negative():
    """Test POST /predict with an unambiguously negative sentence."""
    payload = {"text": "I had a dreadful experience and the application kept crashing."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_samples"] == 1
    result = data["results"][0]
    assert result["label"] == "NEGATIVE"
    assert result["score"] >= 0.8


def test_predict_batch():
    """Test POST /predict with a batch list of sentences."""
    payload = {
        "texts": [
            "We had a fantastic time presenting our computational thinking lab!",
            "The error messages were completely unhelpful and confusing."
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_samples"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["label"] == "POSITIVE"
    assert data["results"][1]["label"] == "NEGATIVE"


def test_predict_empty_payload():
    """Test POST /predict with empty JSON payload returns validation error (422)."""
    response = client.post("/predict", json={})
    assert response.status_code in (400, 422)
    data = response.json()
    assert data.get("success") is False or "detail" in data


def test_predict_blank_string():
    """Test POST /predict with whitespace-only string returns validation error (422)."""
    response = client.post("/predict", json={"text": "     "})
    assert response.status_code in (400, 422)


def test_predict_invalid_data_types():
    """Test POST /predict with invalid data types returns 422."""
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422

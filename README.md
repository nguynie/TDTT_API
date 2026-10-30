# Practice 01: Application Programming Interface (Web API)
**Course**: Computational Thinking (Tư duy tính toán)  
**Faculty of Information Technology — University of Science, VNU-HCM**  
**Lab Instructors**: Nguyen Ngoc Minh Chau, Le Duc Khoan  

---

## Student Information

- **Full Name**: Nguyen Ngoc Phuc Nguyen
- **Student ID**: 25127434
- **Repository**: [https://github.com/Nguynie/TDTT_API](https://github.com/Nguynie/TDTT_API)
- **Demonstration Video**: [Watch Demo Video on YouTube / Google Drive](https://youtu.be/your-demo-video-link)

---

## Selected Hugging Face Model

- **Model Identifier**: [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- **Model Type**: Text Classification / Sentiment Analysis
- **Architecture**: DistilBERT (Transformer-based, distilled version of BERT)
- **Task**: Binary Sentiment Classification (`POSITIVE` / `NEGATIVE`)
- **Key Advantages**:
  - **Lightweight & Fast**: 40% smaller and 60% faster than BERT-base while retaining 97% of language understanding capabilities.
  - **High Accuracy**: Fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset, reaching over 91.3% accuracy.
  - **Low Latency**: Ideal for real-time Web API deployment on both CPU and GPU (CUDA).

---

## System Functionality Overview

This project implements a production-grade Web API built with **FastAPI** that deploys the Hugging Face sentiment analysis model for real-time inference.

### Key Features:
1. **RESTful Architecture**: Clean, intuitive endpoints adhering to HTTP standards.
2. **FastAPI Lifespan Manager**: Model is loaded directly into memory during application startup to avoid request-time cold-start delay.
3. **Hardware Acceleration**: Automatic GPU (CUDA) detection with fallback to CPU.
4. **Strict Schema Validation**: Powered by **Pydantic v2**, ensuring incoming data is non-empty, well-formatted, and trimmed.
5. **Comprehensive Error Handling**: Returns structured JSON error responses with standard HTTP status codes (`400 Bad Request`, `422 Unprocessable Entity`, `500 Internal Server Error`).
6. **Batch Inference Support**: Accepts both single text sentences and batch lists of texts.
7. **Automated Testing Suite**: Includes tests using Python's `requests` library and `pytest` with `TestClient`.
8. **Interactive API Documentation**: Built-in Swagger UI (`/docs`) and ReDoc (`/redoc`).

---

## Source Code Structure

```
TDTT_API/
├── app/
│   ├── __init__.py            # Package declaration
│   ├── config.py              # Application settings and environment configuration
│   ├── schemas.py             # Pydantic v2 request/response schemas & validation
│   ├── model.py               # Hugging Face transformers pipeline manager
│   ├── routes.py              # API routes: GET /, GET /health, POST /predict
│   └── main.py                # FastAPI application entrypoint, CORS, exception handlers
├── tests/
│   ├── __init__.py            # Test package
│   ├── test_api_requests.py   # API integration tests using `requests` library
│   └── test_api_client.py     # Automated unit/integration tests using pytest
├── requirements.txt           # Project dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # Comprehensive project documentation
```

---

## Installation and Setup Instructions

### Prerequisites
- Python 3.10+ (Recommended: **Python 3.11**)
- Git
- Internet connection (for initial package and model download)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nguynie/TDTT_API.git
cd TDTT_API
```

### Step 2: Create and Activate Virtual Environment
```bash
# On Linux / macOS:
python3.11 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

### Start Development Server with Uvicorn
Run the following command from the root project directory:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Output:
```text
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
2026-10-29 20:45:00 [INFO] app: Initializing Hugging Face Sentiment Analysis API...
2026-10-29 20:45:00 [INFO] app.model: Loading Hugging Face model: distilbert/distilbert-base-uncased-finetuned-sst-2-english ...
2026-10-29 20:45:02 [INFO] app.model: Model successfully loaded on cuda:0 (NVIDIA GeForce RTX 3050 Laptop GPU)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

The server is now live at: `http://127.0.0.1:8000`

### Interactive Documentation
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints and Usage Examples

### 1. Root Endpoint (`GET /`)
Returns system information, active model identifier, and links to documentation.

#### Example Request (cURL):
```bash
curl -X GET http://127.0.0.1:8000/
```

#### Example Response (`200 OK`):
```json
{
  "name": "Hugging Face Sentiment Analysis API",
  "version": "1.0.0",
  "description": "A production-ready Web API built with FastAPI and Hugging Face Transformers providing real-time sentiment analysis using DistilBERT.",
  "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "model_url": "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "endpoints": {
    "GET /": "Service introduction and metadata",
    "GET /health": "System and model readiness check",
    "POST /predict": "Execute sentiment inference on input text(s)"
  },
  "documentation": {
    "swagger_ui": "/docs",
    "redoc": "/redoc",
    "openapi_json": "/openapi.json"
  }
}
```

---

### 2. Health Check (`GET /health`)
Checks whether the server and machine learning model are operational and identifies the compute device.

#### Example Request (cURL):
```bash
curl -X GET http://127.0.0.1:8000/health
```

#### Example Response (`200 OK`):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "device": "cuda:0 (NVIDIA GeForce RTX 3050 Laptop GPU)",
  "timestamp": "2026-10-29T13:45:10.123456+00:00"
}
```

---

### 3. Sentiment Prediction (`POST /predict`)
Receives text data and returns sentiment classification (`POSITIVE` or `NEGATIVE`) along with confidence score and latency.

#### Example 3.1: Single Text Analysis (cURL)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI and Hugging Face work seamlessly together! I love this project."}'
```

#### Example Response (`200 OK`):
```json
{
  "success": true,
  "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "total_samples": 1,
  "total_time_ms": 11.25,
  "results": [
    {
      "text": "FastAPI and Hugging Face work seamlessly together! I love this project.",
      "label": "POSITIVE",
      "score": 0.9998,
      "inference_time_ms": 10.84
    }
  ]
}
```

#### Example 3.2: Batch Text Analysis (cURL)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "The computer science faculty provided fantastic support for our lab.",
      "The server crashed repeatedly and the error documentation was completely missing."
    ]
  }'
```

#### Example Response (`200 OK`):
```json
{
  "success": true,
  "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "total_samples": 2,
  "total_time_ms": 18.42,
  "results": [
    {
      "text": "The computer science faculty provided fantastic support for our lab.",
      "label": "POSITIVE",
      "score": 0.9997,
      "inference_time_ms": 9.12
    },
    {
      "text": "The server crashed repeatedly and the error documentation was completely missing.",
      "label": "NEGATIVE",
      "score": 0.9994,
      "inference_time_ms": 8.95
    }
  ]
}
```

---

### 4. Calling the API Using Python `requests`

```python
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. Health check
health_res = requests.get(f"{BASE_URL}/health")
print("Health:", health_res.json())

# 2. Prediction
payload = {
    "text": "I am having an amazing day learning modern web frameworks!"
}
predict_res = requests.post(f"{BASE_URL}/predict", json=payload)
data = predict_res.json()
print("Prediction Result:")
print(json.dumps(data, indent=2))
```

---

### 5. Input Validation & Error Handling Examples

#### Case A: Empty Request Body (`{}`)
- **Status Code**: `422 Unprocessable Entity`
- **Response**:
```json
{
  "success": false,
  "error": "Validation Error",
  "detail": "Either 'text' (non-empty string) or 'texts' (non-empty list of strings) must be provided."
}
```

#### Case B: Blank / Whitespace String (`{"text": "   "}`)
- **Status Code**: `422 Unprocessable Entity`
- **Response**:
```json
{
  "success": false,
  "error": "Validation Error",
  "detail": "Field 'text': 'text' must not be empty or blank whitespace."
}
```

---

## Testing the API

### Method 1: Using the `requests` Library Test Script

Make sure the FastAPI server is running in one terminal, then run:

```bash
python tests/test_api_requests.py
```

To run against a custom host or port:
```bash
python tests/test_api_requests.py --base-url http://127.0.0.1:8000
```

### Method 2: Running Automated Pytest Suite
```bash
pytest tests/test_api_client.py -v
```

---

## Demonstration Video

- **Video Link**: [Watch Demo Video](https://youtu.be/your-demo-video-link)
- **Duration**: Maximum 5 minutes
- **Content**: Model introduction, architecture overview, execution demo, test runs with multiple inputs, and results explanation.

---

## License and Acknowledgments
- Assignment developed for **Practice 01: Application Programming Interface**, Faculty of Information Technology, University of Science, VNU-HCM.
- Hugging Face Model: [DistilBERT Base Uncased Finetuned SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) by the Hugging Face team.

# Video Demonstration Presentation Script (Maximum 5 Minutes)

This script is structured to help you deliver an A+ demonstration video fulfilling all criteria in **Section 5: Demonstration Video Requirements** of the Computational Thinking Practice 01 assignment.

---

## Timing Breakdown and Outline

| Time | Section | Key Talking Points & Actions |
|---|---|---|
| **0:00 - 0:45** | **1. Introduction** | Introduce yourself (Name, Student ID, Class) and the selected Hugging Face model (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`). |
| **0:45 - 2:00** | **2. Code Architecture** | Walk through the repository files: `main.py`, `routes.py`, `schemas.py`, `model.py`, and `requirements.txt`. |
| **2:00 - 3:00** | **3. Running the Server** | Show the terminal, activate virtual environment, start Uvicorn (`uvicorn app.main:app --reload`), and display interactive docs at `/docs`. |
| **3:00 - 4:15** | **4. Testing the API** | Run `python tests/test_api_requests.py` and test interactive inputs in Swagger UI (Positive, Negative, and Error handling). |
| **4:15 - 5:00** | **5. Analysis & Conclusion** | Explain the JSON outputs (labels, confidence scores, latency), recap key achievements, and conclude. |

---

## Step-by-Step Speaking Script

### Part 1: Introduction (0:00 - 0:45)
- **What to show on screen**: Face camera or Title slide / GitHub repository homepage.
- **Script**:
  > *"Hello instructors and everyone! My name is **[Your Full Name]**, student ID **[Your Student ID]**, from class **[Your Class]** at the Faculty of Information Technology, University of Science, VNU-HCM.*
  >
  > *Today, I am presenting my Practice 01 project for Computational Thinking: Deploying a Hugging Face model as a Web API using FastAPI.*
  >
  > *For this assignment, I selected the model **`distilbert/distilbert-base-uncased-finetuned-sst-2-english`**, which is a high-performance transformer model fine-tuned on the Stanford Sentiment Treebank (SST-2) for binary sentiment classification."*

---

### Part 2: Source Code Architecture (0:45 - 2:00)
- **What to show on screen**: VS Code or your code editor showing the project directory tree.
- **Script**:
  > *"Let's take a quick look at the project structure:*
  >
  > 1. *`app/main.py`: The FastAPI application entrypoint. It utilizes an asynchronous `lifespan` manager to preload the transformer model into GPU/CPU memory on startup, avoiding cold-start latency for user requests.*
  > 2. *`app/routes.py`: Implements the three required endpoints: `GET /` for system metadata, `GET /health` for operational checks, and `POST /predict` for sentiment classification.*
  > 3. *`app/schemas.py`: Built using Pydantic v2. It validates incoming request payloads (such as preventing blank whitespace or empty inputs) and structures JSON responses.*
  > 4. *`app/model.py`: Wraps Hugging Face's `transformers.pipeline`, handling device detection (CUDA GPU or CPU fallback), text truncation, and inference timing.*
  > 5. *`tests/test_api_requests.py`: A dedicated test suite utilizing Python's `requests` library to test all endpoints and status codes, as specified in the assignment prompt."*

---

### Part 3: Running the Application (2:00 - 3:00)
- **What to show on screen**: Terminal and Web Browser.
- **Script**:
  > *"Now, let's run the application.*
  >
  > *First, we activate our virtual environment:*
  > ```bash
  > source .venv/bin/activate
  > ```
  > *Then, we start the Uvicorn ASGI server:*
  > ```bash
  > uvicorn app.main:app --reload
  > ```
  > *As we can see in the logs, the model loads directly into memory.*
  >
  > *Let's switch to the browser and visit `http://127.0.0.1:8000/docs`. FastAPI automatically generates an interactive OpenAPI/Swagger UI where we can inspect and test all endpoints."*

---

### Part 4: Testing the API (3:00 - 4:15)
- **What to show on screen**: Terminal running the test script, or Swagger UI.
- **Script**:
  > *"Now let's test our API with diverse inputs using our automated requests test script:*
  > ```bash
  > python tests/test_api_requests.py
  > ```
  > *Let's observe the results:*
  >
  > 1. *First input (Positive): `'Antigravity and FastAPI make building AI web applications astonishingly fast and delightful!'`*
  >    *The model returned label **POSITIVE** with a confidence score of over **99.9%**!*
  > 2. *Second input (Negative): `'The delivery was delayed by two weeks and the package arrived broken. Terrible service.'`*
  >    *The model returned label **NEGATIVE** with over **99.9%** confidence.*
  > 3. *Batch processing: We can also pass multiple texts in a single request, which returns predictions for all sentences simultaneously.*
  > 4. *Error handling: When sending an empty text or invalid payload, the API intercepts it and cleanly returns HTTP **422 Unprocessable Entity** with an informative error message, preventing any model crash."*

---

### Part 5: Results Analysis & Conclusion (4:15 - 5:00)
- **What to show on screen**: GitHub repo or final slide.
- **Script**:
  > *"In summary, the API fulfills all requirements:*
  > - *It delivers high-accuracy sentiment analysis powered by Hugging Face.*
  > - *Provides standard REST endpoints with Pydantic validation and comprehensive error handling.*
  > - *Includes automated testing with Python `requests` and `pytest`, as well as interactive documentation.*
  >
  > *Thank you very much for watching my presentation!"*

---

## Recommendations for Recording
- **Resolution**: 1080p (1920x1080) is recommended.
- **Audio**: Use a headset or clear microphone in a quiet room.
- **Time limit**: Keep it strictly under 5 minutes (target 4:00 - 4:45).
- **Upload**: Upload to YouTube (Unlisted) or Google Drive (set access to **Anyone with the link can view**), and paste the link into `README.md`.

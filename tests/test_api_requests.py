#!/usr/bin/env python3
"""API Test Script using the `requests` library.

Fulfills the requirement from HCMUS FIT Practice 01:
'At least one Python file for testing the API using the requests library.'
"""

import argparse
import json
import sys
import requests


def print_header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def test_root_endpoint(base_url: str) -> bool:
    print_header("1. Testing GET / (Root Endpoint)")
    url = f"{base_url}/"
    try:
        response = requests.get(url, timeout=10)
        print(f"URL: {url}")
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        print("[PASS] Root endpoint test passed.")
        return True
    except Exception as exc:
        print(f"[FAIL] Root endpoint test failed: {exc}")
        return False


def test_health_endpoint(base_url: str) -> bool:
    print_header("2. Testing GET /health (Health Check)")
    url = f"{base_url}/health"
    try:
        response = requests.get(url, timeout=10)
        print(f"URL: {url}")
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "status" in data
        assert "model_name" in data
        assert "device" in data
        assert "timestamp" in data
        print("[PASS] Health check endpoint test passed.")
        return True
    except Exception as exc:
        print(f"[FAIL] Health check endpoint test failed: {exc}")
        return False


def test_predict_positive(base_url: str) -> bool:
    print_header("3. Testing POST /predict (Positive Sentiment Input)")
    url = f"{base_url}/predict"
    payload = {
        "text": "Antigravity and FastAPI make building AI web applications astonishingly fast and delightful!"
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload)}")
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1
        prediction = data["results"][0]
        assert prediction["label"] == "POSITIVE"
        assert prediction["score"] > 0.8
        print(
            f"[PASS] Positive prediction test passed. Predicted: {prediction['label']} ({prediction['score'] * 100:.2f}%)"
        )
        return True
    except Exception as exc:
        print(f"[FAIL] Positive prediction test failed: {exc}")
        return False


def test_predict_negative(base_url: str) -> bool:
    print_header("4. Testing POST /predict (Negative Sentiment Input)")
    url = f"{base_url}/predict"
    payload = {
        "text": "The delivery was delayed by two weeks and the package arrived broken. Terrible service."
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload)}")
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1
        prediction = data["results"][0]
        assert prediction["label"] == "NEGATIVE"
        assert prediction["score"] > 0.8
        print(
            f"[PASS] Negative prediction test passed. Predicted: {prediction['label']} ({prediction['score'] * 100:.2f}%)"
        )
        return True
    except Exception as exc:
        print(f"[FAIL] Negative prediction test failed: {exc}")
        return False


def test_predict_batch(base_url: str) -> bool:
    print_header("5. Testing POST /predict (Batch Input)")
    url = f"{base_url}/predict"
    payload = {
        "texts": [
            "I genuinely love studying artificial intelligence at HCMUS!",
            "I am deeply disappointed with the latest software update, it crashes constantly."
        ]
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["success"] is True
        assert data["total_samples"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["label"] == "POSITIVE"
        assert data["results"][1]["label"] == "NEGATIVE"
        print("[PASS] Batch prediction test passed.")
        return True
    except Exception as exc:
        print(f"[FAIL] Batch prediction test failed: {exc}")
        return False


def test_error_handling_empty_payload(base_url: str) -> bool:
    print_header("6. Testing Error Handling: Empty Payload")
    url = f"{base_url}/predict"
    payload = {}
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Payload: {json.dumps(payload)}")
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))

        assert response.status_code in (400, 422), f"Expected 400/422, got {response.status_code}"
        print("[PASS] Correctly rejected empty payload.")
        return True
    except Exception as exc:
        print(f"[FAIL] Empty payload test failed: {exc}")
        return False


def test_error_handling_blank_text(base_url: str) -> bool:
    print_header("7. Testing Error Handling: Blank / Whitespace-only String")
    url = f"{base_url}/predict"
    payload = {"text": "      "}
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Payload: {json.dumps(payload)}")
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))

        assert response.status_code in (400, 422), f"Expected 400/422, got {response.status_code}"
        print("[PASS] Correctly rejected whitespace-only input.")
        return True
    except Exception as exc:
        print(f"[FAIL] Blank text test failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run API tests using the requests library against FastAPI server."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI server (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    print(f"Starting test suite against: {args.base_url}")
    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_predict_positive,
        test_predict_negative,
        test_predict_batch,
        test_error_handling_empty_payload,
        test_error_handling_blank_text,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        if test_fn(args.base_url):
            passed += 1
        else:
            failed += 1

    print_header("TEST SUMMARY")
    print(f"Total Tests : {len(tests)}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    if failed > 0:
        print("\n[FAIL] Some tests failed.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All requests library tests passed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()

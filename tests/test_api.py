
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_prediction():
    payload = {
        "age_years": 5,
        "km_driven": 25000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": 4,
        "accidents": 0,
        "brand_tier": "premium",
        "city_zone": "Central"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], (int, float))


def test_engine_cc_above_limit():
    payload = {
        "age_years": 5,
        "km_driven": 25000,
        "engine_cc": 1000,
        "owners": 1,
        "service_records": 4,
        "accidents": 0,
        "brand_tier": "premium",
        "city_zone": "Central"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_negative_service_records():
    payload = {
        "age_years": 5,
        "km_driven": 25000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": -2,
        "accidents": 0,
        "brand_tier": "premium",
        "city_zone": "Central"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_request_field():
    payload = {
        "age_years": 5,
        "km_driven": 25000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": 4,
        "accidents": 0,
        "brand_tier": "premium",
        "city_zone": "Central",
        "record_id": "R001"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

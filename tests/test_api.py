from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_predict_valid():
    payload = {
        "age": 35,
        "days_as_customer": 180,
        "total_payment_amount": 1500.5,
        "total_payments_count": 12,
        "avg_payment_amount": 125.04,
        "days_since_last_activity": 5,
        "total_activity_duration": 360.0,
        "avg_activity_duration": 30.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "churn_probability" in data
    assert "churn_prediction" in data
    
    probs = data["churn_probability"] if isinstance(data["churn_probability"], list) else [data["churn_probability"]]
    preds = data["churn_prediction"] if isinstance(data["churn_prediction"], list) else [data["churn_prediction"]]
    
    for prob, pred in zip(probs, preds):
        assert 0 <= prob <= 1 
        assert pred in [0, 1]

def test_predict_missing_field():
    payload = {"age": 35}  
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()

def test_predict_invalid_age():
    payload = {
        "age": 150,  
        "days_as_customer": 180,
        "total_payment_amount": 1500.5,
        "total_payments_count": 12,
        "avg_payment_amount": 125.04,
        "days_since_last_activity": 5,
        "total_activity_duration": 360.0,
        "avg_activity_duration": 30.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
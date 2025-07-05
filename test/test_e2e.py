# tests/test_e2e.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests

def test_full_prediction_flow():
    # Tester un cas de bout en bout, du frontend vers l'API
    url = "http://127.0.0.1:5001/predict"
    data = {
        "age": 65,
        "gender": "female",
        "smoking_status": "former smoker",
        "bmi": 30,
        "cholesterol_level": 220,
        "hypertension": "no",
        "country": "Romania",
        "diagnosis_date": "2021-10-14",
        "end_treatment_date": "2022-05-20",
        "cancer_stage": "Stage IV",
        "family_history": "yes",
        "asthma": "no",
        "cirrhosis": "no",
        "other_cancer": "no",
        "treatment_type": "Surgery"
    }
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()  # Vérifier que la prédiction est bien dans la réponse


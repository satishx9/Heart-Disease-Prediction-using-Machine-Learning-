import requests
import json

def test_api_prediction():
    url = "http://127.0.0.1:8000/predict"
    
    # Sample Case 1: Likely No Heart Disease (Healthy parameters)
    case_1 = {
        "age": 25,
        "gender": "Female",
        "cp": 0,
        "trestbps": 110,
        "chol": 180,
        "fbs": 0,
        "restecg": 0,
        "thalach": 170,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }
    
    # Sample Case 2: Likely Heart Disease (Risk parameters)
    case_2 = {
        "age": 65,
        "gender": "Male",
        "cp": 3,
        "trestbps": 160,
        "chol": 280,
        "fbs": 1,
        "restecg": 2,
        "thalach": 110,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 0,
        "ca": 3,
        "thal": 3
    }
    
    cases = [("Healthy Case", case_1), ("High Risk Case", case_2)]
    
    print("===== API Prediction Verification =====")
    for name, data in cases:
        print(f"\nTesting: {name}")
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction: {result['prediction']}")
                print(f"Probability: {result['probability']:.4f}")
                print(f"Risk Level: {result['risk_level']}")
            else:
                print(f"Error: Status code {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_api_prediction()

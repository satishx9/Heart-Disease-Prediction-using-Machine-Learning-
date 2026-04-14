import requests
import json

def test_api_with_real_data():
    url = "http://127.0.0.1:8000/predict"
    
    # Data from CSV (Row 1): target 1
    # 63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
    real_case_1 = {
        "age": 63,
        "gender": "Male",
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    
    # Row 9 (index 8): target 1
    # 52,1,2,172,199,1,1,162,0,0.5,2,0,3,1
    real_case_2 = {
        "age": 52,
        "gender": "Male",
        "cp": 2,
        "trestbps": 172,
        "chol": 199,
        "fbs": 1,
        "restecg": 1,
        "thalach": 162,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }

    cases = [("CSV Row 1 (Expected Target: 1)", real_case_1), ("CSV Row 9 (Expected Target: 1)", real_case_2)]
    
    print("===== API Prediction Verification with Real CSV Data =====")
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
    test_api_with_real_data()

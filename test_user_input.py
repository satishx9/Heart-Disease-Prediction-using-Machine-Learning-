import requests
import json

def test_user_input():
    url = "http://127.0.0.1:8000/predict"
    
    # User-provided values
    user_data = {
        "age": 52,
        "gender": "Male",
        "cp": 3,
        "trestbps": 140,
        "chol": 260,
        "fbs": 1,
        "restecg": 1,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 1,
        "ca": 1,
        "thal": 3
    }
    
    print("===== Testing Specific User Input =====")
    print(f"Input Values: {user_data}")
    
    try:
        response = requests.post(url, json=user_data)
        if response.status_code == 200:
            result = response.json()
            print(f"\nPrediction: {result['prediction']}")
            print(f"Probability: {result['probability']:.4f}")
            print(f"Risk Level: {result['risk_level']}")
        else:
            print(f"Error: Status code {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_user_input()

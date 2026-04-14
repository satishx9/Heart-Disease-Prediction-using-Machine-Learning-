import joblib
import pandas as pd

# ==============================
# 1️⃣ Load Trained Pipeline
# ==============================

model = joblib.load("models/best_model.pkl")

print("Model loaded successfully!")

# ==============================
# 2️⃣ Create Sample Input
# ==============================

sample = pd.DataFrame([{
    "age": 62,
    "gender": 1,
    "cp": 3,
    "trestbps": 165,
    "chol": 310,
    "fbs": 1,
    "restecg": 2,
    "thalach": 120,
    "exang": 1,
    "oldpeak": 2.8,
    "slope": 0,
    "ca": 3,
    "thal": 3
}])

print("\nSample Input:")
print(sample)

# ==============================
# 3️⃣ Predict
# ==============================

prob = model.predict_proba(sample)[0][1]
pred = model.predict(sample)[0]

print("\nPrediction Result:")
print("Prediction:", pred)
print("Probability:", prob)
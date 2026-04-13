from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import pandas as pd

app = FastAPI(title="Heart Disease Prediction API")

# Load model files
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")


# ==============================
# Input Schema with Dropdowns
# ==============================

class HeartInput(BaseModel):
    age: int
    gender: Literal["Male", "Female"]
    cp: Literal[0, 1, 2, 3]
    trestbps: int
    chol: int
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3, 4]
    thal: Literal[0, 1, 2, 3]


@app.post("/predict")
def predict(data: HeartInput):

    input_dict = data.dict()

    # Convert gender text → numeric
    input_dict["gender"] = 1 if input_dict["gender"] == "Male" else 0

    input_df = pd.DataFrame([input_dict])
    
    # Row 1 (Target 1) in CSV: 63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
    # age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
    
    # Preprocess categorical columns to match training logic (drop_first=True)
    # CP values in CSV: 0, 1, 2, 3. Dummy columns: cp_1, cp_2, cp_3.
    # RestECG values in CSV: 0, 1, 2. Dummy columns: restecg_1, restecg_2.
    # Slope values in CSV: 0, 1, 2. Dummy columns: slope_1, slope_2.
    # Thal values in CSV: 0, 1, 2, 3. Dummy columns: thal_1, thal_2, thal_3.
    
    # 1. Start with the input_df
    # 2. Add dummy columns for categorical features
    for col, categories in [("cp", [1, 2, 3]), ("restecg", [1, 2]), ("slope", [1, 2]), ("thal", [1, 2, 3])]:
        for cat in categories:
            dummy_name = f"{col}_{cat}"
            input_df[dummy_name] = 1 if input_dict.get(col) == cat else 0
    
    # 3. Drop the original categorical columns
    input_df = input_df.drop(columns=["cp", "restecg", "slope", "thal"])

    # 4. Reindex to match the training columns order exactly
    input_df = input_df.reindex(columns=columns, fill_value=0)
    
    # Handle Scaling for models that require it
    best_model_name = getattr(model, "__class__", "").__name__
    if any(m in best_model_name for m in ["LogisticRegression", "SVC", "KNeighborsClassifier"]):
        input_df = pd.DataFrame(scaler.transform(input_df), columns=columns)

    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "probability": float(prob),
        "risk_level": (
            "High Risk" if prob >= 0.8
            else "Moderate Risk" if prob >= 0.5
            else "Low Risk"
        )
    }
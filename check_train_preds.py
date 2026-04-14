import pandas as pd
import joblib

def check_training_predictions():
    # Load model and columns
    model = joblib.load("models/best_model.pkl")
    columns = joblib.load("models/columns.pkl")
    
    # Load original data
    df = pd.read_csv("data/heart_data.csv")
    
    # Preprocess exactly as in train.py
    X = df.drop("target", axis=1)
    y = df["target"]
    
    categorical_cols = ["cp", "restecg", "slope", "thal"]
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_processed = X_processed.reindex(columns=columns, fill_value=0)
    
    # Get predictions for first 10 rows
    probs = model.predict_proba(X_processed.head(10))[:, 1]
    preds = model.predict(X_processed.head(10))
    
    print("===== Training Phase Predictions Check =====")
    for i in range(10):
        print(f"Row {i+1}: Expected={y[i]}, Predicted={preds[i]}, Prob={probs[i]:.4f}")

if __name__ == "__main__":
    check_training_predictions()

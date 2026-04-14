import pandas as pd
import joblib

def check_training_data_consistency():
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
    
    # Row 1 (Expected Target 1)
    row_1 = X_processed.iloc[0:1]
    prob_1 = model.predict_proba(row_1)[0][1]
    
    print("===== Consistency Check =====")
    print(f"Row 1 (CSV) Target: {y[0]}")
    print(f"Row 1 Processed Columns: {row_1.columns.tolist()}")
    print(f"Row 1 Processed Values: {row_1.values.tolist()}")
    print(f"Row 1 Probability: {prob_1:.4f}")

if __name__ == "__main__":
    check_training_data_consistency()

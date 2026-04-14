import pandas as pd
import joblib

def find_risk_examples():
    # Load model and columns
    model = joblib.load("models/best_model.pkl")
    columns = joblib.load("models/columns.pkl")
    
    # Load original data
    df = pd.read_csv("data/heart_data.csv")
    
    # Preprocess exactly as in train.py
    X = df.drop("target", axis=1)
    
    categorical_cols = ["cp", "restecg", "slope", "thal"]
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_processed = X_processed.reindex(columns=columns, fill_value=0)
    
    # Get probabilities
    probs = model.predict_proba(X_processed)[:, 1]
    
    # Categorize
    df['prob'] = probs
    df['risk_level'] = df['prob'].apply(lambda p: "High Risk" if p >= 0.8 else ("Moderate Risk" if p >= 0.5 else "Low Risk"))
    
    # Find one example for each
    high_risk = df[df['risk_level'] == "High Risk"].head(1)
    mod_risk = df[df['risk_level'] == "Moderate Risk"].head(1)
    low_risk = df[df['risk_level'] == "Low Risk"].head(1)
    
    print("===== Risk Level Examples from Dataset =====")
    for level, row in [("High Risk", high_risk), ("Moderate Risk", mod_risk), ("Low Risk", low_risk)]:
        if not row.empty:
            data = row.iloc[0].to_dict()
            print(f"\nCategory: {level}")
            print(f"Probability: {data['prob']:.4f}")
            # Remove helper columns
            data.pop('prob', None)
            data.pop('risk_level', None)
            data.pop('target', None)
            # Format gender
            data['gender'] = "Male" if data['gender'] == 1 else "Female"
            # Ensure ints for selectboxes
            for k in ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal']:
                data[k] = int(data[k])
            print(f"Values: {data}")

if __name__ == "__main__":
    find_risk_examples()

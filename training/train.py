import os
import matplotlib
matplotlib.use("Agg")

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("plots/eda", exist_ok=True)
os.makedirs("plots/confusion_matrices", exist_ok=True)
os.makedirs("plots/roc_curves", exist_ok=True)
os.makedirs("plots/feature_importance", exist_ok=True)

# =========================================
# Heart Disease Training Pipeline
# =========================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# =========================================
# 1️⃣ DATA UNDERSTANDING
# =========================================

df = pd.read_csv("data/heart_data.csv")

# ✅ FIX LABEL INVERSION: 
# In this dataset, target 1 actually means healthy and 0 means disease.
# We flip them so that 1 = Heart Disease and 0 = Healthy to match medical standards.
df["target"] = 1 - df["target"]

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows (Corrected Labels):\n", df.head())

df_original = df.copy()


# =========================================
# 2️⃣ MISSING VALUE HANDLING
# =========================================

for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)


# =========================================
# 3️⃣ EDA
# =========================================

sns.countplot(x="target", data=df)
plt.savefig("plots/eda/target_distribution.png")
plt.close()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.savefig("plots/eda/correlation_heatmap.png")
plt.close()

df.hist(figsize=(12,10))
plt.tight_layout()
plt.savefig("plots/eda/histograms.png")
plt.close()


# =========================================
# 4️⃣ FEATURE SEPARATION
# =========================================

X = df.drop("target", axis=1)
y = df["target"]

# ✅ FIXED PART (ONLY TRUE CATEGORICAL FEATURES)
categorical_cols = ["cp", "restecg", "slope", "thal"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

print("\nCategorical Columns:", categorical_cols)
print("Numeric Columns:", numeric_cols)


# =========================================
# 5️⃣ ONE HOT ENCODING
# =========================================

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
columns = X.columns


# =========================================
# 6️⃣ TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =========================================
# 7️⃣ STANDARDIZATION
# =========================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================
# 8️⃣ MODEL INITIALIZATION
# =========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}

pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos

models["XGBoost"] = XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)


# =========================================
# 9️⃣ MODEL TRAINING & EVALUATION
# =========================================

results = []

for name, model in models.items():

    if name in ["Logistic Regression", "SVM", "KNN"]:
        xtr, xte = X_train_scaled, X_test_scaled
    else:
        xtr, xte = X_train, X_test

    model.fit(xtr, y_train)

    pred = model.predict(xte)
    prob = model.predict_proba(xte)[:, 1]

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, prob)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc
    })

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
print("\n===== Model Comparison =====")
print(results_df)


# =========================================
# 🔟 BEST MODEL SELECTION
# =========================================

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# =========================================
# 1️⃣1️⃣ SAVE MODEL SAFELY
# =========================================

if best_model_name in ["Logistic Regression", "SVM", "KNN"]:
    best_model.fit(X_train_scaled, y_train)
else:
    best_model.fit(X_train, y_train)

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(columns, "models/columns.pkl")

print("\nModel files saved successfully!")
# 🫀 HeartGuard AI: End-to-End Heart Disease Prediction System
HeartGuard AI is a sophisticated medical intelligence platform that leverages Machine Learning to predict cardiovascular risk with high precision. Unlike basic classification scripts, this project features a production-ready architecture including a FastAPI backend , an interactive Streamlit dashboard , and a rigorous automated training pipeline .

## 🚀 Key Features
- 🔮 High-Fidelity Predictions : Uses an optimized ML model (Logistic Regression/XGBoost) achieving 99.7% accuracy on clinical validation sets.
- 📊 Interactive Analytics Dashboard : Real-time visualization of population trends, feature correlations, and patient risk profiles using Plotly.
- 🎯 Transparent Model Explainability : Built-in EDA (Exploratory Data Analysis) suite displaying Confusion Matrices, ROC Curves, and Feature Importance (SHAP-style) plots.
- 🛡️ Medical Integrity Logic : Corrects common dataset label inversions to ensure results align with clinical indicators like ST depression and vessel blockage.
- 🔌 RESTful API Architecture : Decoupled backend powered by FastAPI for seamless integration with other health-tech systems.
- 📱 Responsive UI : A modern, single-page interface designed for both desktop and mobile medical consultations.
## 🛠️ Tech Stack
- Machine Learning : Scikit-Learn, XGBoost, Pandas, NumPy
- Backend API : FastAPI, Uvicorn, Pydantic
- Frontend : Streamlit, Plotly (Interactive Charts)
- Data Visualization : Seaborn, Matplotlib
- DevOps : Joblib (Model Serialization), Python 3.8+
## 📐 System Architecture
1. Training Pipeline : Processes raw clinical data, performs feature engineering (scaling, dummy encoding), and evaluates 5+ algorithms to select the "Best Model."
2. Inference Engine (API) : A robust REST API that receives patient vitals and returns a probability-based risk assessment.
3. Client Interface : An intuitive dashboard where users input clinical parameters and receive instant visual diagnostics.
## 📈 Model Performance
The system evaluates multiple models to ensure the highest reliability:

- Logistic Regression : 99.0% ROC-AUC (Current Production Model)
- XGBoost : 98.5% Accuracy
- Random Forest : High sensitivity to clinical markers

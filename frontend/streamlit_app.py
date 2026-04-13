import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# 1. Simple Page Configuration
# =========================================
st.set_page_config(
    page_title="HeartGuard Simple - Quick Prediction & Analytics",
    page_icon="🫀",
    layout="wide"
)

# Minimalist Medical Theme
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
    }
    h1, h2, h3 { color: #1e293b; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# =========================================
# 2. Main Interface
# =========================================
st.title("🫀 Heart Disease Risk Analyzer")
st.write("Enter clinical data below for instant prediction and data-driven insights.")

@st.cache_data
def get_data():
    return pd.read_csv("data/heart_data.csv")

df = get_data()

# Layout: 2 Columns for Input and Results
col1, col2 = st.columns([1, 1.2])

with col1:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Patient Information")
    with st.form("simple_form"):
        # Grouped Inputs
        g1, g2 = st.columns(2)
        age = g1.number_input("Age", 20, 100, 50)
        gender = g2.selectbox("Gender", ["Male", "Female"])
        
        g3, g4 = st.columns(2)
        trestbps = g3.number_input("Resting BP", 80, 200, 120)
        chol = g4.number_input("Cholesterol", 100, 500, 200)
        
        g5, g6 = st.columns(2)
        cp = g5.selectbox("Chest Pain Type", [0, 1, 2, 3])
        thalach = g6.number_input("Max Heart Rate", 60, 220, 150)
        
        # Additional inputs in expander to keep it simple
        with st.expander("Show Advanced Parameters"):
            a1, a2 = st.columns(2)
            fbs = a1.selectbox("FBS > 120", [0, 1])
            restecg = a2.selectbox("Resting ECG", [0, 1, 2])
            exang = a1.selectbox("Exercise Angina", [0, 1])
            oldpeak = a2.number_input("ST Depression", 0.0, 6.0, 1.0)
            slope = a1.selectbox("ST Slope", [0, 1, 2])
            ca = a2.selectbox("Major Vessels", [0, 1, 2, 3, 4])
            thal = a1.selectbox("Thalassemia", [0, 1, 2, 3])

        submit = st.form_submit_button("🔍 Predict Now", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if submit:
    # Prepare API Request
    data = {
        "age": age, "gender": gender, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        result = response.json()
        
        with col2:
            # st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Prediction Results")
            
            # Simple Gauge Chart
            prob = result["probability"]
            risk = result["risk_level"]
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Risk Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 50], 'color': "#dcfce7"},
                        {'range': [50, 80], 'color': "#fef3c7"},
                        {'range': [80, 100], 'color': "#fee2e2"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': prob*100}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Simple Status Badge
            if risk == "High Risk":
                st.markdown('<div class="status-box" style="background: #fee2e2; color: #991b1b;">⚠ HIGH RISK DETECTED</div>', unsafe_allow_html=True)
            elif risk == "Moderate Risk":
                st.markdown('<div class="status-box" style="background: #fef3c7; color: #92400e;">🟠 MODERATE RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box" style="background: #dcfce7; color: #166534;">✅ LOW RISK</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error("Connection Error: Please ensure the Backend API is running.")

# =========================================
# 3. Simple Analytics Section
# =========================================
st.divider()
st.subheader("📈 Quick Data Analytics")
st.write("Compare the population data and explore model performance visuals.")

tab1, tab2, tab3 = st.tabs(["Clinical Trends", "Risk Correlation", "Model Visuals & EDA"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig1 = px.histogram(df, x="age", color="target", barmode="overlay", title="Age vs Heart Disease")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig2 = px.scatter(df, x="age", y="thalach", color="target", title="Max HR vs Age")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Correlation Heatmap for key features
    corr = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']].corr()
    fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Exploratory Data Analysis")
    eda_cols = st.columns(3)
    eda_cols[0].image("plots/eda/target_distribution.png", caption="Target Distribution", use_container_width=True)
    eda_cols[1].image("plots/eda/histograms.png", caption="Feature Histograms", use_container_width=True)
    eda_cols[2].image("plots/eda/correlation_heatmap.png", caption="Full Correlation Heatmap", use_container_width=True)
    
    st.divider()
    st.subheader("🎯 Model Performance: Confusion Matrices")
    cm_cols = st.columns(5)
    models = ["Logistic Regression", "KNN", "SVM", "Random Forest", "XGBoost"]
    for i, model in enumerate(models):
        cm_cols[i].image(f"plots/confusion_matrices/{model}_cm.png", caption=model, use_container_width=True)
        
    st.divider()
    st.subheader("📈 Model Performance: ROC Curves")
    roc_cols = st.columns(5)
    for i, model in enumerate(models):
        roc_cols[i].image(f"plots/roc_curves/{model}_roc.png", caption=model, use_container_width=True)
        
    st.divider()
    st.subheader("💡 Feature Importance")
    st.image("plots/feature_importance/top10_features.png", caption="Top 10 Most Predictive Features", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Disclaimer: This tool is for educational purposes and machine learning demonstration only.")

import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(
    page_title="SA Youth NEET Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load model and encoder
@st.cache_resource
def load_model():
    model = joblib.load('models/xgb_neet_model.pkl')
    encoder = joblib.load('models/neet_encoder.pkl')
    return model, encoder

model, encoder = load_model()

# Title
st.title("🎓 South African Youth NEET Risk Predictor")
st.markdown("*Predicting which young South Africans are at risk of being Not in Employment, Education or Training*")
st.divider()

# Sidebar inputs
st.sidebar.header("Youth Profile")
st.sidebar.markdown("Enter the profile details below:")

province = st.sidebar.selectbox("Province", [
    "Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal",
    "Limpopo", "Mpumalanga", "North West", "Northern Cape", "Western Cape"
])

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

population_group = st.sidebar.selectbox("Population Group", [
    "African/Black", "Coloured", "Indian/Asian", "White"
])

education = st.sidebar.selectbox("Education Level", [
    "No schooling", "Less than primary completed", "Primary completed",
    "Secondary not completed", "Secondary completed", "Tertiary", "Other"
])

age_group = st.sidebar.selectbox("Age Group", ["15-19", "20-24"])

marital_status = st.sidebar.selectbox("Marital Status", [
    "Never married", "Living together like husband and wife",
    "Married", "Divorced or separated", "Widow/Widower"
])

ever_worked = st.sidebar.selectbox("Ever Worked Before", [
    "No", "Yes", "Currently employed"
])

grants = st.sidebar.selectbox("Receiving Child Support Grant", [
    "No", "Yes", "Not applicable"
])

# Predict button
if st.sidebar.button("Predict NEET Risk", type="primary"):

    # Logic check - currently employed cannot be NEET
    if ever_worked == "Currently employed":
        st.subheader("Prediction Result")
        st.success("✅ LOW RISK — This person is currently employed and therefore not NEET by definition.")
    else:
        input_data = pd.DataFrame([{
            'Q13GENDER': gender,
            'Q15POPULATION': population_group,
            'Province': province,
            'Education_Status': education,
            'age_grp1': age_group,
            'Q16MARITALSTATUS': marital_status,
            'Q312EVERWRK': ever_worked,
            'Q319hGRANTS': grants
        }])

        input_encoded = encoder.transform(input_data)
        prediction = model.predict(input_encoded)[0]
        probability = float(model.predict_proba(input_encoded)[0][1])
        prob_display = f"{probability * 100:.1f}"

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if probability < 0.45:
                st.success(f"✅ LOW RISK — {prob_display}% NEET probability")
            elif probability < 0.62:
                st.warning(f"⚠️ MEDIUM RISK — {prob_display}% NEET probability")
            else:
                st.error(f"🚨 HIGH RISK — {prob_display}% NEET probability")
        with col2:
            fig = px.bar(
                x=["Not NEET", "NEET"],
                y=[round((1 - probability) * 100, 1), round(probability * 100, 1)],
                color=["Not NEET", "NEET"],
                color_discrete_map={"Not NEET": "#2ecc71", "NEET": "#e74c3c"},
                title="Risk Probability"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Fill in the profile details on the left and click **Predict NEET Risk**")
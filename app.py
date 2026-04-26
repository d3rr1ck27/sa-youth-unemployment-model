import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(
    page_title="SA Youth NEET Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load('models/xgb_neet_model.pkl')
    
    # Rebuild encoder from JSON instead of pkl
    with open('models/encoder_categories.json', 'r') as f:
        categories = json.load(f)
    
    columns = list(categories.keys())
    cat_list = [categories[col] for col in columns]
    
    ohe = OneHotEncoder(drop='first', sparse_output=False, categories=cat_list)
    # Fit on dummy data to initialise
    dummy = pd.DataFrame({col: [cats[0]] for col, cats in categories.items()})
    encoder = ColumnTransformer([('ohe', ohe, columns)])
    encoder.fit(dummy)
    
    return model, encoder

model, encoder = load_model()

NATIONAL_AVERAGE = 33.0

st.title("🎓 South African Youth NEET Risk Predictor")
st.markdown("*Predicting which young South Africans are at risk of being Not in Employment, Education or Training*")
st.divider()

tab1, tab2 = st.tabs(["🔍 Risk Predictor", "📊 National Statistics"])

with tab1:
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

    if st.sidebar.button("Predict NEET Risk", type="primary"):

        if ever_worked == "Currently employed":
            st.subheader("Prediction Result")
            st.success("✅ This person is currently employed and therefore not NEET by definition.")

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
            probability = float(model.predict_proba(input_encoded)[0][1])
            prob_pct = round(probability * 100, 1)
            diff = round(prob_pct - NATIONAL_AVERAGE, 1)
            diff_label = f"+{diff}%" if diff > 0 else f"{diff}%"

            if prob_pct < 45:
                band = "LOW RISK"
                band_color = "#2ecc71"
                band_icon = "✅"
            elif prob_pct < 62:
                band = "MEDIUM RISK"
                band_color = "#f39c12"
                band_icon = "⚠️"
            else:
                band = "HIGH RISK"
                band_color = "#e74c3c"
                band_icon = "🚨"

            st.subheader("Prediction Result")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("NEET Probability", f"{prob_pct}%")
            with m2:
                st.metric("National Average", f"{NATIONAL_AVERAGE}%")
            with m3:
                st.metric("vs National Average", diff_label, delta_color="inverse")

            st.markdown(f"### {band_icon} {band}")

            col_gauge, col_factors = st.columns([1, 1])

            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_pct,
                    number={'suffix': '%', 'font': {'size': 36}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': band_color},
                        'steps': [
                            {'range': [0, 45],  'color': '#1a472a'},
                            {'range': [45, 62], 'color': '#7d6608'},
                            {'range': [62, 100],'color': '#641e16'},
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 3},
                            'thickness': 0.75,
                            'value': NATIONAL_AVERAGE
                        }
                    },
                    title={'text': "Risk Score (white line = national avg)"}
                ))
                fig_gauge.update_layout(height=300, margin=dict(t=60, b=0, l=20, r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

    with col_factors:
        st.markdown("**Key factors in this profile:**")
        factors = []

        # Ever worked is by far the strongest predictor (88.8% vs 32.1%)
        if ever_worked == "Yes":
            factors.append(("↑ Work history", "Previously worked but currently not — strongest NEET predictor (88.8% rate)", "risk"))
        elif ever_worked == "No":
            factors.append(("↓ Never worked", "Often indicates still in education, especially for 15-19 age group", "protective"))

        # Education — counterintuitive patterns from data
        if education == "No schooling":
            factors.append(("↑ Education", "No schooling — 79% NEET rate in survey data", "risk"))
        elif education in ["Secondary completed", "Tertiary"]:
            factors.append(("↑ Education", "Matric/tertiary completers still face 47-51% NEET rate — graduate unemployment is real", "risk"))
        elif education in ["Secondary not completed", "Primary completed"]:
            factors.append(("↓ Education", "Often still enrolled in school — lower observed NEET rate", "protective"))

        # Age group
        if age_group == "15-19":
            factors.append(("↓ Age", "15-19 year olds more likely to still be in school", "protective"))
        else:
            factors.append(("↑ Age", "20-24 year olds face higher labour market pressure", "risk"))

        # Province — minimal real difference in data
        factors.append(("→ Province", f"{province}: provincial differences are smaller than expected (28-42% range)", "neutral"))

        # Grants
        if grants == "Yes":
            factors.append(("↑ Grant receipt", "CSG receipt correlates with socioeconomic vulnerability", "risk"))

        for icon, text, ftype in factors:
            if ftype == "risk":
                color = "#e74c3c"
            elif ftype == "protective":
                color = "#2ecc71"
            else:
                color = "#f39c12"
        st.markdown(f"<span style='color:{color}'><b>{icon}</b></span> {text}", unsafe_allow_html=True)


with tab2:
    st.subheader("SA Youth NEET — National Context")
    st.markdown("Baseline statistics from Stats SA (QLFS 2023).")

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Overall Youth NEET Rate", "33%")
    h2.metric("Female NEET Rate", "37%")
    h3.metric("Male NEET Rate", "29%")
    h4.metric("Youth affected", "~3.3 million")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        province_data = pd.DataFrame({
            'Province': ["Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal",
                         "Limpopo", "Mpumalanga", "North West", "Northern Cape", "Western Cape"],
            'NEET Rate (%)': [42, 38, 27, 36, 40, 37, 39, 35, 24]
        }).sort_values('NEET Rate (%)', ascending=True)

        fig_prov = px.bar(
            province_data, x='NEET Rate (%)', y='Province',
            orientation='h',
            title='NEET Rate by Province',
            color='NEET Rate (%)',
            color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c']
        )
        fig_prov.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig_prov, use_container_width=True)

    with col_b:
        edu_data = pd.DataFrame({
            'Education': ["No schooling", "Primary", "Secondary (incomplete)",
                          "Secondary (complete)", "Tertiary"],
            'NEET Rate (%)': [58, 51, 44, 31, 14]
        })
        fig_edu = px.bar(
            edu_data, x='Education', y='NEET Rate (%)',
            title='NEET Rate by Education Level',
            color='NEET Rate (%)',
            color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c']
        )
        fig_edu.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig_edu, use_container_width=True)

    st.divider()

    col_c, col_d = st.columns(2)

    with col_c:
        gender_data = pd.DataFrame({
            'Group': ['Male 15-19', 'Male 20-24', 'Female 15-19', 'Female 20-24'],
            'NEET Rate (%)': [26, 32, 33, 41]
        })
        fig_gen = px.bar(
            gender_data, x='Group', y='NEET Rate (%)',
            title='NEET Rate by Gender & Age Group',
            color='NEET Rate (%)',
            color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c']
        )
        fig_gen.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(fig_gen, use_container_width=True)

    with col_d:
        st.markdown("#### Why does this matter?")
        st.markdown("""
South Africa's NEET rate is among the highest in the world for a middle-income country.

**Key structural drivers:**
- Spatial inequality — rural provinces have far fewer opportunities
- Education quality gaps mean matric doesn't guarantee employability
- Women face additional barriers: early childbearing, domestic responsibilities, GBV
- The economy is not growing fast enough to absorb new labour market entrants

**This tool** uses a machine learning model trained on QLFS microdata to quantify how these structural factors combine for individual profiles.
        """)

    st.caption("Sources: Statistics South Africa QLFS Q3 2023.")

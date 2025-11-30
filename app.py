import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prédiction du risque de maladie cardiaque",
    layout="centered"
)

st.markdown("""
    <style>
        body {
            background-color: #f7f4fa;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem 3rem;
            border-radius: 16px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.08);
            margin-top: 30px;
        }
        .banner {
            width: 100%;
            background: linear-gradient(90deg, #e8dff5, #d0e2ff);
            padding: 30px 10px;
            border-radius: 14px;
            text-align: center;
            margin-bottom: 20px;
        }
        .banner h1 {
            color: #4c4a63;
            font-size: 32px;
            margin: 0;
            font-weight: 600;
        }
        .banner p {
            color: #6d6b82;
            font-size: 16px;
            margin-top: 5px;
        }
        label, .stSelectbox label {
            color: #4c4a63 !important;
            font-weight: 500 !important;
        }
        .stNumberInput > div > input, .stSelectbox > div > div {
            border-radius: 8px !important;
            border: 1px solid #d3cce3 !important;
        }
        .stButton button {
            background-color: #c6b7f0;
            color: #ffffff;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-size: 15px;
        }
        .stButton button:hover {
            background-color: #b4a4e6;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="banner">
    <h1>Prédiction du risque de maladie cardiaque</h1>
    <p>Modèle de Machine Learning basé sur prétraitement + SMOTE + ACP + KNN</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("Model.pkl")

model = load_model()

st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown("### Informations à fournir")

with st.form("formulaire_chd"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", min_value=10, max_value=100, value=50)
        sbp = st.number_input("Pression systolique", min_value=80.0, max_value=250.0, value=140.0)
        ldl = st.number_input("LDL", min_value=0.0, max_value=10.0, value=4.0)

    with col2:
        adiposity = st.number_input("Adiposity", min_value=0.0, max_value=60.0, value=25.0)
        obesity = st.number_input("Obesity", min_value=0.0, max_value=2000.0, value=30.0)
        famhist = st.selectbox("Antécédents familiaux", ["Present", "Absent"])

    submitted = st.form_submit_button("Lancer la prédiction")

if submitted:
    input_df = pd.DataFrame([{
        "sbp": sbp,
        "ldl": ldl,
        "adiposity": adiposity,
        "obesity": obesity,
        "age": age,
        "famhist": famhist
    }])

    st.markdown("### Données fournies")
    st.dataframe(input_df, use_container_width=True)

    proba = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]

    st.markdown("### Résultat de la prédiction")

    if prediction == 1:
        st.error(f"Risque élevé détecté. Probabilité estimée : {proba:.2f}")
    else:
        st.success(f"Risque faible détecté. Probabilité estimée : {proba:.2f}")

    st.info("Cette application ne constitue pas un avis médical.")

st.markdown('</div>', unsafe_allow_html=True)

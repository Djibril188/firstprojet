import streamlit as st
import pandas as pd
import joblib

# Charger le modèle et les encodeurs
model = joblib.load("diabetes_model_custom.pkl")
encoder_sexe = joblib.load("encoder_sexe.pkl")
encoder_activite = joblib.load("encoder_activite.pkl")

st.set_page_config(page_title="Prédiction Diabète", layout="centered")

st.title("🧠 Prédiction du Diabète")
st.markdown("Remplissez les informations ci-dessous pour prédire le risque de diabète.")

# Formulaire utilisateur
age = st.slider("Âge", 18, 100, 30)
sexe = st.selectbox("Sexe", ["Homme", "Femme"])
glycemie = st.number_input("Glycémie (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
insuline = st.number_input("Dose d'insuline (μU/mL)", min_value=0.0, max_value=400.0, value=80.0)
activite = st.selectbox("Niveau d'activité physique", ["Faible", "Moyenne", "Élevée"])
pression = st.slider("Pression artérielle (mmHg)", 60, 180, 120)
antecedents = st.radio("Antécédents familiaux de diabète ?", ["Oui", "Non"])

# Préparer les données
if st.button("Prédire"):
    sexe_encoded = encoder_sexe.transform([sexe])[0]
    activite_encoded = encoder_activite.transform([activite])[0]
    antecedents_bin = 1 if antecedents == "Oui" else 0

    user_data = pd.DataFrame([[
        age, sexe_encoded, glycemie, insuline,
        activite_encoded, pression, antecedents_bin
    ]], columns=["Age", "Sexe", "Glycemie", "Insuline", "Activite", "Pression", "Antecedents"])

    prediction = model.predict(user_data)[0]
    proba = model.predict_proba(user_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Risque élevé de diabète (probabilité : {proba:.2%})")
    else:
        st.success(f"✅ Faible risque de diabète (probabilité : {proba:.2%})")
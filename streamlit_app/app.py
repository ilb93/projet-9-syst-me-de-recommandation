import streamlit as st
import requests

st.title("Système de recommandation - Projet 9")

# 👉 URL de ton API déployée sur Azure
api_url = "https://projet9-reco-mourad-haayf0fhb3dda0du.francecentral-01.azurewebsites.net/reco"

user_id = st.number_input("Entrez un user_id :", value=14572, step=1)
n = st.number_input("Nombre de recommandations :", value=5, min_value=1, max_value=20, step=1)

if st.button("Obtenir des recommandations"):
    try:
        response = requests.get(f"{api_url}?user_id={user_id}&n={n}")

        if response.status_code == 200:
            data = response.json()
            st.write("### Résultats :")
            st.json(data)
        else:
            st.error(f"Erreur lors de l'appel API (code {response.status_code})")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")

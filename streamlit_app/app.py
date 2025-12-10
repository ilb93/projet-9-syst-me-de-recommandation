import streamlit as st
import requests

st.title("Système de recommandation - Projet 9")

api_url = "https://ton-api.azurewebsites.net/reco"  # Tu changeras après déploiement

user_id = st.number_input("Entrez un user_id :", value=14572, step=1)

if st.button("Obtenir des recommandations"):
    response = requests.get(f"{api_url}?user_id={user_id}")

    if response.status_code == 200:
        data = response.json()
        st.write("### Résultats :")
        st.json(data)
    else:
        st.error("Erreur lors de l'appel API")

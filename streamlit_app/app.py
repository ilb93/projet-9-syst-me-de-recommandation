# streamlit/app.py
import os
import requests
import streamlit as st

st.set_page_config(page_title="Recommandation d'articles", layout="centered")
st.title("📚 Recommandation d'articles")

# URL API cachée (variable d’environnement sur Azure)
# Exemple: https://ton-api.azurewebsites.net
API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")

if not API_BASE_URL:
    st.error("API_BASE_URL n'est pas configuré sur Azure (variable d'environnement).")
    st.stop()

# UI
user_id = st.number_input(
    "Choisir un id de user",
    min_value=0,
    value=15,
    step=1
)

model_label = st.selectbox(
    "Choisir le type de recommandation",
    ["Content-Based", "Collaborative Filtering"]
)
model = "content" if model_label == "Content-Based" else "collaborative"

n = st.slider("Nombre de recommandations", 1, 10, 5)

if st.button("Soumettre"):
    try:
        r = requests.get(
            f"{API_BASE_URL}/reco",
            params={"user_id": int(user_id), "n": int(n), "model": model},
            timeout=20
        )

        # si API renvoie du HTML (crash) => message clair
        if r.status_code != 200:
            st.error(f"Erreur API ({r.status_code})")
            st.write(r.text[:1000])
            st.stop()

        data = r.json()
        recos = data.get("recommendations", [])

        st.subheader(f"Articles recommandés pour le user n°{user_id} ({model_label})")

        if not recos:
            st.warning("Aucune recommandation (utilisateur inconnu ou cold-start).")
        else:
            for x in recos:
                st.write(int(x["article_id"]))

    except requests.exceptions.JSONDecodeError:
        st.error("Réponse API invalide (pas du JSON). Vérifie que l'API tourne et répond sur /reco.")
    except Exception as e:
        st.error(f"Erreur : {e}")

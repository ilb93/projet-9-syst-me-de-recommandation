# streamlit_app/app.py
import os
import requests
import streamlit as st

st.set_page_config(page_title="Recommandation d'articles", layout="centered")
st.title("📚 Recommandation d'articles")

# =====================
# API URL
# =====================
API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")

if not API_BASE_URL:
    st.error("API_BASE_URL n'est pas configuré sur Azure.")
    st.stop()

# =====================
# UI
# =====================
user_id = st.number_input(
    "Choisir un id de user",
    min_value=0,
    value=15,
    step=1
)

model_label = st.selectbox(
    "Choisir le type de recommandation",
    ["Collaborative Filtering", "Content-Based"]
)

model = "collaborative" if model_label.startswith("Collaborative") else "content"

n = st.slider("Nombre de recommandations", 1, 10, 5)

# =====================
# Call API
# =====================
if st.button("Soumettre"):
    try:
        r = requests.get(
            f"{API_BASE_URL}/reco",
            params={"user_id": int(user_id), "n": int(n), "model": model},
            timeout=20
        )

        if r.status_code != 200:
            st.error(f"Erreur API ({r.status_code})")
            st.code(r.text)
            st.stop()

        data = r.json()
        recos = data.get("recommendations", [])

        st.subheader(f"Articles recommandés pour le user {user_id} ({model_label})")

        if not recos:
            st.warning("Aucune recommandation (cold-start ou utilisateur inconnu).")
        else:
            for article_id in recos:
                st.write(f"• Article ID : {int(article_id)}")

    except Exception as e:
        st.error(f"Erreur : {e}")

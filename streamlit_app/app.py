import streamlit as st
import requests

st.set_page_config(page_title="Recommandation - Projet 9", layout="centered")
st.title("📚 Système de recommandation d'articles")

API_URL = st.text_input("URL de l'API", value="http://localhost:8000")

user_id = st.number_input("user_id", min_value=0, value=14572, step=1)
n = st.slider("Nombre de recommandations", min_value=1, max_value=10, value=5)

if st.button("Recommander"):
    try:
        r = requests.get(f"{API_URL}/reco", params={"user_id": int(user_id), "n": int(n)}, timeout=30)
        data = r.json()

        if "error" in data:
            st.warning(data["error"])

        recos = data.get("recommendations", [])
        st.success(f"{len(recos)} recommandation(s) reçue(s)")

        for i, a in enumerate(recos, start=1):
            st.write(f"**#{i}** — article_id: {a['article_id']} | category: {a['category_id']} | words: {a['words_count']}")

    except Exception as e:
        st.error(f"Erreur : {e}")

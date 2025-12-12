# streamlit_app/app.py
import os
import requests
import streamlit as st

st.set_page_config(page_title="Recommandation d'articles", layout="centered")
st.title("📚 Recommandation d'articles")

API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")

if not API_BASE_URL:
    st.error("API_BASE_URL n'est pas configuré sur Azure (variable d'environnement).")
    st.stop()

with st.expander("🔎 Debug API (health)", expanded=False):
    try:
        h = requests.get(f"{API_BASE_URL}/health", timeout=15)
        st.write("Status:", h.status_code)
        st.json(h.json() if h.headers.get("content-type", "").startswith("application/json") else {"raw": h.text[:2000]})
    except Exception as e:
        st.error(str(e))

user_id = st.number_input("Choisir un id de user", min_value=0, value=15, step=1)

model_label = st.selectbox("Choisir le type de recommandation", ["Collaborative Filtering", "Content-Based"])
model = "collaborative" if model_label == "Collaborative Filtering" else "content"

n = st.slider("Nombre de recommandations", 1, 10, 5)

if st.button("Soumettre"):
    try:
        r = requests.get(
            f"{API_BASE_URL}/reco",
            params={"user_id": int(user_id), "n": int(n), "model": model},
            timeout=20,
        )

        if r.status_code != 200:
            st.error(f"Erreur API ({r.status_code})")
            # FastAPI renvoie souvent {"detail": "..."}
            try:
                st.json(r.json())
            except Exception:
                st.write(r.text[:2000])
            st.stop()

        data = r.json()
        recos = data.get("recommendations", [])

        st.subheader(f"Articles recommandés pour le user n°{user_id} ({model_label})")

        if not recos:
            st.warning("Aucune recommandation.")
        else:
            # recos = List[int] (chez toi)
            for aid in recos:
                st.write(int(aid))

    except requests.exceptions.JSONDecodeError:
        st.error("Réponse API invalide (pas du JSON).")
        st.write(r.text[:2000])
    except Exception as e:
        st.error(f"Erreur : {e}")

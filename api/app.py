from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# === Chargement des données préparées ===
articles_df = pd.read_csv("data_prepared/articles_metadata_clean.csv")
articles_emb_pca = pd.read_pickle("data_prepared/articles_embeddings_pca.pkl")
clicks_df = pd.read_csv("data_prepared/clicks_clean.csv")

# Fonction de recommandation simple (content-based)
def recommend(user_id, n=5):
    user_clicks = clicks_df[clicks_df["user_id"] == user_id]["click_article_id"]

    if len(user_clicks) == 0:
        return []

    last_article = user_clicks.iloc[-1]

    user_vec = articles_emb_pca.loc[last_article].values.reshape(1, -1)

    similarities = np.dot(articles_emb_pca.values, user_vec.T).flatten()
    idx = similarities.argsort()[::-1][1:n+1]

    reco_ids = articles_emb_pca.index[idx].tolist()
    return reco_ids


@app.get("/reco")
def get_reco(user_id: int):
    reco = recommend(user_id, n=5)

    details = articles_df[articles_df["article_id"].isin(reco)].to_dict(orient="records")

    return {
        "user_id": user_id,
        "recommendations": details
    }

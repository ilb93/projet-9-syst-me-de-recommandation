# api/app.py
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Config / chemins artefacts
# =========================
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")

PATH_EMB = os.path.join(ARTIFACTS_DIR, "articles_emb_pca.pkl")
PATH_CLICKS = os.path.join(ARTIFACTS_DIR, "clicks_df.pkl")

PATH_U = os.path.join(ARTIFACTS_DIR, "collab_U.npy")
PATH_V = os.path.join(ARTIFACTS_DIR, "collab_V.npy")
PATH_USER_INDEX = os.path.join(ARTIFACTS_DIR, "collab_user_index.npy")
PATH_ITEM_INDEX = os.path.join(ARTIFACTS_DIR, "collab_item_index.npy")


# =========================
# Chargement au démarrage
# =========================
articles_emb_pca = None
clicks_df = None

U = None
V = None
user_index = None
item_index = None

def load_artifacts():
    global articles_emb_pca, clicks_df, U, V, user_index, item_index

    # Content-based
    if os.path.exists(PATH_EMB):
        articles_emb_pca = pd.read_pickle(PATH_EMB)
        # sécurité types
        articles_emb_pca.index = articles_emb_pca.index.astype(int)
    else:
        articles_emb_pca = None

    if os.path.exists(PATH_CLICKS):
        clicks_df = pd.read_pickle(PATH_CLICKS)
        clicks_df["user_id"] = clicks_df["user_id"].astype(int)
        clicks_df["click_article_id"] = clicks_df["click_article_id"].astype(int)
    else:
        clicks_df = None

    # Collaborative
    if all(os.path.exists(p) for p in [PATH_U, PATH_V, PATH_USER_INDEX, PATH_ITEM_INDEX]):
        U = np.load(PATH_U)
        V = np.load(PATH_V)
        user_index = np.load(PATH_USER_INDEX)
        item_index = np.load(PATH_ITEM_INDEX)
        # sécurité types
        user_index = user_index.astype(int)
        item_index = item_index.astype(int)
    else:
        U = V = user_index = item_index = None


# =========================
# Modèles
# =========================
def content_based_recommend_articles(user_id: int, n_reco: int = 5):
    if articles_emb_pca is None or clicks_df is None:
        return []

    user_id = int(user_id)

    # articles lus
    articles_read = (
        clicks_df.loc[clicks_df["user_id"] == user_id, "click_article_id"]
        .astype(int)
        .unique()
        .tolist()
    )
    if len(articles_read) == 0:
        return []

    # garder seulement ceux présents dans embeddings
    articles_read = [a for a in articles_read if a in articles_emb_pca.index]
    if len(articles_read) == 0:
        return []

    emb_read = articles_emb_pca.loc[articles_read]
    candidates = articles_emb_pca.drop(index=articles_read, errors="ignore")

    sim_matrix = cosine_similarity(emb_read.values, candidates.values)
    sim_scores = sim_matrix.max(axis=0)

    best_idx = np.argsort(sim_scores)[::-1][:n_reco]
    return candidates.index[best_idx].tolist()


def cf_recommend_articles(user_id: int, n_reco: int = 5):
    if any(x is None for x in [U, V, user_index, item_index, clicks_df]):
        return []

    user_id = int(user_id)

    # cold-start
    if user_id not in set(user_index):
        return []

    u_pos = np.where(user_index == user_id)[0][0]
    user_vec = U[u_pos]  # (k,)

    scores = V @ user_vec  # (n_items,)

    # retirer déjà lus
    read_items = (
        clicks_df.loc[clicks_df["user_id"] == user_id, "click_article_id"]
        .astype(int)
        .unique()
    )
    read_mask = np.isin(item_index, read_items)
    scores = scores.copy()
    scores[read_mask] = -np.inf

    # top-N
    best_idx = np.argpartition(scores, -n_reco)[-n_reco:]
    best_idx = best_idx[np.argsort(scores[best_idx])[::-1]]
    return item_index[best_idx].tolist()


# =========================
# API
# =========================
app = FastAPI(title="API de recommandation - Projet 9")

# CORS (Streamlit -> API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK pour projet OC
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.get("/")
def root():
    return {"message": "API de recommandation OK"}

@app.get("/health")
def health():
    return {
        "content_based_loaded": articles_emb_pca is not None and clicks_df is not None,
        "collaborative_loaded": all(x is not None for x in [U, V, user_index, item_index, clicks_df]),
    }

@app.get("/reco")
def get_recommendations(user_id: int, n: int = 5, model: str = "content"):
    """
    model: 'content' ou 'collaborative'
    """
    model = (model or "content").lower().strip()

    # user inconnu
    if clicks_df is None or user_id not in set(clicks_df["user_id"].unique()):
        return {"user_id": int(user_id), "model": model, "recommendations": []}

    if model == "content":
        reco_ids = content_based_recommend_articles(user_id, n_reco=n)
    elif model in ["collaborative", "collab", "cf"]:
        reco_ids = cf_recommend_articles(user_id, n_reco=n)
    else:
        return {"user_id": int(user_id), "model": model, "recommendations": []}

    return {
        "user_id": int(user_id),
        "model": model,
        "recommendations": [{"article_id": int(x)} for x in reco_ids],
    }

from fastapi import FastAPI
from typing import Dict, List
import numpy as np
import pandas as pd
import os

app = FastAPI(title="API de recommandation - Projet 9")

# ====== Chargement des artefacts au démarrage ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

U_PATH = os.path.join(REPO_DIR, "models", "collaborative", "U.npy")
V_PATH = os.path.join(REPO_DIR, "models", "collaborative", "V.npy")
USER_INDEX_PATH = os.path.join(REPO_DIR, "models", "collaborative", "user_index.npy")
ITEM_INDEX_PATH = os.path.join(REPO_DIR, "models", "collaborative", "item_index.npy")
ARTICLES_PATH = os.path.join(REPO_DIR, "data_prepared", "articles_metadata_clean.csv")

U = np.load(U_PATH)
V = np.load(V_PATH)
user_index = np.load(USER_INDEX_PATH)     # array de vrais user_id
item_index = np.load(ITEM_INDEX_PATH)     # array de vrais article_id
articles_df = pd.read_csv(ARTICLES_PATH)

# Pour accélérer les lookups
user_set = set(user_index.tolist())
item_set = set(item_index.tolist())

@app.get("/")
def root():
    return {"message": "API de recommandation OK", "model": "collaborative"}

def cf_recommend_articles(user_id: int, n_reco: int = 5) -> List[int]:
    # cold-start
    if user_id not in user_set:
        return []

    u_pos = int(np.where(user_index == user_id)[0][0])
    user_vec = U[u_pos]  # (k,)

    scores = V @ user_vec  # (n_items,)

    # Top-N (sans filtrer les déjà lus ici, car on n'a pas clicks_train_df côté API)
    # => simple et rapide, acceptable pour MVP
    best_idx = np.argpartition(scores, -n_reco)[-n_reco:]
    best_idx = best_idx[np.argsort(scores[best_idx])[::-1]]

    return item_index[best_idx].astype(int).tolist()

@app.get("/reco")
def get_recommendations(user_id: int, n: int = 5) -> Dict:
    reco_ids = cf_recommend_articles(user_id=user_id, n_reco=n)

    if len(reco_ids) == 0:
        return {"user_id": user_id, "recommendations": [], "error": "Utilisateur inconnu ou pas de reco"}

    reco_details = articles_df[articles_df["article_id"].isin(reco_ids)].copy()

    # Pour garder l'ordre des reco
    order = {aid: i for i, aid in enumerate(reco_ids)}
    reco_details["rank"] = reco_details["article_id"].map(order)
    reco_details = reco_details.sort_values("rank").drop(columns=["rank"])

    recos = []
    for _, row in reco_details.iterrows():
        recos.append({
            "article_id": int(row["article_id"]),
            "category_id": int(row["category_id"]),
            "created_at_ts": int(row["created_at_ts"]),
            "publisher_id": int(row["publisher_id"]),
            "words_count": int(row["words_count"])
        })

    return {"user_id": user_id, "recommendations": recos}

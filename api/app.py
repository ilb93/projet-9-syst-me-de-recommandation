import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data_prepared"
MODELS_DIR = ROOT_DIR / "models"
COLLAB_DIR = MODELS_DIR / "collaborative"

ARTICLES_EMB_PCA = DATA_DIR / "articles_embeddings_pca.pkl"
CLICKS = DATA_DIR / "clicks_clean.xls"

CF_U = COLLAB_DIR / "cf_U.npy"
CF_V = COLLAB_DIR / "cf_V.npy"
CF_USER_INDEX = COLLAB_DIR / "cf_user_index.npy"
CF_ITEM_INDEX = COLLAB_DIR / "cf_item_index.npy"


# =========================
# App
# =========================
app = FastAPI(title="API Recommandation - Projet 9")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Globals (lazy-loaded)
# =========================
_U = _V = None
_user_index = _item_index = None
_embeddings = None
_articles_ids = None
_clicks_df = None


# =========================
# Utils
# =========================
def _to_int_array(arr):
    return np.array(arr, dtype=int)


def load_collab():
    global _U, _V, _user_index, _item_index

    if _U is not None:
        return

    _U = np.load(CF_U)
    _V = np.load(CF_V)
    _user_index = _to_int_array(np.load(CF_USER_INDEX, allow_pickle=True))
    _item_index = _to_int_array(np.load(CF_ITEM_INDEX, allow_pickle=True))


def load_content():
    global _embeddings, _articles_ids, _clicks_df

    if _embeddings is not None:
        return

    if CLICKS.exists():
        _clicks_df = pd.read_excel(CLICKS)
    else:
        _clicks_df = None

    if ARTICLES_EMB_PCA.exists():
        obj = pd.read_pickle(ARTICLES_EMB_PCA)
        if isinstance(obj, pd.DataFrame):
            _embeddings = obj.values
            _articles_ids = _to_int_array(obj.index)
        else:
            _embeddings = np.array(obj)
            _articles_ids = np.arange(len(_embeddings))
    else:
        _embeddings = None
        _articles_ids = None


# =========================
# Recommendation logic
# =========================
def recommend_collaborative(user_id: int, n: int) -> List[int]:
    load_collab()

    matches = np.where(_user_index == user_id)[0]
    if len(matches) == 0:
        return list(_item_index[:n])

    u_idx = matches[0]
    scores = _U[u_idx] @ _V.T
    top_idx = np.argsort(scores)[::-1][:n]
    return [int(_item_index[i]) for i in top_idx]


def recommend_content(user_id: int, n: int) -> List[int]:
    load_content()

    if _embeddings is None or _articles_ids is None:
        return list(range(n))

    if _clicks_df is None:
        return list(_articles_ids[:n])

    user_col = next((c for c in _clicks_df.columns if c.lower() == "user_id"), None)
    article_col = next((c for c in _clicks_df.columns if "article" in c.lower()), None)

    if not user_col or not article_col:
        return list(_articles_ids[:n])

    dfu = _clicks_df[_clicks_df[user_col].astype(int) == user_id]
    if dfu.empty:
        return list(_articles_ids[:n])

    last_article = int(dfu.iloc[-1][article_col])

    try:
        idx = np.where(_articles_ids == last_article)[0][0]
    except Exception:
        return list(_articles_ids[:n])

    vec = _embeddings[idx]
    sims = (_embeddings @ vec) / (
        np.linalg.norm(_embeddings, axis=1) * np.linalg.norm(vec) + 1e-12
    )
    sims[idx] = -1

    top_idx = np.argsort(sims)[::-1][:n]
    return [int(_articles_ids[i]) for i in top_idx]


# =========================
# Routes
# =========================
@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/reco")
def reco(
    user_id: int = Query(..., ge=0),
    n: int = Query(5, ge=1, le=50),
    model: str = Query("collaborative", pattern="^(collaborative|content)$"),
):
    if model == "collaborative":
        recs = recommend_collaborative(user_id, n)
    else:
        recs = recommend_content(user_id, n)

    return {
        "user_id": user_id,
        "model": model,
        "count": len(recs),
        "recommendations": [{"article_id": int(x)} for x in recs],
    }


@app.get("/debug/users")
def debug_users():
    load_collab()
    return {
        "nb_users": int(len(_user_index)),
        "nb_items": int(len(_item_index)),
        "sample_users": _user_index[:10].tolist(),
        "sample_items": _item_index[:10].tolist(),
    }

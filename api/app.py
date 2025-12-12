import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATA_DIR = (BASE_DIR / "data_prepared").resolve()
MODELS_DIR = (ROOT_DIR / "models").resolve()
COLLAB_DIR = (MODELS_DIR / "collaborative").resolve()

# Fichiers attendus (selon ton repo)
ARTICLES_EMB_PCA = DATA_DIR / "articles_embeddings_pca.pkl"
ARTICLES_META = DATA_DIR / "articles_metadata_clean.xls"
CLICKS = DATA_DIR / "clicks_clean.xls"

CF_U = COLLAB_DIR / "cf_U.npy"
CF_V = COLLAB_DIR / "cf_V.npy"
CF_USER_INDEX = COLLAB_DIR / "cf_user_index.npy"
CF_ITEM_INDEX = COLLAB_DIR / "cf_item_index.npy"


# =========================
# App
# =========================
app = FastAPI(title="API Recommandation - Projet 9")

# CORS (Streamlit sur un autre domaine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok pour PoC
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Lazy-loaded globals
# =========================
_U = None
_V = None
_user_index = None
_item_index = None

_embeddings = None
_articles_ids = None  # mapping index -> article_id
_clicks_df = None


# =========================
# Utils
# =========================
def _to_int_array(arr: np.ndarray) -> np.ndarray:
    try:
        return arr.astype(int)
    except Exception:
        return np.array([int(x) for x in arr], dtype=int)


def load_collab():
    global _U, _V, _user_index, _item_index
    if _U is not None:
        return

    if not (CF_U.exists() and CF_V.exists() and CF_USER_INDEX.exists() and CF_ITEM_INDEX.exists()):
        raise FileNotFoundError(
            f"Missing collaborative files in {COLLAB_DIR}. "
            f"Need: cf_U.npy, cf_V.npy, cf_user_index.npy, cf_item_index.npy"
        )

    _U = np.load(CF_U)
    _V = np.load(CF_V)
    _user_index = np.load(CF_USER_INDEX, allow_pickle=True)
    _item_index = np.load(CF_ITEM_INDEX, allow_pickle=True)

    # ✅ IMPORTANT: cast int pour éviter mismatch str/int
    _user_index = _to_int_array(_user_index)
    _item_index = _to_int_array(_item_index)


def load_content_based():
    """
    Content-based minimal:
    - embeddings PCA (pickle)
    - clicks pour savoir ce que l'utilisateur a vu
    - sinon fallback simple
    """
    global _embeddings, _articles_ids, _clicks_df

    if _embeddings is not None:
        return

    # clicks
    if CLICKS.exists():
        _clicks_df = pd.read_excel(CLICKS)
        # colonnes attendues souvent: user_id / click_article_id / article_id etc.
    else:
        _clicks_df = None

    # embeddings
    if ARTICLES_EMB_PCA.exists():
        obj = pd.read_pickle(ARTICLES_EMB_PCA)
        # selon comment tu as sauvé : ça peut être np.ndarray ou DataFrame
        if isinstance(obj, pd.DataFrame):
            # si dataframe: index ou colonne contient l'id article
            _embeddings = obj.values
            # essaye d'extraire ids
            if obj.index is not None:
                try:
                    _articles_ids = _to_int_array(obj.index.to_numpy())
                except Exception:
                    _articles_ids = np.arange(len(obj), dtype=int)
            else:
                _articles_ids = np.arange(len(obj), dtype=int)
        elif isinstance(obj, np.ndarray):
            _embeddings = obj
            _articles_ids = np.arange(len(obj), dtype=int)
        else:
            # fallback: tente conversion
            _embeddings = np.array(obj)
            _articles_ids = np.arange(len(_embeddings), dtype=int)
    else:
        _embeddings = None
        _articles_ids = None


# =========================
# Reco functions
# =========================
def recommend_collaborative(user_id: int, n: int) -> List[int]:
    load_collab()
    user_id = int(user_id)

    matches = np.where(_user_index == user_id)[0]
    if len(matches) == 0:
        # ✅ cold-start fallback: renvoie quand même des items valides
        return [int(x) for x in _item_index[:n]]

    u_idx = int(matches[0])
    scores = _U[u_idx] @ _V.T
    top_idx = np.argsort(scores)[::-1][:n]
    return [int(_item_index[i]) for i in top_idx]


def recommend_content(user_id: int, n: int) -> List[int]:
    """
    Content-based simplifié :
    - si on retrouve des clics user -> on prend le dernier article cliqué,
      on renvoie les plus similaires cosinus.
    - sinon fallback items (si collab dispo) ou ids articles embeddings.
    """
    load_content_based()

    # si pas d'embeddings, fallback
    if _embeddings is None or _articles_ids is None:
        # fallback collab si dispo
        try:
            load_collab()
            return [int(x) for x in _item_index[:n]]
        except Exception:
            return list(range(n))

    # Si pas de clicks, fallback "top ids"
    if _clicks_df is None:
        return [int(x) for x in _articles_ids[:n]]

    # Trouver colonne user_id
    cols = set(_clicks_df.columns.str.lower())
    # heuristiques
    user_col = None
    for c in _clicks_df.columns:
        if c.lower() in ["user_id", "userid", "user"]:
            user_col = c
            break

    article_col = None
    for c in _clicks_df.columns:
        if c.lower() in ["article_id", "click_article_id", "id_article", "article"]:
            article_col = c
            break

    if user_col is None or article_col is None:
        return [int(x) for x in _articles_ids[:n]]

    dfu = _clicks_df[_clicks_df[user_col].astype(int) == int(user_id)]
    if dfu.empty:
        return [int(x) for x in _articles_ids[:n]]

    last_article = int(dfu.iloc[-1][article_col])

    # map article_id -> index dans embeddings
    # _articles_ids est (index -> article_id)
    try:
        idx = int(np.where(_articles_ids == last_article)[0][0])
    except Exception:
        return [int(x) for x in _articles_ids[:n]]

    # cosinus similarity
    vec = _embeddings[idx]
    denom = (np.linalg.norm(_embeddings, axis=1) * (np.linalg.norm(vec) + 1e-12)) + 1e-12
    sims = (_embeddings @ vec) / denom

    # exclure l'article lui-même
    sims[idx] = -1
    top_idx = np.argsort(sims)[::-1][:n]
    return [int(_articles_ids[i]) for i in top_idx]


# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "API de recommandation OK"}


@app.get("/reco")
def reco(
    user_id: int = Query(..., ge=0),
    n: int = Query(5, ge=1, le=50),
    model: str = Query("collaborative", pattern="^(collaborative|content)$"),
):
    if model == "collaborative":
        recs = recommend_collaborative(user_id=user_id, n=n)
    else:
        recs = recommend_content(user_id=user_id, n=n)

    return {
        "user_id": int(user_id),
        "n": int(n),
        "model": model,
        "recommendations": recs,
        "count": int(len(recs)),
    }


@app.get("/debug/users")
def debug_users():
    load_collab()
    return {
        "nb_users": int(len(_user_index)),
        "nb_items": int(len(_item_index)),
        "user_id_min": int(_user_index.min()),
        "user_id_max": int(_user_index.max()),
        "sample_user_ids": [int(x) for x in _user_index[:20]],
        "sample_item_ids": [int(x) for x in _item_index[:20]],
    }

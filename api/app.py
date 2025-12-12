# api/app.py
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# =========================
# Paths (repo)
# =========================
BASE_DIR = Path(__file__).resolve().parent          # .../api
ROOT_DIR = BASE_DIR.parent                          # repo root

DATA_DIR = (BASE_DIR / "data_prepared").resolve()
MODELS_DIR = (ROOT_DIR / "models").resolve()
COLLAB_DIR = (MODELS_DIR / "collaborative").resolve()

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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Lazy globals
# =========================
_U: Optional[np.ndarray] = None
_V: Optional[np.ndarray] = None
_user_index: Optional[np.ndarray] = None
_item_index: Optional[np.ndarray] = None

_embeddings: Optional[np.ndarray] = None
_articles_ids: Optional[np.ndarray] = None
_clicks_df: Optional[pd.DataFrame] = None


# =========================
# Utils
# =========================
def _to_int_array(arr: np.ndarray) -> np.ndarray:
    # arr peut être dtype object (strings) -> on force proprement
    try:
        return arr.astype(int)
    except Exception:
        return np.array([int(x) for x in arr], dtype=int)


def _safe_load_npy(path: Path) -> np.ndarray:
    # allow_pickle=True au cas où les index ont été sauvés en object
    return np.load(path, allow_pickle=True)


def load_collab() -> None:
    global _U, _V, _user_index, _item_index

    if _U is not None:
        return

    missing = [p.name for p in [CF_U, CF_V, CF_USER_INDEX, CF_ITEM_INDEX] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing collaborative files in {COLLAB_DIR}: {missing}")

    _U = _safe_load_npy(CF_U)
    _V = _safe_load_npy(CF_V)
    _user_index = _safe_load_npy(CF_USER_INDEX)
    _item_index = _safe_load_npy(CF_ITEM_INDEX)

    _user_index = _to_int_array(np.array(_user_index))
    _item_index = _to_int_array(np.array(_item_index))

    # Sécurité: float32
    _U = np.asarray(_U, dtype=np.float32)
    _V = np.asarray(_V, dtype=np.float32)

    # Check shapes
    if _U.ndim != 2 or _V.ndim != 2:
        raise ValueError(f"U/V must be 2D. Got U:{_U.shape} V:{_V.shape}")
    if _U.shape[1] != _V.shape[1]:
        raise ValueError(f"Latent dim mismatch: U:{_U.shape} V:{_V.shape}")


def load_content() -> None:
    global _embeddings, _articles_ids, _clicks_df

    if _embeddings is not None:
        return

    _clicks_df = None
    if CLICKS.exists():
        _clicks_df = pd.read_excel(CLICKS)

    _embeddings = None
    _articles_ids = None
    if ARTICLES_EMB_PCA.exists():
        obj = pd.read_pickle(ARTICLES_EMB_PCA)

        if isinstance(obj, pd.DataFrame):
            _embeddings = obj.values.astype(np.float32, copy=False)
            try:
                _articles_ids = _to_int_array(obj.index.to_numpy())
            except Exception:
                _articles_ids = np.arange(len(obj), dtype=int)

        elif isinstance(obj, np.ndarray):
            _embeddings = obj.astype(np.float32, copy=False)
            _articles_ids = np.arange(len(_embeddings), dtype=int)

        else:
            arr = np.array(obj)
            _embeddings = arr.astype(np.float32, copy=False)
            _articles_ids = np.arange(len(_embeddings), dtype=int)


# =========================
# Reco functions
# =========================
def recommend_collaborative(user_id: int, n: int) -> List[int]:
    load_collab()

    user_id = int(user_id)
    matches = np.where(_user_index == user_id)[0]

    # cold-start -> renvoie des items valides
    if matches.size == 0:
        return [int(x) for x in _item_index[:n]]

    u_idx = int(matches[0])

    # scores = U[u] dot V.T
    scores = _U[u_idx] @ _V.T
    top_idx = np.argsort(scores)[::-1][:n]
    return [int(_item_index[i]) for i in top_idx]


def recommend_content(user_id: int, n: int) -> List[int]:
    load_content()

    if _embeddings is None or _articles_ids is None:
        # fallback collab si dispo
        try:
            load_collab()
            return [int(x) for x in _item_index[:n]]
        except Exception:
            return list(range(n))

    if _clicks_df is None or _clicks_df.empty:
        return [int(x) for x in _articles_ids[:n]]

    # colonnes (heuristiques)
    user_col = next((c for c in _clicks_df.columns if c.lower() in ["user_id", "userid", "user"]), None)
    art_col = next((c for c in _clicks_df.columns if c.lower() in ["article_id", "click_article_id", "id_article", "article"]), None)

    if user_col is None or art_col is None:
        return [int(x) for x in _articles_ids[:n]]

    dfu = _clicks_df[_clicks_df[user_col].astype(int) == int(user_id)]
    if dfu.empty:
        return [int(x) for x in _articles_ids[:n]]

    last_article = int(dfu.iloc[-1][art_col])

    pos = np.where(_articles_ids == last_article)[0]
    if pos.size == 0:
        return [int(x) for x in _articles_ids[:n]]

    idx = int(pos[0])

    vec = _embeddings[idx]
    denom = (np.linalg.norm(_embeddings, axis=1) * (np.linalg.norm(vec) + 1e-12)) + 1e-12
    sims = (_embeddings @ vec) / denom
    sims[idx] = -1

    top_idx = np.argsort(sims)[::-1][:n]
    return [int(_articles_ids[i]) for i in top_idx]


# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "API de recommandation OK"}


@app.get("/health")
def health() -> Dict[str, Any]:
    # Ne crash pas : juste état des fichiers + tailles
    files = {
        "CF_U": str(CF_U),
        "CF_V": str(CF_V),
        "CF_USER_INDEX": str(CF_USER_INDEX),
        "CF_ITEM_INDEX": str(CF_ITEM_INDEX),
        "ARTICLES_EMB_PCA": str(ARTICLES_EMB_PCA),
        "CLICKS": str(CLICKS),
    }
    exists = {k: Path(v).exists() for k, v in files.items()}
    return {
        "status": "ok",
        "cwd": os.getcwd(),
        "base_dir": str(BASE_DIR),
        "root_dir": str(ROOT_DIR),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "collab_dir": str(COLLAB_DIR),
        "files_exist": exists,
    }


@app.get("/debug/files")
def debug_files():
    # Très utile si Azure ne pack pas ce que tu crois
    def ls(p: Path) -> List[str]:
        if not p.exists():
            return []
        return sorted([x.name for x in p.iterdir()])

    return {
        "repo_root": str(ROOT_DIR),
        "api_dir": str(BASE_DIR),
        "data_prepared": ls(DATA_DIR),
        "models": ls(MODELS_DIR),
        "collaborative": ls(COLLAB_DIR),
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
        "U_shape": list(_U.shape),
        "V_shape": list(_V.shape),
    }


@app.get("/reco")
def reco(
    user_id: int = Query(..., ge=0),
    n: int = Query(5, ge=1, le=50),
    model: str = Query("collaborative", regex="^(collaborative|content)$"),
):
    try:
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

    except Exception as e:
        # Important : on remonte le vrai message au lieu de 500 vide
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

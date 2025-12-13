import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# =========================
# Helpers paths
# =========================
def _candidate_dirs() -> List[Path]:
    """
    Azure App Service (Linux) déploie souvent dans /home/site/wwwroot.
    Selon ton workflow, tu peux n’avoir que /home/site/wwwroot/api (sans models à la racine).
    On cherche donc dans plusieurs emplacements.
    """
    here = Path(__file__).resolve()
    base_dir = here.parent          # .../api
    root_dir = base_dir.parent      # .../(repo root) si présent localement

    candidates = [
        root_dir,                     # repo root (local)
        base_dir,                     # api/
        Path.cwd(),                   # current working dir
        Path("/home/site/wwwroot"),   # Azure root
        Path("/home/site/wwwroot/api"),
        Path("/tmp"),                 # parfois extraction temporaire
    ]

    out = []
    seen = set()
    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            continue
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def _find_first_existing(rel_path: str) -> Optional[Path]:
    """
    rel_path exemple: "models/collaborative/cf_U.npy" ou "api/data_prepared/clicks_clean.xls"
    """
    for root in _candidate_dirs():
        p = (root / rel_path).resolve()
        if p.exists():
            return p
    return None


def _read_table_smart(path: Path) -> pd.DataFrame:
    """
    Lit une table en gérant:
    - CSV
    - Excel (.xls/.xlsx)
    - cas fréquent: CSV renommé .xls => sniff du header (ex: 'user_id,')
    """
    suffix = path.suffix.lower()

    # 1) Extensions explicites
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    # 2) Excel ou pseudo-excel
    if suffix in [".xls", ".xlsx"]:
        # sniff rapide: si ça ressemble à du CSV (ex: b"user_id,")
        try:
            with open(path, "rb") as f:
                head = f.read(128).lower()
            if b"," in head and (b"user_id" in head or b"article" in head):
                return pd.read_csv(path)
        except Exception:
            pass

        # vrai excel
        if suffix == ".xls":
            # nécessite xlrd (xls)
            return pd.read_excel(path, engine="xlrd")
        return pd.read_excel(path)

    # 3) Sniff générique si extension inconnue
    try:
        with open(path, "rb") as f:
            head = f.read(128).lower()
        if b"," in head:
            return pd.read_csv(path)
    except Exception:
        pass

    raise ValueError(f"Format non supporté: {path}")


def _to_int_array(arr: np.ndarray) -> np.ndarray:
    try:
        return arr.astype(int)
    except Exception:
        return np.array([int(x) for x in arr], dtype=int)


# =========================
# App
# =========================
app = FastAPI(title="API Recommandation - Projet 9")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok PoC
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
_articles_ids = None
_clicks_df = None


# =========================
# Loaders
# =========================
def load_collab():
    global _U, _V, _user_index, _item_index
    if _U is not None:
        return

    # IMPORTANT:
    # Sur Azure, tu n'as souvent QUE /home/site/wwwroot/api
    # donc on supporte:
    # - models/... (local ou si tu déploies toute la repo)
    # - api/models/... (solution simple: dupliquer les artefacts dans api/)
    CF_U = _find_first_existing("models/collaborative/cf_U.npy") or _find_first_existing("api/models/collaborative/cf_U.npy")
    CF_V = _find_first_existing("models/collaborative/cf_V.npy") or _find_first_existing("api/models/collaborative/cf_V.npy")
    CF_USER_INDEX = _find_first_existing("models/collaborative/cf_user_index.npy") or _find_first_existing("api/models/collaborative/cf_user_index.npy")
    CF_ITEM_INDEX = _find_first_existing("models/collaborative/cf_item_index.npy") or _find_first_existing("api/models/collaborative/cf_item_index.npy")

    missing = []
    if CF_U is None: missing.append("cf_U.npy")
    if CF_V is None: missing.append("cf_V.npy")
    if CF_USER_INDEX is None: missing.append("cf_user_index.npy")
    if CF_ITEM_INDEX is None: missing.append("cf_item_index.npy")

    if missing:
        raise FileNotFoundError(
            "Missing collaborative files. "
            "Ton workflow Azure déploie probablement seulement le dossier 'api/' "
            "et donc 'models/' n'est pas présent sur l'App Service. "
            f"Missing: {missing}. "
            "Solution simple: déplacer/dupliquer models/collaborative -> api/models/collaborative et commit."
        )

    _U = np.load(CF_U)
    _V = np.load(CF_V)
    _user_index = np.load(CF_USER_INDEX, allow_pickle=True)
    _item_index = np.load(CF_ITEM_INDEX, allow_pickle=True)

    _user_index = _to_int_array(_user_index)
    _item_index = _to_int_array(_item_index)


def load_content_based():
    global _embeddings, _articles_ids, _clicks_df
    if _embeddings is not None:
        return

    # embeddings (pickle)
    EMB = _find_first_existing("api/data_prepared/articles_embeddings_pca.pkl") or _find_first_existing("data_prepared/articles_embeddings_pca.pkl")

    # clicks:
    # Ton erreur "Expected BOF record; found b'user_id,'"
    # => fichier pas vraiment Excel, probablement CSV renommé .xls
    CLICKS = (
        _find_first_existing("api/data_prepared/clicks_clean.csv")
        or _find_first_existing("data_prepared/clicks_clean.csv")
        or _find_first_existing("api/data_prepared/clicks_clean.xls")
        or _find_first_existing("data_prepared/clicks_clean.xls")
        or _find_first_existing("api/data_prepared/clicks_clean.xlsx")
        or _find_first_existing("data_prepared/clicks_clean.xlsx")
    )

    if CLICKS is not None:
        _clicks_df = _read_table_smart(CLICKS)
    else:
        _clicks_df = None

    if EMB is not None:
        obj = pd.read_pickle(EMB)
        if isinstance(obj, pd.DataFrame):
            _embeddings = obj.values
            try:
                _articles_ids = _to_int_array(obj.index.to_numpy())
            except Exception:
                _articles_ids = np.arange(len(obj), dtype=int)
        elif isinstance(obj, np.ndarray):
            _embeddings = obj
            _articles_ids = np.arange(len(obj), dtype=int)
        else:
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
        return [int(x) for x in _item_index[:n]]

    u_idx = int(matches[0])
    scores = _U[u_idx] @ _V.T
    top_idx = np.argsort(scores)[::-1][:n]
    return [int(_item_index[i]) for i in top_idx]


def recommend_content(user_id: int, n: int) -> List[int]:
    load_content_based()

    if _embeddings is None or _articles_ids is None:
        # fallback -> collab si dispo
        try:
            load_collab()
            return [int(x) for x in _item_index[:n]]
        except Exception:
            return list(range(n))

    if _clicks_df is None or _clicks_df.empty:
        return [int(x) for x in _articles_ids[:n]]

    # colonnes user/article (robuste)
    cols_lower = {c.lower(): c for c in _clicks_df.columns}
    user_col = cols_lower.get("user_id") or cols_lower.get("userid") or cols_lower.get("user")
    article_col = (
        cols_lower.get("article_id")
        or cols_lower.get("click_article_id")
        or cols_lower.get("id_article")
        or cols_lower.get("article")
    )

    if user_col is None or article_col is None:
        return [int(x) for x in _articles_ids[:n]]

    # filtrage user
    dfu = _clicks_df.copy()
    dfu[user_col] = pd.to_numeric(dfu[user_col], errors="coerce")
    dfu = dfu[dfu[user_col] == int(user_id)]
    if dfu.empty:
        return [int(x) for x in _articles_ids[:n]]

    # dernier article cliqué
    dfu[article_col] = pd.to_numeric(dfu[article_col], errors="coerce")
    dfu = dfu.dropna(subset=[article_col])
    if dfu.empty:
        return [int(x) for x in _articles_ids[:n]]

    last_article = int(dfu.iloc[-1][article_col])

    # retrouver son index dans embeddings
    try:
        idx = int(np.where(_articles_ids == last_article)[0][0])
    except Exception:
        return [int(x) for x in _articles_ids[:n]]

    vec = _embeddings[idx]

    # cosine similarity
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
    """
    Permet de voir vite si Azure a bien les fichiers.
    """
    def exists(rel: str) -> bool:
        return _find_first_existing(rel) is not None

    return {
        "cwd": str(Path.cwd()),
        "candidates": [str(p) for p in _candidate_dirs()],

        "has_collab_cf_U": exists("models/collaborative/cf_U.npy") or exists("api/models/collaborative/cf_U.npy"),
        "has_collab_cf_V": exists("models/collaborative/cf_V.npy") or exists("api/models/collaborative/cf_V.npy"),
        "has_collab_user_index": exists("models/collaborative/cf_user_index.npy") or exists("api/models/collaborative/cf_user_index.npy"),
        "has_collab_item_index": exists("models/collaborative/cf_item_index.npy") or exists("api/models/collaborative/cf_item_index.npy"),

        # clicks: csv OU excel
        "has_clicks": (
            exists("api/data_prepared/clicks_clean.csv")
            or exists("data_prepared/clicks_clean.csv")
            or exists("api/data_prepared/clicks_clean.xls")
            or exists("data_prepared/clicks_clean.xls")
            or exists("api/data_prepared/clicks_clean.xlsx")
            or exists("data_prepared/clicks_clean.xlsx")
        ),

        "has_embeddings": exists("api/data_prepared/articles_embeddings_pca.pkl") or exists("data_prepared/articles_embeddings_pca.pkl"),

        # utile pour confirmer le packaging Azure
        "azure_wwwroot_models_exists": Path("/home/site/wwwroot/models").exists(),
        "azure_api_models_exists": Path("/home/site/wwwroot/api/models").exists(),
    }


@app.get("/reco")
def reco(
    user_id: int = Query(..., ge=0),
    n: int = Query(5, ge=1, le=50),
    model: str = Query("collaborative", pattern="^(collaborative|content)$"),
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
            "recommendations": [{"article_id": int(x)} for x in recs],
            "count": int(len(recs)),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

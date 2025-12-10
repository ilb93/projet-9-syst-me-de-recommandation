from fastapi import FastAPI
from typing import List, Dict

app = FastAPI(title="API de recommandation - Projet 9")

@app.get("/")
def root():
    return {"message": "API de recommandation OK"}

@app.get("/reco")
def get_recommendations(user_id: int, n: int = 5) -> Dict:
    """
    Endpoint de test : renvoie n faux IDs d'articles pour l'utilisateur donné.
    (On pourra plus tard brancher le vrai modèle si tu veux.)
    """
    reco_ids = [100000 + i for i in range(n)]

    recos = [{"article_id": rid} for rid in reco_ids]

    return {
        "user_id": user_id,
        "recommendations": recos
    }

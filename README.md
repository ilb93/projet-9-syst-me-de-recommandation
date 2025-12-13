# Système de recommandation d’articles

**Projet 9 – Content-Based & Collaborative Filtering**

---

## Objectif

Ce projet vise à concevoir et déployer un **système de recommandation d’articles** capable de proposer des contenus pertinents à un utilisateur donné, en s’appuyant sur **deux approches complémentaires** :

- Content-Based Filtering  
- Collaborative Filtering  

L’ensemble du système est **déployé sur Azure**, avec une architecture simple et découplée entre une API de recommandation et une interface utilisateur.

---

## Approches de recommandation

### Content-Based Filtering

Cette approche recommande des articles similaires à ceux consultés par l’utilisateur.

**Principe :**
- Représentation des articles sous forme d’**embeddings**
- Réduction de dimension via **PCA**
- Identification du dernier article consulté à partir des données de clics
- Calcul des recommandations par **similarité cosinus**

**Avantages :**
- Fonctionne même avec peu d’utilisateurs
- Indépendant du comportement des autres utilisateurs
- Recommandations cohérentes et explicables

---

### Collaborative Filtering

Cette approche exploite les interactions collectives entre utilisateurs et articles.

**Principe :**
- Utilisation d’un modèle de **factorisation matricielle**
- Décomposition en :
  - matrice utilisateurs × facteurs latents (U)
  - matrice articles × facteurs latents (V)
- Calcul du score via le produit scalaire :


**Avantages :**
- Capte des préférences implicites
- Recommandations plus personnalisées
- Exploite l’intelligence collective

---

## Architecture du projet

 projet-9-syst-me-de-recommandation/
├── api/

│ ├── app.py

│ ├── requirements.txt

│ ├── startup.txt

│ ├── data_prepared/

│ │ ├── clicks_clean.csv

│ │ └── articles_embeddings_pca.pkl

│ └── models/

│ └── collaborative/

│ ├── cf_U.npy

│ ├── cf_V.npy

│ ├── cf_user_index.npy

│ └── cf_item_index.npy

├── streamlit_app/

│ └── app.py

├── notebooks/

└── README.md


---

## Déploiement

Le projet est déployé sur **Azure App Service (Linux)**.

- API REST : FastAPI  
- Interface utilisateur : Streamlit  
- Chargement dynamique des modèles et des données depuis :


---

## Déploiement

Le projet est déployé sur **Azure App Service (Linux)**.

- API REST : FastAPI  
- Interface utilisateur : Streamlit  
- Chargement dynamique des modèles et des données depuis :


---

## API – Endpoints principaux

### Health check


Permet de vérifier la présence des fichiers nécessaires (modèles, données, embeddings).

---

### Recommandation


**Paramètres :**
- user_id : identifiant utilisateur
- n : nombre de recommandations
- model : collaborative ou content

---

## Lancement local (optionnel)

### API

cd api
pip install -r requirements.txt
uvicorn app:app --reload


### Interface Streamlit


---

## Conclusion

Ce projet met en œuvre un **système de recommandation complet**, depuis la phase de modélisation jusqu’au **déploiement cloud**, en intégrant des contraintes réelles de production : formats de données, chemins système, robustesse et scalabilité.

---

## Auteur

Projet réalisé par **Mourad**  
Projet académique en Data / Machine Learning

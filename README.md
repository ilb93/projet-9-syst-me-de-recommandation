Système de recommandation d’articles

Projet 9 – Content-Based & Collaborative Filtering

Objectif du projet

Ce projet a pour objectif de concevoir, implémenter et déployer un système de recommandation d’articles
capable de proposer des contenus pertinents à un utilisateur donné, en s’appuyant sur deux approches complémentaires :

Content-Based Filtering (basé sur le contenu des articles)

Collaborative Filtering (basé sur les comportements d’utilisateurs similaires)

L’ensemble du système est déployé sur Azure, avec une architecture découplée entre une API de recommandation
et une interface utilisateur.

Approches de recommandation
Content-Based Filtering

Cette approche repose sur la similarité entre les articles consultés par un utilisateur.

Principe :

Les articles sont représentés sous forme de vecteurs d’embeddings

Une réduction de dimension est appliquée via PCA

Le dernier article consulté par l’utilisateur est identifié à partir des données de clics

Les recommandations sont calculées à l’aide de la similarité cosinus entre les articles

Avantages :

Fonctionne même avec peu d’utilisateurs

Indépendant du comportement des autres utilisateurs

Recommandations cohérentes et explicables

Collaborative Filtering

Cette approche exploite les interactions collectives entre utilisateurs et articles.

Principe :

Utilisation d’un modèle de factorisation matricielle

Décomposition en :

matrice utilisateurs × facteurs latents (U)

matrice articles × facteurs latents (V)

Le score de recommandation est calculé via le produit scalaire :

score(user, item) = U[user] · V[item]


Avantages :

Capte des préférences implicites

Recommandations plus personnalisées à grande échelle

Exploitation de l’intelligence collective

Architecture du projet
projet-9-syst-me-de-recommandation/
├── api/
│   ├── app.py
│   ├── requirements.txt
│   ├── startup.txt
│   ├── data_prepared/
│   │   ├── clicks_clean.csv
│   │   └── articles_embeddings_pca.pkl
│   └── models/
│       └── collaborative/
│           ├── cf_U.npy
│           ├── cf_V.npy
│           ├── cf_user_index.npy
│           └── cf_item_index.npy
├── streamlit_app/
│   └── app.py
├── notebooks/
└── README.md

Déploiement

Le projet est déployé sur Azure App Service (Linux).

API REST : FastAPI

Interface utilisateur : Streamlit

Chargement dynamique des modèles et des données depuis :

/home/site/wwwroot/api

Points techniques clés

Chargement lazy des modèles (au premier appel)

Détection automatique des formats de fichiers (CSV / Excel)

Gestion robuste des chemins en environnement Azure

Fallbacks sécurisés si certaines données sont absentes

Séparation claire entre logique métier et interface utilisateur

API – Endpoints principaux
Health check
GET /health


Permet de vérifier la présence des fichiers nécessaires (modèles, données, embeddings).

Recommandation
GET /reco?user_id=15&n=5&model=collaborative


Paramètres :

user_id : identifiant utilisateur

n : nombre de recommandations

model : collaborative ou content

Lancement local (optionnel)
API
cd api
pip install -r requirements.txt
uvicorn app:app --reload

Interface Streamlit
cd streamlit_app
streamlit run app.py

Conclusion

Ce projet met en œuvre un système de recommandation complet, depuis la phase de modélisation
jusqu’au déploiement cloud, en intégrant des contraintes réelles de production :
formats de données, chemins système, robustesse et scalabilité.

Il constitue une base solide pour une évolution vers un moteur de recommandation hybride
ou une industrialisation de type MLOps.

Auteur

Projet réalisé par Mourad
Dans le cadre d’un projet académique en Data / Machine Learning.

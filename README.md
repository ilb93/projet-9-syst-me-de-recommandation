# \# 📚 Système de recommandation d’articles  

# \*\*Projet 9 – Recommandation (Content-Based \& Collaborative Filtering)\*\*

# 

# ---

# 

# \## 🎯 Objectif du projet

# 

# Ce projet a pour objectif de concevoir, implémenter et déployer un \*\*système de recommandation d’articles\*\* capable de proposer des contenus pertinents à un utilisateur donné, en s’appuyant sur \*\*deux approches complémentaires\*\* :

# 

# \- \*\*Content-Based Filtering\*\* (basé sur le contenu des articles)

# \- \*\*Collaborative Filtering\*\* (basé sur les comportements d’utilisateurs similaires)

# 

# L’ensemble du système est \*\*déployé sur Azure\*\*, avec :

# \- une \*\*API REST (FastAPI)\*\* pour la logique de recommandation

# \- une \*\*interface Streamlit\*\* pour l’interaction utilisateur

# 

# ---

# 

# \## 🧠 Approches de recommandation

# 

# \### 1️⃣ Content-Based Filtering

# Cette approche recommande des articles similaires à ceux déjà consultés par l’utilisateur.

# 

# \*\*Principe :\*\*

# \- Les articles sont représentés par des \*\*embeddings vectoriels\*\* (réduction de dimension via PCA).

# \- Le dernier article consulté par l’utilisateur est identifié à partir des données de clics.

# \- Les recommandations sont calculées via la \*\*similarité cosinus\*\* entre les vecteurs d’articles.

# 

# \*\*Avantages :\*\*

# \- Fonctionne même avec peu d’utilisateurs

# \- Pas de dépendance directe aux autres profils

# 

# ---

# 

# \### 2️⃣ Collaborative Filtering

# Cette approche repose sur les comportements collectifs des utilisateurs.

# 

# \*\*Principe :\*\*

# \- Utilisation d’un modèle de \*\*factorisation matricielle\*\* :

# &nbsp; - matrice utilisateurs × facteurs latents (`U`)

# &nbsp; - matrice articles × facteurs latents (`V`)

# \- Le score de recommandation est calculé via le produit scalaire :

# &nbsp; 

\# 📚 Système de recommandation d’articles  

\*\*Projet 9 – Recommandation (Content-Based \& Collaborative Filtering)\*\*



---



\## 🎯 Objectif du projet



Ce projet a pour objectif de concevoir, implémenter et déployer un \*\*système de recommandation d’articles\*\* capable de proposer des contenus pertinents à un utilisateur donné, en s’appuyant sur \*\*deux approches complémentaires\*\* :



\- \*\*Content-Based Filtering\*\* (basé sur le contenu des articles)

\- \*\*Collaborative Filtering\*\* (basé sur les comportements d’utilisateurs similaires)



L’ensemble du système est \*\*déployé sur Azure\*\*, avec :

\- une \*\*API REST (FastAPI)\*\* pour la logique de recommandation

\- une \*\*interface Streamlit\*\* pour l’interaction utilisateur



---



\## 🧠 Approches de recommandation



\### 1️⃣ Content-Based Filtering

Cette approche recommande des articles similaires à ceux déjà consultés par l’utilisateur.



\*\*Principe :\*\*

\- Les articles sont représentés par des \*\*embeddings vectoriels\*\* (réduction de dimension via PCA).

\- Le dernier article consulté par l’utilisateur est identifié à partir des données de clics.

\- Les recommandations sont calculées via la \*\*similarité cosinus\*\* entre les vecteurs d’articles.



\*\*Avantages :\*\*

\- Fonctionne même avec peu d’utilisateurs

\- Pas de dépendance directe aux autres profils



---



\### 2️⃣ Collaborative Filtering

Cette approche repose sur les comportements collectifs des utilisateurs.



\*\*Principe :\*\*

\- Utilisation d’un modèle de \*\*factorisation matricielle\*\* :

&nbsp; - matrice utilisateurs × facteurs latents (`U`)

&nbsp; - matrice articles × facteurs latents (`V`)

\- Le score de recommandation est calculé via le produit scalaire :



score(user, item) = U\[user] · V\[item]





\*\*Avantages :\*\*

\- Capte des préférences implicites

\- Recommandations plus “personnalisées” à grande échelle



---



\## 🏗️ Architecture du projet







projet-9-syst-me-de-recommandation/

├── api/

│ ├── app.py # API FastAPI

│ ├── requirements.txt

│ ├── startup.txt # Commande de démarrage Azure

│ ├── data\_prepared/

│ │ ├── clicks\_clean.csv

│ │ └── articles\_embeddings\_pca.pkl

│ └── models/

│ └── collaborative/

│ ├── cf\_U.npy

│ ├── cf\_V.npy

│ ├── cf\_user\_index.npy

│ └── cf\_item\_index.npy

├── streamlit\_app/

│ └── app.py # Interface Streamlit

├── notebooks/

│ └── \*.ipynb # Entraînement \& exploration

└── README.md





---



\## 🚀 Déploiement Cloud (Azure)



\- \*\*API\*\* : Azure App Service (Linux)

\- \*\*Frontend\*\* : Streamlit déployé sur Azure

\- Les modèles et données sont chargés dynamiquement depuis :





/home/site/wwwroot/api





\### Points techniques gérés

\- Chargement \*\*lazy\*\* des modèles (au premier appel)

\- Détection automatique des formats de fichiers (CSV / Excel)

\- Gestion robuste des chemins Azure

\- Fallbacks sécurisés si certaines données sont absentes



---



\## 🔌 API – Endpoints principaux



\### 🔍 Health check

```http

GET /health





Permet de vérifier la présence des fichiers (modèles, données, embeddings).



🎯 Recommandation

GET /reco?user\_id=15\&n=5\&model=collaborative





Paramètres :



user\_id : identifiant utilisateur



n : nombre de recommandations



model : collaborative ou content



Exemple de réponse :



{

&nbsp; "user\_id": 15,

&nbsp; "n": 5,

&nbsp; "model": "content",

&nbsp; "recommendations": \[

&nbsp;   {"article\_id": 96739},

&nbsp;   {"article\_id": 93090},

&nbsp;   {"article\_id": 96212}

&nbsp; ],

&nbsp; "count": 3

}



🖥️ Interface Streamlit



L’interface permet :



de choisir un utilisateur



de sélectionner le type de recommandation



de définir le nombre d’articles



de visualiser les résultats en temps réel



Elle communique exclusivement avec l’API via HTTP.



🧪 Installation locale (optionnel)

1️⃣ Cloner le projet

git clone https://github.com/ilb93/projet-9-syst-me-de-recommandation.git

cd projet-9-syst-me-de-recommandation



2️⃣ Lancer l’API

cd api

pip install -r requirements.txt

uvicorn app:app --reload



3️⃣ Lancer Streamlit

cd streamlit\_app

streamlit run app.py



✅ Résultats et validation



✔️ Les deux modèles fonctionnent en production



✔️ Les recommandations sont cohérentes et différenciées



✔️ L’architecture est scalable et cloud-ready



✔️ Les erreurs de déploiement (paths, formats) ont été identifiées et corrigées



📌 Conclusion



Ce projet démontre la mise en œuvre complète d’un système de recommandation moderne, depuis la phase de modélisation jusqu’au déploiement cloud, en respectant des contraintes réelles de production (formats, chemins, performance, robustesse).



Il constitue une base solide pour :



un moteur de recommandation hybride



une industrialisation MLOps



ou une extension vers des recommandations explicables



👤 Auteur



Projet réalisé par Mourad

Dans le cadre d’un projet académique en Data / Machine Learning.

&nbsp; 




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


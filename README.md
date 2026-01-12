# FBref Data Scraper

Ce projet utilise la bibliothèque `soccerdata` pour collecter les statistiques des joueurs de football des principales ligues européennes sur les cinq dernières saisons.

Il utilise également les données "tests" fournies par SkillCorner dans le cadre de leur hackathon (cf README_skillcorner.md)

## Objectif

Collecter toutes les données statistiques disponibles au niveau des joueurs pour un ensemble sélectionné de ligues européennes sur les cinq dernières saisons.
A terme, l'objectif est d'utiliser ces données pour prédire le "nearest player" comparé au joueur ciblé, et ainsi aider les clubs/agents de joueur dans leurs choix de transferts.

## Ligues ciblées

Le projet se concentre sur les premières divisions (et, lorsqu'elles sont disponibles, les deuxièmes divisions) des pays européens qui exportent fréquemment des joueurs vers les clubs de premier plan européens :

### Europe de l'Ouest
- France (Ligue 1, Ligue 2)
- Angleterre (Premier League, Championship)
- Espagne (La Liga, Segunda División)
- Allemagne (Bundesliga, 2. Bundesliga)
- Italie (Serie A, Serie B)

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Structure du projet

- `data/` - Données collectées
- `scripts/` - Scripts permettant de récupérer et cleaner les données
  - `scraping/` - Scripts de scraping des données
  - `merging/` - Scripts d'aggrégation des données scrapées
- `notebooks/` - Notebooks Jupyter pour l'exploration et l'analyse des données
- `streamlit/` - Premiers streamlits générés à partir des données scrappées
  - `fbref/` - Focus sur les données récupérées sur fbref
  - `aus/` - Focus sur les données récupérées via skillcornerdata sur la ligue australienne

## Utilisation

Pour visualiser les premiers résultats de nos modèles de prédictions, une fois dans le dossier SKILLCORNERDATA :

```python
cd streamlit/fbref
run app.py
```

Une fois sur le streamlit, vous pouvez choisir un des joueurs qui vous intéresse et rechercher, en fonction de plusieurs caractéristiques, des joueurs qui correspondent au votre.


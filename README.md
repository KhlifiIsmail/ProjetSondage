# Application Avancée de Théorie de Sondage

Une application web interactive pour l'échantillonnage statistique, développée pour le projet de Théorie de Sondage de l'École supérieure de la statistique et l'Analyse de l'information.

![Application Screenshot](https://i.imgur.com/placeholder.png)

## Fonctionnalités

### Interface Utilisateur

- Design moderne et professionnel avec une interface intuitive
- Navigation par onglets pour une expérience utilisateur fluide
- Personnalisation de l'interface (thèmes de couleurs, etc.)
- Visualisations interactives avec Plotly
- Tableaux de données stylisés et téléchargeables

### Méthode Aléatoire Simple sans Remise (SAS)

- Tirage d'échantillons aléatoires sans remise
- Statistiques descriptives complètes (moyenne, écart-type, CV, etc.)
- Tableaux comparatifs échantillon-cadre
- Tests de représentativité (Chi² et tests de Student)
- Calcul d'intervalles de confiance avec niveau personnalisable
- Visualisations interactives des distributions
- Analyse bivariée entre variables (avec tests statistiques)

### Méthode de Stratification à Allocation Proportionnelle

- Allocation proportionnelle automatique par strate
- Tableaux et visualisations des allocations
- Statistiques descriptives globales et par strate
- Distribution des variables auxiliaires
- Analyse de l'efficacité de la stratification (effet de plan, etc.)
- Intervalles de confiance stratifiés
- Analyse bivariée par strate

### Analyse Statistique Avancée

- Tests de représentativité automatiques
- Score global de représentativité de l'échantillon
- Diagnostics d'échantillonnage
- Analyse de corrélation et tests d'indépendance
- Analyses ANOVA pour variables mixtes (catégorielles/numériques)
- Visualisation des intervalles de confiance
- Exportation de tous les résultats en format Excel

## Installation et Exécution

### Prérequis

- Python 3.8 ou supérieur
- Les bibliothèques Python listées dans `requirements.txt`

### Installation

1. Clonez ce dépôt ou téléchargez les fichiers source
2. Installez les dépendances requises:

```bash
pip install -r requirements.txt
```

### Structure du Projet

- `app.py`: Application principale Streamlit
- `utils.py`: Fonctions utilitaires pour l'échantillonnage et l'analyse statistique
- `requirements.txt`: Liste des dépendances Python
- `README.md`: Documentation du projet

### Exécution de l'Application

```bash
streamlit run app.py
```

L'application sera accessible dans votre navigateur à l'adresse `http://localhost:8501`

## Guide d'Utilisation

### 1. Téléchargement du Cadre d'Échantillonnage

- Utilisez le panneau latéral pour télécharger le fichier Excel "Cadre Tunisie.xlsx"
- Vous pouvez visualiser un aperçu des données dans l'expander "Informations sur le cadre"

### 2. Sélection de la Méthode d'Échantillonnage

- Choisissez entre "Aléatoire Simple sans Remise (SAS)" et "Stratification à Allocation Proportionnelle"
- Pour chaque méthode, les paramètres spécifiques s'afficheront automatiquement

### 3. Configuration des Paramètres

- **Pour la méthode SAS** :

  - Définissez la taille de l'échantillon
  - Sélectionnez une variable comparative principale et secondaire (optionnelle)
  - Ajustez le niveau de confiance pour les intervalles

- **Pour la méthode stratifiée** :
  - Définissez la taille de l'échantillon
  - Sélectionnez la variable de stratification
  - Choisissez une variable auxiliaire (optionnelle)
  - Ajustez le niveau de confiance pour les analyses statistiques

### 4. Génération et Analyse des Échantillons

- Cliquez sur le bouton "Générer l'échantillon"
- Explorez les différents onglets pour analyser les résultats :
  - Données de l'échantillon
  - Statistiques descriptives
  - Visualisations comparatives
  - Analyses avancées (tests statistiques, intervalles de confiance, etc.)

### 5. Exportation des Résultats

- Utilisez les boutons de téléchargement pour exporter :
  - L'échantillon généré
  - Les tableaux statistiques
  - Les tableaux d'allocation (pour la méthode stratifiée)
  - Les résultats complets combinés

## Fonctionnalités Avancées

### Analyse de Représentativité

L'application calcule automatiquement un score de représentativité basé sur plusieurs tests statistiques :

- Tests du Chi² pour les variables catégorielles
- Tests t de Student pour les variables numériques
- Le score global indique le pourcentage de variables pour lesquelles l'échantillon est représentatif

### Personnalisation Visuelle

- Vous pouvez choisir parmi plusieurs thèmes de couleurs dans le panneau latéral
- Toutes les visualisations sont interactives (zoom, survol pour plus d'informations, etc.)

### Analyse Bivariée

L'application propose des analyses bivariées complètes :

- Pour deux variables catégorielles : tableaux croisés, test du Chi² et V de Cramer
- Pour deux variables numériques : nuage de points avec régression, corrélation de Pearson et Spearman
- Pour une variable catégorielle et une numérique : boxplots, ANOVA et calcul de l'Eta²

### Diagnostics de la Stratification

Pour la méthode stratifiée, des diagnostics spécifiques sont disponibles :

- Calcul de l'effet de plan (DEFF)
- Estimation de la taille d'échantillon effective
- Comparaison des corrélations entre variables par strate

## Contribution et Développement Futur

Cette application a été développée pour le projet de Théorie de Sondage de l'École supérieure de la statistique et l'Analyse de l'information. N'hésitez pas à l'améliorer ou à la personnaliser selon vos besoins.

Suggestions pour les développements futurs :

- Ajout de méthodes d'échantillonnage supplémentaires (systématique, en grappes, etc.)
- Implémentation de techniques de post-stratification
- Intégration avec des sources de données externes
- Exportation des visualisations en format haute résolution

## Auteurs

[Votre nom et le nom de votre coéquipier]

## Remerciements

- École supérieure de la statistique et l'Analyse de l'information
- Enseignants du cours Théorie de Sondage

---

Développé avec ❤️ pour le projet de Théorie de Sondage  
École supérieure de la statistique et l'Analyse de l'information | 2024-2025

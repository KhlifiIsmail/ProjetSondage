# Application de Théorie de Sondage

Cette application web permet de tirer automatiquement des échantillons selon les méthodes d'échantillonnage suivantes :
1. **Aléatoire Simple sans Remise (SAS)**
2. **Stratification à Allocation Proportionnelle**

## Fonctionnalités

### Méthode Aléatoire Simple sans Remise (SAS)
- Sélection d'un échantillon de taille n sans remise
- Génération de statistiques descriptives
- Tableau comparatif échantillon-cadre
- Visualisation graphique des comparaisons

### Méthode de Stratification à Allocation Proportionnelle
- Sélection d'une variable de stratification (Régions, Gouvernorats, Délégations)
- Utilisation de variables auxiliaires (Milieu urbain/rural, taille des blocs)
- Tableau d'allocation proportionnelle
- Génération d'échantillons stratifiés
- Visualisations graphiques des allocations

### Autres fonctionnalités
- Interface utilisateur intuitive et esthétique
- Possibilité de télécharger tous les résultats (échantillons, statistiques, tableaux, graphiques)
- Visualisations interactives

## Installation et exécution

### Prérequis
- Python 3.8 ou supérieur
- Les bibliothèques Python suivantes : streamlit, pandas, numpy, matplotlib, seaborn

### Installation des dépendances
```bash
pip install streamlit pandas numpy matplotlib seaborn
```

### Exécution de l'application
1. Assurez-vous que tous les fichiers sont dans le même répertoire :
   - `app.py` (application principale)
   - `utils.py` (fonctions utilitaires)

2. Lancez l'application avec Streamlit :
```bash
streamlit run app.py
```

3. L'application s'ouvrira automatiquement dans votre navigateur web par défaut à l'adresse `http://localhost:8501`

## Utilisation

1. **Téléchargement du cadre d'échantillonnage**
   - Utilisez le menu latéral pour télécharger le fichier Excel "Cadre Tunisie.xlsx"
   - Un aperçu des données téléchargées s'affichera

2. **Sélection de la méthode d'échantillonnage**
   - Choisissez entre "Aléatoire Simple sans Remise" et "Stratification à Allocation Proportionnelle"

3. **Paramétrage**
   - Définissez la taille de l'échantillon souhaitée
   - Sélectionnez les variables de comparaison ou de stratification selon la méthode choisie

4. **Génération de l'échantillon**
   - Cliquez sur le bouton "Générer l'échantillon"
   - Consultez les résultats qui s'affichent (échantillon, statistiques, graphiques)

5. **Téléchargement des résultats**
   - Utilisez les boutons de téléchargement pour sauvegarder les différents résultats

## Structure du projet

- `app.py` : Application principale Streamlit
- `utils.py` : Fonctions utilitaires pour l'échantillonnage et l'analyse statistique
- `README.md` : Documentation du projet

## Auteurs

[Votre nom et le nom de votre coéquipier]

École supérieure de la statistique et l'Analyse de l'information  
Année universitaire: 2024-2025
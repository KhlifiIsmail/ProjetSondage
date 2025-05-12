
"""
Script de lancement pour l'Application de Théorie de Sondage.
Ce script vérifie l'environnement, installe les dépendances manquantes,
et lance l'application Streamlit.
"""

import os
import sys
import subprocess
import pkg_resources

def check_dependencies():
    """Vérifie si toutes les dépendances sont installées."""
    with open('requirements.txt', 'r') as f:
        required = [line.strip().split('==')[0] for line in f.readlines()]
    
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = set(required) - installed
    
    return missing

def install_dependencies(packages):
    """Installe les dépendances manquantes."""
    print(f"Installation des dépendances manquantes: {', '.join(packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + list(packages))
    print("Dépendances installées avec succès.")

def check_files():
    """Vérifie si tous les fichiers nécessaires sont présents."""
    required_files = ['app.py', 'utils.py', 'requirements.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Erreur: Fichiers manquants: {', '.join(missing_files)}")
        print("Veuillez vous assurer que tous les fichiers nécessaires sont présents.")
        return False
    
    return True

def launch_app():
    """Lance l'application Streamlit."""
    print("\nLancement de l'Application de Théorie de Sondage...")
    print("L'application sera accessible dans votre navigateur à l'adresse: http://localhost:8501")
    print("Appuyez sur Ctrl+C pour arrêter l'application.")
    
    subprocess.call(["streamlit", "run", "app.py"])

def main():
    """Fonction principale."""
    print("=" * 70)
    print("Initialisation de l'Application de Théorie de Sondage")
    print("=" * 70)
    
    # Vérifier les fichiers
    if not check_files():
        return
    
    # Vérifier et installer les dépendances
    missing_packages = check_dependencies()
    if missing_packages:
        try:
            install_dependencies(missing_packages)
        except Exception as e:
            print(f"Erreur lors de l'installation des dépendances: {e}")
            print("Veuillez installer manuellement les dépendances en exécutant:")
            print("pip install -r requirements.txt")
            return
    else:
        print("Toutes les dépendances sont déjà installées.")
    
    # Lancer l'application
    launch_app()

if __name__ == "__main__":
    main()
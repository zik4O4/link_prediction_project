"""
SCRIPT PRINCIPAL
================
Exécute toutes les étapes du pipeline
"""

import sys
import os

# Add the project root (parent of src/) to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_loader import load_datasets, print_dataset_stats
from src.feature_extraction import prepare_dataset
from src.train_models import compare_with_without_our_measure
from src.visualization import plot_comparison_all_datasets, plot_accuracy_table

def main():
    """Pipeline complet"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "REPRODUCTION DES EXPÉRIENCES")
    print(" " * 15 + "Link Prediction avec Our_measure (parameter-free)")
    print("=" * 80 + "\n")
    
    # ======================================================================
    # ÉTAPE 1 : Chargement des datasets
    # ======================================================================
    print("📂 ÉTAPE 1/4: Chargement des datasets")
    print("-" * 80)
    
    datasets = load_datasets()
    print_dataset_stats(datasets)
    
    # ======================================================================
    # ÉTAPE 2 : Extraction des features
    # ======================================================================
    print("\n📊 ÉTAPE 2/4: Extraction des features")
    print("-" * 80)
    
    prepared_data = {}
    
    for name, G in datasets.items():
        print(f"\n🔄 Traitement de {name}...")
        X_train, X_test, y_train, y_test = prepare_dataset(G, verbose=True)
        prepared_data[name] = (X_train, X_test, y_train, y_test)
    
    # ======================================================================
    # ÉTAPE 3 : Entraînement des modèles
    # ======================================================================
    print("\n🤖 ÉTAPE 3/4: Entraînement des modèles")
    print("-" * 80)
    
    all_results = {}
    
    for name, (X_train, X_test, y_train, y_test) in prepared_data.items():
        print(f"\n📈 Dataset: {name}")
        print("=" * 80)
        
        results = compare_with_without_our_measure(
            X_train, X_test, y_train, y_test, verbose=True
        )
        
        all_results[name] = results
    
    # ======================================================================
    # ÉTAPE 4 : Visualisation
    # ======================================================================
    print("\n📊 ÉTAPE 4/4: Génération des visualisations")
    print("-" * 80)
    
    datasets_names = list(datasets.keys())
    
    plot_comparison_all_datasets(all_results, datasets_names)
    plot_accuracy_table(all_results, datasets_names)
    
    # ======================================================================
    # CONCLUSION
    # ======================================================================
    print("\n" + "=" * 80)
    print("✅ EXPÉRIENCES TERMINÉES AVEC SUCCÈS !")
    print("=" * 80)
    print("\n📁 Fichiers générés:")
    print("   - results/figures/results_comparison.png")
    print("   - results/tables/results_table.csv")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
"""
PARTIE 5 : VISUALISATION
========================
Visualise les résultats de l'entraînement pour tous les datasets (Facebook, dolphins, power grid, football, etc.).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def plot_comparison_all_datasets(all_results, datasets_names, output_dir='results/figures'):
    """Crée les graphiques comparatifs pour tous les datasets"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_datasets = len(datasets_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Link Prediction: Performance Comparison (With vs Without Our_measure)', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, dataset_name in enumerate(datasets_names):
        ax = axes[idx]
        
        results = all_results[dataset_name]
        results_without = results['without']
        results_with = results['with']
        
        models = list(results_without.keys())
        acc_without = [results_without[m]['metrics']['accuracy'] for m in models if 'metrics' in results_without[m]]
        acc_with = [results_with[m]['metrics']['accuracy'] for m in models if 'metrics' in results_with[m]]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, acc_without, width, 
                       label='Without Our_measure', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, acc_with, width, 
                       label='With Our_measure', alpha=0.8, color='coral')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(f'{dataset_name} Dataset', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
    
    if num_datasets < 6:
        for i in range(num_datasets, 6):
            fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    output_path = f'{output_dir}/results_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure sauvegardée: {output_path}")
    
    plt.show()

def plot_accuracy_table(all_results, datasets_names, output_dir='results/tables'):
    """Crée un tableau récapitulatif pour tous les datasets"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    
    for dataset_name in datasets_names:
        results = all_results[dataset_name]
        
        for model_name in results['without'].keys():
            if 'metrics' not in results['without'][model_name] or 'metrics' not in results['with'][model_name]:
                continue
            acc_without = results['without'][model_name]['metrics']['accuracy']
            acc_with = results['with'][model_name]['metrics']['accuracy']
            gain = acc_with - acc_without
            
            data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Without': f"{acc_without:.4f}",
                'With': f"{acc_with:.4f}",
                'Gain': f"{gain:+.4f}"
            })
    
    df = pd.DataFrame(data)
    
    print("\n" + "=" * 80)
    print("TABLEAU RÉCAPITULATIF DES RÉSULTATS POUR TOUS LES DATASETS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    output_path = f'{output_dir}/results_table.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Tableau sauvegardé: {output_path}")
    
    return df

def plot_feature_importance(all_results, datasets_names, output_dir='results/figures'):
    """Visualise l'importance des features pour RF et XGB sur tous les datasets"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name in datasets_names:
        results = all_results[dataset_name]['with']  # Utilise les résultats avec Our_measure
        
        for model_name in ['RF', 'XGB']:
            if model_name in results and 'feature_importance' in results[model_name]['metrics']:
                importance = results[model_name]['metrics']['feature_importance']
                
                features = list(importance.keys())
                values = list(importance.values())
                
                plt.figure(figsize=(10, 6))
                plt.barh(features, values, color='skyblue')
                plt.xlabel('Importance')
                plt.ylabel('Features')
                plt.title(f'Feature Importance - {model_name} on {dataset_name}')
                plt.tight_layout()
                
                output_path = f'{output_dir}/{dataset_name.lower()}_{model_name.lower()}_importance.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✅ Figure sauvegardée: {output_path}")
                plt.close()

if __name__ == "__main__":
    # Pipeline complet : chargement, extraction, entraînement, visualisation
    from data_loader import load_datasets
    from feature_extraction import prepare_multiple_datasets
    from train_models import train_on_multiple_datasets, compare_with_without_our_measure
    
    datasets = load_datasets()
    feature_datasets = prepare_multiple_datasets(datasets, max_samples=500)
    
    print("=" * 60)
    print("VISUALISATION DES RÉSULTATS POUR TOUS LES DATASETS")
    print("=" * 60)
    
    # Entraînement sur tous les datasets
    results_all = train_on_multiple_datasets(feature_datasets, tune_hyperparams=True, cv_folds=3, verbose=False)  # Réduit verbose pour éviter spam
    
    # Comparaisons avec/sans Our_measure pour chaque dataset
    all_comparisons = {}
    for name, (X_train, X_test, y_train, y_test) in feature_datasets.items():
        comparison = compare_with_without_our_measure(X_train, X_test, y_train, y_test, verbose=False)
        all_comparisons[name] = comparison
    
    datasets_names = list(all_comparisons.keys())
    
    # Visualisations
    plot_comparison_all_datasets(all_comparisons, datasets_names)
    plot_accuracy_table(all_comparisons, datasets_names)
    plot_feature_importance(all_comparisons, datasets_names)
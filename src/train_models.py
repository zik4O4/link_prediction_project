"""
PARTIE 4 : ENTRAÎNEMENT DES MODÈLES
=======================================
Entraîne et évalue des modèles de classification pour la prédiction de liens.
Modèles : KNN, LR, RF, NN, SVM, XGBoost (si disponible).
Comparaisons avec/sans Our_measure.
Entraînement sur tous les datasets (Facebook, dolphins, power grid, football, etc.).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

def get_model(model_name, random_state=42):
    """Retourne une instance du modèle spécifié"""
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LR': LogisticRegression(max_iter=1000, random_state=random_state),
        'RF': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'NN': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state) if 'SVM' else None,
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state) if XGB_AVAILABLE else None
    }
    
    if model_name not in models:
        raise ValueError(f"Modèle inconnu: {model_name}. Disponibles: {list(models.keys())}")
    
    model = models[model_name]
    if model is None:
        raise ImportError(f"Modèle {model_name} nécessite une dépendance non installée (e.g., xgboost pour XGB)")
    
    return model

def needs_scaling(model_name):
    """Détermine si le modèle nécessite un scaling"""
    return model_name in ['KNN', 'LR', 'NN', 'SVM']

def train_single_model(model_name, X_train, X_test, y_train, y_test, 
                      scale=None, cv_folds=5, tune_hyperparams=False):
    """
    Entraîne et évalue un modèle unique
    
    Args:
        model_name: nom du modèle
        X_train, X_test: features
        y_train, y_test: labels
        scale: forcer scaling (None = auto)
        cv_folds: nombre de folds pour CV
        tune_hyperparams: activer tuning des hyperparamètres
    
    Returns:
        dict: métriques et modèle
    """
    if X_train.empty or len(y_train) == 0:
        raise ValueError("Dataset d'entraînement vide")
    
    # Scaling
    do_scale = scale if scale is not None else needs_scaling(model_name)
    if do_scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None
    
    # Modèle
    model = get_model(model_name)
    
    # Hyperparameter tuning (optionnel)
    if tune_hyperparams:
        param_grids = {
            'KNN': {'n_neighbors': [3, 5, 7]},
            'LR': {'C': [0.1, 1, 10]},
            'RF': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
            'NN': {'hidden_layer_sizes': [(32,), (64, 32), (128, 64)]},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
            'XGB': {'n_estimators': [50, 100], 'max_depth': [3, 6]}
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='f1')
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
    
    # Entraînement
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    # Cross-validation
    if cv_folds > 1:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='f1')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
    
    # Feature importance (pour RF et XGB)
    if hasattr(model, 'feature_importances_'):
        metrics['feature_importance'] = dict(zip(X_train.columns, model.feature_importances_))
    
    return {
        'metrics': metrics,
        'model': model,
        'scaler': scaler
    }

def train_all_models(X_train, X_test, y_train, y_test, with_our_measure=True, 
                    models_list=None, verbose=True, **kwargs):
    """
    Entraîne tous les modèles disponibles
    
    Args:
        X_train, X_test: features
        y_train, y_test: labels
        with_our_measure: inclure Our_measure
        models_list: liste des modèles (None = tous)
        verbose: afficher progression
        **kwargs: arguments pour train_single_model
    
    Returns:
        dict: résultats par modèle
    """
    if not with_our_measure:
        X_train = X_train.drop(columns=['Our_measure'], errors='ignore')
        X_test = X_test.drop(columns=['Our_measure'], errors='ignore')
    
    if models_list is None:
        models_list = ['KNN', 'LR', 'RF', 'NN', 'SVM']
        if XGB_AVAILABLE:
            models_list.append('XGB')
    
    if verbose:
        feature_status = "AVEC" if with_our_measure else "SANS"
        print(f"\n🤖 Entraînement des modèles ({feature_status} Our_measure)")
        print(f"   Features: {X_train.shape[1]} | Échantillons: {len(y_train)}")
        print("-" * 60)
    
    results = {}
    iterator = tqdm(models_list) if TQDM_AVAILABLE and verbose else models_list
    
    for model_name in iterator:
        try:
            result = train_single_model(
                model_name, X_train, X_test, y_train, y_test, **kwargs
            )
            results[model_name] = result
            
            if verbose:
                acc = result['metrics']['accuracy']
                f1 = result['metrics']['f1']
                print(f"   {model_name:<5} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        except Exception as e:
            print(f"❌ Erreur pour {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def compare_with_without_our_measure(X_train, X_test, y_train, y_test, 
                                    models_list=None, verbose=True, **kwargs):
    """Compare AVEC et SANS Our_measure"""
    
    if verbose:
        print("\n" + "=" * 60)
        print("COMPARAISON AVEC / SANS OUR_MEASURE")
        print("=" * 60)
    
    results_without = train_all_models(
        X_train.copy(), X_test.copy(), y_train, y_test,
        with_our_measure=False, models_list=models_list, verbose=verbose, **kwargs
    )
    
    results_with = train_all_models(
        X_train.copy(), X_test.copy(), y_train, y_test,
        with_our_measure=True, models_list=models_list, verbose=verbose, **kwargs
    )
    
    if verbose:
        print("\n📊 RÉSULTATS COMPARATIFS:")
        print("-" * 60)
        print(f"{'Modèle':<8} {'Sans Our_measure':<18} {'Avec Our_measure':<18} {'Gain':<10}")
        print("-" * 60)
        
        for model_name in results_without.keys():
            if 'error' in results_without[model_name]:
                continue
            acc_without = results_without[model_name]['metrics']['accuracy']
            acc_with = results_with[model_name]['metrics']['accuracy']
            gain = acc_with - acc_without
            
            print(f"{model_name:<8} {acc_without:.4f}            {acc_with:.4f}            {gain:+.4f}")
        
        print("=" * 60)
    
    return {
        'without': results_without,
        'with': results_with
    }

def save_models(results, output_dir="models", prefix="model"):
    """Sauvegarde les modèles entraînés"""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, result in results.items():
        if 'model' in result:
            filename = f"{output_dir}/{prefix}_{model_name}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(result, f)
    
    print(f"💾 Modèles sauvegardés dans {output_dir}/")

def plot_results(results, output_file=None):
    """Visualise les performances des modèles"""
    try:
        import matplotlib.pyplot as plt
        
        models = []
        accuracies = []
        f1_scores = []
        
        for model_name, result in results.items():
            if 'metrics' in result:
                models.append(model_name)
                accuracies.append(result['metrics']['accuracy'])
                f1_scores.append(result['metrics']['f1'])
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Modèles')
        ax.set_ylabel('Score')
        ax.set_title('Performances des Modèles')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"📊 Figure sauvegardée: {output_file}")
        else:
            plt.show()
    
    except ImportError:
        print("⚠️  Matplotlib non installé. Installez avec: pip install matplotlib")

def train_on_multiple_datasets(datasets, **kwargs):
    """
    Entraîne sur tous les datasets
    
    Args:
        datasets: dict {nom: (X_train, X_test, y_train, y_test)} depuis feature_extraction
        **kwargs: arguments pour train_all_models
    
    Returns:
        dict: résultats par dataset
    """
    results = {}
    for name, (X_train, X_test, y_train, y_test) in datasets.items():
        print(f"\n🏷️  Dataset: {name}")
        try:
            results[name] = train_all_models(X_train, X_test, y_train, y_test, **kwargs)
        except Exception as e:
            print(f"❌ Erreur pour {name}: {e}")
    
    return results

if __name__ == "__main__":
    # Test de l'entraînement
    from data_loader import load_datasets
    from feature_extraction import prepare_multiple_datasets
    
    datasets = load_datasets()
    feature_datasets = prepare_multiple_datasets(datasets, max_samples=500)
    
    print("=" * 60)
    print("TEST DE L'ENTRAÎNEMENT DES MODÈLES SUR TOUS LES DATASETS")
    print("=" * 60)
    
    # Entraînement sur tous les datasets
    results_all = train_on_multiple_datasets(feature_datasets, tune_hyperparams=True, cv_folds=3, verbose=True)
    
    # Comparaison avec/sans Our_measure pour chaque dataset
    for name, (X_train, X_test, y_train, y_test) in feature_datasets.items():
        print(f"\n🔍 Comparaison pour {name}:")
        comparison = compare_with_without_our_measure(X_train, X_test, y_train, y_test, verbose=True)
        
        # Sauvegarde des modèles
        save_models(results_all[name], prefix=f"{name.lower()}")
        
        # Visualisation des performances
        plot_results(results_all[name], output_file=f"models/{name.lower()}_performance.png")
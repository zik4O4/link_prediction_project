"""
PARTIE 3 : EXTRACTION DES FEATURES
===================================
Prépare les données pour l'apprentissage supervisé :
- Génère des liens positifs et négatifs
- Calcule les features pour chaque paire
- Équilibre les classes
- Sauvegarde et visualisation optionnelles
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from similarity_metrics import compute_all_metrics
import os

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def generate_positive_samples(G, max_samples=None):
    """
    Génère des échantillons positifs (liens existants)
    
    Args:
        G: graphe NetworkX
        max_samples: nombre maximum d'échantillons (None = tous)
    
    Returns:
        list: liste de tuples (u, v, label=1)
    
    Raises:
        ValueError: si le graphe est vide
    """
    if G.number_of_edges() == 0:
        raise ValueError("Le graphe n'a pas d'arêtes")
    
    pos_edges = list(G.edges())
    if max_samples:
        pos_edges = pos_edges[:max_samples]
    
    pos_samples = [(u, v, 1) for u, v in pos_edges]
    return pos_samples

def generate_negative_samples(G, num_samples, max_attempts_factor=10):
    """
    Génère des échantillons négatifs (liens inexistants)
    
    Args:
        G: graphe NetworkX
        num_samples: nombre d'échantillons négatifs à générer
        max_attempts_factor: facteur pour limiter les tentatives (évite boucle infinie)
    
    Returns:
        list: liste de tuples (u, v, label=0)
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return []
    
    neg_samples = set()  # Utilise un set pour éviter les doublons
    attempts = 0
    max_attempts = num_samples * max_attempts_factor
    
    while len(neg_samples) < num_samples and attempts < max_attempts:
        u, v = np.random.choice(nodes, 2, replace=False)
        
        # Vérifie que ce n'est pas un lien existant et pas déjà dans les négatifs
        edge_tuple = tuple(sorted((u, v)))
        if not G.has_edge(u, v) and edge_tuple not in neg_samples:
            neg_samples.add(edge_tuple)
        
        attempts += 1
    
    if len(neg_samples) < num_samples:
        print(f"⚠️  Seulement {len(neg_samples)} échantillons négatifs générés (sur {num_samples} demandés)")
    
    return [(u, v, 0) for u, v in neg_samples]

def extract_features(G, edge_samples, verbose=True, use_cache=False):
    """
    Extrait les features pour chaque paire de nœuds
    
    Args:
        G: graphe NetworkX
        edge_samples: liste de (u, v, label)
        verbose: afficher la progression
        use_cache: utiliser un cache pour les métriques (pour optimisation)
    
    Returns:
        X: DataFrame des features
        y: array des labels
    """
    features_list = []
    labels = []
    
    total = len(edge_samples)
    iterator = tqdm(edge_samples) if TQDM_AVAILABLE and verbose else edge_samples
    
    cache = {} if use_cache else None
    
    for sample in iterator:
        u, v, label = sample
        
        # Clé de cache
        key = tuple(sorted((u, v)))
        if cache and key in cache:
            metrics = cache[key]
        else:
            metrics = compute_all_metrics(G, u, v)
            if cache:
                cache[key] = metrics
        
        features_list.append(metrics)
        labels.append(label)
    
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    return X, y

def prepare_dataset(G, test_ratio=0.2, balance_classes=True, max_samples=None, verbose=True):
    """
    Prépare le dataset complet pour l'apprentissage
    
    Args:
        G: graphe NetworkX
        test_ratio: proportion du test set
        balance_classes: équilibrer positifs et négatifs
        max_samples: nombre max d'échantillons par classe (None = tous)
        verbose: afficher les informations
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    
    if verbose:
        print(f"\n🔄 Préparation du dataset...")
        print(f"   Graphe: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    
    # Générer les échantillons positifs
    pos_samples = generate_positive_samples(G, max_samples)
    
    # Générer les négatifs (équilibrés si demandé)
    num_neg = len(pos_samples) if balance_classes else min(max_samples or len(pos_samples), len(pos_samples))
    neg_samples = generate_negative_samples(G, num_neg)
    
    all_samples = pos_samples + neg_samples
    np.random.shuffle(all_samples)
    
    if verbose:
        print(f"   Échantillons: {len(pos_samples)} positifs, {len(neg_samples)} négatifs")
        print(f"\n📊 Extraction des features...")
    
    # Extraire les features
    X, y = extract_features(G, all_samples, verbose=verbose)
    
    if verbose:
        print(f"   ✅ {X.shape[0]} échantillons, {X.shape[1]} features")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )
    
    if verbose:
        print(f"   Train: {len(y_train)} | Test: {len(y_test)}\n")
    
    return X_train, X_test, y_train, y_test

def save_dataset_to_csv(X_train, X_test, y_train, y_test, output_dir="datasets/features", prefix="dataset"):
    """
    Sauvegarde les datasets en CSV
    
    Args:
        X_train, X_test: DataFrames des features
        y_train, y_test: arrays des labels
        output_dir: dossier de sortie
        prefix: préfixe des fichiers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f"{output_dir}/{prefix}_X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/{prefix}_X_test.csv", index=False)
    pd.DataFrame(y_train, columns=['label']).to_csv(f"{output_dir}/{prefix}_y_train.csv", index=False)
    pd.DataFrame(y_test, columns=['label']).to_csv(f"{output_dir}/{prefix}_y_test.csv", index=False)
    
    print(f"💾 Datasets sauvegardés dans {output_dir}/")

def plot_feature_distributions(X, y, output_file=None):
    """
    Visualise la distribution des features
    
    Args:
        X: DataFrame des features
        y: array des labels
        output_file: fichier de sortie pour la figure (None = afficher)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df = X.copy()
        df['label'] = y
        
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(X.columns):
            if i < len(axes):
                sns.histplot(data=df, x=col, hue='label', ax=axes[i], alpha=0.7)
                axes[i].set_title(col)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"📊 Figure sauvegardée: {output_file}")
        else:
            plt.show()
    
    except ImportError:
        print("⚠️  Matplotlib/Seaborn non installés. Installez avec: pip install matplotlib seaborn")

def prepare_multiple_datasets(datasets, **kwargs):
    """
    Prépare les datasets pour tous les graphes
    
    Args:
        datasets: dict {nom: graphe} depuis data_loader
        **kwargs: arguments pour prepare_dataset
    
    Returns:
        dict: {nom: (X_train, X_test, y_train, y_test)}
    """
    results = {}
    for name, G in datasets.items():
        print(f"\n🏷️  Dataset: {name}")
        try:
            results[name] = prepare_dataset(G, **kwargs)
        except Exception as e:
            print(f"❌ Erreur pour {name}: {e}")
    
    return results

if __name__ == "__main__":
    # Test de l'extraction
    from data_loader import load_datasets
    
    datasets = load_datasets()
    
    print("=" * 60)
    print("TEST DE L'EXTRACTION DES FEATURES")
    print("=" * 60)
    
    # Préparation pour tous les datasets (au lieu de seulement Facebook)
    feature_datasets = prepare_multiple_datasets(datasets, max_samples=1000, verbose=True)
    
    # Sauvegarde et visualisation pour chaque dataset
    for name, (X_train, X_test, y_train, y_test) in feature_datasets.items():
        print(f"\n📋 Features extraites pour {name} (échantillon):")
        print(X_train.head())
        
        # Sauvegarde
        save_dataset_to_csv(X_train, X_test, y_train, y_test, prefix=name.lower())
        
        # Visualisation
        plot_feature_distributions(X_train, y_train, output_file=f"datasets/features/{name.lower()}_distributions.png")
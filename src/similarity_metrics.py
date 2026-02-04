"""
PARTIE 2 : MÉTRIQUES DE SIMILARITÉ
===================================
Implémente toutes les métriques de similarité :
- CN, AA, PA, JA, RA, Salton, HD, HP, LHN, Sorensen
- Our_measure (parameter-free, optimisé)
"""

import networkx as nx
import numpy as np
from itertools import islice
from collections import defaultdict

# ============================================================
# MÉTRIQUE PARAMETER-FREE (Our_measure) - OPTIMISÉ
# ============================================================

def compute_our_measure(G, u, v, l=3, max_paths=10000):
    """
    Calcule la métrique parameter-free J(u,v) selon la formule exacte.
    
    J(x, y) = |path_{x,y}^{length ≤ l}| / (|Γ(x)| + |Γ(y)| + 1)
    
    where:
    - |path_{x,y}^{length ≤ l}| is the number of simple paths between nodes x and y
      with length less than or equal to l.
      Note that only simple paths are considered; paths such as
      1 → 2 → 5 → 2 → 1 are not included.
    - l is the path length.
    - |Γ(x)| is the number of neighbors (degree) of node x.
    
    Args:
        G: graphe NetworkX (non-dirigé)
        u, v: nœuds à comparer
        l: longueur maximale des chemins (défaut: 3)
        max_paths: limite pour éviter l'explosion (défaut: 10000, ajustez si besoin)
    
    Returns:
        float: valeur de similarité
    
    Raises:
        ValueError: si u ou v n'existe pas
    """
    if u not in G or v not in G:
        raise ValueError(f"Nœuds {u} ou {v} n'existent pas dans le graphe")
    
    if u == v:
        return 0.0
    
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    denominator = deg_u + deg_v + 1
    
    # Comptage des chemins simples de longueur 1 à l
    num_paths = 0
    queue = [(u, [u], 0)]  # (nœud, chemin, longueur_actuelle)
    
    while queue and num_paths < max_paths:
        current, path, length = queue.pop(0)
        
        if length > l:
            continue
        
        if current == v and length > 0:
            num_paths += 1
            continue  # Continue pour compter tous les chemins, pas seulement un
        
        for neighbor in G.neighbors(current):
            if neighbor not in path:  # Pas de cycles
                new_path = path + [neighbor]
                queue.append((neighbor, new_path, length + 1))
    
    return num_paths / denominator

# ============================================================
# MÉTRIQUES CLASSIQUES - AVEC GESTION D'ERREURS
# ============================================================

def common_neighbors(G, u, v):
    """Common Neighbors (CN)"""
    if u not in G or v not in G:
        return 0
    return len(list(nx.common_neighbors(G, u, v)))

def adamic_adar(G, u, v):
    """Adamic-Adar (AA) - Calcul direct pour plus de contrôle"""
    if u not in G or v not in G:
        return 0.0
    cn = list(nx.common_neighbors(G, u, v))
    if not cn:
        return 0.0
    return sum(1 / np.log(G.degree(w)) for w in cn if G.degree(w) > 1)

def preferential_attachment(G, u, v):
    """Preferential Attachment (PA)"""
    if u not in G or v not in G:
        return 0
    return G.degree(u) * G.degree(v)

def jaccard_coefficient(G, u, v):
    """Jaccard (JA)"""
    if u not in G or v not in G:
        return 0.0
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = len(neighbors_u & neighbors_v)
    union = len(neighbors_u | neighbors_v)
    return intersection / union if union > 0 else 0.0

def resource_allocation(G, u, v):
    """Resource Allocation (RA)"""
    if u not in G or v not in G:
        return 0.0
    ra = 0.0
    for w in nx.common_neighbors(G, u, v):
        deg_w = G.degree(w)
        if deg_w > 0:
            ra += 1.0 / deg_w
    return ra

def salton_index(G, u, v):
    """Salton Index"""
    if u not in G or v not in G:
        return 0.0
    cn = common_neighbors(G, u, v)
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    return cn / np.sqrt(deg_u * deg_v) if deg_u > 0 and deg_v > 0 else 0.0

def hub_depressed(G, u, v):
    """Hub Depressed (HD)"""
    if u not in G or v not in G:
        return 0.0
    cn = common_neighbors(G, u, v)
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    min_deg = min(deg_u, deg_v)
    return cn / min_deg if min_deg > 0 else 0.0

def hub_promoted(G, u, v):
    """Hub Promoted (HP)"""
    if u not in G or v not in G:
        return 0.0
    cn = common_neighbors(G, u, v)
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    max_deg = max(deg_u, deg_v)
    return cn / max_deg if max_deg > 0 else 0.0

def leicht_holme_newman(G, u, v):
    """Leicht-Holme-Newman (LHN)"""
    if u not in G or v not in G:
        return 0.0
    cn = common_neighbors(G, u, v)
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    product = deg_u * deg_v
    return cn / product if product > 0 else 0.0

def sorensen_index(G, u, v):
    """Sorensen Index"""
    if u not in G or v not in G:
        return 0.0
    cn = common_neighbors(G, u, v)
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    sum_deg = deg_u + deg_v
    return (2 * cn) / sum_deg if sum_deg > 0 else 0.0

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def compute_all_metrics(G, u, v):
    """
    Calcule toutes les métriques de similarité pour une paire (u, v)
    
    Args:
        G: graphe NetworkX
        u, v: nœuds à comparer
    
    Returns:
        dict: {nom_métrique: valeur}
    """
    metrics = {}
    metrics['CN'] = common_neighbors(G, u, v)
    metrics['AA'] = adamic_adar(G, u, v)
    metrics['PA'] = preferential_attachment(G, u, v)
    metrics['JA'] = jaccard_coefficient(G, u, v)
    metrics['RA'] = resource_allocation(G, u, v)
    metrics['Salton'] = salton_index(G, u, v)
    metrics['HD'] = hub_depressed(G, u, v)
    metrics['HP'] = hub_promoted(G, u, v)
    metrics['LHN'] = leicht_holme_newman(G, u, v)
    metrics['Sorensen'] = sorensen_index(G, u, v)
    metrics['Our_measure'] = compute_our_measure(G, u, v)
    return metrics

def compute_metrics_for_all_pairs(G, non_edges_only=True, sample_size=None):
    """
    Calcule les métriques pour toutes les paires de nœuds.
    
    Args:
        G: graphe NetworkX
        non_edges_only: si True, calcule seulement pour les paires non-connectées
        sample_size: nombre maximum de paires à calculer (pour les gros graphes)
    
    Returns:
        list: [(u, v, metrics_dict), ...]
    """
    results = []
    nodes = list(G.nodes())
    pairs = []
    
    if non_edges_only:
        non_edges = list(nx.non_edges(G))
        pairs = non_edges[:sample_size] if sample_size else non_edges
    else:
        from itertools import combinations
        pairs = list(combinations(nodes, 2))[:sample_size] if sample_size else combinations(nodes, 2)
    
    for u, v in pairs:
        metrics = compute_all_metrics(G, u, v)
        results.append((u, v, metrics))
    
    return results

def normalize_metrics(metrics_dict, method='minmax'):
    """
    Normalise les métriques pour une meilleure comparabilité.
    
    Args:
        metrics_dict: dict de métriques
        method: 'minmax' ou 'zscore'
    
    Returns:
        dict: métriques normalisées
    """
    normalized = {}
    values = list(metrics_dict.values())
    
    if method == 'minmax':
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            for k, v in metrics_dict.items():
                normalized[k] = (v - min_val) / (max_val - min_val)
        else:
            normalized = metrics_dict.copy()
    elif method == 'zscore':
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val > 0:
            for k, v in metrics_dict.items():
                normalized[k] = (v - mean_val) / std_val
        else:
            normalized = metrics_dict.copy()
    
    return normalized

# ============================================================
# TEST ET EXEMPLE D'UTILISATION
# ============================================================

if __name__ == "__main__":
    # Test des métriques
    G = nx.karate_club_graph()
    
    print("=" * 60)
    print("TEST DES MÉTRIQUES DE SIMILARITÉ")
    print("=" * 60 + "\n")
    
    test_pairs = [(0, 33), (0, 2), (1, 3)]
    
    for u, v in test_pairs:
        print(f"Paire ({u}, {v}):")
        metrics = compute_all_metrics(G, u, v)
        
        for name, value in metrics.items():
            print(f"  {name:15s} = {value:.4f}")
        print()
    
    # Exemple d'utilisation pour toutes les paires non-connectées (échantillon)
    print("Exemple : Calcul pour 5 paires non-connectées :")
    sample_results = compute_metrics_for_all_pairs(G, sample_size=5)
    for u, v, mets in sample_results:
        print(f"({u}, {v}): CN={mets['CN']}, Our_measure={mets['Our_measure']:.4f}")
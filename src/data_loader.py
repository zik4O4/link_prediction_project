"""
PARTIE 1 : CHARGEMENT DES DATASETS
===================================
Charge les 4 datasets réels :
1. Facebook (ego-facebook.edges)
2. Dolphins (soc-dolphins.mtx)
3. Football (football.mtx)
4. Power Grid (power-US-Grid.mtx)
"""

import networkx as nx
import os

def load_mtx_file(filepath):
    """
    Charge un graphe depuis un fichier Matrix Market (.mtx)
    
    Args:
        filepath: chemin vers le fichier .mtx
    
    Returns:
        G: graphe NetworkX
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    print(f"   📂 Chargement depuis: {filepath}")
    
    try:
        from scipy.io import mmread
        
        # Charger la matrice
        matrix = mmread(filepath)
        
        # Convertir en graphe NetworkX
        G = nx.from_scipy_sparse_array(matrix)
        
        # Supprimer les self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Garder la plus grande composante connexe
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # Renuméroter les nœuds de 0 à n-1
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        
        print(f"   ✅ Chargé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
        
        return G
        
    except Exception as e:
        print(f"   ❌ Erreur lors du chargement MTX: {e}")
        raise

def load_edge_list_file(filepath, delimiter=None, comments='#'):
    """
    Charge un graphe depuis un fichier edge list
    
    Args:
        filepath: chemin vers le fichier
        delimiter: séparateur (None = auto)
        comments: caractère de commentaire
    
    Returns:
        G: graphe NetworkX
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    print(f"   📂 Chargement depuis: {filepath}")
    
    G = nx.Graph()
    
    with open(filepath, 'r') as f:
        for line in f:
            # Ignorer les commentaires
            if line.startswith(comments) or line.startswith('%'):
                continue
            
            line = line.strip()
            if not line:
                continue
            
            # Séparer les nœuds
            if delimiter:
                parts = line.split(delimiter)
            else:
                parts = line.split()
            
            if len(parts) >= 2:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    G.add_edge(u, v)
                except ValueError:
                    continue
    
    if G.number_of_nodes() == 0:
        raise ValueError(f"Le fichier {filepath} ne contient aucune arête valide")
    
    # Garder la plus grande composante connexe
    if not nx.is_connected(G):
        print(f"   ⚠️  Graphe non connexe, extraction de la plus grande composante")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # Renuméroter les nœuds de 0 à n-1
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    
    print(f"   ✅ Chargé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    
    return G

# ============================================================
# CHARGEMENT DES 4 DATASETS
# ============================================================

def load_facebook():
    """1. Facebook (ego-network)"""
    filepath = 'datasets/facebook/ego-facebook.edges'
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ Fichier non trouvé: {filepath}\n"
            f"💡 Placez 'ego-facebook.edges' dans datasets/facebook/"
        )
    
    return load_edge_list_file(filepath, delimiter=',')  # <-- Add this: specify comma as delimiter

def load_dolphins():
    """2. Dolphins Social Network"""
    filepath = 'datasets/dolphins/soc-dolphins.mtx'
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ Fichier non trouvé: {filepath}\n"
            f"💡 Placez 'soc-dolphins.mtx' dans datasets/dolphins/"
        )
    
    return load_mtx_file(filepath)

def load_football():
    """3. American Football"""
    filepath = 'datasets/football/football.mtx'
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ Fichier non trouvé: {filepath}\n"
            f"💡 Placez 'football.mtx' dans datasets/football/"
        )
    
    return load_mtx_file(filepath)

def load_power():
    """4. US Power Grid"""
    filepath = 'datasets/power/power-US-Grid.mtx'
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ Fichier non trouvé: {filepath}\n"
            f"💡 Placez 'power-US-Grid.mtx' dans datasets/power/"
        )
    
    return load_mtx_file(filepath)

# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def load_datasets():
    """
    Charge les 4 datasets
    
    Returns:
        dict: {nom_dataset: graphe NetworkX}
    """
    datasets = {}
    
    print("\n" + "=" * 70)
    print(" " * 20 + "CHARGEMENT DES DATASETS")
    print("=" * 70 + "\n")
    
    # 1. Facebook
    print("1️⃣ Facebook (Ego-network)")
    print("-" * 70)
    try:
        datasets['Facebook'] = load_facebook()
        print(f"   ✅ {datasets['Facebook'].number_of_nodes()} nœuds, "
              f"{datasets['Facebook'].number_of_edges()} arêtes\n")
    except Exception as e:
        print(f"   ❌ Erreur: {e}\n")
        raise
    
    # 2. Dolphins
    print("2️⃣ Dolphins (Social Network)")
    print("-" * 70)
    try:
        datasets['Dolphins'] = load_dolphins()
        print(f"   ✅ {datasets['Dolphins'].number_of_nodes()} nœuds, "
              f"{datasets['Dolphins'].number_of_edges()} arêtes\n")
    except Exception as e:
        print(f"   ❌ Erreur: {e}\n")
        raise
    
    # 3. Football
    print("3️⃣ Football (American College)")
    print("-" * 70)
    try:
        datasets['Football'] = load_football()
        print(f"   ✅ {datasets['Football'].number_of_nodes()} nœuds, "
              f"{datasets['Football'].number_of_edges()} arêtes\n")
    except Exception as e:
        print(f"   ❌ Erreur: {e}\n")
        raise
    
    # 4. Power Grid
    print("4️⃣ Power Grid (US Infrastructure)")
    print("-" * 70)
    try:
        datasets['Power'] = load_power()
        print(f"   ✅ {datasets['Power'].number_of_nodes()} nœuds, "
              f"{datasets['Power'].number_of_edges()} arêtes\n")
    except Exception as e:
        print(f"   ❌ Erreur: {e}\n")
        raise
    
    print("=" * 70)
    print(f"✅ {len(datasets)} datasets chargés avec succès !")
    print("=" * 70 + "\n")
    
    return datasets

def print_dataset_stats(datasets):
    """Affiche des statistiques détaillées"""
    
    print("\n" + "=" * 90)
    print(" " * 30 + "STATISTIQUES DES DATASETS")
    print("=" * 90)
    
    print(f"\n{'Dataset':<15} {'Nœuds':>8} {'Arêtes':>10} {'Densité':>10} "
          f"{'Degré moy':>12} {'Diamètre':>10}")
    print("-" * 90)
    
    for name, G in datasets.items():
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        density = nx.density(G)
        avg_degree = sum(dict(G.degree()).values()) / nodes
        
        try:
            diameter = nx.diameter(G) if nx.is_connected(G) else "N/A"
        except:
            diameter = "N/A"
        
        print(f"{name:<15} {nodes:>8} {edges:>10} {density:>10.4f} "
              f"{avg_degree:>12.2f} {diameter:>10}")
    
    print("=" * 90 + "\n")

if __name__ == "__main__":
    try:
        datasets = load_datasets()
        print_dataset_stats(datasets)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        print("\n💡 Vérifiez que tous les fichiers sont dans les bons dossiers:")
        print("   - datasets/facebook/ego-facebook.edges")
        print("   - datasets/dolphins/soc-dolphins.mtx")
        print("   - datasets/football/football.mtx")
        print("   - datasets/power/power-US-Grid.mtx")
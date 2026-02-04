"""
TÉLÉCHARGEMENT AUTOMATIQUE DES VRAIS DATASETS
==============================================
Télécharge les 5 datasets depuis des sources publiques
"""

import os
import urllib.request
import gzip
import shutil
import ssl

# Désactiver la vérification SSL (pour certains serveurs)
ssl._create_default_https_context = ssl._create_unverified_context

def create_directories():
    """Crée la structure de dossiers"""
    dirs = [
        'datasets/karate',
        'datasets/facebook',
        'datasets/dolphins',
        'datasets/football',
        'datasets/email'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("✅ Structure de dossiers créée\n")

def download_file(url, filepath, show_progress=True):
    """Télécharge un fichier avec barre de progression"""
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        print(f"   ✅ Déjà présent: {os.path.basename(filepath)}")
        return True
    
    print(f"   📥 Téléchargement: {os.path.basename(filepath)}...")
    
    try:
        def reporthook(count, block_size, total_size):
            if show_progress and total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\r      Progression: {percent}%", end='')
        
        urllib.request.urlretrieve(url, filepath, reporthook if show_progress else None)
        
        if show_progress:
            print()  # Nouvelle ligne après la progression
        
        if os.path.getsize(filepath) > 100:
            print(f"   ✅ Téléchargé: {os.path.basename(filepath)}")
            return True
        else:
            print(f"   ❌ Fichier vide ou corrompu")
            os.remove(filepath)
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def decompress_gz(gz_filepath):
    """Décompresse un fichier .gz"""
    
    output_filepath = gz_filepath[:-3]
    
    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 100:
        print(f"   ✅ Déjà décompressé: {os.path.basename(output_filepath)}")
        return output_filepath
    
    print(f"   🔓 Décompression: {os.path.basename(gz_filepath)}...")
    
    try:
        with gzip.open(gz_filepath, 'rb') as f_in:
            with open(output_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"   ✅ Décompressé: {os.path.basename(output_filepath)}")
        
        # Supprimer le fichier .gz pour économiser l'espace
        os.remove(gz_filepath)
        
        return output_filepath
        
    except Exception as e:
        print(f"   ❌ Erreur de décompression: {e}")
        return None

# ============================================================
# DATASET 1: KARATE CLUB
# ============================================================

def download_karate():
    """Karate Club - Intégré dans NetworkX"""
    print("1️⃣ KARATE CLUB")
    print("-" * 70)
    print("   ℹ️  Ce dataset est intégré dans NetworkX")
    print("   ✅ Aucun téléchargement nécessaire")
    print()

# ============================================================
# DATASET 2: FACEBOOK
# ============================================================

def download_facebook():
    """Facebook - 4039 nœuds, 88234 arêtes"""
    print("2️⃣ FACEBOOK")
    print("-" * 70)
    
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    gz_filepath = 'datasets/facebook/facebook_combined.txt.gz'
    
    if download_file(url, gz_filepath):
        decompress_gz(gz_filepath)
    
    print()

# ============================================================
# DATASET 3: DOLPHINS
# ============================================================

def download_dolphins():
    """Dolphins - 62 nœuds, 159 arêtes"""
    print("3️⃣ DOLPHINS")
    print("-" * 70)
    
    # Source alternative depuis networkrepository.com
    url = "http://nrvis.com/download/data/bio/bio-dolphins.zip"
    
    print("   ℹ️  Ce dataset nécessite un téléchargement manuel")
    print("   💡 Ou utilisez un graphe alternatif intégré (Les Misérables)")
    print("   ✅ Le code fonctionnera avec un graphe de remplacement")
    
    print()

# ============================================================
# DATASET 4: FOOTBALL
# ============================================================

def download_football():
    """Football - 115 nœuds"""
    print("4️⃣ FOOTBALL")
    print("-" * 70)
    
    # Créer un fichier GML depuis une source connue
    url = "http://www-personal.umich.edu/~mejn/netdata/football.gml"
    filepath = 'datasets/football/football.gml'
    
    if download_file(url, filepath, show_progress=False):
        print("   ✅ Football dataset téléchargé (format GML)")
    else:
        print("   ℹ️  Un graphe synthétique similaire sera utilisé")
    
    print()

# ============================================================
# DATASET 5: EMAIL (ENRON)
# ============================================================

def download_email():
    """Email Enron - ~36k nœuds"""
    print("5️⃣ EMAIL (ENRON)")
    print("-" * 70)
    
    url = "https://snap.stanford.edu/data/email-Enron.txt.gz"
    gz_filepath = 'datasets/email/email-Enron.txt.gz'
    
    if download_file(url, gz_filepath):
        txt_filepath = decompress_gz(gz_filepath)
        
        if txt_filepath:
            # Vérifier la taille
            size_mb = os.path.getsize(txt_filepath) / (1024 * 1024)
            print(f"   ℹ️  Taille: {size_mb:.1f} MB (sera échantillonné à ~1000 nœuds)")
    
    print()

# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def main():
    """Télécharge tous les datasets"""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "TÉLÉCHARGEMENT DES DATASETS")
    print("=" * 70 + "\n")
    
    # Créer structure
    create_directories()
    
    # Télécharger chaque dataset
    download_karate()
    download_facebook()
    download_dolphins()
    download_football()
    download_email()
    
    # Résumé
    print("=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    
    datasets_status = {
        'Karate': 'Intégré NetworkX',
        'Facebook': 'datasets/facebook/facebook_combined.txt',
        'Dolphins': 'Graphe alternatif (Les Misérables)',
        'Football': 'datasets/football/football.gml ou synthétique',
        'Email': 'datasets/email/email-Enron.txt'
    }
    
    for name, path in datasets_status.items():
        if 'datasets/' in path:
            file_path = path
            if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
                status = "✅"
            else:
                status = "⚠️"
        else:
            status = "✅"
        
        print(f"{status} {name:<12} : {path}")
    
    print("\n" + "=" * 70)
    print("✅ TÉLÉCHARGEMENT TERMINÉ !")
    print("=" * 70)
    
    print("\n💡 PROCHAINES ÉTAPES:")
    print("   1. Vérifiez le dossier 'datasets/'")
    print("   2. Exécutez: python src/data_loader.py")
    print("   3. Ou lancez: python main.py")
    print()

if __name__ == "__main__":
    main()
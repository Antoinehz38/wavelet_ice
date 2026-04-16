import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def charger_donnees(dossier_racine):
    """Parcourt les dossiers et charge les données des JSON dans un DataFrame Pandas."""
    lignes = []
    chemin_racine = Path(dossier_racine)
    
    # Recherche récursive de tous les fichiers finissant par _energy.json
    fichiers_json = list(chemin_racine.rglob('*_energy.json'))
    
    if not fichiers_json:
        print(f"Aucun fichier *_energy.json trouvé dans {dossier_racine}")
        return pd.DataFrame()

    for fichier in fichiers_json:
        # Extraire le nom de la transformation depuis le nom du fichier
        transformation = fichier.name.replace('_energy.json', '')
        
        with open(fichier, 'r') as f:
            try:
                donnees = json.load(f)
            except json.JSONDecodeError:
                print(f"Erreur de lecture du fichier {fichier}")
                continue
                
        # Extraire les métriques pour chaque modulation
        for modulation, metriques in donnees.items():
            bleed_overall = metriques.get('bleed', {}).get('overall', 0.0)
            underfill_overall = metriques.get('underfill', {}).get('overall', 0.0)
            count = metriques.get('total_count', 1)
            
            lignes.append({
                'Transformation': transformation,
                'Modulation': modulation,
                'Bleed_Overall': bleed_overall,
                'Underfill_Overall': underfill_overall,
                'Count': count
            })
            
    return pd.DataFrame(lignes)

def generer_rapports(df, dossier_sortie="rapports"):
    """Génère et sauvegarde les visualisations."""
    if df.empty:
        return
        
    os.makedirs(dossier_sortie, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Analyse Globale par Transformation (Moyenne générale)
    # ---------------------------------------------------------
    df_global = df.groupby('Transformation')[['Bleed_Overall', 'Underfill_Overall']].mean().reset_index()
    
    # Passage en format "long" pour Seaborn
    df_global_melted = df_global.melt(id_vars='Transformation', 
                                      var_name='Metrique', 
                                      value_name='Moyenne')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_global_melted, x='Transformation', y='Moyenne', hue='Metrique', palette='Set2')
    plt.title('Moyenne globale de Bleed et Underfill par Transformation')
    plt.ylabel('Valeur Overall')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(dossier_sortie, 'global_par_transformation.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. Heatmap : Bleed par Transformation et Modulation
    # ---------------------------------------------------------
    pivot_bleed = df.pivot_table(values='Bleed_Overall', 
                                 index='Modulation', 
                                 columns='Transformation', 
                                 aggfunc='mean')
                                 
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_bleed, annot=True, cmap='YlOrRd', fmt=".4f", linewidths=.5)
    plt.title('Heatmap: Moyenne du "Bleed Overall" par Modulation et Transformation')
    plt.ylabel('Modulation')
    plt.xlabel('Transformation')
    plt.tight_layout()
    plt.savefig(os.path.join(dossier_sortie, 'heatmap_bleed.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Heatmap : Underfill par Transformation et Modulation
    # ---------------------------------------------------------
    pivot_underfill = df.pivot_table(values='Underfill_Overall', 
                                     index='Modulation', 
                                     columns='Transformation', 
                                     aggfunc='mean')
                                     
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_underfill, annot=True, cmap='PuBu', fmt=".4f", linewidths=.5)
    plt.title('Heatmap: Moyenne de "l\'Underfill Overall" par Modulation et Transformation')
    plt.ylabel('Modulation')
    plt.xlabel('Transformation')
    plt.tight_layout()
    plt.savefig(os.path.join(dossier_sortie, 'heatmap_underfill.png'), dpi=300)
    plt.close()

    print(f"✅ Analyse terminée ! Les graphiques ont été sauvegardés dans le dossier : '{dossier_sortie}'")

def make_pipeline_analyse_energy(dossier_racine, dossier_sortie="rapports"):
    print(f"Extraction des données depuis {dossier_racine}...")
    df_donnees = charger_donnees(dossier_racine)
    
    if not df_donnees.empty:
        print("Génération des graphiques en cours...")
        generer_rapports(df_donnees, dossier_sortie)
    else:
        print("Aucune donnée à analyser.")

if __name__ == "__main__":
    # Paramétrage de la ligne de commande
    parser = argparse.ArgumentParser(description="Analyse des métriques d'énergie par transformation et modulation.")
    parser.add_argument("dossier", type=str, help="Chemin vers le dossier contenant les sous-dossiers de signaux.")
    parser.add_argument("--sortie", type=str, default="rapport_visuel", help="Dossier où sauvegarder les images (défaut: rapport_visuel).")
    
    args = parser.parse_args()
    
    # Exécution
    print(f"Extraction des données depuis {args.dossier}...")
    df_donnees = charger_donnees(args.dossier)
    
    if not df_donnees.empty:
        print("Génération des graphiques en cours...")
        generer_rapports(df_donnees, args.sortie)
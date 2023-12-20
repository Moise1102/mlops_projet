import pandas as pd
import numpy as np 
import os
import sys
sys.path.append('..')

def extract(chemin_data: str, colonnes_a_retirer: list) -> pd.DataFrame:
    
    """
        Fonction qui récupère le fichier à traiter et retire les colonnes qui ne sont pas nécessaire pour l'étude 
        et retourne la base de données avec les colonnes supprimées.
        
        :param chemin_data: (str) Le chemin permettant d'accéder au fichier.
        
        :param colonnes_a_retirer: (list) Une liste des noms des colonnes à supprimer.
        
        :return: (pandas.core.frame.DataFrame) le Data sans les colonnes à retirer (qui ont été données en paramètres)
    """
    
    # Vérification - le chemin donné en paramètres existe
    
    if os.path.exists(chemin_data) == True:
        df = pd.read_csv(chemin_data)
        
        # Vérification - les colonnes données en paramètres existent
     
        verif_colonnes_presentes =  all(colonne in df.columns for colonne in colonnes_a_retirer)
        
        if verif_colonnes_presentes == True :
            df = df.drop(colonnes_a_retirer, axis=1)
            
            return df
     
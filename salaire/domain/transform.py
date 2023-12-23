import numpy as np
import pandas as pd
from salaire.infrastructure.extract import extract
from salaire.domain.decorators import log_return_shape
from sklearn.impute import KNNImputer
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

@log_return_shape
def redocadge (data: pd.DataFrame)  -> pd.DataFrame :
    """
    fonction qui fait toutes les transformations de features  ingineering
    
    Parameters:
    - data (DataFrame): Le DataFrame contenant les données.

    Returns:
    - DataFrame: Le DataFrame avec les colonnes transformées 
    
    """
    
    #
    data['SALAIRE'] = data['SALPRSFIN'].apply(lambda x: 1 if x > 2000 else 0)
    data=data.drop('SALPRSFIN',axis = 1)
    
    regroupement_mapping_numerique = {'02I': 1, '02T': 1,
                                  '03I': 2, '03T': 2, '04I': 2, '04T': 2, '05': 2,
                                  '06I': 3, '07I': 3, '06T': 3, '07T': 3,
                                  '09L': 4, '09M': 4, '10L': 4, '11L': 4,
                                  '12L': 4, '10M': 4, '11M': 4, '12M': 4,
                                  '13L': 5, '14L': 5, '13M': 5, '14M': 5,
                                  '15': 6, '16': 6,
                                  '17': 7, '18L': 7, '18M': 7,
                                  '01':8,
                                  '08':9,
                                  }

    # Remplacer les modalités dans la colonne 'PHD' avec les nouvelles valeurs de regroupement numériques
    data['PHD'] = data['PHD'].replace(regroupement_mapping_numerique)
    
    
    return data


@log_return_shape
def cleaned(data: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Supprime les valeurs aberrantes des variables spécifiées dans un DataFrame
    et remplace les valeurs manquantes en utilisant un KNNImputer.

    Parameters:
    - data (DataFrame): Le DataFrame contenant les données.
    - variables (list): La liste des noms de variables pour lesquelles supprimer les valeurs aberrantes.

    Returns:
    - DataFrame: Le DataFrame sans les valeurs aberrantes pour les variables spécifiées.
    """
    
    imputer = KNNImputer(n_neighbors=3)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data_imputed = data.dropna()
    
    data_cleaned = data_imputed.copy()

    for variable in variables:
        Q1 = data_cleaned[variable].quantile(0.25)
        Q3 = data_cleaned[variable].quantile(0.75)

        IQR = Q3 - Q1

        borne_inf = Q1 - 1.5 * IQR
        borne_sup = Q3 + 1.5 * IQR

        
        data_cleaned = data_cleaned[(data_cleaned[variable] >= borne_inf) & (data_cleaned[variable] <= borne_sup)]

    
    data_cleaned_no_missing = data_cleaned.dropna()
   
    return data_cleaned_no_missing



@log_return_shape
def calculate_cramer_v(data: pd.DataFrame, categorical_vars: list) -> pd.DataFrame:
    """
    Calcule le V de Cramer pour chaque paire de variables catégorielles dans le DataFrame.

    Parameters:
    - data (pd.DataFrame): Le DataFrame contenant les données.
    - categorical_vars (list): La liste des noms de variables catégorielles.

    Returns:
    - pd.DataFrame: Un DataFrame contenant les valeurs du V de Cramer pour chaque paire de variables.
    """
    
    contingency_tables = {}
    for var1 in categorical_vars:
        for var2 in categorical_vars:
            contingency_table = pd.crosstab(data[var1], data[var2])
            contingency_tables[(var1, var2)] = contingency_table

    
    cramer_v_values = {}
    for (var1, var2), contingency_table in contingency_tables.items():
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        cramer_v_values[(var1, var2)] = cramers_v

    
    cramer_df = pd.DataFrame(index=categorical_vars, columns=categorical_vars)
    for var1 in categorical_vars:
        for var2 in categorical_vars:
            cramer_df.loc[var1, var2] = cramer_v_values.get((var1, var2), cramer_v_values.get((var2, var1)))

    
    cramer_df = cramer_df.astype(float)

    return cramer_df


@log_return_shape
def plot_cramer_heatmap(cramer_df: pd.DataFrame):
    """
    Crée et affiche une heatmap du V de Cramer.

    Parameters:
    - cramer_df (pd.DataFrame): Le DataFrame contenant les valeurs du V de Cramer.
    """
    # Créez une heatmap avec seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(cramer_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("V de Cramer Heatmap")
    plt.show()

@log_return_shape
def etl(chemin_data: str, colonnes_a_retirer: list,variables: list) -> pd.DataFrame:

    """
    fonction qui prend un dataframe fait toute les transformation nécéssaire
    et la charge 

    Parameters:
    ---------------
    - chemin_data (str) : chemin vers le fichier
    
    - colonnes_a_retirer (list) : list des colonnes à retirer

    - variables (list) : variables pour laquelle ont veux calculer les données abbérantes 


    Return :
    Un DataFrame propre et recoder prêt à l'emploi

    """

    data=extract(chemin_data,colonnes_a_retirer)
    data= redocadge(data)
    data = cleaned(data,variables)

    return data

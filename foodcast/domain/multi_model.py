import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def model(data: pd.DataFrame):
    """
    Entraîne et évalue plusieurs modèles de classification sur le DataFrame d'entrée.

    Parameters:
    - data (pd.DataFrame): Le DataFrame contenant les données. Il doit inclure une colonne 'SALAIRE' comme variable cible.

    Returns:
    - pd.DataFrame: Un DataFrame contenant les résultats de l'évaluation des modèles.

    Cette fonction effectue les étapes suivantes :
    1. Divise les données d'entrée en ensembles d'entraînement et de test.
    2. Définit un ensemble de modèles de classification, y compris DecisionTree, KNeighbors, LogisticRegression et RandomForest.
    3. Spécifie des grilles d'hyperparamètres pour chaque modèle pour l'ajustement des hyperparamètres.
    4. Itère sur chaque modèle, créant un pipeline avec ou sans mise à l'échelle en fonction du type de modèle.
    5. Utilise GridSearchCV pour trouver les meilleurs hyperparamètres pour chaque modèle.
    6. Affiche et enregistre les meilleurs hyperparamètres, la précision, le score et le rapport de classification pour chaque modèle sur l'ensemble de test.

    Remarque : Cette fonction suppose que la variable cible est 'SALAIRE' dans le DataFrame d'entrée.

    Example:
    data = pd.read_csv('votre_data.csv')
    results = model(data)
    """

    y = data.SALAIRE
    X = data.drop('SALAIRE', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    models = {
        'DecisionTree': DecisionTreeClassifier(),
        'KNeighbors': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier()
    }

    param_grid = {
        'DecisionTree': {'model__criterion': ['gini', 'entropy'], 'model__max_depth': [None, 5, 10, 15]},
        'KNeighbors': {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']},
        'LogisticRegression': {'model__C': [0.1, 1, 10], 'model__penalty': ['l1', 'l2']},
        'RandomForest': {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 5, 10, 15],
                         'model__criterion': ['gini', 'entropy']}
    }

    for model_name, model in models.items():
        if 'KNeighbors' in model_name or 'LogisticRegression' in model_name:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('model', model)
            ])

        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        y_pred = grid_search.predict(X_test)
        print(f"Accuracy for {model_name}: {accuracy_score(y_test, y_pred)}")
        print(f"Score for {model_name}: {grid_search.score(X_test, y_test)}")
        print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
        print("=" * 80)

"""Pipeline ML : prédiction du churn client.

Pipeline complet :
CSV -> nettoyage -> train/test split -> preprocessing -> Random Forest
-> évaluation -> sauvegarde du pipeline complet
"""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET = "Exited"

COLONNES_A_SUPPRIMER = (
    "RowNumber",
    "CustomerId",
    "Surname",
    "Geography",
)

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_ESTIMATORS = 100


# ---------------------------------------------------------------------------
# Chargement et préparation des données
# ---------------------------------------------------------------------------

def prepare_data(
    csv_path: str | Path,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Charge et prépare les données pour l'entraînement.

    Le preprocessing des variables est volontairement effectué plus tard
    dans train_model(), afin d'éviter toute fuite de données entre train
    et test.

    Args:
        csv_path: Chemin vers le fichier CSV.
        test_size: Proportion du dataset réservée au test.
        random_state: Graine aléatoire.

    Returns:
        x_train, x_test, y_train, y_test

    Raises:
        FileNotFoundError: Si le fichier CSV n'existe pas.
        ValueError: Si les données sont invalides.
    """

    # -----------------------------------------------------------------------
    # Validation des paramètres
    # -----------------------------------------------------------------------

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size doit être strictement compris entre 0 et 1."
        )

    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Fichier CSV introuvable : {csv_path}"
        )

    # -----------------------------------------------------------------------
    # Lecture du CSV
    # -----------------------------------------------------------------------

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "Le fichier CSV est vide."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Impossible de lire correctement le fichier CSV : {csv_path}"
        ) from exc

    if df.empty:
        raise ValueError("Le dataset ne contient aucune ligne.")

    # -----------------------------------------------------------------------
    # Validation des colonnes
    # -----------------------------------------------------------------------

    if TARGET not in df.columns:
        raise ValueError(
            f"La colonne cible '{TARGET}' est absente du dataset."
        )

    if "Gender" not in df.columns:
        raise ValueError(
            "La colonne 'Gender' est absente du dataset."
        )

    # -----------------------------------------------------------------------
    # Suppression des doublons
    # -----------------------------------------------------------------------

    initial_rows = len(df)

    df = df.drop_duplicates(ignore_index=True)

    removed_duplicates = initial_rows - len(df)

    if removed_duplicates:
        print(
            f"{removed_duplicates} doublon(s) supprimé(s)."
        )

    # -----------------------------------------------------------------------
    # Nettoyage de la variable cible
    # -----------------------------------------------------------------------

    # Une ligne sans cible ne peut pas être utilisée pour l'entraînement.
    df = df.dropna(subset=[TARGET])

    if df.empty:
        raise ValueError(
            "Aucune donnée exploitable après suppression des cibles manquantes."
        )

    # -----------------------------------------------------------------------
    # Vérification de la cible
    # -----------------------------------------------------------------------

    unique_targets = df[TARGET].dropna().unique()

    if len(unique_targets) < 2:
        raise ValueError(
            "La variable cible doit contenir au moins deux classes."
        )

    # -----------------------------------------------------------------------
    # Suppression des colonnes inutiles
    # -----------------------------------------------------------------------

    df = df.drop(
        columns=COLONNES_A_SUPPRIMER,
        errors="ignore",
    )

    # -----------------------------------------------------------------------
    # Séparation X / y
    # -----------------------------------------------------------------------

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    if x.empty:
        raise ValueError(
            "Aucune variable explicative disponible."
        )

    # -----------------------------------------------------------------------
    # Train / Test split
    # -----------------------------------------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Dataset total : {len(df)} lignes")
    print(f"Train         : {len(x_train)} lignes")
    print(f"Test          : {len(x_test)} lignes")
    print(f"Variables     : {x.shape[1]}")

    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Construit et entraîne le pipeline complet.

    Le pipeline contient :
        - Imputation des variables numériques
        - Imputation + OneHotEncoder pour les variables catégorielles
        - Random Forest

    Le preprocessing est appris uniquement sur le train set.

    Args:
        x_train: Variables explicatives du train.
        y_train: Variable cible du train.
        n_estimators: Nombre d'arbres.
        random_state: Graine aléatoire.

    Returns:
        Pipeline entraîné.
    """

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    if x_train is None or x_train.empty:
        raise ValueError("x_train est vide.")

    if y_train is None or y_train.empty:
        raise ValueError("y_train est vide.")

    if len(x_train) != len(y_train):
        raise ValueError(
            "x_train et y_train doivent avoir le même nombre de lignes."
        )

    if n_estimators <= 0:
        raise ValueError(
            "n_estimators doit être supérieur à 0."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "y_train doit contenir au moins deux classes."
        )

    # -----------------------------------------------------------------------
    # Identification des colonnes
    # -----------------------------------------------------------------------

    categorical_columns = x_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_columns = x_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    if not numeric_columns and not categorical_columns:
        raise ValueError(
            "Aucune colonne numérique ou catégorielle exploitable."
        )

    # -----------------------------------------------------------------------
    # Preprocessing numérique
    # -----------------------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Preprocessing catégoriel
    # -----------------------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Preprocessor global
    # -----------------------------------------------------------------------

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # -----------------------------------------------------------------------
    # Random Forest
    # -----------------------------------------------------------------------

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_features="sqrt",
    )

    # -----------------------------------------------------------------------
    # Pipeline final
    # -----------------------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    # -----------------------------------------------------------------------
    # Entraînement
    # -----------------------------------------------------------------------

    print(
        f"Entraînement du Random Forest "
        f"avec {n_estimators} arbres..."
    )

    model.fit(x_train, y_train)

    print("Entraînement terminé.")

    return model


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """
    Évalue le modèle sur le jeu de test.

    Returns:
        Accuracy du modèle.
    """

    if model is None:
        raise ValueError("Le modèle est None.")

    if x_test is None or x_test.empty:
        raise ValueError("x_test est vide.")

    if y_test is None or y_test.empty:
        raise ValueError("y_test est vide.")

    if len(x_test) != len(y_test):
        raise ValueError(
            "x_test et y_test doivent avoir le même nombre de lignes."
        )

    # -----------------------------------------------------------------------
    # Prédiction
    # -----------------------------------------------------------------------

    y_pred = model.predict(x_test)

    # -----------------------------------------------------------------------
    # Accuracy
    # -----------------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    print(
        f"\nAccuracy : {accuracy * 100:.2f} %"
    )

    # -----------------------------------------------------------------------
    # Matrice de confusion
    # -----------------------------------------------------------------------

    print("\nMatrice de confusion :")

    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    # -----------------------------------------------------------------------
    # Classification report
    # -----------------------------------------------------------------------

    print("\nRapport de classification :")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    return accuracy


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_model(
    model: Pipeline,
    path: str | Path = "classifier.joblib",
) -> None:
    """
    Sauvegarde le pipeline complet.

    Le fichier contient à la fois :
        - le preprocessing
        - l'encodeur
        - le modèle Random Forest
    """

    if model is None:
        raise ValueError(
            "Impossible de sauvegarder un modèle None."
        )

    output_path = Path(path)

    # Crée le dossier automatiquement s'il n'existe pas.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
        compress=3,
    )

    print(
        f"Modèle sauvegardé dans : {output_path}"
    )


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_model(
    path: str | Path = "classifier.joblib",
) -> Pipeline:
    """
    Charge un pipeline sauvegardé.
    """

    model_path = Path(path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(
            f"Impossible de charger le modèle : {model_path}"
        ) from exc

    if not hasattr(model, "predict"):
        raise ValueError(
            "Le fichier chargé ne contient pas un modèle compatible."
        )

    print(
        f"Modèle chargé depuis : {model_path}"
    )

    return model

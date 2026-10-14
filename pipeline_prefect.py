"""Pipeline Prefect du projet de prédiction du churn (Atelier 3).

Utilise mainO.py et model_pipelineO.py SANS les modifier :
le formatage et la qualité sont vérifiés et signalés, jamais réécrits.

Usage :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow code
    python pipeline_prefect.py --flow train        (alias : entrainement)
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow install
"""

import argparse
import os
import subprocess  # nosec B404 - commandes fixes et connues uniquement
import sys

from prefect import flow, get_run_logger, task

from model_pipelineO import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

# Mêmes valeurs par défaut que dans mainO.py
DEFAULT_CSV_PATH = "Churn_Modelling.csv"
DEFAULT_MODEL_PATH = "classifier.joblib"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100

# Étapes qualité / sécurité : uniquement les fichiers du projet
PROJECT_FILES = ["model_pipelineO.py", "mainO.py", "pipeline_prefect.py"]

# Étape 06 : dépôt git distant (à remplacer par votre URL)
REPO_URL = os.getenv("REPO_URL", "https://github.com/firasayari10/ml-project-prefect.git")
REPO_DIR = os.getenv("REPO_DIR", ".")


def run(cmd, blocking=True):
    """Exécute une commande et envoie la sortie dans les logs Prefect.

    blocking=True  : la tâche échoue si la commande échoue.
    blocking=False : un échec est seulement signalé (warning).
    """
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(  # nosec B603 - pas de shell, arguments fixes
        cmd, capture_output=True, text=True, check=False
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        if output:
            logger.info(output)
        return output
    if blocking:
        logger.error(output)
        raise RuntimeError(f"Échec ({result.returncode}) : {' '.join(cmd)}")
    logger.warning("Problèmes détectés (non bloquant) :\n%s", output)
    return output


# --------------------------------------------------------------------------
# Étape 06 : récupération du projet
# --------------------------------------------------------------------------
@task(name="Récupération du projet (git)")
def git_pull():
    """Clone le dépôt s'il n'existe pas, sinon fait un pull."""
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        run(["git", "-C", REPO_DIR, "pull"])
    else:
        run(["git", "clone", REPO_URL, REPO_DIR])


# --------------------------------------------------------------------------
# Dépendances et suivi du code (étape 04)
# --------------------------------------------------------------------------
@task(name="Installer les dépendances", retries=1)
def install_dependencies():
    """Installe les paquets de requirements.txt."""
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    """Vérifie le formatage avec black (--check : aucun fichier modifié)."""
    run(
        [sys.executable, "-m", "black", "--check", "--diff", *PROJECT_FILES],
        blocking=False,
    )


@task(name="Qualité du code")
def code_quality():
    """Analyse la qualité du code avec flake8."""
    run(
        [sys.executable, "-m", "flake8", "--max-line-length=100", *PROJECT_FILES],
        blocking=False,
    )


@task(name="Sécurité du code")
def code_security():
    """Analyse de sécurité avec bandit (sévérité moyenne et haute)."""
    run([sys.executable, "-m", "bandit", "-ll", *PROJECT_FILES])


@task(name="Tests unitaires")
def unit_tests():
    """Lance les tests unitaires du dossier tests/."""
    run([sys.executable, "-m", "pytest", "tests", "-v"])


# --------------------------------------------------------------------------
# Données et modèle
# --------------------------------------------------------------------------
@task(name="Préparation des données")
def prepare_data_task(data_path=DEFAULT_CSV_PATH):
    """Retourne x_train, x_test, y_train, y_test."""
    return prepare_data(
        data_path,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )


@task(name="Entrainement du modèle")
def train_model_task(x_train, y_train):
    """Entraîne le pipeline Random Forest."""
    return train_model(
        x_train,
        y_train,
        n_estimators=DEFAULT_N_ESTIMATORS,
        random_state=DEFAULT_RANDOM_STATE,
    )


@task(name="Sauvegarder le modèle")
def save_model_task(model, model_path=DEFAULT_MODEL_PATH):
    """Sauvegarde le modèle avec joblib."""
    save_model(model, model_path)


@task(name="Charger le modèle")
def load_model_task(model_path=DEFAULT_MODEL_PATH):
    """Charge le modèle sauvegardé."""
    return load_model(model_path)


@task(name="Evaluer le modèle")
def evaluate_model_task(model, x_test, y_test):
    """Évalue le modèle et retourne l'accuracy."""
    accuracy = evaluate_model(model, x_test, y_test)
    get_run_logger().info("Accuracy : %.2f %%", accuracy * 100)
    return accuracy


# --------------------------------------------------------------------------
# Flows
# --------------------------------------------------------------------------
@flow(name="install")
def install_flow():
    """Installation des dépendances."""
    install_dependencies()


@flow(name="code")
def code_flow():
    """Formatage, qualité, sécurité et tests."""
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train")
def train_flow(data_path: str = DEFAULT_CSV_PATH, model_path: str = DEFAULT_MODEL_PATH):
    """Préparation des données + entraînement + sauvegarde."""
    x_train, _, y_train, _ = prepare_data_task(data_path)
    model = train_model_task(x_train, y_train)
    save_model_task(model, model_path)


@flow(name="evaluate")
def evaluate_flow(
    data_path: str = DEFAULT_CSV_PATH, model_path: str = DEFAULT_MODEL_PATH
):
    """Chargement du modèle + évaluation."""
    _, x_test, _, y_test = prepare_data_task(data_path)
    model = load_model_task(model_path)
    return evaluate_model_task(model, x_test, y_test)


@flow(name="all")
def all_flow(
    data_path: str = DEFAULT_CSV_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    pull_repo: bool = False,
    install: bool = True,
):
    """Pipeline complet : (git) -> dépendances -> code -> données -> modèle."""
    if pull_repo:
        git_pull()
    if install:
        install_dependencies()
    format_code()
    code_quality()
    code_security()
    unit_tests()
    x_train, x_test, y_train, y_test = prepare_data_task(data_path)
    model = train_model_task(x_train, y_train)
    save_model_task(model, model_path)
    return evaluate_model_task(model, x_test, y_test)


FLOWS = {
    "all": all_flow,
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline MLOps avec Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()

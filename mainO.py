"""Point d'entrée du pipeline ML de prédiction du churn."""

import argparse
import sys
from pathlib import Path

from model_pipelineO import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


# ---------------------------------------------------------------------------
# Configuration par défaut
# ---------------------------------------------------------------------------

DEFAULT_CSV_PATH = "Churn_Modelling.csv"
DEFAULT_MODEL_PATH = "classifier.joblib"

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100


# ---------------------------------------------------------------------------
# Arguments CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    """Crée et configure le parser de ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline ML de prédiction du churn client "
            "avec Random Forest."
        )
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Entraîner et sauvegarder un nouveau modèle.",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Évaluer le modèle sauvegardé.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=(
            f"Chemin vers le fichier CSV "
            f"(défaut : {DEFAULT_CSV_PATH})."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=(
            f"Chemin vers le modèle "
            f"(défaut : {DEFAULT_MODEL_PATH})."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=(
            f"Proportion du dataset utilisée pour le test "
            f"(défaut : {DEFAULT_TEST_SIZE})."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=(
            f"Graine aléatoire "
            f"(défaut : {DEFAULT_RANDOM_STATE})."
        ),
    )

    parser.add_argument(
        "--estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help=(
            f"Nombre d'arbres du Random Forest "
            f"(défaut : {DEFAULT_N_ESTIMATORS})."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    model_path: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
) -> None:
    """Prépare les données, entraîne et sauvegarde le modèle."""

    print("=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)

    print(f"\nDataset : {Path(data_path).resolve()}")
    print(f"Modèle  : {Path(model_path).resolve()}")

    # -----------------------------------------------------------------------
    # Préparation
    # -----------------------------------------------------------------------

    print("\n[1/3] Préparation des données...")

    x_train, x_test, y_train, y_test = prepare_data(
        data_path,
        test_size=test_size,
        random_state=random_state,
    )

    # -----------------------------------------------------------------------
    # Entraînement
    # -----------------------------------------------------------------------

    print("\n[2/3] Entraînement du modèle...")

    model = train_model(
        x_train,
        y_train,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    # -----------------------------------------------------------------------
    # Sauvegarde
    # -----------------------------------------------------------------------

    print("\n[3/3] Sauvegarde du modèle...")

    save_model(
        model,
        model_path,
    )

    # -----------------------------------------------------------------------
    # Évaluation
    # -----------------------------------------------------------------------

    print("\nÉvaluation sur le jeu de test...")

    evaluate_model(
        model,
        x_test,
        y_test,
    )

    print("\nEntraînement terminé.")


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate(
    data_path: str,
    model_path: str,
    test_size: float,
    random_state: int,
) -> None:
    """Charge un modèle existant et l'évalue."""

    print("=" * 60)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 60)

    print(f"\nDataset : {Path(data_path).resolve()}")
    print(f"Modèle  : {Path(model_path).resolve()}")

    # -----------------------------------------------------------------------
    # Préparation du jeu de test
    # -----------------------------------------------------------------------

    print("\n[1/2] Préparation des données...")

    _, x_test, _, y_test = prepare_data(
        data_path,
        test_size=test_size,
        random_state=random_state,
    )

    # -----------------------------------------------------------------------
    # Chargement
    # -----------------------------------------------------------------------

    print("\n[2/2] Chargement du modèle...")

    model = load_model(model_path)

    # -----------------------------------------------------------------------
    # Évaluation
    # -----------------------------------------------------------------------

    print("\nÉvaluation...")

    evaluate_model(
        model,
        x_test,
        y_test,
    )

    print("\nÉvaluation terminée.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Point d'entrée principal."""

    parser = create_parser()
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Vérification des arguments
    # -----------------------------------------------------------------------

    if not args.train and not args.evaluate:
        parser.print_help()
        return 0

    # -----------------------------------------------------------------------
    # Validation test-size
    # -----------------------------------------------------------------------

    if not 0 < args.test_size < 1:
        parser.error(
            "--test-size doit être strictement compris entre 0 et 1."
        )

    # -----------------------------------------------------------------------
    # Validation estimators
    # -----------------------------------------------------------------------

    if args.estimators <= 0:
        parser.error(
            "--estimators doit être supérieur à 0."
        )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------

    try:
        if args.train:
            train(
                data_path=args.data,
                model_path=args.model,
                test_size=args.test_size,
                random_state=args.random_state,
                n_estimators=args.estimators,
            )

        # -------------------------------------------------------------------
        # Evaluate uniquement si --evaluate est fourni sans --train.
        #
        # Si les deux sont présents, train() a déjà évalué le nouveau modèle.
        # On évite donc une seconde préparation + évaluation inutile.
        # -------------------------------------------------------------------

        elif args.evaluate:
            evaluate(
                data_path=args.data,
                model_path=args.model,
                test_size=args.test_size,
                random_state=args.random_state,
            )

    except (FileNotFoundError, ValueError) as exc:
        print(
            f"\nErreur : {exc}",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        print(
            f"\nErreur inattendue : {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())

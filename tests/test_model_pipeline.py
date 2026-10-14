"""Tests unitaires de model_pipelineO.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_pipelineO import (  # noqa: E402
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


@pytest.fixture
def csv_file(tmp_path):
    """Petit dataset churn au même format que Churn_Modelling.csv."""
    n = 100
    df = pd.DataFrame(
        {
            "RowNumber": range(n),
            "CustomerId": range(1000, 1000 + n),
            "Surname": ["X"] * n,
            "CreditScore": [600 + i for i in range(n)],
            "Geography": ["France", "Spain"] * (n // 2),
            "Gender": ["Male", "Female"] * (n // 2),
            "Age": [20 + i % 50 for i in range(n)],
            "Tenure": [i % 10 for i in range(n)],
            "Balance": [1000.0 * i for i in range(n)],
            "NumOfProducts": [1 + i % 3 for i in range(n)],
            "HasCrCard": [i % 2 for i in range(n)],
            "IsActiveMember": [(i + 1) % 2 for i in range(n)],
            "EstimatedSalary": [50000.0 + i for i in range(n)],
            "Exited": [1 if i % 4 == 0 else 0 for i in range(n)],
        }
    )
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return path


def test_prepare_data_split(csv_file):
    x_train, x_test, y_train, y_test = prepare_data(csv_file, test_size=0.2)
    assert len(x_train) == 80 and len(x_test) == 20
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert "Exited" not in x_train.columns


def test_prepare_data_missing_file():
    with pytest.raises(FileNotFoundError):
        prepare_data("fichier_inexistant.csv")


def test_train_save_load_evaluate(csv_file, tmp_path):
    x_train, x_test, y_train, y_test = prepare_data(csv_file)
    model = train_model(x_train, y_train, n_estimators=10)
    model_path = tmp_path / "model.joblib"
    save_model(model, model_path)
    loaded = load_model(model_path)
    accuracy = evaluate_model(loaded, x_test, y_test)
    assert 0.0 <= accuracy <= 1.0


def test_train_model_invalid_estimators(csv_file):
    x_train, _, y_train, _ = prepare_data(csv_file)
    with pytest.raises(ValueError):
        train_model(x_train, y_train, n_estimators=0)

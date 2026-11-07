
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import load_data
from features import separate_features_target, create_preprocessor


def test_separate_features_target():
    df = load_data()

    X, y = separate_features_target(df)

    assert X.shape == (7043, 19)
    assert len(y) == 7043
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns


def test_target_encoding():
    df = load_data()

    X, y = separate_features_target(df)

    assert set(y.unique()) == {0, 1}


def test_preprocessor_creation():
    df = load_data()
    X, y = separate_features_target(df)

    preprocessor = create_preprocessor(X)

    assert preprocessor is not None


def test_preprocessing_output():
    df = load_data()
    X, y = separate_features_target(df)

    preprocessor = create_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == 7043
    assert X_transformed.shape[1] == 45


def test_no_missing_values_after_preprocessing():
    df = load_data()
    X, y = separate_features_target(df)

    preprocessor = create_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    import pandas as pd

    assert pd.isna(X_transformed).sum() == 0


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import load_data


def test_load_data():
    df = load_data()

    assert df is not None
    assert not df.empty


def test_data_shape():
    df = load_data()

    assert df.shape == (7043, 21)


def test_required_columns():
    df = load_data()

    required_columns = [
        "customerID",
        "gender",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    ]

    for column in required_columns:
        assert column in df.columns


def test_customer_id_uniqueness():
    df = load_data()

    assert df["customerID"].is_unique


def test_target_values():
    df = load_data()

    assert set(df["Churn"].unique()) == {"Yes", "No"}

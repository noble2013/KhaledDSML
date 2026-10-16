import pandas as pd
import pytest
from khaled_dsml import sqlite_import_frame, sqlite_group_summary


def test_sqlite_pipeline(tmp_path):
    df = pd.DataFrame({"wilaya": ["Alger", "Alger", "Oran"],
                       "montant_da": [10., 20., 15.]})
    path = tmp_path / "example.db"
    assert sqlite_import_frame(df, path) == 3
    result = sqlite_group_summary(path).set_index("category")
    assert result.loc["Alger", "total"] == 30
    assert result.loc["Oran", "mean"] == 15


def test_sqlite_invalid_name(tmp_path):
    with pytest.raises(ValueError):
        sqlite_import_frame(pd.DataFrame({"x": [1]}), tmp_path / "a.db", table="orders;DROP TABLE")

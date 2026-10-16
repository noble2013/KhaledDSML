import numpy as np
import pandas as pd
import pytest
from khaled_dsml import (generate_transactions, csv_chunk_summary,
    csv_chunk_monthly, csv_chunk_missing, map_reduce_word_count,
    json_documents_to_table)


def test_generate_reproducibly():
    a, b = generate_transactions(100, seed=20), generate_transactions(100, seed=20)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 100 and "montant_da" in a


def test_invalid_generation():
    with pytest.raises(ValueError):
        generate_transactions(-1)


def test_chunked_aggregation_matches_pandas(tmp_path):
    df = generate_transactions(1055, missing_rate=.07)
    file = tmp_path / "transactions.csv"
    df.to_csv(file, index=False)
    chunk = csv_chunk_summary(file, chunksize=53)
    expected = df.groupby("wilaya")["montant_da"].agg(["count", "sum", "mean"])
    for column in ["count", "sum", "mean"]:
        pd.testing.assert_series_equal(chunk[column].sort_index(), expected[column].sort_index(), check_names=False,
                                       atol=1e-8, rtol=1e-8, check_dtype=False)
    assert chunk["sum"].sum() == pytest.approx(df["montant_da"].sum(), rel=1e-10)


def test_chunk_monthly(tmp_path):
    df = generate_transactions(1077)
    file = tmp_path / "transactions.csv"
    df.to_csv(file, index=False)
    actual = csv_chunk_monthly(file, chunksize=100)
    expected = df.groupby(df.date.dt.to_period("M"))["montant_da"].sum()
    for month, val in actual.items():
        assert val == pytest.approx(expected[month], rel=1e-12, abs=1e-6)


def test_chunk_missing(tmp_path):
    df = generate_transactions(501, missing_rate=.13)
    file = tmp_path / "tx.csv"
    df.to_csv(file, index=False)
    report = csv_chunk_missing(file, chunksize=41)
    assert report.loc["montant_da", "count"] == int(df.montant_da.isna().sum())


def test_single_group_chunk(tmp_path):
    df = pd.DataFrame({"montant_da": [1, 2, None, 4]})
    path = tmp_path / "only.csv"
    df.to_csv(path, index=False)
    r = csv_chunk_summary(path, group_by=None, chunksize=2)
    assert r.loc["ALL", "count"] == 3 and r.loc["ALL", "sum"] == 7


def test_mapreduce_words():
    assert map_reduce_word_count(["البيانات البيانات Python", "Python البيانات"])["البيانات"] == 3


def test_nosql_flattening():
    r = json_documents_to_table([{"client": {"name": "Ali"}, "amount": 12}])
    assert r.loc[0, "client.name"] == "Ali"

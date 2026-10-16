"""Memory-aware big-data teaching tools (local, chunked, non-distributed)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd


def generate_transactions(n_rows=10_000, seed=42, missing_rate=.02) -> pd.DataFrame:
    """Reproducible synthetic Algerian transactions, amounts in Algerian dinars."""
    if n_rows < 1 or not 0 <= missing_rate < 1:
        raise ValueError("Require n_rows >= 1 and 0 <= missing_rate < 1")
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-01-01")
    df = pd.DataFrame({
        "id_client": rng.integers(1, 50_000, n_rows),
        "wilaya": rng.choice(["Alger", "Oran", "Setif", "Constantine", "Annaba", "Batna"], n_rows),
        "categorie": rng.choice(["Alimentation", "Electronique", "Vetements", "Transport"], n_rows),
        "montant_da": np.round(rng.lognormal(7.5, .9, n_rows), 2),
        "date": base + pd.to_timedelta(rng.integers(0, 365*86400, n_rows), unit="s"),
    })
    miss = rng.random(n_rows) < missing_rate
    df.loc[miss, "montant_da"] = np.nan
    return df


def csv_chunk_summary(path: str | Path, *, amount_column="montant_da",
                      group_by="wilaya", chunksize=100_000) -> pd.DataFrame:
    """Streaming count/sum/mean by category; never loads the whole CSV."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    usecols = [amount_column] if group_by is None else [group_by, amount_column]
    sums: dict = {}
    counts: dict = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk[amount_column] = pd.to_numeric(chunk[amount_column], errors="coerce")
        if group_by is None:
            pairs = [("ALL", (int(chunk[amount_column].count()),
                               float(chunk[amount_column].sum())))]
        else:
            group = chunk.groupby(group_by, dropna=False)[amount_column].agg(["count", "sum"])
            pairs = [(k, (int(row["count"]), float(row["sum"]))) for k, row in group.iterrows()]
        for key, (count, total) in pairs:
            # Use a stable string key also for missing group values.
            k = "<MISSING>" if pd.isna(key) else str(key)
            sums[k] = sums.get(k, 0.0) + total
            counts[k] = counts.get(k, 0) + count
    result = pd.DataFrame({"group": list(sums),
                           "count": [counts[k] for k in sums],
                           "sum": [sums[k] for k in sums]})
    if result.empty:
        return pd.DataFrame(columns=["count", "sum", "mean"],
                            index=pd.Index([], name=group_by or "group"))
    result["mean"] = result["sum"] / result["count"].replace(0, np.nan)
    return result.set_index("group").rename_axis(group_by or "group").sort_values("sum", ascending=False)


def csv_chunk_missing(path: str | Path, *, chunksize=100_000) -> pd.DataFrame:
    """Count missing observations over all CSV chunks."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    total = 0
    missing = None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        total += len(chunk)
        counts = chunk.isna().sum()
        missing = counts if missing is None else missing.add(counts, fill_value=0)
    if missing is None:
        return pd.DataFrame(columns=["count", "percent"])
    return pd.DataFrame({"count": missing.astype(int), "percent": missing / total * 100})


def csv_chunk_monthly(path: str | Path, *, date_column="date",
                      amount_column="montant_da", chunksize=100_000) -> pd.Series:
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    sums = {}
    for chunk in pd.read_csv(path, usecols=[date_column, amount_column], chunksize=chunksize):
        dates = pd.to_datetime(chunk[date_column], errors="coerce")
        amounts = pd.to_numeric(chunk[amount_column], errors="coerce")
        grouped = amounts.groupby(dates.dt.to_period("M")).sum()
        for period, value in grouped.items():
            if pd.notna(period):
                sums[period] = sums.get(period, 0.0) + float(value)
    result = pd.Series(sums, dtype=float, name="total_da").sort_index()
    result.index.name = "month"
    return result


def map_reduce_word_count(documents, *, min_length=2) -> dict[str, int]:
    """Small local MapReduce illustration, not an Apache Spark implementation."""
    if min_length < 1:
        raise ValueError("min_length must be >= 1")
    mapped = (Counter(w.lower() for w in re.findall(r"\b\w+\b", str(doc), flags=re.UNICODE)
                      if len(w) >= min_length) for doc in documents)
    reduced: Counter = Counter()
    for part in mapped:
        reduced.update(part)
    return dict(reduced.most_common())


def json_documents_to_table(documents: list[dict]) -> pd.DataFrame:
    """Flatten document-style JSON as an introduction to NoSQL data."""
    return pd.json_normalize(documents)

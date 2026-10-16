"""SQLite demonstrations for TD2: SQL versus document-style data."""
from __future__ import annotations
import re
import sqlite3
from pathlib import Path
import pandas as pd


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("SQL identifier must be ASCII letters, digits or underscores")
    return '"' + name + '"'


def sqlite_import_frame(df: pd.DataFrame, database: str | Path,
                        table="transactions", if_exists="replace") -> int:
    """Store a pandas DataFrame in a local SQLite database."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    _safe_name(table)
    if if_exists not in {"replace", "append", "fail"}:
        raise ValueError("if_exists must be replace, append or fail")
    with sqlite3.connect(str(database)) as conn:
        count = df.to_sql(table, conn, if_exists=if_exists, index=False)
    return int(count)


def sqlite_group_summary(database: str | Path, *, table="transactions",
                         group_by="wilaya", amount_column="montant_da") -> pd.DataFrame:
    """Return count/sum/mean, computed by SQLite using a safe fixed SQL pattern."""
    t, g, a = _safe_name(table), _safe_name(group_by), _safe_name(amount_column)
    sql = (f"SELECT {g} AS category, COUNT({a}) AS count, SUM({a}) AS total, "
           f"AVG({a}) AS mean FROM {t} GROUP BY {g} ORDER BY total DESC")
    with sqlite3.connect(str(database)) as conn:
        return pd.read_sql_query(sql, conn)

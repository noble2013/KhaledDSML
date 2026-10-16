"""Big Data TD1: million-row dataset, missingness, memory and CSV streaming."""
from pathlib import Path
from khaled_dsml import (generate_transactions, quality_report, optimize_memory,
                         csv_chunk_summary, csv_chunk_missing, csv_chunk_monthly)

rows = 1_000_000  # Reduce to 10_000 if memory is limited
file = Path("transactions.csv")
df = generate_transactions(rows)
print("before", quality_report(df))
optimized = optimize_memory(df)
print("after", quality_report(optimized))
df.to_csv(file, index=False)
print("summary", csv_chunk_summary(file, chunksize=200_000))
print("monthly", csv_chunk_monthly(file, chunksize=200_000))
print("missing", csv_chunk_missing(file, chunksize=200_000))

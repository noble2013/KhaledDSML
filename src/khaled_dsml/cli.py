"""Command line interface: demo, inspect, and quiz."""
from __future__ import annotations

import argparse
import json
from .bigdata import generate_transactions
from .quality import quality_report
from .quizzes import get_questions


def main(argv=None):
    parser = argparse.ArgumentParser(prog="khaled-dsml", description="KhaledDSML educational toolkit")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a synthetic dataset example")
    demo.add_argument("--rows", type=int, default=1000)
    inspect = commands.add_parser("inspect", help="Inspect a CSV file (loads it into RAM)")
    inspect.add_argument("csv")
    quiz = commands.add_parser("quiz", help="Print questions without exposing the solutions")
    quiz.add_argument("--course", choices=["all", "data_science", "machine_learning", "big_data"], default="all")
    quiz.add_argument("--count", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "demo":
        df = generate_transactions(n_rows=args.rows)
        print(json.dumps(quality_report(df), ensure_ascii=False, indent=2))
    elif args.command == "inspect":
        import pandas as pd
        print(json.dumps(quality_report(pd.read_csv(args.csv)), ensure_ascii=False, indent=2))
    else:
        for q in get_questions(args.course, args.count):
            print(f"\n{q.id}: {q.question}")
            for label, choice in zip("ABCD", q.choices):
                print(f"  {label}. {choice}")


if __name__ == "__main__":
    main()

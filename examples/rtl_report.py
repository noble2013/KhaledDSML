"""Export one or more hypothesis tests as a clean offline Arabic report.

Run after installing the development branch:
    python examples/rtl_report.py
"""
from khaled_dsml import (
    normality_test,
    pearson_test,
    save_test_report,
)


def main() -> None:
    tests = [
        normality_test([12, 13, 15, 17, 16, 14, 13, 18]),
        pearson_test([1, 2, 3, 4, 5], [10, 11, 12, 15, 16]),
    ]
    filename = save_test_report(
        tests, "rtl_statistical_report.html", title="تقرير إحصائي — علم البيانات"
    )
    print("Report saved to:", filename.resolve())


if __name__ == "__main__":
    main()

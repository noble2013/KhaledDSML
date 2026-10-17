"""Safe, offline Arabic RTL HTML reports for statistical hypothesis tests.

The generated document needs no JavaScript, CDN, or internet connection.
All user-visible text from results and titles is HTML-escaped.
"""
from __future__ import annotations

from collections.abc import Iterable
from html import escape
from math import isfinite
from pathlib import Path

from ._results import TestResult


def _normalize_results(results: TestResult | Iterable[TestResult]) -> list[TestResult]:
    if isinstance(results, TestResult):
        return [results]
    if isinstance(results, (str, bytes)):
        raise TypeError("results must be a TestResult or an iterable of TestResult")
    try:
        items = list(results)
    except TypeError as exc:
        raise TypeError("results must be a TestResult or an iterable of TestResult") from exc
    if not items:
        raise ValueError("Provide at least one statistical test result")
    if not all(isinstance(item, TestResult) for item in items):
        raise TypeError("Every result must be a TestResult from KhaledDSML")
    return items


def _number(value: float, digits: int) -> str:
    number = float(value)
    if not isfinite(number):
        return str(number)
    if 0 < abs(number) < 10 ** (-digits):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def _p_value(value: float, digits: int) -> str:
    p = float(value)
    threshold = 10 ** (-digits)
    if p < threshold:
        return f"p &lt; {threshold:.{digits}f}"
    return f"p = {p:.{digits}f}"


def render_test_report(
    results: TestResult | Iterable[TestResult], *,
    title: str = "تقرير الاختبارات الإحصائية",
    digits: int = 4,
) -> str:
    """Return a complete UTF-8 HTML document with RTL Arabic explanations.

    Accepts a single ``TestResult`` or several test results. It never interprets
    a p-value as proof that the null hypothesis is true. The formatting of very
    small p-values avoids displaying an incorrect literal zero.
    """
    if not isinstance(digits, int) or isinstance(digits, bool) or not 1 <= digits <= 10:
        raise ValueError("digits must be an integer between 1 and 10")
    items = _normalize_results(results)
    rows = []
    for item in items:
        decision = "رفض الفرضية الصفرية" if item.reject_null else "عدم رفض الفرضية الصفرية"
        note = f'<p class="note">{escape(str(item.notes))}</p>' if item.notes else ""
        rows.append(
            "<article class=\"result\">"
            f"<h2>{escape(str(item.test))}</h2>"
            f"<p><strong>الفرضية الصفرية:</strong> {escape(str(item.null_hypothesis_ar))}</p>"
            f"<p><strong>حجم العينة:</strong> <bdi dir=\"ltr\">{int(item.sample_size)}</bdi></p>"
            f"<p><strong>إحصائية الاختبار:</strong> <bdi dir=\"ltr\">{_number(item.statistic, digits)}</bdi></p>"
            f"<p><strong>مستوى الدلالة:</strong> <bdi dir=\"ltr\">{_number(item.alpha, digits)}</bdi></p>"
            f"<p><strong>القيمة الاحتمالية:</strong> <bdi dir=\"ltr\">{_p_value(item.p_value, digits)}</bdi></p>"
            f"<p><strong>القرار:</strong> {decision}</p>"
            f"<p class=\"interpretation\">{escape(str(item.interpretation_ar))}</p>"
            '<div class="english" dir="ltr" lang="en">'
            f'<p><strong>Null hypothesis:</strong> {escape(str(item.null_hypothesis_en))}</p>'
            f'<p><strong>Interpretation:</strong> {escape(str(item.interpretation_en))}</p>'
            f'{("<p>" + escape(str(item.notes_en)) + "</p>") if item.notes_en else ""}'
            '</div>'
            f"{note}</article>"
        )
    heading = escape(str(title))
    cards = "\n".join(rows)
    return (
        "<!doctype html>\n"
        '<html lang="ar" dir="rtl">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{heading}</title>\n"
        "<style>\n"
        ".khaled-dsml-report{font-family:Tahoma,Arial,sans-serif;font-size:19px;font-weight:600;direction:rtl;text-align:right;"
        "max-width:880px;margin:2rem auto;padding:0 1rem;color:#25243a;"
        "background:#fff;line-height:1.9}\n"
        ".khaled-dsml-report h1{font-size:1.5rem}.khaled-dsml-report h2{font-size:1.16rem;margin:0 0 .75rem}\n"
        ".khaled-dsml-report .result{border:1px solid #bfc4cf;border-radius:10px;"
        "padding:1.2rem;margin:1rem 0;break-inside:avoid}\n"
        ".khaled-dsml-report .result p{margin:.35rem 0}.khaled-dsml-report .interpretation{font-weight:600}"
        ".khaled-dsml-report .note{font-size:.93rem;color:#555}\n"
        ".khaled-dsml-report .english{direction:ltr;text-align:left;font-size:19px;font-weight:600;border-top:1px solid #ddd;margin-top:.8rem;padding-top:.5rem}\n"
        ".khaled-dsml-report bdi{font-family:Consolas,monospace;unicode-bidi:isolate}\n"
        "@media print{.khaled-dsml-report{margin:0;max-width:none}.khaled-dsml-report .result{border:1px solid #aaa}}\n"
        "</style>\n</head>\n<body>\n" 
        '<div class="khaled-dsml-report" dir="rtl" lang="ar">\n'
        f"<h1>{heading}</h1>\n"
        '<p>النتائج إحصائية وليست إثباتًا للسببية. عدم رفض الفرضية الصفرية لا يثبت صحتها.</p>\n'
        f"{cards}\n</div>\n</body>\n</html>\n"
    )


def save_test_report(
    results: TestResult | Iterable[TestResult],
    path: str | Path,
    *,
    title: str = "تقرير الاختبارات الإحصائية",
    digits: int = 4,
) -> Path:
    """Save an offline RTL HTML report encoded in UTF-8 and return its path."""
    destination = Path(path)
    destination.write_text(
        render_test_report(results, title=title, digits=digits),
        encoding="utf-8",
    )
    return destination


def display_test_report(
    results: TestResult | Iterable[TestResult], *,
    title: str = "تقرير الاختبارات الإحصائية",
    digits: int = 4,
) -> None:
    """Display an RTL HTML report in Jupyter or Google Colab.

    IPython is intentionally an optional dependency for non-notebook users.
    """
    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise ImportError(
            "display_test_report needs IPython; install khaled-dsml[notebooks]"
        ) from exc
    display(HTML(render_test_report(results, title=title, digits=digits)))

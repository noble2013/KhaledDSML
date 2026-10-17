"""Bold Arabic RTL / English LTR interpretations for Google Colab and Jupyter.

The renderer is intentionally plain like a notebook output: no cards, colored
borders, remote fonts, JavaScript, or notebook-wide styles. Dynamic text is
HTML-escaped; each language is aligned with explicit direction attributes and
inline styles. This is a presentation helper, not a statistical test.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from html import escape
from pathlib import Path

from ._results import TestResult


def _interpretations(result: TestResult | Mapping) -> tuple[str, str]:
    if isinstance(result, TestResult):
        ar, en = result.interpretation_ar, result.interpretation_en
    elif isinstance(result, Mapping):
        ar = result.get("interpretation_ar", result.get("ar"))
        en = result.get("interpretation_en", result.get("en"))
    else:
        raise TypeError(
            "Expected a KhaledDSML TestResult or dictionary with "
            "interpretation_ar/interpretation_en (or ar/en)"
        )
    if not isinstance(ar, str) or not ar.strip():
        raise ValueError("Arabic interpretation must be a nonempty string")
    if not isinstance(en, str) or not en.strip():
        raise ValueError("English interpretation must be a nonempty string")
    return ar, en


def _normalize_results(results) -> list:
    if isinstance(results, (TestResult, Mapping)):
        return [results]
    if isinstance(results, (str, bytes)) or not isinstance(results, Iterable):
        raise TypeError("Expected a result or an iterable of bilingual results")
    items = list(results)
    if not items:
        raise ValueError("Provide at least one bilingual result")
    return items


def _validate_options(title: str | None, font_size: int) -> None:
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None")
    if type(font_size) is not int or not 14 <= font_size <= 36:
        raise ValueError("font_size must be an integer between 14 and 36")


def render_bilingual_results(
    results: TestResult | Mapping | Iterable,
    *,
    title: str | None = None,
    font_size: int = 23,
) -> str:
    """Render one or many bilingual interpretations in a borderless document.

    Arabic: right aligned, 700 weight, RTL. English: left aligned, 700 weight,
    LTR. Default title is omitted to match the compact Google Colab example.
    The complete HTML string is also suitable for printing in a browser.
    """
    _validate_options(title, font_size)
    items = _normalize_results(results)
    panels = []
    for item in items:
        ar, en = _interpretations(item)
        panels.append(
            '<section class="kdsml-pair" '
            'style="margin:20px 0 36px;padding:0;border:0;">'
            '<div class="kdsml-ar" lang="ar" dir="rtl" '
            'style="direction:rtl !important;text-align:right !important;'
            'font-family:Tahoma,Arial,sans-serif;'
            f'font-size:{font_size}px;font-weight:700;line-height:1.9;'
            'margin:0 0 20px;padding:0;white-space:pre-wrap;'
            'overflow-wrap:break-word;">'
            f'{escape(ar)}</div>'
            '<div class="kdsml-en" lang="en" dir="ltr" '
            'style="direction:ltr !important;text-align:left !important;'
            'font-family:Arial,Helvetica,sans-serif;'
            f'font-size:{font_size}px;font-weight:700;line-height:1.9;'
            'margin:0;padding:0;white-space:pre-wrap;'
            'overflow-wrap:break-word;">'
            f'{escape(en)}</div>'
            '</section>'
        )
    label = title if title is not None else "Bilingual interpretation | تفسير النتائج"
    heading = (
        '<h2 lang="ar" dir="rtl" style="direction:rtl !important;'
        'text-align:right !important;font-family:Tahoma,Arial,sans-serif;'
        'font-size:24px;font-weight:700;margin:0 0 14px">'
        f'{escape(title)}</h2>'
        if title else ""
    )
    return (
        '<!doctype html><html lang="ar" dir="rtl"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(label)}</title></head><body>'
        '<div class="khaled-dsml-bilingual" lang="ar" dir="rtl" '
        'style="box-sizing:border-box;max-width:1100px;'
        'margin:0 auto;padding:10px 20px;'
        'background:#fff;color:#222;border:0;'
        'font-family:Tahoma,Arial,sans-serif;">'
        + heading + "".join(panels) +
        '</div></body></html>'
    )


def render_bilingual_result(
    result: TestResult | Mapping, *,
    title: str | None = None,
    font_size: int = 23,
) -> str:
    """Backward-compatible singular alias with the new plain notebook style."""
    return render_bilingual_results(result, title=title, font_size=font_size)


def display_bilingual_results(
    results: TestResult | Mapping | Iterable, *,
    title: str | None = None,
    font_size: int = 23,
) -> None:
    """Display multiple Arabic-right / English-left interpretations in one cell."""
    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise ImportError(
            'Install IPython or khaled-dsml[notebooks] to display HTML reports'
        ) from exc
    display(HTML(render_bilingual_results(results, title=title, font_size=font_size)))


def display_bilingual_result(
    result: TestResult | Mapping, *,
    title: str | None = None,
    font_size: int = 23,
) -> None:
    """Display one bilingual result (Jupyter/Colab, no surrounding card)."""
    display_bilingual_results(result, title=title, font_size=font_size)


def save_bilingual_results(
    results: TestResult | Mapping | Iterable, path: str | Path, *,
    title: str | None = None,
    font_size: int = 23,
) -> Path:
    """Write UTF-8 HTML with all interpretations, ready for browser printing."""
    destination = Path(path)
    destination.write_text(
        render_bilingual_results(results, title=title, font_size=font_size),
        encoding="utf-8",
    )
    return destination


def save_bilingual_result(
    result: TestResult | Mapping, path: str | Path, *,
    title: str | None = None,
    font_size: int = 23,
) -> Path:
    """Backward-compatible singular save helper."""
    return save_bilingual_results(result, path, title=title, font_size=font_size)

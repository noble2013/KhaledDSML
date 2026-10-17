"""Accessible Arabic RTL / English LTR presentation for notebook output.

A notebook's print() stream uses the host terminal's text direction. This
module returns isolated HTML with inline alignment and safe escaping so that
mixed-language reports remain readable in Jupyter and Google Colab without
changing notebook-wide CSS or requiring an internet connection.
"""
from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path

from ._results import TestResult


def _interpretations(result: TestResult | Mapping) -> tuple[str, str]:
    if isinstance(result, TestResult):
        ar = result.interpretation_ar
        en = result.interpretation_en
    elif isinstance(result, Mapping):
        ar = result.get("interpretation_ar", result.get("ar"))
        en = result.get("interpretation_en", result.get("en"))
    else:
        raise TypeError(
            "Expected a KhaledDSML TestResult or a dictionary with "
            "interpretation_ar/interpretation_en (or ar/en)"
        )
    if not isinstance(ar, str) or not ar.strip():
        raise ValueError("Arabic interpretation must be a nonempty string")
    if not isinstance(en, str) or not en.strip():
        raise ValueError("English interpretation must be a nonempty string")
    return ar, en


def render_bilingual_result(
    result: TestResult | Mapping, *,
    title: str = "التفسير الإحصائي | Statistical interpretation",
    font_size: int = 21,
) -> str:
    """Return a safe self-contained Arabic-right / English-left HTML document.

    All dynamic fields are HTML-escaped. Inline styles are intentionally scoped
    to the panel, so Colab themes do not change the text alignment. Use
    display_bilingual_result() for convenient display inside notebook cells.
    """
    if type(font_size) is not int or not 14 <= font_size <= 36:
        raise ValueError("font_size must be an integer from 14 through 36")
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    arabic, english = _interpretations(result)
    ar = escape(arabic)
    en = escape(english)
    heading = escape(title)
    return (
        '<!doctype html><html lang="ar" dir="rtl"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{heading}</title></head><body>'
        '<section class="khaled-dsml-bilingual" '
        'style="box-sizing:border-box;max-width:1000px;margin:12px auto;'
        'padding:18px;border:1px solid #aeb8c8;border-radius:12px;'
        'background:#fff;color:#1f2937;">'
        '<h2 dir="rtl" lang="ar" style="text-align:right !important;'
        'direction:rtl !important;font-family:Tahoma,Arial,sans-serif;'
        'font-size:22px;font-weight:700;line-height:1.8;margin:0 0 12px">'
        f'{heading}</h2>'
        '<div class="kdsml-ar" dir="rtl" lang="ar" '
        'style="direction:rtl !important;text-align:right !important;'
        'font-family:Tahoma,Arial,sans-serif;'
        f'font-size:{font_size}px;font-weight:700;line-height:2;'
        'white-space:pre-wrap;padding:12px 8px;overflow-wrap:anywhere;">'
        f'{ar}</div>'
        '<hr style="border:0;border-top:1px solid #d4dbe5;margin:12px 0">'
        '<div class="kdsml-en" dir="ltr" lang="en" '
        'style="direction:ltr !important;text-align:left !important;'
        'font-family:Arial,Helvetica,sans-serif;'
        f'font-size:{font_size}px;font-weight:600;line-height:1.75;'
        'white-space:pre-wrap;padding:12px 8px;overflow-wrap:anywhere;">'
        f'{en}</div>'
        '</section></body></html>'
    )


def display_bilingual_result(
    result: TestResult | Mapping, *,
    title: str = "التفسير الإحصائي | Statistical interpretation",
    font_size: int = 21,
) -> None:
    """Show a readable RTL/LTR bilingual card in Jupyter or Google Colab."""
    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise ImportError(
            'Install IPython or khaled-dsml[notebooks] to display HTML reports'
        ) from exc
    display(HTML(render_bilingual_result(result, title=title, font_size=font_size)))


def save_bilingual_result(
    result: TestResult | Mapping, path: str | Path, *,
    title: str = "التفسير الإحصائي | Statistical interpretation",
    font_size: int = 21,
) -> Path:
    """Save a readable and self-contained UTF-8 report for browser printing."""
    destination = Path(path)
    destination.write_text(
        render_bilingual_result(result, title=title, font_size=font_size),
        encoding="utf-8",
    )
    return destination

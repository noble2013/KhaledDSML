"""Offline guards for bilingual author information and Colab description rendering."""
from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NOTEBOOK = ROOT / "notebooks/24_library_description_rtl_author.ipynb"
GUIDE = ROOT / "docs/COLAB_DESCRIPTION_RTL_AR_EN.md"


def test_readme_has_prominent_right_aligned_author_card():
    text = README.read_text(encoding="utf-8")
    opening = text[:3400]
    assert '<div dir="rtl" lang="ar" align="right">' in opening
    assert "<table dir=\"rtl\">" in opening
    for label in (
        "الاسم", "اللقب", "خالد", "علي", "Khaled Ali",
        "الأستاذ الدكتور", "الجامعة", "جامعة الشهيد حمه لخضر",
        "الكلية", "كلية العلوم الاقتصادية والتجارية وعلوم التسيير",
    ):
        assert label in opening
    assert 'lang="en" dir="ltr"' in opening
    assert "https://pypi.org/project/khaled-dsml/0.10.0/" in opening
    assert NOTEBOOK.name in text


def test_colab_notebook_is_runnable_and_renders_bilingual_markup():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 2
    install = "".join(code[0]["source"]).strip()
    assert install.startswith("%pip -q install")
    assert "markdown beautifulsoup4 requests" in install
    body = "".join(code[1]["source"])
    compile(body, f"{NOTEBOOK.name}:render", "exec")
    for token in (
        "raw.githubusercontent.com/noble2013/KhaledDSML/main/README.md",
        "markdown.markdown(", "BeautifulSoup(",
        'tag["dir"] = "rtl" if rtl else "ltr"',
        "display(HTML(", 'lang="ar" dir="rtl"',
        'direction: ltr !important',
    ):
        assert token in body
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)


def test_guide_distinguishes_legacy_academic_title_and_official_rank():
    text = GUIDE.read_text(encoding="utf-8")
    assert "الأستاذ الدكتور" in text
    assert "الرتبة الوظيفية الرسمية" in text
    assert "لا نضيف رتبة إدارية محددة إلا بعد تأكيدها" in text
    assert "PyPI" in text and "GitHub" in text

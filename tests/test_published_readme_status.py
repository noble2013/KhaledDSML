"""Public GitHub README must not contradict released PyPI version.

The PyPI version's long description is an immutable publish-time snapshot:
these tests protect the live GitHub content, not a rewritten PyPI archive.
"""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
README=ROOT/"README.md"
PUBDOC=ROOT/"docs/PUBLISH_AR.md"
RELEASEDOC=ROOT/"docs/FINAL_RELEASE_0_10_AR_EN.md"

def test_public_readme_reflects_published_stable_release():
    text=README.read_text(encoding="utf-8")
    intro=text.split("## لماذا هذه المكتبة؟",1)[0]
    assert "الإصدار الرسمي الأحدث" in intro
    assert "https://pypi.org/project/khaled-dsml/0.10.0/" in intro
    assert "https://github.com/noble2013/KhaledDSML/releases/tag/v0.10.0" in intro
    assert "نسخة المصدر النهائية: 0.10.0 — جاهزة" not in text
    assert "الإصدار الرسمي المنشور على PyPI: 0.9.0" not in text
    assert "موجودة في فرع التطوير فقط" not in text

def test_readme_gives_published_install_and_accurate_import():
    text=README.read_text(encoding="utf-8")
    assert 'python -m pip install "khaled-dsml==0.10.0"' in text
    assert 'from khaled_dsml import describe_numeric, display_descriptive_report' in text
    assert "display_descriptive_report(summary)" in text
    assert 'khaled-dsml[distributed]==0.10.0' in text
    assert 'khaled-dsml[econometrics,distributed]==0.9.0' not in text

def test_release_guides_acknowledge_real_publication():
    for file in (PUBDOC,RELEASEDOC):
        text=file.read_text(encoding="utf-8")
        assert "https://pypi.org/project/khaled-dsml/0.10.0/" in text
        assert "39072405164" in text
        assert "17 أكتوبر 2026" in text

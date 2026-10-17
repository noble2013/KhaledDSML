# KhaledDSML 0.10.0 — تم النشر على PyPI | Published stable release
## Verified stable release | الإصدار المستقر المنشور

> **تحديث حالة النشر — 17 أكتوبر 2026، 22:49 (الجزائر):** نُشر **[KhaledDSML 0.10.0 رسميًا على PyPI](https://pypi.org/project/khaled-dsml/0.10.0/)**، وصدر [GitHub Release v0.10.0](https://github.com/noble2013/KhaledDSML/releases/tag/v0.10.0)، واكتمل [GitHub Actions Publish to PyPI](https://github.com/noble2013/KhaledDSML/actions/runs/39072405164) بنجاح بعد اجتياز اختبارات المصدر والحزم. الفقرات اللاحقة التي تصف ما «يجب فعله قبل النشر» تُقرأ **كسجل خطوات الإصدار المنفَّذة**، لا كتصريح بأن 0.10.0 ما زال غير منشور.


**المؤلف:** خالد علي — جامعة الشهيد حمه لخضر، الوادي، الجزائر.

**Source and published version:** `0.10.0` ([PyPI](https://pypi.org/project/khaled-dsml/0.10.0/)).
**Previous GitHub release:** `v0.9.0`; current [GitHub release `v0.10.0`](https://github.com/noble2013/KhaledDSML/releases/tag/v0.10.0). README changes in GitHub do not retroactively replace metadata on the already published PyPI version.

### معيار إعلان جاهزية النشر

Publish **only when** a workflow for the exact `main` commit containing `0.10.0` passes **all seven jobs**: four Python 3.10–3.13 test suites; genuine Dask/PySpark distributed integration; execution of ten advanced Colab notebooks **14–23**; and building/twine-checking wheel and sdist with **two separately installed clean virtual environments** that import the installed package rather than the source checkout.

The final CI also verifies `tools/release_gate.py --target pypi --tag v0.10.0 --prerelease false`. This is a **validation-only** command; it does not publish or create a GitHub Release.

**Scientific limitations:** `256/256` catalogue rows have candidate public API mappings, **not 256 independently certified scientific methods**. Entries #85 (retrospective observed-power fallacy) and #117 (Poisson vs ZIP classical Vuong misuse) are deliberately linked to **restricted, safer substitutes**; never claim they are full equivalents. Bibliographic review, assumptions and expert sign-off remain necessary for research-grade certification. CI passing establishes software packaging/test readiness, not scientific certification.

### Windows PowerShell: full build, validation AND install

Run from the project root containing `pyproject.toml` after the final merge:

```powershell
git checkout main
git pull origin main

py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,econometrics,plots,exports,unitroot,quantiles]"

.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe tools\notebook_smoke.py
.\.venv\Scripts\python.exe tools\release_gate.py --target source
.\.venv\Scripts\python.exe tools\release_gate.py --target pypi --tag v0.10.0 --prerelease false

.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m twine check dist\*
.\.venv\Scripts\python.exe tools\check_distributions.py dist
.\.venv\Scripts\python.exe tools\install_dist_smoke.py dist

.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps ".\dist\khaled_dsml-0.10.0-py3-none-any.whl"
.\.venv\Scripts\python.exe -c "import khaled_dsml, importlib.metadata as m; print(khaled_dsml.__version__, m.version('khaled-dsml'))"
```

Expected: `0.10.0 0.10.0` and success without test failures.

**Windows note:** Close open Python/Jupyter processes holding SQLite/temporary files before deleting temporary environments. `tools/install_dist_smoke.py` installs the package twice in temporary independent venvs; internet access to PyPI is needed for dependencies unless pip is configured with an offline wheel cache.

### GitHub PyPI publication (only by owner's explicit decision)

After all checks are green for **the exact `main` stable commit**:

1. Make sure the `pypi` GitHub Environment exists with suitable required reviewers / protections; configure the **PyPI Trusted Publisher** for owner `noble2013`, repository `KhaledDSML`, workflow `release.yml`, and environment `pypi` (no token shared with ChatGPT).
2. In the GitHub Releases interface select **Draft a new release**, create tag `v0.10.0` **from the verified stable `main` commit**, and use title `KhaledDSML v0.10.0`.
3. Keep the `Pre-release` option **unchecked**. Publish the GitHub Release only when you want public PyPI publication. This triggers `.github/workflows/release.yml`; creating or merging an ordinary PR cannot publish.
4. Confirm the PyPI workflow succeeded, package `khaled-dsml==0.10.0` appears at https://pypi.org/project/khaled-dsml/, and do a clean external install:

```powershell
py -3.13 -m venv .publishvenv
.\.publishvenv\Scripts\python.exe -m pip install --upgrade pip
.\.publishvenv\Scripts\python.exe -m pip install "khaled-dsml==0.10.0"
.\.publishvenv\Scripts\python.exe -c "import khaled_dsml; print(khaled_dsml.__version__)"
```

Because PyPI distribution filenames are immutable, a mistakenly uploaded release cannot be silently overwritten.

**TestPyPI:** An optional TestPyPI trial for `0.10.0rc1` uses a different manual, prerelease-only workflow, requires the owner's explicit `TESTPYPI` confirmation and can only be executed from a GitHub main branch containing `0.10.0rc1`. **Do not try the current prerelease-only TestPyPI workflow against final stable `0.10.0`.** It is safe to run the full local/CI installation gates without a TestPyPI upload. Actual PyPI OIDC connectivity remains unverified until an authorized publication attempt.

### مراجع الوثائق

- [دليل نشر المشروع](PUBLISH_AR.md)
- [القيود العلمية لأعمال القدرة اللاحقة وVuong](ZERO_INFLATION_AND_PRECISION_0_10_AR_EN.md)
- [تدقيق كتالوج الـ256 بندًا](CATALOG_AUDIT_0_10_AR_EN.md)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [Python Packaging: PEP 440](https://peps.python.org/pep-0440/)

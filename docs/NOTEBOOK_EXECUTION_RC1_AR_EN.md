> **تحديث:** أصبح مسار smoke الحالي يشغل الدفاتر **14–23** (10 دفاتر)، وليس 19–23 فقط. راجع [ضوابط الإصدار النهائي](FINAL_RELEASE_0_10_AR_EN.md). يُحتفظ بهذه الوثيقة لسجل تطوير RC1.

# KhaledDSML 0.10.0rc1 — Offline notebook science smoke tests
## تشغيل الأمثلة العلمية فعليًا قبل الإصدار

**Source branch:** `feature/rc1-notebook-smoke-and-publish-hardening` applied after merged PR19. **Version stays `0.10.0rc1`; production PyPI remains unchanged.**

### Goal | الهدف

Previous CI verified callable functions and that notebook JSON files are parseable, but **did not execute entire teaching examples as the students would**. This new gate runs each executable Python code cell from Colab notebooks **19, 20, 21, 22 and 23** in order, on synthetic data, in isolated temporary working directories.

The installed optional dependencies are `[dev,econometrics,plots,exports,unitroot,quantiles]`. Only a single initial standalone `%pip install ...` cell is intentionally skipped because CI already installed these packages. Other notebook magics or shell commands raise errors rather than being silently discarded.

The headless smoke runner replaces only the front-end **display** function with the real `render_analysis_dashboard` HTML renderer: each analysis function and each report generator is still executed, and every generated report must contain valid offline RTL Arabic support. The Office-export notebook 20 **actually creates and opens** editable XLSX and DOCX outputs within a temporary directory (never a network upload). The test exits nonzero at the first cell raising an exception; no success is fabricated.

Run locally after installing the extras:

```powershell
git checkout main
git pull origin main
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,econometrics,plots,exports,unitroot,quantiles]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe tools\notebook_smoke.py
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe tools\check_distributions.py dist
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps ".\dist\khaled_dsml-0.10.0rc1-py3-none-any.whl"
```

### CI release gates | ضوابط CI

The pipeline now has **seven jobs** in total: four Python 3.10–3.13 tests, one real Dask/PySpark integration, one offline `notebook-smoke` teaching workflow, and one dependent `distributions` build/install/metadata + 256-row source-archive validation. The final build depends on ALL prior scientific/API and notebook smoke jobs.

Production PyPI workflow separately checks:
- GitHub release commit must be an **ancestor of `origin/main`**, preventing release of an unreviewed divergent branch.
- Source version must be stable (not dev/a/b/rc), GitHub tag must match and release must not be flagged `prerelease`.
- All unit tests and notebooks 19–23 must run before signed OIDC publishing.

TestPyPI workflow separately checks:
- Workflow must have been manually triggered from **`refs/heads/main`**.
- User must explicitly enter `TESTPYPI`.
- Source version must be prerelease and tests/notebooks/build must succeed.
- Publishing remains controlled by GitHub environment `testpypi` and Trusted Publisher permissions.

**These safeguards do not themselves publish.** The repository owner must separately authorize a TestPyPI run and configure environments/OIDC appropriately.

### Scientific honesty | النزاهة العلمية

All 256 catalogue rows have candidate source APIs, **not** 256 full methodological approvals. Entries #85 and #117 have restricted replacement methods; the normal post-hoc observed power and invalid Poisson-vs-ZIP Vuong inference are NOT represented as valid completed methods. Executing teaching notebooks checks that examples work; it does not certify assumptions or scientific correctness for every possible dataset.

### Main resources

- [Publication guide / دليل النشر](PUBLISH_AR.md)
- [Release candidate audit / تدقيق جاهزية الإصدار](RELEASE_CANDIDATE_0_10_AR_EN.md)
- [Zero-inflation and power methodology / ضوابط الاستدلال](ZERO_INFLATION_AND_PRECISION_0_10_AR_EN.md)
- [Catalogue methods audit](CATALOG_AUDIT_0_10_AR_EN.md)

> **تاريخي / Archived:** يوثق هذا الملف مرحلة `0.10.0rc1` السابقة. الإصدار المصدري اللاحق `0.10.0` وإجراءات جاهزيته موثقة في [دليل الإصدار النهائي](FINAL_RELEASE_0_10_AR_EN.md). لا تخلط بين الإصدارين؛ لم ينشر `0.10.0` تلقائيًا إلى PyPI.

# KhaledDSML 0.10.0rc1 — Release candidate readiness
## جاهزية إصدار الاختبار الأول (عربي / English)

### Status | الحالة

- **Source version:** `0.10.0rc1` — PEP 440 release candidate; **not yet uploaded** to PyPI or TestPyPI by this change.
- **Last published stable before RC preparation:** `0.9.0`.
- **Supported target versions:** Python 3.10–3.13.
- **Research trace:** All 256 catalogue IDs have importable candidate APIs. This does **not** certify 256 methods. Original #85 maps to externally specified effect sensitivity, **not observed power**; #117 maps to strictly nonnested Vuong plus a separate van den Broek Poisson zero-excess score, **not Poisson-vs-ZIP Vuong**.

### Distribution evidence | سلامة التوزيع

The release candidate gate checks all of the following (see also [actual notebook execution](NOTEBOOK_EXECUTION_RC1_AR_EN.md)):

1. `pyproject.toml` version equals the public `khaled_dsml.__version__` exactly.
2. Version follows the project's canonical PEP 440 scheme; a stable PyPI release must have plain `X.Y.Z` syntax and matching `vX.Y.Z` GitHub tag.
3. `MANIFEST.in` includes `docs/CATALOG_256_SOURCE_TRACE_V010.tsv`, and actual generated `sdist` contains 256 catalogue rows.
4. Built `wheel` contains package `__init__.py` and distribution metadata matching `0.10.0rc1`.
5. Four Python CI targets pass independent numerical tests; actual Dask/PySpark tests run with Java 17; **Colab notebooks 19–23 run as real offline research examples**; distributions build, validate and install.
6. Both publishing workflows use up-to-date test extras `econometrics`, `plots`, `exports`, `unitroot` and `quantiles`. They never need saved PyPI passwords or GitHub secrets.
7. TestPyPI requires manual `TESTPYPI` confirmation, rejects stable source versions and requires configured OIDC. Production PyPI rejects release candidates and prerelease GitHub Releases.
8. A fresh `pip install` after building the wheel verifies that the installed Python import version matches the source.

**Never equate packaging checks with a scientific methods audit**. Independent methodological validation remains necessary before researchers infer causality, use asymptotic p-values on small samples, or assume equivalence between a restricted API and a broad named test.

### Python / PowerShell sample workflow

```powershell
git checkout main
git pull origin main
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,econometrics,plots,exports,unitroot,quantiles]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe tools\release_gate.py --target source
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m twine check dist\*
.\.venv\Scripts\python.exe tools\check_distributions.py dist
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps ".\dist\khaled_dsml-0.10.0rc1-py3-none-any.whl"
.\.venv\Scripts\python.exe -c "import khaled_dsml; print(khaled_dsml.__version__)"
```

For a **local trial only**, no GitHub Release, API token, TestPyPI workflow or production publish action is needed. The source code is not the published PyPI package until the separate OIDC workflow succeeds.

### Scientific quality review / مراجعة مستقلة مطلوبة

The catalogue lists supported candidates across traditional inference, multivariate methods, time-series, big data, model evaluation, psychometrics, numerical sketches, monitoring, and forensic **screens**. Some functions intentionally reject unsupported use cases; none promises certification for arbitrary data distributions or experimental designs.

Before advancing to `0.10.0` stable, review a representative independent numerical validation for each METHOD (not merely an import), broken assumptions, missing-data treatment, multiplicity, computational complexity, numerical stability and Arabic/English interpretations. Recheck all 256 original catalogue entries against their exact method definitions, rather than source/API mapping alone.

### Official reference material

- [PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [PEP 440 version ordering](https://peps.python.org/pep-0440/)
- [Project-specific publication guide](PUBLISH_AR.md)
- [256-item evidence matrix](CATALOG_AUDIT_0_10_AR_EN.md)
- [Scientific restricted alternatives](ZERO_INFLATION_AND_PRECISION_0_10_AR_EN.md)

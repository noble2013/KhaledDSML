# KhaledDSML 0.10 — Phillips–Perron and numerical research integrity
## فيليبس–بيرون وGRIMMER وفحص منحنى قيم p

**Status / الحالة:** `0.10.0.dev0` development branch `feature/v0.10-unitroot-forensic-pcurve`. This is NOT an official stable PyPI 0.10 release. Published stable remains **0.9.0**.

## 1 — Phillips–Perron | اختبار فيليبس–بيرون

**Public API:** `phillips_perron_unit_root(series,trend="c",test_type="tau",lags=None,alpha=.05)`.

The computation delegates to independently maintained `arch.unitroot.PhillipsPerron`; installed with `"khaled-dsml[unitroot]"`. The null is a **unit root**; alternatives are stationarity around no trend (`n`), a constant (`c`), or constant and linear trend (`ct`). The choices `tau` and `rho` correspond to the standard Phillips–Perron variants. `lags` is the **Newey–West long-run covariance bandwidth**, not the ADF order of lagged differenced regressors. P-values are based on MacKinnon approximations from `arch`, and critical values are provided in the result table. The function rejects missing, constant or undersized observations.

**العربية:** عدم رفض فرض جذر الوحدة لا يثبت بشكل قطعي عدم الاستقرار. قد تغيّر الانقطاعات الهيكلية، والموسمية، والاختيارات المتكررة للاتجاه أو التأخر، الاستدلال. لا تطبق الدالة اختبارات الوحدات الكسرية أو اختبارات الانقطاع.

**Reference implementation:** <https://bashtage.github.io/arch/unitroot/generated/arch.unitroot.PhillipsPerron.html>.

## 2 — GRIMMER-style integer mean + sample SD | تدقيق المتوسط والانحراف المعياري

**Public API:** `grimmer_variance_consistency(reported_mean,reported_sd,sample_size,decimals_mean=2,decimals_sd=2,minimum_score=1,maximum_score=5)`.

The algorithm enumerates **all reachable pairs of exact integer totals and squared totals** of the specified integer score scale using bounded integer bitset dynamic programming, with `n=2..50` and a scale spanning at most nine consecutive integer scores. It matches BOTH a printed rounded mean and a printed **sample** standard deviation using **ROUND_HALF_UP** at their declared decimal precisions. `ddof=1`; it does not confuse sample SD with population SD or standard error. An arithmetic witness (sum and squared sum) is returned when one exists.

**العربية:** مخالفة GRIMMER حسابية فقط وقد تعكس سياسة تقريب مختلفة أو عدم إدخال جميع الحالات أو اختلاف المقاييس، وهي ليست إثباتًا للتلاعب. قبول المتوسط والانحراف لا يثبت أن البيانات الخام صحيحة.

The result does not reconstruct the original dataset. It only proves that at least one integer sequence on the declared scale is arithmetically feasible.

## 3 — Significant p-value half-curve | نصف منحنى قيم p الدالة

**Public API:** `pcurve_right_skew_screen(significant_p_values,threshold=.05,midpoint=None,alpha=.05)`.

A deliberately **limited half-curve right-skew screen**: for at least five independent **exact** significant p-values strictly within `(0,threshold)`, it counts those below the midpoint (by default `threshold/2`) and compares that count with a one-sided binomial reference under a **conditional-uniform** null. It uses `scipy.stats.binomtest` for the actual p-value.

**Critically, this is NOT full p-curve analysis.** It does not estimate statistical power, correct selective-reporting bias, test p-hacking conclusively, combine dependent findings, or substitute for preregistered systematic review. Results gathered after searching published papers may strongly violate the reference null. Different significance rules, rounded p-values and correlated studies invalidate naïve interpretation.

**العربية:** هدف الأداة تنبيه استكشافي محدود إلى توزيع القيم الدالة، وليس إثبات النشر الانتقائي أو تقدير قوة الدراسات. يجب جمع القيم الدقيقة وفق بروتوكول مسبق وتقييم استقلالية نتائج الأبحاث.

## Reproducible example | مثال تدريسي

```python
import numpy as np
from khaled_dsml import (
    phillips_perron_unit_root,
    grimmer_variance_consistency,
    pcurve_right_skew_screen,
    display_analysis_dashboard,
)
rng=np.random.default_rng(2026)

pp=phillips_perron_unit_root(np.cumsum(rng.normal(size=200)),trend="c")
display_analysis_dashboard(pp,theme="ocean")

grimmer=grimmer_variance_consistency(
    "2.00","1.41",sample_size=2,
    minimum_score=1,maximum_score=3
)
display_analysis_dashboard(grimmer,theme="amethyst")

half=pcurve_right_skew_screen([.001,.002,.011,.018,.022,.027,.03,.041])
display_analysis_dashboard(half,theme="aurora")
```

### Running tests and building on Windows | الاختبار والبناء والتثبيت

```powershell
git checkout main
git pull origin main
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,econometrics,exports,unitroot]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps ".\dist\khaled_dsml-0.10.0.dev0-py3-none-any.whl"
```

## Test and catalogue trace | التحقق العلمي

`tests/test_unitroot_forensic_specialist.py` cross-checks PP statistics, approximated p-values, critical values and Newey–West lag choices against direct independent `arch` invocations. GRIMMER feasible moment pairs are tested against **full Cartesian enumeration of integer score datasets** for small samples and independent known SD examples. The half-curve screening p-value is checked against SciPy's exact `binomtest` reference.

Catalogue IDs **#198, #240 and #243** have now been mapped as source-level API candidates. **248/256 entries mapped, 8 without any mapped public callable**, and the mapping must NOT be described as 248 fully scientifically certified statistical procedures.

**Remaining unlinked IDs:** #83 small-sample generic Bayes factors, #85 observed post-hoc power, #117 zero-inflated Vuong comparison, #123 SEM global fit indices, #124 measurement invariance, #188 SHAP stability, #229 t-digest, #241 SPRITE. Specialized libraries and independent references must be reviewed before promising those methodological features.

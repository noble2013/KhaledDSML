# Research methods under development — 0.10.0.dev0

## A. Henze–Zirkler test | اختبار هينز–زيركلر

**English.** Tests the composite null that complete independent observations arise from a nonsingular multivariate normal distribution. The implementation uses the HZ Gaussian-kernel statistic with the conventional smoothing bandwidth and ML covariance whitening. Its p-value is **parametric Monte Carlo**, not an undocumented lognormal approximation. Each independent standard-normal null sample is re-centred and re-whitened; affine invariance makes this calibration valid under unknown Gaussian means and nonsingular covariance. P-value = `(1 + number of simulated statistics >= observed)/(B+1)`. Minimum recommended `B=199` (minimum accepted 99). For large samples set `n_resamples=999` or more to resolve small p-values.

**العربية.** يختبر الاعتدالية المشتركة لعينة مستقلة وكاملة، مع معايرة مونت كارلو وإعادة تقدير المتوسطات ومصفوفة التغاير في كل تكرار. لا يعني عدم الرفض إثبات الاعتدالية. القيمة الاحتمالية محدودة الدقة وفق عدد المحاكاة، ويجب معالجة القيم المفقودة بطريقة معلنة وعدم الخلط بين اختبار التوزيع وشرط استقلالية المشاهدات.

```python
import numpy as np
from khaled_dsml import henze_zirkler_normality_test, display_analysis_dashboard

x = np.random.default_rng(17).normal(size=(90, 3))
result = henze_zirkler_normality_test(
    x, alpha=0.05, n_resamples=199, random_state=2026
)
display_analysis_dashboard(result, theme="aurora")
```

**Cost:** `O((B+1)*n²)` pairwise distances; restricted to `n<=700` to avoid accidental quadratic workloads. No external pingouin or cloud dependency.

## B. MNAR delta scenario | سيناريوهات دلتا الافتراضية للبيانات المفقودة

**English.** The `mnar_delta_sensitivity` API takes an original numeric frame with its original missing mask, **at least two** completed frames preserving all observed values, one target column and hypothetical additive `deltas`. Each delta changes *only previously missing target values*. The completed-dataset mean for each imputation then changes by `delta * missing_fraction`. The table reports means across imputations and their descriptive range. **These are scenario summaries, not pooled confidence intervals, p-values, or evidence of an identified MNAR mechanism.** A valid inferential treatment would require a compatible fitted model and additional MNAR assumptions.

**العربية.** يسمح بتحليل حساسية وصفي تحت سيناريوهات افتراضية يحددها الباحث. تُزاح القيم التي كانت مفقودة أصلًا فقط، وتبقى القيم المرصودة كما هي. لا يقدّم الاختبار نموذجًا مثبتًا لآلية MNAR، ولا يختلق قيمة احتمالية أو فاصل ثقة.

```python
import pandas as pd
from khaled_dsml import mnar_delta_sensitivity, display_analysis_dashboard

original = pd.DataFrame({"income": [100., None, 180., None, 250.], "age": [24, 30, 41, 35, 50]})
a = original.fillna({"income": 150.})
b = original.fillna({"income": 170.})
result = mnar_delta_sensitivity(
    original, [a, b], column="income", deltas=[-30., -15., 0., 15., 30.]
)
display_analysis_dashboard(result, theme="amethyst")
```

## Validation / التحقق

Dedicated `tests/test_v010_research.py` exercises independent HZ kernel computation, affine invariance, Monte Carlo resolution/reproducibility, a clear non-Gaussian alternative, parameter and rank validation, preservation of observed values, MNAR delta algebra and Analytics Studio rendering. All tests must pass Python 3.10–3.13, including existing Spark/Dask checks, before merge. The code on this branch is **not** the published v0.9.0 release.

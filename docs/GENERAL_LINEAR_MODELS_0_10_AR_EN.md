# KhaledDSML 0.10 — General Linear Models | النماذج الخطية العامة

**Development only / نسخة مصدر غير منشورة:** `feature/v0.10-general-linear-models`, `0.10.0.dev0`. The last officially published PyPI distribution remains **0.9.0**.

These methods require optional `statsmodels` (install `"khaled-dsml[econometrics]"`) and are imported lazily when used. The returned result dictionaries work with bilingual offline Analytics Studio reports, Arabic RTL and English LTR. All examples below are synthetic.

## 1. One-factor ANCOVA | تحليل التغاير

`ancova_one_factor(response,groups,covariate,alpha=.05)` implements classical ANCOVA with one categorical grouping variable and one continuous covariate. It fits three OLS formulas on the **same complete, independent cases**:

- Reduced: `response ~ covariate`.
- Additive: `response ~ covariate + group`. The added **partial F group test** checks adjusted group means under a common covariate slope.
- Interaction: `response ~ covariate * group`. A **separate diagnostic F** tests homogeneity of slopes by comparing additive against interaction. A significant interaction makes interpretation of common-slope group means questionable.

The adjustment evaluates group means at the *pooled* observed covariate mean, not an arbitrary group baseline. Outputs include F, numerator/denominator df, p-value, partial eta squared, interaction/slope-homogeneity F and p-value, and a group table of adjusted predictions.

**العربية:** يقتضي ANCOVA توازي الانحدارات، استقلالية الوحدات، تجانسًا ملائمًا للتباين وشروطًا مناسبة لبواقي OLS. إذا كان التفاعل دالًا، فلا تعتمد المتوسطات المعدلة ذات الميل المشترك بلا تحليل آخر. **الضبط الإحصائي لا ينشئ سببية من بيانات رصدية.**

## 2. One-way MANOVA | تحليل التباين متعدد المتغيرات

`manova_one_factor(outcomes,groups,alpha=.05)` takes a complete numeric matrix with at least **two continuous outcomes** and independent grouped subjects. It uses `statsmodels.multivariate.manova.MANOVA` and reports:

- **Wilks' lambda** — default designated decision, with approximate F and p.
- **Pillai's trace**.
- **Hotelling–Lawley trace**.
- **Roy's greatest root**.

The pooled **within-group** covariance must be full rank; the code verifies this and minimum per-group sample size. The four tests share the same grouped multivariate null, and should not be counted as four independent confirmations. Multivariate normality and equal covariance assumptions need external diagnostics.

**العربية:** MANOVA لا يحل محل اختبار القياسات المتكررة على الأشخاص أنفسهم. يشترط استقلال الحالات وتماثل صياغة القياس ومصفوفة تغاير داخلية كاملة الرتبة. دلالة ويلكس تقريبية تبعًا لشروط النموذج.

## 3. Repeated-measures ANOVA | تحليل تباين القياسات المتكررة

`repeated_measures_one_factor(wide,alpha=.05)` expects **one row per subject**, **one column per occasion**, no missing visits, and a balanced design. It uses `statsmodels.stats.anova.AnovaRM` on a long-format internally reconstructed table. Reported F, df and p are **uncorrected** for violation of the sphericity assumption. No Greenhouse–Geisser/Huynh–Feldt correction is claimed. For missing visits, complex group-by-time factors, or correlated observations with heterogeneous residuals, consider appropriately specified mixed-effects models.

**العربية:** تعالج هذه الدالة القياسات المتكررة لفرد واحد ضمن صف واحد. لا يُنصح باستخدام دلالة F الكلاسيكية عندما تنتهك الكروية دون تصحيح، ولا تُستخدم لإكمال الزيارات المفقودة اصطناعيًا.

## Example / مثال تطبيقي

```python
import numpy as np
import pandas as pd
from khaled_dsml import (
    ancova_one_factor, manova_one_factor,
    repeated_measures_one_factor, display_analysis_dashboard,
)
rng = np.random.default_rng(12)
groups = ["A"]*24 + ["B"]*24
cov = rng.normal(10,2,48)
outcome = 5 + .9*cov + np.repeat([0,1.2],24) + rng.normal(0,1,48)
display_analysis_dashboard(ancova_one_factor(outcome,groups,cov),theme="ocean")

matrix = rng.multivariate_normal([0,0],[[1,.3],[.3,1]],size=48)
matrix += np.repeat([[0,0],[.6,-.2]],24,axis=0)
display_analysis_dashboard(manova_one_factor(matrix,groups),theme="amethyst")

wide = pd.DataFrame(rng.normal(size=(18,3)) +
    np.array([0,.3,.8]),columns=["T0","T1","T2"])
display_analysis_dashboard(repeated_measures_one_factor(wide),theme="aurora")
```

## Scientific references and independent checks | مراجع التحقق

`tests/test_general_linear_models.py` verifies:
- ANCOVA added-variable F by **independently computed** residual sums of squares and F distribution, plus adjusted predictions against `statsmodels`.
- MANOVA Wilks' lambda independently from **within-group E and between-group H determinant ratio** `det(E)/det(E+H)`, plus all four statistics versus `statsmodels`.
- Repeated ANOVA F using a direct **subjects × occasions sum-of-squares** decomposition, plus exact `AnovaRM` reference.
- Degenerate designs, collinear outcomes/covariates, nonfinite values, group-size constraints, incomplete visits, and bilingual HTML.
- CI across Python 3.10–3.13, true Spark/Dask integration, and wheel/sdist build/install remain mandatory before merge.

**Catalogue:** these three methods link original **IDs #37, #38, #39**. This is **source-level candidate mapping**, not verified completion of every requirement inside the original 256 scientific catalogue items. Published stable remains 0.9.0.

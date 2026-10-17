# KhaledDSML 0.4.0 — الحزمة الأكاديمية الموسعة | Extended Academic Analysis Suite

**الحالة:** نسخة تطويرية (Unreleased). الإصدار الرسمي على PyPI هو 0.3.0.  
**Author / المؤلف:** Dr. Khaled Ali — University of El Oued, Algeria.

## ما تمت إضافته | Implemented modules

| وحدة / Module | وظائف تنفيذية جديدة / APIs | ملاحظات علمية |
|---|---|---|
| `inference_extended` | `welch_anova_test`, `mood_median_test`, `brunner_munzel_test`, `mcnemar_test`, `cochran_q_test` | استقلال/ازدواج العينات حسب الاختبار |
| النسب | `one_proportion_z_test`, `two_proportions_z_test`, `wilson_proportion_interval` | تحقق التقريب الطبيعي؛ Wilson للعينات الصغيرة |
| الارتباط | `point_biserial_test`, `partial_correlation_test`, `cramers_v`, `phi_coefficient`, `fisher_z_correlation_test` | مقارنة Fisher تخص عينتين مستقلتين |
| أحجام الأثر | `glass_delta`, `eta_squared`, `omega_squared`, `cohen_w`, `contingency_risk_measures` | لا تعني سببية |
| `psychometrics_extended` | `item_total_statistics`, `split_half_reliability`, `test_retest_correlation`, `cohen_kappa_agreement`, `fleiss_kappa_agreement` | ثبات الاستبيانات والاتفاق |
| البناء العاملي | `factor_scree_table`, `parallel_analysis`, `exploratory_factor_analysis`, `convergent_validity`, `htmt_ratio`, `fornell_larcker_matrix`, `harman_single_factor` | EFA استكشافي؛ HARMAN ليس دليل غياب الانحياز |
| `ml_extended` | `classification_diagnostics`, `calibration_summary`, `detailed_regression_metrics`, `cluster_validation_metrics`, `binary_group_fairness`, `time_series_split_indices`, `repeated_cv_indices`, `model_overfit_gap` | ECE معيار مرتبط بالتقسيم؛ العدالة ليست استنتاجًا قانونيًا |
| `regression_extended` | `regression_white_test`, `regression_goldfeld_quandt_test`, `regression_reset_test`, `regression_rainbow_test`, `regression_influence_table`, `chow_structural_break_test`, `regression_cusum_test`, `model_information_criteria`, `logistic_hosmer_lemeshow`, `glm_overdispersion_diagnostic`, `two_way_anova` | تتطلب نماذج ملائمة؛ Chow نقطة تحول مسبقة |
| `timeseries_extended` | `kpss_stationarity_test`, `zivot_andrews_break_test`, `granger_predictivity_tests`, `engle_granger_cointegration_test`, `johansen_cointegration_test`, `diebold_mariano_test` | اختبار Granger **تنبؤي** وليس إثباتًا للسببية |

هذا التوسع **ليس تنفيذًا كاملًا لبنود الكتالوج الـ256**. البنود التي لا تزال غير مغطاة تشمل، من بين أمور أخرى: Little MCAR، Mardia/Henze-Zirkler، نماذج SEM/CFA وتكافؤ القياس، بعض الاختبارات البايزية، مقارنة AUC بأسلوب DeLong، تحليل NLP المتقدم، اختبارات العناقيد المتخصصة، تصدير Word/Excel وفق APA، وباقي اختبارات النزاهة العلمية. تُعد الوظيفة مكتملة فقط بعد اختبارها والتحقق منها مقابل مرجع مستقل.

## التثبيت والتجربة | Google Colab

```python
%pip install --upgrade statsmodels
%pip install --no-deps --force-reinstall --no-cache-dir "git+https://github.com/noble2013/KhaledDSML.git@feature/v0.4-academic-analysis-suite"
```

ثم اختر **Runtime → Restart session**، وبعدها نفذ أمثلة النتائج. لا تستخدم `pip install khaled-dsml==0.4.0` قبل نشره رسميًا.

## تحليل النسب + تفسير ثنائي اللغة

```python
from khaled_dsml import (
    two_proportions_z_test, wilson_proportion_interval,
    display_analysis_dashboard,
)

r = two_proportions_z_test(73, 120, 49, 120)
interval = wilson_proportion_interval(73, 120)
display_analysis_dashboard(
    [r, interval],
    title_ar="تحليل مؤشرات الاستجابة",
    title_en="RESEARCH INDICATORS · COMPARATIVE PROPORTIONS",
    theme="aurora",
)
```

## تحليل استبيان وقياس الثبات

```python
import numpy as np, pandas as pd
from khaled_dsml import (
    item_total_statistics, factor_scree_table,
    parallel_analysis, exploratory_factor_analysis,
)
rng = np.random.default_rng(41)
latent = rng.normal(size=120)
items = pd.DataFrame({
    f"q{i}": latent + rng.normal(scale=.45, size=120)
    for i in range(1, 7)
})
display(item_total_statistics(items))
display(factor_scree_table(items))
display(parallel_analysis(items, n_iter=100, seed=42))
efa = exploratory_factor_analysis(items, n_factors=1)
display(efa["loadings"])
```

**تنبيه:** موافقة EFA أو ارتفاع Cronbach alpha لا تثبت تلقائيًا صدق بناء القياس، وبيانات Likert الرتبية قد تحتاج ارتباطات polychoric وتحليلًا تأكيديًا مستقلًا.

## تقييم التصنيف والمعايرة والإنصاف

```python
from khaled_dsml import classification_diagnostics, display_analysis_dashboard

labels=[0,0,1,1,0,1,0,1]
scores=[.07,.21,.68,.90,.29,.71,.42,.94]
result = classification_diagnostics(labels, scores, n_bins=4)
display_analysis_dashboard(result, theme="ocean",
    title_ar="تقييم جودة نموذج التصنيف",
    title_en="MODEL VALIDATION · CLASSIFICATION")
```

تُعرض المقاييس المحسوبة من البيانات دون توليد قيم مصطنعة. `average_precision` مختلف عن `pr_auc` الهندسي؛ ومؤشر التفاوت بين المجموعات لا يحل محل مراجعة التحيز والسياق.

## اختبارات الانحدار

```python
import numpy as np
import statsmodels.api as sm
from khaled_dsml import (
    regression_white_test, regression_influence_table,
    display_analysis_dashboard,
)
rng=np.random.default_rng(17)
x=np.linspace(0,10,120)
y=2+1.5*x+rng.normal(size=120)
X=sm.add_constant(x)
fitted=sm.OLS(y,X).fit()

r=regression_white_test(fitted.resid,X)
display_analysis_dashboard(r,theme="amethyst")
display(regression_influence_table(fitted).head())
```

## فحص السلاسل الزمنية

```python
import numpy as np
from khaled_dsml import (
    kpss_stationarity_test, diebold_mariano_test,
    display_analysis_dashboard,
)
rng=np.random.default_rng(2)
stationary=rng.normal(size=120)
test=kpss_stationarity_test(stationary)
display_analysis_dashboard(test,theme="aurora",chart_data=stationary)
```

**مهم:** KPSS يختبر السكون كفرضية صفرية، بعكس ADF. وفشل رفض العدمية لا يثبت السكون. وكل القيم الاحتمالية تعتمد على تصميم العينة وصحة الفروض.

## أمان البحث والواجهة | Research safeguards

- Arabic paragraphs are explicitly RTL/right-aligned, English paragraphs LTR/left-aligned.
- تظل واجهة Analytics Studio ذات التنسيقات الإبداعية `aurora`, `ocean`, `amethyst`.
- لا تُعرض نتيجة احصائية لم تحسبها الدالة. لا تُحذف ملاحظات دون قرار بشري.
- رفض فرضية العدم لا يثبت سببية، وقبولها الإجرائي لا يثبت صحتها.
- بيانات التدريب والتقييم تُفصل زمنيًا أو بحسب المستجيبين وفق تصميم الدراسة.
- اختبارات الإحصاء الاستدلالي ينبغي إعادة التحقق منها على عينات بحثية حقيقية مع خبير إحصائي قبل النشر العلمي.

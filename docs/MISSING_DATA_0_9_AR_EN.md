# KhaledDSML 0.9 — Missing-Data Inference and Multiple Imputation

**حالة الإصدار:** تطويري وغير منشور بعد؛ أحدث إصدار PyPI رسمي هو **0.8.0**.  
**Author / المؤلف:** Khaled Ali — University of El Oued.

## المنهج العلمي | Scientific scope

تُعالج هذه الوحدة البيانات المفقودة دون افتراض أن أي آلية للفقد قد ثبتت تلقائيًا. يجب عدم حذف البيانات أو تعويضها تلقائيًا دون مراجعة الباحث لتصميم جمع البيانات، وسبب الفقد، وطبيعة المتغيرات.

| API | الوظيفة | القيود |
|---|---|---|
| `little_mcar_test(data)` | اختبار Little بافتراض الاعتدالية متعددة المتغيرات، وتقدير متوسط/تغاير نموذج البيانات غير المكتملة بواسطة EM | Asymptotic chi-square; requires enough complete cases; no fully empty rows |
| `missingness_association_screen(data)` | فحص ارتباط غياب كل متغير بالمتغيرات العددية المرصودة باستعمال point-biserial وHolm | Exploratory; cannot prove MAR or exclude MNAR |
| `multiple_impute_train_test_numeric(train,test)` | عدة نسخ تعويضية احتمالية باستخدام IterativeImputer وBayesianRidge | Fits only on training; no categorical features; may violate variable bounds |
| `rubin_pool_scalar(estimates, variances)` | دمج تقديرات عددية أحادية البعد وتبايناتها حسب Rubin | Scalar pooling; assumes congenial imputation/analysis models; does not invent a p-value |

### لماذا اختبار Little؟

فرضية العدم لاختبار Little هي **MCAR**: أن آلية غياب البيانات مستقلة عن القيم المرصودة وغير المرصودة في المجتمع، وفق الفروض الإحصائية للاختبار. **رفض MCAR لا يثبت MAR أو MNAR**، وعدم رفض الاختبار لا يثبت MCAR. الأنماط القليلة المشاهدات قد تجعل تقريب \(\chi^2\) ضعيفًا.

الإحصائية الأساسية:

\[
\chi^2_{Little}=\sum_r n_r (\bar{x}_{r,obs}-\hat{\mu}_{obs})^\top
\hat{\Sigma}_{obs}^{-1}(\bar{x}_{r,obs}-\hat{\mu}_{obs})
\]

درجات الحرية \(df=\sum_r p_r-p\)، مع حذف الأنماط عديمة المتغيرات المرصودة من التطبيق الحالي. يجري تقدير \(\hat\mu\) و\(\hat\Sigma\) عبر EM مع التوقع الشرطي والتغاير الشرطي، بدل حشو القيم الناقصة بمتوسطات بسيطة.

## Colab experiment / التطبيق

```python
%pip install --force-reinstall --no-cache-dir --no-deps "git+https://github.com/noble2013/KhaledDSML.git@feature/v0.9-missing-data-inference"
```

Restart the Python runtime after installation.

```python
import numpy as np
import pandas as pd
import khaled_dsml as kd

rng=np.random.default_rng(25)
frame=pd.DataFrame(
    rng.multivariate_normal(
        [500,100,70],
        [[2000,90,40],[90,120,20],[40,20,70]],
        size=210,
    ),
    columns=["الدخل","الإنفاق","الثقة"]
)
frame.loc[::8,"الإنفاق"]=np.nan
frame.loc[::9,"الثقة"]=np.nan
result=kd.little_mcar_test(frame)
kd.display_analysis_dashboard(
    result,theme="aurora",
    title_ar="تقييم آلية غياب البيانات",
    title_en="LITTLE MCAR · MISSING-DATA AUDIT",
    font_size=22,
)
```

The report displays **actual** chi-square statistic, degrees of freedom, p-value, missingness patterns and notes. A threshold decision does not establish the absence of unmeasured confounding or confirm MAR.

## استكمال متعدد دون تسرب تدريب/اختبار

```python
train=frame.iloc[:160].copy()
test=frame.iloc[160:].copy()

completed=kd.multiple_impute_train_test_numeric(
    train,test,n_imputations=5,seed=42
)
for completed_train,completed_test in completed["imputations"]:
    assert completed_train.notna().all().all()
    assert completed_test.notna().all().all()
```

كل مُعَوِّض يُلائم النموذج على **التدريب فقط** ثم يحوّل الاختبار. إذا كان في الاختبار توزيع مختلف، لا يعيد ملاءمة المُعَوِّض له. هذا لا يحل جميع أنواع تسرب المعلومات من التصميم نفسه.

### تجميع النتائج وفق Rubin

```python
pool=kd.rubin_pool_scalar(
    estimates=[.38,.42,.44,.41,.40],
    variances=[.01,.011,.009,.012,.01]
)
kd.display_analysis_dashboard(
    pool,theme="amethyst",
    title_ar="دمج نتائج الاستكمال المتعدد",
    title_en="RUBIN POOLING · MULTIPLE IMPUTATION",
    font_size=22,
)
```

يُقدّم التقدير وفاصل الثقة والتباين بين الاستكمالات وداخلها. قيمة p لا تُختلق، ودرجات الحرية تعتمد على تقريب Rubin؛ لا تشمل هذه المرحلة تصحيح عينات صغيرة من نوع Barnard–Rubin.

## Arabic/English reporting | تصميم التقرير

تعمل المخرجات ضمن **Analytics Studio** بجداول خفيفة ومقروءة، مع عناوين عربية RTL يمينًا بخط واضح، والأسماء الإنجليزية LTR يسارًا، وتدرجات `aurora` و`ocean` و`amethyst`. التقارير قابلة للطباعة إلى PDF من HTML دون اتصال.

## Limitations / تنبيهات

- جميع الدوال تعتمد على بيانات رقمية؛ ينبغي معالجة متغيرات الفئات والترتيب والنسب المقيدة ضمن نموذج ملائم قبل الاستكمال.
- Simple IterativeImputer posterior draws are not a general-purpose fully specified joint Bayesian imputation model.
- MCAR/MAR/MNAR are assumptions about an observation mechanism, not labels determinable from one diagnostic.
- If data are clustered or longitudinal, independent-row asymptotics and bootstrap/imputation methods need study-specific modifications.
- This increment addresses a subset of the original **256-item academic methods catalogue**; other specialized missing-data and SEM methods remain outside its tested scope.

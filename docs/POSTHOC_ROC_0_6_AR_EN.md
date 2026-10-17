# KhaledDSML 0.6.0 — Post-hoc Comparisons & Paired ROC-AUC

**حالة الإصدار / Release status:** تطويري، لم يُنشر بعد / Unreleased.  
**Stable on PyPI:** `0.5.0`. **Research suite:** جامعة الوادي — Khaled Ali.

تهدف هذه الدفعة إلى إكمال اختبارات المقارنات البعدية وتقييم الفرق بين نموذجَي تصنيف على عينة الاختبار **نفسها**، مع إنتاج جداول علمية ثنائية اللغة ضمن Analytics Studio.

## جدول الوظائف الجديدة

| API | Purpose / الغرض | Study assumptions |
|---|---|---|
| `tukey_hsd_comparisons` | Tukey HSD لكل المجموعات | Independent, approximately normal errors and homoscedasticity |
| `games_howell_comparisons` | Games–Howell لتباينات غير متجانسة | Independent groups; within-group normality; Welch Satterthwaite |
| `dunn_posthoc` | Dunn بعد Kruskal–Wallis | Independent groups, tie correction, Holm/Bonferroni |
| `dunnett_control_comparisons` | Dunnett مقارنة علاجات بمجموعة ضابطة | Independent normal groups with common variance |
| `nemenyi_friedman_posthoc` | Nemenyi بعد Friedman لقياسات مترابطة | Matched blocks, no ties; asymptotic studentized-range |
| `delong_auc_comparison` | DeLong للمقارنة بين ROC-AUC مترابطتين | Scores on the **same independent subjects**, held-out evaluation |
| `stratified_paired_auc_bootstrap` | فاصل ثقة Bootstrap لفرق AUC | Same cases, stratified paired resampling; **no invented p-value** |

كل اختبار يُرجع كائن `dict` يحتوي على `interpretation_ar` و`interpretation_en`، و`table` عند المقارنات المتعددة، مع احترام التفسير من اليمين إلى اليسار للعربية ومن اليسار إلى اليمين للإنجليزية.

## التثبيت التجريبي

```python
%pip install --force-reinstall --no-cache-dir --no-deps "git+https://github.com/noble2013/KhaledDSML.git@feature/v0.6-posthoc-roc-comparison"
```

أعد تشغيل الجلسة بعد التثبيت (**Runtime → Restart session**). لا تُثبت `0.6.0` من PyPI قبل إعلان نشرها الرسمي.

## 1. المقارنات البعدية والتحليل المرئي

```python
import numpy as np
from khaled_dsml import (
    games_howell_comparisons, tukey_hsd_comparisons,
    dunn_posthoc, display_analysis_dashboard,
)

rng=np.random.default_rng(12)
groups={
    "الوادي":rng.normal(65,8,30),
    "الجزائر":rng.normal(74,12,32),
    "بسكرة":rng.normal(69,16,34),
}

result=games_howell_comparisons(groups)
display_analysis_dashboard(
    result, title_ar="مقارنة الأداء بين المجموعات",
    title_en="MULTIPLE COMPARISONS · GAMES–HOWELL",
    theme="aurora", font_size=22,
)
```

تظهر المقارنات المعتمدة على البيانات الحقيقية في جدول قابل للتمرير داخل الصفحة، مع قيمة p لكل مقارنة، وفواصل الثقة.

## 2. Dunnett مقابل الضبط

```python
from khaled_dsml import dunnett_control_comparisons, display_analysis_dashboard

result=dunnett_control_comparisons(groups,control="الوادي",seed=42)
display_analysis_dashboard(
    result,title_ar="مقارنة العلاجات بمجموعة ضابطة",
    title_en="DUNNETT · TREATMENT VS CONTROL",
    theme="ocean",
)
```

لا يجري Dunnett مقارنة كل علاج بكل علاج، بل يضبط تعدد المقارنات بالنسبة إلى مجموعة الضبط فقط.

## 3. اختبار DeLong لنموذجين على نفس الاختبار

```python
import numpy as np
from khaled_dsml import (
    delong_auc_comparison, stratified_paired_auc_bootstrap,
    display_analysis_dashboard,
)

rng=np.random.default_rng(7)
labels=np.array([0]*70+[1]*70)
model_a=np.r_[rng.normal(0,1,70),rng.normal(1.0,1,70)]
model_b=np.r_[rng.normal(0,1,70),rng.normal(.5,1,70)]

delong=delong_auc_comparison(labels,model_a,model_b)
bootstrap=stratified_paired_auc_bootstrap(
    labels,model_a,model_b,repetitions=500,seed=7
)

display_analysis_dashboard(
    [delong,bootstrap],
    title_ar="مقارنة دقة التمييز بين نموذجين",
    title_en="CORRELATED ROC AREAS · MODEL COMPARISON",
    theme="amethyst",
)
```

**تنبيهات علمية:** DeLong يعطي `p_value` تقريبيًا للفرق بين ROC-AUC المترابطتين فقط، ولا يثبت جودة المعايرة أو السببية. إذا كانت التنبؤات متطابقة وفرق AUC بلا تباين، **ترفض الدالة اختلاق p-value**. أما Bootstrap فيعيد فاصل ثقة فقط دون إحصائية دلالة مصطنعة.

## 4. المقارنات بين الاستقلال والترابط

Tukey، Games–Howell، Dunn، Dunnett مُخصصة لمجموعات **مستقلة**. أما Nemenyi–Friedman فهي للشروط **المتكررة لدى الأشخاص أنفسهم**. لا تستخدم اختبارات المجموعات المستقلة على بيانات القياسات المتكررة.

## 5. القيود والمراجعة البحثية

هذه المرحلة لا تدّعي استكمال الـ256 اختبارًا في الكتالوج. ما يزال العمل على بعض اختبارات CFA/SEM والبايزية والوساطة والمعايرة المتقدمة، بالإضافة إلى ضبط الوثائق مع بنود الكتالوج الأصلية. يجب الإبلاغ عن خطة المقارنات واختيار الاختبار قبل فحص النتائج، وتطبيق تصحيح تعدد المقارنات إن كان مناسبًا.

# KhaledDSML 0.8.0 — Statistical Mediation and Moderation | الوساطة والتعديل الإحصائي

**Status:** Development branch, unreleased. **Published stable:** `0.7.0`.  
**Author:** Dr. Khaled Ali — University of El Oued, Algeria.

## About this module

This academic release implements six regression-based research utilities for studies involving mediation and moderation. **Regression mediation is descriptive and associational**; causal mediation requires temporal ordering, identification assumptions, defensible confounder control, and an appropriate design. The user must prespecify an analysis protocol.

| Name | الغرض | Important assumption |
|---|---|---|
| `mediation_paths(x,m,y,controls=...)` | معاملات a وb وc وc′ والارتباط غير المباشر a×b | Aligned cases, full-rank linear OLS |
| `sobel_mediation_test(x,m,y)` | اختبار Sobel التقريبي للمنتج | Normal delta approximation; no claim of finite-sample exactness |
| `bootstrap_mediation_interval(...)` | فاصل Bootstrap المئيني المزدوج | Paired-case resampling; independent observational units |
| `moderation_interaction_test(x,w,y)` | اختبار معامل X×W وتغير R² | OLS interaction; HC3 optional; nested F assumes homoscedasticity |
| `simple_slopes_table(x,w,y)` | الميل المشروط عند مستويات W المختارة | Uses full coefficient covariance for SE and CI |
| `johnson_neyman_region(x,w,y)` | مجالات Johnson–Neyman ضمن نطاق W المرصود | Only statistical significance, not a causal region |

The **Analytics Studio** dashboard supports creative Aurora/Ocean/Amethyst themes, offline HTML export, bold RTL/right-aligned Arabic text, and LTR/left-aligned English. Result tables display a clearly styled Arabic title above the English column key, while numeric values remain readable.

## Experiment in Google Colab / التجربة

Install the stable dependencies, then the development branch (not released on PyPI yet):

```python
%pip install "khaled-dsml[econometrics]==0.7.0"
%pip install --no-deps --force-reinstall --no-cache-dir "git+https://github.com/noble2013/KhaledDSML.git@feature/v0.8-mediation-moderation-suite"
```

Restart session (**Runtime → Restart session**) to avoid importing an earlier version.

## Example 1 — الوساطة الإحصائية

```python
import numpy as np
import khaled_dsml as kd

rng=np.random.default_rng(42)
n=150
x=rng.normal(size=n)
covariate=rng.normal(size=n)
m=.6*x+.3*covariate+rng.normal(size=n)
y=.4*x+1.0*m+.2*covariate+rng.normal(size=n)

paths=kd.mediation_paths(x,m,y,controls=covariate)
sobel=kd.sobel_mediation_test(x,m,y,controls=covariate)
bootstrap=kd.bootstrap_mediation_interval(
    x,m,y,controls=covariate,repetitions=500,seed=42
)

kd.display_analysis_dashboard(
    [paths,sobel,bootstrap],
    title_ar="لوحة تحليل المسارات والوساطة",
    title_en="MEDIATION PATHS · INDIRECT ASSOCIATION",
    theme="aurora",font_size=22,
)
```

Sobel reports its approximate two-sided p-value; bootstrap reports a percentile interval **without inventing a hypothesis-test p-value**.

## Example 2 — التعديل والتفاعل بين المتغيرات

```python
rng=np.random.default_rng(7)
n=190
x=rng.normal(size=n)
w=rng.uniform(-2,2,size=n)
y=1+.5*x+.4*w+.7*x*w+rng.normal(scale=.65,size=n)

interaction=kd.moderation_interaction_test(x,w,y,robust="HC3")
slopes=kd.simple_slopes_table(x,w,y,moderator_values=[-1,0,1])
regions=kd.johnson_neyman_region(x,w,y)

kd.display_analysis_dashboard(
    [interaction,slopes,regions],
    title_ar="اختبار التعديل والميل المشروط",
    title_en="MODERATION · SIMPLE SLOPES · JOHNSON–NEYMAN",
    theme="amethyst",font_size=22,
)
```

The main interaction coefficient describes the estimated change in X's association with Y per 1-unit change in W. **Neither the conditional-slope p-value nor the Johnson–Neyman range establishes causality.** For values of W outside the observed range, do not extrapolate without domain justification.

## Export to a stylish offline HTML / حفظ التقرير

```python
kd.save_analysis_dashboard(
    [paths,sobel,bootstrap,interaction,slopes,regions],
    "KhaledDSML_Mediation_Moderation_AR_EN.html",
    title_ar="التقرير الجامعي المتكامل لتحليل الوساطة والتعديل",
    title_en="ACADEMIC PATH ANALYSIS · MEDIATION AND MODERATION",
    theme="ocean",
    font_size=22,
)
```

HTML output can be opened offline and printed to PDF from the browser. Computed numbers are preserved; report generation does not infer missing statistics.

## Scope and scientific limitations / حدود الاستخدام

- Models assume independent records and finite aligned quantitative inputs; listwise cleaning, imputation, causal adjustment and survey weights must be handled upstream with a documented methodology.
- Bootstrap percentile intervals are not BCa or exact significance tests. If observations are clustered, repeated, or longitudinal, simple row bootstrap may be invalid.
- Sobel uses an asymptotic normal approximation for a product often non-normal at modest sample sizes.
- Nested F for ΔR² is classical homoscedastic OLS. HC3 refers to interaction t, not the nested F.
- Johnson–Neyman boundaries are only reported inside the observed moderator range.
- The functionality is part of a larger **256-item academic catalog** and does not imply all catalog items are implemented.

# KhaledDSML 0.10 — Specialized inference, drift and editable reports
## الارتباطات المتخصصة ومراقبة التغير والتقارير الثنائية

**Development branch:** `feature/v0.10-specialized-inference-monitoring` • `0.10.0.dev0`. **Published PyPI stable remains 0.9.0** until an explicitly verified release.

### 1. Polychoric correlation | الارتباط الكامن للمتغيرات الرتبية

`polychoric_latent_correlation(x, y)` estimates the correlation of **two latent normally distributed variables** underlying observed ordered **integer-coded categories**. It estimates ordinal thresholds by cumulative sample marginal proportions, then maximizes a multinomial log likelihood whose cell probabilities come from a bivariate Gaussian distribution. Requires >=80 independent paired cases and a minimum of four observations per ordinal category. The return includes the estimated latent correlation and full contingency counts, **without a fabricated p-value or standard error**.

**العربية:** يتطلب فرض التوزيع الطبيعي الثنائي الكامن وصحة ترتيب رموز الفئات واستقلال الحالات. لا يُستنتج منه تأثير سببي. لا يُعد بديلًا آليًا لمقاييس الارتباط المتينة خارج هذا النموذج.

### 2. Williams–Steiger overlapping correlation test | اختبار ارتباطين تابعين

`williams_steiger_overlapping_correlations(x,y,z)` compares Pearson `r(x,y)` and `r(x,z)` **on the same complete iid sample**, considering `r(y,z)`. Returns Williams–Steiger t with `n−3` degrees of freedom, the 3×3 sample correlation matrix and an approximate parametric two-sided p-value. Not for independent samples or non-overlapping correlations.

**العربية:** يلزم استقلال ثلاثيات الملاحظات وملاءمة التوزيع الطبيعي الثلاثي ووجود مصفوفة ارتباط قابلة للتقدير. الفرق بين معاملين لا يثبت سببية أي منهما.

### 3. Conover–Iman | مقارنات كونوفِر–إيمان البعدية

`conover_iman_posthoc(group1,group2,group3,...)` implements pairwise pooled-rank differences after Kruskal–Wallis, using the Conover–Iman pooled-rank t reference and **Holm multiple-testing adjustment**. **This implementation deliberately rejects tied pooled ranks** rather than silently claiming a tie correction that has not been validated. At least three independent groups, each containing four or more finite observations.

**العربية:** لا تُدرج القيم المتساوية ضمن النسخة الحالية؛ استخدم طريقة خاصة بالروابط أو تحليلًا بديلًا عند وجودها، ولا تفسّر دلالة المقارنات كدليل سببي.

### 4. Effect size r | حجم الأثر

`effect_size_r_from_z(z,n)` returns a **signed** descriptive `r=z/√n` in settings where this approximation is scientifically meaningful (e.g. some rank tests). Rejects calculated magnitudes outside [-1,1]. **This is not a universal r from arbitrary statistics** and has no p-value.

### 5. PELT | كشف نقاط تغير المتوسط

`pelt_mean_shift(values,penalty=...,min_segment_size=5)` minimizes the exact piecewise-constant **within-segment sum of squares** plus `penalty × number_of_changes`. It uses a dynamic programming recurrence with PELT pruning delayed for the minimum segment size, and reports zero-based **start indexes of new segments**. Cost is SSE, not distribution-free; penalty is user-chosen, and the pruning is exact for this objective under valid arithmetic. Does not promise worst-case linear runtime or calibrated hypothesis test p-values.

### 6. DDM | مراقبة أخطاء التنبؤ

`ddm_binary_error_monitor(errors,min_instances=30)` tracks online Bernoulli error rate `p` plus binomial SD `√(p(1-p)/n)`, storing a best reference after warmup, and flags warning/drift at 2/3 SD multiples. Resets after drift by default. **Heuristic drift alarms** lack global false-positive control; serial dependence and repeated monitoring can be misleading.

### 7. ADWIN-style | نافذة تكيفية بحدود هوفدينغ

`adwin_bounded_mean_monitor(values,delta=.01,min_subwindow=15)` checks admissible split points of a finite [0,1] bounded time stream, discards the older prefix at detected changes, and uses an explicit **finite-horizon conservative Hoeffding union bound**. **It is an uncompressed ADWIN-style implementation, not optimized bucket-compressed ADWIN2.** The bound requires iid observations under no-change conditions, a finite prespecified horizon and observed values truly bounded by [0,1].

### 8. Excel and Word exports | تصدير Excel وWord

Two real **editable** file exporters are provided. Install the optional dependencies:

```powershell
py -3.13 -m pip install -e ".[econometrics,exports,dev]"
```

Use:

```python
from khaled_dsml import save_analysis_xlsx, save_analysis_docx
from khaled_dsml import pelt_mean_shift
result = pelt_mean_shift([0.0]*30 + [4.0]*30, penalty=5)
save_analysis_xlsx(result, "analysis.xlsx")
save_analysis_docx(result, "analysis.docx")
```

Excel includes a **Summary** sheet and, if present, a **Data Table** sheet, with formatted headers and Arabic right-to-left summaries. Values beginning with `=, +, -, @` are escaped when passed as text, reducing spreadsheet formula-injection risk. Word includes Arabic RTL titles/interpretation, English LTR interpretation, a metric table and output-table evidence. The exported report reflects caller data, **does not sign, verify or recalculate any study**. Both methods write files on the caller's own machine, without network upload.

### Scientific quality gate | الفحص المستقل

`tests/test_specialized_inference_monitoring_exports.py` verifies:
- Williams–Steiger explicit t formula and SciPy t reference.
- Conover–Iman independent pooled-rank computation and Holm-adjusted p-values.
- Polychoric recovery from simulated latent Gaussian ordinal measurements and rejection of invalid codes.
- PELT changepoints and objective against an **unpruned independent dynamic-programming oracle** across several randomized signals and segment-size constraints.
- DDM and bounded-window shift/no-shift synthetic behavior.
- Real XLSX roundtrip with formula-trigger text **as plain text**, and DOCX OOXML presence of Arabic bidi formatting, English text and tables.
- Offline bilingual Analytics Studio escaping.

**Catalogue source-level mapping:** 245 out of 256 records now have callable API candidates; 11 have no mapping. This figure **does not certify scientifically complete implementations**. Those still unlinked include generic beyond-Beta-binomial Bayes factors, ill-advised observed post-hoc power, zero-inflated-model Vuong, structural-equation SEM indices/invariance, SHAP stability, Phillips–Perron, t-digest, GRIMMER, SPRITE and p-curve.

### Release integrity | حالة النشر

Current `0.10.0.dev0` source contains developing features. Do not call it a released stable `0.10.0` or PyPI 0.10 without publishing checks, changelog audit, source code licensing and independent scientific validation of the desired methods.

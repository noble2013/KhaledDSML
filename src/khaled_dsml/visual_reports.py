"""Beautiful, accessible bilingual analysis dashboards for Jupyter and Colab.

Rendered offline with scoped CSS and optional inline SVG; no external fonts,
scripts, internet access, or third-party visualization dependencies are needed.
Values displayed here come only from the caller's statistical results.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from html import escape
import math
import pandas as pd
from pathlib import Path

from ._results import TestResult


_METRICS = (
    ("posterior_mean", "المتوسط البعدي", "Posterior mean"),
    ("posterior_variance", "التباين البعدي", "Posterior variance"),
    ("credible_lower", "حد المصداقية الأدنى", "Credible lower"),
    ("credible_upper", "حد المصداقية الأعلى", "Credible upper"),
    ("predictive_mean_successes", "المتوسط التنبؤي للنجاحات", "Predictive successes"),
    ("probability_a_greater_b_given_h1", "احتمال تفوق المجموعة أ وفق النموذج البديل", "Pr(A > B | H1)"),
    ("log_bayes_factor_10", "لوغاريتم عامل بايز BF10", "Log Bayes factor 10"),
    ("bayes_factor_10", "عامل بايز BF10", "Bayes factor 10"),
    ("posterior_mean_a", "المتوسط البعدي للمجموعة أ", "Group A posterior"),
    ("posterior_mean_b", "المتوسط البعدي للمجموعة ب", "Group B posterior"),
    ("successes", "حالات النجاح", "Successes"),
    ("trials", "عدد المحاولات", "Trials"),
    ("predictive_trials", "المحاولات المستقبلية", "Future trials"),
    ("n_reference", "حجم العينة المرجعية", "Reference rows"),
    ("n_current", "حجم العينة الحالية", "Current rows"),
    ("n_bins", "فئات المقارنة", "Bins"),
    ("population_stability_index", "مؤشر ثبات المجتمع PSI", "Population stability (PSI)"),
    ("jensen_shannon_divergence_bits", "تباعد جنسن–شانون (بت)", "Jensen–Shannon (bits)"),
    ("wasserstein_distance", "مسافة فاسرشتاين", "Wasserstein distance"),
    ("ks_statistic", "إحصائية كولموغوروف–سميرنوف", "KS statistic"),
    ("ks_p_value", "قيمة p لاختبار KS", "KS p-value"),
    ("variance_sample", "تباين العينة", "Sample variance"),
    ("std_sample", "الانحراف المعياري للعينة", "Sample std."),
    ("variance_population", "تباين المجتمع", "Population variance"),
    ("n_studies", "عدد الدراسات", "Studies"),
    ("Q", "إحصائية كوكْران Q", "Cochran Q"),
    ("I_squared_percent", "مؤشر عدم التجانس I² %", "I² (%)"),
    ("tau_squared_dl", "تباين التأثيرات بين الدراسات", "DL tau²"),
    ("fixed_effect_estimate", "تقدير التأثير الثابت", "Fixed-effect estimate"),
    ("fixed_effect_standard_error", "خطأ التأثير الثابت", "Fixed-effect SE"),
    ("Q_p_value", "قيمة p لعدم التجانس", "Heterogeneity p-value"),
    ("intercept", "مقطع انحدار إيغر", "Egger intercept"),
    ("intercept_standard_error", "الخطأ المعياري لمقطع إيغر", "Intercept SE"),
    ("precision_slope", "ميل الدقة في انحدار إيغر", "Precision slope"),
    ("n_discordant", "الحالات المختلفة بين النموذجين", "Discordant predictions"),
    ("both_correct", "تنبؤات صحيحة للنموذجين", "Both correct"),
    ("both_incorrect", "تنبؤات خاطئة للنموذجين", "Both incorrect"),
    ("only_a_correct", "صحيحة عند أ فقط", "Only A correct"),
    ("only_b_correct", "صحيحة عند ب فقط", "Only B correct"),
    ("accuracy_a", "دقة النموذج أ", "Model A accuracy"),
    ("accuracy_b", "دقة النموذج ب", "Model B accuracy"),
    ("mape_percent", "متوسط خطأ MAPE %", "MAPE (%)"),
    ("median_ape_percent", "وسيط الخطأ النسبي %", "Median APE (%)"),
    ("n_rules", "عدد قواعد الجودة", "Contract rules"),
    ("n_failed_rules", "القواعد المخالفة", "Failed rules"),
    ("n_alarms", "عدد إنذارات التغيّر", "Change alarms"),
    ("n_rows", "عدد صفوف البيانات", "Data rows"),
    ("n_columns", "عدد أعمدة البيانات", "Data columns"),
    ("n_features", "عدد السمات", "Features"),
    ("outer_cv", "طيات التحقق الخارجية", "Outer folds"),
    ("inner_cv", "طيات التحقق الداخلية", "Inner folds"),
    ("outer_score_mean", "متوسط أداء الطيات الخارجية", "Outer CV score mean"),
    ("outer_score_std", "تشتت أداء الطيات الخارجية", "Outer CV score SD"),
    ("observed_cv_score", "الأداء المرصود", "Observed CV score"),
    ("null_score_mean", "متوسط أداء تبديل الملصقات", "Permutation-null score"),
    ("monte_carlo_resolution", "دقة خطوة مونت كارلو", "Monte Carlo resolution"),
    ("baseline_holdout_score", "الأداء الأساسي على الاختبار", "Holdout baseline score"),
    ("top_mean_score_decrease", "أعلى انخفاض في الأداء", "Top feature importance"),
    ("last_validation_score", "آخر أداء للتحقق", "Last validation score"),
    ("n_train_sizes", "أحجام التدريب المختبرة", "Training sizes"),
    ("mean_cv_score", "متوسط خط الأساس", "Dummy CV score mean"),
    ("sd_cv_score", "تشتت خط الأساس", "Dummy CV score SD"),
    ("n_samples", "حجم العينة", "Sample size"),
    ("n_resamples", "محاكاة مونت كارلو", "Monte Carlo replicates"),
    ("monte_carlo_exceedances", "محاكاة تجاوزت الإحصائية", "Null exceedances"),
    ("beta", "معلمة التنعيم", "HZ smoothing beta"),
    ("n_scenarios", "سيناريوهات الحساسية", "Sensitivity scenarios"),
    ("n_missing", "القيم المستكملة محل الدراسة", "Imputed target cells"),
    ("baseline_mean", "المتوسط المرجعي دون إزاحة", "Mean at zero shift"),
    ("n_patterns", "أنماط الفقد", "Missingness patterns"),
    ("small_patterns", "أنماط بعينات صغيرة", "Sparse patterns"),
    ("em_iterations", "تكرارات خوارزمية EM", "EM iterations"),
    ("degrees_of_freedom", "درجات الحرية", "Degrees of freedom"),
    ("tested_pairs", "الأزواج المختبرة", "Tested pairs"),
    ("n_imputations", "مرات الاستكمال", "Imputations"),
    ("n_train", "حجم التدريب", "Training rows"),
    ("n_test", "حجم الاختبار", "Testing rows"),
    ("estimate", "المعامل المجمّع", "Pooled estimate"),
    ("standard_error", "الخطأ المعياري", "Std. error"),
    ("within_variance", "التباين داخل الاستكمال", "Within variance"),
    ("between_variance", "التباين بين الاستكمالات", "Between variance"),
    ("total_variance", "إجمالي التباين", "Total variance"),

    ("n_covariates", "عدد متغيرات الضبط", "Covariates"),
    ("a_path", "معامل المسار أ", "Path a"),
    ("b_path", "معامل المسار ب", "Path b"),
    ("direct_effect", "الأثر المباشر", "Direct association"),
    ("indirect_effect", "الأثر غير المباشر", "Indirect association"),
    ("total_effect", "الأثر الكلي", "Total association"),
    ("indirect_se", "الخطأ المعياري للوساطة", "Indirect SE"),
    ("interaction_coefficient", "معامل التفاعل", "Interaction coefficient"),
    ("delta_r2", "الزيادة في معامل التحديد", "Delta R²"),
    ("r2_main_effects", "تحديد النموذج الأساسي", "Main-effects R²"),
    ("r2_interaction", "تحديد نموذج التفاعل", "Interaction R²"),
    ("nested_f", "إحصائية مقارنة النموذجين", "Nested F"),
    ("nested_f_p_value", "دلالة مقارنة النموذجين", "Nested F p-value"),
    ("moderator_min", "أدنى مستوى للمعدل", "Observed W minimum"),
    ("moderator_max", "أعلى مستوى للمعدل", "Observed W maximum"),

    ("n_variables", "عدد المتغيرات", "Variables"),
    ("n_items", "عدد البنود", "Items"),
    ("n_targets", "عدد الحالات", "Targets"),
    ("n_raters", "عدد المحكّمين", "Raters"),
    ("n_pairable_units", "وحدات بتقييمات متكررة", "Pairable units"),
    ("tokens", "عدد الكلمات", "Tokens"),
    ("types", "المفردات الفريدة", "Word types"),
    ("unique_types", "تنوع المفردات", "Unique types"),
    ("omega_total_one_factor", "أوميغا لعامل واحد", "One-factor omega"),
    ("krippendorff_alpha", "ألفا كريبندورف", "Krippendorff α"),
    ("icc", "الاتفاق داخل الفئات", "ICC"),
    ("trustworthiness", "موثوقية التضمين", "Trustworthiness"),
    ("mean_ari", "متوسط استقرار العناقيد", "Mean bootstrap ARI"),
    ("sd_ari", "تشتت استقرار العناقيد", "Bootstrap ARI SD"),
    ("ttr", "التنوع المعجمي", "Type-token ratio"),
    ("shannon_entropy_bits", "إنتروبيا الكلمات", "Entropy (bits)"),
    ("slope", "ميل قانون زيف", "Zipf slope"),
    ("r_squared", "معامل تفسير زيف", "Zipf R²"),
    ("skewness_p_value", "دلالة الالتواء", "Mardia skewness p"),
    ("kurtosis_p_value", "دلالة التفلطح", "Mardia kurtosis p"),
    ("W", "إحصائية الكروية", "Mauchly W"),
    ("difference", "فرق التقديرات", "Difference"),
    ("equivalence_margin", "هامش التكافؤ", "Equivalence margin"),
    ("score_difference", "فرق أداء النموذجين", "Score difference"),
    ("power", "قوة الاختبار", "Power"),
    ("n_per_group", "العينة لكل مجموعة", "N per group"),

    ("n_valid", "مشاهدات صالحة", "Valid observations"),
    ("mean", "المتوسط", "Mean"),
    ("median", "الوسيط", "Median"),
    ("std", "الانحراف المعياري", "Std. deviation"),
    ("statistic", "إحصائية الاختبار", "Test statistic"),
    ("p_value", "القيمة الاحتمالية", "p-value"),
    ("alpha", "مستوى الدلالة", "Significance"),
    ("roc_auc", "المساحة تحت ROC", "ROC-AUC"),
    ("roc_auc_a", "مساحة ROC للنموذج الأول", "ROC-AUC A"),
    ("roc_auc_b", "مساحة ROC للنموذج الثاني", "ROC-AUC B"),
    ("auc_difference", "فرق المساحة تحت ROC", "ROC-AUC difference"),
    ("variance_difference", "تباين فرق AUC", "AUC diff. variance"),
    ("significant_comparisons", "مقارنات دالة إحصائيًا", "Significant pairs"),
    ("n_groups", "عدد المجموعات", "Number of groups"),

    ("balanced_accuracy", "الدقة المتوازنة", "Balanced accuracy"),
    ("mcc", "معامل ماثيوز", "MCC"),
    ("cohen_kappa", "اتفاق كابا", "Cohen κ"),
    ("pr_auc", "المساحة تحت PR", "PR-AUC"),
    ("average_precision", "متوسط الدقة", "Avg. precision"),
    ("log_loss", "خسارة لوغاريتمية", "Log loss"),
    ("brier", "مقياس براير", "Brier"),
    ("ece", "خطأ المعايرة", "ECE"),
    ("r2", "معامل التحديد", "R²"),
    ("adjusted_r2", "التحديد المعدل", "Adj. R²"),
    ("kappa", "معامل الاتفاق", "Kappa"),
    ("htmt", "نسبة HTMT", "HTMT"),
    ("ave", "متوسط التباين المستخلص", "AVE"),
    ("composite_reliability", "الثبات المركب", "Composite reliability"),
    ("odds_ratio", "نسبة الأرجحية", "Odds ratio"),
    ("relative_risk", "الخطر النسبي", "Relative risk"),
    ("proportion", "النسبة", "Proportion"),
    ("lower", "الحد الأدنى", "Lower bound"),
    ("upper", "الحد الأعلى", "Upper bound"),
    ("pearson_dispersion", "تشتت بيرسون", "Pearson dispersion"),
    ("calinski_harabasz", "مؤشر كالينسكي", "Calinski-Harabasz"),

    ("accuracy", "الدقة", "Accuracy"),
    ("precision", "الإحكام", "Precision"),
    ("recall", "الاسترجاع", "Recall"),
    ("f1", "مؤشر F1", "F1 score"),
    ("mae", "الخطأ المطلق", "MAE"),
    ("rmse", "جذر مربع الخطأ", "RMSE"),
    ("mase", "الخطأ المُقاس", "MASE"),
    ("smape_percent", "الخطأ النسبي", "sMAPE %"),
    ("change_index", "موضع التغير (0-based)", "Change index"),
    ("n_rows", "عدد الصفوف", "Rows"),
    ("n_columns", "عدد الأعمدة", "Columns"),
    ("duplicate_rows", "الصفوف المكررة", "Duplicate rows"),
    ("missing_cells", "الخلايا المفقودة", "Missing cells"),
)
_TITLES = {
    "Numeric distribution drift profile": ("ملف انجراف التوزيع الرقمي", "Numeric distribution drift"),
    "Welford streaming summary": ("ملخص ولفورد المتدفق", "Welford online summary"),
    "Sample Ratio Mismatch (SRM)": ("اختبار عدم تطابق توزيع المجموعات", "Sample Ratio Mismatch"),
    "Cochran Q and I-squared heterogeneity": ("اختبار عدم التجانس Q وI²", "Cochran Q and I²"),
    "DerSimonian-Laird random-effects meta-analysis": ("التحليل البعدي للتأثيرات العشوائية", "DerSimonian–Laird random effects"),
    "Egger small-study effects regression": ("انحدار إيغر لتأثيرات الدراسات الصغيرة", "Egger small-study regression"),
    "McNemar paired prediction comparison": ("اختبار ماكنمار للمقارنة المزدوجة", "Paired McNemar test"),
    "Strict MAPE forecast accuracy": ("متوسط نسبة الخطأ المطلق الصارم", "Strict MAPE"),
    "GRIM integer-mean arithmetic consistency": ("فحص GRIM لاتساق المتوسط", "GRIM arithmetic consistency"),
    "Student t reported-p recalculation": ("إعادة حساب قيمة p لاختبار t", "Reported t p-value audit"),
    "DataFrame schema contract validation": ("فحص عقد بنية وجودة البيانات", "Data schema contract validation"),
    "Page-Hinkley sequential change monitor": ("مراقبة تغيّر بيج–هينكلي المتتابعة", "Page–Hinkley monitoring"),
    "Nested cross-validation evaluation": ("التقييم المتداخل للتحقق المتقاطع", "Nested cross-validation"),
    "Permutation CV score significance": ("اختبار دلالة الأداء بتبديل الملصقات", "Permutation score significance"),
    "Holdout permutation feature importance": ("أهمية السمات بالتبديل على الاختبار المستقل", "Holdout permutation importance"),
    "Cross-validation learning curve": ("منحنى التعلّم والتحقق المتقاطع", "Cross-validation learning curve"),
    "Cross-validation dummy baseline": ("خط الأساس الساذج والتحقق المتقاطع", "Dummy CV baseline"),
    "Bayesian Beta-binomial posterior": ("التوزيع البعدي بايز-ثنائي الحدين", "Bayesian Beta–binomial posterior"),
    "Bayesian independent-binomial model comparison": ("مقارنة بايزية لمعدلات المجموعتين", "Bayesian comparison of binomial groups"),
    "Henze-Zirkler multivariate normality (Monte Carlo)": ("اختبار هينز–زيركلر للاعتدالية متعددة المتغيرات", "Henze–Zirkler multivariate normality"),
    "MNAR delta-shift sensitivity (descriptive)": ("تحليل حساسية دلتا للبيانات المفقودة", "MNAR delta-shift sensitivity"),
    "Mann-Kendall trend (normal approximation)": ("اختبار اتجاه مان–كندال", "Mann–Kendall trend"),
    "Pettitt single change-point": ("اختبار نقطة التغيّر", "Pettitt change point"),
    "Shapiro-Wilk": ("اختبار شابيرو–ويلك", "Shapiro–Wilk"),
    "Box-Pierce": ("اختبار الارتباط الذاتي للبواقي", "Box–Pierce residual test"),
    "Engle ARCH-LM": ("اختبار تغير التباين الشرطي", "Engle ARCH-LM"),
}
_PALETTES = {
    "aurora": ("#11263b", "#243f60", "#4bd2c5", "#d6faf4", "#edf9f8"),
    "ocean": ("#123158", "#215584", "#72c4ff", "#e0f3ff", "#edf6ff"),
    "amethyst": ("#302351", "#5a3e86", "#d6aeff", "#f3e8ff", "#f8f5ff"),
}


def _entries(results) -> list:
    if isinstance(results, (TestResult, Mapping)):
        return [results]
    if isinstance(results, (str, bytes)) or not isinstance(results, Iterable):
        raise TypeError("Expected a TestResult, a result dictionary, or an iterable of results")
    items = list(results)
    if not items:
        raise ValueError("Provide at least one analysis result")
    return items


def _parse_result(item) -> tuple[str, str, str, str, dict]:
    if isinstance(item, TestResult):
        ar, en = item.interpretation_ar, item.interpretation_en
        values = {
            "statistic": item.statistic, "p_value": item.p_value,
            "alpha": item.alpha, "n_samples": item.sample_size,
            "reject_null": item.reject_null,
            "notes": item.notes, "notes_en": item.notes_en,
        }
        label = item.test
    elif isinstance(item, Mapping):
        values = dict(item)
        ar = values.get("interpretation_ar", values.get("ar"))
        en = values.get("interpretation_en", values.get("en"))
        if (not ar or not en) and "mean" in values and "n_valid" in values:
            from .descriptive import interpret_descriptive
            interpretation = interpret_descriptive(values)
            ar, en = interpretation["ar"], interpretation["en"]
        elif (not ar or not en) and "roc_auc" in values and "f1" in values:
            from .ml_evaluation import interpret_classification_metrics
            interpretation = interpret_classification_metrics(values)
            ar, en = interpretation["ar"], interpretation["en"]
        label = str(values.get("test", values.get("name", "Statistical analysis")))
    else:
        raise TypeError("Each result must be a TestResult or dictionary")
    if not isinstance(ar, str) or not ar.strip():
        raise ValueError("Result must have a nonempty Arabic interpretation")
    if not isinstance(en, str) or not en.strip():
        raise ValueError("Result must have a nonempty English interpretation")
    ar_title, en_title = _TITLES.get(label, (label, label))
    return ar_title, en_title, ar, en, values


def _display_number(key, value) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        try:
            # NumPy numerical scalars support float() without being built-in types.
            import numbers
            if not isinstance(value, numbers.Real):
                return None
        except ImportError:
            return None
    number = float(value)
    if not math.isfinite(number):
        return "—"
    if key in {"n_samples", "n_valid", "n_rows", "n_columns",
               "duplicate_rows", "missing_cells", "change_index"}:
        return str(int(number))
    if key == "p_value" and number < 0.0001:
        return "p < 0.0001"
    if key == "smape_percent":
        return f"{number:.2f}%"
    return f"{number:,.4f}".rstrip("0").rstrip(".") if abs(number) < 10000 else f"{number:,.1f}"


def _metrics_block(values: Mapping, n: int) -> str:
    tiles = []
    for key, ar, en in _METRICS:
        if key not in values:
            continue
        display = _display_number(key, values[key])
        if display is None:
            continue
        tiles.append(
            f'<div class="kdsml-metric"><div class="kdsml-metric-top">'
            f'<span dir="rtl" lang="ar">{escape(ar)}</span>'
            f'<span dir="ltr" lang="en">{escape(en)}</span></div>'
            f'<strong dir="ltr">{escape(display)}</strong></div>'
        )
        if len(tiles) >= 6:
            break
    return (f'<div class="kdsml-metrics">{"".join(tiles)}</div>' if tiles else "")


def _sparkline(values: Sequence[float] | None) -> str:
    if values is None:
        return ""
    if isinstance(values, (str, bytes)):
        raise TypeError("chart_data requires numeric observations")
    try:
        v = [float(a) for a in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("chart_data must consist of numeric observations") from exc
    if len(v) < 2 or any(not math.isfinite(x) for x in v):
        raise ValueError("chart_data needs two or more finite observations")
    # A deterministic downsample prevents giant HTML for high-frequency datasets.
    if len(v) > 120:
        v = [v[round(i * (len(v) - 1) / 119)] for i in range(120)]
    minimum, maximum = min(v), max(v)
    spread = maximum - minimum
    w, h, pad = 620, 112, 12
    points = []
    for index, value in enumerate(v):
        x = pad + index * (w - 2 * pad) / (len(v) - 1)
        y = h / 2 if spread == 0 else h - pad - (value - minimum) * (h - 2 * pad) / spread
        points.append(f"{x:.1f},{y:.1f}")
    line = " ".join(points)
    return (
        '<figure class="kdsml-chart">'
        '<figcaption><span lang="ar" dir="rtl">مسار القيم المرصودة</span>'
        '<span lang="en" dir="ltr">Observed series</span></figcaption>'
        '<svg viewBox="0 0 620 112" role="img" '
        'aria-label="Plot of observed values in their given order" '
        'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">'
        '<line x1="0" y1="100" x2="620" y2="100" stroke="#d9e5e9" stroke-width="1"/>'
        f'<polyline points="{line}" fill="none" stroke="var(--kdsml-accent)" '
        'stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg><div class="kdsml-chart-range" dir="ltr">'
        f'<span>Min {minimum:,.3g}</span><span>Max {maximum:,.3g}</span>'
        '</div></figure>'
    )


_TABLE_HEADINGS_AR = {
    "moderator_value": "قيمة المتغير المعدّل",
    "moderator_lower": "بداية مجال المعدّل",
    "moderator_upper": "نهاية مجال المعدّل",
    "slope": "الميل المشروط",
    "standard_error": "الخطأ المعياري",
    "ci_lower": "الحد الأدنى لفاصل الثقة",
    "ci_upper": "الحد الأعلى لفاصل الثقة",
    "p_value": "القيمة الاحتمالية",
    "reject_null": "رفض فرضية العدم",
    "slope_significant": "دلالة الميل",
    "group_a": "المجموعة الأولى",
    "group_b": "المجموعة الثانية",
    "mean_difference": "فرق المتوسطين",
    "n_samples": "حجم العينة",
    "count": "العدد",
    "missing_count": "عدد القيم المفقودة",
    "mean_predicted": "متوسط الاحتمال المتوقع",
    "fraction_positive": "نسبة الفئة الموجبة",
}


def _table_block(values: Mapping) -> str:
    """Inline responsive preview of an actual result DataFrame, never invented rows.

    Large tables are truncated to a safe 25x8 preview. Results must contain a
    pandas DataFrame under 'table', 'bins' or 'group_rates' for this to appear.
    """
    for key in ("table", "bins", "group_rates"):
        frame = values.get(key)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        selected = frame.iloc[:25, :8]
        headers = "".join(
            '<th scope="col">'
            + (f'<span class="kdsml-th-ar" lang="ar" dir="rtl">'
               f'{escape(_TABLE_HEADINGS_AR[str(c)])}</span>'
               if str(c) in _TABLE_HEADINGS_AR else "")
            + f'<span class="kdsml-th-en" lang="en" dir="ltr">{escape(str(c))}</span>'
            + '</th>'
            for c in selected.columns
        )
        rows = []
        for _, row in selected.iterrows():
            cols = []
            for value in row:
                if pd.isna(value):
                    text = "—"
                elif isinstance(value, (float, int)) and not isinstance(value, bool):
                    text = f"{value:.4g}"
                else:
                    text = str(value)
                cols.append(f'<td dir="auto">{escape(text)}</td>')
            rows.append("<tr>" + "".join(cols) + "</tr>")
        truncated = len(frame)>25 or len(frame.columns)>8
        hint = ("<p class=\"kdsml-table-hint\" dir=\"ltr\">"
                "Preview truncated to 25 rows × 8 columns</p>" if truncated else "")
        return (
            '<section class="kdsml-table-panel">'
            '<h3 lang="ar" dir="rtl">جدول نتائج التحليل</h3>'
            '<p lang="en" dir="ltr">ANALYTICAL RESULTS TABLE</p>'
            '<div class="kdsml-table-scroll"><table dir="ltr">'
            f'<thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody>'
            f'</table></div>{hint}</section>'
        )
    return ""


def _render_item(item, index: int) -> str:
    ar_title, en_title, ar, en, values = _parse_result(item)
    decision = values.get("reject_null")
    if isinstance(decision, bool):
        cls = "rejected" if decision else "not-rejected"
        status_ar = "رفض الفرضية الصفرية" if decision else "عدم رفض الفرضية الصفرية"
        status_en = "Reject H₀" if decision else "Do not reject H₀"
        pill = (f'<span class="kdsml-decision {cls}">'
                f'<span dir="rtl" lang="ar">{status_ar}</span> · '
                f'<span dir="ltr" lang="en">{status_en}</span></span>')
    else:
        pill = ""
    metric_html = _metrics_block(values, index)
    table_html = _table_block(values)
    notes_ar = values.get("notes", "")
    notes_en = values.get("notes_en", "")
    notes = ""
    if notes_ar or notes_en:
        parts = []
        if notes_ar:
            parts.append(f'<p dir="rtl" lang="ar">{escape(str(notes_ar))}</p>')
        if notes_en:
            parts.append(f'<p dir="ltr" lang="en">{escape(str(notes_en))}</p>')
        notes = '<aside class="kdsml-notes">' + "".join(parts) + '</aside>'
    return (
        '<article class="kdsml-result">'
        '<div class="kdsml-result-heading"><span class="kdsml-number" dir="ltr">'
        f'{index:02d}</span><div class="kdsml-result-names">'
        f'<h2 dir="rtl" lang="ar">{escape(ar_title)}</h2>'
        f'<p dir="ltr" lang="en">{escape(en_title)}</p></div>{pill}</div>'
        + metric_html +
        '<div class="kdsml-explanations"><div class="kdsml-ar" dir="rtl" lang="ar">'
        '<span class="kdsml-lang">الشرح والتفسير</span>'
        f'<p>{escape(ar)}</p></div>'
        '<div class="kdsml-en" dir="ltr" lang="en">'
        '<span class="kdsml-lang">Explanation & interpretation</span>'
        f'<p>{escape(en)}</p></div></div>{table_html}{notes}</article>'
    )


_CSS = """
.kdsml-studio{--kdsml-ink:#11263b;--kdsml-shadow:rgba(23,51,77,.09);
 --kdsml-accent:#24b5ab;--kdsml-soft:#eaf9f6;--kdsml-title:#162f49;
 font-family:Tahoma,"Segoe UI",Arial,sans-serif;max-width:1100px;margin:12px auto;
 padding:20px;color:#24374d;background:#f6f9fc;direction:rtl;text-align:right;
 border:1px solid #e4eaf0;border-radius:22px;box-sizing:border-box}
.kdsml-studio *{box-sizing:border-box}
.kdsml-studio .kdsml-hero{position:relative;overflow:hidden;border-radius:19px;
 padding:30px 30px 28px;background:linear-gradient(123deg,var(--kdsml-ink),var(--kdsml-mid));
 color:white;min-height:180px;isolation:isolate}
.kdsml-studio .kdsml-hero:before{content:"";position:absolute;width:290px;height:290px;
 border:1px solid rgba(255,255,255,.13);border-radius:50%;left:-80px;top:-100px;z-index:-1;
 box-shadow:0 0 0 36px rgba(255,255,255,.035),0 0 0 81px rgba(255,255,255,.025)}
.kdsml-studio .kdsml-brand{font:700 12px Arial,sans-serif;letter-spacing:.19em;
 text-transform:uppercase;color:var(--kdsml-accent-light);direction:ltr;text-align:left}
.kdsml-studio .kdsml-hero h1{font-size:clamp(25px,4vw,36px);font-weight:800;
 margin:13px 0 4px;line-height:1.55;direction:rtl;text-align:right;color:white}
.kdsml-studio .kdsml-hero p{margin:0;font-size:16px;font-weight:600;line-height:1.75;
 text-align:left;direction:ltr;color:#e3edf7}
.kdsml-studio .kdsml-hero-foot{display:flex;gap:10px;flex-wrap:wrap;
 margin-top:19px;justify-content:flex-start}
.kdsml-studio .kdsml-chip{border:1px solid rgba(255,255,255,.3);border-radius:999px;
 padding:6px 12px;font:600 12px Arial,sans-serif;background:rgba(255,255,255,.085)}
.kdsml-studio .kdsml-result{margin-top:18px;padding:24px 25px 22px;
 background:#fff;border:1px solid #e3eaf0;border-radius:18px;
 box-shadow:0 6px 21px var(--kdsml-shadow);break-inside:avoid}
.kdsml-studio .kdsml-result-heading{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.kdsml-studio .kdsml-number{display:inline-grid;place-items:center;flex:0 0 43px;
 height:43px;background:var(--kdsml-soft);border-radius:13px;color:var(--kdsml-ink);
 font:800 15px Arial,sans-serif}
.kdsml-studio .kdsml-result-names{flex:1;min-width:165px}
.kdsml-studio .kdsml-result-names h2{font-size:21px;font-weight:800;
 color:var(--kdsml-title);line-height:1.5;margin:0;text-align:right}
.kdsml-studio .kdsml-result-names p{font:600 13px Arial,sans-serif;
 color:#63758b;margin:3px 0 0;text-align:left}
.kdsml-studio .kdsml-decision{font:700 12px Tahoma,Arial,sans-serif;
 background:#ebf7f3;border:1px solid #b2e1d4;border-radius:999px;
 color:#146647;padding:8px 12px;max-width:100%;line-height:1.7}
.kdsml-studio .kdsml-decision.not-rejected{color:#815318;
 background:#fff7ea;border-color:#f4d5a4}
.kdsml-studio .kdsml-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
 gap:11px;margin:23px 0}
.kdsml-studio .kdsml-metric{border:1px solid #e1e9f0;background:#f8fbfe;
 border-radius:13px;padding:15px 17px;min-width:0}
.kdsml-studio .kdsml-metric-top{display:flex;justify-content:space-between;
 align-items:flex-start;gap:6px;flex-wrap:wrap}
.kdsml-studio .kdsml-metric-top span:first-child{font:700 13px Tahoma,Arial,sans-serif;color:#445d76}
.kdsml-studio .kdsml-metric-top span:last-child{font:500 10px Arial,sans-serif;
 color:#728397}
.kdsml-studio .kdsml-metric strong{display:block;text-align:right;font:800 24px Arial,sans-serif;
 letter-spacing:.015em;color:#17314c;margin-top:9px;overflow-wrap:anywhere}
.kdsml-studio .kdsml-explanations{display:grid;grid-template-columns:minmax(0,1fr);
 gap:14px;border-top:1px solid #e3e9ef;padding-top:21px}
.kdsml-studio .kdsml-ar{direction:rtl !important;text-align:right !important;
 background:linear-gradient(95deg,var(--kdsml-soft),#fff);
 padding:18px 21px;border-right:4px solid var(--kdsml-accent);border-radius:12px}
.kdsml-studio .kdsml-en{direction:ltr !important;text-align:left !important;
 background:#f6f8fb;padding:18px 21px;border-left:4px solid #8aa1bc;border-radius:12px}
.kdsml-studio .kdsml-lang{display:block;font-size:12px;font-weight:700;
 color:#617790;letter-spacing:.03em;margin-bottom:9px}
.kdsml-studio .kdsml-ar p{font-family:Tahoma,Arial,sans-serif;font-size:var(--kdsml-text-size);
 font-weight:700;line-height:2;margin:0;text-align:right !important;
 overflow-wrap:anywhere;white-space:pre-wrap;color:#24374b}
.kdsml-studio .kdsml-en p{font-family:Arial,Helvetica,sans-serif;
 font-size:var(--kdsml-text-size);font-weight:650;line-height:1.8;
 margin:0;text-align:left !important;overflow-wrap:anywhere;
 white-space:pre-wrap;color:#293c52}
.kdsml-studio .kdsml-table-panel{margin-top:21px;padding:18px;
 border:1px solid #e2e8f0;border-radius:14px;background:#f9fcff}
.kdsml-studio .kdsml-table-panel h3{font-size:18px;font-weight:800;
 text-align:right;color:#244762;margin:0}
.kdsml-studio .kdsml-table-panel p{font-size:11px;font-weight:650;
 color:#6d8297;margin:4px 0 13px;text-align:left}
.kdsml-studio .kdsml-table-scroll{width:100%;overflow-x:auto;
 -webkit-overflow-scrolling:touch;scrollbar-color:#b5c5d6 transparent}
.kdsml-studio .kdsml-table-scroll table{border-collapse:collapse;
 width:100%;min-width:420px;background:#fff;text-align:left;
 font:600 13px Arial,sans-serif;color:#264056}
.kdsml-studio .kdsml-table-scroll th{padding:12px 13px;
 background:#eaf4f8;color:#274763;border-bottom:2px solid #b2d9dd}
.kdsml-studio .kdsml-table-scroll th .kdsml-th-ar{
 display:block;text-align:right;font:800 13px Tahoma,Arial,sans-serif;
 direction:rtl;white-space:normal;min-width:86px;line-height:1.5}
.kdsml-studio .kdsml-table-scroll th .kdsml-th-en{
 display:block;text-align:left;font:700 11px Arial,sans-serif;
 color:#597489;direction:ltr;line-height:1.4}
.kdsml-studio .kdsml-table-scroll td{padding:10px 13px;
 border-bottom:1px solid #e8edf1;white-space:nowrap}
.kdsml-studio .kdsml-table-scroll tbody tr:nth-child(even){background:#f8fbfd}
.kdsml-studio .kdsml-table-panel .kdsml-table-hint{margin-top:12px;font-size:11px}
.kdsml-studio .kdsml-notes{margin-top:17px;padding:10px 15px;border-radius:9px;
 border:1px dashed #bbcbd8;background:#fcfdfd;color:#465b70}
.kdsml-studio .kdsml-notes p{margin:5px 0;font-size:13px;line-height:1.7;font-weight:600}
.kdsml-studio .kdsml-chart{margin:15px 0 8px;padding:16px 20px;background:#fff;
 border:1px solid #e3eaf0;border-radius:16px;break-inside:avoid}
.kdsml-studio .kdsml-chart figcaption{display:flex;justify-content:space-between;
 font-weight:700;color:#314a64;margin-bottom:12px;gap:12px}
.kdsml-studio .kdsml-chart figcaption span:last-child{font:600 12px Arial,sans-serif}
.kdsml-studio .kdsml-chart svg{display:block;width:100%;height:125px}
.kdsml-studio .kdsml-chart-range{display:flex;justify-content:space-between;
 font:600 11px Arial,sans-serif;color:#586d82}
.kdsml-studio .kdsml-footer{text-align:center;font:600 12px Arial,sans-serif;
 color:#698094;line-height:1.7;margin:22px 0 2px;direction:ltr}
@media(max-width:680px){.kdsml-studio{padding:10px;border-radius:13px}
.kdsml-studio .kdsml-hero{padding:20px;border-radius:13px}
.kdsml-studio .kdsml-result{padding:16px 14px}
.kdsml-studio .kdsml-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}
.kdsml-studio .kdsml-metric strong{font-size:20px}}
@media(max-width:420px){.kdsml-studio .kdsml-metrics{grid-template-columns:1fr}}
@media print{.kdsml-studio{background:#fff;border:0;max-width:none;
 border-radius:0;padding:0;color:#111;--kdsml-soft:#f5f5f5}
.kdsml-studio .kdsml-hero{background:#f2f4f6 !important;color:#111 !important;
 border:1px solid #c2c2c2;print-color-adjust:economy;-webkit-print-color-adjust:economy}
.kdsml-studio .kdsml-hero h1,.kdsml-studio .kdsml-hero p,
.kdsml-studio .kdsml-brand{color:#111 !important}
.kdsml-studio .kdsml-chip{border:1px solid #aaa;color:#111}
.kdsml-studio .kdsml-result{box-shadow:none;border:1px solid #ccc}
.kdsml-studio .kdsml-ar,.kdsml-studio .kdsml-en,.kdsml-studio .kdsml-metric{
 background:#fff;border-color:#bbb}
.kdsml-studio .kdsml-result,.kdsml-studio .kdsml-chart{break-inside:avoid}}
"""


def render_analysis_dashboard(
    results,
    *,
    title_ar: str = "لوحة التحليل الإحصائي",
    title_en: str = "STATISTICAL INSIGHTS",
    theme: str = "aurora",
    font_size: int = 20,
    chart_data: Sequence[float] | None = None,
) -> str:
    """Return print-ready UTF-8 HTML for one/many scientific analysis results.

    The only chart is a sparkline of caller-provided, time-ordered numbers;
    the function does NOT infer trends, causality, or significance from it.
    Themes: aurora, ocean, amethyst. Text is always HTML-escaped.
    """
    if theme not in _PALETTES:
        raise ValueError(f"theme must be one of {', '.join(_PALETTES)}")
    if type(font_size) is not int or not 14 <= font_size <= 36:
        raise ValueError("font_size must be an integer from 14 to 36")
    if not isinstance(title_ar, str) or not isinstance(title_en, str):
        raise TypeError("titles must be strings")
    items = _entries(results)
    content = "".join(_render_item(item, i) for i, item in enumerate(items, 1))
    graph = _sparkline(chart_data)
    ink, mid, light, light2, soft = _PALETTES[theme]
    styles = (
        f"--kdsml-ink:{ink};--kdsml-mid:{mid};"
        f"--kdsml-accent:{light};--kdsml-accent-light:{light2};"
        f"--kdsml-soft:{soft};--kdsml-text-size:{font_size}px"
    )
    return (
        '<!doctype html><html lang="ar" dir="rtl"><head>'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(title_ar)}</title><style>{_CSS}</style></head><body>'
        f'<main class="kdsml-studio" dir="rtl" style="{styles}">'
        '<header class="kdsml-hero"><div class="kdsml-brand">'
        'KHALED DSML · ANALYTICS STUDIO</div>'
        f'<h1 lang="ar" dir="rtl">{escape(title_ar)}</h1>'
        f'<p lang="en" dir="ltr">{escape(title_en)}</p>'
        '<div class="kdsml-hero-foot">'
        f'<span class="kdsml-chip" dir="ltr">{len(items)} '
        'results</span><span class="kdsml-chip" dir="ltr">AR / EN</span>'
        '</div></header>'
        + graph + content +
        '<footer class="kdsml-footer">KhaledDSML · Scientific interpretation · '
        'Correlations and test outcomes do not establish causality.</footer>'
        '</main></body></html>'
    )


def display_analysis_dashboard(results, **kwargs) -> None:
    """Show an impressive yet accessible analytics report in Colab or Jupyter."""
    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise ImportError(
            'Install IPython or khaled-dsml[notebooks] for notebook display'
        ) from exc
    display(HTML(render_analysis_dashboard(results, **kwargs)))


def save_analysis_dashboard(results, path: str | Path, **kwargs) -> Path:
    """Save an offline interactive-free, print-friendly HTML dashboard."""
    target = Path(path)
    target.write_text(render_analysis_dashboard(results, **kwargs), encoding="utf-8")
    return target

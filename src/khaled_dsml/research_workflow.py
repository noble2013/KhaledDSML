"""Research workflows with explicit assumptions, human overrides and bilingual results.

Auto-selection is a convenience, NOT a substitute for prespecified protocols.
Normality and equal-variance tests have low power on small samples and the
decision tree can inflate type-I error if used indiscriminately.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd
from scipy import stats

from .audit import audit_dataset
from ._results import TestResult


def _vector(values,minimum=3):
    x=np.asarray(values,dtype=float).reshape(-1)
    if len(x)<minimum or not np.isfinite(x).all():
        raise ValueError(f"Need >= {minimum} finite observations; handle missing values explicitly")
    if np.std(x)==0:
        raise ValueError("Constant observations cannot support this test selection")
    return x


def _test_summary(test, statistic, p, n, alpha, assumptions, notes_ar, notes_en):
    if not np.isfinite([statistic,p]).all() or not 0<=p<=1:
        raise ValueError("Selected test yielded undefined statistics")
    rejected=bool(p<alpha)
    return {
        "test":test,"statistic":float(statistic),"p_value":float(p),
        "alpha":alpha,"reject_null":rejected,"n_samples":int(n),
        "assumption_checks":assumptions,
        "interpretation_ar":(
            ("توجد أدلة إحصائية ضد تساوي المجموعتين. " if rejected
             else "لا نرفض فرضية تساوي المجموعتين؛ لا يعني هذا إثباتها. ")
            +notes_ar),
        "interpretation_en":(
            ("Statistical evidence against equality between groups. " if rejected
             else "Do not reject group equality; this does not prove the null. ")
            +notes_en),
    }


def compare_groups(group_a, group_b, *, paired=False, method="auto", alpha=.05):
    """Compare two groups, reporting the chosen test and normality checks.

    method='auto' chooses Welch t for approximately normal independent samples,
    else Mann–Whitney; for paired data chooses paired t or Wilcoxon. The choice
    is exploratory, not a valid substitute for a preregistered analysis plan.
    """
    if not 0<alpha<1:
        raise ValueError("alpha must be (0,1)")
    x,y=_vector(group_a),_vector(group_b)
    if paired and len(x)!=len(y):
        raise ValueError("Paired samples must have equal length")
    if method not in ("auto","welch","mann_whitney","paired_t","wilcoxon"):
        raise ValueError("Unknown comparison method")
    if paired and method in ("welch","mann_whitney"):
        raise ValueError("Independent-group methods cannot test paired observations")
    if not paired and method in ("paired_t","wilcoxon"):
        raise ValueError("Paired-group methods require paired=True")
    target=x-y if paired else x
    tested=[target] if paired else [x,y]
    normality=[
        float(stats.shapiro(z).pvalue) if len(z)>=3 and len(z)<=5000 else None
        for z in tested
    ]
    looks_normal=all(p is not None and p>=alpha for p in normality)
    selection=("paired_t" if paired else "welch") if looks_normal else (
        "wilcoxon" if paired else "mann_whitney"
    )
    chosen=selection if method=="auto" else method
    if chosen=="welch":
        r=stats.ttest_ind(x,y,equal_var=False)
    elif chosen=="mann_whitney":
        r=stats.mannwhitneyu(x,y,alternative="two-sided")
    elif chosen=="paired_t":
        r=stats.ttest_rel(x,y)
    else:
        try:
            r=stats.wilcoxon(x,y,alternative="two-sided")
        except ValueError as exc:
            raise ValueError("Wilcoxon needs nonzero paired differences") from exc
    assumptions={
        "normality_shapiro_p":normality,
        "paired":bool(paired),
        "selected_by":"heuristic" if method=="auto" else "researcher",
    }
    title={
        "welch":"Welch independent t",
        "mann_whitney":"Mann–Whitney U",
        "paired_t":"Paired t",
        "wilcoxon":"Wilcoxon signed-rank",
    }[chosen]
    return _test_summary(
        title,r.statistic,r.pvalue,len(x)+len(y) if not paired else len(x),
        alpha,assumptions,
        "اختيار الاختبار استكشافي؛ لا تفسر عدم رفض الاعتدالية بوصفه إثباتًا لها.",
        "Automatic selection is exploratory; failure to reject normality is not proof of it.",
    )


def research_quality_gate(
    data, *, max_missing_fraction=.2,
    allow_missing=False, allow_duplicates=False,
) -> dict:
    """Fail closed when selected data-quality warnings are not acknowledged.

    The gate NEVER modifies data. An explicit allow flag only bypasses a
    particular stop condition; it does not remove warnings or certify validity.
    """
    if not isinstance(data,pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if not 0<=max_missing_fraction<=1:
        raise ValueError("max_missing_fraction must be in [0,1]")
    report=audit_dataset(data,missing_alert=max_missing_fraction)
    n=len(data)
    missing_share=float(data.isna().mean().max()) if n and len(data.columns) else 0
    if report["duplicate_rows"] and not allow_duplicates:
        raise ValueError("Repeated complete rows require explicit allow_duplicates=True after review")
    if missing_share>max_missing_fraction and not allow_missing:
        raise ValueError("Excessive missing values require explicit allow_missing=True after review")
    report["approved_overrides"]={
        "allow_missing":bool(allow_missing),"allow_duplicates":bool(allow_duplicates),
    }
    report["interpretation_ar"]+=" تم السماح بالمؤشرات التي تجاوزتها صراحةً فقط، إن وجدت."
    report["interpretation_en"]+=" Explicitly overridden warnings remain present."
    return report


def apa_style_test_summary(result) -> dict:
    """Conservative APA-inspired one-line summaries; not full APA 7 tables.

    This helper cannot invent degrees of freedom, confidence intervals,
    corrected p-values or effect sizes that the source did not provide.
    """
    if isinstance(result,TestResult):
        data=result.to_dict()
    elif isinstance(result,Mapping):
        data=dict(result)
    else:
        raise TypeError("Expected TestResult or mapping")
    needed=["statistic","p_value","test"]
    if any(k not in data for k in needed):
        raise ValueError("Missing test name, statistic, or p-value")
    stat=float(data["statistic"])
    p=float(data["p_value"])
    if not np.isfinite(stat) or not np.isfinite(p) or not 0<=p<=1:
        raise ValueError("Invalid numeric summary")
    p_label="p < .001" if p<.001 else f"p = {p:.3f}"
    sentence=f"{data['test']}: statistic = {stat:.3f}, {p_label}."
    return {
        "formatted":sentence,
        "interpretation_ar":(
            "صياغة مختصرة بأسلوب APA وليست جدول APA كاملاً؛ "
            "درجات الحرية وحجم الأثر وفواصل الثقة يجب ذكرها إن توفرت."
        ),
        "interpretation_en":(
            "APA-inspired abbreviated report, not full APA 7 formatting; "
            "add actual degrees of freedom, effect size, and confidence intervals where available."
        ),
    }

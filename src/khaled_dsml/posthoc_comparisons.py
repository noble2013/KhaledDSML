"""Research-grade post-hoc group comparisons with transparent assumptions.

Outputs are ordinary bilingual dictionaries and pandas tables, compatible
with the offline, RTL/LTR Analytics Studio renderer. No automatic hypothesis
is chosen after seeing outcomes.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _groups(groups, *, minimum=3):
    if not isinstance(groups, dict) or len(groups) < minimum:
        raise ValueError(f"Expected a mapping of at least {minimum} named groups")
    if len({str(k) for k in groups}) != len(groups):
        raise ValueError("Group names must be unique after conversion to text")
    values = {}
    for label, group in groups.items():
        try:
            arr = np.asarray(group, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Group observations must be numeric") from exc
        if arr.ndim != 1 or arr.size < 3 or not np.isfinite(arr).all():
            raise ValueError("Each group requires >=3 finite numeric observations")
        if np.var(arr, ddof=1) <= 0:
            raise ValueError("All groups require nonzero within-group variance")
        values[str(label)] = arr
    return values


def _alpha(alpha):
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")


def _correct(p, *, method="holm"):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Correction requires finite p-values in [0, 1]")
    if method == "none":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(1., p*len(p))
    if method == "holm":
        order = np.argsort(p)
        ordered = np.minimum(1., p[order]*(len(p)-np.arange(len(p))))
        ordered = np.maximum.accumulate(ordered)
        corrected = np.empty_like(ordered)
        corrected[order] = ordered
        return corrected
    raise ValueError("Correction must be holm, bonferroni, or none")


def _wrap(name, frame, groups, alpha, ar, en):
    return {
        "test": name,
        "table": frame,
        "alpha": alpha,
        "n_samples": sum(map(len, groups.values())),
        "n_groups": len(groups),
        "significant_comparisons": int(frame["reject_null"].sum()),
        "interpretation_ar": ar,
        "interpretation_en": en,
    }


def tukey_hsd_comparisons(groups, *, alpha=.05):
    """Tukey HSD post-hoc test for independent, homoscedastic normal groups.

    Uses SciPy's studentized-range implementation including unequal group n.
    """
    _alpha(alpha)
    data = _groups(groups)
    names = list(data)
    result = stats.tukey_hsd(*(data[k] for k in names))
    conf = result.confidence_interval(confidence_level=1-alpha)
    rows = []
    for i, j in combinations(range(len(names)), 2):
        p = float(result.pvalue[i,j])
        rows.append({
            "group_a": names[i], "group_b": names[j],
            "mean_difference": float(result.statistic[i,j]),
            "ci_lower": float(conf.low[i,j]), "ci_upper": float(conf.high[i,j]),
            "p_value": p, "reject_null": bool(p < alpha),
        })
    return _wrap(
        "Tukey HSD pairwise means", pd.DataFrame(rows), data, alpha,
        "مقارنات Tukey بعد ANOVA بافتراض مجموعات مستقلة وتجانس التباين؛ فواصل ثقة متزامنة.",
        "Tukey HSD requires independent groups, approximately normal errors and "
        "equal variances; its intervals account for multiple comparisons.",
    )


def games_howell_comparisons(groups, *, alpha=.05):
    """Games–Howell all-pairs tests: Welch SE/df and studentized-range p-values.

    Appropriate for independent samples with unequal variances, assuming
    reasonable within-group normality; asymptotic uncertainty for tiny groups.
    """
    _alpha(alpha)
    data = _groups(groups)
    names = list(data)
    k = len(names)
    rows = []
    for i, j in combinations(range(k),2):
        a, b = data[names[i]], data[names[j]]
        vi, vj = np.var(a,ddof=1)/len(a), np.var(b,ddof=1)/len(b)
        se = np.sqrt(vi + vj)
        df = (vi+vj)**2/(vi**2/(len(a)-1)+vj**2/(len(b)-1))
        diff = float(np.mean(a)-np.mean(b))
        q = np.sqrt(2)*abs(diff)/se
        p = float(stats.studentized_range.sf(q, k, df))
        critical = float(stats.studentized_range.ppf(1-alpha,k,df))
        half = critical*se/np.sqrt(2)
        rows.append({
            "group_a":names[i], "group_b":names[j],
            "mean_difference":diff, "ci_lower":float(diff-half),
            "ci_upper":float(diff+half),
            "welch_df":float(df), "p_value":p, "reject_null":bool(p<alpha),
        })
    return _wrap(
        "Games-Howell unequal-variance post-hoc", pd.DataFrame(rows), data, alpha,
        "اختبار Games–Howell للمقارنات البعدية، لا يفترض تجانس التباين بين المجموعات المستقلة.",
        "Games–Howell uses Welch standard errors, Satterthwaite degrees of freedom "
        "and the studentized-range distribution for unequal independent variances.",
    )


def dunn_posthoc(groups, *, alpha=.05, correction="holm"):
    """Dunn pairwise rank comparisons with pooled tie correction.

    Inputs are independent groups. P-values are asymptotic normal, then
    Holm- or Bonferroni-corrected when requested.
    """
    _alpha(alpha)
    data = _groups(groups)
    names = list(data)
    sample = np.concatenate([data[n] for n in names])
    n = len(sample)
    ranks = stats.rankdata(sample)
    _, ties = np.unique(sample, return_counts=True)
    tie_factor = 1-sum(t**3-t for t in ties)/(n**3-n)
    if tie_factor <= 0:
        raise ValueError("All observations have the same rank")
    means = {}
    pos = 0
    for name in names:
        next_pos = pos+len(data[name])
        means[name] = float(ranks[pos:next_pos].mean())
        pos=next_pos
    rows = []
    raw = []
    for a,b in combinations(names,2):
        diff = means[a]-means[b]
        se = np.sqrt(n*(n+1)/12*tie_factor*(1/len(data[a])+1/len(data[b])))
        z = diff/se
        p = float(2*stats.norm.sf(abs(z)))
        raw.append(p)
        rows.append({
            "group_a":a,"group_b":b,"mean_rank_difference":diff,
            "z_statistic":float(z),"raw_p_value":p,
        })
    adjusted = _correct(raw,method=correction)
    for row,p in zip(rows,adjusted):
        row["p_value"]=float(p)
        row["reject_null"]=bool(p<alpha)
    return _wrap(
        "Dunn pairwise rank comparisons",pd.DataFrame(rows),data,alpha,
        "اختبار Dunn اللامعلمي بعد Kruskal–Wallis، مع تصحيح الروابط والمقارنات المتعددة.",
        f"Dunn rank comparisons for independent groups with pooled tie correction; "
        f"p-value multiplicity adjustment: {correction}.",
    )


def dunnett_control_comparisons(groups, *, control, alpha=.05, seed=42):
    """Dunnett many-to-one means comparing treatments with one control.

    Uses SciPy Dunnett; independent normal groups with a common variance
    required. Standardized comparison is treatment minus control.
    """
    _alpha(alpha)
    data = _groups(groups,minimum=3)
    if str(control) not in data:
        raise KeyError("Specified control must be an existing group name")
    try:
        # scipy>=1.11 uses scipy.stats.dunnett(..., random_state=...)
        from scipy.stats import dunnett
    except ImportError as exc:
        raise ImportError("Dunnett requires SciPy >=1.11") from exc
    other = [key for key in data if key != str(control)]
    import inspect
    params = inspect.signature(dunnett).parameters
    keyword = {"rng":seed} if "rng" in params else {"random_state":seed}
    result = dunnett(*(data[k] for k in other),control=data[str(control)],
                     alternative="two-sided",**keyword)
    ci = result.confidence_interval(confidence_level=1-alpha)
    rows = []
    for index,label in enumerate(other):
        p = float(result.pvalue[index])
        diff = float(data[label].mean()-data[str(control)].mean())
        rows.append({
            "group_a":label,"group_b":str(control),
            "mean_difference":diff,
            "ci_lower":float(ci.low[index]),"ci_upper":float(ci.high[index]),
            "p_value":p,"reject_null":bool(p<alpha),
        })
    return _wrap(
        "Dunnett treatment versus control",pd.DataFrame(rows),data,alpha,
        "مقارنة Dunnett لكل علاج مقابل الضبط بتصحيح المقارنات؛ لا يجري مقارنة كل علاج بغيره.",
        "Dunnett compares each independent treatment to the control, not all "
        "treatment pairs; assumes normally distributed errors and common variance.",
    )


def nemenyi_friedman_posthoc(groups, *, alpha=.05):
    """Nemenyi comparison of within-block average ranks after Friedman.

    Paired measurements are rows/blocks. Within-row ties are not supported,
    because the simple studentized-range approximation lacks tie correction.
    """
    _alpha(alpha)
    data = _groups(groups)
    names = list(data)
    n = len(data[names[0]])
    if any(len(data[name]) != n for name in names):
        raise ValueError("Friedman–Nemenyi requires equally sized matched measurements")
    frame = np.column_stack([data[name] for name in names])
    if any(len(np.unique(row)) != len(names) for row in frame):
        raise ValueError("Within-block tied ranks require a tie-adjusted procedure")
    ranks = np.apply_along_axis(stats.rankdata, 1, frame)
    means = ranks.mean(axis=0)
    k = len(names)
    sd = np.sqrt(k*(k+1)/(6*n))
    rows=[]
    for i,j in combinations(range(k),2):
        q = float(np.sqrt(2)*abs(means[i]-means[j])/sd)
        p = float(stats.studentized_range.sf(q,k,np.inf))
        rows.append({
            "group_a":names[i],"group_b":names[j],
            "mean_rank_difference":float(means[i]-means[j]),
            "p_value":p,"reject_null":bool(p<alpha),
        })
    return _wrap(
        "Friedman-Nemenyi paired-rank post-hoc",pd.DataFrame(rows),data,alpha,
        "مقارنات Nemenyi بعد Friedman على رتب داخل كتل مترابطة دون تعادلات.",
        "Nemenyi compares within-block mean ranks after Friedman, with an "
        "asymptotic studentized-range approximation and no tied ranks.",
    )

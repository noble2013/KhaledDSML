"""Data drift and online quality monitoring with explicit calibration limits.

Drift scores describe distributional discrepancies; only the independent-sample
KS test and the pre-specified sample-ratio test produce inferential p-values.
PSI cutoffs, if chosen by practitioners, are not universal significance tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


def _finite_numeric(sample, name, *, minimum=20):
    try:
        x = np.asarray(sample, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and finite") from exc
    if x.ndim != 1 or len(x) < minimum or not np.isfinite(x).all():
        raise ValueError(f"{name} must be 1D with at least {minimum} finite numbers")
    return x


def distribution_drift_profile(
    reference, current, *, bins=10, smoothing=1e-6, alpha=0.05,
):
    """Compare numerical samples using PSI, JS(bits), Wasserstein and KS.

    Reference quantile bins are fixed before inspecting current values.
    Identical quantiles are consolidated, and extreme bins remain open-ended.
    Additive probability smoothing is ONLY for PSI and Jensen-Shannon.
    No universal significance threshold exists for PSI or Wasserstein. KS
    p-values require independent, representative observations and do not
    adjust for repeated monitoring or multiple comparison.
    """
    ref = _finite_numeric(reference, "reference")
    cur = _finite_numeric(current, "current")
    if (type(bins) is not int or not 2 <= bins <= 100
        or isinstance(smoothing, (bool, np.bool_))
        or not np.isscalar(smoothing) or not np.isfinite(smoothing)
        or not 0 < smoothing <= 0.1
        or isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha)
        or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("bins 2..100, smoothing (0,0.1], alpha (0,1) required")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError("Reference distribution lacks distinct quantile boundaries")
    edges[0], edges[-1] = -np.inf, np.inf
    rcounts = np.histogram(ref, bins=edges)[0]
    ccounts = np.histogram(cur, bins=edges)[0]
    effective = len(rcounts)
    rp = (rcounts / len(ref) + smoothing)
    cp = (ccounts / len(cur) + smoothing)
    rp /= rp.sum()
    cp /= cp.sum()
    contribution = (cp - rp) * np.log(cp / rp)
    psi = float(contribution.sum())
    js = float(jensenshannon(rp, cp, base=2.0) ** 2)
    ks = stats.ks_2samp(ref, cur, alternative="two-sided", method="auto")
    distance = float(stats.wasserstein_distance(ref, cur))
    return {
        "test": "Numeric distribution drift profile",
        "n_reference": len(ref), "n_current": len(cur),
        "n_bins": int(effective), "alpha": float(alpha),
        "population_stability_index": max(0., psi),
        "jensen_shannon_divergence_bits": max(0., js),
        "wasserstein_distance": distance,
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "reject_equal_distributions_ks": bool(ks.pvalue < alpha),
        "table": pd.DataFrame({
            "bin_lower": edges[:-1], "bin_upper": edges[1:],
            "reference_count": rcounts, "current_count": ccounts,
            "reference_share_smoothed": rp, "current_share_smoothed": cp,
            "psi_contribution": contribution,
        }),
        "interpretation_ar": (
            "مؤشرات PSI وJensen–Shannon وWasserstein وصفية؛ لا توجد عتبة دلالة "
            "عالمية لها. تعتمد حدود الفئات على بيانات المرجع فقط. قيمة p "
            "لاختبار KS تتطلب عينتين مستقلتين ولا تعالج تعدد الفحوص أو المراقبة المتكررة."
        ),
        "interpretation_en": (
            "PSI, Jensen-Shannon (bits), and Wasserstein are descriptive drift "
            "scores, not hypothesis-test p-values. Quantile bins are fitted "
            "on the reference only. The KS p-value assumes independent samples "
            "and is not corrected for repeated checks or multiple testing."
        ),
    }


def welford_streaming_summary(chunks):
    """Stable single-pass mean and sample variance across streamed numeric chunks.

    Uses the Welford/Chan merge formula, never concatenating all input chunks.
    Missing or infinite values must be handled by the caller's explicit policy.
    """
    total, mean, m2 = 0, 0., 0.
    for chunk in chunks:
        x = np.asarray(chunk, dtype=float)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("Each chunk must be a finite 1D numeric vector")
        if not len(x):
            continue
        n2 = len(x)
        local_mean = float(np.mean(x))
        local_m2 = float(np.sum((x - local_mean) ** 2))
        if not total:
            total, mean, m2 = n2, local_mean, local_m2
            continue
        delta = local_mean - mean
        updated = total + n2
        m2 += local_m2 + delta * delta * total * n2 / updated
        mean += delta * n2 / updated
        total = updated
    if total < 2:
        raise ValueError("At least two observed numeric values are needed")
    sample = max(0.0, m2 / (total - 1))
    return {
        "test": "Welford streaming summary",
        "n_samples": int(total),
        "mean": float(mean),
        "variance_sample": float(sample),
        "std_sample": float(np.sqrt(sample)),
        "variance_population": float(max(0.0, m2 / total)),
        "interpretation_ar": (
            "حساب تجميعي متين عدديًا للمتوسط والتباين بتحديثات مستمرة، "
            "من دون تجميع السجلات كلها في الذاكرة. لا يتضمن استدلالًا احتماليًا."
        ),
        "interpretation_en": (
            "Numerically stable mergeable streaming mean and variance without "
            "concatenating raw input chunks. Descriptive only; no hypothesis "
            "test or statistical p-value is implied."
        ),
    }


def sample_ratio_mismatch(observed_counts, expected_proportions, *, alpha=0.05):
    """Pearson chi-square Sample Ratio Mismatch against a preregistered allocation.

    Appropriate for independently assigned, mutually exclusive A/B arms.
    Minimum expected count of 5 for each arm; repeated sequential peeking
    changes false-positive behavior and is NOT supported by this p-value.
    """
    obs = np.asarray(observed_counts)
    expected = np.asarray(expected_proportions, dtype=float)
    if (obs.ndim != 1 or expected.ndim != 1 or obs.size != expected.size
        or obs.size < 2 or obs.size > 200 or obs.dtype.kind not in "iu"
        or np.any(obs < 0) or not np.isfinite(expected).all()
        or np.any(expected <= 0) or not np.isclose(expected.sum(), 1., atol=1e-10)
        or isinstance(alpha, (bool, np.bool_)) or not 0 < alpha < 1):
        raise ValueError("Provide >=2 integer nonnegative counts and positive probabilities summing to one")
    if obs.sum() <= 0:
        raise ValueError("No observations")
    exp = obs.sum() * expected
    if np.any(exp < 5):
        raise ValueError("All expected arm counts must be at least five for chi-square approximation")
    chi = float(np.sum((obs - exp) ** 2 / exp))
    p = float(stats.chi2.sf(chi, len(obs)-1))
    return {
        "test": "Sample Ratio Mismatch (SRM)",
        "statistic": chi, "p_value": p,
        "degrees_of_freedom": len(obs)-1, "alpha": float(alpha),
        "reject_null": bool(p < alpha), "n_samples": int(obs.sum()),
        "n_groups": int(len(obs)),
        "table": pd.DataFrame({
            "group": np.arange(1, len(obs) + 1),
            "observed": obs.astype(int), "expected": exp,
            "planned_share": expected,
        }),
        "null_hypothesis_ar": "تتبع أعداد المجموعات نسب التخصيص المحددة قبل التجربة.",
        "null_hypothesis_en": "Observed arm counts follow the prespecified assignment proportions.",
        "interpretation_ar": (
            "اختبار نسب توزيع مجموعات التجربة المحددة مسبقًا، تحت استقلالية "
            "التخصيص وصلاحية تقريب كاي تربيع. لا يصح اعتبار المراقبة المتكررة "
            "اختبارًا واحدًا؛ الرفض ينبه لخلل محتمل ولا يثبت سببه."
        ),
        "interpretation_en": (
            "Chi-square check against prespecified randomized allocation "
            "proportions, requiring independent assignments and adequate "
            "expected counts. Repeated peeking is not adjusted; rejection "
            "flags possible allocation imbalance but does not diagnose its cause."
        ),
    }

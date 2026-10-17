"""SHAP explanation stability diagnostics on precomputed aligned matrices.

The caller owns the SHAP calculation, model training, background dataset,
holdout isolation and feature correspondence. This module does NOT compute
SHAP values or infer fairness, causal effects or global model reliability.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats


def shap_explanation_stability(shap_runs, *, feature_names=None):
    """Compare aligned SHAP matrices across at least three model/refit runs.

    Input is a numeric 3D array (runs, same_holdout_rows, same_features).
    Every run must explain the same rows, target and reference definition,
    with the same feature order. Computes a Spearman rank association across
    per-feature mean absolute SHAP importance vectors and a per-feature
    importance coefficient of variation. Flat rank vectors are rejected.
    Does not compare feature attribution sign when background / output units
    differ and does not refit or compute any SHAP estimates.
    """
    try: arr=np.asarray(shap_runs,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("SHAP matrices must be finite numeric arrays") from exc
    if (arr.ndim!=3 or not 3<=arr.shape[0]<=50
        or not 10<=arr.shape[1]<=100000
        or not 3<=arr.shape[2]<=200
        or not np.isfinite(arr).all()):
        raise ValueError("Need 3..50 runs x >=10 same holdout rows x 3..200 features")
    nrun,nrow,nfeat=arr.shape
    if feature_names is None:
        names=[f"feature_{i+1}" for i in range(nfeat)]
    elif (not isinstance(feature_names,(list,tuple))
        or len(feature_names)!=nfeat
        or len(set(feature_names))!=nfeat
        or any(not isinstance(v,str) or not v.strip() for v in feature_names)):
        raise ValueError("Feature names must be distinct nonempty ordered strings")
    else:names=list(feature_names)
    importance=np.abs(arr).mean(axis=1)
    if (not np.isfinite(importance).all()
        or np.any(np.ptp(importance,axis=1)<=1e-12)):
        raise ValueError("Rank correlation is undefined when an entire run's importances are tied")
    pairs=[]
    for a,b in combinations(range(nrun),2):
        rho=float(stats.spearmanr(importance[a],importance[b]).statistic)
        if not np.isfinite(rho):
            raise ValueError("Undefined rank correlation across SHAP runs")
        pairs.append({"run_a":a+1,"run_b":b+1,"spearman_rank":rho})
    avg=importance.mean(axis=0)
    sd=importance.std(axis=0,ddof=1)
    cv=np.where(avg>1e-12,sd/np.maximum(avg,1e-12),np.nan)
    table=pd.DataFrame({
        "feature":names,"mean_absolute_shap":avg,
        "sd_mean_absolute_shap_across_runs":sd,
        "coefficient_of_variation":cv,
    })
    pair_table=pd.DataFrame(pairs)
    return {
        "test":"Precomputed SHAP global-importance stability",
        "n_runs":nrun,"n_samples":nrow,"n_features":nfeat,
        "mean_pairwise_spearman":float(pair_table["spearman_rank"].mean()),
        "min_pairwise_spearman":float(pair_table["spearman_rank"].min()),
        "n_pairwise_comparisons":len(pair_table),
        "pairwise_rank_table":pair_table,
        "table":table,
        "interpretation_ar":(
            "فحص ثبات ترتيب الأهمية المطلقة لمصفوفات SHAP محسوبة "
            "مسبقًا على حالات الاختبار والسمات نفسها. يتطلب نفس "
            "مقياس المخرجات والخلفية المرجعية وترتيب السمات في كل "
            "تشغيل. لا تحسب الدالة SHAP ولا تؤكد استقرار التنبؤ "
            "أو العدالة أو العلاقة السببية."
        ),
        "interpretation_en":(
            "Compares global mean-absolute SHAP importance across "
            "precomputed, perfectly aligned runs on identical held-out "
            "rows, feature ordering and common background/output scale. "
            "Spearman agreement does not establish explanation validity, "
            "predictive stability, causality or fairness. SHAP is not "
            "computed or refitted here."
        ),
    }

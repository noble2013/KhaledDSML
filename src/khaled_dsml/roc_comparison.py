"""Paired ROC-AUC comparison using empirical DeLong influence functions.

The test compares discriminative ROC areas on the *same held-out individuals*.
It does not compare precision-recall AUC, calibration or clinical usefulness,
and it never treats correlated predictions as independent observations.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _validate(y_true, scores_a, scores_b):
    y = np.asarray(y_true)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if any(v.ndim != 1 for v in (y,a,b)) or not len(y)==len(a)==len(b):
        raise ValueError("Need aligned one-dimensional labels and two scoring vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Prediction scores must be finite and not contain missing values")
    if not np.isin(y,[0,1]).all():
        raise ValueError("Binary labels must be encoded as 0/1")
    if (y==1).sum()<4 or (y==0).sum()<4:
        raise ValueError("At least four positive and four negative independent cases are required")
    return y.astype(int),a,b


def _auc_influence(positive,negative):
    # Positive-vs-negative Mann-Whitney kernel, giving ties exactly half credit.
    kernel=(positive[:,None]>negative[None,:]).astype(float)
    kernel+=(positive[:,None]==negative[None,:])*.5
    return float(kernel.mean()), kernel.mean(axis=1), kernel.mean(axis=0)


def delong_auc_comparison(y_true, scores_a, scores_b, *, alpha=.05):
    """Asymptotic paired DeLong two-sided test and normal CI for AUC difference.

    Covariance uses positive and negative case-wise influence terms, with
    sample variance ddof=1 for each class. If the covariance of the difference
    is zero, p-value is not identifiable and ValueError is raised.
    """
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha must be in (0,1)")
    y,a,b=_validate(y_true,scores_a,scores_b)
    positive=y==1
    pos1,neg1=a[positive],a[~positive]
    pos2,neg2=b[positive],b[~positive]
    auc1,v10_1,v01_1=_auc_influence(pos1,neg1)
    auc2,v10_2,v01_2=_auc_influence(pos2,neg2)
    m,n=len(pos1),len(neg1)
    pos_cov=np.cov(np.stack([v10_1,v10_2]),ddof=1)/m
    neg_cov=np.cov(np.stack([v01_1,v01_2]),ddof=1)/n
    cov=pos_cov+neg_cov
    variance=float(cov[0,0]+cov[1,1]-2*cov[0,1])
    if not np.isfinite(variance) or variance<=1e-14:
        raise ValueError(
            "Paired AUC-difference variance is zero or ill-conditioned; "
            "DeLong p-value/CI is undefined for these score vectors"
        )
    diff=auc1-auc2
    z=diff/np.sqrt(variance)
    p=float(2*stats.norm.sf(abs(z)))
    critical=stats.norm.ppf(1-alpha/2)
    se=np.sqrt(variance)
    # Normal CI may overshoot feasible AUC differences; bound only the displayed
    # interval, while leaving the point estimate and test unchanged.
    ci_low=float(max(-1,diff-critical*se))
    ci_high=float(min(1,diff+critical*se))
    return {
        "test":"DeLong paired ROC-AUC comparison",
        "roc_auc_a":auc1, "roc_auc_b":auc2,
        "auc_difference":float(diff),
        "variance_difference":variance,
        "statistic":float(z),"p_value":p,"alpha":alpha,
        "ci_lower":ci_low,"ci_upper":ci_high,
        "reject_null":bool(p<alpha),"n_samples":len(y),
        "n_positive":m,"n_negative":n,
        "interpretation_ar":(
            "اختبار DeLong يقارن المساحة تحت منحنى ROC لتنبؤين على الأشخاص أنفسهم. "
            "يفترض استقلال الحالات بين الأشخاص؛ ولا يثبت السببية أو دقة المعايرة."
        ),
        "interpretation_en":(
            "DeLong compares correlated ROC areas for two scores on the same "
            "independent individuals; it does not evaluate calibration, causality "
            "or performance on new data unless scores were held out."
        ),
    }


def stratified_paired_auc_bootstrap(
    y_true, scores_a, scores_b, *, repetitions=1000, confidence_level=.95, seed=42,
):
    """Paired stratified percentile-bootstrap CI for ROC-AUC difference.

    Resamples paired scores and labels *within* class strata. It provides a
    descriptive uncertainty interval but no fictitious hypothesis p-value.
    """
    y,a,b=_validate(y_true,scores_a,scores_b)
    if (type(repetitions) is not int or not 100<=repetitions<=20000
        or not np.isfinite(confidence_level) or not 0<confidence_level<1):
        raise ValueError("Use 100-20000 repetitions and confidence level in (0,1)")
    rng=np.random.default_rng(seed)
    pos=np.flatnonzero(y==1)
    neg=np.flatnonzero(y==0)
    observed=float(roc_auc_score(y,a)-roc_auc_score(y,b))
    deltas=np.empty(repetitions)
    for k in range(repetitions):
        idx=np.concatenate([
            rng.choice(pos,len(pos),replace=True),
            rng.choice(neg,len(neg),replace=True),
        ])
        labels=y[idx]
        deltas[k]=roc_auc_score(labels,a[idx])-roc_auc_score(labels,b[idx])
    tail=(1-confidence_level)/2
    low,high=np.quantile(deltas,[tail,1-tail])
    return {
        "test":"Paired stratified bootstrap ROC-AUC difference",
        "auc_difference":observed, "ci_lower":float(low),
        "ci_upper":float(high), "confidence_level":confidence_level,
        "replications":repetitions, "n_samples":len(y),
        "interpretation_ar":(
            "فاصل ثقة Bootstrap مزدوج مع الحفاظ على الفئات. "
            "هذه ليست قيمة p لاختبار DeLong، ولا تضمن التعميم خارج العينة."
        ),
        "interpretation_en":(
            "Paired class-stratified percentile-bootstrap interval; no "
            "hypothesis-test p-value is claimed and out-of-sample generalization "
            "depends on evaluation design."
        ),
    }

"""Extended model evaluation: calibration, clustering, fairness and leakage controls.

No metric is a fairness guarantee, unbiased validation proof or safety verdict.
These functions compute from held-out labels/predictions provided by the caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit, RepeatedStratifiedKFold, RepeatedKFold


def _binary(y_true, scores):
    y=np.asarray(y_true).reshape(-1)
    p=np.asarray(scores,dtype=float).reshape(-1)
    if len(y)<4 or len(y)!=len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Need >=4 aligned binary labels and probabilities in [0,1]")
    if not np.isin(y,[0,1]).all() or np.unique(y).size<2:
        raise ValueError("Labels must contain both 0 and 1")
    return y.astype(int),p


def classification_diagnostics(y_true, probabilities, *, threshold=.5, n_bins=10) -> dict:
    """Classification metrics, PR trapezoidal AUC, Brier, log-loss and ECE.

    AP and geometric PR AUC differ; they are reported separately.
    Bin-wise ECE is descriptive and depends on bins, not a hypothesis test.
    """
    y,p=_binary(y_true,probabilities)
    if not 0<=threshold<=1 or type(n_bins) is not int or not 2<=n_bins<=100:
        raise ValueError("threshold in [0,1], and n_bins an integer between 2 and 100")
    predicted=(p>=threshold).astype(int)
    precision,recall,_=metrics.precision_recall_curve(y,p)
    cal=calibration_summary(y,p,n_bins=n_bins)
    return {
        "accuracy":float(metrics.accuracy_score(y,predicted)),
        "balanced_accuracy":float(metrics.balanced_accuracy_score(y,predicted)),
        "precision":float(metrics.precision_score(y,predicted,zero_division=0)),
        "recall":float(metrics.recall_score(y,predicted,zero_division=0)),
        "f1":float(metrics.f1_score(y,predicted,zero_division=0)),
        "mcc":float(metrics.matthews_corrcoef(y,predicted)),
        "cohen_kappa":float(metrics.cohen_kappa_score(y,predicted)),
        "roc_auc":float(metrics.roc_auc_score(y,p)),
        "average_precision":float(metrics.average_precision_score(y,p)),
        "pr_auc":float(metrics.auc(recall,precision)),
        "log_loss":float(metrics.log_loss(y,p,labels=[0,1])),
        "brier":float(metrics.brier_score_loss(y,p)),
        "ece":cal["expected_calibration_error"],
        "n_samples":len(y),
        "interpretation_ar":"هذه مقاييس تقييم خارج العينة إن كانت المدخلات من عينة اختبار مستقلة؛ PR-AUC ليست مساوية دائمًا لمتوسط الدقة، والمعايرة تختلف عن التمييز.",
        "interpretation_en":"Metrics are out-of-sample only if inputs came from held-out data; trapezoidal PR-AUC differs from average precision and calibration differs from ranking.",
    }


def calibration_summary(y_true, probabilities, *, n_bins=10) -> dict:
    """Equal-width calibration bins with weighted expected calibration error."""
    y,p=_binary(y_true,probabilities)
    if type(n_bins) is not int or not 2<=n_bins<=100:
        raise ValueError("n_bins must be an integer in [2,100]")
    indices=np.minimum((p*n_bins).astype(int),n_bins-1)
    out=[]
    ece=0.
    for b in range(n_bins):
        mask=indices==b
        n=int(mask.sum())
        if not n:
            continue
        accuracy=float(y[mask].mean())
        confidence=float(p[mask].mean())
        gap=abs(accuracy-confidence)
        ece+=n/len(y)*gap
        out.append({
            "bin":b+1,"count":n,
            "mean_predicted":confidence,"observed_fraction":accuracy,
            "absolute_gap":gap,
        })
    return {
        "bins":pd.DataFrame(out),
        "expected_calibration_error":float(ece),
        "n_samples":len(y),
        "interpretation_ar":"ECE يعتمد على اختيار سلال المعايرة ولا يمثل قياسًا شاملًا لجودة النموذج.",
        "interpretation_en":"ECE depends on binning and does not fully characterize predictive uncertainty.",
    }


def detailed_regression_metrics(actual, predicted, *, n_predictors=None) -> dict:
    """MAE, RMSE, R2, sMAPE and optional adjusted R2; MAPE if no actual zero."""
    y=np.asarray(actual,dtype=float).reshape(-1)
    f=np.asarray(predicted,dtype=float).reshape(-1)
    if len(y)<3 or len(y)!=len(f) or not np.isfinite(y).all() or not np.isfinite(f).all():
        raise ValueError("Need >=3 aligned finite observations")
    err=np.abs(y-f)
    divisor=np.abs(y)+np.abs(f)
    smape=100*np.mean(np.divide(2*err,divisor,out=np.zeros_like(err),where=divisor>0))
    r2=float(metrics.r2_score(y,f))
    result={
        "mae":float(err.mean()),"rmse":float(np.sqrt(np.mean((y-f)**2))),
        "r2":r2,"smape_percent":float(smape),
        "mape_percent":float(100*np.mean(err/np.abs(y))) if np.all(y!=0) else None,
        "n_samples":len(y),
        "interpretation_ar":"مقاييس خطأ تنبؤ؛ MAPE غير معرّف عند وجود أصفار فعلية، وR² لا يثبت السببية.",
        "interpretation_en":"Forecast errors; MAPE is undefined when actual values include zero, and R2 does not establish causality.",
    }
    if n_predictors is not None:
        if type(n_predictors) is not int or n_predictors<1 or len(y)<=n_predictors+1:
            raise ValueError("Adjusted R2 requires n_samples > n_predictors + 1")
        result["adjusted_r2"]=float(1-(1-r2)*(len(y)-1)/(len(y)-n_predictors-1))
    return result


def cluster_validation_metrics(X, labels, *, reference=None) -> dict:
    """Silhouette, Davies-Bouldin, Calinski-Harabasz, optional ARI and NMI."""
    x=np.asarray(X,dtype=float)
    y=np.asarray(labels).reshape(-1)
    if x.ndim!=2 or x.shape[0]!=len(y) or x.shape[0]<4 or not np.isfinite(x).all():
        raise ValueError("Need finite n-by-p features aligned with cluster assignments")
    distinct=np.unique(y)
    if not 2<=len(distinct)<len(y):
        raise ValueError("Clustering scores require 2 through n-1 distinct clusters")
    result={
        "silhouette":float(metrics.silhouette_score(x,y)),
        "davies_bouldin":float(metrics.davies_bouldin_score(x,y)),
        "calinski_harabasz":float(metrics.calinski_harabasz_score(x,y)),
        "n_samples":len(y),
        "interpretation_ar":"مؤشرات جودة تجميع هندسيّ؛ لا تكفي لإثبات وجود عناقيد حقيقية أو منفعة اجتماعية.",
        "interpretation_en":"Geometric clustering measures are not evidence that clusters are substantively meaningful.",
    }
    if reference is not None:
        truth=np.asarray(reference).reshape(-1)
        if len(truth)!=len(y):
            raise ValueError("reference labels must align with samples")
        result["adjusted_rand"]=float(metrics.adjusted_rand_score(truth,y))
        result["normalized_mutual_information"]=float(metrics.normalized_mutual_info_score(truth,y))
    return result


def binary_group_fairness(y_true, predictions, protected_group, *, positive_label=1) -> dict:
    """Demographic parity, equal opportunity, FPR gap and equality of odds.

    Requires exactly two group values, and both classes in each group for
    equality-of-odds comparisons. Results diagnose disparity, not legality.
    """
    y=np.asarray(y_true).reshape(-1)
    pred=np.asarray(predictions).reshape(-1)
    g=np.asarray(protected_group).reshape(-1)
    if len(y)<4 or len(y)!=len(pred) or len(y)!=len(g):
        raise ValueError("Need aligned labels, predictions and groups")
    levels=pd.unique(g)
    if len(levels)!=2:
        raise ValueError("Exactly two protected groups are supported")
    if not np.isin(y,[0,1]).all() or not np.isin(pred,[0,1]).all() or positive_label!=1:
        raise ValueError("Provide binary 0/1 actual and predicted labels, positive_label=1")
    rates=[]
    for level in levels:
        mask=g==level
        actual=y[mask]
        predicted=pred[mask]
        pos=actual==1
        neg=actual==0
        if not pos.any() or not neg.any():
            raise ValueError("Both actual classes must occur within each group")
        rates.append({
            "group":str(level),"n":int(mask.sum()),
            "selection_rate":float(predicted.mean()),
            "tpr":float(predicted[pos].mean()),
            "fpr":float(predicted[neg].mean()),
        })
    a,b=rates
    parity=abs(a["selection_rate"]-b["selection_rate"])
    tpr_gap=abs(a["tpr"]-b["tpr"])
    fpr_gap=abs(a["fpr"]-b["fpr"])
    high=max(a["selection_rate"],b["selection_rate"])
    ratio=min(a["selection_rate"],b["selection_rate"])/high if high else None
    return {
        "group_rates":pd.DataFrame(rates),
        "demographic_parity_gap":parity,"equal_opportunity_gap":tpr_gap,
        "false_positive_rate_gap":fpr_gap,
        "equalized_odds_gap":max(tpr_gap,fpr_gap),
        "selection_rate_ratio":ratio,
        "interpretation_ar":"هذه مقاييس فجوة حسب مجموعتين؛ قاعدة 80% مؤشر فرز وليست حكمًا قانونيًا أو ضمانًا للعدالة.",
        "interpretation_en":"These are group disparity diagnostics; the 80% rule is a screening heuristic, not a legal or ethical fairness verdict.",
    }


def time_series_split_indices(n_samples, *, n_splits=5, gap=0, test_size=None) -> list:
    """Chronological expanding-window splits, excluding user-specified gap."""
    if type(n_samples) is not int or n_samples<4 or type(n_splits) is not int or n_splits<2:
        raise ValueError("Need >=4 observations and n_splits>=2")
    if type(gap) is not int or gap<0:
        raise ValueError("gap must be a nonnegative integer")
    if test_size is not None and (type(test_size) is not int or test_size<1):
        raise ValueError("test_size must be a positive integer")
    splits=TimeSeriesSplit(n_splits=n_splits,gap=gap,test_size=test_size)
    return [(train,test) for train,test in splits.split(np.arange(n_samples))]


def repeated_cv_indices(y, *, folds=5, repeats=2, stratified=True, seed=42) -> list:
    """Seeded repeated classification/regression CV indices, no pipeline fitting."""
    labels=np.asarray(y).reshape(-1)
    if type(folds) is not int or folds<2 or type(repeats) is not int or repeats<1:
        raise ValueError("folds>=2 and repeats>=1 required")
    if len(labels)<folds:
        raise ValueError("Sample size smaller than number of folds")
    splitter=(RepeatedStratifiedKFold(n_splits=folds,n_repeats=repeats,random_state=seed)
              if stratified else
              RepeatedKFold(n_splits=folds,n_repeats=repeats,random_state=seed))
    try:
        return [(tr,te) for tr,te in splitter.split(np.zeros((len(labels),1)),labels)]
    except ValueError as exc:
        raise ValueError("Invalid class counts for stratified repeated cross-validation") from exc


def model_overfit_gap(train_score, validation_score, *, higher_is_better=True) -> dict:
    """Metric gap; does not prove overfitting without uncertainty/trial design."""
    if not np.isfinite([train_score,validation_score]).all():
        raise ValueError("Scores must be finite")
    gap=float(train_score-validation_score if higher_is_better else validation_score-train_score)
    return {
        "optimism_gap":gap, "train_score":float(train_score),"validation_score":float(validation_score),
        "interpretation_ar":"فجوة الأداء قد تشير إلى التفاؤل، ولا تثبت فرط التعلّم دون تقييم مستقل.",
        "interpretation_en":"A train-validation gap may indicate optimism but is not definitive evidence of overfitting.",
    }

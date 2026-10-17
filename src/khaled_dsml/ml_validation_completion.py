"""Additional evidence-first iid model evaluation and clustering screens.

Holdout data must remain excluded from model fitting, hyperparameter search,
feature engineering and decision threshold selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import NearestNeighbors


def _numeric_matrix(X, *, minimum=20):
    try:a=np.asarray(X,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("Finite numeric feature matrix required") from exc
    if (a.ndim!=2 or a.shape[0]<minimum or a.shape[1]<1
        or a.shape[1]>30 or not np.isfinite(a).all()):
        raise ValueError("Need complete numeric 2D matrix with enough rows and <=30 features")
    return a


def leave_one_out_cv_evaluation(estimator, X, y, *, scoring="accuracy",
                                max_rows=200):
    """Exact leave-one-out CV, without hyperparameter selection.

    Every model is refitted n times; preprocessing belongs inside a Pipeline.
    Per-fold R2 and ROC AUC are undefined with a single held-out observation.
    This is for iid rows, not subjects with repeated records or time-series.
    """
    if (type(max_rows) is not int or not 10<=max_rows<=1000
        or not isinstance(scoring,str) or
        scoring in ("r2","roc_auc","roc_auc_ovr","roc_auc_ovo")):
        raise ValueError("max_rows must be 10..1000; use a single-case scorer")
    scorer=get_scorer(scoring)
    if not hasattr(X,"shape") or len(X.shape)!=2:
        raise ValueError("X must be 2D")
    a=np.asarray(y)
    if a.ndim!=1 or len(a)!=len(X) or len(a)<10 or len(a)>max_rows or pd.isna(a).any():
        raise ValueError("Aligned 1D complete labels and 10..max_rows observations required")
    if is_classifier(estimator):
        _,counts=np.unique(a,return_counts=True)
        if len(counts)<2 or counts.min()<2:
            raise ValueError("Each classification label needs at least two cases")
    scores=cross_val_score(clone(estimator),X,y,cv=LeaveOneOut(),
                           scoring=scoring,error_score="raise",n_jobs=1)
    if not np.isfinite(scores).all():raise ValueError("Nonfinite leave-one-out scores")
    return {
        "test":"Leave-one-out model evaluation","n_samples":len(a),
        "mean_cv_score":float(scores.mean()),"scoring":scoring,
        "table":pd.DataFrame({"case":np.arange(len(a)),"heldout_score":scores}),
        "interpretation_ar":(
            "تحقق Leave-One-Out يعيد تدريب النموذج لكل مشاهدة مستقلة، "
            "مع تجهيز البيانات داخل Pipeline. يجب ألا تختار المعلمات "
            "على نفس النتائج ولا يُستخدم مع عينات زمنية أو مجموعات مترابطة."
        ),
        "interpretation_en":(
            "Exact Leave-One-Out CV refits the full estimator for each "
            "independent case. Preprocessing belongs inside its Pipeline; "
            "not nested tuning, grouped validation, temporal validation "
            "or an independent confirmatory holdout."
        ),
    }


def hopkins_cluster_tendency(X, *, sample_size=None, random_state=42):
    """Hopkins statistic with axis-aligned uniform reference in feature space.

    H=sum(dist(reference to data)) / (sum(dist(reference to data)) +
    sum(dist(observed to other observed))). Values near 0.5 often correspond
    to unclustered data under this artificial uniform-null design.
    Sensitive to feature scaling, geometry, dimension and sampling seed.
    """
    a=_numeric_matrix(X,minimum=30)
    n,p=a.shape
    widths=np.ptp(a,axis=0)
    if np.any(widths<=1e-12):
        raise ValueError("No constant features in Hopkins calculation")
    if sample_size is None:sample_size=min(100,n//2)
    if type(sample_size) is not int or not 5<=sample_size<=n//2:
        raise ValueError("sample_size must be within [5,n//2]")
    if isinstance(random_state,(bool,np.bool_)) or not isinstance(random_state,(int,np.integer)):
        raise ValueError("random_state must be an integer")
    rng=np.random.default_rng(int(random_state))
    ids=rng.choice(n,size=sample_size,replace=False)
    artificial=rng.uniform(a.min(axis=0),a.max(axis=0),size=(sample_size,p))
    fit=NearestNeighbors(n_neighbors=2).fit(a)
    u=fit.kneighbors(artificial,n_neighbors=1)[0][:,0]
    w=fit.kneighbors(a[ids],n_neighbors=2)[0][:,1]
    denom=float(u.sum()+w.sum())
    if denom<=1e-14:raise ValueError("Degenerate distance geometry")
    h=float(u.sum()/denom)
    return {
        "test":"Hopkins clustering tendency",
        "n_samples":n,"n_features":p,"sample_size":sample_size,
        "hopkins":h,
        "interpretation_ar":(
            "مؤشر Hopkins استكشافي بمقارنة مسافات مشاهدات حقيقية "
            "بنقاط موحدة مصطنعة في حدود البيانات. يتأثر بمقاييس السمات "
            "والبعد والبذرة؛ ليس اختبارًا وحيدًا حاسمًا لقابلية التجميع."
        ),
        "interpretation_en":(
            "Exploratory Hopkins statistic relative to an axis-aligned "
            "uniform reference. Feature scaling, geometry, dimensionality "
            "and random seed strongly affect the result. No p-value "
            "or universal clustering conclusion is implied."
        ),
    }


def holdout_noise_robustness(fitted_estimator, X_holdout, y_holdout, *,
                             scoring="accuracy", noise_sd=.1,
                             n_repeats=20, random_state=42):
    """Fixed-fitted-model sensitivity to Gaussian noise on a numeric holdout.

    Perturbations use noise_sd TIMES each holdout feature's sample SD.
    Uses no model refitting or holdout labels in training. This evaluates
    random input corruption only (not robustness to selection bias or shift).
    """
    a=_numeric_matrix(X_holdout,minimum=20)
    yy=np.asarray(y_holdout)
    if yy.ndim!=1 or len(yy)!=len(a) or pd.isna(yy).any():
        raise ValueError("Holdout labels must match rows without missing values")
    if (isinstance(noise_sd,(bool,np.bool_)) or not np.isscalar(noise_sd)
        or not np.isfinite(noise_sd) or noise_sd<=0
        or type(n_repeats) is not int or not 2<=n_repeats<=500
        or isinstance(random_state,(bool,np.bool_)) or not isinstance(random_state,(int,np.integer))):
        raise ValueError("Positive finite noise, 2..500 repeats and integer seed required")
    scorer=get_scorer(scoring)
    base=float(scorer(fitted_estimator,X_holdout,y_holdout))
    if not np.isfinite(base):raise ValueError("Baseline score nonfinite")
    rng=np.random.default_rng(int(random_state))
    scales=a.std(axis=0,ddof=1)
    records=[]
    for i in range(n_repeats):
        pert=a+rng.normal(size=a.shape)*scales*noise_sd
        if isinstance(X_holdout,pd.DataFrame):
            pert=pd.DataFrame(pert,columns=X_holdout.columns,index=X_holdout.index)
        score=float(scorer(fitted_estimator,pert,y_holdout))
        if not np.isfinite(score):raise ValueError("Noise score nonfinite")
        records.append({"replicate":i+1,"score":score,"score_change":score-base})
    table=pd.DataFrame(records)
    return {
        "test":"Holdout noise-perturbation robustness",
        "n_samples":len(a),"n_features":a.shape[1],
        "n_repeats":n_repeats,"noise_sd":float(noise_sd),
        "baseline_holdout_score":base,
        "mean_noisy_score":float(table["score"].mean()),
        "mean_score_change":float(table["score_change"].mean()),
        "table":table,
        "interpretation_ar":(
            "فحص حساسية استكشافي لنموذج مدرّب سلفًا عند إضافة "
            "ضوضاء غاوسية نسبية للسمات في مجموعة اختبار منفصلة. "
            "لا يعادل اختبار تغير المجتمع أو إثبات مقاومة الهجمات."
        ),
        "interpretation_en":(
            "Descriptive perturbation of numeric held-out features by "
            "Gaussian noise scaled to feature SD, with a fixed pretrained "
            "estimator. Not evidence of causal robustness, adversarial "
            "resistance, or protection against selection bias."
        ),
    }

"""Focused functional-form and predictive-stability research screens."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def box_tidwell_logistic_screen(outcome, positive_predictor, *, alpha=.05):
    """Single-positive-covariate Box–Tidwell logarithmic interaction in logit.

    Fit binomial-logit y ~ intercept + x + x*ln(x). H0: x log x term
    coefficient is zero. This is a one-predictor functional-form screen,
    not multivariable Box–Tidwell, a proof of linearity, or a correction
    for endogeneity. Requires positive x, two binary outcome classes and
    independent rows; separation invalidates Wald inference.
    """
    try:
        y=np.asarray(outcome,dtype=float)
        x=np.asarray(positive_predictor,dtype=float)
    except (ValueError,TypeError) as exc:
        raise ValueError("Binary response and positive finite numeric predictor required") from exc
    if (y.ndim!=1 or x.ndim!=1 or len(y)!=len(x) or len(y)<50
        or not np.isfinite(y).all() or not np.isfinite(x).all()
        or not np.isin(y,[0,1]).all() or min(int((y==1).sum()),int((y==0).sum()))<10
        or np.any(x<=0) or np.ptp(x)<=1e-10
        or isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha)
        or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError(">=50 aligned independent binary rows, both labels >=10, positive varying x")
    import statsmodels.api as sm
    design=np.column_stack((np.ones(len(x)),x,x*np.log(x)))
    if np.linalg.matrix_rank(design)<3:
        raise ValueError("Box–Tidwell terms are linearly dependent")
    fit=sm.GLM(y,design,family=sm.families.Binomial()).fit()
    coef=float(fit.params[2]);stderr=float(fit.bse[2])
    if not np.isfinite(coef) or not np.isfinite(stderr) or stderr<=1e-12:
        raise ValueError("Non-estimable Box–Tidwell term, possibly complete separation")
    stat=coef/stderr
    p=float(2*stats.norm.sf(abs(stat)))
    return {
        "test":"One-predictor Box-Tidwell logit screen",
        "n_samples":len(y),"interaction_coefficient":coef,
        "interaction_standard_error":stderr,"statistic":float(stat),
        "p_value":p,"alpha":float(alpha),
        "reject_logit_linearity":bool(p<alpha),
        "interpretation_ar":(
            "فحص Box–Tidwell للاستقامة على مقياس لوغاريتم الأرجحية "
            "لمتغير موجب واحد، عبر الحد x·ln(x). يعتمد Wald على "
            "الاستقلال والنموذج الصحيح وعدم الانفصال التام؛ "
            "عدم الرفض ليس إثباتًا للصيغة الخطية أو السببية."
        ),
        "interpretation_en":(
            "One-positive-predictor Box–Tidwell logit functional-form "
            "screen using x*log(x), with Wald asymptotic p-value. "
            "No separation, independent rows and appropriate binomial "
            "model are required. Nonsignificance does not prove linearity."
        ),
    }


def prediction_seed_stability(estimator, X_train, y_train, X_holdout, *,
                              random_state_parameter="random_state",
                              seeds=(7,17,37,57,77), task="classification"):
    """Refit model from scratch under recorded seeds; describe prediction spread.

    Fitted preprocessors must belong to the supplied estimator/Pipeline.
    Holdout X remains untouched during training. Different seeds can also
    alter stochastic optimization or split definitions, so results describe
    only the specified seed mechanism, not global reproducibility.
    """
    from sklearn.base import clone
    if not hasattr(X_train,"shape") or not hasattr(X_holdout,"shape") or len(X_train.shape)!=2 or len(X_holdout.shape)!=2:
        raise ValueError("Need 2D X_train and X_holdout")
    if (len(X_train)<20 or len(X_holdout)<10 or X_train.shape[1]!=X_holdout.shape[1]
        or len(y_train)!=len(X_train) or task not in ("classification","regression")
        or not isinstance(random_state_parameter,str)
        or not random_state_parameter or not isinstance(seeds,(tuple,list))
        or not 3<=len(seeds)<=30 or any(type(z) is not int for z in seeds)
        or len(set(seeds))!=len(seeds)):
        raise ValueError("Require aligned training/holdout, 3..30 unique integer seeds and task")
    if random_state_parameter not in estimator.get_params(deep=True):
        raise ValueError("random_state_parameter not found on estimator")
    predictions=[]
    for seed in seeds:
        m=clone(estimator)
        m.set_params(**{random_state_parameter:seed})
        m.fit(X_train,y_train)
        p=np.asarray(m.predict(X_holdout))
        if p.ndim!=1 or len(p)!=len(X_holdout) or pd.isna(p).any():
            raise ValueError("Nonfinite/missing holdout predictions")
        predictions.append(p)
    p=np.asarray(predictions)
    if task=="classification":
        consistency=np.array([
            np.unique(p[:,i],return_counts=True)[1].max()/len(seeds)
            for i in range(p.shape[1])
        ])
        label="modal_agreement_fraction"
    else:
        values=np.asarray(p,dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite regression predictions")
        consistency=values.std(axis=0,ddof=1)
        label="sd_across_seeds"
    return {
        "test":"Holdout prediction stability across seeds",
        "n_train":len(X_train),"n_test":len(X_holdout),
        "n_seeds":len(seeds),"task":task,
        "mean_stability_metric":float(consistency.mean()),
        "metric_name":label,
        "table":pd.DataFrame({"holdout_row":np.arange(len(X_holdout)),label:consistency}),
        "interpretation_ar":(
            "فحص وصفي لتغير التنبؤ عند إعادة ملاءمة النموذج ببذور "
            "عشوائية مسجلة، باستخدام مجموعة لم تدخل في التدريب. "
            "الاتفاق العالي لا يضمن عدالة النموذج أو استقراره عند تغير "
            "المجتمع أو وجود تسرب بيانات خارج Pipeline."
        ),
        "interpretation_en":(
            "Descriptive sensitivity of holdout predictions to estimator "
            "random seeds. Preprocessing must remain fold/train-only. "
            "Stability across these seeds does not establish general "
            "robustness, fairness or validity under dataset shift."
        ),
    }


def gap_statistic_kmeans(X, *, max_clusters=5, n_reference=10, random_state=42):
    """Tibshirani-style gap statistic against box-uniform reference K-means.

    Gap(k) = mean(log W*_k) - log W_k with s_k=sd(log W*_k)*sqrt(1+1/B).
    Select smallest k whose gap(k)>=gap(k+1)-s(k+1), else k_max.
    KMeans inertia is within-cluster sum of squared Euclidean distances.
    Axis-aligned uniform reference and scaling strongly affect conclusions.
    """
    from sklearn.cluster import KMeans
    try:a=np.asarray(X,dtype=float)
    except (ValueError,TypeError) as exc:
        raise ValueError("Numeric data required") from exc
    if (a.ndim!=2 or a.shape[0]<30 or not 1<=a.shape[1]<=20
        or not np.isfinite(a).all() or np.any(np.ptp(a,axis=0)<=1e-12)
        or type(max_clusters) is not int or not 2<=max_clusters<=min(10,len(a)//5)
        or type(n_reference) is not int or not 5<=n_reference<=100
        or isinstance(random_state,(bool,np.bool_)) or not isinstance(random_state,(int,np.integer))):
        raise ValueError("Finite 2D varying data, k=2..10 and reference count 5..100 required")
    rng=np.random.default_rng(int(random_state))
    low=a.min(axis=0);high=a.max(axis=0)
    output=[]
    for k in range(1,max_clusters+1):
        model=KMeans(n_clusters=k,n_init=10,random_state=int(random_state)).fit(a)
        w=float(model.inertia_)
        if w<=0:raise ValueError("Within-cluster dispersion is degenerate")
        null_logs=[]
        for i in range(n_reference):
            synthetic=rng.uniform(low,high,size=a.shape)
            km=KMeans(n_clusters=k,n_init=3,random_state=int(random_state)+i).fit(synthetic)
            if km.inertia_<=0:raise ValueError("Reference dispersion degenerate")
            null_logs.append(np.log(km.inertia_))
        gap=float(np.mean(null_logs)-np.log(w))
        spread=float(np.std(null_logs,ddof=1)*np.sqrt(1+1/n_reference))
        output.append({"k":k,"gap":gap,"reference_sd_corrected":spread,
                       "observed_inertia":w})
    table=pd.DataFrame(output)
    selected=max_clusters
    for i in range(max_clusters-1):
        if table.loc[i,"gap"]>=table.loc[i+1,"gap"]-table.loc[i+1,"reference_sd_corrected"]:
            selected=i+1
            break
    return {
        "test":"K-means gap statistic box-reference",
        "n_samples":len(a),"n_features":a.shape[1],
        "n_reference":n_reference,"selected_k":selected,
        "table":table,
        "interpretation_ar":(
            "إحصائية Gap تقارن تجمّع K-means بعينات موحدة مصطنعة "
            "داخل صندوق حدود السمات، مع قاعدة اختيار أصغر k "
            "التي تتجاوز عتبة المرجع. تتأثر بالقياس والبعد والتوزيع "
            "المرجعي، ولا تثبت وجود عناقيد حقيقية."
        ),
        "interpretation_en":(
            "K-means gap statistic compares observed inertia to a "
            "box-uniform synthetic reference, with Tibshirani's "
            "one-standard-error k-selection rule. Scaling, geometry "
            "and the reference distribution affect the conclusion."
        ),
    }

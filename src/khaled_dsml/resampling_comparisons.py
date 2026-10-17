"""Prespecified paired resampling inference for model comparison.

Input differences must come from exactly the required train/test pairing.
These methods DO NOT train models, establish independent test observations,
or correct for model/hyperparameter selection conducted on the same folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def corrected_resampled_t(differences, *, n_train, n_test, alpha=.05):
    """Nadeau–Bengio corrected repeated-holdout paired t statistic.

    t = mean(d)/sqrt((1/N+n_test/n_train)*sample_var(d)), df=N-1.
    Requires same train/test cases for models A and B on each repeat.
    The variance correction is approximate and not a guarantee of nominal
    type-I error under arbitrary dependence or adaptive model selection.
    """
    try:d=np.asarray(differences,dtype=float)
    except (TypeError,ValueError) as exc:raise ValueError("Numeric paired differences required") from exc
    if (d.ndim!=1 or len(d)<5 or not np.isfinite(d).all()
        or type(n_train) is not int or type(n_test) is not int
        or min(n_train,n_test)<5 or isinstance(alpha,(bool,np.bool_))
        or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError("Require >=5 finite paired repeat differences and positive train/test sizes")
    variance=float(d.var(ddof=1))
    correction=1/len(d)+n_test/n_train
    if variance<=1e-14:raise ValueError("Paired differences have no usable variance")
    statistic=float(d.mean()/np.sqrt(correction*variance))
    p=float(2*stats.t.sf(abs(statistic),len(d)-1))
    return {
        "test":"Nadeau-Bengio corrected resampled t",
        "n_resamples":len(d),"mean_score_difference":float(d.mean()),
        "variance_correction":float(correction),"statistic":statistic,
        "degrees_of_freedom":len(d)-1,"p_value":p,"alpha":float(alpha),
        "table":pd.DataFrame({"repeat":np.arange(1,len(d)+1),"score_difference":d}),
        "interpretation_ar":(
            "اختبار t المصحّح تقريبي لفروق أداء نموذجين على تقسيمات "
            "تدريب/اختبار مزدوجة، ويتضمن نسبة الاختبار إلى التدريب. "
            "لا يصلح دون معرفة تصميم السحب ولا يعالج اختيار النموذج "
            "بعد فحص النتائج أو الارتباط غير المعتاد."
        ),
        "interpretation_en":(
            "Approximate corrected resampled paired t from prespecified "
            "matched train/test runs, using the Nadeau–Bengio variance "
            "inflation 1/N + n_test/n_train. Not valid for arbitrarily "
            "dependent folds or post-selection claims."
        ),
    }


def five_by_two_cv_comparison(paired_differences, *, method="paired_t", alpha=.05):
    """Dietterich paired 5x2cv t or Alpaydin combined 5x2cv F.

    Input MUST be shape (5,2): five independent 50:50 random splits, each
    evaluated in both reciprocal train/test directions for the same two
    classifiers/regressors and metric. Data matrix is the paired A-B
    score difference. Cross-validation design cannot be inferred from values.
    """
    try:d=np.asarray(paired_differences,dtype=float)
    except (ValueError,TypeError) as exc:raise ValueError("5x2 paired differences must be numeric") from exc
    if (d.shape!=(5,2) or not np.isfinite(d).all()
        or method not in ("paired_t","combined_f")
        or isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha)
        or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError("Require complete 5x2 paired differences, supported method, alpha (0,1)")
    centered=d-d.mean(axis=1,keepdims=True)
    s2=(centered**2).sum(axis=1)
    total=float(s2.sum())
    if total<=1e-14:raise ValueError("5x2 within-split variance is degenerate")
    if method=="paired_t":
        test=float(d[0,0]/np.sqrt(total/5))
        p=float(2*stats.t.sf(abs(test),5))
        test_name="Dietterich paired 5x2cv t"
    else:
        test=float(np.sum(d*d)/(2*total))
        p=float(stats.f.sf(test,10,5))
        test_name="Alpaydin combined 5x2cv F"
    return {
        "test":test_name,"statistic":test,"p_value":p,"alpha":float(alpha),
        "degrees_of_freedom":5 if method=="paired_t" else (10,5),
        "mean_score_difference":float(d.mean()),"method":method,
        "table":pd.DataFrame({"split":np.repeat(np.arange(1,6),2),
                              "direction":[1,2]*5,"score_difference":d.ravel(),
                              "split_variance":np.repeat(s2,2)}),
        "interpretation_ar":(
            "اختبار فروق أداء نموذجين وفق تصميم 5×2cv المزدوج المحدد "
            "مسبقًا، بتبادل دوري التدريب والاختبار في خمس تقسيمات "
            "نصفية. القيم المدخلة يجب أن تأتي من هذا التصميم فعلًا، "
            "ولا يضمن الاختبار صحة الاختيار المتكرر للنماذج أو استقلاليتها."
        ),
        "interpretation_en":(
            "Dietterich paired t or Alpaydin combined F from actual paired "
            "5x2cv score differences only. Requires five randomized "
            "half-splits with swapped train/test directions for both "
            "models. Selection/peeking invalidates nominal inference."
        ),
    }

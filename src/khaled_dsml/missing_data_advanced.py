"""Missing-data research methods with explicit diagnostics and no silent deletion.

Little's MCAR statistic uses multivariate normal EM estimates and an asymptotic
chi-square null; non-rejection does not prove MCAR, nor does rejection identify
whether the true process is MAR or MNAR. Multiple imputation trains only on
the supplied training split and never silently observes test responses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _numeric(data, *, min_rows=30, min_cols=2, min_observed=8):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame with numeric research columns")
    if (len(data)<min_rows or data.shape[1]<min_cols
        or not data.columns.is_unique):
        raise ValueError("Not enough rows/columns, or ambiguous duplicate column names")
    if any(not pd.api.types.is_numeric_dtype(t) for t in data.dtypes):
        raise TypeError("Explicitly encode categorical variables before using numeric methods")
    x=data.to_numpy(dtype=float)
    if np.isinf(x).any():
        raise ValueError("Infinite values are not missing data and must be handled explicitly")
    if any(np.isfinite(x[:,j]).sum()<min_observed for j in range(x.shape[1])):
        raise ValueError("Each column requires at least eight observed values")
    return x


def _em_multivariate_normal(x, *, tolerance=1e-7, max_iter=300):
    """Incomplete-data multivariate-normal ML by EM, including conditional covariance."""
    n,p=x.shape
    if not 0<tolerance<.01 or type(max_iter) is not int or max_iter<20:
        raise ValueError("tolerance must be (0,.01); max_iter must be >=20")
    observed=np.isfinite(x)
    if (observed.all() or (~observed).all(axis=1).any()):
        raise ValueError("Need some but not complete missingness; fully missing rows are unsupported")
    complete=observed.all(axis=1)
    if complete.sum()<p+3:
        raise ValueError("At least p+3 complete cases required to anchor covariance estimation")
    mu=np.mean(x[complete],axis=0)
    sigma=np.cov(x[complete],rowvar=False,ddof=0)
    if np.min(np.linalg.eigvalsh(sigma))<=1e-9:
        raise ValueError("Complete-case covariance must be positive definite")
    keys, inverse=np.unique(observed,axis=0,return_inverse=True)
    for iteration in range(max_iter):
        summed=np.zeros(p)
        second=np.zeros((p,p))
        for k,mask in enumerate(keys):
            subset=x[inverse==k]
            o=np.flatnonzero(mask)
            missing=np.flatnonzero(~mask)
            expected=subset.copy()
            if len(missing):
                v=sigma[np.ix_(o,o)]
                if np.min(np.linalg.eigvalsh(v))<=1e-11:
                    raise ValueError("EM covariance is ill-conditioned")
                cross=sigma[np.ix_(missing,o)]
                adjustment=np.linalg.solve(v,cross.T)
                expected[:,missing]=mu[missing]+(
                    subset[:,o]-mu[o])@adjustment
                conditional=sigma[np.ix_(missing,missing)]-cross@adjustment
            summed+=expected.sum(axis=0)
            second+=expected.T@expected
            if len(missing):
                second[np.ix_(missing,missing)]+=len(subset)*conditional
        next_mu=summed/n
        next_sigma=second/n-np.outer(next_mu,next_mu)
        next_sigma=(next_sigma+next_sigma.T)/2
        if np.min(np.linalg.eigvalsh(next_sigma))<=1e-11:
            raise ValueError("Normal missing-data covariance became singular")
        relative=max(
            np.max(abs(next_mu-mu))/max(1,np.max(abs(mu))),
            np.max(abs(next_sigma-sigma))/max(1,np.max(abs(sigma))),
        )
        mu,sigma=next_mu,next_sigma
        if relative<tolerance:
            return mu,sigma,iteration+1
    raise ValueError("Missing-data EM did not converge; review sample and increase max_iter")


def little_mcar_test(data, *, alpha=.05, tolerance=1e-7, max_iter=300):
    """Little's large-sample chi-square test under joint multivariate normality.

    Statistic sum_r n_r*(mean_r - ML_mean)_observed' * covariance_r^-1
    * (mean_r - ML_mean)_observed. Degrees of freedom sum_r(p_observed)-p.
    Patterns with fewer than 5 observations are flagged as approximation risks.
    """
    if isinstance(alpha,bool) or not np.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha must be strictly between 0 and 1")
    x=_numeric(data)
    n,p=x.shape
    mu,cov,iterations=_em_multivariate_normal(x,tolerance=tolerance,max_iter=max_iter)
    keys,inverse=np.unique(np.isfinite(x),axis=0,return_inverse=True)
    statistic=0.
    df=-p
    low=0
    rows=[]
    for k,mask in enumerate(keys):
        subset=x[inverse==k]
        o=np.flatnonzero(mask)
        nr=len(subset)
        df+=len(o)
        if nr<5:
            low+=1
        difference=subset[:,o].mean(axis=0)-mu[o]
        local=cov[np.ix_(o,o)]
        statistic+=nr*float(difference@np.linalg.solve(local,difference))
        rows.append({
            "observed_pattern": "".join("1" if b else "0" for b in mask),
            "n_rows":nr, "n_observed_variables":len(o),
        })
    if df<1:
        raise ValueError("Not enough distinct missingness patterns for MCAR test")
    p_value=float(stats.chi2.sf(statistic,df))
    return {
        "test": "Little MCAR asymptotic chi-square",
        "statistic":float(statistic), "p_value":p_value,
        "degrees_of_freedom":int(df), "alpha":alpha,
        "reject_null":bool(p_value<alpha),
        "n_samples":n,"n_variables":p,"n_patterns":len(keys),
        "small_patterns":low,"em_iterations":iterations,
        "table":pd.DataFrame(rows),
        "interpretation_ar":(
            "اختبار Little يعتمد على اعتدالية متعددة المتغيرات وتقدير EM. "
            "رفض العدمية يقدم دليلًا ضد MCAR ولا يحدد MAR أو MNAR؛ "
            "عدم الرفض لا يثبت MCAR، والأنماط الصغيرة تضعف تقريب كاي تربيع."
        ),
        "interpretation_en":(
            "Little's approximate test assumes multivariate normality and "
            "identifiable EM covariance. Rejection suggests non-MCAR, not MAR "
            "versus MNAR. Non-rejection does not prove MCAR; sparse patterns "
            "may invalidate chi-square calibration."
        ),
    }


def missingness_association_screen(data, *, alpha=.05):
    """Exploratory numeric point-biserial missingness indicators vs observed columns.

    Applies Holm correction across *all* tested pairs. Missingness in the
    predictor itself is excluded pairwise with explicit n_used. This cannot
    prove MAR, MCAR or MNAR, and repeated persons need special handling.
    """
    if not 0<alpha<1:
        raise ValueError("alpha must be (0,1)")
    x=_numeric(data,min_rows=20)
    names=list(data.columns)
    rows=[]
    for j,target in enumerate(names):
        indicator=~np.isfinite(x[:,j])
        if not indicator.any() or indicator.all():
            continue
        for k,predictor in enumerate(names):
            if j==k:
                continue
            valid=np.isfinite(x[:,k])
            outcome=indicator[valid].astype(int)
            predictor_values=x[valid,k]
            if (len(outcome)<12 or outcome.sum()<4
                or (len(outcome)-outcome.sum())<4
                or predictor_values.std()<=0):
                continue
            corr,p=stats.pointbiserialr(outcome,predictor_values)
            if not np.isfinite([corr,p]).all():
                continue
            rows.append({
                "missing_target":target, "observed_predictor":predictor,
                "n_used":int(valid.sum()), "missing_group_n":int(outcome.sum()),
                "point_biserial_r":float(corr),"raw_p_value":float(p),
            })
    if not rows:
        raise ValueError("Insufficient observed predictors and missing/observed group sizes")
    frame=pd.DataFrame(rows)
    probs=frame["raw_p_value"].to_numpy()
    ix=np.argsort(probs)
    corrected=np.maximum.accumulate(
        np.minimum(1,probs[ix]*(len(probs)-np.arange(len(probs)))
    )
    adjusted=np.empty_like(corrected)
    adjusted[ix]=corrected
    frame["holm_p_value"]=adjusted
    frame["reject_null"]=adjusted<alpha
    return {
        "test":"Missingness versus observed predictors (exploratory)",
        "table":frame.sort_values("holm_p_value").reset_index(drop=True),
        "alpha":alpha, "n_samples":len(x),
        "tested_pairs":len(rows),
        "significant_comparisons":int((adjusted<alpha).sum()),
        "interpretation_ar":(
            "فحص استكشافي للارتباط بين غياب متغير والمتغيرات المرصودة مع تصحيح Holm. "
            "لا يثبت آلية MAR ولا يستبعد MNAR؛ تجنب استنتاج سببية الفقد."
        ),
        "interpretation_en":(
            "Exploratory point-biserial missingness screening with Holm "
            "multiplicity adjustment. Associations neither prove MAR nor "
            "exclude MNAR; records must be independent for these p-values."
        ),
    }


def multiple_impute_train_test_numeric(
    train, test, *, n_imputations=5, seed=42, max_iter=10,
):
    """Fit stochastic chained regression imputers on TRAIN only, transform TEST.

    Returns a list of pairs (completed_train, completed_test) as pandas frames.
    Basic sklearn IterativeImputer BayesianRidge posterior draws assume an
    approximate Gaussian regression model, not validated categorical bounds.
    Test values never affect training imputer fit, preventing this leakage path.
    """
    if type(n_imputations) is not int or not 2<=n_imputations<=20:
        raise ValueError("n_imputations must be an integer from 2 to 20")
    if type(max_iter) is not int or max_iter<2 or max_iter>100:
        raise ValueError("max_iter must lie between 2 and 100")
    train_x=_numeric(train,min_rows=15)
    test_x=_numeric(test,min_rows=1,min_observed=0)
    if list(train.columns)!=list(test.columns):
        raise ValueError("Train and test column names/order must match exactly")
    if any(np.isfinite(train_x[:,k]).sum()<8 for k in range(train_x.shape[1])):
        raise ValueError("All training features need at least eight observed values")
    if not np.isnan(train_x).any():
        raise ValueError("Training set has no missing cells to impute; no stochastic MI needed")
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    out=[]
    for k in range(n_imputations):
        imputer=IterativeImputer(
            random_state=seed+k, sample_posterior=True,
            max_iter=max_iter, skip_complete=False,
        )
        completed_train=imputer.fit_transform(train_x)
        completed_test=imputer.transform(test_x)
        if (not np.isfinite(completed_train).all()
            or not np.isfinite(completed_test).all()):
            raise ValueError("Imputation produced non-finite values")
        out.append((
            pd.DataFrame(completed_train,index=train.index,columns=train.columns),
            pd.DataFrame(completed_test,index=test.index,columns=test.columns),
        ))
    return {
        "imputations":out, "n_imputations":n_imputations,
        "n_train":len(train),"n_test":len(test),
        "interpretation_ar":(
            "استكمال متعدد باحتمالات من نموذج BayesianRidge عبر IterativeImputer "
            "مع الملاءمة على التدريب فقط. قد تكون القيم المستكملة خارج حدود المجال، "
            "وليس دليلًا على MAR أو صلاحية التنبؤ."
        ),
        "interpretation_en":(
            "Stochastic chained-regression multiple imputation fitted to "
            "training only, then transformed on test. Gaussian draws may "
            "violate plausible bounds; valid Rubin inference requires a "
            "congenial analysis model and missingness assumptions."
        ),
    }


def rubin_pool_scalar(estimates, variances, *, confidence_level=.95):
    """Rubin pooled scalar mean and total variance, including finite-m df.

    Does not invent a p-value. Normal approximation when between-imputation
    variance is exactly zero. Within-imputation variances are sampling SE^2.
    """
    q=np.asarray(estimates,dtype=float)
    u=np.asarray(variances,dtype=float)
    if (q.ndim!=1 or u.ndim!=1 or len(q)!=len(u) or len(q)<2
        or not np.isfinite(q).all() or not np.isfinite(u).all()
        or (u<0).any() or not 0<confidence_level<1):
        raise ValueError("Provide >=2 paired finite scalar estimates and nonnegative sampling variances")
    m=len(q)
    mean=float(q.mean())
    within=float(u.mean())
    between=float(q.var(ddof=1))
    total=within+(1+1/m)*between
    if total<=0:
        raise ValueError("Pooled uncertainty is zero; inference is undefined")
    if between<=0:
        df=float("inf")
        critical=float(stats.norm.ppf((1+confidence_level)/2))
    else:
        df=float((m-1)*(1+within/((1+1/m)*between))**2)
        critical=float(stats.t.ppf((1+confidence_level)/2,df))
    se=np.sqrt(total)
    return {
        "estimate":mean,"standard_error":float(se),
        "ci_lower":float(mean-critical*se),"ci_upper":float(mean+critical*se),
        "degrees_of_freedom":df, "n_imputations":m,
        "within_variance":within,"between_variance":between,
        "total_variance":float(total),"confidence_level":confidence_level,
        "interpretation_ar":(
            "دمج Rubin لتقدير أحادي البعد مع تباين داخل التعويضات وبينها. "
            "لا يختلق دلالة p؛ صلاحية الفواصل تتطلب نماذج استكمال وتحليل متوافقة."
        ),
        "interpretation_en":(
            "Rubin scalar pooling of within- and between-imputation uncertainty. "
            "No test p-value is asserted; interval validity depends on "
            "compatible imputation and analysis models and suitable sample sizes."
        ),
    }

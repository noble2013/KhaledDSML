"""Psychometric validity, reliability, agreement, and factor diagnostics.

Scores and psychometric thresholds are descriptive aids, never a substitute
for test design, external validity studies, or evidence of unidimensionality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _items(frame, *, min_items=2, min_rows=5):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame with questionnaire items as columns")
    if frame.shape[1] < min_items or frame.shape[0] < min_rows:
        raise ValueError("Not enough participants or questionnaire items")
    if not frame.columns.is_unique:
        raise ValueError("Item names must be unique")
    x = frame.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Explicitly handle missing or infinite item responses first")
    return x


def item_total_statistics(items):
    """Corrected item-total Pearson correlations and Cronbach alpha if deleted.

    Uses listwise complete cases. Correlations with a constant item/remaining
    sum are undefined and returned as NaN, not zero.
    """
    frame = pd.DataFrame(items) if not isinstance(items, pd.DataFrame) else items
    x = _items(frame)
    k, n = x.shape[1], x.shape[0]
    out = []
    for i, name in enumerate(frame.columns):
        remaining = np.delete(x, i, axis=1)
        score = remaining.sum(axis=1)
        corr = (float(np.corrcoef(x[:, i], score)[0,1])
                if np.std(x[:, i],ddof=1)>0 and np.std(score,ddof=1)>0 else np.nan)
        if k > 2 and np.var(score, ddof=1) > 0:
            variances = remaining.var(axis=0,ddof=1)
            alpha = (k-1)/(k-2)*(1-variances.sum()/np.var(score,ddof=1))
        else:
            alpha = np.nan
        out.append({
            "item":name,"mean":x[:,i].mean(),"std":x[:,i].std(ddof=1),
            "corrected_item_total_r":corr,"alpha_if_deleted":float(alpha),
        })
    return pd.DataFrame(out).set_index("item")


def split_half_reliability(items, *, method="odd_even") -> dict:
    """Spearman-Brown corrected correlation between two fixed item halves."""
    frame = pd.DataFrame(items) if not isinstance(items,pd.DataFrame) else items
    x = _items(frame,min_items=4)
    if method not in ("odd_even", "first_second"):
        raise ValueError("method must be odd_even or first_second")
    k=x.shape[1]
    if method=="odd_even":
        left,right=x[:,::2],x[:,1::2]
    else:
        left,right=x[:,:k//2],x[:,k//2:]
    a,b=left.sum(axis=1),right.sum(axis=1)
    if a.std(ddof=1)==0 or b.std(ddof=1)==0:
        raise ValueError("Half-test scores have no variation")
    r=float(np.corrcoef(a,b)[0,1])
    sb=float(2*r/(1+r)) if r>-1 else float("nan")
    return {
        "half_correlation":r, "spearman_brown":sb,
        "interpretation_ar":"ثبات تجزئة نصفية يعتمد على طريقة تقسيم البنود، وليس دليلًا على أحادية البعد.",
        "interpretation_en":"Split-half reliability depends on item allocation and does not prove unidimensionality.",
    }


def test_retest_correlation(first, second) -> dict:
    """Pearson test-retest association, not absolute agreement."""
    x,y=np.asarray(first,dtype=float).reshape(-1),np.asarray(second,dtype=float).reshape(-1)
    if len(x)<5 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Need >=5 finite paired observations")
    if x.std()==0 or y.std()==0:
        raise ValueError("Scores must vary at both occasions")
    r=stats.pearsonr(x,y)
    return {
        "correlation":float(r.statistic),"p_value":float(r.pvalue),"n_samples":len(x),
        "interpretation_ar":"الارتباط بين الزمنين يقيس الاستقرار النسبي، لا الاتفاق المطلق أو غياب انحياز القياس.",
        "interpretation_en":"Retest correlation measures rank stability, not absolute agreement or freedom from bias.",
    }


def cohen_kappa_agreement(rater_a, rater_b, *, weights=None) -> dict:
    """Cohen kappa for paired raters; optional ordinal linear/quadratic weights."""
    from sklearn.metrics import cohen_kappa_score
    a,b=np.asarray(rater_a),np.asarray(rater_b)
    if a.ndim!=1 or b.ndim!=1 or len(a)!=len(b) or len(a)<2:
        raise ValueError("Need equal-length paired 1D ratings")
    if weights not in (None,"linear","quadratic"):
        raise ValueError("weights must be None, linear, or quadratic")
    value=float(cohen_kappa_score(a,b,weights=weights))
    if not np.isfinite(value):
        raise ValueError("Kappa undefined because marginal ratings are degenerate")
    return {
        "kappa":value,"n_samples":len(a),"weights":weights,
        "interpretation_ar":"كابا تقيس اتفاق المحكّمين فوق الاتفاق المتوقع مصادفةً؛ تتأثر بشيوع الفئات.",
        "interpretation_en":"Kappa measures agreement beyond chance and is sensitive to category prevalence.",
    }


def fleiss_kappa_agreement(category_counts) -> dict:
    """Fleiss multi-rater kappa from n_subjects x n_categories counts."""
    counts=np.asarray(category_counts,dtype=float)
    if counts.ndim!=2 or counts.shape[0]<2 or counts.shape[1]<2 or not np.isfinite(counts).all() or (counts<0).any() or (counts!=np.floor(counts)).any():
        raise ValueError("Need nonnegative integer item-by-category vote counts")
    votes=counts.sum(axis=1)
    if min(votes)<2 or not np.all(votes==votes[0]):
        raise ValueError("All items need the same number of >=2 raters")
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for Fleiss kappa') from exc
    k=float(fleiss_kappa(counts,method="fleiss"))
    if not np.isfinite(k):
        raise ValueError("Kappa undefined for this vote distribution")
    return {
        "kappa":k,"n_items":counts.shape[0],"raters_per_item":int(votes[0]),
        "interpretation_ar":"اتفاق متعدد المحكّمين فوق الصدفة، مع افتراض فئات مستقلة ومحكّمين قابلين للتبادل.",
        "interpretation_en":"Chance-corrected multi-rater agreement for interchangeable raters and distinct categories.",
    }


def factor_scree_table(items) -> pd.DataFrame:
    """Eigenvalues and explained variance of questionnaire correlation matrix."""
    x=_items(pd.DataFrame(items) if not isinstance(items,pd.DataFrame) else items)
    if (x.std(axis=0,ddof=1)==0).any():
        raise ValueError("All questionnaire items must vary")
    r=np.corrcoef(x,rowvar=False)
    eig=np.linalg.eigvalsh(r)[::-1]
    eig=np.maximum(eig,0)
    return pd.DataFrame({
        "component":np.arange(1,len(eig)+1),
        "eigenvalue":eig,
        "explained_variance_ratio":eig/eig.sum(),
        "cumulative_explained_variance":np.cumsum(eig/eig.sum()),
        "kaiser_above_one":eig>1,
    })


def parallel_analysis(items, *, n_iter=200, seed=42, quantile=.95) -> pd.DataFrame:
    """Horn-style Monte Carlo correlation eigenvalue parallel analysis.

    Random normal matrices of identical n,p are used. This is a heuristic for
    selecting number of factors, not a test establishing a valid factor model.
    """
    data=np.asarray(_items(pd.DataFrame(items) if not isinstance(items,pd.DataFrame) else items))
    n,p=data.shape
    if n<=p or p<2 or (data.std(axis=0,ddof=1)==0).any():
        raise ValueError("Need n>p and nonconstant items")
    if type(n_iter) is not int or n_iter<30 or not 0<quantile<1:
        raise ValueError("Use >=30 iterations and a quantile in (0,1)")
    observed=np.linalg.eigvalsh(np.corrcoef(data,rowvar=False))[::-1]
    rng=np.random.default_rng(seed)
    simulated=np.empty((n_iter,p))
    for j in range(n_iter):
        v=rng.normal(size=(n,p))
        simulated[j]=np.linalg.eigvalsh(np.corrcoef(v,rowvar=False))[::-1]
    reference=np.quantile(simulated,quantile,axis=0)
    return pd.DataFrame({
        "component":np.arange(1,p+1),
        "observed_eigenvalue":observed,
        "random_eigenvalue_quantile":reference,
        "retain":observed>reference,
    })


def exploratory_factor_analysis(items, *, n_factors=2, random_state=42) -> dict:
    """Orthogonal unrotated exploratory factor analysis via sklearn FA.

    For ordinal Likert data, Pearson-based FA is only an approximation.
    No rotation, factor naming, or confirmatory fit claim is made.
    """
    from sklearn.decomposition import FactorAnalysis
    frame=pd.DataFrame(items) if not isinstance(items,pd.DataFrame) else items
    x=_items(frame)
    n,p=x.shape
    if n<=p or type(n_factors) is not int or not 1<=n_factors<p:
        raise ValueError("Need n>p and 1<=n_factors<p")
    if (x.std(axis=0,ddof=1)==0).any():
        raise ValueError("All items must vary")
    z=(x-x.mean(axis=0))/x.std(axis=0,ddof=1)
    model=FactorAnalysis(n_components=n_factors,random_state=random_state).fit(z)
    loadings=model.components_.T
    return {
        "loadings":pd.DataFrame(loadings,index=frame.columns,
                                columns=[f"factor_{i+1}" for i in range(n_factors)]),
        "uniqueness":pd.Series(model.noise_variance_,index=frame.columns),
        "log_likelihood_per_sample":float(model.score(z)),
        "interpretation_ar":"تحليل عوامل استكشافي بلا دوران؛ لا يثبت صدق البناء ويحتاج تحليلًا تأكيديًا مستقلًا.",
        "interpretation_en":"Unrotated exploratory factor analysis; construct validity requires separate confirmatory evidence.",
    }


def convergent_validity(loadings, *, error_variances=None) -> dict:
    """AVE and composite reliability from *standardized* single-factor loadings."""
    l=np.asarray(loadings,dtype=float).reshape(-1)
    if len(l)<2 or not np.isfinite(l).all() or np.any(np.abs(l)>1):
        raise ValueError("Need at least two standardized loadings in [-1,1]")
    if error_variances is None:
        errors=1-l*l
    else:
        errors=np.asarray(error_variances,dtype=float).reshape(-1)
    if len(errors)!=len(l) or not np.isfinite(errors).all() or (errors<0).any():
        raise ValueError("Need nonnegative residual variances matching loadings")
    ave=float(np.sum(l*l)/(np.sum(l*l)+np.sum(errors)))
    cr=float(np.sum(l)**2/(np.sum(l)**2+np.sum(errors)))
    return {
        "ave":ave,"composite_reliability":cr,
        "interpretation_ar":"AVE وCR يعتمدان على تشبعات معيارية وافتراضات نموذج القياس، ولا يثبتان بمفردهما الصدق.",
        "interpretation_en":"AVE and composite reliability assume standardized loadings and an appropriate measurement model.",
    }


def htmt_ratio(items, groups) -> dict:
    """Heterotrait–monotrait ratio for exactly two questionnaire constructs."""
    frame=pd.DataFrame(items) if not isinstance(items,pd.DataFrame) else items
    x=_items(frame,min_items=4)
    if not isinstance(groups,(list,tuple)) or len(groups)!=2:
        raise ValueError("groups must contain exactly two sets of distinct item column names")
    a,b=[list(group) for group in groups]
    if len(a)<2 or len(b)<2 or set(a)&set(b) or any(col not in frame.columns for col in a+b):
        raise ValueError("Need two nonoverlapping constructs with >=2 valid items each")
    corr=frame[a+b].corr().abs()
    between=corr.loc[a,b].to_numpy().mean()
    within_a=corr.loc[a,a].to_numpy()[np.triu_indices(len(a),1)].mean()
    within_b=corr.loc[b,b].to_numpy()[np.triu_indices(len(b),1)].mean()
    if within_a<=0 or within_b<=0 or not np.isfinite([between,within_a,within_b]).all():
        raise ValueError("HTMT undefined for zero within-construct correlations")
    return {
        "htmt":float(between/np.sqrt(within_a*within_b)),
        "interpretation_ar":"HTMT مؤشر لتمييز البعدين؛ ينبغي تقدير عدم اليقين وفحص صلاحية بنية القياس.",
        "interpretation_en":"HTMT is a discriminant-validity indicator, not proof without construct-model assessment.",
    }


def fornell_larcker_matrix(correlations, ave) -> pd.DataFrame:
    """Show sqrt(AVE) on diagonal and construct correlations off-diagonal."""
    matrix=pd.DataFrame(correlations).astype(float)
    if matrix.shape[0]!=matrix.shape[1] or not matrix.index.equals(matrix.columns):
        raise ValueError("Need square named construct-correlation matrix")
    if not np.isfinite(matrix.to_numpy()).all() or not np.allclose(matrix,matrix.T,atol=1e-8):
        raise ValueError("Correlation matrix must be finite and symmetric")
    averages=pd.Series(ave)
    if not matrix.index.isin(averages.index).all():
        raise KeyError("AVE values must be keyed by all construct names")
    out=matrix.copy()
    for name in matrix.index:
        v=float(averages[name])
        if not 0<=v<=1:
            raise ValueError("AVE must be in [0,1]")
        out.loc[name,name]=np.sqrt(v)
    return out


def harman_single_factor(items) -> dict:
    """Descriptive first principal-component share, NOT a method-bias test."""
    table=factor_scree_table(items)
    share=float(table["explained_variance_ratio"].iloc[0])
    return {
        "first_component_variance_ratio":share,
        "interpretation_ar":"نسبة التباين للمكون الأول مؤشر وصفي فقط؛ اختبار هارمان لا يكفي لاستبعاد تحيز الطريقة المشتركة.",
        "interpretation_en":"First-component variance is descriptive; Harman's single-factor check cannot rule out common-method bias.",
    }

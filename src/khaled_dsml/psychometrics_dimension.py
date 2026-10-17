"""Reliability and dimension diagnostics with explicit methodological limits.

All reliability summaries rely on particular measurement assumptions. They
do not establish construct validity, measurement invariance, or causality.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def omega_single_factor(items):
    """One-factor omega total approximation via standardized factor analysis.

    This is NOT hierarchical omega; assumes one common factor and independent
    residuals, and is inappropriate when a strong multidimensional model fits.
    """
    from sklearn.decomposition import FactorAnalysis
    x = np.asarray(items, dtype=float)
    if (x.ndim != 2 or x.shape[0] < 15 or x.shape[1] < 3
        or x.shape[0] <= x.shape[1] or not np.isfinite(x).all()
        or np.any(x.std(axis=0,ddof=1) <= 0)):
        raise ValueError("Need complete n-by-p numeric responses, n>p and at least 3 variable items")
    standardized = (x-x.mean(axis=0))/x.std(axis=0,ddof=1)
    model = FactorAnalysis(n_components=1,random_state=42).fit(standardized)
    lam = model.components_[0]
    errors = np.asarray(model.noise_variance_,dtype=float)
    if np.any(errors <= 0):
        raise ValueError("Estimated item-specific variances must be positive")
    common = float(lam.sum()**2)
    value = float(common/(common+errors.sum()))
    return {
        "omega_total_one_factor":value,"n_samples":len(x),"n_items":x.shape[1],
        "loadings":lam.copy(),
        "interpretation_ar":"أوميغا نموذج عامل واحد مع أخطاء مستقلة؛ لا يثبت صدق الأداة أو أحادية البعد.",
        "interpretation_en":"One-factor omega assumes independent uniquenesses; it does not prove construct validity or unidimensionality.",
    }


def intraclass_correlation(ratings, *, model="ICC2") -> dict:
    """Single-rater, two-way random absolute (ICC2) or mixed consistency (ICC3).

    Rows are targets, columns raters; balanced, complete numerical ratings
    and independent targets are required. No confidence interval is invented.
    """
    x = np.asarray(ratings, dtype=float)
    if (x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 2
        or not np.isfinite(x).all() or model not in ("ICC2","ICC3")):
        raise ValueError("Require complete numeric n-by-r ratings, n>=5, r>=2 and model ICC2/ICC3")
    n, k = x.shape
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    ms_rows = float(k*np.sum((row_means-grand)**2)/(n-1))
    ms_cols = float(n*np.sum((col_means-grand)**2)/(k-1))
    residual = x-row_means[:,None]-col_means[None,:]+grand
    mse = float(np.sum(residual**2)/((n-1)*(k-1)))
    denominator = (ms_rows+(k-1)*mse +
                   (k*(ms_cols-mse)/n if model=="ICC2" else 0))
    if denominator<=0:
        raise ValueError("ICC undefined because target/measurement variance is degenerate")
    icc=float((ms_rows-mse)/denominator)
    return {
        "icc":icc,"model":model,"n_targets":n,"n_raters":k,
        "interpretation_ar":(
            "ICC2 للاتفاق المطلق بين محكّمين عشوائيين؛ ICC3 لاتساق محكّمين ثابتين، "
            "والفترتان والثبات يعتمدان على تصميم القياس."
        ),
        "interpretation_en":(
            "ICC2 estimates single-rater absolute agreement with random raters; "
            "ICC3 measures consistency with fixed raters. No confidence interval is inferred."
        ),
    }


def krippendorff_nominal_alpha(ratings) -> dict:
    """Krippendorff nominal alpha with missing individual ratings.

    Rows=units, columns=raters; missing values are None/NaN. Units with fewer
    than two valid ratings are excluded from the coincidence distribution.
    """
    a = np.asarray(ratings,dtype=object)
    if a.ndim!=2 or a.shape[0]<3 or a.shape[1]<2:
        raise ValueError("Need units-by-raters data with at least three units and two raters")
    coincidence = {}
    totals = Counter()
    n_pairable_units = 0
    for row in a:
        valid = [str(value) for value in row if not pd.isna(value)]
        m = len(valid)
        if m<2:
            continue
        n_pairable_units += 1
        counts=Counter(valid)
        for k,count in counts.items():
            totals[k]+=count
        for c, ni in counts.items():
            for d, nj in counts.items():
                pair=(c,d)
                coincidence[pair]=coincidence.get(pair,0.) + ni*(nj-(1 if c==d else 0))/(m-1)
    N=sum(totals.values())
    if n_pairable_units<2 or N<4 or len(totals)<2:
        raise ValueError("Need at least two pairable units and two nominal rating categories")
    disagreement=sum(value for (c,d),value in coincidence.items() if c!=d)/N
    expected=sum(ni*nj for i,ni in totals.items() for j,nj in totals.items() if i!=j)/(N*(N-1))
    if expected<=0:
        raise ValueError("Nominal disagreement expected by chance must be positive")
    return {
        "krippendorff_alpha":float(1-disagreement/expected),
        "n_pairable_units":n_pairable_units,
        "n_valid_ratings":N,
        "interpretation_ar":"ألفا كريبندورف الاسمية تسمح بفقد بعض تقييمات المحكّمين؛ الاتفاق يعتمد على تعريف الفئات.",
        "interpretation_en":"Nominal Krippendorff alpha tolerates missing ratings; agreement depends on category design.",
    }


def trustworthiness_embedding(original, embedded, *, n_neighbors=5) -> dict:
    """Trustworthiness of a low-dimensional embedding vs original distances."""
    from sklearn.manifold import trustworthiness
    x, z = np.asarray(original,dtype=float),np.asarray(embedded,dtype=float)
    if (x.ndim!=2 or z.ndim!=2 or x.shape[0]!=z.shape[0]
        or len(x)<8 or z.shape[1]<1 or not np.isfinite(x).all()
        or not np.isfinite(z).all() or type(n_neighbors) is not int
        or n_neighbors<1 or n_neighbors >= (len(x)-1)/2):
        raise ValueError("Require two finite n-by-p embeddings, same n, 1<=neighbors<(n-1)/2")
    score=float(trustworthiness(x,z,n_neighbors=n_neighbors))
    return {
        "trustworthiness":score,"n_samples":len(x),"neighbors":n_neighbors,
        "interpretation_ar":"مدى احتفاظ التمثيل المنخفض بالجيران المحليين؛ ليس ضمانًا لحفظ البنية العالمية.",
        "interpretation_en":"Local-neighbor preservation, not a guarantee of global geometry or cluster validity.",
    }


def distance_correlation(a, b) -> float:
    """Biased sample distance-correlation statistic in [0,1], no p-value.

    Uses Euclidean double-centered pairwise distance matrices, O(n²).
    It is descriptive; significance requires a suitable permutation design.
    """
    x,y=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if x.ndim==1:
        x=x[:,None]
    if y.ndim==1:
        y=y[:,None]
    if (x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<5
        or len(x)>2000 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Need 5-2000 aligned finite observations; O(n²) memory")
    if np.any(x.std(axis=0)==0) or np.any(y.std(axis=0)==0):
        raise ValueError("Constant input dimensions are not supported")
    A=squareform(pdist(x))
    B=squareform(pdist(y))
    A=A-A.mean(axis=0)[None,:]-A.mean(axis=1)[:,None]+A.mean()
    B=B-B.mean(axis=0)[None,:]-B.mean(axis=1)[:,None]+B.mean()
    xy=float(np.mean(A*B))
    xx=float(np.mean(A*A))
    yy=float(np.mean(B*B))
    if xx<=0 or yy<=0:
        raise ValueError("Distance correlation undefined for zero distance variance")
    return float(min(1,max(0,np.sqrt(max(0,xy)/np.sqrt(xx*yy)))))


def bootstrap_cluster_stability(
    X, *, n_clusters=3, repetitions=30, fraction=.8, seed=42,
):
    """Illustrative KMeans cluster stability under bootstrap resampling.

    Fits reference KMeans once and bootstrapped KMeans repeatedly, then ARI
    on all original observations; not proof that substantive clusters exist.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    data=np.asarray(X,dtype=float)
    if (data.ndim!=2 or len(data)<20 or not np.isfinite(data).all()
        or type(n_clusters) is not int or not 2<=n_clusters<len(data)//2
        or type(repetitions) is not int or not 10<=repetitions<=1000
        or not .5<=fraction<=1):
        raise ValueError("Need >=20 finite samples, valid clusters, >=10 repetitions, 0.5<=fraction<=1")
    rng=np.random.default_rng(seed)
    reference=KMeans(n_clusters=n_clusters,random_state=seed,n_init=10).fit(data)
    k=int(np.ceil(len(data)*fraction))
    scores=[]
    for j in range(repetitions):
        idx=rng.choice(len(data),size=k,replace=True)
        if len(np.unique(data[idx],axis=0))<n_clusters:
            raise ValueError("Insufficient distinct bootstrap samples for the requested clusters")
        model=KMeans(n_clusters=n_clusters,random_state=seed+j+1,n_init=10).fit(data[idx])
        scores.append(float(adjusted_rand_score(reference.labels_,model.predict(data))))
    arr=np.asarray(scores)
    return {
        "mean_ari":float(arr.mean()),"sd_ari":float(arr.std(ddof=1)),
        "lower_ari":float(np.quantile(arr,.025)),
        "upper_ari":float(np.quantile(arr,.975)),
        "replications":repetitions,"scores":scores,
        "interpretation_ar":"استقرار KMeans بإعادة أخذ العينات مؤشر تجريبي ويعتمد على طريقة التجميع وعدد العناقيد.",
        "interpretation_en":"KMeans bootstrap label stability is conditional on the chosen algorithm and k; not evidence of meaningful clusters.",
    }

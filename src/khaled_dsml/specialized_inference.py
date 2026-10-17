"""Specialized limited-scope tests for ordinal associations and post-hoc ranks.

These functions deliberately reject unsupported inference rather than output
made-up significance. An effect-size estimate alone is never a causal claim.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar


def _alpha(value):
    if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or not 0<value<1:
        raise ValueError("alpha must be a finite probability strictly inside (0,1)")
    return float(value)


def williams_steiger_overlapping_correlations(x,y,z,*,alpha=.05):
    """Williams/Steiger t test of r(x,y) == r(x,z) in ONE iid sample.

    Uses the Williams 1959 / Steiger 1980 dependent-overlapping
    correlation statistic with determinant of the 3x3 correlation matrix.
    Requires multivariate-normal iid observations for parametric t inference,
    n >= 10, full-rank trivariate correlation and no constant columns.
    """
    alpha=_alpha(alpha)
    try:a=np.column_stack([np.asarray(v,dtype=float) for v in (x,y,z)])
    except (ValueError,TypeError) as exc:
        raise ValueError("Aligned numeric 1D vectors required") from exc
    if a.ndim!=2 or a.shape[1]!=3 or len(a)<10 or not np.isfinite(a).all():
        raise ValueError("Need >=10 complete matched iid triples")
    if np.any(a.std(axis=0,ddof=1)<=1e-12):
        raise ValueError("Each correlation variable must vary")
    r=np.corrcoef(a,rowvar=False)
    rxy,rxz,ryz=float(r[0,1]),float(r[0,2]),float(r[1,2])
    determinant=float(np.linalg.det(r))
    if determinant<=1e-10 or abs(ryz)>=1-1e-12:
        raise ValueError("Dependent-correlation test requires positive-definite correlation matrix")
    n=len(a)
    denom=(2*(n-1)/(n-3)*determinant
           + ((rxy+rxz)**2/4)*(1-ryz)**3)
    if denom<=0:raise ValueError("Williams statistic undefined")
    t=float((rxy-rxz)*np.sqrt((n-1)*(1+ryz)/denom))
    p=float(2*stats.t.sf(abs(t),n-3))
    return {
        "test":"Williams-Steiger overlapping correlations t",
        "n_samples":n,"r_xy":rxy,"r_xz":rxz,"r_yz":ryz,
        "correlation_matrix_determinant":determinant,
        "statistic":t,"degrees_of_freedom":n-3,"p_value":p,
        "alpha":alpha,"reject_equal_correlations":bool(p<alpha),
        "table":pd.DataFrame(r,index=["X","Y","Z"],columns=["X","Y","Z"]),
        "interpretation_ar":(
            "اختبار t لويلامز–ستيغر يقارن ارتباط X مع Y وارتباط X مع Z "
            "عندما تشترك المقارنتان في العينة والمتغير X. يفترض "
            "مشاهدات مستقلة وملائمة طبيعية ثلاثية؛ ليس لاختبار "
            "ارتباط عينتين مستقلتين، ولا يثبت السببية."
        ),
        "interpretation_en":(
            "Williams–Steiger dependent-overlapping correlation t test "
            "compares r(X,Y) with r(X,Z) on the same iid triples. "
            "Trivariate normality and full-rank correlations support "
            "the parametric reference. Not for independent samples."
        ),
    }


def effect_size_r_from_z(z_statistic,n_samples):
    """Signed descriptive r=z/sqrt(n), only for valid z-based effect mapping.

    This common rank-test approximation should not be applied to arbitrary
    z or nonindependent sampling. r may exceed |1| for a general z input:
    such cases are rejected instead of silently clipping the effect.
    """
    if (isinstance(z_statistic,(bool,np.bool_)) or not np.isscalar(z_statistic)
        or not np.isfinite(z_statistic) or type(n_samples) is not int
        or n_samples<2):
        raise ValueError("Provide a finite z statistic and integer n>=2")
    value=float(z_statistic/np.sqrt(n_samples))
    if abs(value)>1+1e-10:
        raise ValueError("z/sqrt(n) outside [-1,1]; check statistic and sample count")
    return {
        "test":"Z-based signed effect size r",
        "n_samples":n_samples,"z_statistic":float(z_statistic),
        "effect_size_r":value,
        "interpretation_ar":(
            "حجم أثر وصفي تقريبي r = z/√n لبعض اختبارات الرتب. "
            "لا يجوز تعميم التحويل على كل إحصائية z أو بيانات "
            "مترابطة، ولا يُستنتج منه إثبات الدلالة أو السببية."
        ),
        "interpretation_en":(
            "Signed descriptive r=z/sqrt(n) for settings where this "
            "rank-test effect-size conversion is justified. Not a "
            "universal transformation for all z statistics, or a p-value."
        ),
    }


def conover_iman_posthoc(*groups,alpha=.05,adjust="holm"):
    """Conover–Iman all-pairs post-Kruskal comparisons WITHOUT tied ranks.

    S²=sum((rank - mean_rank)^2)/(N-1); pooled factor
    (N-1-H)/(N-k), with Kruskal H and k groups. Two-sided t on N-k df.
    The tie-free scope is deliberate; tied ranks need a separately verified
    correction. Holm controls familywise error for the tested family.
    """
    alpha=_alpha(alpha)
    if len(groups)<3 or len(groups)>15 or adjust!="holm":
        raise ValueError("Conover–Iman here needs 3..15 independent groups and Holm correction")
    try:samples=[np.asarray(g,dtype=float) for g in groups]
    except (TypeError,ValueError) as exc:
        raise ValueError("Groups must contain finite numeric observations") from exc
    if any(g.ndim!=1 or len(g)<4 or not np.isfinite(g).all() for g in samples):
        raise ValueError("Every group needs at least 4 finite independent observations")
    pooled=np.concatenate(samples)
    if len(np.unique(pooled))!=len(pooled):
        raise ValueError("Conover–Iman in this implementation requires no tied pooled ranks")
    n=len(pooled);k=len(samples);rank=stats.rankdata(pooled)
    means=[];offset=0
    for sample in samples:
        means.append(float(rank[offset:offset+len(sample)].mean()))
        offset+=len(sample)
    h=float(stats.kruskal(*samples).statistic)
    s2=float(np.var(rank,ddof=1))
    factor=(n-1-h)/(n-k)
    if factor<=0:raise ValueError("Within-group rank dispersion is too small")
    records=[]
    for i,j in combinations(range(k),2):
        t=abs(means[i]-means[j])/np.sqrt(s2*factor*(1/len(samples[i])+1/len(samples[j])))
        p=float(2*stats.t.sf(t,n-k))
        records.append({"group_i":i+1,"group_j":j+1,"t_statistic":float(t),
                        "raw_p":p})
    table=pd.DataFrame(records)
    raw=table["raw_p"].to_numpy()
    sorted_idx=np.argsort(raw)
    adj=np.empty(len(raw))
    adj[sorted_idx]=np.clip(
        np.maximum.accumulate((len(raw)-np.arange(len(raw)))*raw[sorted_idx]),0,1
    )
    table["holm_p"]=adj
    table["reject_adjusted"]=adj<alpha
    return {
        "test":"Conover-Iman tie-free posthoc comparisons",
        "n_samples":n,"n_groups":k,"alpha":alpha,
        "kruskal_h":h,"n_pairs":len(records),
        "n_significant_pairs":int(table["reject_adjusted"].sum()),
        "table":table,
        "interpretation_ar":(
            "مقارنات كونوفِر–إيمان للرتب بعد Kruskal–Wallis مع Holm "
            "لتعديل القيم الاحتمالية. هذه النسخة تتحقق من عدم تساوي "
            "الرتب وترفض البيانات ذات الروابط بدل تطبيق تصحيح غير مدقق. "
            "تستلزم عينات مستقلة؛ استكشافية إذا اختيرت المقارنات بعد النتائج."
        ),
        "interpretation_en":(
            "Tie-free Conover–Iman pairwise pooled-rank comparisons after "
            "Kruskal–Wallis, with Holm familywise adjustment. "
            "Tied pooled ranks are rejected rather than claiming an "
            "untested correction. Assumes independent groups."
        ),
    }


def polychoric_latent_correlation(x,y,*,max_categories=8):
    """Ordinal latent Gaussian correlation by contingency-table MLE.

    Categories must be ordered integer codes. Thresholds are the observed
    marginal normal quantiles; rho maximizes the bivariate-normal rectangle
    multinomial log likelihood, conditional on estimated thresholds. The
    estimate presumes an underlying bivariate normal latent pair and iid
    observations. No misleading naive Wald SE or p-value is returned.
    """
    try:
        a=np.asarray(x);b=np.asarray(y)
        af=np.asarray(x,dtype=float);bf=np.asarray(y,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("Ordinal categories must be integer numeric codes") from exc
    if (a.ndim!=1 or b.ndim!=1 or len(a)!=len(b) or len(a)<80
        or not np.isfinite(af).all() or not np.isfinite(bf).all()
        or not np.array_equal(af,np.floor(af))
        or not np.array_equal(bf,np.floor(bf))
        or type(max_categories) is not int or not 2<=max_categories<=15):
        raise ValueError(">=80 aligned, finite ordinal integer pairs and max_categories 2..15")
    la,ia=np.unique(a,return_inverse=True)
    lb,ib=np.unique(b,return_inverse=True)
    if not 2<=len(la)<=max_categories or not 2<=len(lb)<=max_categories:
        raise ValueError("Require 2..max_categories ordered levels per variable")
    n=len(a)
    counts=np.zeros((len(la),len(lb)),dtype=int)
    np.add.at(counts,(ia,ib),1)
    if min(counts.sum(axis=0).min(),counts.sum(axis=1).min())<4:
        raise ValueError("Each ordinal level must contain at least 4 cases")
    tha=stats.norm.ppf(np.r_[0,np.cumsum(counts.sum(axis=1))/n])
    thb=stats.norm.ppf(np.r_[0,np.cumsum(counts.sum(axis=0))/n])
    from scipy.stats import multivariate_normal
    from scipy.special import ndtr

    def cdf(u,v,rho):
        if u==-np.inf or v==-np.inf:return 0.
        if u==np.inf:return float(ndtr(v))
        if v==np.inf:return float(ndtr(u))
        return float(multivariate_normal.cdf((u,v),cov=[[1,rho],[rho,1]]))
    def negloglik(rho):
        score=0.
        for i in range(len(la)):
            for j in range(len(lb)):
                nij=int(counts[i,j])
                if nij==0:continue
                p=(cdf(tha[i+1],thb[j+1],rho)-cdf(tha[i],thb[j+1],rho)
                   -cdf(tha[i+1],thb[j],rho)+cdf(tha[i],thb[j],rho))
                if p<=0:return 1e20
                score-=nij*np.log(p)
        return score
    optimization=minimize_scalar(negloglik,method="bounded",
                                 bounds=(-.995,.995),
                                 options={"xatol":1e-5,"maxiter":150})
    if not optimization.success or not np.isfinite(optimization.fun):
        raise ValueError("Polychoric likelihood optimization failed")
    return {
        "test":"Polychoric latent-normal ordinal association",
        "n_samples":n,"n_levels_x":len(la),"n_levels_y":len(lb),
        "polychoric_rho":float(optimization.x),
        "negative_log_likelihood":float(optimization.fun),
        "table":pd.DataFrame(counts,
                             index=[str(v) for v in la],
                             columns=[str(v) for v in lb]),
        "interpretation_ar":(
            "تقدير ارتباط بوليكوريك بين قياسين رتبيين وفق فرض "
            "التوزيع الطبيعي الثنائي للمتغيرين الكامنين وعتبات الفئات "
            "من التكرارات الهامشية. لا تتضمن الدالة خطأً معياريًا "
            "أو قيمة p أو فاصل ثقة موثوقًا؛ تجنب تفسيره سببياً."
        ),
        "interpretation_en":(
            "Polychoric association from a latent bivariate Gaussian "
            "ordinal threshold model, with marginal-normal thresholds "
            "and bivariate rectangle likelihood optimization. "
            "No p-value, reliable standard error or causal claim is implied."
        ),
    }

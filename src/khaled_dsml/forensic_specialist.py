"""Research arithmetic and selective-reporting SCREENS, not misconduct tests."""
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation, localcontext
import numpy as np
import pandas as pd
from scipy import stats


def _decimal(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(label+" must not be boolean")
    try: d=Decimal(str(value))
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError(label+" must be numeric") from exc
    if not d.is_finite(): raise ValueError(label+" must be finite")
    return d


def grimmer_variance_consistency(reported_mean, reported_sd, sample_size, *,
                                  decimals_mean=2, decimals_sd=2,
                                  minimum_score=1, maximum_score=5):
    """Exact bounded-integer GRIMMER-style mean and *sample SD* feasibility.

    Enumerate reachable (sum, squared sum) with integer bitset DP.
    Sample SD uses ddof=1. Both reported values use ROUND_HALF_UP. Bound
    runtime with n<=50 and <=9 consecutive score levels. Failing is a
    reporting arithmetic flag, NEVER evidence of research fabrication.
    """
    if (type(sample_size) is not int or not 2<=sample_size<=50
        or type(decimals_mean) is not int or not 0<=decimals_mean<=4
        or type(decimals_sd) is not int or not 0<=decimals_sd<=4
        or type(minimum_score) is not int or type(maximum_score) is not int
        or not -100<=minimum_score<=maximum_score<=100
        or maximum_score-minimum_score>8):
        raise ValueError("Require n=2..50, 0..4 decimal places, ordered <=9 integer scale levels")
    mu=_decimal(reported_mean,"reported_mean")
    sd=_decimal(reported_sd,"reported_sd")
    qm=Decimal(1).scaleb(-decimals_mean)
    qs=Decimal(1).scaleb(-decimals_sd)
    if (mu!=mu.quantize(qm) or sd!=sd.quantize(qs)
        or sd<0 or mu<minimum_score-1 or mu>maximum_score+1):
        raise ValueError("Reported mean and SD must have specified rounding and valid scale")
    width=maximum_score-minimum_score
    # A bit at position q means normalized squared total Q=q is possible.
    reachable={0:1}
    for _ in range(sample_size):
        fresh={}
        for total,mask in reachable.items():
            for value in range(width+1):
                key=total+value
                fresh[key]=fresh.get(key,0) | (mask << (value*value))
        reachable=fresh
    n=Decimal(sample_size)
    targets=[
        (tot,mask) for tot,mask in reachable.items()
        if (Decimal(tot+sample_size*minimum_score)/n).quantize(
            qm,rounding=ROUND_HALF_UP)==mu
    ]
    witness=None
    checked=0
    with localcontext() as ctx:
        ctx.prec=70
        for total,mask in targets:
            while mask:
                lowest=mask & -mask
                q=lowest.bit_length()-1
                mask^=lowest
                checked+=1
                variance=(Decimal(q)-Decimal(total*total)/n)/(n-1)
                if variance<0:continue
                s=variance.sqrt()
                if s.quantize(qs,rounding=ROUND_HALF_UP)==sd:
                    witness={
                        "integer_total":int(total+sample_size*minimum_score),
                        "integer_squared_total":int(q+2*minimum_score*total+
                            sample_size*minimum_score**2),
                        "possible_sample_sd":float(s),
                    }
                    break
            if witness is not None:break
    return {
        "test":"GRIMMER rounded mean and sample SD feasibility",
        "n_samples":sample_size,"reported_mean":str(mu),
        "reported_sd":str(sd),"decimals_mean":decimals_mean,
        "decimals_sd":decimals_sd,"minimum_score":minimum_score,
        "maximum_score":maximum_score,
        "arithmetically_consistent":witness is not None,
        "witness_moments":witness,
        "n_compatible_mean_totals":len(targets),
        "n_moment_pairs_inspected":checked,
        "interpretation_ar":(
            "فحص GRIMMER لإمكان المتوسط والانحراف المعياري للعينة "
            "من درجات صحيحة محدودة، وفق تقريب نصف للأعلى وddof=1. "
            "التناقض الحسابي يستدعي مراجعة التقرير ولا يدل على "
            "التزوير؛ والتوافق لا يثبت صحة البيانات الأصلية."
        ),
        "interpretation_en":(
            "Exact bounded-integer GRIMMER-style feasibility of a rounded "
            "mean and SAMPLE SD (ddof=1), both ROUND_HALF_UP. A mismatch "
            "is not proof of misconduct and a match does not authenticate "
            "any observations."
        ),
    }


def pcurve_right_skew_screen(significant_p_values, *, threshold=.05,
                             midpoint=None, alpha=.05):
    """Exploratory half-p-curve binomial right-skew screen, NOT full p-curve.

    Under a prespecified conditional-uniform null with independent exact
    significant p-values, the count below midpoint is Binomial(n,midpoint/
    threshold). Does not estimate underlying power, test p-hacking or
    correct selective publication, dependencies and post-hoc study search.
    """
    if (isinstance(threshold,(bool,np.bool_)) or not np.isscalar(threshold)
        or not np.isfinite(threshold) or not 0<threshold<1
        or isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha)
        or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError("threshold and alpha must be finite probabilities")
    if midpoint is None:midpoint=threshold/2
    if (isinstance(midpoint,(bool,np.bool_)) or not np.isscalar(midpoint)
        or not np.isfinite(midpoint) or not 0<midpoint<threshold):
        raise ValueError("midpoint must be strictly inside the significance cutoff")
    try:p=np.asarray(significant_p_values,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("Require exact numeric p-values") from exc
    if (p.ndim!=1 or len(p)<5 or not np.isfinite(p).all()
        or np.any(p<=0) or np.any(p>=threshold)):
        raise ValueError("Require >=5 exact significant numeric p-values (0,threshold)")
    lower=int(np.count_nonzero(p<midpoint))
    n=len(p)
    prob=float(midpoint/threshold)
    b=stats.binomtest(lower,n,prob,alternative="greater")
    return {
        "test":"Exploratory significant-p right-skew half-curve",
        "n_samples":n,"n_p_below_midpoint":lower,
        "n_p_above_or_equal_midpoint":n-lower,
        "significance_cutoff":float(threshold),"split_midpoint":float(midpoint),
        "expected_below_midpoint_under_uniform":prob,
        "statistic":lower,"p_value":float(b.pvalue),"alpha":float(alpha),
        "reject_uniform_significant_p":bool(b.pvalue<alpha),
        "table":pd.DataFrame({"p_value":p,
                              "half":["lower" if x<midpoint else "upper" for x in p]}),
        "interpretation_ar":(
            "فحص استكشافي لنزوع قيم p الدالة نحو الصغر مقابل فرض "
            "التوزيع المنتظم المشروط بالدلالة واستقلال الاختبارات. "
            "لا يمثل p-curve كاملًا أو تقديرًا للقدرة ولا إثباتًا "
            "لانحياز النشر؛ الانتقاء والتبعية قد يفسدان المرجع."
        ),
        "interpretation_en":(
            "Exploratory half-curve right-skew count tested against "
            "conditional-uniform independent exact significant p-values. "
            "NOT full p-curve, a statistical power estimator, or proof "
            "of selective reporting. Selection and dependence may "
            "invalidate the test."
        ),
    }

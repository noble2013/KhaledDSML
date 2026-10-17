"""Prespecified experimental monitoring and research diagnostics.

No function here establishes causality from observational data. Sequential
boundaries, multiple looks and covariate-adjusted A/B comparisons require
design review before confirmatory use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bernoulli_sprt(observations, *, p0, p1, alpha=.05, beta=.20):
    """Wald's simple-vs-simple sequential probability ratio test.

    H0: Bernoulli(p0); H1: Bernoulli(p1), with 0<p0<p1<1, iid trials.
    Evidence is logged until the FIRST boundary crossing. alpha/beta are
    design targets and only approximations for realized error probabilities.
    If no boundary crosses, the result is INCONCLUSIVE, not acceptance of H0.
    """
    x = np.asarray(observations)
    if (x.ndim != 1 or not len(x) or x.dtype.kind not in "biu"
        or not np.isin(x, [0, 1]).all()):
        raise ValueError("observations must be a nonempty 1D binary integer series")
    params = (p0,p1,alpha,beta)
    if any(isinstance(v,(bool,np.bool_)) or not np.isscalar(v) or not np.isfinite(v)
           for v in params) or not (0 < p0 < p1 < 1 and 0 < alpha < 1 and 0 < beta < 1 and alpha+beta<1):
        raise ValueError("Require 0<p0<p1<1 and positive alpha/beta with alpha+beta<1")
    high = float(np.log((1-beta)/alpha))
    low = float(np.log(beta/(1-alpha)))
    llr = np.cumsum(np.where(x==1, np.log(p1/p0), np.log((1-p1)/(1-p0))))
    high_hits = np.flatnonzero(llr >= high)
    low_hits = np.flatnonzero(llr <= low)
    candidates = [int(v[0]) for v in (high_hits,low_hits) if len(v)]
    stop = min(candidates) if candidates else len(x)-1
    decision = ("favor_H1" if llr[stop]>=high else
                "favor_H0" if llr[stop]<=low else "inconclusive")
    return {
        "test":"Bernoulli Wald SPRT", "n_samples":int(stop+1),
        "n_available":len(x), "p0":float(p0), "p1":float(p1),
        "alpha_design":float(alpha), "beta_design":float(beta),
        "log_likelihood_ratio":float(llr[stop]),"lower_log_bound":low,
        "upper_log_bound":high,"decision":decision,"stopping_index":int(stop),
        "table":pd.DataFrame({"trial":np.arange(1,stop+2),"outcome":x[:stop+1],
                              "cumulative_log_lr":llr[:stop+1]}),
        "interpretation_ar":(
            "اختبار SPRT ثنائي لفرضيتين محددتين مسبقًا، تحت استقلالية "
            "تجارب برنولي. إذا لم تتجاوز نسبة الاحتمال أي حد فالنتيجة "
            "غير حاسمة، وليست قبولًا لفرض العدم. ألفا وبيتا هدفان تصميميان "
            "تقريبيان، ولا يجوز اختيار الفرضيات بعد رؤية المسار."
        ),
        "interpretation_en":(
            "Wald Bernoulli SPRT for two prespecified simple probabilities "
            "and iid trials. No boundary crossing means inconclusive, "
            "not acceptance of H0. Nominal alpha/beta thresholds are "
            "approximate design targets; do not tune hypotheses post hoc."
        ),
    }


def cuped_randomized_effect(outcome, treatment, pre_experiment, *, alpha=.05):
    """CUPED adjusted treatment difference, independent 1:1-style randomized arms.

    Theta = pooled sample Cov(Y,X)/Var(X); adjusts Y - theta*(X - mean(X)).
    Returns Welch SE and an exploratory t p-value using adjusted outcomes.
    Estimating theta on the same data can make simple t inference approximate;
    split/cross-fit nuisance parameters for confirmatory precision and
    preregister the pre-treatment covariate. No causal interpretation absent
    valid randomized assignment.
    """
    try:
        y=np.asarray(outcome,dtype=float)
        x=np.asarray(pre_experiment,dtype=float)
        t=np.asarray(treatment)
    except (ValueError,TypeError) as exc:
        raise ValueError("Numeric outcome, pre-experiment covariate and binary group needed") from exc
    if (y.ndim!=1 or x.ndim!=1 or t.ndim!=1 or len(y)!=len(x) or len(t)!=len(y)
        or len(y)<20 or not np.isfinite(y).all() or not np.isfinite(x).all()
        or t.dtype.kind not in "biu" or not np.isin(t,[0,1]).all()
        or (t==1).sum()<8 or (t==0).sum()<8 or x.var(ddof=1)<=1e-14
        or isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha)
        or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError("Aligned finite vectors, two arms of >=8, varying pre-X, alpha (0,1)")
    theta=float(np.cov(y,x,ddof=1)[0,1]/np.var(x,ddof=1))
    adjusted=y-theta*(x-x.mean())
    a=adjusted[t==1];b=adjusted[t==0]
    diff=float(a.mean()-b.mean())
    raw=float(y[t==1].mean()-y[t==0].mean())
    va=float(a.var(ddof=1)/len(a));vb=float(b.var(ddof=1)/len(b))
    se=float(np.sqrt(va+vb))
    if se<=1e-12:
        raise ValueError("Adjusted variance too small for Welch inference")
    df=float((va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)))
    test=float(diff/se)
    p=float(2*stats.t.sf(abs(test),df))
    critical=float(stats.t.ppf(1-alpha/2,df))
    return {
        "test":"CUPED adjusted randomized difference","n_samples":len(y),
        "n_treated":int(len(a)),"n_control":int(len(b)),
        "theta":theta,"raw_difference":raw,"adjusted_difference":diff,
        "standard_error":se,"degrees_of_freedom":df,"statistic":test,
        "p_value":p,"ci_lower":diff-critical*se,"ci_upper":diff+critical*se,
        "alpha":float(alpha),
        "table":pd.DataFrame({"arm":["control","treated"],
                              "n":[len(b),len(a)],
                              "mean_adjusted":[float(b.mean()),float(a.mean())]}),
        "interpretation_ar":(
            "تعديل CUPED يستعمل متغيرًا مقاسًا قبل التجربة لتقليل تشتت "
            "المخرجات؛ الحساب الاستدلالي هنا تقريب Welch لأن معامل التعديل "
            "قُدّر على البيانات نفسها. لا إثبات للسببية دون تخصيص عشوائي "
            "صحيح، ويجب تثبيت متغير الضبط مسبقًا."
        ),
        "interpretation_en":(
            "CUPED adjusts outcome using a prespecified pre-treatment "
            "covariate. Welch inference is approximate because the adjustment "
            "coefficient is estimated on the same outcome data. Valid "
            "randomization is essential for causal interpretation."
        ),
    }


def shard_distribution_consistency(shards, *, alpha=.05):
    """Compare independent numeric partitions using a prespecified KS screen.

    Each shard compared with all other shards combined. Holm-adjusted
    p-values control familywise error for the collected tests only, under
    their validity assumptions. Overlapping shards, repeated monitoring and
    dependent records invalidate the simple interpretation.
    """
    if (not isinstance(shards,(list,tuple)) or not 2<=len(shards)<=50
        or isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha)
        or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError("Provide 2..50 shard vectors and alpha in (0,1)")
    arrays=[]
    for part in shards:
        try: a=np.asarray(part,dtype=float)
        except (ValueError,TypeError) as exc:
            raise ValueError("All shards must be finite numeric vectors") from exc
        if a.ndim!=1 or len(a)<20 or not np.isfinite(a).all():
            raise ValueError("Each shard must have at least 20 finite numbers")
        arrays.append(a)
    from scipy.stats import ks_2samp
    records=[]
    for i,arr in enumerate(arrays):
        rest=np.concatenate(arrays[:i]+arrays[i+1:])
        s=ks_2samp(arr,rest,alternative="two-sided")
        records.append({"shard":i,"n":len(arr),"mean":float(arr.mean()),
                        "ks_statistic":float(s.statistic),"raw_p":float(s.pvalue)})
    tab=pd.DataFrame(records)
    from statsmodels.stats.multitest import multipletests
    _,corrected,_,_=multipletests(tab["raw_p"].to_numpy(),alpha=alpha,method="holm")
    tab["holm_p"]=corrected
    tab["flagged"]=corrected<alpha
    return {
        "test":"Sharded numeric distribution consistency",
        "n_shards":len(shards),"n_samples":sum(map(len,arrays)),
        "n_flagged":int(tab["flagged"].sum()),"alpha":float(alpha),
        "table":tab,
        "interpretation_ar":(
            "فحص تجانُس توزيعات أقسام مستقلة باستعمال KS مع تصحيح Holm "
            "للمقارنات في هذا التقرير. يحتاج استقلال الحالات وتحديد الأقسام "
            "قبل الفحص، ولا يعالج المراقبة المتكررة أو الترتيب الزمني."
        ),
        "interpretation_en":(
            "Each disjoint numeric shard is compared to its complement by "
            "a two-sample KS test, Holm-adjusted within this family. "
            "Requires independent observations and prespecified shards; "
            "not appropriate for overlapping windows or repeated peeking."
        ),
    }


def practical_proportion_difference(success_a, total_a, success_b, total_b, *,
                                    minimum_relevant_difference=.01,alpha=.05):
    """Separate significance from a prespecified practical difference.

    Wald-normal two-sample p-value uses pooled null variance; the effect
    difference and unpooled normal CI are descriptive approximation, not
    a statistically justified equivalence or noninferiority procedure.
    """
    if (any(type(x) is not int for x in (success_a,total_a,success_b,total_b))
        or total_a<30 or total_b<30 or not 0<=success_a<=total_a
        or not 0<=success_b<=total_b or not np.isfinite(minimum_relevant_difference)
        or not 0<minimum_relevant_difference<1 or not np.isfinite(alpha)
        or not 0<alpha<1):
        raise ValueError("Integer successes within totals >=30, relevant gap and alpha in (0,1)")
    pa=success_a/total_a;pb=success_b/total_b
    pooled=(success_a+success_b)/(total_a+total_b)
    null_se=float(np.sqrt(pooled*(1-pooled)*(1/total_a+1/total_b)))
    if null_se<=0:
        raise ValueError("Pooled null variance vanishes")
    z=(pa-pb)/null_se
    p=float(2*stats.norm.sf(abs(z)))
    se=float(np.sqrt(pa*(1-pa)/total_a+pb*(1-pb)/total_b))
    quant=float(stats.norm.ppf(1-alpha/2))
    effect=float(pa-pb)
    return {
        "test":"Statistical versus practical proportion difference",
        "n_samples":total_a+total_b,"proportion_a":pa,"proportion_b":pb,
        "difference":effect,"minimum_relevant_difference":minimum_relevant_difference,
        "practically_relevant_point_estimate":bool(abs(effect)>=minimum_relevant_difference),
        "statistic":float(z),"p_value":p,"alpha":float(alpha),
        "ci_lower":effect-quant*se,"ci_upper":effect+quant*se,
        "interpretation_ar":(
            "تمييز إحصائي وصفي بين دلالة الفرق وقيمته العملية المحددة "
            "قبل جمع البيانات. تجاوز نقطة التقدير للحد العملي لا يُثبت "
            "التكافؤ أو تفوّقًا عمليًا بثقة؛ الفاصل الطبيعي تقريبي."
        ),
        "interpretation_en":(
            "Separates a two-proportion null z-test from a prespecified "
            "practical-effect threshold. Crossing the threshold by a "
            "point estimate is not a valid equivalence or noninferiority "
            "claim; the reported unpooled normal CI is approximate."
        ),
    }

"""Transparent representativeness screens and sample nonresponse probes.

Checks detect discrepancies only. They do not establish random sampling,
unit-level independence, causal impact or measurement invariance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sample_population_comparison(population, sample, *, numeric=(), categorical=(),
                                 alpha=.05):
    """KS for numerical columns; chi-square GOF for categorical proportions.

    A categorical level with reference probability zero and observed count
    positive is reported as a structural mismatch (p=0), not subjected to a
    chi-square approximation. Expected counts <5 are rejected as too sparse.
    Holm correction is applied across all tested columns, not repetitions.
    """
    if not isinstance(population,pd.DataFrame) or not isinstance(sample,pd.DataFrame):
        raise TypeError("population and sample must be DataFrames")
    if (not isinstance(numeric,(list,tuple)) or not isinstance(categorical,(list,tuple))
        or not numeric and not categorical or
        set(numeric)&set(categorical) or
        len(set(numeric))+len(set(categorical))>50
        or len(set(numeric))!=len(numeric) or len(set(categorical))!=len(categorical)
        or isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha)
        or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError("Declare unique disjoint numeric/categorical columns and alpha in (0,1)")
    if len(population)<30 or len(sample)<30:
        raise ValueError("Reference and sample must contain >=30 rows")
    rows=[]
    for field in numeric:
        if not isinstance(field,str) or field not in population or field not in sample:
            raise ValueError("Unknown numeric field")
        a=np.asarray(population[field],dtype=float)
        b=np.asarray(sample[field],dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Numeric fields must be completely finite")
        ks=stats.ks_2samp(a,b)
        rows.append({"column":field,"method":"KS","statistic":float(ks.statistic),
                     "p_value":float(ks.pvalue),"detail":"Independent two-sample KS"})
    for field in categorical:
        if not isinstance(field,str) or field not in population or field not in sample:
            raise ValueError("Unknown categorical field")
        ref=population[field];cur=sample[field]
        if ref.isna().any() or cur.isna().any():
            raise ValueError("Categorical fields may not contain missing levels")
        p=ref.value_counts(normalize=True)
        o=cur.value_counts()
        if not set(o.index).issubset(set(p.index)):
            rows.append({"column":field,"method":"chi-square GOF",
                         "statistic":float("inf"),"p_value":0.,
                         "detail":"Sample includes categories absent in reference"})
            continue
        observed=o.reindex(p.index,fill_value=0).to_numpy(dtype=float)
        expected=len(cur)*p.to_numpy(dtype=float)
        if np.any(expected<5):
            raise ValueError("All expected categorical counts must be >=5 for chi-square")
        ref_stat=stats.chisquare(observed,f_exp=expected)
        rows.append({"column":field,"method":"chi-square GOF",
                     "statistic":float(ref_stat.statistic),
                     "p_value":float(ref_stat.pvalue),
                     "detail":"Reference frequencies treated as fixed target"})
    table=pd.DataFrame(rows)
    raw=table["p_value"].to_numpy()
    order=np.argsort(raw)
    adj=np.empty(len(raw))
    ordered=np.maximum.accumulate((len(raw)-np.arange(len(raw)))*raw[order])
    adj[order]=np.clip(ordered,0,1)
    table["p_value_holm"]=adj
    table["flagged"]=adj<alpha
    return {
        "test":"Numeric and categorical sample representativeness screen",
        "n_population":len(population),"n_sample":len(sample),
        "n_columns_tested":len(table),"n_flagged":int(table["flagged"].sum()),
        "alpha":float(alpha),"table":table,
        "interpretation_ar":(
            "مقارنة استكشافية بين العينة والمرجع: KS للأعمدة العددية "
            "وكا² الملاءمة للفئات، مع Holm عبر الأعمدة. يفترض الاختبار "
            "استقلال العينات وثبات نسب المجتمع المرجعية. عدم رفض الفروق "
            "لا يثبت تمثيل العينة ولا سلامة التصميم."
        ),
        "interpretation_en":(
            "Exploratory comparison: numerical two-sample KS and categorical "
            "chi-square goodness-of-fit against fixed reference proportions, "
            "Holm-adjusted over columns. Independence and design assumptions "
            "matter; nonsignificance never proves representative sampling."
        ),
    }


def early_late_nonresponse_screen(early, late, *, alpha=.05):
    """Exploratory Welch test for two disjoint early/late respondent groups.

    Late respondents are NOT nonrespondents, so this does not identify or
    rule out actual nonresponse bias. Timing effects may be confounded.
    """
    try:
        a=np.asarray(early,dtype=float);b=np.asarray(late,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("Numeric early/late values required") from exc
    if (a.ndim!=1 or b.ndim!=1 or min(len(a),len(b))<5
        or not np.isfinite(a).all() or not np.isfinite(b).all()
        or a.var(ddof=1)<=1e-14 or b.var(ddof=1)<=1e-14
        or isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha)
        or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError("Disjoint early/late samples >=5 with positive variances")
    va=a.var(ddof=1)/len(a);vb=b.var(ddof=1)/len(b)
    t=(a.mean()-b.mean())/np.sqrt(va+vb)
    df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
    p=float(2*stats.t.sf(abs(t),df))
    return {
        "test":"Early-late respondent Welch screen",
        "n_early":len(a),"n_late":len(b),
        "difference":float(a.mean()-b.mean()),"statistic":float(t),
        "degrees_of_freedom":float(df),"p_value":p,"alpha":float(alpha),
        "table":pd.DataFrame({"group":["early","late"],"n":[len(a),len(b)],
                              "mean":[a.mean(),b.mean()]}),
        "interpretation_ar":(
            "مقارنة استكشافية بين مستجيبين مبكرين ومتأخرين؛ لا تمثل "
            "المجموعة المتأخرة غير المستجيبين، ولذلك لا تثبت غياب "
            "تحيز عدم الاستجابة أو وجوده. يجب مراجعة أثر توقيت الإجابة."
        ),
        "interpretation_en":(
            "Exploratory Welch comparison between disjoint early and "
            "late respondents. Late response is not nonresponse, so this "
            "does not quantify, prove or rule out nonresponse bias."
        ),
    }


def scientific_report_alerts(result, *, min_n=30, alpha=.05):
    """Non-diagnostic quality warnings derived from explicit result metadata.

    A zero-warning result is NOT scientific approval or proof that a model
    is valid. Flags missing assumption information, small n and missing
    bilingual text; does not infer unobserved assumptions from p-values.
    """
    from collections.abc import Mapping
    if not isinstance(result,Mapping) or type(min_n) is not int or min_n<2:
        raise ValueError("Provide mapping result and min_n>=2")
    if isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha in (0,1) required")
    alerts=[]
    count=result.get("n_samples")
    if count is None:
        alerts.append(("unknown_sample_size","Missing n_samples; verify sample size explicitly"))
    elif isinstance(count,(int,np.integer)) and not isinstance(count,(bool,np.bool_)) and count<min_n:
        alerts.append(("small_sample","Sample smaller than configured n threshold"))
    elif not isinstance(count,(int,np.integer)) or count<=0:
        alerts.append(("invalid_sample_size","Invalid n_samples metadata"))
    for lang in ("ar","en"):
        val=result.get("interpretation_"+lang)
        if not isinstance(val,str) or not val.strip():
            alerts.append(("missing_interpretation_"+lang,"Missing bilingual interpretation"))
    if "p_value" in result:
        p=result["p_value"]
        if isinstance(p,(bool,np.bool_)) or not isinstance(p,(int,float,np.number)) or not np.isfinite(p) or not 0<=p<=1:
            alerts.append(("invalid_p_value","p_value is not a finite probability"))
    if result.get("slope_homogeneity_rejected") is True:
        alerts.append(("slope_heterogeneity","ANCOVA common-slope assumption flagged"))
    if result.get("sphericity_correction","").startswith("none;"):
        alerts.append(("uncorrected_sphericity","Repeated ANOVA has no sphericity correction"))
    if result.get("n_alarms",0) and "Page-Hinkley" in str(result.get("test","")):
        alerts.append(("uncalibrated_alarm","Sequential alarm requires calibration and design review"))
    return {
        "test":"Research output quality alerts",
        "n_alerts":len(alerts),
        "passed_metadata_screen":len(alerts)==0,
        "table":pd.DataFrame(alerts,columns=["flag","detail"]),
        "interpretation_ar":(
            "تنبيهات آلية لفحص بيانات وصف النتيجة فقط؛ لا تتحقق آليًا "
            "من الافتراضات غير المسجلة أو التحيز أو السببية. عدم "
            "ظهور تنبيه لا يساوي اعتمادًا علميًا للنتيجة."
        ),
        "interpretation_en":(
            "Metadata-only report alerts for documented conditions. "
            "A clean screen cannot verify unreported assumptions, "
            "selection bias, causal identification or research validity."
        ),
    }

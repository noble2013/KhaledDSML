"""Constrained small integer-score reconstruction inspired by SPRITE.

Returns A possible synthetic score distribution satisfying rounded published
summary statistics, never the original dataset. Constraint reconstruction
is NOT evidence of misconduct or a faithful full SPRITE stochastic procedure.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import pandas as pd


def _dec(v):
    if isinstance(v,bool):raise ValueError("Reports must be numeric")
    try:d=Decimal(str(v))
    except (ValueError,TypeError,InvalidOperation) as exc:
        raise ValueError("Reports must be numeric") from exc
    if not d.is_finite():raise ValueError("Reports must be finite")
    return d


def sprite_integer_constraint_witness(reported_mean, reported_sd, n_samples, *,
                                      minimum_score=1, maximum_score=5,
                                      decimals_mean=2, decimals_sd=2,
                                      fixed_counts=None):
    """Find one bounded integer histogram matching rounded mean and sample SD.

    Enumerates finite histograms (n<=18, <=6 score levels) in deterministic
    lexicographic order. Optional fixed_counts are exact number of cases in
    listed categories. Reports a possible synthetic reconstruction only, not
    a probability, a p-value or an inference about the actual study.
    """
    if (type(n_samples) is not int or not 2<=n_samples<=18
        or type(minimum_score) is not int or type(maximum_score) is not int
        or maximum_score<minimum_score or maximum_score-minimum_score>5
        or type(decimals_mean) is not int or not 0<=decimals_mean<=4
        or type(decimals_sd) is not int or not 0<=decimals_sd<=4):
        raise ValueError("Require n=2..18, integer scale <=6 categories and precisions 0..4")
    mu=_dec(reported_mean);sd=_dec(reported_sd)
    dm=Decimal(1).scaleb(-decimals_mean)
    ds=Decimal(1).scaleb(-decimals_sd)
    if mu!=mu.quantize(dm) or sd!=sd.quantize(ds) or sd<0:
        raise ValueError("Reported moments must have the stated decimal precision and SD>=0")
    fixed_counts={} if fixed_counts is None else fixed_counts
    if (not isinstance(fixed_counts,dict) or
        any(type(k) is not int or not minimum_score<=k<=maximum_score
            or type(v) is not int or v<0 for k,v in fixed_counts.items())
        or sum(fixed_counts.values())>n_samples):
        raise ValueError("fixed_counts must map valid integer score levels to nonnegative counts")
    levels=list(range(minimum_score,maximum_score+1))
    remaining=n_samples-sum(fixed_counts.values())
    witness=None
    examined=0
    with localcontext() as ctx:
        ctx.prec=65
        def recurse(index,left,totals):
            nonlocal witness,examined
            if witness is not None:return
            if index==len(levels):
                if left!=0:return
                examined+=1
                sums=sum(score*count for score,count in zip(levels,totals))
                average=Decimal(sums)/Decimal(n_samples)
                if average.quantize(dm,rounding=ROUND_HALF_UP)!=mu:return
                sq=sum(score*score*count for score,count in zip(levels,totals))
                var=(Decimal(sq)-Decimal(sums*sums)/Decimal(n_samples))/Decimal(n_samples-1)
                if var<0:return
                s=var.sqrt()
                if s.quantize(ds,rounding=ROUND_HALF_UP)==sd:
                    witness=tuple(totals)
                return
            score=levels[index]
            required=fixed_counts.get(score,0)
            for extra in range(left+1):
                recurse(index+1,left-extra,totals+[required+extra])
                if witness is not None:return
        recurse(0,remaining,[])
    if witness is None:
        table=pd.DataFrame(columns=["score","frequency"])
        synthetic=[]
    else:
        table=pd.DataFrame({"score":levels,"frequency":list(witness)})
        synthetic=[score for score,cnt in zip(levels,witness) for _ in range(cnt)]
    return {
        "test":"Bounded SPRITE-style integer score constraint witness",
        "n_samples":n_samples,"reported_mean":str(mu),"reported_sd":str(sd),
        "one_feasible_reconstruction_found":witness is not None,
        "n_histograms_examined":examined,
        "synthetic_scores":synthetic,
        "table":table,
        "interpretation_ar":(
            "بحث عددي مقيّد يستلهم فكرة SPRITE لإيجاد توزيع اصطناعي "
            "ممكن على سلم درجات صحيح يطابق متوسطًا وانحرافًا معياريًا "
            "مدوّنين وفق التقريب المحدد. لا يسترجع البيانات الأصلية "
            "ولا يطبق الخوارزمية العشوائية الكاملة، ولا يثبت "
            "وجود تزوير أو غيابه."
        ),
        "interpretation_en":(
            "Finite deterministic bounded-integer histogram search inspired "
            "by SPRITE: identifies ONE synthetic witness satisfying rounded "
            "mean, sample SD and optional fixed score frequencies. "
            "Not the full stochastic SPRITE algorithm, not original "
            "data recovery and not evidence of research misconduct."
        ),
    }

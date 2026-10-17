"""Unicode-aware language corpus diagnostics and paired score uncertainty.

Designed for Arabic and English text without downloading tokenizers/models.
Whitespace/punctuation tokenization is a transparent baseline, not full Arabic
morphological analysis. Corpus comparisons need adequate independent texts.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import re
import numpy as np
import pandas as pd
from scipy import stats

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_AR_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")


def corpus_tokens(text, *, lowercase=True, strip_arabic_diacritics=True):
    """Return simple Arabic/Latin Unicode word tokens (not a stemmer)."""
    if not isinstance(text, str):
        raise TypeError("Input text must be a Unicode string")
    result = _AR_DIACRITICS.sub("",text) if strip_arabic_diacritics else text
    if lowercase:
        result = result.casefold()
    return _WORD_RE.findall(result)


def lexical_diversity(text_or_tokens):
    """Type/token ratio and Shannon entropy (not length-invariant)."""
    tokens = corpus_tokens(text_or_tokens) if isinstance(text_or_tokens,str) else list(text_or_tokens)
    if len(tokens)<1 or any(not isinstance(t,str) or not t for t in tokens):
        raise ValueError("Provide one or more nonempty word tokens")
    counts = Counter(tokens)
    probs = np.array(list(counts.values()),dtype=float)/len(tokens)
    return {
        "tokens":len(tokens), "types":len(counts),
        "ttr":float(len(counts)/len(tokens)),
        "shannon_entropy_bits":float(-np.dot(probs,np.log2(probs))),
        "interpretation_ar":"مؤشر التنوع المعجمي TTR يتأثر بطول النص، فلا تقارن مدونات متفاوتة الطول دون تصحيح.",
        "interpretation_en":"Type-token ratio depends on corpus length; unequal-length comparisons require controls.",
    }


def zipf_frequency_analysis(text_or_tokens, *, min_types=5):
    """Log-rank versus log-frequency regression, descriptive only."""
    tokens = corpus_tokens(text_or_tokens) if isinstance(text_or_tokens,str) else list(text_or_tokens)
    if not tokens or any(not isinstance(t,str) or not t for t in tokens):
        raise ValueError("Corpus must contain valid word tokens")
    freqs = Counter(tokens).most_common()
    if len(freqs)<min_types or min_types<3:
        raise ValueError("At least min_types>=3 distinct word types required")
    f = np.array([n for _,n in freqs],dtype=float)
    if np.ptp(f) == 0:
        raise ValueError("Zipf log-rank fit needs at least two distinct token frequencies")
    ranks = np.arange(1,len(f)+1,dtype=float)
    slope, intercept, corr, p, _ = stats.linregress(np.log(ranks),np.log(f))
    table = pd.DataFrame({"rank":ranks.astype(int),
                          "word":[word for word,_ in freqs],
                          "frequency":f.astype(int)})
    return {
        "slope":float(slope),"intercept":float(intercept),"r_squared":float(corr**2),
        "n_samples":len(tokens),"unique_types":len(freqs),"table":table,
        "interpretation_ar":"انحدار وصفي لتردد الكلمات حسب الرتبة، وليس اختبارًا قاطعًا لانطباق قانون Zipf.",
        "interpretation_en":"Descriptive log-rank/frequency fit; it is not a definitive statistical test of Zipf's law.",
    }


def dunning_corpus_keyness(corpus_a, corpus_b, *, min_total=3, max_types=5000):
    """Dunning G² word keyness between two independent token corpora.

    Holm familywise adjustment is applied across actually compared words.
    Direction is based on the relative frequency in A vs B. This is a
    corpus-level association and does not identify ideological intent.
    """
    a = corpus_tokens(corpus_a) if isinstance(corpus_a,str) else list(corpus_a)
    b = corpus_tokens(corpus_b) if isinstance(corpus_b,str) else list(corpus_b)
    if min(len(a),len(b))<10 or any(not isinstance(t,str) or not t for t in a+b):
        raise ValueError("Need >=10 valid word tokens in each corpus")
    if type(min_total) is not int or min_total<1 or type(max_types) is not int or max_types<2:
        raise ValueError("min_total>=1 and max_types>=2 must be integers")
    ca,cb=Counter(a),Counter(b)
    vocab=[word for word,n in (ca+cb).most_common() if n>=min_total]
    if len(vocab)>max_types:
        raise ValueError("Too many vocabulary types; raise max_types explicitly for a full comparison")
    if not vocab:
        raise ValueError("No words meet the minimum frequency")
    n1,n2=len(a),len(b)
    rows=[]
    for word in vocab:
        aa,bb=ca[word],cb[word]
        table=np.array([[aa,n1-aa],[bb,n2-bb]],dtype=float)
        if np.any(table.sum(axis=0)<=0):
            continue
        res=stats.chi2_contingency(table,correction=False,lambda_="log-likelihood")
        rows.append({
            "word":word,"count_a":aa,"count_b":bb,
            "rate_a":aa/n1,"rate_b":bb/n2,
            "g_squared":float(res.statistic),
            "p_value":float(res.pvalue),
            "overrepresented_in":"A" if aa/n1>bb/n2 else "B" if aa/n1<bb/n2 else "equal",
        })
    if not rows:
        raise ValueError("All tokens have identical global occurrence; no contrast is defined")
    result=pd.DataFrame(rows)
    order=np.argsort(result["p_value"].to_numpy())
    ps=result["p_value"].to_numpy()[order]
    adjusted=np.maximum.accumulate(np.minimum(1.,ps*(len(ps)-np.arange(len(ps)))))
    out=np.empty_like(adjusted)
    out[order]=adjusted
    result["holm_p_value"]=out
    return result.sort_values(["holm_p_value","g_squared"],ascending=[True,False]).reset_index(drop=True)


def paired_bootstrap_score_difference(
    reference, predictions_a, predictions_b, metric: Callable,
    *, n_resamples=1000, confidence_level=.95, seed=42,
):
    """Paired percentile-bootstrap uncertainty on any corpus-level score.

    All arrays are resampled at matching document indices, and metric(y, p)
    must return a scalar score. No permutation p-value or causality claim is
    fabricated; percentile CIs should not be used as exact hypothesis tests.
    """
    if not callable(metric):
        raise TypeError("metric must be callable as metric(reference, prediction)")
    r,a,b=list(reference),list(predictions_a),list(predictions_b)
    n=len(r)
    if n<10 or len(a)!=n or len(b)!=n or type(n_resamples) is not int or n_resamples<100 or not 0<confidence_level<1:
        raise ValueError("Need >=10 paired documents, >=100 bootstrap replications, confidence level in (0,1)")
    def _value(ref,pred):
        val=float(metric(ref,pred))
        if not np.isfinite(val):
            raise ValueError("metric must yield a finite numeric scalar")
        return val
    point=_value(r,a)-_value(r,b)
    rng=np.random.default_rng(seed)
    diffs=[]
    for _ in range(n_resamples):
        ids=rng.integers(0,n,size=n)
        ref=[r[i] for i in ids]
        aa=[a[i] for i in ids]
        bb=[b[i] for i in ids]
        diffs.append(_value(ref,aa)-_value(ref,bb))
    tail=(1-confidence_level)/2
    low,high=np.quantile(diffs,[tail,1-tail])
    return {
        "score_difference":float(point),
        "lower":float(low),"upper":float(high),
        "confidence_level":confidence_level,
        "n_samples":n,"replications":n_resamples,
        "interpretation_ar":"فاصل Bootstrap المزدوج لأداء نموذجين؛ لا يمثل قيمة احتمالية لاختبار فرضية صفرية.",
        "interpretation_en":"Paired percentile-bootstrap score difference; not an exact null-hypothesis p-value.",
    }

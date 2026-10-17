"""Arabic and English corpus metrics with calibrated interpretation limits."""
import numpy as np
import pytest
from khaled_dsml import (
    corpus_tokens, lexical_diversity, zipf_frequency_analysis,
    dunning_corpus_keyness, paired_bootstrap_score_difference,
)


def test_arabic_tokenization_and_diacritics():
    words=corpus_tokens("كِتاب، كِتاب! مدارس تستقبل الطلاب. Hello 2026")
    assert words[:2]==["كتاب","كتاب"]
    assert "hello" in words
    assert "2026" in words


def test_lexical_ttr_varies_by_length_and_entropy():
    r=lexical_diversity("العلم العلم التعلم القراءة القراءة")
    assert r["tokens"]==5
    assert r["types"]==3
    assert r["ttr"]==pytest.approx(.6)
    assert r["shannon_entropy_bits"]>0


def test_zipf_fit_is_descriptive_and_table_is_real():
    tokens=(["أ"]*15+["ب"]*10+["ج"]*7+["د"]*4+
            ["هـ"]*3+["و"]*2+["ز"]*1)
    result=zipf_frequency_analysis(tokens)
    assert result["slope"]<0
    assert 0<=result["r_squared"]<=1
    assert result["table"]["frequency"].iloc[0]==15
    assert "not" in result["interpretation_en"]


def test_dunning_keyness_holm_and_direction():
    a=(["اقتصاد"]*45+["تجارة"]*10+["تعليم"]*25+
       ["عمل"]*10+["سوق"]*10)
    b=(["اقتصاد"]*5+["تجارة"]*10+["تعليم"]*25+
       ["عمل"]*10+["سوق"]*50)
    table=dunning_corpus_keyness(a,b)
    assert table.loc[table["word"]=="اقتصاد","overrepresented_in"].iloc[0]=="A"
    assert table.loc[table["word"]=="سوق","overrepresented_in"].iloc[0]=="B"
    assert (table["holm_p_value"]>=table["p_value"]).all()
    assert table["holm_p_value"].between(0,1).all()


def test_paired_bootstrap_difference_reproducible_without_fake_p_value():
    refs=list(range(20))
    a=list(range(20))
    b=[x+1 for x in refs]
    metric=lambda y,p: float(-np.mean((np.array(y)-p)**2))
    x=paired_bootstrap_score_difference(refs,a,b,metric,
                                        n_resamples=100,seed=2)
    y=paired_bootstrap_score_difference(refs,a,b,metric,
                                        n_resamples=100,seed=2)
    assert x["score_difference"]==pytest.approx(1)
    assert x==y
    assert "p_value" not in x
    with pytest.raises(ValueError):
        paired_bootstrap_score_difference(refs,a,b,metric,n_resamples=10)


def test_zipf_rejects_uniform_counts_instead_of_nan_r2():
    with pytest.raises(ValueError, match="distinct token frequencies"):
        zipf_frequency_analysis(["one","two","three","four","five"])

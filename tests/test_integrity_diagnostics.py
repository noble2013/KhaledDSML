"""Independent numerical references and research integrity boundary tests."""
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error

from khaled_dsml import (
    mcnemar_paired_predictions, mean_absolute_percentage_error_strict,
    grim_mean_consistency, recalculate_t_test_p, validate_dataframe_contract,
    page_hinkley_change_monitor, render_analysis_dashboard,
)


def _examples():
    # The same 100 subjects are scored by both models. A-only=20, B-only=5.
    target = np.array([1, 0] * 50)
    a_correct = np.array([True]*55 + [True]*20 + [False]*5 + [False]*20)
    b_correct = np.array([True]*55 + [False]*20 + [True]*5 + [False]*20)
    a = np.where(a_correct, target, 1-target)
    b = np.where(b_correct, target, 1-target)
    return target, a, b


def test_mcnemar_exact_against_scipy_and_paired_counts():
    y, a, b = _examples()
    r = mcnemar_paired_predictions(y, a, b)
    assert (r["both_correct"],r["only_a_correct"],r["only_b_correct"],
            r["both_incorrect"]) == (55,20,5,20)
    assert r["n_discordant"] == 25
    assert r["statistic"] == 5
    assert r["p_value"] == pytest.approx(stats.binomtest(20, 25, .5).pvalue)
    assert r["accuracy_a"] == pytest.approx(.75)
    assert r["accuracy_b"] == pytest.approx(.60)
    assert r["table"].iloc[0,1] == 55
    assert r["table"].iloc[1,1] == 20


def test_mcnemar_chisquare_correction_and_zero_discordance():
    y, a, b = _examples()
    for corrected in (True,False):
        r = mcnemar_paired_predictions(y,a,b,exact=False,
                                      continuity_correction=corrected)
        reference = (15-(1 if corrected else 0))**2/25
        assert r["statistic"] == pytest.approx(reference)
        assert r["p_value"] == pytest.approx(stats.chi2.sf(reference,1))
    tie = mcnemar_paired_predictions(y,a,a)
    assert tie["n_discordant"] == 0
    assert tie["p_value"] == 1
    assert tie["statistic"] == 0


@pytest.mark.parametrize("y,a,b", [
    ([1,0], [1], [1,0]), ([1,0], [1,0], [0,np.nan]),
    ([1,0], [1,0], [0]), ([1], [1], [1]),
])
def test_mcnemar_rejects_misaligned_or_missing(y,a,b):
    with pytest.raises(ValueError):
        mcnemar_paired_predictions(y,a,b)


def test_mape_matches_sklearn_only_when_actuals_nonzero():
    actual = [10,20,30,40]
    pred = [11,18,33,36]
    r = mean_absolute_percentage_error_strict(actual,pred)
    assert r["mape_percent"] == pytest.approx(
        100*mean_absolute_percentage_error(actual,pred))
    assert r["median_ape_percent"] == pytest.approx(10)
    assert "p_value" not in r


@pytest.mark.parametrize("actual,pred", [
    ([0,2],[1,2]),([1,np.nan],[1,2]),([1,2],[1]),([1],[2]),
    ([1,2],[1,np.inf]),([1,2],["1","?"]),
])
def test_mape_rejects_unsafe_actuals_and_invalid_inputs(actual,pred):
    with pytest.raises(ValueError):
        mean_absolute_percentage_error_strict(actual,pred)


def test_grim_matches_exhaustive_feasible_integer_sum_reference():
    # Exhaustively test every nearby rounded report, including half-up boundary.
    for n in (2,3,7,9,20,32):
        totals = set(range(n, 5*n+1))
        attainable = {
            str((Decimal(k)/Decimal(n)).quantize(Decimal(".01"),rounding=ROUND_HALF_UP))
            for k in totals
        }
        for raw in range(100,501,3):
            candidate = Decimal(raw)/Decimal(100)
            displayed = str(candidate.quantize(Decimal(".01")))
            r = grim_mean_consistency(displayed,n,decimals=2,
                                      minimum_score=1,maximum_score=5)
            assert r["arithmetically_consistent"] == (displayed in attainable)
    assert grim_mean_consistency("3.00",20)["arithmetically_consistent"]
    assert not grim_mean_consistency("3.21",7)["arithmetically_consistent"]


@pytest.mark.parametrize("kwargs",[
    {"reported_mean":"3.214","sample_size":10},
    {"reported_mean":"nan","sample_size":10},
    {"reported_mean":"3.00","sample_size":True},
    {"reported_mean":"3.00","sample_size":-1},
    {"reported_mean":"3.00","sample_size":10,"decimals":-1},
    {"reported_mean":"3.00","sample_size":10,"minimum_score":6},
])
def test_grim_rejects_invalid_scope(kwargs):
    with pytest.raises(ValueError):
        grim_mean_consistency(**kwargs)


@pytest.mark.parametrize("t,df,tail",[
    (2.31,12,"two-sided"), (-1.7,19,"greater"),(-1.7,19,"less"),
    (0.,4,"two-sided"), (4.2,250,"two-sided")
])
def test_t_report_recalculation_matches_scipy(t,df,tail):
    r = recalculate_t_test_p(t,df,alternative=tail)
    expected = (2*stats.t.sf(abs(t),df) if tail=="two-sided"
                else stats.t.sf(t,df) if tail=="greater" else stats.t.cdf(t,df))
    assert r["p_value"] == pytest.approx(expected)
    assert "reported_p" not in r
    checked = recalculate_t_test_p(t,df,alternative=tail,reported_p=r["p_rounded"])
    assert checked["rounded_report_consistent"] is True


def test_t_report_detects_mismatching_rounded_p_and_unsupported_inequalities():
    r = recalculate_t_test_p(2.3,20,reported_p="0.900")
    assert r["rounded_report_consistent"] is False
    with pytest.raises(ValueError):
        recalculate_t_test_p(2.3,20,reported_p="<.001")
    with pytest.raises(ValueError):
        recalculate_t_test_p(2.3,0)
    with pytest.raises(ValueError):
        recalculate_t_test_p(2.3,20,alternative="wrong")


def test_contract_counts_without_raw_row_leak_or_mutation():
    df = pd.DataFrame({"age":[20,25,np.nan,25],
                       "arm":["A","B","X","A"],"id":[1,2,2,4],
                       "secret":["private1","private2","private3","private4"]})
    original = df.copy(deep=True)
    schema = {
        "age":{"kind":"numeric","nullable":False,"min":21,"max":30},
        "arm":{"kind":"string","allowed_values":["A","B"]},
        "id":{"kind":"integer","unique":True},
        "absent":{"kind":"numeric","nullable":False},
    }
    r = validate_dataframe_contract(df,schema,allow_extra_columns=False)
    assert r["passed"] is False
    t = r["table"]
    assert t.loc[(t.column=="age") & (t.rule=="not_null"),"violations"].iloc[0] == 1
    assert t.loc[(t.column=="age") & (t.rule=="min"),"violations"].iloc[0] == 1
    assert t.loc[(t.column=="id") & (t.rule=="unique"),"violations"].iloc[0] == 1
    assert t.loc[(t.column=="arm") & (t.rule=="allowed_values"),"violations"].iloc[0] == 1
    assert t.loc[t.rule=="no_extra_columns","violations"].iloc[0] == 1
    assert not any("private" in text for text in t.astype(str).to_numpy().ravel())
    pd.testing.assert_frame_equal(df,original)


def test_contract_pass_case_and_invalid_rules():
    df=pd.DataFrame({"x":[1,2,3],"name":["a","b","c"]})
    r=validate_dataframe_contract(df,{
        "x":{"kind":"integer","min":0,"max":3,"nullable":False,"unique":True},
        "name":{"kind":"string","nullable":False}
    },allow_extra_columns=False)
    assert r["passed"] is True
    assert r["n_failed_rules"] == 0
    for bad in ({"x":{"kind":"complex"}}, {"x":{"min":0}},
                {"x":{"kind":"integer","min":4,"max":1}},
                {"x":{"kind":"integer","nullable":"no"}}):
        with pytest.raises(ValueError):
            validate_dataframe_contract(df,bad)


def test_page_hinkley_constant_baseline_then_shift():
    seq=np.r_[np.zeros(100),np.ones(120)*4]
    result=page_hinkley_change_monitor(seq,delta=.05,threshold=10,min_instances=15,
                                       direction="up")
    assert result["n_alarms"] >= 1
    assert result["alarm_indices"][0] >= 100
    assert "p_value" not in result
    assert result["n_alarms"] == len(result["alarm_directions"])
    assert all(v=="up" for v in result["alarm_directions"])


def test_page_hinkley_no_false_alarm_for_constant_stream_and_direction():
    for d in ("up","down","both"):
        r=page_hinkley_change_monitor(np.zeros(150),threshold=2,min_instances=10,direction=d)
        assert r["n_alarms"] == 0
    a=page_hinkley_change_monitor(np.r_[np.ones(80)*4,np.zeros(120)],
                                  delta=.05,threshold=10,direction="down")
    assert a["alarm_indices"][0] >= 80


@pytest.mark.parametrize("kwargs",[
    {"threshold":0},{"delta":-1},{"min_instances":1},
    {"direction":"sideways"},{"reset_after_alarm":7}
])
def test_page_hinkley_rejects_invalid_configurations(kwargs):
    with pytest.raises(ValueError):
        page_hinkley_change_monitor(np.arange(40),**kwargs)


def test_all_six_methods_render_in_arabic_and_english_without_external_resources():
    y,a,b=_examples()
    outputs=[
        mcnemar_paired_predictions(y,a,b),
        mean_absolute_percentage_error_strict([10,20],[11,18]),
        grim_mean_consistency("3.21",7),
        recalculate_t_test_p(2.3,20),
        validate_dataframe_contract(pd.DataFrame({"x":[1,2]}),{"x":{"kind":"integer"}}),
        page_hinkley_change_monitor(np.zeros(40),min_instances=10),
    ]
    for result in outputs:
        html=render_analysis_dashboard(result,theme="ocean")
        assert 'dir="rtl"' in html
        assert result["interpretation_ar"] in html
        assert result["interpretation_en"] in html
        assert "http://" not in html and "https://" not in html

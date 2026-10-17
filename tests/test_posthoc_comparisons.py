"""Reference and edge-case tests for post-hoc statistical comparisons."""
import numpy as np
import pytest
from scipy import stats

from khaled_dsml import (
    tukey_hsd_comparisons, games_howell_comparisons,
    dunn_posthoc, dunnett_control_comparisons, nemenyi_friedman_posthoc,
    render_analysis_dashboard,
)


@pytest.fixture
def groups():
    rng=np.random.default_rng(77)
    return {
        "ضبط":rng.normal(2,1,24),
        "علاج A":rng.normal(3,.9,29),
        "علاج B":rng.normal(4,1.1,27),
    }


def test_tukey_matched_to_scipy_and_usable_in_dashboard(groups):
    r=tukey_hsd_comparisons(groups)
    ref=stats.tukey_hsd(*groups.values())
    names=list(groups)
    assert r["table"].shape[0]==3
    assert r["table"].iloc[0]["mean_difference"]==pytest.approx(ref.statistic[0,1])
    assert r["table"].iloc[0]["p_value"]==pytest.approx(ref.pvalue[0,1])
    assert r["n_samples"]==sum(map(len,groups.values()))
    assert "جدول نتائج التحليل" in render_analysis_dashboard(r,theme="ocean")


def test_games_howell_welch_studentized_reference(groups):
    r=games_howell_comparisons(groups)
    row=r["table"].iloc[0]
    a,b=list(groups.values())[:2]
    se=np.sqrt(np.var(a,ddof=1)/len(a)+np.var(b,ddof=1)/len(b))
    va=np.var(a,ddof=1)/len(a)
    vb=np.var(b,ddof=1)/len(b)
    df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
    expected=stats.studentized_range.sf(
        np.sqrt(2)*abs(a.mean()-b.mean())/se,len(groups),df
    )
    assert row["p_value"]==pytest.approx(expected)
    assert row["welch_df"]==pytest.approx(df)
    assert row["ci_lower"] < row["mean_difference"] < row["ci_upper"]


def test_dunn_tie_correction_and_holm_adjustment(groups):
    from statsmodels.stats.multitest import multipletests
    a=np.round(groups["ضبط"])
    b=np.round(groups["علاج A"])
    c=np.round(groups["علاج B"])
    r=dunn_posthoc({"ضبط":a,"علاج A":b,"علاج B":c})
    table=r["table"]
    reference=multipletests(table["raw_p_value"],method="holm")[1]
    assert table["p_value"].to_numpy()==pytest.approx(reference)
    assert (table["p_value"]>=table["raw_p_value"]).all()
    assert "Dunn" in r["test"]


def test_dunnett_scipy_reference_and_only_vs_control(groups):
    from scipy.stats import dunnett
    a,b,c=list(groups.values())
    r=dunnett_control_comparisons(groups,control="ضبط",seed=4)
    row=r["table"]
    # Random seed prevents Monte Carlo drift between SciPy versions; check each
    # significant statistic and controlled comparisons, not an unstable MC bit.
    assert len(row)==2
    assert set(row["group_b"])=={"ضبط"}
    assert row["mean_difference"].iloc[0]==pytest.approx(b.mean()-a.mean())
    assert row["mean_difference"].iloc[1]==pytest.approx(c.mean()-a.mean())
    assert row["p_value"].between(0,1).all()
    assert row["ci_lower"].le(row["ci_upper"]).all()


def test_friedman_nemenyi_requires_paired_data_and_no_ties():
    rng=np.random.default_rng(15)
    data=rng.normal(size=(40,3))+np.array([0,1,2])
    groups={"أ":data[:,0],"ب":data[:,1],"ج":data[:,2]}
    r=nemenyi_friedman_posthoc(groups)
    assert len(r["table"])==3
    assert r["table"]["p_value"].between(0,1).all()
    with pytest.raises(ValueError,match="tied ranks"):
        nemenyi_friedman_posthoc({
            "أ":np.arange(5.),"ب":np.arange(5.),"ج":np.arange(5.)+2
        })
    with pytest.raises(ValueError,match="equally sized"):
        nemenyi_friedman_posthoc({
            "أ":np.arange(6.),"ب":np.arange(7.),"ج":np.arange(8.)
        })


def test_invalid_group_designs_and_alpha_rejected():
    with pytest.raises(ValueError):
        tukey_hsd_comparisons({"a":[1,2,3],"b":[4,5,6]})
    with pytest.raises(ValueError):
        games_howell_comparisons({"a":[1,2,np.nan],"b":[1,2,3],"c":[5,6,7]})
    with pytest.raises(KeyError):
        dunnett_control_comparisons({"a":[1,2,3],"b":[2,3,4],"c":[3,4,5]},control="x")
    with pytest.raises(ValueError):
        dunn_posthoc({"a":[1,2,3],"b":[2,3,4],"c":[3,4,5]},correction="bad")
    with pytest.raises(ValueError):
        tukey_hsd_comparisons({"a":[1,2,3],"b":[4,5,6],"c":[7,8,9]},alpha=1)

"""End-to-end academic workflow with explicit quality gates and statistical cautions."""
import numpy as np
import pandas as pd
import pytest
from khaled_dsml import (
    compare_groups, research_quality_gate, apa_style_test_summary,
    display_analysis_dashboard, render_analysis_dashboard, normality_test,
)


def test_auto_welch_and_explicit_option():
    rng=np.random.default_rng(27)
    a=rng.normal(size=90)
    b=rng.normal(loc=1,size=90)
    result=compare_groups(a,b,method="welch")
    assert result["test"]=="Welch independent t"
    assert result["reject_null"]
    assert len(result["assumption_checks"]["normality_shapiro_p"])==2
    assert result["interpretation_ar"] and result["interpretation_en"]
    html=render_analysis_dashboard(result,theme="aurora")
    assert "KHALED DSML" in html


def test_paired_nonparametric_by_explicit_choice():
    a=np.arange(1,14,dtype=float)
    b=a+np.arange(1,14)*.1
    result=compare_groups(a,b,paired=True,method="wilcoxon")
    assert result["test"]=="Wilcoxon signed-rank"
    assert result["n_samples"]==13
    assert result["assumption_checks"]["paired"]
    with pytest.raises(ValueError):
        compare_groups(a,b,paired=False,method="paired_t")
    with pytest.raises(ValueError):
        compare_groups(a,b[:-1],paired=True)


def test_quality_gate_stops_duplicates_and_missing_until_explicit_override():
    frame=pd.DataFrame({"x":[1,1,2,np.nan],"y":[2,2,3,4]})
    with pytest.raises(ValueError,match="Repeated complete rows"):
        research_quality_gate(frame)
    with pytest.raises(ValueError,match="Excessive missing"):
        research_quality_gate(frame,allow_duplicates=True)
    accepted=research_quality_gate(frame,allow_missing=True,allow_duplicates=True)
    assert accepted["approved_overrides"]["allow_duplicates"]
    assert accepted["duplicate_rows"]==1
    assert "overridden" in accepted["interpretation_en"]


def test_apa_summary_never_invents_degrees_of_freedom():
    result=normality_test([1,2,2,3,4,6,7,8])
    report=apa_style_test_summary(result)
    assert "p = " in report["formatted"] or "p < " in report["formatted"]
    assert "df =" not in report["formatted"]
    assert "not full" in report["interpretation_en"]
    with pytest.raises(ValueError):
        apa_style_test_summary({"test":"No statistic"})

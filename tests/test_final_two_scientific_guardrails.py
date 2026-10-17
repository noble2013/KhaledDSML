"""Independent scientific reference checks and misuse adversarial tests.

References: R vcdExtra::zero.test, van den Broek (1995);
Wilson (2015) forbids ZIP vs Poisson classical Vuong;
Hoenig & Heisey (2001) observed-power caution.
"""
from html import escape

import numpy as np
import pytest
from scipy import stats

from khaled_dsml import (
    poisson_excess_zero_score, vuong_strictly_nonnested,
    post_study_precision_scenarios, render_analysis_dashboard,
)


def test_van_den_broek_matches_published_r_score_formula_and_scipy():
    rng=np.random.default_rng(90)
    y=rng.poisson(lam=1.4,size=850)
    result=poisson_excess_zero_score(y)
    n=len(y);lam=y.mean();p0=np.exp(-lam);n0=(y==0).sum()
    corrected=n*p0*(1-p0)-n*lam*p0*p0
    numerator=n0-n*p0
    expected_z=numerator/np.sqrt(corrected)
    assert result["lambda_hat"]==pytest.approx(lam)
    assert result["efficient_zero_variance"]==pytest.approx(corrected)
    assert result["statistic"]==pytest.approx(expected_z)
    assert result["chi_square_statistic"]==pytest.approx(expected_z**2)
    assert result["p_value"]==pytest.approx(stats.norm.sf(expected_z))
    assert result["chi_square_two_sided_p_value"]==pytest.approx(
        stats.chi2.sf(expected_z**2,df=1))
    assert "Vuong" in result["interpretation_en"]


def test_directional_zero_inflation_score_detects_excess_but_not_deficit():
    rng=np.random.default_rng(10)
    regular=rng.poisson(lam=2,size=1200)
    zip_like=np.where(rng.random(1200)<.45,0,
                      rng.poisson(lam=2,size=1200))
    excess=poisson_excess_zero_score(zip_like)
    assert excess["excess_zero_alarm"] is True
    assert excess["statistic"]>0
    assert excess["p_value"]<.01
    # A deficit of zeros must never become positive evidence of *excess* zeros.
    deficit=np.repeat(np.array([0,1,2,3],dtype=int),[8,300,300,300])
    r=poisson_excess_zero_score(deficit)
    assert r["statistic"]<0
    assert r["excess_zero_alarm"] is False
    assert r["p_value"]>.5
    assert "p_value" in poisson_excess_zero_score(regular)


@pytest.mark.parametrize("bad",[
    [0]*60, [2.2]*60, [1,2]*10,
    [1]*49+[float("nan")], [1]*59+[-1], [1]*59+[True],
])
def test_zero_score_rejects_invalid_or_unsupported_count_models(bad):
    with pytest.raises(ValueError):
        poisson_excess_zero_score(bad)


def test_vuong_independent_reference_on_genuinely_nonnested_log_densities():
    rng=np.random.default_rng(14)
    observations=rng.lognormal(mean=.2,sigma=.55,size=330)
    log_a=stats.lognorm.logpdf(observations,s=.55,scale=np.exp(.2))
    log_b=stats.expon.logpdf(observations,scale=observations.mean())
    r=vuong_strictly_nonnested(
        log_a,log_b,model_a_family="lognormal",
        model_b_family="exponential",
        relationship="strictly_nonnested",
    )
    differences=log_a-log_b
    z=np.sqrt(len(differences))*differences.mean()/differences.std(ddof=1)
    assert r["statistic"]==pytest.approx(z)
    assert r["p_value"]==pytest.approx(2*stats.norm.sf(abs(z)))
    assert r["favored_if_significant"]=="A"
    assert r["mean_loglik_difference_a_minus_b"]==pytest.approx(
        differences.mean())
    assert "NOT" in r["interpretation_en"]


def test_vuong_blocks_poisson_zip_and_unverified_relationship():
    a=np.linspace(-4,-1,100)
    b=np.linspace(-3,-.8,100)**2*(-.3)
    for names in [("Poisson","ZIP"),("Poisson","zero-inflated Poisson"),
                  ("zip","Poisson"),("zero_inflated_poisson","ordinary poisson")]:
        with pytest.raises(ValueError,match="overlap"):
            vuong_strictly_nonnested(
                a,b,model_a_family=names[0],model_b_family=names[1],
                relationship="strictly_nonnested",
            )
    with pytest.raises(ValueError,match="Declare"):
        vuong_strictly_nonnested(
            a,b,model_a_family="gamma",model_b_family="lognormal",
            relationship="unknown")
    with pytest.raises(ValueError,match="Indistinguishable"):
        vuong_strictly_nonnested(
            a,a,model_a_family="gamma",model_b_family="lognormal",
            relationship="strictly_nonnested")


def test_poststudy_precision_and_scenarios_match_normal_reference_not_observed_power():
    r=post_study_precision_scenarios(
        estimate=.25,standard_error=.12,
        externally_specified_effects=[0.,.1,.3,.6],alpha=.05,
    )
    zcrit=stats.norm.ppf(.975)
    assert r["ci_lower"]==pytest.approx(.25-zcrit*.12)
    assert r["ci_upper"]==pytest.approx(.25+zcrit*.12)
    table=r["table"]
    for a,expected_power in zip(
        table["externally_hypothesized_true_effect"],
        table["hypothetical_normal_power"],
    ):
        ref=(stats.norm.sf(zcrit-a/.12)+stats.norm.cdf(-zcrit-a/.12))
        assert expected_power==pytest.approx(ref)
    assert table["hypothetical_normal_power"].iloc[0]==pytest.approx(.05)
    assert r["observed_power_computed"] is False
    assert "achieved power" in r["interpretation_en"]
    assert "p_value" not in r


def test_poststudy_does_not_bind_scenarios_to_observed_p_value():
    scenarios=[.2,.4,.6]
    r1=post_study_precision_scenarios(.01,.15,scenarios)
    r2=post_study_precision_scenarios(2.50,.15,scenarios)
    # Same study precision + *externally fixed* true effects yields identical
    # power scenarios no matter how the observed study outcome changed.
    np.testing.assert_allclose(
        r1["table"]["hypothetical_normal_power"],
        r2["table"]["hypothetical_normal_power"],
    )
    assert r1["ci_lower"]!=r2["ci_lower"]
    for kwargs in [
        {"standard_error":0},{"standard_error":-1},
        {"externally_specified_effects":[]},
        {"externally_specified_effects":[.2,.2]},
        {"externally_specified_effects":[float("nan")]},
        {"alpha":True},
    ]:
        args=dict(estimate=.3,standard_error=.15,
                  externally_specified_effects=scenarios)
        args.update(kwargs)
        with pytest.raises(ValueError):
            post_study_precision_scenarios(**args)


def test_scientific_dashboard_explains_direction_and_not_equivalent_power():
    rng=np.random.default_rng(5)
    sample=rng.poisson(1.5,size=400)
    examples=[
        poisson_excess_zero_score(sample),
        vuong_strictly_nonnested(
            stats.norm.logpdf(sample.astype(float),loc=1.5,scale=1.0),
            stats.laplace.logpdf(sample.astype(float),loc=1.5,scale=1.0),
            model_a_family="normal",model_b_family="laplace",
            relationship="strictly_nonnested",
        ),
        post_study_precision_scenarios(.3,.1,[.1,.2,.5]),
    ]
    for result in examples:
        html=render_analysis_dashboard(result,theme="ocean")
        assert 'dir="rtl"' in html
        assert escape(result["interpretation_ar"]) in html
        assert escape(result["interpretation_en"]) in html

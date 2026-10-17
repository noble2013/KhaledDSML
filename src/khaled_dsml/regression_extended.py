"""Classical regression diagnostics: linearity, heteroskedasticity and influence.

Functions accept fitted statsmodels models unless stated otherwise; statistical
assumptions and full model diagnostics are the caller's responsibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._results import make_result


def _sm():
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for regression tests') from exc
    return sm


def _alpha(alpha):
    if not 0<alpha<1:
        raise ValueError("alpha must be in (0,1)")


def regression_white_test(residuals, design, *, alpha=.05):
    """White general heteroskedasticity test. Design must include intercept."""
    _alpha(alpha); _sm()
    from statsmodels.stats.diagnostic import het_white
    y=np.asarray(residuals,dtype=float).reshape(-1)
    x=np.asarray(design,dtype=float)
    if x.ndim!=2 or len(y)!=len(x) or len(x)<8 or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Need finite residuals and aligned 2D design with intercept")
    if not np.any(np.all(np.isclose(x,1),axis=0)):
        raise ValueError("White test design must contain an intercept column")
    stat,p,_,_=het_white(y,x)
    return make_result("White heteroskedasticity",stat,p,len(y),alpha,
        "تباين الأخطاء ثابت وفق نموذج الاختبار.",
        "اختبار مرتبط بملاءمة نموذج المتوسط وعدم وجود اعتمادية زمنية غير معالجة.",
        null_en="The conditional error variance is homoskedastic.",
        notes_en="Model specification and residual dependence affect validity.")


def regression_goldfeld_quandt_test(y, design, *, split=.5, alpha=.05):
    """Goldfeld–Quandt variance test, observations must be ordered appropriately."""
    _alpha(alpha); _sm()
    from statsmodels.stats.diagnostic import het_goldfeldquandt
    outcome=np.asarray(y,dtype=float).reshape(-1)
    X=np.asarray(design,dtype=float)
    if X.ndim!=2 or len(outcome)!=len(X) or len(X)<12 or not np.isfinite(outcome).all() or not np.isfinite(X).all():
        raise ValueError("Need finite aligned outcome and design, at least 12 cases")
    if not .25<=split<=.75:
        raise ValueError("split must lie in [.25,.75]")
    stat,p,_=het_goldfeldquandt(outcome,X,split=split,alternative="two-sided")
    return make_result("Goldfeld-Quandt",stat,p,len(outcome),alpha,
        "تباين الأخطاء متساوٍ في شطري العينة.",
        "يجب ترتيب الصفوف وفق المتغير المحدد مسبقًا؛ لا ترتبها بعد رؤية النتائج.",
        null_en="The error variances of the two ordered subsamples are equal.",
        notes_en="Observation ordering must be prespecified, not selected after inspecting the test.")


def regression_reset_test(fitted_ols, *, power=3, alpha=.05):
    """Ramsey RESET of omitted nonlinear combinations of fitted values."""
    _alpha(alpha); _sm()
    from statsmodels.stats.diagnostic import linear_reset
    if type(power) is not int or power<2:
        raise ValueError("power must be integer >=2")
    r=linear_reset(fitted_ols,power=power,use_f=True)
    return make_result("Ramsey RESET",float(r.fvalue),float(r.pvalue),
        int(fitted_ols.nobs),alpha,
        "لا توجد دلائل على خطأ شكل الدالة ضمن توسعات RESET المستخدمة.",
        "رفض العدمية لا يحدد المتغير المفقود أو الشكل الصحيح للدالة.",
        null_en="Selected higher-order fitted-value terms do not improve the mean specification.",
        notes_en="Rejection does not identify the omitted predictor or correct functional form.")


def regression_rainbow_test(fitted_ols, *, fraction=.5, alpha=.05):
    """Rainbow linearity test using statsmodels OLS fitted result."""
    _alpha(alpha); _sm()
    from statsmodels.stats.diagnostic import linear_rainbow
    if not 0<fraction<1:
        raise ValueError("fraction must lie in (0,1)")
    stat,p=linear_rainbow(fitted_ols,frac=fraction)
    return make_result("Rainbow linearity",stat,p,int(fitted_ols.nobs),alpha,
        "تتلاءم الصيغة الخطية مع السلوك المفترض وفق اختبار Rainbow.",
        "يعتمد الاختبار على ترتيب الصفوف وطريقة اختيار الجزء المركزي.",
        null_en="The linear specification is adequate according to the Rainbow test.",
        notes_en="Inference depends on observation ordering and central fraction.")


def regression_influence_table(fitted_ols) -> pd.DataFrame:
    """Cook's D, leverage, DFFITS and maximum |DFBETAS| for fitted OLS."""
    _sm()
    influence=fitted_ols.get_influence()
    cooks,_=influence.cooks_distance
    hat=influence.hat_matrix_diag
    dffits,_=influence.dffits
    dfbetas=np.asarray(influence.dfbetas,dtype=float)
    n=int(fitted_ols.nobs)
    return pd.DataFrame({
        "cooks_d":cooks,
        "leverage":hat,
        "dffits":dffits,
        "max_abs_dfbetas":np.max(np.abs(dfbetas),axis=1),
        "cooks_reference_4_over_n":np.full(n,4/n),
        "interpretation_ar":np.repeat("مؤشر للمراجعة وليس قاعدة حذف تلقائي",n),
        "interpretation_en":np.repeat("Review flag, never automatic deletion rule",n),
    },index=getattr(fitted_ols.model.data,"row_labels",None))


def chow_structural_break_test(y, design, *, break_index, alpha=.05):
    """Chow F test at a *prespecified* structural break, including intercept."""
    _alpha(alpha)
    values=np.asarray(y,dtype=float).reshape(-1)
    x=np.asarray(design,dtype=float)
    if x.ndim!=2 or len(x)!=len(values) or not np.isfinite(values).all() or not np.isfinite(x).all():
        raise ValueError("Need a finite aligned outcome and design matrix")
    n,k=x.shape
    if type(break_index) is not int or break_index<k+2 or break_index>n-k-2:
        raise ValueError("break_index must leave more observations than regressors in both segments")
    for part in (x,x[:break_index],x[break_index:]):
        if np.linalg.matrix_rank(part)!=k:
            raise ValueError("Model design must have full column rank in both periods")
    def ssr(a,b):
        coef=np.linalg.lstsq(a,b,rcond=None)[0]
        return float(np.sum((b-a@coef)**2))
    pooled=ssr(x,values)
    two=ssr(x[:break_index],values[:break_index])+ssr(x[break_index:],values[break_index:])
    den_df=n-2*k
    if two<=0 or den_df<=0:
        raise ValueError("Break test undefined for perfect subgroup fit")
    f=max(0,(pooled-two)/k)/(two/den_df)
    p=stats.f.sf(f,k,den_df)
    return make_result("Chow prespecified break",f,p,n,alpha,
        "معاملات الانحدار مستقرة في جزأي العينة حسب نقطة القطع المحددة مسبقًا.",
        f"فهرس القطع={break_index}؛ اختبار لا يصلح لاختيار القطع بعد الاطلاع على البيانات.",
        null_en="Regression coefficients are stable at the prespecified split.",
        notes_en=f"Split index={break_index}; post-hoc split selection invalidates the nominal p-value.")


def regression_cusum_test(fitted_ols, *, alpha=.05):
    """CUSUM test of OLS residual parameter stability (statsmodels reference)."""
    _alpha(alpha); _sm()
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    resid=np.asarray(fitted_ols.resid,dtype=float)
    ddof=int(fitted_ols.df_model)+1
    stat,p,_=breaks_cusumolsresid(resid,ddof=ddof)
    return make_result("OLS CUSUM",stat,p,len(resid),alpha,
        "معاملات الانحدار ثابتة عبر ترتيب العينة.",
        "يتطلب ترتيبًا زمنيًا ملائمًا واستيفاء شروط نموذج OLS.",
        null_en="OLS regression coefficients remain stable across ordered observations.",
        notes_en="Requires meaningful ordering and appropriate OLS conditions.")


def model_information_criteria(models) -> pd.DataFrame:
    """Comparable fitted-model AIC/BIC; require common likelihood/sample."""
    if not hasattr(models,"items") or len(models)<2:
        raise ValueError("Provide at least two named fitted statsmodels models")
    rows=[]
    n_values=set()
    for name,model in models.items():
        if not hasattr(model,"aic") or not hasattr(model,"bic") or not hasattr(model,"nobs"):
            raise TypeError("Values must be fitted likelihood-based model results")
        n_values.add(int(model.nobs))
        rows.append({"model":str(name),"aic":float(model.aic),"bic":float(model.bic),
                     "log_likelihood":float(model.llf),"n_samples":int(model.nobs)})
    if len(n_values)!=1:
        raise ValueError("AIC/BIC compare models fitted on identical sample sizes")
    return pd.DataFrame(rows).set_index("model").sort_values("aic")


def logistic_hosmer_lemeshow(y_true, probability, *, groups=10, alpha=.05):
    """Hosmer–Lemeshow calibration grouping diagnostic.

    Depends on grouping and approximate chi-square reference; prefer
    calibration plots and proper scoring rules as complementary checks.
    """
    _alpha(alpha)
    y=np.asarray(y_true,dtype=float).reshape(-1)
    p=np.asarray(probability,dtype=float).reshape(-1)
    if len(y)<30 or len(y)!=len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Need >=30 paired finite actual outcomes and probabilities")
    if not np.isin(y,[0,1]).all() or np.any((p<=0)|(p>=1)):
        raise ValueError("Use binary 0/1 targets and strictly interior predicted probabilities")
    if type(groups) is not int or groups<3 or groups>len(y)//5:
        raise ValueError("Need 3 or more groups with at least 5 observations per group")
    order=np.argsort(p)
    bins=np.array_split(order,groups)
    chi2=0
    for idx in bins:
        observed=y[idx].sum()
        expected=p[idx].sum()
        variance=expected*(1-expected/len(idx))
        if variance<=0:
            raise ValueError("Degenerate expected grouping")
        chi2+=(observed-expected)**2/variance
    pvalue=stats.chi2.sf(chi2,groups-2)
    return make_result("Hosmer-Lemeshow grouped calibration",chi2,pvalue,len(y),alpha,
        "الفروق بين الأحداث المرصودة والمتوقعة ضمن المجموعات متوافقة مع النموذج.",
        "قيمة p تقريبية وتعتمد على طريقة التجميع؛ لا تستخدمها وحدها لتقييم المعايرة.",
        null_en="Observed and expected event frequencies are consistent within grouped risk bins.",
        notes_en="The approximate p-value is grouping-sensitive and is not a standalone calibration assessment.")


def glm_overdispersion_diagnostic(fitted_glm) -> dict:
    """Pearson chi-square/df dispersion heuristic for a fitted GLM."""
    if not hasattr(fitted_glm,"pearson_chi2") or not hasattr(fitted_glm,"df_resid"):
        raise TypeError("Expected a fitted statsmodels GLM result")
    df=float(fitted_glm.df_resid)
    if df<=0:
        raise ValueError("Residual degrees of freedom must be positive")
    ratio=float(fitted_glm.pearson_chi2/df)
    return {
        "pearson_dispersion":ratio,"df_residual":df,
        "interpretation_ar":"نسبة تشتت Pearson وصفية وتعتمد على عائلة GLM؛ ارتفاعها ليس إثباتًا تلقائيًا لسوء التحديد.",
        "interpretation_en":"Pearson dispersion is family-dependent and does not alone prove model misspecification.",
    }


def two_way_anova(data, *, outcome, factor_a, factor_b, typ=2) -> pd.DataFrame:
    """Two-factor OLS ANOVA with interaction; balance/design assumptions apply."""
    _sm()
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    if len({outcome,factor_a,factor_b})!=3:
        raise ValueError("Outcome and factor names must differ")
    df=pd.DataFrame(data)
    if any(c not in df.columns for c in (outcome,factor_a,factor_b)):
        raise KeyError("Unknown column in two_way_anova")
    if typ not in (1,2,3):
        raise ValueError("typ must be 1, 2, or 3")
    clean=df[[outcome,factor_a,factor_b]].dropna().copy()
    clean.columns=["_y","_a","_b"]
    if not pd.api.types.is_numeric_dtype(clean["_y"]) or clean["_a"].nunique()<2 or clean["_b"].nunique()<2:
        raise ValueError("Need numeric outcome and >=2 levels of each factor")
    if len(clean)<10:
        raise ValueError("Need at least 10 complete observations")
    fitted=smf.ols("_y ~ C(_a)*C(_b)",data=clean).fit()
    if fitted.df_resid<=0:
        raise ValueError("Insufficient residual degrees of freedom")
    return anova_lm(fitted,typ=typ)

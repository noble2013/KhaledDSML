# KhaledDSML 0.10 — Model validation and integrity | تقييم وتحقق نماذج تعلّم الآلة

**Branch / الفرع:** `feature/v0.10-ml-validation-expansion`, `0.10.0.dev0`. This is source development; the official PyPI release remains **0.9.0**. All examples use synthetic data, and all report interpretations are Arabic RTL plus English LTR.

## 1. Nested CV / التحقق المتقاطع المتداخل

`nested_cv_model_evaluation(estimator, X, y, param_grid, outer_cv=3, inner_cv=3, scoring="accuracy")` uses `GridSearchCV` only within each outer training fold. Outer test observations never enter hyperparameter selection. Reports a table of outer holdout scores and corresponding training-only best parameters, mean and SD (descriptive). It does **not** return a trained production model or invent a p-value/confidence interval. Model preprocessing, feature selection and imputation must all be inside the estimator's **Pipeline**.

The current helper supports independently sampled rows and stratified classification or ordinary KFold regression. **Not for repeated individuals, grouped cases, temporal streams, spatially clustered units or nested survey samples.** Select an appropriate group/time splitter and scientific design before trusting any score.

## 2. Permutation model score / اختبار أداء تبديل الملصقات

`permutation_score_significance(estimator, X, y, n_permutations=99, scoring="accuracy")` compares ordinary cross-validated predictive performance against repeated label shuffles and refits. SciKit-learn's finite-simulation p-value is `(1 + # null scores >= observed)/(B + 1)`; its granularity is `1/(B+1)`. **Under H0 labels must be exchangeable.** Selection of hypotheses after testing, dependent observations, leakage or a globally optimized feature subset can invalidate interpretation. It does not prove a causal relationship or generalize to nonexchangeable clustered/time-series cases.

## 3. Feature importance on a reserved holdout / أهمية السمات

`holdout_permutation_importance(fitted_estimator, X_holdout, y_holdout, scoring="accuracy", n_repeats=10)` shuffles a single feature at a time on a **held-out dataset**, keeping the fitted model unchanged. Returns average score decrease and across-repeat dispersion for each feature. Highly correlated features may share or hide importance, and a negative score decrease may be legitimate. This is **not causal importance**, a confidence interval, or a feature-selection method to be applied to data used to fit the estimator.

## 4. Learning curve / منحنى التعلّم

`learning_curve_diagnostic(estimator, X, y, cv=3, train_sizes=(.5,.75,1.), scoring="accuracy")` clones and retrains an estimator within CV for increasing subset sizes. Reports training and validation scores. The output is descriptive and CV fold results are correlated: **not an independent external validation**.

## 5. Dummy baseline / خط الأساس الساذج

`dummy_baseline_cv(X,y,task="classification",scoring="accuracy")` fits `DummyClassifier(strategy="most_frequent")` separately per training fold, or `DummyRegressor(strategy="mean")` for regression. Reported scores are held-out CV fold scores. This prevents accidentally computing a dummy response statistic using evaluation labels. For regression use an appropriate scorer such as `"neg_mean_absolute_error"`.

## Reproducible teaching example | مثال تدريسي قابل للتنفيذ

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from khaled_dsml import (
    nested_cv_model_evaluation, permutation_score_significance,
    holdout_permutation_importance, learning_curve_diagnostic,
    dummy_baseline_cv, display_analysis_dashboard,
)

X0, y0 = make_classification(
    n_samples=120, n_features=5, n_informative=3,
    n_redundant=0, random_state=17
)
X = pd.DataFrame(X0, columns=[f"x{i}" for i in range(5)])
model = Pipeline([
    ("scale", StandardScaler()),
    ("logit", LogisticRegression(solver="liblinear", max_iter=350)),
])
nested = nested_cv_model_evaluation(
    model, X, y0, {"logit__C": [.1, 1.]},
    inner_cv=3, outer_cv=3
)
display_analysis_dashboard(nested, theme="amethyst")

dummy = dummy_baseline_cv(X, y0)
display_analysis_dashboard(dummy, theme="aurora")
```

For importance, partition the data first, fit the estimator on the training subset only and pass the genuinely reserved test subset to `holdout_permutation_importance`. Do not select the model or features on that same evaluation subset after inspecting the result.

## Coverage and review scope | التغطية والمراجعة

- **Five new callable candidates:** IDs #167, #177, #179, #187 and #193.
- **One shared method scope** (#178): Y-randomization via the label-permutation predictive-score test. This is not evidence of every form of Y-randomization.
- **Seven legacy implementations/API references reconciled:** #26 and #200 (`ljung_box_test`), #98 (`breusch_pagan_test`), #125 (`cronbach_alpha`), #196 (`adf_test`), #207 (`regression_cusum_test`, specifically **regression residual** CUSUM), and #227 (`adjust_pvalues`, not a full experimental sequential correction protocol). For #26/#200/#98/#125/#196 the existing functions in `econometrics.py` were newly exported at package root, without importing statsmodels eagerly.
- **209 / 256** catalogue rows now have candidate public callable evidence, **47** remain unmapped. These are mapping figures, NOT scientifically validated completion counts. Review each composite method and every independent numerical assumption before sign-off.

## Quality gate | بوابة الجودة

`tests/test_ml_validation_advanced.py` compares all five new routines to explicit scikit-learn reference calculations with the same random seeds and fold definitions, validates input/error cases, checks estimator cloning, tests Arabic/English Analytics Studio rendering, and ensures no invented p-values for descriptive statistics. Full tests are run on Python 3.10–3.13; real Spark/Dask integration and distribution build/install remain mandatory to merge. Existing v0.9.0 PyPI remains unchanged.

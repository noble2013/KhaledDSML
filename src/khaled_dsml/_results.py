"""Bilingual research result types and conservative inferential interpretations."""
from dataclasses import asdict, dataclass
import math

_H0_EN = {
    "Shapiro-Wilk": "The observations are drawn from a normal distribution.",
    "Jarque-Bera": "Population skewness and kurtosis are consistent with normality.",
    "Student t-test": "The two population means are equal.",
    "Welch t-test": "The two population means are equal.",
    "Paired t-test": "The population mean of paired differences is zero.",
    "Mann-Whitney U": "The distributions of the two groups are identical.",
    "Wilcoxon signed-rank": "The paired difference distribution is symmetric about zero.",
    "One-way ANOVA": "All group population means are equal.",
    "Kruskal-Wallis": "The group distributions are identical.",
    "Brown-Forsythe (Levene)": "The groups have equal population variances.",
    "Pearson Chi-square independence": "The categorical variables are independent.",
    "Pearson correlation": "The population linear correlation is zero.",
    "Spearman correlation": "The population rank correlation is zero.",
    "Two-sample Kolmogorov-Smirnov": "The two samples come from the same distribution.",
    "Distribution drift / KS": "The reference and current data follow the same distribution.",
    "One-sample t-test": "The population mean equals the hypothesized value.",
    "D'Agostino K-squared": "The sampled distribution is normal.",
    "Bartlett": "All group population variances are equal.",
    "Fligner-Killeen": "All group population variances are equal.",
    "Kendall tau": "The population rank association is zero.",
    "Fisher exact": "The row and column variables are independent (odds ratio is one).",
    "Chi-square goodness-of-fit": "The observed category probabilities match the expected probabilities.",
    "Friedman": "The repeated-treatment distributions have identical location under the test assumptions.",
}


@dataclass(frozen=True)
class TestResult:
    test: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    interpretation_ar: str
    null_hypothesis_ar: str
    sample_size: int
    notes: str = ""
    # Appended fields preserve all existing positional TestResult calls.
    interpretation_en: str = ""
    null_hypothesis_en: str = ""
    notes_en: str = ""

    def __post_init__(self):
        if not self.interpretation_en:
            message = (
                f"Reject the null hypothesis at significance level {self.alpha:g}; "
                "the data provide statistical evidence against it."
                if self.reject_null else
                f"Do not reject the null hypothesis at significance level {self.alpha:g}; "
                "this does not establish that it is true."
            )
            object.__setattr__(self, "interpretation_en", message)
        if not self.null_hypothesis_en:
            object.__setattr__(
                self, "null_hypothesis_en",
                _H0_EN.get(self.test, "Refer to the test-specific null hypothesis given in Arabic."),
            )

    def to_dict(self) -> dict:
        return asdict(self)


def make_result(
    name: str, statistic: float, p_value: float, n: int,
    alpha: float, null_ar: str, notes: str = "",
    null_en: str = "", notes_en: str = "",
) -> TestResult:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    if not math.isfinite(float(p_value)) or not (0 <= float(p_value) <= 1):
        raise ValueError("Test returned an invalid p-value; inspect assumptions and data")
    reject = bool(p_value < alpha)
    interpretation = (
        f"نرفض الفرضية الصفرية عند مستوى دلالة {alpha:g}؛ "
        "توجد أدلة إحصائية ضدها."
        if reject else
        f"لا نرفض الفرضية الصفرية عند مستوى دلالة {alpha:g}؛ "
        "لا يعني ذلك إثبات صحتها."
    )
    return TestResult(
        name, float(statistic), float(p_value), float(alpha),
        reject, interpretation, null_ar, int(n), notes,
        null_hypothesis_en=null_en, notes_en=notes_en,
    )

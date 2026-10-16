"""Simple, teaching-oriented results for statistical hypothesis tests."""
from dataclasses import asdict, dataclass
import math


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

    def to_dict(self) -> dict:
        return asdict(self)


def make_result(
    name: str, statistic: float, p_value: float, n: int,
    alpha: float, null_ar: str, notes: str = "",
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
    return TestResult(name, float(statistic), float(p_value), float(alpha),
                      reject, interpretation, null_ar, int(n), notes)

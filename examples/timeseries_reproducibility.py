"""Classroom example for KhaledDSML 0.3.0 (unreleased development branch).

No external datasets or internet access are required.
"""
from pathlib import Path

import numpy as np

from khaled_dsml import (
    arch_lm_test, box_pierce_test, forecast_accuracy_report,
    mann_kendall_trend_test, pettitt_change_point_test,
    reproducibility_manifest, save_reproducibility_manifest,
    verify_reproducibility_manifest,
)


def main():
    rng = np.random.default_rng(42)
    series = np.linspace(70, 100, 40) + rng.normal(scale=1.5, size=40)
    trend = mann_kendall_trend_test(series)
    print("العربية:", trend.interpretation_ar)
    print("English:", trend.interpretation_en)

    shifted = np.r_[np.full(20, 10), np.full(20, 20)]
    change = pettitt_change_point_test(shifted)
    print(change["interpretation_ar"])
    print(change["interpretation_en"])

    residuals = rng.normal(size=120)
    print(box_pierce_test(residuals, lags=6).interpretation_en)
    print(arch_lm_test(residuals, lags=4).interpretation_ar)

    training = [100, 110, 120, 130, 140]
    forecasts = forecast_accuracy_report([150, 160], [148, 163], training)
    print(forecasts["interpretation_ar"])
    print(forecasts["interpretation_en"])

    data = Path("economic_monthly_data.csv")
    data.write_text("month,sales" + chr(10) + "2026-01,100" + chr(10),
                    encoding="utf-8")
    manifest = reproducibility_manifest(data, seed=42)
    path = save_reproducibility_manifest(manifest, "research_manifest.json")
    print("Saved:", path)
    print(verify_reproducibility_manifest(manifest, data))


if __name__ == "__main__":
    main()

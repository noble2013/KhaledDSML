"""Economic demand example: monthly sales forecasts, synthetic data."""
import numpy as np
import pandas as pd
from khaled_dsml import split_data, make_regressor, regression_metrics

rng = np.random.default_rng(2026)
X = pd.DataFrame({"price_da": rng.uniform(40, 100, 200),
                  "ad_budget_da": rng.uniform(2000, 20000, 200),
                  "wilaya": rng.choice(["El Oued", "Oran", "Alger"], 200)})
y = 1500 - 8 * X["price_da"] + .02 * X["ad_budget_da"] + rng.normal(0, 30, 200)
X_train, X_test, y_train, y_test = split_data(X, y)
model = make_regressor(X_train, "ridge", alpha=1.0)
model.fit(X_train, y_train)
print(regression_metrics(y_test, model.predict(X_test)))

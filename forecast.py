
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA


# ==================================================
# 1. LOAD AND PREPARE DATA
# ==================================================
df = pd.read_csv("dataset/hour_engineered.csv")

df["timestamp"] = (
    pd.to_datetime(df["dteday"])
    + pd.to_timedelta(df["hr"], unit="h")
)

df = df.sort_values("timestamp").reset_index(drop=True)

# Keep the original observed target values.
df = df.dropna(subset=["cnt"]).reset_index(drop=True)

y = df["cnt"].astype(float)
timestamps = df["timestamp"]

print("=" * 55)
print("BIKE SHARING DEMAND - ARMA FORECAST")
print("=" * 55)

print("Total observations:", len(y))
print("Duplicate timestamps:", timestamps.duplicated().sum())

full_index = pd.date_range(
    start=timestamps.min(),
    end=timestamps.max(),
    freq="h"
)

missing_timestamps = full_index.difference(
    pd.DatetimeIndex(timestamps)
)

print("Missing hourly timestamps:", len(missing_timestamps))


# ==================================================
# 2. ADF STATIONARITY TEST
# ==================================================
print("\n" + "=" * 55)
print("ADF STATIONARITY TEST")
print("=" * 55)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    adf_result = adfuller(y)

print("ADF statistic:", round(adf_result[0], 4))
print("p-value:", adf_result[1])

if adf_result[1] < 0.05:
    print("Evidence of stationarity.")
else:
    print("Stationarity has not been established.")


# ==================================================
# 3. PLOT ORIGINAL TIME SERIES
# ==================================================
plt.figure(figsize=(12, 5))
plt.plot(timestamps, y)

plt.title("Hourly Bike Sharing Demand")
plt.xlabel("Time")
plt.ylabel("Bike Rentals (cnt)")

plt.tight_layout()
plt.show()


# ==================================================
# 4. PLOT AUTOCORRELATION FUNCTION
# ==================================================
fig, ax = plt.subplots(figsize=(12, 5))

plot_acf(y, lags=48, ax=ax)

ax.set_title("ACF - Original Bike Sharing Demand")
ax.set_xlabel("Lag (observations)")
ax.set_ylabel("Autocorrelation")

plt.tight_layout()
plt.show()


# ==================================================
# 5. CHRONOLOGICAL 80/20 TRAIN/TEST SPLIT
# ==================================================
split_idx = int(len(y) * 0.8)

train = y.iloc[:split_idx].reset_index(drop=True)
test = y.iloc[split_idx:].reset_index(drop=True)
test_timestamps = timestamps.iloc[split_idx:].reset_index(drop=True)

print("\n" + "=" * 55)
print("TRAIN/TEST SPLIT")
print("=" * 55)

print("Training observations:", len(train))
print("Testing observations:", len(test))


# ==================================================
# 6. FIT ARMA(2,1)
# ==================================================
# ARMA(p,q) is represented as ARIMA(p,0,q) in statsmodels.
# Integer indexes make the model operate on consecutive
# observed rows, even where actual timestamps are missing.
p, q = 2, 1

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    fitted_model = ARIMA(
        train,
        order=(p, 0, q),
        trend="c"
    ).fit()

print("\nFitted model: ARMA(2,1)")


# ==================================================
# 7. ROLLING ONE-STEP-AHEAD FORECASTS
# ==================================================
actual_values = []
predicted_values = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for actual_value in test:

        # Forecast the next observed row.
        prediction = fitted_model.forecast(steps=1).iloc[0]

        actual_values.append(float(actual_value))
        predicted_values.append(float(prediction))

        # Update the model with the actual observation.
        # Parameters remain fixed; the model state is updated.
        fitted_model = fitted_model.append(
            [float(actual_value)],
            refit=False
        )


# ==================================================
# 8. EVALUATE MODEL
# ==================================================
actual = np.asarray(actual_values)
predicted = np.asarray(predicted_values)

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print("\n" + "=" * 55)
print("ARMA(2,1) TEST RESULTS")
print("=" * 55)

print("Evaluated test observations:", len(actual))
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# ==================================================
# 9. PLOT ACTUAL VS PREDICTED DEMAND
# ==================================================
results = pd.DataFrame({
    "timestamp": test_timestamps,
    "Actual": actual,
    "Predicted": predicted
})

plt.figure(figsize=(14, 6))

plt.plot(
    results["timestamp"],
    results["Actual"],
    label="Actual Demand",
    alpha=0.75
)

plt.plot(
    results["timestamp"],
    results["Predicted"],
    label="ARMA(2,1) Forecast",
    alpha=0.8
)

plt.title("ARMA(2,1): Actual vs Predicted Bike Demand")
plt.xlabel("Time")
plt.ylabel("Bike Rentals (cnt)")
plt.legend()

plt.tight_layout()
plt.show()

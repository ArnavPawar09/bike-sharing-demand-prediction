import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from statsmodels.discrete.discrete_model import NegativeBinomial


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv("dataset/hour_engineered.csv")

df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

print("=" * 60)
print("BIKE-SHARING DEMAND PREDICTION")
print("=" * 60)

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())


# ============================================================
# 2. DEFINE TARGET AND FEATURES
# ============================================================

# Target: total hourly rentals
y = df["cnt"]

# Exclude leakage columns and redundant features
excluded_columns = [
    "cnt",
    "casual",
    "registered",
    "instant",
    "dteday",
    "atemp",
    "mnth",
    "day_of_month",
    "hour_sin",
    "hour_cos"
]

X = df.drop(columns=excluded_columns)

# We will model different hourly patterns on working and
# non-working days using hour-workingday interactions.
# Therefore, workingday itself is omitted to avoid redundancy.
X = X.drop(columns=["workingday"])

# Categorical predictors
categorical_features = [
    "season",
    "yr",
    "month",
    "hr",
    "holiday",
    "weekday",
    "weathersit"
]

# Continuous numerical predictors
numerical_features = [
    "temp",
    "hum",
    "windspeed"
]

# Add hourly interaction features
for hour in range(24):
    X[f"hour_{hour}_workingday"] = (
        (df["hr"] == hour) &
        (df["workingday"] == 1)
    ).astype(int)

interaction_features = [
    f"hour_{hour}_workingday" for hour in range(24)
]

numerical_features += interaction_features

# Confirm that the target contains non-negative counts
assert (y >= 0).all(), "Target contains negative rental counts."


# ============================================================
# 3. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

# First 80% for training, final 20% for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print("\nTRAIN-TEST SPLIT")
print("-" * 60)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training period:",
      df["dteday"].iloc[0],
      "to",
      df["dteday"].iloc[split_index - 1])
print("Testing period:",
      df["dteday"].iloc[split_index],
      "to",
      df["dteday"].iloc[-1])


# ============================================================
# 4. CALCULATE MEAN, VARIANCE AND DISPERSION
# ============================================================

target_mean = y_train.mean()
target_variance = y_train.var()

print("\nCOUNT DISTRIBUTION")
print("-" * 60)
print(f"Training target mean:     {target_mean:.2f}")
print(f"Training target variance: {target_variance:.2f}")
print(f"Variance / mean ratio:    "
      f"{target_variance / target_mean:.2f}")

if target_variance > target_mean:
    print("Initial indication: variance exceeds mean.")
    print("The data may be overdispersed.")
else:
    print("Initial indication: no overall overdispersion "
          "by this simple check.")


# ============================================================
# 5. PREPROCESSING
# ============================================================

# Fit the encoder using training data only.
# The same encoded feature matrix will be used for all models.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ],
    remainder="drop"
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Convert sparse matrices to dense arrays for statsmodels
if hasattr(X_train_encoded, "toarray"):
    X_train_encoded = X_train_encoded.toarray()

if hasattr(X_test_encoded, "toarray"):
    X_test_encoded = X_test_encoded.toarray()

X_train_encoded = np.asarray(X_train_encoded, dtype=float)
X_test_encoded = np.asarray(X_test_encoded, dtype=float)

# Add an intercept for the statsmodels negative binomial model
X_train_nb = sm.add_constant(
    X_train_encoded,
    has_constant="add"
)

X_test_nb = sm.add_constant(
    X_test_encoded,
    has_constant="add"
)

print("\nNumber of encoded predictors:",
      X_train_encoded.shape[1])


# ============================================================
# 6. MODEL EVALUATION FUNCTION
# ============================================================

results = {}
fitted_models = {}
predictions = {}

def evaluate_model(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)

    train_pred = np.asarray(model.predict(Xtr))
    test_pred = np.asarray(model.predict(Xte))

    train_mae = mean_absolute_error(ytr, train_pred)
    test_mae = mean_absolute_error(yte, test_pred)

    train_rmse = np.sqrt(mean_squared_error(ytr, train_pred))
    test_rmse = np.sqrt(mean_squared_error(yte, test_pred))

    train_r2 = r2_score(ytr, train_pred)
    test_r2 = r2_score(yte, test_pred)

    results[name] = {
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train R2": train_r2,
        "Test R2": test_r2
    }

    fitted_models[name] = model
    predictions[name] = test_pred

    print(f"\n{name}")
    print("-" * 45)
    print(f"Training MAE:  {train_mae:.2f}")
    print(f"Testing MAE:   {test_mae:.2f}")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE:  {test_rmse:.2f}")
    print(f"Training R2:   {train_r2:.4f}")
    print(f"Testing R2:    {test_r2:.4f}")


# ============================================================
# 7. MULTIPLE LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

evaluate_model(
    "Multiple Linear Regression",
    linear_model,
    X_train_encoded,
    y_train,
    X_test_encoded,
    y_test
)


# ============================================================
# 8. POISSON REGRESSION
# ============================================================

poisson_model = Pipeline([
    (
        "regressor",
        PoissonRegressor(
            alpha=1.0,
            max_iter=2000
        )
    )
])

evaluate_model(
    "Poisson Regression",
    poisson_model,
    X_train_encoded,
    y_train,
    X_test_encoded,
    y_test
)


# ============================================================
# 9. NEGATIVE BINOMIAL REGRESSION
# ============================================================

print("\nFitting negative binomial regression...")

try:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        nb_model = NegativeBinomial(
            endog=y_train.to_numpy(),
            exog=X_train_nb,
            loglike_method="nb2"
        ).fit(
            disp=False,
            maxiter=500
        )

    if caught_warnings:
        print("\nNegative binomial fitting warnings:")
        for warning in caught_warnings:
            print("-", warning.message)

    if not nb_model.mle_retvals.get("converged", False):
        print("\nWARNING: Negative binomial optimization "
              "did not converge. Interpret its results cautiously.")

    nb_train_pred = np.asarray(
        nb_model.predict(X_train_nb)
    )
    nb_test_pred = np.asarray(
        nb_model.predict(X_test_nb)
    )

    results["Negative Binomial Regression"] = {
        "Train MAE": mean_absolute_error(
            y_train, nb_train_pred
        ),
        "Test MAE": mean_absolute_error(
            y_test, nb_test_pred
        ),
        "Train RMSE": np.sqrt(
            mean_squared_error(y_train, nb_train_pred)
        ),
        "Test RMSE": np.sqrt(
            mean_squared_error(y_test, nb_test_pred)
        ),
        "Train R2": r2_score(y_train, nb_train_pred),
        "Test R2": r2_score(y_test, nb_test_pred)
    }

    fitted_models["Negative Binomial Regression"] = nb_model
    predictions["Negative Binomial Regression"] = nb_test_pred

    print("\nNegative Binomial Regression")
    print("-" * 45)

    for metric, value in results[
        "Negative Binomial Regression"
    ].items():
        print(f"{metric}: {value:.4f}")

    print(f"Estimated dispersion alpha: "
          f"{nb_model.params[-1]:.6f}")

except Exception as error:
    print("\nNegative binomial model could not be fitted.")
    print("Reason:", error)


# ============================================================
# 10. MODEL-BASED POISSON DISPERSION DIAGNOSTIC
# ============================================================

# Pearson dispersion = Pearson chi-square / residual degrees
# of freedom. Values substantially above 1 can indicate
# conditional overdispersion.

poisson_predictions_train = np.maximum(
    predictions_train := np.asarray(
        poisson_model.predict(X_train_encoded)
    ),
    1e-8
)

pearson_chi_square = np.sum(
    (y_train.to_numpy() - poisson_predictions_train) ** 2
    / poisson_predictions_train
)

residual_df = len(y_train) - X_train_encoded.shape[1] - 1

pearson_dispersion = pearson_chi_square / residual_df

print("\nPOISSON DISPERSION DIAGNOSTIC")
print("-" * 45)
print(f"Pearson chi-square: {pearson_chi_square:.2f}")
print(f"Residual degrees of freedom: {residual_df}")
print(f"Pearson dispersion statistic: "
      f"{pearson_dispersion:.4f}")

if pearson_dispersion > 1.5:
    print("The statistic suggests potential conditional "
          "overdispersion.")
else:
    print("The statistic does not show strong overdispersion "
          "by this rough threshold.")

# Note: this is a diagnostic, not a formal hypothesis test.


# ============================================================
# 11. FINAL MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(results).T

print("\n" + "=" * 75)
print("FINAL REGRESSION MODEL COMPARISON")
print("=" * 75)
print(comparison.round(4).to_string())

if not comparison.empty:
    best_name = comparison["Test RMSE"].idxmin()

    print("\nBest model by test RMSE:", best_name)
    print(f"Test RMSE: "
          f"{comparison.loc[best_name, 'Test RMSE']:.2f}")
    print(f"Test MAE:  "
          f"{comparison.loc[best_name, 'Test MAE']:.2f}")
    print(f"Test R2:   "
          f"{comparison.loc[best_name, 'Test R2']:.4f}")


# ============================================================
# 12. ACTUAL VS PREDICTED PLOT
# ============================================================

import matplotlib.pyplot as plt

best_predictions = predictions[best_name]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_predictions, alpha=0.3)

upper_limit = max(
    float(y_test.max()),
    float(np.max(best_predictions))
)

plt.plot(
    [0, upper_limit],
    [0, upper_limit],
    "r--",
    label="Perfect predictions"
)

plt.xlabel("Actual Rentals")
plt.ylabel("Predicted Rentals")
plt.title(f"Actual vs Predicted: {best_name}")
plt.legend()
plt.tight_layout()
plt.show()

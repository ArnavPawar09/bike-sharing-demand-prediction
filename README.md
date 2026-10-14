# Bike Sharing Demand Prediction

A Predictive Analytics project that analyses hourly bike-sharing demand and builds predictive models using regression techniques and time-series forecasting.

## Project Overview

Bike-sharing demand varies according to time of day, season, weather, and working-day patterns. Predicting demand can help bike-sharing systems plan bicycle availability and improve resource allocation.

This project uses the UCI Bike Sharing Dataset to explore these patterns, compare regression models, and evaluate an ARMA time-series forecasting model.

## Objectives

- Perform exploratory data analysis (EDA) on hourly bike-sharing data.
- Clean and prepare the dataset for modelling.
- Engineer useful time-based features.
- Compare multiple linear regression, Poisson regression, and negative binomial regression.
- Select the best-performing regression model using test-set metrics.
- Apply ARMA(2,1) for time-series forecasting.
- Evaluate the models using appropriate regression and forecasting metrics.

## Dataset

**Source:** [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

The project uses the hourly dataset, `hour.csv`, containing **17,379 observations** from 2011–2012.

The target variable is `cnt`, representing the total number of bike rentals recorded in an hour.

Important variables include:

| Variable | Description |
|---|---|
| `cnt` | Total hourly bike rentals; prediction target |
| `casual` | Rentals by casual users |
| `registered` | Rentals by registered users |
| `hr` | Hour of the day |
| `season` | Season |
| `mnth` | Month |
| `weekday` | Day of the week |
| `workingday` | Whether the day is a working day |
| `weathersit` | Weather category |
| `temp` | Normalized temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |

The `casual` and `registered` variables were excluded from the regression predictors because their sum equals `cnt`, which would introduce target leakage.

## Exploratory Data Analysis

The EDA stage examined the target distribution, hourly demand patterns, correlations, outliers, and seasonal variation.

Key observations:

- Hourly rental counts were strongly right-skewed.
- Demand showed prominent peaks around approximately 8 AM and 5 PM.
- Demand varied across seasons.
- `casual` and `registered` were strongly correlated with `cnt` because they are components of the target.
- `temp` and `atemp` were highly correlated, so `temp` was retained while `atemp` was excluded from the regression models.
- High-demand observations were retained unless there was evidence that they were erroneous.

## Data Preparation and Feature Engineering

The project uses Python and pandas for data preparation.

The workflow included:

1. Checking the data for missing values and duplicate rows.
2. Preserving the original dataset files.
3. Creating a cleaned dataset.
4. Engineering additional date and time features, including month, day of the month, and cyclical hour features.
5. Encoding categorical variables for regression modelling.
6. Excluding target-derived variables and unnecessary columns from the predictor set.
7. Using a chronological 80/20 train/test split.

The cyclical hour features represent the periodic nature of time, where the end of the day is close to the beginning of the next day.

## Regression Models

Three regression models were evaluated.

### 1. Multiple Linear Regression

Models the target as a linear combination of explanatory variables. It serves as a baseline for comparison.

### 2. Poisson Regression

A count-data regression model suitable for non-negative count outcomes. The Poisson model assumes that the conditional mean and variance are equal.

### 3. Negative Binomial Regression

A count-data model that can accommodate overdispersion, where the variance exceeds the mean.

The observed target mean was approximately **174.64**, while its variance was approximately **27,874.11**. The variance-to-mean ratio was approximately **159.61**, providing strong evidence of overdispersion and motivating the use of negative binomial regression.

### Regression Results

All metrics below are from the recorded chronological test-set evaluation.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 78.16 | 105.01 | 0.7732 |
| Poisson Regression | 58.66 | 79.59 | 0.8697 |
| Negative Binomial Regression | **47.38** | **77.23** | **0.8773** |

Negative binomial regression was selected as the best-performing regression model among the three evaluated candidates because it achieved the lowest MAE and RMSE and the highest R².

## Time-Series Forecasting: ARMA(2,1)

A separate time-series forecasting experiment was conducted using the hourly rental-count sequence.

### Stationarity Analysis

The Augmented Dickey–Fuller (ADF) test was used to examine stationarity.

| Measure | Result |
|---|---:|
| ADF statistic | -6.8229 |
| ADF p-value | 1.98 × 10⁻⁹ |

The p-value was below the 0.05 significance level, providing evidence against the unit-root null hypothesis. Therefore, the original series was used without differencing.

The Autocorrelation Function (ACF) was also examined. ARMA(2,1) was selected manually rather than through an exhaustive parameter search.

In statsmodels, ARMA(p,q) is represented as ARIMA(p,0,q), meaning the differencing order is zero.

### Forecasting Procedure

1. Sort the observations chronologically.
2. Split the data into 80% training and 20% testing observations.
3. Fit ARMA(2,1) on the training observations.
4. Generate rolling one-step-ahead forecasts for the test observations.
5. Update the model state using each actual observation while keeping model parameters fixed.
6. Evaluate the predictions using MAE and RMSE.

### ARMA Results

| Measure | Result |
|---|---:|
| Training observations | 13,903 |
| Test observations evaluated | 3,476 |
| MAE | 72.91 |
| RMSE | 111.67 |

The actual-versus-predicted plot shows that the model captures the broad repeated demand pattern, although some individual peaks are overestimated or underestimated.

These are rolling one-step-ahead results: actual observations from earlier test rows are used to update the model state before later predictions.

## Evaluation Metrics

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted rental counts. Lower values indicate smaller average errors.

### Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction error. It penalizes large errors more strongly than MAE.

### R² Score

Used for the regression models to measure performance relative to a mean-prediction baseline. Higher values generally indicate a better fit on the evaluated data.

Classification metrics such as accuracy, precision, recall, and F1-score were not used because the target is a numerical count rather than a discrete class label.

## Technologies Used

- **Python**
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib and Seaborn** — visualization
- **scikit-learn** — preprocessing, regression estimators, and evaluation metrics
- **statsmodels** — negative binomial regression, ADF testing, ACF analysis, and ARMA forecasting
- **Visual Studio Code** — development environment

## Project Structure

```text
bike-sharing-demand-prediction/
├── dataset/
│   ├── bike+sharing+dataset/
│   │   ├── hour.csv
│   │   ├── day.csv
│   │   └── Readme.txt
│   ├── hour_cleaned.csv
│   └── hour_engineered.csv
├── main.py
├── model.py
├── forecast.py
├── requirements.txt
├── .gitignore
├── README.md
└── Bike_Sharing_Demand_Prediction_Project_Report.docx
```

The project report may be included in the repository if desired.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bike-sharing-demand-prediction.git
cd bike-sharing-demand-prediction
```

Replace `YOUR_USERNAME` with the relevant GitHub username.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run exploratory data analysis

```bash
python main.py
```

### 5. Run regression modelling

```bash
python model.py
```

### 6. Run ARMA forecasting

```bash
python forecast.py
```

Run these commands from the repository root so that the relative dataset paths resolve correctly.

## Assumptions and Limitations

- The recorded rental counts are treated as a reasonable measure of hourly demand.
- The chronological split evaluates predictions on later observations rather than randomly selected rows.
- There are **165 missing hourly timestamps** in the dataset.
- The final ARMA implementation does not impute missing rental counts. It treats consecutive observed rows as consecutive model steps, even where the actual elapsed time exceeds one hour.
- The ARMA model order was selected manually; exhaustive parameter optimization was not performed.
- The ADF result supports stationarity under the test assumptions but does not guarantee constant variance or the absence of daily patterns.
- The ARMA model uses the history of rental counts and does not directly incorporate weather or calendar predictors.
- ARMA results are based on rolling one-step-ahead forecasts, not a single forecast of the entire test period made in advance.
- Reported results apply to this dataset, feature set, split, and implementation; performance may differ on other periods or datasets.

## Conclusion

This project demonstrates a complete workflow for hourly bike-sharing demand prediction, from exploratory data analysis and feature engineering to regression modelling and time-series forecasting.

Negative binomial regression was the strongest of the evaluated regression candidates, achieving a test MAE of 47.38, RMSE of 77.23, and R² of 0.8773. The separate ARMA(2,1) experiment evaluated all 3,476 test observations and achieved an MAE of 72.91 and RMSE of 111.67.

The experiments illustrate two approaches to demand prediction: regression using explanatory features and time-series forecasting using the history of observed demand.

## References

1. Fanaee-T, H. and Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence, 2, 113–127.
2. [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
3. [scikit-learn documentation](https://scikit-learn.org/)
4. [statsmodels documentation](https://www.statsmodels.org/)

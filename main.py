
import pandas as pd

# Load the original hourly dataset
df = pd.read_csv("dataset/bike+sharing+dataset/hour.csv")

# Display basic information
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())

#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of hourly bike rentals
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="cnt", bins=40, kde=True)
plt.title("Distribution of Hourly Bike Rentals")
plt.xlabel("Total Rentals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Hourly demand throughout the day
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="hr", y="cnt", errorbar=None)
plt.title("Average Bike Rentals by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Rentals")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

# Correlation heatmap for numeric variables
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include="number").corr(),
            cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------

# Box plot to inspect outliers in hourly rental counts
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["cnt"])
plt.title("Box Plot of Hourly Bike Rentals")
plt.xlabel("Total Rentals")
plt.tight_layout()
plt.show()

# Average rentals by day of the week
plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="weekday", y="cnt", errorbar=None)
plt.title("Average Bike Rentals by Day of the Week")
plt.xlabel("Weekday (0 = Sunday)")
plt.ylabel("Average Rentals")
plt.tight_layout()
plt.show()

# Average rentals by season
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="season", y="cnt", errorbar=None)
plt.title("Average Bike Rentals by Season")
plt.xlabel("Season (1 = Spring, 2 = Summer, 3 = Fall, 4 = Winter)")
plt.ylabel("Average Rentals")
plt.tight_layout()
plt.show()

# Compare demand on working days and non-working days
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="workingday", y="cnt", errorbar=None)
plt.title("Average Rentals: Working vs Non-Working Days")
plt.xlabel("Working Day (0 = No, 1 = Yes)")
plt.ylabel("Average Rentals")
plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------------

from pathlib import Path

# Work on a copy; never modify the original dataset
cleaned_df = df.copy()

# Remove duplicate rows if any are found
cleaned_df = cleaned_df.drop_duplicates()

# Save the cleaned dataset
output_path = Path("dataset/hour_cleaned.csv")
cleaned_df.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved to: {output_path}")
print("Original shape:", df.shape)
print("Cleaned shape:", cleaned_df.shape)

#---------------------------------------------------------------------------------

# Load the cleaned dataset
cleaned_df = pd.read_csv("dataset/hour_cleaned.csv")

# Convert the date column to datetime
cleaned_df["dteday"] = pd.to_datetime(cleaned_df["dteday"])

# Extract calendar features
cleaned_df["month"] = cleaned_df["dteday"].dt.month
cleaned_df["day_of_month"] = cleaned_df["dteday"].dt.day

# Encode hour cyclically
import numpy as np

cleaned_df["hour_sin"] = np.sin(2 * np.pi * cleaned_df["hr"] / 24)
cleaned_df["hour_cos"] = np.cos(2 * np.pi * cleaned_df["hr"] / 24)

# Save the engineered dataset
cleaned_df.to_csv("dataset/hour_engineered.csv", index=False)

print("\nEngineered dataset saved.")
print("Engineered shape:", cleaned_df.shape)
print("\nNew features:")
print(cleaned_df[["hr", "month", "day_of_month",
                  "hour_sin", "hour_cos"]].head())


#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def main():
    # 1) Read CSV
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")

    # 2) Aggregate monthly totals, sort by YEAR_MONTH
    monthly_df = (
        df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
          .sum()
          .sort_values("YEAR_MONTH")
    )

    # Convert to a float Series
    y = monthly_df["ITEMS"].astype(float)

    # Suppose we want the first 40 obs for training, last 6 for testing
    train_size = 40
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]  # the remaining (46 - 40 = 6) observations

    print(f"Training Set Length: {len(y_train)}")
    print(f"Test Set Length: {len(y_test)}")

    # 3) Fit ARIMA(1,1,0) with linear trend on the training set
    model = ARIMA(y_train, order=(1,1,0), trend="t")
    results = model.fit()

    print("\nTraining Summary:")
    print(results.summary())

    # 4) Forecast the test period
    # steps = length of the test set, so we do multi-step forecast
    steps_ahead = len(y_test)
    forecast = results.forecast(steps=steps_ahead)

    # 5) Compare forecast with actual test data
    forecast_index = y_test.index  # same time indices as the test set
    forecast_series = pd.Series(forecast, index=forecast_index)

    # Print or plot
    print("\nForecast vs. Actual Test Data:")
    comparison_df = pd.DataFrame({
        "Actual": y_test,
        "Forecast": forecast_series
    })
    print(comparison_df)

    # 6) Plot training, test, and forecast
    plt.figure(figsize=(9,5))
    plt.plot(y_train.index, y_train, label="Train", marker='o')
    plt.plot(y_test.index, y_test, label="Test", marker='o')
    plt.plot(forecast_series.index, forecast_series, label="Forecast", marker='x')
    plt.title("ARIMA(1,1,0) Train/Test Split Forecast")
    plt.xlabel("Time Index")
    plt.ylabel("Monthly Items")
    plt.legend()
    plt.tight_layout()
    plt.savefig("arima_train_test_forecast.png")
    plt.show()

if __name__ == "__main__":
    main()

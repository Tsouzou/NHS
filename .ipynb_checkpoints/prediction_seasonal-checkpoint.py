#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def main():
    # 1) Read CSV
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")

    # 2) Aggregate monthly totals and sort
    monthly_df = (df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
                     .sum()
                     .sort_values("YEAR_MONTH"))
    y = monthly_df["ITEMS"].astype(float)

    # Suppose you have 46 data points; train on the first 40
    train_size = 40
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # 3) Fit ARIMA(1,1,0) with a linear trend ('t')
    model = ARIMA(y_train, order=(1,1,0), trend="t")
    results = model.fit()
    print(results.summary())

    # 4) Forecast the same number of steps as y_test
    steps_ahead = len(y_test)
    forecast_obj = results.get_forecast(steps=steps_ahead)

    # 5) Extract the predicted mean + 60% confidence interval
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci = forecast_obj.conf_int(alpha=0.40)  # 60% => alpha=0.40

    # 6) Plot
    plt.figure(figsize=(8,5))

    # Plot training + test data
    plt.plot(y.index[:train_size], y_train, label="Training Data", marker="o")
    plt.plot(y.index[train_size:], y_test, label="Test Data", marker="o")

    # Plot forecast mean
    plt.plot(forecast_mean.index, forecast_mean.values,
             label="Forecast Mean", marker="x")

    # Shaded 60% confidence interval
    plt.fill_between(
        forecast_mean.index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        color='k', alpha=0.1, label="60% CI"
    )

    plt.title("ARIMA(1,1,0) with trend='t' - 60% Confidence Interval")
    plt.xlabel("Time Index")
    plt.ylabel("ITEMS")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

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

    # Set train size = 43, test size = total - 43
    train_size = 43
    test_size = len(y) - train_size
    # Forecast an additional 3 points beyond the test set
    N_future = 3

    # 3) Split data
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]  # test portion

    # 4) Fit ARIMA(1,1,0) with a linear trend ('t')
    model = ARIMA(y_train, order=(1,1,0), trend="t")
    results = model.fit()
    print(results.summary())

    # 5) Forecast the same number of steps as y_test using 80% CI => alpha=0.20
    steps_ahead_test = len(y_test)
    forecast_test_obj = results.get_forecast(steps=steps_ahead_test)
    forecast_test_mean = forecast_test_obj.predicted_mean
    forecast_test_ci = forecast_test_obj.conf_int(alpha=0.20)  # 80% CI for test

    # 6) Forecast test + future (3) => total of test_size + N_future
    forecast_full_obj = results.get_forecast(steps=steps_ahead_test + N_future)
    forecast_full_mean = forecast_full_obj.predicted_mean
    forecast_full_ci = forecast_full_obj.conf_int(alpha=0.20)  # 80% CI for full

    # Separate the future portion
    forecast_future_mean = forecast_full_mean.iloc[test_size:]
    forecast_future_ci = forecast_full_ci.iloc[test_size:]

    # 7) Create indices for plotting
    x_index = np.arange(len(y))        # 0 .. len(y)-1
    train_index = x_index[:train_size] # 0..42
    test_index = x_index[train_size:]  # 43.. end
    future_index = np.arange(train_size + test_size, 
                             train_size + test_size + N_future)
    # future_index covers the 3 future points beyond the test.

    # 8) Plot
    plt.figure(figsize=(8,5))

    # (a) Training + test data
    plt.plot(train_index, y_train, label="Training Data", marker="o")
    plt.plot(test_index, y_test, label="Test Data", marker="o")

    # (b) Test forecast (mean)
    plt.plot(test_index, forecast_test_mean.values, 
             label="Test Forecast (mean)", marker="x")

    # (b.1) Test forecast 80% CI
    plt.fill_between(
        test_index,
        forecast_test_ci.iloc[:, 0],
        forecast_test_ci.iloc[:, 1],
        color='orange', alpha=0.2, label="Test 80% CI"
    )

    # (c) Future forecasts (3 points)
    plt.plot(future_index, forecast_future_mean.values, 
             label="Future (3 steps) Forecast", marker="s", color="green")

    # (c.1) Future 80% CI
    plt.fill_between(
        future_index,
        forecast_future_ci.iloc[:, 0],
        forecast_future_ci.iloc[:, 1],
        color='green', alpha=0.2, label="Future 80% CI"
    )

    plt.title("ARIMA(1,1,0) with trend='t' - Train=43, Test + 3 future (80% CI)")
    plt.xlabel("Time Index")
    plt.ylabel("ITEMS")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

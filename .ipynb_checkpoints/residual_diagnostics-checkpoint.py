#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import pacf

def main():
    # 1) Read your CSV
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")

    # 2) Aggregate monthly totals and sort by month
    monthly_df = (
        df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
          .sum()
          .sort_values("YEAR_MONTH")
    )

    # Convert the series to float, just to ensure correct dtype
    y = monthly_df["ITEMS"].astype(float)

    # 3) Fit ARIMA(1,1,0) specifying a 't' (linear) trend
    model = ARIMA(y, order=(1,1,0), trend="t")
    results = model.fit()

    # 4) Print summary
    print(results.summary())

    # 5) Extract Residuals
    resid = results.resid

    # 6) Plot Residuals Over Time
    plt.figure(figsize=(8,4))
    plt.plot(resid, marker='o', color='blue')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.title("Residuals of ARIMA(1,1,0) Model")
    plt.xlabel("Time Index")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig("resid_timeseries.png")
    plt.show()

    

    # 8) Ljung–Box Test to Check for Any Autocorrelation
    lb_results = acorr_ljungbox(resid, lags=[12,20], return_df=True)
    print("\nLjung–Box Test Results (lags=12,20):")
    print(lb_results)

    # 9) OPTIONALLY, PRINT PACF VALUES (example: up to lag=12)
    # Note: This won't appear in the model summary by default—it's computed separately.
    max_lag = 12
    pacf_vals = pacf(resid, nlags=max_lag)
    print(f"\nPartial Autocorrelation Function (PACF) values up to lag={max_lag}:")
    for lag in range(len(pacf_vals)):
        print(f"  Lag {lag}: {pacf_vals[lag]:.4f}")

if __name__ == "__main__":
    main()

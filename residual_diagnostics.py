#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

def main():
    # 1) Read CSV
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")

    # 2) Aggregate monthly totals
    monthly_df = (
        df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
          .sum()
          .sort_values("YEAR_MONTH")
    )

    # 3) Prepare the time series
    y = monthly_df["ITEMS"].astype(float)

    # 4) Fit ARIMA(1,1,0)
    model = ARIMA(y, order=(1,1,0))
    results = model.fit()

    print(results.summary())

    # 5) Extract Residuals
    resid = results.resid  # or results.resid (both yield the same in ARIMA)

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

    # 7) ACF & PACF of Residuals
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sm.graphics.tsa.plot_acf(resid, lags=12, ax=axes[0], title="ACF of Residuals")
    sm.graphics.tsa.plot_pacf(resid, lags=12, ax=axes[1], title="PACF of Residuals")
    plt.tight_layout()
    plt.savefig("resid_acf_pacf.png")
    plt.show()

    # 8) Ljung–Box Test to Check for Any Autocorrelation
    lb_results = acorr_ljungbox(resid, lags=[12,20], return_df=True)
    print("\nLjung–Box Test Results (lags=12,20):")
    print(lb_results)
    
    print("\nIf p-values are high, there's no evidence of remaining autocorrelation.\n")

if __name__ == "__main__":
    main()

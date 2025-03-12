#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

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
    #    This effectively includes a drift term in the differenced space
    model = ARIMA(y, order=(1,1,0), trend="t")
    results = model.fit()

    # 4) Print summary
    #    Watch for 'time' or 'linear' parameter in the summary, 
    #    which corresponds to the slope in the original data 
    #    (drift in the differenced space).
    print(results.summary())

    # 5) Example: Plot the fitted values vs. actual data
    fitted_vals = results.fittedvalues
    plt.figure(figsize=(8,5))
    plt.plot(y.index, y.values, label="Actual Data", marker="o")
    plt.plot(y.index, fitted_vals, label="Fitted", marker="x")
    plt.title("ARIMA(1,1,0) with trend='t'")
    plt.xlabel("Time Index")
    plt.ylabel("ITEMS")
    plt.legend()
    plt.tight_layout()
    plt.savefig("arima110_trend_t_fit.png")
    plt.show()

    # 6) Access parameters, interpret slope as 'mu' in the differenced domain
    print("\nModel Parameters:")
    print(results.params)  # One of them (likely 'time') is your linear slope

if __name__ == "__main__":
    main()

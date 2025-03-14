#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def main():
    # 1) Read your CSV
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")

    # 2) Group by YEAR_MONTH to get the total monthly ITEM counts, sorted by date
    monthly_totals = (
        df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
          .sum()
          .sort_values("YEAR_MONTH")
    )
    
    # OPTIONAL: If you want a proper DateTime index, you could convert YEAR_MONTH (e.g., 202101) to a date:
    # But for demonstration, let's keep it simple—statsmodels ARIMA can run on an integer index.
    
    # 3) Extract just the counts as a numeric Series
    #    We'll let ARIMA(1,1,0) difference automatically (that's the "1" in the middle).
    y = monthly_totals["ITEMS"].astype(float)

    # 4) Fit an ARIMA(1,1,0) model
    #    This means: AR(1) on the differenced series, no MA term, and difference=1.
    #    We expect a negative AR(1) coefficient if there's a bounce-back effect between months.
    model = ARIMA(y, order=(1,1,0))
    results = model.fit()
    
    # 5) Print summary to see the AR(1) coefficient (labeled AR.L1)
    print(results.summary())
    
    # 6) Optionally, we can plot the fitted vs. actual
    fitted_values = results.fittedvalues  # The in-sample predictions
    plt.figure(figsize=(8,5))
    plt.plot(y.index, y.values, label="Actual ITEMS", marker='o')
    plt.plot(y.index, fitted_values, label="Fitted (ARIMA(1,1,0))", marker='x')
    plt.title("Monthly Antidepressant Items: ARIMA(1,1,0) Fit")
    plt.xlabel("Time Index")
    plt.ylabel("ITEMS")
    plt.legend()
    plt.tight_layout()
    plt.savefig("arima110_fit.png")
    plt.show()


if __name__ == "__main__":
    main()

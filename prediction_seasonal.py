#!/usr/bin/env python3
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def main():
    # 1) Read & group by YEAR_MONTH
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")
    monthly_df = (
        df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
          .sum()
          .sort_values("YEAR_MONTH")
    )

    y = monthly_df["ITEMS"].astype(float)

    # 2) Fit a SARIMAX model with a 12-month seasonal AR(1) term
    #    Non-seasonal: (p=1, d=1, q=0)
    #    Seasonal: (P=1, D=0, Q=0, s=12)
    #    trend='t' (optional) for a linear trend in the original data
    model = SARIMAX(
        endog=y,
        order=(1,1,0),
        seasonal_order=(1,0,0,12),
        trend='t'
    )
    results = model.fit(disp=False)

    # 3) Print summary: look specifically for 'ar.S.L12' row
    print(results.summary())

    # If 'ar.S.L12' is near zero or has a p-value above 0.05, 
    # it means the model sees no strong seasonal effect at lag 12.
    
if __name__ == "__main__":
    main()

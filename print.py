#!/usr/bin/env python3
import pandas as pd

def main():
    # 1) Read your CSV
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")
    
    # 2) Group by YEAR_MONTH to get the total number of items each month
    monthly_trend = (
        df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
          .sum()
          .sort_values("YEAR_MONTH")
    )
    
    # 3) Print the monthly trend
    print("\nMonthly Trend for All Antidepressant Items:")
    print(monthly_trend.to_string(index=False))

if __name__ == "__main__":
    main()

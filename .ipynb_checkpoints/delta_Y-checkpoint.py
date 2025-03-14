#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1) Read the CSV data
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")
    
    # 2) Aggregate by YEAR_MONTH to get monthly totals of ITEMS (if not already done)
    monthly_totals = (
        df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
          .sum()
          .sort_values("YEAR_MONTH")
    )
    
    # 3) Create a string column for plotting on the x-axis
    monthly_totals["YEAR_MONTH_STR"] = monthly_totals["YEAR_MONTH"].astype(str)
    
    # 4) Log-transform the ITEMS: Y_t = ln(ITEMS_t)
    monthly_totals["Y"] = np.log(monthly_totals["ITEMS"])
    
    # 5) Compute the first difference: ΔY_t = Y_{t+1} - Y_t
    monthly_totals["dY"] = monthly_totals["Y"].diff()
    
    # 6) Plot ΔY_t over time - note the first entry is NaN, so we slice [1:]
    plt.figure(figsize=(8,5))
    plt.plot(
        monthly_totals["YEAR_MONTH_STR"].iloc[1:],
        monthly_totals["dY"].iloc[1:],
        marker="o",
        color="blue"
    )
    plt.title(r"$\Delta Y_t$ Over Time (Log-Differences of Monthly Items)", fontsize=14)
    plt.xlabel("Month (YYYYMM)")
    plt.ylabel(r"$\Delta Y_t$")
    plt.xticks(rotation=70)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 7) Save the plot to a file and display
    plt.savefig("plot_dY_over_time.png")
    plt.show()

if __name__ == "__main__":
    main()

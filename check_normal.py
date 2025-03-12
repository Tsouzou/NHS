#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

def main():
    # 1) Read the CSV data
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")

    # 2) Group by YEAR_MONTH to get monthly total ITEMS (if not already aggregated)
    monthly_totals = (
        df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
          .sum()
          .sort_values("YEAR_MONTH")
    )
    
    # Create a string for plotting on the x-axis
    monthly_totals["YEAR_MONTH_STR"] = monthly_totals["YEAR_MONTH"].astype(str)
    
    # 3) Log-transform the ITEMS: Y_t = ln(ITEMS)
    monthly_totals["Y"] = np.log(monthly_totals["ITEMS"])
    
    # 4) Compute first difference: ΔY_t = Y_{t+1} - Y_t
    monthly_totals["dY"] = monthly_totals["Y"].diff()
    
    # Slice off the first NaN row (diff can't compute at t=0)
    dY = monthly_totals["dY"].iloc[1:].dropna()
    
    # A) HISTOGRAM & KERNEL DENSITY
    plt.figure(figsize=(7,4))
    plt.hist(dY, bins=15, density=True, color="skyblue", edgecolor="black", alpha=0.7)
    
    # mean & std of dY
    mu, sigma = dY.mean(), dY.std()
    x_vals = np.linspace(dY.min(), dY.max(), 200)
    normal_pdf = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_vals - mu)/sigma)**2)
    plt.plot(x_vals, normal_pdf, color="red", linewidth=2, label="Normal PDF")
    
    plt.title("Histogram of ΔY (Log-Differences)")
    plt.xlabel("ΔY")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dY_histogram.png")
    plt.show()
    
    # B) QQ Plot
    sm.qqplot(dY, line='45', fit=True)
    plt.title("QQ Plot of ΔY")
    plt.tight_layout()
    plt.savefig("dY_qqplot.png")
    plt.show()
    
    # C) Normality Test (Shapiro–Wilk)
    w_stat, p_val = stats.shapiro(dY)
    print(f"Shapiro–Wilk Test for dY: W={w_stat:.4f}, p-value={p_val:.4f}")
    if p_val < 0.05:
        print("=> We reject the null hypothesis of normality at 5% level.")
    else:
        print("=> We cannot reject normality at 5% level.")
    
    # D) ACF/PACF for checking independence
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sm.graphics.tsa.plot_acf(dY, lags=12, ax=axes[0], title="ACF of ΔY")
    sm.graphics.tsa.plot_pacf(dY, lags=12, ax=axes[1], title="PACF of ΔY")
    plt.tight_layout()
    plt.savefig("dY_acf_pacf.png")
    plt.show()
    
    # E) Ljung–Box test (portmanteau test) for autocorrelation at lag=20
    lb_results = acorr_ljungbox(dY, lags=[20], return_df=True)
    print("\nLjung–Box Test (lag=20):")
    print(lb_results)
    # If p-value is high, no significant autocorrelation => good for i.i.d. assumption
    
    print("\nAll checks completed. Examine plots and test results to assess normality/i.i.d.\n")

if __name__ == "__main__":
    main()

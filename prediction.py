#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def main():
    # 1) Read CSV
    df = pd.read_csv("BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv")

    # 2) Aggregate monthly totals and sort
    monthly_df = (
        df.groupby("YEAR_MONTH", as_index=False)["ITEMS"]
          .sum()
          .sort_values("YEAR_MONTH")
    )
    y = monthly_df["ITEMS"].astype(float)

    # -----------------------------------------------
    # Use the last 5 data points for testing, then
    # forecast 5 points into the future.
    test_size = 5
    N_future = 5
    train_size = len(y) - test_size
    # -----------------------------------------------

    # 3) Split data
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # 4) Fit ARIMA(1,1,0) with a linear trend ('t')
    model = ARIMA(y_train, order=(1,1,0), trend="t")
    results = model.fit()

    params = results.params
    conf_int_90 = results.conf_int(alpha=0.10)  # 90% CI for parameters
    x1_val       = params.get('x1', None)
    arL1_val     = params.get('ar.L1', None)
    sigma2_val   = params.get('sigma2', None)
    sigma_val    = np.sqrt(sigma2_val) if sigma2_val is not None else None

    # Confidence intervals for each parameter
    x1_ci        = conf_int_90.loc['x1']     if 'x1'     in conf_int_90.index else None
    arL1_ci      = conf_int_90.loc['ar.L1']  if 'ar.L1'  in conf_int_90.index else None
    sigma2_ci    = conf_int_90.loc['sigma2'] if 'sigma2' in conf_int_90.index else None

    sigma_ci_lower, sigma_ci_upper = None, None
    if sigma2_ci is not None and sigma2_ci[0] > 0 and sigma2_ci[1] > 0:
        sigma_ci_lower = np.sqrt(sigma2_ci[0])
        sigma_ci_upper = np.sqrt(sigma2_ci[1])

    print("Parameter estimates (90% CI):")
    if x1_val is not None and x1_ci is not None:
        print(f"  x1       = {x1_val:.4f} "
              f"(90% CI: {x1_ci[0]:.4f}, {x1_ci[1]:.4f})")

    if arL1_val is not None and arL1_ci is not None:
        print(f"  ar.L1    = {arL1_val:.4f} "
              f"(90% CI: {arL1_ci[0]:.4f}, {arL1_ci[1]:.4f})")

    if sigma_val is not None and sigma_ci_lower is not None and sigma_ci_upper is not None:
        print(f"  sigma    = {sigma_val:.4f} "
              f"(90% CI: {sigma_ci_lower:.4f}, {sigma_ci_upper:.4f})")
    ########################################################################

    # 5) Create forecasts
    steps_ahead_test = len(y_test)
    forecast_test_obj = results.get_forecast(steps=steps_ahead_test)
    forecast_full_obj = results.get_forecast(steps=steps_ahead_test + N_future)

    # 6) Indices for plotting
    x_index      = np.arange(len(y))        
    train_index  = x_index[:train_size] 
    test_index   = x_index[train_size:]  
    future_index = np.arange(train_size + test_size,
                             train_size + test_size + N_future)

    ########################################################################
    # MULTI-LAYER CONFIDENCE REGIONS (darker in center = higher probability)
    ########################################################################
    # Define multiple confidence levels from wider (lighter) to narrower (darker).
    conf_levels = [0.90, 0.70, 0.50]
    # Corresponding alpha for the fill regions (increasingly darker).
    fill_alphas = [0.2, 0.4, 0.6]

    plt.figure(figsize=(10,5))

    # (a) Training + Test data
    plt.plot(train_index, y_train, label="Training Data", marker="o", color="black")
    plt.plot(test_index, y_test, label="Test Data", marker="o", color="blue")

    # Plot layered CIs for TEST portion
    for i, conf in enumerate(conf_levels):
        alpha_ = 1.0 - conf
        ci_test = forecast_test_obj.conf_int(alpha=alpha_)
        lower_test = ci_test.iloc[:,0]
        upper_test = ci_test.iloc[:,1]
        plt.fill_between(
            test_index,
            lower_test,
            upper_test,
            color="orange",
            alpha=fill_alphas[i],
            label=f"Test {int(conf*100)}% CI" if i == 0 else None
        )

    # Plot layered CIs for FUTURE portion
    forecast_full_ci = {}
    for conf in conf_levels:
        alpha_ = 1.0 - conf
        forecast_full_ci[conf] = forecast_full_obj.conf_int(alpha=alpha_)

    for i, conf in enumerate(conf_levels):
        ci_future_part = forecast_full_ci[conf].iloc[test_size:]
        lower_future = ci_future_part.iloc[:,0]
        upper_future = ci_future_part.iloc[:,1]
        plt.fill_between(
            future_index,
            lower_future,
            upper_future,
            color="green",
            alpha=fill_alphas[i],
            label=f"Future {int(conf*100)}% CI" if i == 0 else None
        )

    plt.title("ARIMA(1,1,0) with Layered Confidence Intervals (Darker = More Likely)")
    plt.xlabel("Time Index")
    plt.ylabel("ITEMS")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

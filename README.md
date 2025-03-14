## The final work is Report.pdf, converted from Report.ipynb


The coding process begins by reading and grouping raw data by YEAR_MONTH, then sorting the monthly totals in chronological order. A log transform is applied to these totals to stabilize variance, and the first difference of the log values removes longer-term trends. Diagnostics such as histograms, QQ plots, and the Shapiro–Wilk test confirm that the differenced data is largely normal. ACF, PACF, and Ljung–Box tests indicate a negative AR(1) “bounce-back” effect, leading to an ARIMA(1,1,0) model that captures how one month’s surge tends to be offset the following month. Optional train–test splits validate the model’s predictive ability, while attempts to add a seasonal component reveal no strong yearly cycle in this relatively short dataset. The final result is a concise time-series explanation: the monthly data rises over time, yet each large monthly jump quickly reverts, yielding near-white noise residuals once the negative lag-1 dependence is accounted for.

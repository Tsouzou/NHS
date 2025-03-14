## The final work is Report.pdf, converted from Report.ipynb


An initial step involved reading raw data from the file “BSA_ODP_PCA_REGIONAL_DRUG_SUMMARY.csv,” followed by grouping the records by YEAR_MONTH. This aggregation produced an overview of monthly antidepressant item counts, which were then sorted chronologically for proper time-series analysis. To make the data more tractable, a log transform was applied to the monthly totals, stabilizing variance. The first difference of these log values was then computed to remove the baseline trend that is inherent in long-term growth.

After obtaining the differenced log values, multiple diagnostic steps were taken. Histograms and QQ plots were generated to investigate whether these differences appeared normally distributed, and the Shapiro–Wilk test provided a formal check on normality. ACF and PACF plots were then employed to identify any correlation across lags, while a Ljung–Box test further quantified if significant autocorrelation persisted. These diagnostics revealed negative autocorrelation at lag one but no major leftover correlation at other lags once the right model was selected.

The chosen model was an ARIMA(1,1,0) specification, fitting a simple but effective equation for the differenced data. The negative AR(1) coefficient signified a bounce-back effect: if prescribing rose sharply one month, the next month was likely to revert toward the previous level. Residual checks showed that this configuration captured the major time-dependent patterns, leaving behind what resembled a white noise process. An optional train–test split was introduced to ensure the model’s forecasting capability over an unseen portion of the data, thereby validating how robustly it predicted future observations based on past trends.

Further exploration touched on potential seasonal behavior by incorporating a seasonal extension with SARIMAX(1,1,0)×(1,0,0,12). However, the estimated seasonal component turned out to be small and statistically insignificant within the relatively short span of available observations. As a result, forecasts remained near-linear, governed primarily by a mild drift and the short-run corrective factor. Domain knowledge could still motivate adding targeted features such as month-specific dummy variables if it were strongly believed that certain months experience bigger—or more systematic—departures from typical levels.

In conclusion, the coding journey examined each step from data import and aggregation through model construction and verification. The final ARIMA(1,1,0) model, assisted by negative lag-one dependence in the differenced data, forwarded a concise explanation for the month-to-month prescribing changes, leaving random residuals with minimal correlation. This outcome serves as a reliable basis for further time-series investigation, whether for refining the negative AR(1) process or exploring deeper seasonal structures when more comprehensive evidence—or data—becomes available.

Time Series Stationarity & Unit Root Tests Explorer
A Streamlit application by Dr Merwan Roudane  for teaching and exploring the fundamentals of stationarity in time series.

Introduction to Stationarity

Definitions of strict vs. weak (covariance) stationarity

Interactive AR(1) vs. non-stationary series generator

Side-by-side plots, ACF/PACF, summary statistics, and ADF test results

Trend Stationary vs. Difference Stationary

Contrast deterministic-trend (TS) processes vs. difference-stationary (DS) processes

Interactive sliders for trend slopes or drift terms

Visualize raw, detrended, and differenced series

ADF tests before/after transformation, with clear pass/fail boxes

Unit Root Tests

Theory behind unit roots in AR(1) processes

Augmented Dickey–Fuller (ADF) specifications (none, constant, trend)

KPSS as a complement, plus overview of PP, DF-GLS, Ng-Perron, breakpoint tests

Tabbed examples showing simulated data and test outputs

Order of Integration

Definition: an I(d) series becomes stationary after d differences

Simulations of I(0), I(1), I(2) series

Automatic differencing, ADF on each level, and conclusion of estimated d

Making Series Stationary

Catalogue of transformations: detrending, differencing, seasonal adjustment, logs/Box-Cox

Interactive demo: choose non-stationary pattern (trend, seasonal, heteroskedastic, random‐walk)

Apply multiple transforms at once, see plots annotated with ADF pass/fail

ACF/PACF comparison and decision-process checklist

Optimal Lag Selection

Why lag order matters for unit‐root tests and AR modeling

Generate AR(1), AR(2), ARMA(1,1), or seasonal AR(4) processes

Compute AIC, BIC, HQIC for a range of lags, plot criteria vs. lag

Fit the chosen lag, show fitted vs. actual, plus residual diagnostics

Interactive ADF Test

Upload your own series or synthesize borderline cases

Choose regression form (“none”, constant, trend) and autolag method

See ADF statistic, p-value, critical values, and interpret pass/fail

Generate Your Own Series

Quick AR/MA/ARMA simulator with slider controls

Instantly plot the series and its ACF/PACF for teaching or testing hypotheses

PDF Viewer

Upload a PDF (e.g. course notes, slides) so it’s saved on the server as STATIONARITY.pdf

Instantly embed that file in the app for reference

Persists until restarted, making it easy to bundle documentation alongside your interactive exploration

Key strengths:

Fully interactive: no code changes required to explore dozens of parameter combinations.

Illustrated with real‐time ADF/KPSS testing and diagnostics.

Modular layout, easy to extend with new tests (e.g. Phillips–Perron) or features (e.g. GARCH modeling).

Self-documenting: upload your own PDF reference manual for students.

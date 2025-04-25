import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
import os
import base64
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Time Series Stationarity Explorer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .main-header { font-size:2.5rem; color:#1E3A8A; text-align:center; margin-bottom:1rem; }
    .section-header { font-size:1.8rem; color:#1E3A8A; margin-top:1rem; margin-bottom:0.5rem; }
    .subsection-header { font-size:1.4rem; color:#2563EB; margin-top:0.8rem; margin-bottom:0.5rem; }
    .concept-box { background-color:#F3F4F6; padding:1rem; border-radius:0.5rem; margin-bottom:1rem; }
    .formula { background-color:#E5E7EB; padding:0.5rem; border-radius:0.3rem; font-family:monospace; overflow-x:auto; }
    .result-box { background-color:#ECFDF5; padding:0.8rem; border-radius:0.4rem; margin-top:0.5rem; }
    .warning-box { background-color:#FEF3C7; padding:0.8rem; border-radius:0.4rem; margin-top:0.5rem; }
</style>
""", unsafe_allow_html=True)

# Helper: Embed a local PDF file
def show_static_pdf(path: str):
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode('utf-8')
    iframe = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(iframe, unsafe_allow_html=True)

# Function: PDF upload & viewer
def interactive_pdf_viewer():
    st.header("PDF Viewer")
    uploaded = st.file_uploader("Upload a PDF to serve as STATIONARITY.pdf", type="pdf")
    if uploaded:
        with open("STATIONARITY.pdf", "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("✅ Saved as STATIONARITY.pdf")
    if os.path.exists("STATIONARITY.pdf"):
        show_static_pdf("STATIONARITY.pdf")
    else:
        st.info("No PDF uploaded yet. Use the uploader above to set STATIONARITY.pdf.")

# Main application
def main():
    st.markdown('<h1 class="main-header">Time Series Stationarity & Unit Root Tests Explorer</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to Section",
        [
            "Introduction to Stationarity",
            "TS vs DS Processes",
            "Unit Root Tests",
            "Integration Order",
            "Making Series Stationary",
            "Lag Selection",
            "Interactive ADF Test",
            "Generate Your Own Series",
            "PDF Viewer"
        ]
    )

    # Dispatch sections
    if section == "Introduction to Stationarity":
        stationarity_intro()
    elif section == "TS vs DS Processes":
        ts_vs_ds()
    elif section == "Unit Root Tests":
        unit_root_tests()
    elif section == "Integration Order":
        integration_order()
    elif section == "Making Series Stationary":
        making_stationary()
    elif section == "Lag Selection":
        lag_selection()
    elif section == "Interactive ADF Test":
        interactive_adf()
    elif section == "Generate Your Own Series":
        generate_series()
    elif section == "PDF Viewer":
        interactive_pdf_viewer()

    # Sidebar credit
    st.sidebar.markdown(
        """
        ---
        **Created by**  
        Dr Merwan Roudane   
        _For educational purposes_
        """,
        unsafe_allow_html=True
    )

def stationarity_intro():
	st.markdown('<h2 class="section-header">Introduction to Stationarity</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">What is Stationarity?</h3>
        <p>A time series is <b>stationary</b> if its statistical properties do not change over time. There are two types of stationarity:</p>
        <ul>
            <li><b>Strict Stationarity:</b> The joint distribution of observations does not change with time</li>
            <li><b>Weak (Covariance) Stationarity:</b> Mean, variance, and autocovariances are constant over time</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Conditions for Weak Stationarity</h3>
        <ol>
            <li>Constant mean: $E[Y_t] = \mu$ for all $t$</li>
            <li>Constant variance: $Var[Y_t] = \sigma^2$ for all $t$</li>
            <li>Autocovariance depends only on lag: $Cov[Y_t, Y_{t+h}] = \gamma_h$ for all $t$ and lag $h$</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

	# Interactive visualization
	st.markdown('<h3 class="subsection-header">Interactive Visualization of Stationary vs Non-Stationary Series</h3>',
				unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		# User inputs for stationary AR process
		st.markdown("### Stationary AR(1) Process")
		ar_coef = st.slider("AR Coefficient (φ)", min_value=-0.99, max_value=0.99, value=0.7, step=0.1)
		ar_mean = st.slider("Process Mean", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
		ar_noise = st.slider("Noise Standard Deviation", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

	with col2:
		# User inputs for non-stationary processes
		st.markdown("### Non-Stationary Process")
		process_type = st.selectbox(
			"Select Type",
			["Random Walk", "Random Walk with Drift", "Trend Stationary", "Heteroskedastic"]
		)

		if process_type == "Random Walk with Drift":
			drift = st.slider("Drift Term", min_value=-1.0, max_value=1.0, value=0.1, step=0.05)
		elif process_type == "Trend Stationary":
			trend_coef = st.slider("Trend Coefficient", min_value=-1.0, max_value=1.0, value=0.1, step=0.05)
		elif process_type == "Heteroskedastic":
			vol_increase = st.slider("Volatility Growth Factor", min_value=1.0, max_value=5.0, value=1.5, step=0.1)

	# Generate data
	np.random.seed(42)  # For reproducibility
	t = 200  # Number of time points
	time = np.arange(t)

	# Stationary AR(1) process
	ar1 = np.zeros(t)
	ar1[0] = ar_mean + np.random.normal(0, ar_noise)
	for i in range(1, t):
		ar1[i] = ar_mean + ar_coef * (ar1[i - 1] - ar_mean) + np.random.normal(0, ar_noise)

	# Non-stationary process
	ns = np.zeros(t)
	ns[0] = 0
	if process_type == "Random Walk":
		for i in range(1, t):
			ns[i] = ns[i - 1] + np.random.normal(0, 1)

	elif process_type == "Random Walk with Drift":
		for i in range(1, t):
			ns[i] = drift + ns[i - 1] + np.random.normal(0, 1)

	elif process_type == "Trend Stationary":
		for i in range(t):
			ns[i] = trend_coef * i + np.random.normal(0, 1)

	elif process_type == "Heteroskedastic":
		for i in range(t):
			ns[i] = np.random.normal(0, 1 + (i / t) * vol_increase)

	# Plot the series
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

	# Plot stationary series
	ax1.plot(time, ar1)
	ax1.set_title(f"Stationary AR(1) Process with φ={ar_coef}", fontsize=14)
	ax1.set_xlabel("Time")
	ax1.set_ylabel("Value")
	ax1.grid(True, alpha=0.3)

	# Plot non-stationary series
	ax2.plot(time, ns)
	ax2.set_title(f"Non-Stationary Process: {process_type}", fontsize=14)
	ax2.set_xlabel("Time")
	ax2.set_ylabel("Value")
	ax2.grid(True, alpha=0.3)

	plt.tight_layout()
	st.pyplot(fig)

	# ACF and PACF
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

	# ACF and PACF for stationary series
	sm.graphics.tsa.plot_acf(ar1, lags=20, ax=ax1, alpha=0.5)
	ax1.set_title("ACF - Stationary Series")

	sm.graphics.tsa.plot_pacf(ar1, lags=20, ax=ax2, alpha=0.5)
	ax2.set_title("PACF - Stationary Series")

	# ACF and PACF for non-stationary series
	sm.graphics.tsa.plot_acf(ns, lags=20, ax=ax3, alpha=0.5)
	ax3.set_title(f"ACF - Non-Stationary Series ({process_type})")

	sm.graphics.tsa.plot_pacf(ns, lags=20, ax=ax4, alpha=0.5)
	ax4.set_title(f"PACF - Non-Stationary Series ({process_type})")

	plt.tight_layout()
	st.pyplot(fig)

	# Statistical properties
	st.markdown('<h3 class="subsection-header">Statistical Properties Comparison</h3>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("### Stationary Series Properties")
		st.markdown(f"**Mean:** {ar1.mean():.4f}")
		st.markdown(f"**Variance:** {ar1.var():.4f}")
		st.markdown(f"**First half mean:** {ar1[:t // 2].mean():.4f}")
		st.markdown(f"**Second half mean:** {ar1[t // 2:].mean():.4f}")
		st.markdown(f"**First half variance:** {ar1[:t // 2].var():.4f}")
		st.markdown(f"**Second half variance:** {ar1[t // 2:].var():.4f}")

		# Perform ADF test for stationary series
		adf_result = adfuller(ar1)
		st.markdown("**ADF Test Results:**")
		st.markdown(f"Test Statistic: {adf_result[0]:.4f}")
		st.markdown(f"p-value: {adf_result[1]:.4f}")

		if adf_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (Reject unit root hypothesis)</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (Failed to reject unit root hypothesis)</div>',
						unsafe_allow_html=True)

	with col2:
		st.markdown("### Non-Stationary Series Properties")
		st.markdown(f"**Mean:** {ns.mean():.4f}")
		st.markdown(f"**Variance:** {ns.var():.4f}")
		st.markdown(f"**First half mean:** {ns[:t // 2].mean():.4f}")
		st.markdown(f"**Second half mean:** {ns[t // 2:].mean():.4f}")
		st.markdown(f"**First half variance:** {ns[:t // 2].var():.4f}")
		st.markdown(f"**Second half variance:** {ns[t // 2:].var():.4f}")

		# Perform ADF test for non-stationary series
		adf_result = adfuller(ns)
		st.markdown("**ADF Test Results:**")
		st.markdown(f"Test Statistic: {adf_result[0]:.4f}")
		st.markdown(f"p-value: {adf_result[1]:.4f}")

		if adf_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (Reject unit root hypothesis)</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (Failed to reject unit root hypothesis)</div>',
						unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Why Stationarity Matters</h3>
        <ul>
            <li>Most time series models assume stationarity</li>
            <li>Non-stationary data can lead to spurious regressions</li>
            <li>Statistical tests have different distributions under non-stationarity</li>
            <li>Forecasts are more reliable with stationary data</li>
            <li>Statistical properties like mean and variance are more meaningful</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def ts_vs_ds():
	st.markdown('<h2 class="section-header">Trend Stationary vs. Difference Stationary Processes</h2>',
				unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Trend Stationary (TS) Processes</h3>
        <p>A process that is stationary around a deterministic trend.</p>
        <div class="formula">
            $Y_t = f(t) + \varepsilon_t$
        </div>
        <p>where $f(t)$ is a deterministic function (often linear: $\alpha + \beta t$) and $\varepsilon_t$ is a stationary process.</p>
        <p><b>Key insight:</b> Removing the trend makes the series stationary.</p>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Difference Stationary (DS) Processes</h3>
        <p>A process that becomes stationary after differencing.</p>
        <div class="formula">
            $\Delta Y_t = Y_t - Y_{t-1} = \mu + \varepsilon_t$
        </div>
        <p>where $\varepsilon_t$ is a stationary process. Random walk (with or without drift) is an example.</p>
        <p><b>Key insight:</b> Differencing makes the series stationary.</p>
    </div>
    """, unsafe_allow_html=True)

	# Interactive visualization
	st.markdown('<h3 class="subsection-header">Interactive Visualization: TS vs DS</h3>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		# Parameters for TS
		st.markdown("### Trend Stationary Process")
		trend_intercept = st.slider("Trend Intercept (α)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
		trend_slope = st.slider("Trend Slope (β)", min_value=-1.0, max_value=1.0, value=0.1, step=0.01)
		ar_coef_ts = st.slider("AR Coefficient for TS", min_value=-0.9, max_value=0.9, value=0.5, step=0.1)

	with col2:
		# Parameters for DS
		st.markdown("### Difference Stationary Process")
		drift = st.slider("Drift Term (μ)", min_value=-0.5, max_value=0.5, value=0.05, step=0.01)

	# Generate data
	np.random.seed(42)  # For reproducibility
	t = 200  # Number of time points
	time = np.arange(t)

	# Trend Stationary process
	trend = trend_intercept + trend_slope * time
	noise_ts = np.zeros(t)
	noise_ts[0] = np.random.normal(0, 1)
	for i in range(1, t):
		noise_ts[i] = ar_coef_ts * noise_ts[i - 1] + np.random.normal(0, 1)
	ts_process = trend + noise_ts

	# Difference Stationary process
	ds_process = np.zeros(t)
	ds_process[0] = 0  # Start at 0
	for i in range(1, t):
		ds_process[i] = ds_process[i - 1] + drift + np.random.normal(0, 1)

	# Plot original series
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

	# Plot TS
	ax1.plot(time, ts_process, label='TS Process')
	ax1.plot(time, trend, 'r--', label='Deterministic Trend')
	ax1.set_title("Trend Stationary (TS) Process", fontsize=14)
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	# Plot DS
	ax2.plot(time, ds_process)
	ax2.set_title("Difference Stationary (DS) Process", fontsize=14)
	ax2.grid(True, alpha=0.3)

	plt.tight_layout()
	st.pyplot(fig)

	# Transformations to make stationary
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

	# Detrended TS
	detrended_ts = ts_process - trend
	ax1.plot(time, detrended_ts)
	ax1.set_title("Detrended TS Process", fontsize=14)
	ax1.grid(True, alpha=0.3)

	# First difference of TS
	diff_ts = np.diff(ts_process)
	ax2.plot(time[1:], diff_ts)
	ax2.set_title("First Difference of TS Process", fontsize=14)
	ax2.grid(True, alpha=0.3)

	# Detrended DS
	linear_trend_ds = np.polyfit(time, ds_process, 1)
	trend_line_ds = linear_trend_ds[0] * time + linear_trend_ds[1]
	detrended_ds = ds_process - trend_line_ds
	ax3.plot(time, detrended_ds)
	ax3.set_title("Detrended DS Process", fontsize=14)
	ax3.grid(True, alpha=0.3)

	# First difference of DS
	diff_ds = np.diff(ds_process)
	ax4.plot(time[1:], diff_ds)
	ax4.set_title("First Difference of DS Process", fontsize=14)
	ax4.grid(True, alpha=0.3)

	plt.tight_layout()
	st.pyplot(fig)

	# Statistical tests
	st.markdown('<h3 class="subsection-header">Statistical Tests</h3>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("### Trend Stationary Process")

		# ADF test on original TS process
		adf_result_ts = adfuller(ts_process)
		st.markdown("**ADF Test on Original TS Process:**")
		st.markdown(f"Test Statistic: {adf_result_ts[0]:.4f}")
		st.markdown(f"p-value: {adf_result_ts[1]:.4f}")

		if adf_result_ts[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (Reject unit root hypothesis)</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (Failed to reject unit root hypothesis)</div>',
						unsafe_allow_html=True)

		# ADF test on detrended TS process
		adf_result_ts_detrended = adfuller(detrended_ts)
		st.markdown("**ADF Test on Detrended TS Process:**")
		st.markdown(f"Test Statistic: {adf_result_ts_detrended[0]:.4f}")
		st.markdown(f"p-value: {adf_result_ts_detrended[1]:.4f}")

		if adf_result_ts_detrended[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (Reject unit root hypothesis)</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (Failed to reject unit root hypothesis)</div>',
						unsafe_allow_html=True)

	with col2:
		st.markdown("### Difference Stationary Process")

		# ADF test on original DS process
		adf_result_ds = adfuller(ds_process)
		st.markdown("**ADF Test on Original DS Process:**")
		st.markdown(f"Test Statistic: {adf_result_ds[0]:.4f}")
		st.markdown(f"p-value: {adf_result_ds[1]:.4f}")

		if adf_result_ds[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (Reject unit root hypothesis)</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (Failed to reject unit root hypothesis)</div>',
						unsafe_allow_html=True)

		# ADF test on differenced DS process
		adf_result_ds_diff = adfuller(diff_ds)
		st.markdown("**ADF Test on Differenced DS Process:**")
		st.markdown(f"Test Statistic: {adf_result_ds_diff[0]:.4f}")
		st.markdown(f"p-value: {adf_result_ds_diff[1]:.4f}")

		if adf_result_ds_diff[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (Reject unit root hypothesis)</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (Failed to reject unit root hypothesis)</div>',
						unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Key Differences Between TS and DS Processes</h3>
        <table>
            <tr>
                <th>Feature</th>
                <th>Trend Stationary (TS)</th>
                <th>Difference Stationary (DS)</th>
            </tr>
            <tr>
                <td>Effect of Shocks</td>
                <td>Temporary (mean-reverting)</td>
                <td>Permanent (persistent)</td>
            </tr>
            <tr>
                <td>Variance</td>
                <td>Constant or bounded</td>
                <td>Increases with time</td>
            </tr>
            <tr>
                <td>Appropriate Transformation</td>
                <td>Detrending</td>
                <td>Differencing</td>
            </tr>
            <tr>
                <td>Forecast Behavior</td>
                <td>Returns to trend</td>
                <td>Follows random walk</td>
            </tr>
            <tr>
                <td>Example</td>
                <td>GDP growth around long-term trend</td>
                <td>Asset prices</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="warning-box">
        <h3 class="subsection-header">Caution: Impact of Misspecification</h3>
        <ul>
            <li>Treating a DS process as TS: Inadequate removal of stochastic trend leads to spurious results</li>
            <li>Treating a TS process as DS: Over-differencing introduces unnecessary moving average components</li>
            <li>Visual inspection alone can be misleading; formal testing is essential</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def unit_root_tests():
	st.markdown('<h2 class="section-header">Unit Root Tests</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">What is a Unit Root?</h3>
        <p>Consider an AR(1) process:</p>
        <div class="formula">
            $Y_t = \phi Y_{t-1} + \varepsilon_t$
        </div>
        <p>A unit root exists when $\phi = 1$, making the process non-stationary (random walk):</p>
        <div class="formula">
            $Y_t = Y_{t-1} + \varepsilon_t \quad \Rightarrow \quad Y_t = Y_0 + \sum_{i=1}^{t} \varepsilon_i$
        </div>
        <p>For stationarity: $|\phi| < 1$ (all roots of the characteristic equation lie outside the unit circle)</p>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Augmented Dickey-Fuller (ADF) Test</h3>
        <p>The ADF test examines the null hypothesis that a time series has a unit root (is non-stationary) against the alternative that it is stationary.</p>

        <h4>General Equation</h4>
        <div class="formula">
            $\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \varepsilon_t$
        </div>

        <p>where:</p>
        <ul>
            <li>$\alpha$ is a constant (drift)</li>
            <li>$\beta t$ is a time trend</li>
            <li>$\gamma = \phi - 1$ (we test $H_0: \gamma = 0$ vs $H_1: \gamma < 0$)</li>
            <li>$p$ lags of $\Delta Y_t$ to control for autocorrelation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

	st.markdown('<h3 class="subsection-header">ADF Test Specifications</h3>', unsafe_allow_html=True)

	tab1, tab2, tab3 = st.tabs(
		["Case 1: No Constant, No Trend", "Case 2: With Constant, No Trend", "Case 3: With Constant and Trend"])

	with tab1:
		st.markdown("""
        <div class="formula">
            $\Delta Y_t = \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \varepsilon_t$
        </div>
        <p><b>Use when:</b> Series clearly has zero mean with no trend</p>
        <p><b>Example:</b> Highly mean-reverting financial series like spreads</p>
        """, unsafe_allow_html=True)

		# Example visualization for Case 1
		np.random.seed(42)
		t = 200
		time = np.arange(t)

		# Generate a mean-zero stationary series
		case1_series = np.zeros(t)
		case1_series[0] = np.random.normal(0, 1)
		for i in range(1, t):
			case1_series[i] = 0.5 * case1_series[i - 1] + np.random.normal(0, 1)

		fig, ax = plt.subplots(figsize=(10, 5))
		ax.plot(time, case1_series)
		ax.axhline(y=0, color='r', linestyle='--')
		ax.set_title("Example: Mean-Zero Series for Case 1", fontsize=14)
		ax.grid(True, alpha=0.3)
		st.pyplot(fig)

		# ADF test result
		adf_result = adfuller(case1_series, regression='n')  # 'nc' for no constant
		st.markdown("**ADF Test Result (No Constant):**")
		st.markdown(f"Test Statistic: {adf_result[0]:.4f}")
		st.markdown(f"p-value: {adf_result[1]:.4f}")

		if adf_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (Reject unit root hypothesis)</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (Failed to reject unit root hypothesis)</div>',
						unsafe_allow_html=True)

	with tab2:
		st.markdown("""
        <div class="formula">
            $\Delta Y_t = \alpha + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \varepsilon_t$
        </div>
        <p><b>Use when:</b> Series has non-zero mean but no visible trend</p>
        <p><b>Example:</b> Mean-reverting series with non-zero equilibrium</p>
        """, unsafe_allow_html=True)

		# Example visualization for Case 2
		np.random.seed(42)
		t = 200
		time = np.arange(t)

		# Generate a non-zero mean stationary series
		case2_series = np.zeros(t)
		mean_level = 5.0
		case2_series[0] = mean_level + np.random.normal(0, 1)
		for i in range(1, t):
			case2_series[i] = mean_level + 0.7 * (case2_series[i - 1] - mean_level) + np.random.normal(0, 1)

		fig, ax = plt.subplots(figsize=(10, 5))
		ax.plot(time, case2_series)
		ax.axhline(y=mean_level, color='r', linestyle='--', label=f'Mean = {mean_level}')
		ax.legend()
		ax.set_title("Example: Non-Zero Mean Series for Case 2", fontsize=14)
		ax.grid(True, alpha=0.3)
		st.pyplot(fig)

		# ADF test result
		adf_result = adfuller(case2_series, regression='c')  # 'c' for constant
		st.markdown("**ADF Test Result (With Constant):**")
		st.markdown(f"Test Statistic: {adf_result[0]:.4f}")
		st.markdown(f"p-value: {adf_result[1]:.4f}")

		if adf_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (Reject unit root hypothesis)</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (Failed to reject unit root hypothesis)</div>',
						unsafe_allow_html=True)

	with tab3:
		st.markdown("""
        <div class="formula">
            $\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \varepsilon_t$
        </div>
        <p><b>Use when:</b> Series exhibits a trend and possibly a non-zero mean</p>
        <p><b>Example:</b> Economic data with growth trends</p>
        """, unsafe_allow_html=True)

		# Example visualization for Case 3
		np.random.seed(42)
		t = 200
		time = np.arange(t)

		# Generate a trend stationary series
		trend_slope = 0.03
		trend = trend_slope * time
		case3_noise = np.zeros(t)
		case3_noise[0] = np.random.normal(0, 1)
		for i in range(1, t):
			case3_noise[i] = 0.6 * case3_noise[i - 1] + np.random.normal(0, 1)
		case3_series = trend + case3_noise

		fig, ax = plt.subplots(figsize=(10, 5))
		ax.plot(time, case3_series)
		ax.plot(time, trend, 'r--', label=f'Trend (slope = {trend_slope})')
		ax.legend()
		ax.set_title("Example: Trend Series for Case 3", fontsize=14)
		ax.grid(True, alpha=0.3)
		st.pyplot(fig)

		# ADF test result
		adf_result = adfuller(case3_series, regression='ct')  # 'ct' for constant and trend
		st.markdown("**ADF Test Result (With Constant and Trend):**")
		st.markdown(f"Test Statistic: {adf_result[0]:.4f}")
		st.markdown(f"p-value: {adf_result[1]:.4f}")

		if adf_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (Reject unit root hypothesis)</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (Failed to reject unit root hypothesis)</div>',
						unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Other Unit Root Tests</h3>
        <h4>Phillips-Perron (PP) Test</h4>
        <ul>
            <li>Non-parametric adjustment to control for autocorrelation</li>
            <li>More robust to heteroskedasticity and serial correlation</li>
            <li>Same null and alternative hypotheses as ADF</li>
        </ul>

        <h4>KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)</h4>
        <ul>
            <li>Reverse null hypothesis: $H_0:$ Series is stationary</li>
            <li>$H_1:$ Series has a unit root</li>
            <li>Useful for confirmatory analysis alongside ADF or PP tests</li>
        </ul>

        <h4>Others</h4>
        <ul>
            <li>Elliott-Rothenberg-Stock Test (DF-GLS): More powerful than standard ADF</li>
            <li>Ng-Perron Test: Suite of tests with good size and power</li>
            <li>Breakpoint Unit Root Tests: Account for structural breaks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

	# Visualization of test comparison
	st.markdown('<h3 class="subsection-header">Comparison of ADF and KPSS Tests</h3>', unsafe_allow_html=True)

	st.markdown("""
    <div class="warning-box">
        <p>Using both ADF and KPSS tests together provides more reliable conclusions:</p>
        <table>
            <tr>
                <th>ADF (H₀: Non-stationary)</th>
                <th>KPSS (H₀: Stationary)</th>
                <th>Conclusion</th>
            </tr>
            <tr>
                <td>Reject H₀</td>
                <td>Fail to Reject H₀</td>
                <td>Strong evidence for stationarity</td>
            </tr>
            <tr>
                <td>Fail to Reject H₀</td>
                <td>Reject H₀</td>
                <td>Strong evidence for non-stationarity</td>
            </tr>
            <tr>
                <td>Reject H₀</td>
                <td>Reject H₀</td>
                <td>Conflicting results, possible structural breaks</td>
            </tr>
            <tr>
                <td>Fail to Reject H₀</td>
                <td>Fail to Reject H₀</td>
                <td>Insufficient evidence, possibly due to small sample</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

	# Interactive visualization of ADF vs KPSS
	st.markdown("### Interactive Example: ADF vs KPSS Tests")

	series_type = st.selectbox(
		"Select Series Type",
		["Stationary", "Non-Stationary", "Trend Stationary", "Borderline Case"]
	)

	# Generate selected series
	np.random.seed(42)
	t = 200
	time = np.arange(t)

	if series_type == "Stationary":
		# AR(1) with phi=0.7
		example_series = np.zeros(t)
		example_series[0] = np.random.normal(0, 1)
		for i in range(1, t):
			example_series[i] = 0.7 * example_series[i - 1] + np.random.normal(0, 1)
		series_description = "AR(1) process with coefficient 0.7"

	elif series_type == "Non-Stationary":
		# Random walk
		example_series = np.zeros(t)
		example_series[0] = 0
		for i in range(1, t):
			example_series[i] = example_series[i - 1] + np.random.normal(0, 1)
		series_description = "Random walk (unit root) process"

	elif series_type == "Trend Stationary":
		# Trend + stationary noise
		trend = 0.05 * time
		noise = np.zeros(t)
		noise[0] = np.random.normal(0, 1)
		for i in range(1, t):
			noise[i] = 0.5 * noise[i - 1] + np.random.normal(0, 1)
		example_series = trend + noise
		series_description = "Trend stationary process with AR(1) noise"

	else:  # Borderline Case
		# Near unit root (phi=0.95)
		example_series = np.zeros(t)
		example_series[0] = np.random.normal(0, 1)
		for i in range(1, t):
			example_series[i] = 0.95 * example_series[i - 1] + np.random.normal(0, 1)
		series_description = "Near unit root process with AR coefficient 0.95"

	# Plot the series
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(time, example_series)
	ax.set_title(f"{series_type} Series: {series_description}", fontsize=14)
	ax.grid(True, alpha=0.3)
	st.pyplot(fig)

	# Perform both tests
	regression_type = st.radio(
		"ADF Test Specification",
		["No Constant, No Trend", "With Constant, No Trend", "With Constant and Trend"]
	)

	if regression_type == "No Constant, No Trend":
		adf_reg = 'n'
	elif regression_type == "With Constant, No Trend":
		adf_reg = 'c'
	else:
		adf_reg = 'ct'

	# Run the tests
	adf_result = adfuller(example_series, regression=adf_reg)

	try:
		kpss_result = kpss(example_series, regression='c' if 'Trend' not in regression_type else 'ct')
	except:
		kpss_result = [0, 0.1]  # Fallback if KPSS fails

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("### ADF Test Results")
		st.markdown(f"**Test Statistic:** {adf_result[0]:.4f}")
		st.markdown(f"**p-value:** {adf_result[1]:.4f}")
		st.markdown(f"**Critical Values:**")
		for key, value in adf_result[4].items():
			st.markdown(f"  - {key}: {value:.4f}")

		if adf_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ Reject H₀: Series is stationary</div>', unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Fail to reject H₀: Series has a unit root</div>',
						unsafe_allow_html=True)

	with col2:
		st.markdown("### KPSS Test Results")
		st.markdown(f"**Test Statistic:** {kpss_result[0]:.4f}")
		st.markdown(f"**p-value:** {kpss_result[1]:.4f}")
		st.markdown(f"**Critical Values:**")
		for key, value in kpss_result[3].items():
			st.markdown(f"  - {key}: {value:.4f}")

		if kpss_result[1] < 0.05:
			st.markdown('<div class="warning-box">❌ Reject H₀: Series is non-stationary</div>', unsafe_allow_html=True)
		else:
			st.markdown('<div class="result-box">✅ Fail to reject H₀: Series is stationary</div>',
						unsafe_allow_html=True)

	# Combined conclusion
	st.markdown("### Combined Test Conclusion")

	adf_reject = adf_result[1] < 0.05
	kpss_reject = kpss_result[1] < 0.05

	if adf_reject and not kpss_reject:
		st.markdown('<div class="result-box">✅ Strong evidence for stationarity</div>', unsafe_allow_html=True)
	elif not adf_reject and kpss_reject:
		st.markdown('<div class="warning-box">❌ Strong evidence for non-stationarity</div>', unsafe_allow_html=True)
	elif adf_reject and kpss_reject:
		st.markdown(
			'<div class="warning-box">⚠️ Conflicting results - possible structural breaks or misspecification</div>',
			unsafe_allow_html=True)
	else:
		st.markdown(
			'<div class="warning-box">⚠️ Insufficient evidence - possibly due to small sample or borderline case</div>',
			unsafe_allow_html=True)


def integration_order():
	st.markdown('<h2 class="section-header">Order of Integration</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Definition</h3>
        <p>A time series $Y_t$ is said to be integrated of order $d$, denoted as $Y_t \sim I(d)$, if it becomes stationary after differencing $d$ times.</p>
        <div class="formula">
            $\Delta^d Y_t \sim I(0)$
        </div>
        <p>where $\Delta^d$ represents the differencing operator applied $d$ times, and $I(0)$ denotes a stationary process.</p>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Common Orders of Integration</h3>
        <ul>
            <li><b>$I(0)$:</b> Series is already stationary (no differencing needed)</li>
            <li><b>$I(1)$:</b> First differences are stationary (most common in economic data)</li>
            <li><b>$I(2)$:</b> Second differences are stationary (less common, sometimes seen in price levels)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

	# Interactive visualization of different integration orders
	st.markdown('<h3 class="subsection-header">Interactive Visualization of Different Integration Orders</h3>',
				unsafe_allow_html=True)

	# Generate example time series of different integration orders
	np.random.seed(42)
	t = 200
	time = np.arange(t)

	# I(0): Stationary process
	i0_series = np.zeros(t)
	i0_series[0] = np.random.normal(0, 1)
	for i in range(1, t):
		i0_series[i] = 0.5 * i0_series[i - 1] + np.random.normal(0, 1)

	# I(1): Random walk process
	i1_series = np.zeros(t)
	i1_series[0] = 0
	for i in range(1, t):
		i1_series[i] = i1_series[i - 1] + np.random.normal(0, 1)

	# I(2): Double integrated process
	i2_series = np.zeros(t)
	i2_series[0] = 0
	i2_series[1] = 0
	for i in range(2, t):
		i2_series[i] = 2 * i2_series[i - 1] - i2_series[i - 2] + np.random.normal(0, 1)

	# Plot original series
	fig, axes = plt.subplots(3, 1, figsize=(12, 10))

	axes[0].plot(time, i0_series)
	axes[0].set_title("I(0) Process: Stationary AR(1)", fontsize=14)
	axes[0].grid(True, alpha=0.3)

	axes[1].plot(time, i1_series)
	axes[1].set_title("I(1) Process: Random Walk", fontsize=14)
	axes[1].grid(True, alpha=0.3)

	axes[2].plot(time, i2_series)
	axes[2].set_title("I(2) Process: Double Integrated", fontsize=14)
	axes[2].grid(True, alpha=0.3)

	plt.tight_layout()
	st.pyplot(fig)

	# Select series for analysis
	series_choice = st.radio(
		"Select Series to Analyze:",
		["I(0) Process", "I(1) Process", "I(2) Process"]
	)

	if series_choice == "I(0) Process":
		series = i0_series
		expected_order = 0
	elif series_choice == "I(1) Process":
		series = i1_series
		expected_order = 1
	else:
		series = i2_series
		expected_order = 2

	# Perform differencing and testing
	st.markdown("### Determining the Order of Integration")

	# Original series
	adf_orig = adfuller(series, regression='c')

	# First difference
	diff1 = np.diff(series)
	adf_diff1 = adfuller(diff1, regression='c')

	# Second difference
	diff2 = np.diff(diff1)
	adf_diff2 = adfuller(diff2, regression='c')

	# Plot the series and its differences
	fig, axes = plt.subplots(3, 1, figsize=(12, 10))

	axes[0].plot(time, series)
	axes[0].set_title(f"Original Series: {series_choice}", fontsize=14)
	axes[0].grid(True, alpha=0.3)

	axes[1].plot(time[1:], diff1)
	axes[1].set_title("First Difference", fontsize=14)
	axes[1].grid(True, alpha=0.3)

	axes[2].plot(time[2:], diff2)
	axes[2].set_title("Second Difference", fontsize=14)
	axes[2].grid(True, alpha=0.3)

	plt.tight_layout()
	st.pyplot(fig)

	# Display ADF test results
	col1, col2, col3 = st.columns(3)

	with col1:
		st.markdown("### Original Series")
		st.markdown(f"**ADF Test Statistic:** {adf_orig[0]:.4f}")
		st.markdown(f"**p-value:** {adf_orig[1]:.4f}")

		if adf_orig[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary</div>', unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary</div>', unsafe_allow_html=True)

	with col2:
		st.markdown("### First Difference")
		st.markdown(f"**ADF Test Statistic:** {adf_diff1[0]:.4f}")
		st.markdown(f"**p-value:** {adf_diff1[1]:.4f}")

		if adf_diff1[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary</div>', unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary</div>', unsafe_allow_html=True)

	with col3:
		st.markdown("### Second Difference")
		st.markdown(f"**ADF Test Statistic:** {adf_diff2[0]:.4f}")
		st.markdown(f"**p-value:** {adf_diff2[1]:.4f}")

		if adf_diff2[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary</div>', unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary</div>', unsafe_allow_html=True)

	# Determine integration order
	if adf_orig[1] < 0.05:
		determined_order = 0
	elif adf_diff1[1] < 0.05:
		determined_order = 1
	elif adf_diff2[1] < 0.05:
		determined_order = 2
	else:
		determined_order = "higher than 2"

	st.markdown("""
    <div class="result-box">
        <h3 class="subsection-header">Conclusion</h3>
        <p>The series appears to be integrated of order <b>{}</b> (I({})).</p>
        <p>Expected integration order for {} is {}.</p>
    </div>
    """.format(determined_order, determined_order, series_choice, expected_order), unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Practical Implications of Integration Order</h3>
        <table>
            <tr>
                <th>Order</th>
                <th>Description</th>
                <th>Examples</th>
                <th>Appropriate Models</th>
            </tr>
            <tr>
                <td>I(0)</td>
                <td>Stationary series, mean-reverting</td>
                <td>Interest rate spreads, growth rates</td>
                <td>ARMA, standard regression</td>
            </tr>
            <tr>
                <td>I(1)</td>
                <td>Non-stationary, stationary in first differences</td>
                <td>GDP, stock prices, exchange rates</td>
                <td>ARIMA, cointegration, error correction</td>
            </tr>
            <tr>
                <td>I(2)</td>
                <td>Non-stationary, need second differences</td>
                <td>Price levels in high inflation economies</td>
                <td>Second-order ARIMA, polynomial trends</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="warning-box">
        <h3 class="subsection-header">Cautions When Determining Integration Order</h3>
        <ul>
            <li>Structural breaks can mimic unit root behavior</li>
            <li>Seasonal patterns may require seasonal differencing</li>
            <li>Near-integrated processes can be difficult to classify</li>
            <li>Over-differencing can introduce unnecessary MA components</li>
            <li>Economic theory should guide interpretation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def making_stationary():
	st.markdown('<h2 class="section-header">Making Time Series Stationary</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Common Transformations</h3>
        <table>
            <tr>
                <th>Non-Stationarity Type</th>
                <th>Transformation Method</th>
                <th>Formula</th>
            </tr>
            <tr>
                <td>Trend (deterministic)</td>
                <td>Detrending</td>
                <td>$Y_t^* = Y_t - f(t)$ where $f(t)$ is trend function</td>
            </tr>
            <tr>
                <td>Trend (stochastic)</td>
                <td>Differencing</td>
                <td>$\Delta Y_t = Y_t - Y_{t-1}$</td>
            </tr>
            <tr>
                <td>Seasonality</td>
                <td>Seasonal differencing</td>
                <td>$\Delta_s Y_t = Y_t - Y_{t-s}$ where $s$ is season length</td>
            </tr>
            <tr>
                <td>Changing variance</td>
                <td>Log/Power transformation</td>
                <td>$\ln(Y_t)$ or $Y_t^\lambda$ (Box-Cox)</td>
            </tr>
            <tr>
                <td>Heteroskedasticity</td>
                <td>ARCH/GARCH modeling</td>
                <td>Explicitly model variance structure</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

	# Interactive demonstration
	st.markdown('<h3 class="subsection-header">Interactive Transformation Demo</h3>', unsafe_allow_html=True)

	col1, col2 = st.columns([1, 2])

	with col1:
		non_stationary_type = st.selectbox(
			"Select Non-Stationarity Type",
			["Linear Trend", "Exponential Trend", "Cyclical", "Heteroskedastic", "Seasonal", "Random Walk"]
		)

		sample_size = st.slider("Sample Size", min_value=100, max_value=500, value=200, step=50)

		if non_stationary_type == "Linear Trend":
			trend_slope = st.slider("Trend Slope", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

		elif non_stationary_type == "Exponential Trend":
			growth_rate = st.slider("Growth Rate (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

		elif non_stationary_type == "Cyclical":
			cycle_amplitude = st.slider("Cycle Amplitude", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
			cycle_frequency = st.slider("Cycle Frequency", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

		elif non_stationary_type == "Heteroskedastic":
			volatility_growth = st.slider("Volatility Growth", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

		elif non_stationary_type == "Seasonal":
			season_length = st.slider("Season Length", min_value=4, max_value=12, value=4, step=1)
			seasonal_amplitude = st.slider("Seasonal Amplitude", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

		elif non_stationary_type == "Random Walk":
			drift = st.slider("Drift Term", min_value=-0.2, max_value=0.2, value=0.05, step=0.01)

		transformation_method = st.multiselect(
			"Select Transformation(s) to Apply",
			["First Differencing", "Detrending", "Log Transform", "Seasonal Differencing", "Deseasonalizing"],
			default=["First Differencing"]
		)

	# Generate data based on selected type
	np.random.seed(42)
	time = np.arange(sample_size)

	if non_stationary_type == "Linear Trend":
		trend = trend_slope * time
		noise = np.random.normal(0, 1, sample_size)
		series = trend + noise

	elif non_stationary_type == "Exponential Trend":
		trend = np.exp(growth_rate / 100 * time)
		noise = np.random.normal(0, max(0.1, trend.std() * 0.1), sample_size)
		series = trend + noise

	elif non_stationary_type == "Cyclical":
		cycle = cycle_amplitude * np.sin(cycle_frequency * time)
		noise = np.random.normal(0, 1, sample_size)
		series = cycle + noise

	elif non_stationary_type == "Heteroskedastic":
		series = np.zeros(sample_size)
		for i in range(sample_size):
			volat = 1 + (i / sample_size) * volatility_growth
			series[i] = np.random.normal(0, volat)

	elif non_stationary_type == "Seasonal":
		season = seasonal_amplitude * np.sin(2 * np.pi * time / season_length)
		trend = 0.05 * time
		noise = np.random.normal(0, 1, sample_size)
		series = trend + season + noise

	elif non_stationary_type == "Random Walk":
		series = np.zeros(sample_size)
		series[0] = 0
		for i in range(1, sample_size):
			series[i] = drift + series[i - 1] + np.random.normal(0, 1)

	# Apply selected transformations
	transformed_series = []
	transformation_names = []

	# Original series
	transformed_series.append(series)
	transformation_names.append("Original Series")

	if "First Differencing" in transformation_method:
		diff = np.diff(series)
		transformed_series.append(diff)
		transformation_names.append("First Difference")

	if "Detrending" in transformation_method:
		# Linear detrending
		x = np.arange(len(series))
		slope, intercept = np.polyfit(x, series, 1)
		trend_line = intercept + slope * x
		detrended = series - trend_line
		transformed_series.append(detrended)
		transformation_names.append("Detrended (Linear)")

	if "Log Transform" in transformation_method:
		# Make sure all values are positive for log transform
		if np.min(series) <= 0:
			log_series = np.log(series - np.min(series) + 1)
		else:
			log_series = np.log(series)
		transformed_series.append(log_series)
		transformation_names.append("Log Transform")

	if "Seasonal Differencing" in transformation_method and len(series) > 12:
		# Use 4 for quarterly, 12 for monthly
		season_period = min(season_length, 12)
		seas_diff = series[season_period:] - series[:-season_period]
		transformed_series.append(seas_diff)
		transformation_names.append(f"Seasonal Diff (s={season_period})")

	if "Deseasonalizing" in transformation_method and non_stationary_type == "Seasonal":
		# Simple seasonal adjustment for demonstration
		deseasonalized = series - season
		transformed_series.append(deseasonalized)
		transformation_names.append("Deseasonalized")

	# Plot results
	with col2:
		num_transforms = len(transformed_series)
		fig, axes = plt.subplots(num_transforms, 1, figsize=(10, 3 * num_transforms))

		if num_transforms == 1:
			axes = [axes]  # Make sure axes is always a list

		for i, (ts, name) in enumerate(zip(transformed_series, transformation_names)):
			x_vals = np.arange(len(ts))
			axes[i].plot(x_vals, ts)
			axes[i].set_title(name, fontsize=12)
			axes[i].grid(True, alpha=0.3)

			# Perform stationarity test
			if len(ts) > 10:  # Ensure enough data points
				try:
					adf_result = adfuller(ts, regression='c')
					is_stationary = adf_result[1] < 0.05

					if is_stationary:
						axes[i].set_facecolor((0.9, 1.0, 0.9))  # Light green for stationary
						stationary_text = "✅ Stationary (p={:.4f})".format(adf_result[1])
					else:
						axes[i].set_facecolor((1.0, 0.9, 0.9))  # Light red for non-stationary
						stationary_text = "❌ Non-stationary (p={:.4f})".format(adf_result[1])

					axes[i].text(0.02, 0.05, stationary_text, transform=axes[i].transAxes,
								 bbox=dict(facecolor='white', alpha=0.8))
				except:
					pass  # Skip if test fails

		plt.tight_layout()
		st.pyplot(fig)

	# ACF and PACF plots for original and transformed series
	st.markdown('<h3 class="subsection-header">ACF and PACF Comparison</h3>', unsafe_allow_html=True)

	selected_series = st.selectbox(
		"Select Series to Analyze",
		transformation_names
	)

	series_idx = transformation_names.index(selected_series)
	selected_data = transformed_series[series_idx]

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

	# ACF plot
	sm.graphics.tsa.plot_acf(selected_data, lags=min(30, len(selected_data) // 3), ax=ax1)
	ax1.set_title(f"ACF: {selected_series}")

	# PACF plot
	sm.graphics.tsa.plot_pacf(selected_data, lags=min(30, len(selected_data) // 3), ax=ax2)
	ax2.set_title(f"PACF: {selected_series}")

	plt.tight_layout()
	st.pyplot(fig)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Transformation Decision Process</h3>
        <ol>
            <li><b>Stabilize variance first</b> - Use log or power transformations if variance changes with level</li>
            <li><b>Remove seasonality</b> - Use seasonal differencing or seasonal adjustment</li>
            <li><b>Remove trend</b> - Use differencing for stochastic trends, detrending for deterministic trends</li>
            <li><b>Check stationarity</b> - Verify with formal tests like ADF or KPSS</li>
            <li><b>Examine ACF/PACF</b> - Should decay quickly for stationary series</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="warning-box">
        <h3 class="subsection-header">Important Considerations</h3>
        <ul>
            <li>Each transformation changes the interpretation of the series</li>
            <li>Over-differencing introduces unnecessary complexity (MA component)</li>
            <li>Some information might be lost during transformation</li>
            <li>Consider domain knowledge in selecting transformations</li>
            <li>Balance statistical requirements with interpretability</li>
            <li>For forecasting, transformations must be reversed at the end</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def lag_selection():
	st.markdown('<h2 class="section-header">Optimal Lag Selection</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Importance of Lag Selection</h3>
        <ul>
            <li>Too few lags: residual autocorrelation remains, biased results</li>
            <li>Too many lags: reduced power, efficiency, and precision</li>
            <li>Impacts unit root test results and model performance</li>
            <li>Critical for correct model specification</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Information Criteria</h3>
        <p>Common information criteria for selecting optimal lag length:</p>
        <div class="formula">
            $\text{AIC} = -2\ln(L) + 2k$
        </div>
        <div class="formula">
            $\text{BIC} = -2\ln(L) + k\ln(n)$
        </div>
        <div class="formula">
            $\text{HQIC} = -2\ln(L) + 2k\ln(\ln(n))$
        </div>
        <p>where $L$ is the likelihood, $k$ is the number of parameters, and $n$ is the sample size.</p>
    </div>
    """, unsafe_allow_html=True)

	# Interactive lag selection
	st.markdown('<h3 class="subsection-header">Interactive Lag Selection Demo</h3>', unsafe_allow_html=True)

	col1, col2 = st.columns([1, 2])

	with col1:
		process_type = st.selectbox(
			"Select Process Type",
			["AR(1)", "AR(2)", "ARMA(1,1)", "Seasonal AR(4)"]
		)

		if process_type == "AR(1)":
			ar_coef = st.slider("AR(1) Coefficient", min_value=-0.9, max_value=0.9, value=0.7, step=0.1)
			true_order = 1

		elif process_type == "AR(2)":
			ar1_coef = st.slider("AR(1) Coefficient", min_value=-0.9, max_value=0.9, value=0.5, step=0.1)
			ar2_coef = st.slider("AR(2) Coefficient", min_value=-0.9, max_value=0.9, value=0.3, step=0.1)
			true_order = 2

		elif process_type == "ARMA(1,1)":
			ar_coef = st.slider("AR Coefficient", min_value=-0.9, max_value=0.9, value=0.7, step=0.1)
			ma_coef = st.slider("MA Coefficient", min_value=-0.9, max_value=0.9, value=0.4, step=0.1)
			true_order = 1  # For AR component

		elif process_type == "Seasonal AR(4)":
			ar_coef = st.slider("Seasonal AR Coefficient", min_value=-0.9, max_value=0.9, value=0.7, step=0.1)
			true_order = 4

		sample_size = st.slider("Sample Size", min_value=100, max_value=1000, value=200, step=100)
		max_lag = st.slider("Maximum Lag to Consider", min_value=1, max_value=20, value=10, step=1)

	# Generate time series
	np.random.seed(42)

	if process_type == "AR(1)":
		# AR(1): φ(B) = 1 – φ·B
		ar = np.r_[1, -ar_coef]  # [1, -φ]
		ma = np.array([1])  # θ(B)=1
		ar_process = ArmaProcess(ar, ma)
		series = ar_process.generate_sample(sample_size)

	elif process_type == "AR(2)":
		# AR(2): φ(B) = 1 – φ₁B – φ₂B²
		ar = np.r_[1, -ar1_coef, -ar2_coef]
		ma = np.array([1])
		ar_process = ArmaProcess(ar, ma)
		series = ar_process.generate_sample(sample_size)

	elif process_type == "ARMA(1,1)":
		# ARMA(1,1): φ(B)=1–φB,  θ(B)=1+θB
		ar = np.r_[1, -ar_coef]
		ma = np.r_[1, ma_coef]
		arma_process = ArmaProcess(ar, ma)
		series = arma_process.generate_sample(sample_size)

	elif process_type == "Seasonal AR(4)":
		# Seasonal AR(4): φ(B)=1 – φ·B⁴
		ar = np.array([1, 0, 0, 0, -ar_coef])
		ma = np.array([1])
		ar_process = ArmaProcess(ar, ma)
		series = ar_process.generate_sample(sample_size)

	# Calculate information criteria for different lag lengths
	aic_values = []
	bic_values = []
	hqic_values = []

	for p in range(1, max_lag + 1):
		model = AutoReg(series, lags=p)
		results = model.fit()
		aic_values.append(results.aic)
		bic_values.append(results.bic)
		hqic_values.append(results.hqic)

	# Plot information criteria
	with col2:
		fig, ax = plt.subplots(figsize=(10, 6))
		lags = list(range(1, max_lag + 1))

		ax.plot(lags, aic_values, 'o-', label='AIC')
		ax.plot(lags, bic_values, 's-', label='BIC')
		ax.plot(lags, hqic_values, '^-', label='HQIC')

		# Add vertical line at true order
		ax.axvline(x=true_order, color='r', linestyle='--', alpha=0.7, label=f'True Order = {true_order}')

		ax.set_xlabel('Lag Order')
		ax.set_ylabel('Information Criterion Value')
		ax.set_title('Information Criteria by Lag Order')
		ax.legend()
		ax.grid(True, alpha=0.3)

		plt.tight_layout()
		st.pyplot(fig)

	# Determine optimal lags
	optimal_aic = np.argmin(aic_values) + 1
	optimal_bic = np.argmin(bic_values) + 1
	optimal_hqic = np.argmin(hqic_values) + 1

	st.markdown("""
    <div class="result-box">
        <h3 class="subsection-header">Optimal Lag Selection Results</h3>
        <ul>
            <li><b>AIC selects:</b> {} lag(s)</li>
            <li><b>BIC selects:</b> {} lag(s)</li>
            <li><b>HQIC selects:</b> {} lag(s)</li>
            <li><b>True order:</b> {} lag(s)</li>
        </ul>
    </div>
    """.format(optimal_aic, optimal_bic, optimal_hqic, true_order), unsafe_allow_html=True)

	# Visualize model fit with selected lags
	selected_criterion = st.selectbox(
		"Select Criterion for Model Fitting",
		["AIC", "BIC", "HQIC", "True Order"]
	)

	if selected_criterion == "AIC":
		selected_lag = optimal_aic
	elif selected_criterion == "BIC":
		selected_lag = optimal_bic
	elif selected_criterion == "HQIC":
		selected_lag = optimal_hqic
	else:
		selected_lag = true_order

	# Fit model with selected lag
	model = AutoReg(series, lags=selected_lag)
	results = model.fit()

	# Plot original series and fitted values
	fig, ax = plt.subplots(figsize=(10, 6))

	ax.plot(series, label='Original Series')
	ax.plot(results.fittedvalues, 'r', label=f'Fitted (Lag={selected_lag})')

	ax.set_title(f'Series vs. Fitted Values with {selected_lag} Lag(s)')
	ax.legend()
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	st.pyplot(fig)

	# Residual diagnostics
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))

	# Residual time plot
	axes[0, 0].plot(results.resid)
	axes[0, 0].set_title('Residuals')
	axes[0, 0].grid(True, alpha=0.3)

	# Residual histogram
	axes[0, 1].hist(results.resid, bins=20, density=True, alpha=0.7)
	axes[0, 1].set_title('Residual Distribution')

	# Residual ACF
	sm.graphics.tsa.plot_acf(results.resid, lags=20, ax=axes[1, 0], alpha=0.5)
	axes[1, 0].set_title('Residual ACF')

	# Residual PACF
	sm.graphics.tsa.plot_pacf(results.resid, lags=20, ax=axes[1, 1], alpha=0.5)
	axes[1, 1].set_title('Residual PACF')

	plt.tight_layout()
	st.pyplot(fig)

	# Model coefficients and diagnostics
	st.markdown('<h3 class="subsection-header">Model Summary</h3>', unsafe_allow_html=True)

	# Extract and display model coefficients
	coefs = results.params

	st.markdown("### AR Coefficients:")
	coef_data = []
	for i in range(len(coefs)):
		if i == 0:
			coef_data.append(["Constant", f"{coefs[i]:.4f}"])
		else:
			coef_data.append([f"AR({i})", f"{coefs[i]:.4f}"])

	coef_df = pd.DataFrame(coef_data, columns=["Parameter", "Coefficient"])
	st.dataframe(coef_df)

	# Model fit statistics
	st.markdown("### Model Fit Statistics:")
	fit_stats = [
		["Log Likelihood", f"{results.llf:.4f}"],
		["AIC", f"{results.aic:.4f}"],
		["BIC", f"{results.bic:.4f}"],
		["HQIC", f"{results.hqic:.4f}"]
	]
	fit_df = pd.DataFrame(fit_stats, columns=["Statistic", "Value"])
	st.dataframe(fit_df)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Characteristics of Information Criteria</h3>
        <table>
            <tr>
                <th>Criterion</th>
                <th>Penalty Term</th>
                <th>Tendency</th>
                <th>Best For</th>
            </tr>
            <tr>
                <td>AIC</td>
                <td>$2k$</td>
                <td>Often selects larger models</td>
                <td>Forecasting when true model is complex</td>
            </tr>
            <tr>
                <td>BIC</td>
                <td>$k\ln(n)$</td>
                <td>More parsimonious models</td>
                <td>Finding the true model</td>
            </tr>
            <tr>
                <td>HQIC</td>
                <td>$2k\ln(\ln(n))$</td>
                <td>Intermediate between AIC and BIC</td>
                <td>Balance between parsimony and fit</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <h3 class="subsection-header">Alternative Approaches to Lag Selection</h3>
        <ul>
            <li><b>Data frequency heuristics:</b> Use 4 lags for quarterly data, 12 for monthly, etc.</li>
            <li><b>Sequential testing:</b> Start with a large model and test significance of highest lag</li>
            <li><b>Rule of thumb:</b> $\text{Int}(T^{1/4})$ where $T$ is sample size</li>
            <li><b>Cross-validation:</b> Select model with best out-of-sample performance</li>
            <li><b>Residual diagnostics:</b> Select the minimum lag that yields white noise residuals</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def interactive_adf():
	st.markdown('<h2 class="section-header">Interactive ADF Test</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <p>This section allows you to test for stationarity and unit roots using your own data or the provided examples.</p>
    </div>
    """, unsafe_allow_html=True)

	# Data source selection
	data_source = st.radio(
		"Select Data Source",
		["Use Example Data", "Upload Your Own Data"]
	)

	if data_source == "Use Example Data":
		example_dataset = st.selectbox(
			"Select Example Dataset",
			["Random Walk", "Trend Stationary", "Seasonal", "Financial Returns", "GDP Growth"]
		)

		# Generate example data
		np.random.seed(42)
		t = 200
		time = np.arange(t)

		if example_dataset == "Random Walk":
			data = np.zeros(t)
			data[0] = 0
			for i in range(1, t):
				data[i] = data[i - 1] + np.random.normal(0, 1)

			data_description = "Random walk without drift, expected I(1)"

		elif example_dataset == "Trend Stationary":
			trend = 0.05 * time
			noise = np.zeros(t)
			for i in range(t):
				if i == 0:
					noise[i] = np.random.normal(0, 1)
				else:
					noise[i] = 0.7 * noise[i - 1] + np.random.normal(0, 1)
			data = trend + noise

			data_description = "Linear trend with stationary AR(1) noise, expected trend stationary"

		elif example_dataset == "Seasonal":
			season = 5 * np.sin(2 * np.pi * time / 12)
			trend = 0.03 * time
			noise = np.random.normal(0, 1, t)
			data = trend + season + noise

			data_description = "Seasonal pattern with trend, needs seasonal differencing"

		elif example_dataset == "Financial Returns":
			# Simulate somewhat realistic financial returns with volatility clustering
			returns = np.zeros(t)
			volatility = np.ones(t)

			for i in range(1, t):
				volatility[i] = 0.9 * volatility[i - 1] + 0.1 * abs(returns[i - 1])
				returns[i] = np.random.normal(0, volatility[i])

			data = returns
			data_description = "Simulated financial returns, expected I(0) with ARCH effects"

		elif example_dataset == "GDP Growth":
			# Simulate somewhat realistic GDP growth rates with persistence
			growth = np.zeros(t)
			growth[0] = 2 + np.random.normal(0, 0.5)

			for i in range(1, t):
				growth[i] = 0.8 * growth[i - 1] + (1 - 0.8) * 2 + np.random.normal(0, 0.5)

			data = growth
			data_description = "Simulated GDP growth rates, expected I(0) with high persistence"

		# Create a DataFrame for the data
		df = pd.DataFrame({
			'Time': time,
			'Value': data
		})

	else:  # Upload data
		uploaded_file = st.file_uploader("Upload CSV file with time series data", type=["csv"])

		if uploaded_file is not None:
			try:
				df = pd.read_csv(uploaded_file)

				# Let user select the column to analyze
				value_column = st.selectbox(
					"Select the column containing your time series values",
					df.columns
				)

				data = df[value_column].values
				time = np.arange(len(data))
				data_description = "User uploaded data"

			except Exception as e:
				st.error(f"Error reading the uploaded file: {e}")
				return
		else:
			st.info("Please upload a CSV file to continue.")
			return

	# Display the data
	st.markdown(f"<h3 class='subsection-header'>Data: {data_description}</h3>", unsafe_allow_html=True)

	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(time, data)
	ax.set_title("Time Series Plot", fontsize=14)
	ax.grid(True, alpha=0.3)
	plt.tight_layout()
	st.pyplot(fig)

	# ADF Test configuration
	st.markdown('<h3 class="subsection-header">Configure Augmented Dickey-Fuller Test</h3>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		regression_type = st.radio(
			"Regression Type",
			["No Constant, No Trend", "With Constant, No Trend", "With Constant and Trend"]
		)

		if regression_type == "No Constant, No Trend":
			regression = 'n'
		elif regression_type == "With Constant, No Trend":
			regression = 'c'
		else:
			regression = 'ct'

	with col2:
		lag_selection = st.radio(
			"Lag Selection Method",
			["Automatic (AIC)", "Automatic (BIC)", "Automatic (t-test)", "Manual"]
		)

		if lag_selection == "Automatic (AIC)":
			autolag = 'AIC'
			maxlag = None
		elif lag_selection == "Automatic (BIC)":
			autolag = 'BIC'
			maxlag = None
		elif lag_selection == "Automatic (t-test)":
			autolag = 't-stat'
			maxlag = None
		else:
			autolag = None
			maxlag = st.slider("Maximum Lag", min_value=0, max_value=30, value=10)

	# Run ADF test
	if autolag is None:
		adf_result = adfuller(data, regression=regression, maxlag=maxlag)
	else:
		adf_result = adfuller(data, regression=regression, autolag=autolag)

	# Display results
	st.markdown('<h3 class="subsection-header">ADF Test Results</h3>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown(f"**Test Statistic:** {adf_result[0]:.4f}")
		st.markdown(f"**p-value:** {adf_result[1]:.4f}")
		st.markdown(f"**Lags Used:** {adf_result[2]}")
		st.markdown(f"**Number of Observations:** {adf_result[3]}")

		if adf_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ Reject null hypothesis - Series is stationary</div>',
						unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Fail to reject null hypothesis - Series has a unit root</div>',
						unsafe_allow_html=True)

	with col2:
		st.markdown("**Critical Values:**")
		for key, value in adf_result[4].items():
			st.markdown(f"  - {key}: {value:.4f}")

	# Additional visualizations
	st.markdown('<h3 class="subsection-header">Additional Analysis</h3>', unsafe_allow_html=True)

	tab1, tab2, tab3 = st.tabs(["ACF/PACF", "Differencing", "Transformation Suggestions"])

	with tab1:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

		# ACF plot
		sm.graphics.tsa.plot_acf(data, lags=min(30, len(data) // 3), ax=ax1, alpha=0.5)
		ax1.set_title("Autocorrelation Function (ACF)")

		# PACF plot
		sm.graphics.tsa.plot_pacf(data, lags=min(30, len(data) // 3), ax=ax2, alpha=0.5)
		ax2.set_title("Partial Autocorrelation Function (PACF)")

		plt.tight_layout()
		st.pyplot(fig)

		# Interpretation
		if np.abs(sm.tsa.acf(data, nlags=10)[1:]).max() > 0.5:
			st.markdown(
				"**ACF Interpretation:** Strong autocorrelation present, suggesting possible non-stationarity or high persistence.")
		else:
			st.markdown("**ACF Interpretation:** Autocorrelation diminishes quickly, suggesting possible stationarity.")

	with tab2:
		# Calculate differenced series
		if len(data) > 1:
			diff1 = np.diff(data)

			# Run ADF test on differenced series
			adf_diff_result = adfuller(diff1, regression=regression)

			fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

			# Original series
			ax1.plot(time, data)
			ax1.set_title("Original Series", fontsize=14)
			ax1.grid(True, alpha=0.3)

			# Differenced series
			ax2.plot(time[1:], diff1)
			ax2.set_title("First Difference", fontsize=14)
			ax2.grid(True, alpha=0.3)

			plt.tight_layout()
			st.pyplot(fig)

			st.markdown("**ADF Test on Differenced Series:**")
			st.markdown(f"Test Statistic: {adf_diff_result[0]:.4f}")
			st.markdown(f"p-value: {adf_diff_result[1]:.4f}")

			if adf_diff_result[1] < 0.05:
				st.markdown(
					'<div class="result-box">✅ Differenced series is stationary, original series is likely I(1)</div>',
					unsafe_allow_html=True)
			else:
				st.markdown(
					'<div class="warning-box">❌ Differenced series still non-stationary, may be I(2) or higher</div>',
					unsafe_allow_html=True)

	with tab3:
		st.markdown("### Transformation Suggestions")

		# Check for obvious trend
		trend_coef = np.polyfit(time, data, 1)[0]
		has_trend = abs(trend_coef) > 0.01  # Arbitrary threshold

		# Check for variance changes
		first_half_var = np.var(data[:len(data) // 2])
		second_half_var = np.var(data[len(data) // 2:])
		variance_ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)
		unstable_variance = variance_ratio > 1.5  # Arbitrary threshold

		# Check for seasonality (simplistic)
		acf_vals = sm.tsa.acf(data, nlags=min(24, len(data) // 3))
		potential_seasonality = False
		for lag in range(4, min(13, len(acf_vals))):
			if acf_vals[lag] > 0.3:  # Arbitrary threshold
				potential_seasonality = True
				seasonal_period = lag
				break

		transformations = []

		if not adf_result[1] < 0.05:  # Non-stationary
			transformations.append(("First differencing", "For stochastic trends or random walks"))

		if has_trend:
			transformations.append(("Detrending", "For deterministic trends"))

		if unstable_variance:
			transformations.append(("Logarithmic transformation", "For stabilizing increasing variance"))
			transformations.append(("Square root transformation", "For milder variance stabilization"))

		if potential_seasonality:
			transformations.append((f"Seasonal differencing (lag={seasonal_period})", "For removing seasonal patterns"))

		if transformations:
			st.markdown("Based on the data characteristics, consider the following transformations:")
			for transform, reason in transformations:
				st.markdown(f"- **{transform}**: {reason}")
		else:
			st.markdown("No obvious transformations needed, the series may already be stationary.")

		# Integration order suggestion
		if adf_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ The series appears to be I(0) - already stationary</div>',
						unsafe_allow_html=True)
		elif len(data) > 1 and adf_diff_result[1] < 0.05:
			st.markdown(
				'<div class="result-box">✅ The series appears to be I(1) - stationary after first differencing</div>',
				unsafe_allow_html=True)
		else:
			st.markdown(
				'<div class="warning-box">❌ The series may be I(2) or higher - consider second differencing</div>',
				unsafe_allow_html=True)


def generate_series():
	st.markdown('<h2 class="section-header">Generate Your Own Time Series</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="concept-box">
        <p>This section allows you to generate your own time series with specific properties to explore stationarity concepts.</p>
    </div>
    """, unsafe_allow_html=True)

	# Model selection
	model_type = st.selectbox(
		"Select Time Series Model",
		["AR(p) Process", "MA(q) Process", "ARMA(p,q) Process", "ARIMA(p,d,q) Process", "Trend + Noise", "Seasonal",
		 "Custom"]
	)

	# Default parameters
	sample_size = st.slider("Sample Size", min_value=50, max_value=1000, value=200, step=50)

	# Model-specific parameters
	if model_type == "AR(p) Process":
		p = st.slider("AR Order (p)", min_value=1, max_value=5, value=1)
		ar_params = []

		for i in range(p):
			ar_coef = st.slider(f"AR({i + 1}) Coefficient", min_value=-0.99, max_value=0.99,
								value=0.7 if i == 0 else 0.0, step=0.05)
			ar_params.append(ar_coef)

		# Create AR process
		ar = np.r_[1, -np.array(ar_params)]
		ma = np.array([1])
		arma_process = ArmaProcess(ar, ma)

		if arma_process.isstationary:
			stationary_status = "✅ Process is stationary"
		else:
			stationary_status = "❌ Process is non-stationary"

		st.markdown(f"**Stationarity Status:** {stationary_status}")

		# Generate data
		np.random.seed(42)
		data = arma_process.generate_sample(sample_size)

		expected_order = "I(0)" if arma_process.isstationary else "I(1) or higher"

	elif model_type == "MA(q) Process":
		q = st.slider("MA Order (q)", min_value=1, max_value=5, value=1)
		ma_params = []

		for i in range(q):
			ma_coef = st.slider(f"MA({i + 1}) Coefficient", min_value=-0.99, max_value=0.99,
								value=0.5 if i == 0 else 0.0, step=0.05)
			ma_params.append(ma_coef)

		# Create MA process
		ar = np.array([1])
		ma = np.r_[1, np.array(ma_params)]
		arma_process = ArmaProcess(ar, ma)

		# MA processes are always stationary
		st.markdown("**Stationarity Status:** ✅ MA processes are always stationary")

		# Generate data
		np.random.seed(42)
		data = arma_process.generate_sample(sample_size)

		expected_order = "I(0)"

	elif model_type == "ARMA(p,q) Process":
		p = st.slider("AR Order (p)", min_value=0, max_value=5, value=1)
		q = st.slider("MA Order (q)", min_value=0, max_value=5, value=1)

		ar_params = []
		for i in range(p):
			ar_coef = st.slider(f"AR({i + 1}) Coefficient", min_value=-0.99, max_value=0.99,
								value=0.7 if i == 0 else 0.0, step=0.05)
			ar_params.append(ar_coef)

		ma_params = []
		for i in range(q):
			ma_coef = st.slider(f"MA({i + 1}) Coefficient", min_value=-0.99, max_value=0.99,
								value=0.5 if i == 0 else 0.0, step=0.05)
			ma_params.append(ma_coef)

		# Create ARMA process
		ar = np.r_[1, -np.array(ar_params)] if p > 0 else np.array([1])
		ma = np.r_[1, np.array(ma_params)] if q > 0 else np.array([1])
		arma_process = ArmaProcess(ar, ma)

		if arma_process.isstationary:
			stationary_status = "✅ Process is stationary"
		else:
			stationary_status = "❌ Process is non-stationary"

		st.markdown(f"**Stationarity Status:** {stationary_status}")

		# Generate data
		np.random.seed(42)
		data = arma_process.generate_sample(sample_size)

		expected_order = "I(0)" if arma_process.isstationary else "I(1) or higher"

	elif model_type == "ARIMA(p,d,q) Process":
		p = st.slider("AR Order (p)", min_value=0, max_value=5, value=1)
		d = st.slider("Differencing Order (d)", min_value=0, max_value=2, value=1)
		q = st.slider("MA Order (q)", min_value=0, max_value=5, value=1)

		ar_params = []
		for i in range(p):
			ar_coef = st.slider(f"AR({i + 1}) Coefficient", min_value=-0.99, max_value=0.99,
								value=0.7 if i == 0 else 0.0, step=0.05)
			ar_params.append(ar_coef)

		ma_params = []
		for i in range(q):
			ma_coef = st.slider(f"MA({i + 1}) Coefficient", min_value=-0.99, max_value=0.99,
								value=0.5 if i == 0 else 0.0, step=0.05)
			ma_params.append(ma_coef)

		# Create ARMA process
		ar = np.r_[1, -np.array(ar_params)] if p > 0 else np.array([1])
		ma = np.r_[1, np.array(ma_params)] if q > 0 else np.array([1])
		arma_process = ArmaProcess(ar, ma)

		# Generate stationary ARMA data
		np.random.seed(42)
		stationary_data = arma_process.generate_sample(sample_size)

		# Apply integration (summing)
		data = stationary_data.copy()
		for _ in range(d):
			data = np.cumsum(data)

		expected_order = f"I({d})"

		st.markdown(f"**Expected Integration Order:** {expected_order}")

	elif model_type == "Trend + Noise":
		trend_type = st.selectbox(
			"Trend Type",
			["Linear", "Quadratic", "Exponential"]
		)

		trend_coef = st.slider("Trend Coefficient", min_value=-0.5, max_value=0.5, value=0.05, step=0.01)
		noise_type = st.selectbox(
			"Noise Type",
			["White Noise", "AR(1) Noise"]
		)

		if noise_type == "AR(1) Noise":
			ar_coef = st.slider("AR(1) Coefficient", min_value=-0.9, max_value=0.9, value=0.7, step=0.1)

		# Generate time and trend
		time = np.arange(sample_size)

		if trend_type == "Linear":
			trend = trend_coef * time
		elif trend_type == "Quadratic":
			trend = trend_coef * time ** 2
		else:  # Exponential
			trend = np.exp(trend_coef * time)

		# Generate noise
		if noise_type == "White Noise":
			noise = np.random.normal(0, 1, sample_size)
		else:  # AR(1) Noise
			noise = np.zeros(sample_size)
			noise[0] = np.random.normal(0, 1)
			for i in range(1, sample_size):
				noise[i] = ar_coef * noise[i - 1] + np.random.normal(0, 1)

		# Combine trend and noise
		data = trend + noise

		expected_order = "Trend Stationary" if noise_type == "AR(1) Noise" and abs(
			ar_coef) < 1 else "Depends on detrending"

	elif model_type == "Seasonal":
		period = st.slider("Seasonal Period", min_value=2, max_value=24, value=12)
		seasonal_amplitude = st.slider("Seasonal Amplitude", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
		trend_coef = st.slider("Trend Coefficient", min_value=-0.5, max_value=0.5, value=0.02, step=0.01)

		# Generate seasonal component
		time = np.arange(sample_size)
		seasonal = seasonal_amplitude * np.sin(2 * np.pi * time / period)
		trend = trend_coef * time
		noise = np.random.normal(0, 0.5, sample_size)

		data = seasonal + trend + noise

		expected_order = "Seasonally stationary after detrending and seasonal adjustment"

	else:  # Custom
		st.markdown("### Custom Time Series Components")

		include_trend = st.checkbox("Include Trend", value=True)
		trend_coef = 0
		if include_trend:
			trend_coef = st.slider("Trend Coefficient", min_value=-0.5, max_value=0.5, value=0.05, step=0.01)

		include_seasonal = st.checkbox("Include Seasonality", value=True)
		period = 12
		seasonal_amplitude = 0
		if include_seasonal:
			period = st.slider("Seasonal Period", min_value=2, max_value=24, value=12)
			seasonal_amplitude = st.slider("Seasonal Amplitude", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

		include_ar = st.checkbox("Include AR Component", value=True)
		ar_coef = 0
		if include_ar:
			ar_coef = st.slider("AR(1) Coefficient", min_value=-0.99, max_value=0.99, value=0.7, step=0.05)

		include_unit_root = st.checkbox("Include Unit Root (Random Walk)", value=False)

		# Generate time
		time = np.arange(sample_size)

		# Generate components
		trend = trend_coef * time if include_trend else np.zeros(sample_size)
		seasonal = seasonal_amplitude * np.sin(2 * np.pi * time / period) if include_seasonal else np.zeros(sample_size)

		# Generate AR component
		ar_component = np.zeros(sample_size)
		if include_ar:
			ar_component[0] = np.random.normal(0, 1)
			for i in range(1, sample_size):
				ar_component[i] = ar_coef * ar_component[i - 1] + np.random.normal(0, 1)

		# Generate random walk component
		rw_component = np.zeros(sample_size)
		if include_unit_root:
			for i in range(1, sample_size):
				rw_component[i] = rw_component[i - 1] + np.random.normal(0, 1)

		# Combine all components
		data = trend + seasonal + ar_component + rw_component

		# Determine expected order
		if include_unit_root:
			expected_order = "I(1) due to random walk component"
		elif include_trend or include_seasonal:
			expected_order = "Stationary after appropriate transformations"
		elif include_ar and abs(ar_coef) < 1:
			expected_order = "I(0) - stationary AR process"
		else:
			expected_order = "I(0) - stationary"

	# Plot the generated series
	time = np.arange(sample_size)

	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(time, data)
	ax.set_title(f"Generated {model_type}", fontsize=14)
	ax.set_xlabel("Time")
	ax.set_ylabel("Value")
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	st.pyplot(fig)

	# Display ACF and PACF
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

	sm.graphics.tsa.plot_acf(data, lags=min(30, sample_size // 3), ax=ax1, alpha=0.5)
	ax1.set_title("ACF")

	sm.graphics.tsa.plot_pacf(data, lags=min(30, sample_size // 3), ax=ax2, alpha=0.5)
	ax2.set_title("PACF")

	plt.tight_layout()
	st.pyplot(fig)

	# Stationarity testing
	st.markdown('<h3 class="subsection-header">Stationarity Testing</h3>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		# ADF test
		adf_result = adfuller(data, regression='c')

		st.markdown("### ADF Test Results")
		st.markdown(f"**Test Statistic:** {adf_result[0]:.4f}")
		st.markdown(f"**p-value:** {adf_result[1]:.4f}")

		if adf_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ Stationary (reject unit root)</div>', unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Non-stationary (fail to reject unit root)</div>',
						unsafe_allow_html=True)

	with col2:
		try:
			# KPSS test
			kpss_result = kpss(data, regression='c')

			st.markdown("### KPSS Test Results")
			st.markdown(f"**Test Statistic:** {kpss_result[0]:.4f}")
			st.markdown(f"**p-value:** {kpss_result[1]:.4f}")

			if kpss_result[1] < 0.05:
				st.markdown('<div class="warning-box">❌ Non-stationary (reject stationarity)</div>',
							unsafe_allow_html=True)
			else:
				st.markdown('<div class="result-box">✅ Stationary (fail to reject stationarity)</div>',
							unsafe_allow_html=True)
		except:
			st.markdown("KPSS test could not be calculated for this series.")

	# First differences
	if len(data) > 1:
		diff1 = np.diff(data)

		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

		ax1.plot(time, data)
		ax1.set_title("Original Series", fontsize=14)
		ax1.grid(True, alpha=0.3)

		ax2.plot(time[1:], diff1)
		ax2.set_title("First Difference", fontsize=14)
		ax2.grid(True, alpha=0.3)

		plt.tight_layout()
		st.pyplot(fig)

		# ADF test on differenced series
		adf_diff_result = adfuller(diff1, regression='c')

		st.markdown("### ADF Test on First Difference")
		st.markdown(f"**Test Statistic:** {adf_diff_result[0]:.4f}")
		st.markdown(f"**p-value:** {adf_diff_result[1]:.4f}")

		if adf_diff_result[1] < 0.05:
			st.markdown('<div class="result-box">✅ Differenced series is stationary</div>', unsafe_allow_html=True)
		else:
			st.markdown('<div class="warning-box">❌ Differenced series is still non-stationary</div>',
						unsafe_allow_html=True)

	# Final summary
	st.markdown("""
    <div class="result-box">
        <h3 class="subsection-header">Summary</h3>
        <p><b>Model:</b> {}</p>
        <p><b>Expected Integration Order:</b> {}</p>
        <p><b>ADF Test Conclusion:</b> {}</p>
        <p><b>First Difference Test:</b> {}</p>
    </div>
    """.format(
		model_type,
		expected_order,
		"Stationary" if adf_result[1] < 0.05 else "Non-stationary",
		"Stationary" if len(data) > 1 and adf_diff_result[1] < 0.05 else "Non-stationary or not applicable"
	), unsafe_allow_html=True)

	# Option to download the generated data
	csv = pd.DataFrame({'time': time, 'value': data}).to_csv(index=False)
	b64 = base64.b64encode(csv.encode()).decode()
	href = f'<a href="data:file/csv;base64,{b64}" download="generated_timeseries.csv">Download Generated Time Series (CSV)</a>'
	st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
	main()
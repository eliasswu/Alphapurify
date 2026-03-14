# AlphaPurify: Factor analytics for quants

**AlphaPurify** provides a fully modular, vectorized, and multiprocessing-enabled framework for factor cleaning, evaluation, exposure decomposition, and portfolio attribution based on factors.

---

## 🚀 Key Features

- ⚡ **High Performance**
  - Nearly fully vectorized architecture powered by Polars
  - Optimized for large-scale cross-sectional panel data
  - Memory-efficient structural safeguards

- 🧩 **Fully Modular Design**
  - Each module can be used independently
  - Seamlessly integrated into custom research pipelines
  - Minimal coupling between components

- 📊 **Comprehensive Factor Research Engine**
  - Cross-sectional IC analysis
  - Horizon autocorrelation
  - Quantile portfolio backtesting
  - Turnover measurement
  - Industry-level attribution
  - Long–short, long-only, and short-only evaluation

- 🧪 **Advanced Factor Cleaning Toolkit**
  - 40+ preprocessing techniques
  - Robust winsorization
  - Regression-based neutralization
  - Polynomial & robust regression options
  - Advanced standardization methods

- 📈 **Exposure & Return Attribution**
  - Systematic exposure decomposition
  - Residual alpha estimation
  - Cumulative attribution curves
  - Interactive Plotly visualizations

- 🕒 **Frequency-Agnostic**
  - Supports intraday, daily, weekly, and high-frequency datasets
  - No structural modifications required

- 🛡 **Look-Ahead Bias Protection**
  - Forward return construction safeguards
  - Rebalancing alignment protection
  - Parameter-level anti-leakage controls

---

## 📦 Installation

```bash
pip install alphapurify

## 📊 Example Workflow

from alphapurify import AlphaPurifier, FactorAnalyzer

# Load your DataFrame
df = ...

# Clean factor
cleaned = (
    AlphaPurifier(df, factor_col="alpha")
    .winsorize(method="mad")
    .neutralize(neutralizer_cols=["size", "industry"])
    .standardize(method="zscore")
    .to_result()
)

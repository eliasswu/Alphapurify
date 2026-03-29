# Copyright (c) Elias Wu
# Licensed under the MIT License.

"""
AlphaPurify -  High-performance Quantitative Factor Analysis & Purification Toolkit

Main Modules:
-------------
- Database: a parallelized data pipeline class designed to efficiently read and write
  symbol-level factor datasets stored as parquet files.

- AlphaPurifier: a chainable factor-cleaning framework offering 40+ preprocessing
  techniques—including winsorization, regression-based neutralization, and advanced
  standardization—for systematic alpha refinement.

- FactorAnalyzer: a fully vectorized, multiprocessing-powered factor research engine
  for long–short, long-only, and short-only portfolio evaluation.
  It performs cross-sectional IC analysis, horizon autocorrelation, quantile portfolio
  backtesting, turnover measurement, and industry-level attribution—while leveraging
  Polars vectorization and parallel computation for high-performance research workflows.
  The framework also provides interactive Plotly-based visualizations for return curves,
  IC dynamics, and rolling statistics, enabling comprehensive and intuitive factor
  diagnostics.

- Exposures: a factor exposure and return attribution engine for long–short,
  long-only, and short-only portfolios.
  The framework decomposes portfolio returns into systematic exposure contributions
  and residual Alpha, computes cumulative attribution curves, and provides interactive
  Plotly visualizations for exposure and performance diagnostics.


Main Features:
-------------
- Fast and highly efficient: Built on a nearly fully vectorized architecture
  (Polars-based), ensuring high-performance computation even for large-scale datasets.

- Extremely easy to use: In most cases, a single well-structured DataFrame is
  sufficient to run the entire workflow from preprocessing to analysis and
  visualization.

- Extensive preprocessing toolkit: Includes a large collection of robust
  winsorization, neutralization, and standardization methods (40+ techniques)
  for institutional-grade factor cleaning.
"""
__version__ = "0.1.8"

from .AlphaPurifier import AlphaPurifier
from .Database import DataBase
from .Exposures import PortfolioExposures, PureExposures
from .FactorAnalyzer import FactorAnalyzer
from .APr_utils import *


# ============================================================
#  Class FactorAnalyzer Usage Example
# ============================================================
# This section demonstrates how to use FactorAnalyzer
# ============================================================


# ============================================================
# 1. You need to have a Dataframe
# ============================================================
# Load your Dataframe:
#
#          datetime       symbol    open   close      volume    alpha 
# 0       2010-01-06  000001.SZ  834.81  822.25   412143.13  12.929126   
# 617     2010-01-07  000001.SZ  822.25  813.27   355336.85  12.780821   
# 1235    2010-01-08  000001.SZ  807.88  811.48   288543.06  12.572600   
# 1853    2010-01-11  000001.SZ  843.79  811.48   442846.02  13.000977   
# 2471    2010-01-12  000001.SZ  810.40  806.09   591795.91  13.290917   
# ...            ...        ...     ...     ...         ...        ...   
# 2893923 2024-12-25  688981.SH   96.48   97.99   985329.07  13.800731   
# 2894795 2024-12-26  688981.SH   98.00   96.73   711045.86  13.474492   
# 2895667 2024-12-27  688981.SH   96.78   97.51  1144716.30  13.950667   
# 2896539 2024-12-30  688981.SH   96.60   99.29   906573.04  13.717427   
# 2897411 2024-12-31  688981.SH   99.00   94.62   878237.97  13.685673   
#
# Your DataFrame must include a time column ("datetime"), an asset identifier column ("symbol"), a price column ("close" or "open"), 
# and your factor column ("alpha") to ensure proper usage.
#
# ============================================================


# ============================================================
# 2. Initialize FactorAnalyzer: These are all the parameters!
# ============================================================
# from alphapurify import FactorAnalyzer
# from alphapurify.FactorAnalyzer import AnalysisConfig, ResearchConfig
# 
# FactorAnalyzer = FactorAnalyzer(
#     base_df=your_df,
#     trade_date_col='datetime',
#     symbol_col='symbol',
#     price_col='close',
#     factor_name='your_factor',
#    
#     research_cfg=ResearchConfig(
#         rebalance_periods = (1,5,10),
#         return_rolling_period = 20,
#         return_horizons = (1,5,10),
#         horizon_rolling_period = 20,
#         bins = 5,
#         fac_shift = None,
#         base_rate = 0.02,
#         overnight = "on"
#     ),
#    
#     analysis_cfg=AnalysisConfig(
#         rank_ic = True,
#         log_scale = True,
#         agg_freq = None,
#         group_by = None,
#         max_workers = -1
#     ))
#                                        
#
# Parameter Explanation:
#
# base_df:
#     Input dataset containing price and factor data.
#     Supports pandas.DataFrame or polars.DataFrame.
#
# trade_date_col:
#     Column name representing the timestamp of each observation.
#
# symbol_col:
#     Column name representing the asset identifier (e.g., stock code).
#
# price_col:
#     Column name used to compute forward returns.
#
# factor_name:
#     Column name of the factor to be evaluated.
#
# ------------------------------------------------------------
# research_cfg (ResearchConfig): Controls how the factor is tested
# ------------------------------------------------------------
#
# rebalance_periods:
#     Tuple of rebalancing intervals (in number of bars).
#     Example: (1, 5, 10) means rebalance every 1, 5, and 10 periods.
#
# return_rolling_period:
#     Window size for smoothing quantile returns (rolling mean).
#
# return_horizons:
#     Forecast horizons (in periods) used for IC / Rank IC calculation.
#
# horizon_rolling_period:
#     Rolling window size for smoothing IC time series.
#
# bins:
#     Number of quantile groups for portfolio sorting.
#     Must be >= 3. Common choice: 5 (quintiles) or 10 (deciles).
#
# fac_shift:
#     Number of periods to shift the factor forward.
#     Used to avoid look-ahead bias. Example: fac_shift=1.
#
# base_rate:
#     Risk-free rate used in performance metrics (e.g., Sharpe ratio).
#
# overnight:
#     Controls handling of overnight returns:
#     - "on"   : include all data
#     - "off"  : exclude overnight returns
#     - "only" : only use overnight returns
#
# ------------------------------------------------------------
# analysis_cfg (AnalysisConfig): Controls analysis behavior
# ------------------------------------------------------------
#
# rank_ic:
#     If True, compute Rank IC (Spearman correlation).
#     If False, compute normal IC (Pearson correlation).
#
# log_scale:
#     If True, cumulative returns are transformed using log scale.
#
# agg_freq:
#     Aggregation frequency for IC and returns.
#     Example: "1d", "1w", "1m".
#     If None, frequency is inferred automatically.
#
# group_by:
#     Dictionary mapping symbol -> group (e.g., industry classification).
#     Enables grouped IC analysis and industry contribution analysis.
#
# max_workers:
#     Number of parallel workers for computation.
#     -1 means using all available CPU cores.
#
# ============================================================


# ============================================================
# 3.Run backtesting and create reports
# ============================================================
#
# FactorAnalyzer.run()
# FactorAnalyzer.create_long_return_sheet(staticPlot:bool=False, return_fig:bool=False)
# FactorAnalyzer.create_long_short_return_sheet(staticPlot:bool=False, return_fig:bool=False)
# FactorAnalyzer.create_short_return_sheet(staticPlot:bool=False, return_fig:bool=False)
# FactorAnalyzer.create_single_fac_ic_sheet(staticPlot:bool=False, return_fig:bool=False)
# 
# Parameter Explanation & Output Reports:
#
# staticPlot:
#     If True, renders the figure as a static (non-interactive) plot.
#     Useful for exporting or embedding in reports where interaction is not needed.
#
# return_fig:
#     If True, returns the Plotly figure object instead of only displaying it.
#     This allows further customization, saving, or integration into dashboards.
#
# ------------------------------------------------------------
# run():
# ------------------------------------------------------------
# Must be called before generating any report.
#
# ------------------------------------------------------------
# create_long_return_sheet():
# ------------------------------------------------------------
# Generates a comprehensive report for long-only (top quantile) performance:
# - Rolling mean returns for top / middle / bottom quantiles
# - Cumulative returns of each quantile
# - (Optional) Industry-level contribution analysis
# - Turnover analysis for top and bottom groups
# - Heatmap of aggregated (monthly or custom frequency) returns
# - Heatmap of performance statistics (Sharpe, Drawdown, etc.)
#
# ------------------------------------------------------------
# create_long_short_return_sheet():
# ------------------------------------------------------------
# Generates a report for long-short strategy performance:
# - Long-short return time series and rolling mean
# - Cumulative long-short returns
# - (Optional) Industry-level long-short contribution
# - Turnover analysis for long-short portfolio
# - Heatmap of aggregated returns
# - Heatmap of long-short performance statistics
#
# ------------------------------------------------------------
# create_short_return_sheet():
# ------------------------------------------------------------
# Generates a report for short-only (bottom quantile) performance:
# - Rolling mean returns (inverted for short perspective)
# - Cumulative short returns
# - (Optional) Industry-level short contribution
# - Turnover analysis
# - Heatmap of aggregated returns
# - Heatmap of short-side performance statistics
#
# ------------------------------------------------------------
# create_single_fac_ic_sheet():
# ------------------------------------------------------------
# Generates a report for factor predictive power (IC analysis):
# - IC / Rank IC time series and rolling mean
# - Cumulative IC
# - (Optional) Industry contribution to IC
# - IC autocorrelation (factor persistence)
# - Q-Q plot for distribution diagnostics
# - Heatmap of aggregated IC values
# - Heatmap of IC statistics (mean, std, t-stat, IR, etc.)
#
# ------------------------------------------------------------
#
# ============================================================


==============================================================
# 4. Cross-sectional Snapshot: Microscopic Alpha Tracer
# ============================================================
# This section demonstrates how to use the white-box diagnostic tracer.
# 
# ============================================================
# 4.1 Execute Microscopic Alpha Tracer: Trace ANY dimension!
# ============================================================
#
# # Ensure you have already invoked `FactorAnalyzer.run()` before tracing.
#
# trace_df = FactorAnalyzer.trace(
#      rebalance_period='W',          # Target rebalancing interval (e.g., 5 or 'W', 'M')
#      date='2012-09-28',             # Target historical trading bar string
#      bins=[1, 5],                   # Selected quantile bins to visualize. None defaults to ALL bins.
#      position='l',                  # Directional perspective: 'l' (Long) or 's' (Short)
#      staticPlot=False,              # True for flattened image, False for interactive Plotly.
#      return_fig=False,              # True to return (fig, df), False to return df only.
#      return_full_df=False           # True to bypass time mask and output un-filtered panel data.
# )
#
# ============================================================
# Parameter Explanation:
# ============================================================
#
# rebalance_period:
#      The target strategy rebalancing interval. Supports either an integer 
#      (number of trading bars) or custom calendar codes (e.g., 'W' for Weekly, 'M' for Monthly).
#
# date:
#      The exact chronological execution target bar string. Acts as the central 
#      node slicing a microscopic micro-panel window ($\pm 1$ Bar) for diagnostic analysis.
#
# bins:
#      List of specific quantile bin coordinates to visualize. 
#      If None, defaults dynamically to all available quantile blocks sorted in descending order.
#
# position:
#      The directional execution perspective modifier. Defaults to 'l'.
#      - 'l': Maps long-leg performance tracking vectors (highest quantile groups match positive alpha).
#      - 's': Maps inverted short-leg metrics (automatically inverts capital allocation weights to `-w`).
#
# staticPlot:
#      Controls rendering interactivity. Defaults to False.
#      - True: Silences reactive UI JavaScript workers and outputs an immutable flattened vector graphic. 
#              Crucial for continuous batch reporting scripts or static compliance documentation.
#      - False: Retains full live Plotly features including interactive vector panning, zooming, and hovering.
#
# return_fig:
#      Controls return container types. Defaults to False.
#      - True: Returns a compiled `(plotly.graph_objects.Figure, pandas.DataFrame)` tuple for custom layouts.
#      - False: Passes back only the sliced, cleaned cross-sectional Pandas DataFrame.
#
# return_full_df:
#      Bypass modifier for time masks. Defaults to False.
#      - False: Automatically filters and slices the data to focus strictly on the target execution window.
#      - True: Disables the filtration mask, passing back the entire unfiltered panel dataset.
#
# ============================================================
# Output Diagnostics Dashboard & Core Layers:
# ============================================================
#
# Executing this pipeline utilizes Plotly's `make_subplots` engine to vertically stack 
# high-density diagnostic layers, breaking down the black-box backtest into actionable views:
#
# ------------------------------------------------------------
# Layer 1: Net Returns Sectional Canvas (Panel of Net Returns - Heatmap)
# ------------------------------------------------------------
# - Y-axis represents quantile groups (Q1 to Qn); X-axis tracks the chronological sub-window.
# - Leverages a "RdYlGn" (Red-Yellow-Green) diverging color scale to map cross-sectional net returns.
# - Each cell embeds explicit high-precision overlays (`texttemplate="%{text:.2%}"`) to pin-point alpha peaks.
#
# ------------------------------------------------------------
# Layer 2: Weight & Return Scatter Cluster (Weights and Returns Scatter)
# ------------------------------------------------------------
# - Y-axis represents individual asset forward returns; X-axis captures prior allocated portfolio weights.
# - Marks precise ticker symbols at cluster centers alongside a $Y=0$ baseline and a median-weight marker.
# - Empowers researchers to instantly assess if high-weight assets drove performance or triggered drawdowns.
#
# ------------------------------------------------------------
# Layer 3: Gross Return Contribution Breakdown (Contributions of Gross Return Bar)
# ------------------------------------------------------------
# - Y-axis logs asset identifiers (Symbols); X-axis computes absolute contribution ($Weight \times Return$).
# - Automatically sorted in descending order to display alpha drivers on top and laggards at the bottom.
# - Isolates exactly which specific components inside the quantile group carried the portfolio.
#
# ------------------------------------------------------------
# Layer 4: Portfolio Rebalancing Dumbbell Curves (Weight Changes)
# ------------------------------------------------------------
# - Automatically triggered when the sliced cross-section overlaps with strategy rebalance dates.
# - Y-axis charts assets; X-axis charts the shifting capital weight allocation.
# - Uses a responsive Dumbbell geometry: Circles represent legacy weights; Diamonds track optimized target targets.
# - Connector lines visualize individual swap paths, tracking dynamic drift corrections and assets turnover.
#
# ============================================================



# ============================================================
# 5. Core Engine Execution & Exposed Intermediate Attributes
# ============================================================
# This section details the execution engine and the structured datasets 
# generated after executing FactorAnalyzer.run().
# ============================================================
# ------------------------------------------------------------
# 5.1 Invoking the Backtesting & IC Analysis Pipeline
# ------------------------------------------------------------
#
# # The run() method acts as the master dispatch interface. It utilizes 
# # an optimized, multi-process parallel architecture backed by PyArrow IPC 
# # memory mapping and Loky workers to process multiple rebalancing periods 
# # and horizons simultaneously without shared-memory overhead.
#
# FactorAnalyzer.run()
#
# ============================================================
# 5.2 Exposed Micro-Panel Dataframes (For Custom Analysis)
# ============================================================
# After execution, the following attributes are bound to your FactorAnalyzer 
# instance. You can access these anytime to build custom visuals or tables.
#
# ------------------------------------------------------------
#  Performance Summary Sheets (Pandas DataFrames)
# ------------------------------------------------------------
#
# self.ls_stats_panel : pd.DataFrame
#      Compiled performance summary statistics across all rebalance periods 
#      for the Long-Short (L-S) strategy portfolio.
#      Metrics include: Annualized Return, Sharpe Ratio, Sortino Ratio, 
#      Calmar Ratio, Maximum Drawdown, Win Rate, Profit/Loss ratio, and PnL.
#
# self.l_stats_panel : pd.DataFrame
#      Compiled performance metrics for the Long-only portfolio (Top Quantile), 
#      processed across all target evaluation intervals.
#
# self.s_stats_panel : pd.DataFrame
#      Compiled performance metrics for the Short-only portfolio (Bottom Quantile), 
#      accounting for transaction frictions and inverted shorting matrices.
#
# self.ic_stats_panel : pd.DataFrame
#      Predictive factor validity diagnostic matrix tracking IC properties 
#      across distinct forecast horizons. 
#      Exposes: Mean IC, IC Standard Deviation, Skewness, Kurtosis, 
#      t-statistic, p-value, and Information Ratio (IR).
#
# ------------------------------------------------------------
# Structural Time Series & Aggregation Panels
# ------------------------------------------------------------
#
# self.returns_dict : dict[any, pd.DataFrame]
#      Key: Rebalance Period. Value: Sliced cross-sectional performance panels.
#      Contains chronological time series for raw, net, and cumulative returns 
#      across all individual quantiles and the long-short spread.
#
# self.ics_dict : dict[any, pd.DataFrame]
#      Key: Horizon Period. Value: Longitudinal predictive validation matrices.
#      Stores raw IC series, Rank IC series, rolling means, and cumulative IC series.
#
# self.ls_monthly_panel / self.l_monthly_panel / self.s_monthly_panel : pd.DataFrame
#      Aggregated return performance records truncated and calculated at the configured 
#      aggregation frequency (e.g., calendar month-end matrices).
#
# self.ic_monthly_panel : pd.DataFrame
#      Aggregated historical factor information storage tracking cumulative 
#      predictive alpha persistence across calendar sub-periods.
#
# ------------------------------------------------------------
# Friction, Risk Attribution, & Heatmap Matrices
# ------------------------------------------------------------
#
# self.ls_turnovers_dict : dict[any, float]
#      Maps rebalancing windows to their average aggregate portfolio turnover rate. 
#      Crucial for identifying factor capacity constraints and decay speeds.
#
# self.agg_dfs_dict : dict[any, pd.DataFrame]
#      Underlying raw trading turnover panels storing granular buy/sell dollar 
#      volumes and multi-tier slippage/tax cost realizations per period.
#
# self.indus_returns_dict : dict[any, pd.DataFrame]
#      Cross-sectional portfolio industry performance tracking maps. Breaks down 
#      the active return contributions across discrete risk-group classifications.
#
# self.ic_indus_contribs_dict : dict[any, pd.DataFrame]
#      Factor information-coefficient decay panels isolated by sector/industry, 
#      allowing researchers to inspect if factor alpha is driven by pure stock 
#      selection or industry bets.
#
# self.heatmap_calendar_dfs_dict / self.heatmap_calendar_s_dfs_dict : dict[any, pd.DataFrame]
#      Pre-formatted pivot tables matching the Year $\times$ Month grid. Stores 
#      the active compound excess returns for long-only and short-only panels.
#
# ============================================================
# 5.3 Quick Code Example: Querying Intermediate Properties
# ============================================================
#
# # Example 1: View the Sharpe and Drawdown summary for Long-Short portfolios
# print(analyzer.ls_stats_panel[["period", "Ann. Sharpe", "Max Drawdown"]])
#
# # Example 2: Pull the raw day-to-day returns panel for a 5-day rebalance interval
# df_5d = analyzer.returns_dict[5]
# print(df_5d[["cum_ret_net_ls", "ret_net_ls"]].tail())
#
# # Example 3: Pull the calendar month excess return grid for the top quantile
# monthly_grid_q5 = analyzer.heatmap_calendar_dfs_dict[5]
# print(monthly_grid_q5.head())
#
# ============================================================


import pandas as pd
import numpy as np
from alphapurify import FactorAnalyzer

def df():
    np.random.seed(42)
    n_stocks = 100
    start_date = "2024-01-01"
    end_date = "2025-12-31"

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    symbols = [f"stock_{i}" for i in range(1, n_stocks + 1)]
    dfs = []

    for sym in symbols:
        n = len(dates)
        drift = np.random.uniform(0.0001, 0.0005)     
        vol = np.random.uniform(0.01, 0.03)            
        
        eps = np.random.randn(n)
        returns = drift + vol * eps
        
        price = 100 * np.exp(np.cumsum(returns))
        
        df_temp = pd.DataFrame({
            "datetime": dates,
            "symbol": sym,
            "close": price,
            "ret": returns
        })
        
        dfs.append(df_temp)

    df = pd.concat(dfs).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    df["future_ret"] = df.groupby("symbol")["ret"].shift(-1)
    noise = np.random.randn(len(df)) * 0.02
    df["factor"] = 0.2 * df["future_ret"] + noise
    df = df.drop(columns=["ret", "future_ret"])
    return df


def test_sheets(df):
    FA = FactorAnalyzer(df,'datetime','symbol','close','factor')
    FA.run()
    res = FA.create_single_fac_full_sheet(return_fig=True)

test_sheets(df())

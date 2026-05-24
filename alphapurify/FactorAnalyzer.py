from datetime import timedelta
import numpy as np
import polars as pl
from joblib import Parallel, delayed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import multiprocessing as mp
import tempfile
import pyarrow as pa
import pandas as pd
import scipy.stats as stats

from dataclasses import dataclass

class ResearchConfig:
    rebalance_periods: tuple[int | str] = ("W","M","Q")
    return_horizons: tuple[int] = (1,5,10)
    horizon_rolling_period: int = 20
    bins: int = 5
    fac_shift: int | None = None
    base_rate: float = 0.02,
    fee_rate: float = 0.0003,
    slippage_rate: float = 0.001,
    tax_rate:float = 0.0,
    tax_rate_direction: str = "sell",
    overnight: str = "on"

@dataclass
class AnalysisConfig:
    rank_ic: bool = True
    log_scale: bool = False
    agg_freq: str | None = None
    group_by: dict | None = None
    max_workers: int = -1

_worker_df = None

class FactorAnalyzer():
    r"""
    FactorAnalyzer

    An enterprise-grade, high-performance factor research and evaluation engine 
    designed for cross-sectional asset pricing statistics and portfolio backtesting. 
    This class orchestrates a complete scientific pipeline for quantitative alpha 
    factor validation, supporting multi-horizon Information Coefficient (IC) diagnostics, 
    quantile portfolio sorting, rigorous turnover/transaction cost accounting, 
    and multi-dimensional risk-return visualization panels.

    To eliminate Python's Global Interpreter Lock (GIL) bottlenecks and minimize memory 
    footprint when processing high-frequency or large-scale historical datasets, the 
    engine utilizes a vectorized Polars core combined with Apache Arrow memory-mapping 
    (`pyarrow.memory_map`) via Joblib multi-processing parallel execution.

    Key Mathematical & Architectural Implementations
    ------------------------------------------------
    1. Look-Ahead Bias Mitigation:
       Supports multi-bar lagging of alpha signals via `fac_shift` to strictly mirror 
       real-world execution.
    
    2. Fractional Grouping & Cumulative Compounding:
       Under procedural rebalancing (`calc_stats_for_period`), rows are partitioned into 
       continuous rebalancing epochs identified by an internal ID 'g'. Portfolio weights 
       are dynamically rolled forward via `forward_fill` inside each epoch to simulate 
       the organic drifting of capital driven by cumulative asset returns (`cum_prod`), 
       avoiding the common backtesting error of assuming constant weights between intervals.
    
    3. Asymmetric Transaction Cost Modeling:
       Integrates buy/sell non-linear frictions by evaluating directional turnover:
       Turnover = sum |w_{t,i} - w_{t^-,i}|
       It incorporates customizable distinct schedules for commissions, slippage, and 
       stamp taxes based on trading direction (`tax_rate_direction` as 'buy', 'sell', or 'both').
    
    4. Advanced Diagnostic Visualization:
       Generates comprehensive diagnostic sheets utilizing Plotly subplots, supporting 
       independent row-level sub-legends, interactive box plots for seasonal alpha 
       stability analysis, empirical Q-Q plots for fat-tail risk mapping, and 
       industry-neutralized performance attributions.

    Parameters
    ----------
    base_df : pd.DataFrame or pl.DataFrame
        The foundational panel data containing asset returns, identifier codes, timestamps, 
        and raw factor scores. Converted internally to a zero-copy or micro-cloned `pl.DataFrame`.
    
    trade_date_col : str
        The column identifier for sequence timestamps (e.g., 'datetime', 'date'). Cast to `pl.Datetime`.
    
    symbol_col : str
        The asset uniqueness identifier (e.g., 'symbol', 'ticker', 'code'). Used as the primary 
        key for cross-sectional alignment and lagging window operations.
    
    price_col : str
        The close price or execution price utilized for forward return metrics computation.
    
    factor_name : str
        The target alpha factor column name to evaluate.
    
    research_cfg : ResearchConfig or dict, optional
        A structured configuration instance containing hyper-parameters for execution, 
        including: rebalance_periods, return_horizons, rolling windows, log_scale flags, 
        overnight filtering types ('off', 'only', 'on'), and quantiles counts (`bins`).
    
    analysis_cfg : AnalysisConfig or dict, optional
        A structured configuration instance governing parallel pooling worker limits (`max_workers`), 
        IC calculation types (Pearson vs. Spearman Rank IC), and temporal aggregation frequencies (`agg_freq`).

    Attributes
    ----------
    base_df : pl.DataFrame
        Internal multi-index panel dataset, optimized, chronologically sorted, and blocked by 
        `[symbol_col, trade_date_col]`.
    
    bins : int
        Number of equal-width or quantile groups for cross-sectional portfolio segmentation (Strictly >= 3).
    
    freq : str
        Inferred data sampling frequency determined dynamically via temporal delta examination (`map_freq`).
    
    returns_dict : dict[int, pl.DataFrame]
        A period-keyed collection storing dataframes of granular portfolio equity metrics, 
        drawdown tracking series, and compound return curves.
    
    ics_dict : dict[int, pl.DataFrame]
        A horizon-keyed dictionary storing precise chronological timeseries of Information 
        Coefficients, rolling statistics, and factor autocorrelations.
    
    ls_stats_panel : pd.DataFrame
        Aggregated summary matrix containing annualized metrics (Sharpe, Sortino, Calmar, 
        Win Rate, Profit-to-Loss ratio, Max Drawdown) specifically for the Long-Short spread portfolio.
    
    l_stats_panel : pd.DataFrame
        Aggregated summary performance matrix for the long leg portfolio (highest quantile group).
    
    s_stats_panel : pd.DataFrame
        Aggregated summary performance matrix for the short leg portfolio (lowest quantile group).
    
    ic_stats_panel : pd.DataFrame
        Comprehensive summary matrix of historical predictive signal strength distributions, 
        detailing IC means, standard deviations, Skewness, Kurtosis, and Student's t-test outcomes (t-stat and p-value).

    Notes
    -----
    - Memory optimization is preserved through Arrow IPC serialized file buffers (`pa.RecordBatchFileWriter`). 
      Worker nodes access memory-mapped addresses concurrently under read-only sharing configurations 
      to eliminate memory duplication overhead.
    - Global shared data context in parallel routines is retrieved via specialized initializers to ensure 
      process isolation stability under different OS execution regimes.
    - Zero-price barriers are handled natively; zero values in the price series are cast into nulls 
      and dropped during forward-return translation blocks to avoid infinity computational traps.

    Typical Pipeline Workflow
    -------------------------
    >>> analyzer = FactorAnalyzer.simple(df=historical_panel, factor_name="alpha_momentum")
    >>> # Execute multi-processing parallel calculation pipeline
    >>> analyzer.run()
    >>> # Invoke individual standalone interactive diagnostic dashboards
    >>> analyzer.create_single_fac_ic_sheet()
    >>> analyzer.create_long_short_return_sheet()
    >>> # Deep dive cross-sectional transaction layer tracing on specific regime drift dates
    >>> snapshot_df = analyzer.trace(rebalence_period=5, date="2026-05-20 09:30:00", position="l")
    """
    def __init__(self,
                 base_df:pd.DataFrame,
                 trade_date_col:str,
                 symbol_col:str,
                 price_col:str,
                 factor_name:str,
                 research_cfg: ResearchConfig | dict | None = None,
                 analysis_cfg: AnalysisConfig | dict | None = None):
        
        if isinstance(research_cfg, dict):
            research_cfg = ResearchConfig(**research_cfg)

        if isinstance(analysis_cfg, dict):
            analysis_cfg = AnalysisConfig(**analysis_cfg)

        self.research_cfg = research_cfg or ResearchConfig()
        self.analysis_cfg = analysis_cfg or AnalysisConfig()  
        
        self.price_col = price_col 
        self.trade_date_col = trade_date_col
        self.symbol_col = symbol_col
        self.factor_name = factor_name
        self.rebalance_periods = self.research_cfg.rebalance_periods
        self.return_horizons = self.research_cfg.return_horizons
        self.horizon_rolling_period = self.research_cfg.horizon_rolling_period
        self.base_rate =  self.research_cfg.base_rate[0]
        self.overnight = self.research_cfg.overnight
        self.bins = self.research_cfg.bins
        self.fac_shift = self.research_cfg.fac_shift
        self.fee_rate = self.research_cfg.fee_rate[0]
        self.slippage_rate = self.research_cfg.slippage_rate[0]
        self.tax_rate = self.research_cfg.tax_rate[0]
        self.tax_rate_direction = self.research_cfg.tax_rate_direction[0]
        
        self.rank_ic = self.analysis_cfg.rank_ic
        self.log_scale = self.analysis_cfg.log_scale
        self.agg_freq = self.analysis_cfg.agg_freq
        self.group_by = self.analysis_cfg.group_by
        self.max_workers = self.analysis_cfg.max_workers
        
        
        pl.Datetime()
        if isinstance(base_df,pd.DataFrame):
            self.base_df= pl.from_pandas(base_df)
        else:
            self.base_df = base_df.clone()
        self.base_df:pl.DataFrame = self.base_df.with_columns(pl.col(self.trade_date_col).cast(pl.Datetime)).sort([self.symbol_col,self.trade_date_col])
        
        self.td = self.base_df[self.trade_date_col][1] - self.base_df[self.trade_date_col][0]
        self.days = (self.base_df[self.trade_date_col][-1] - self.base_df[self.trade_date_col][0]).days
    
        if self.fac_shift:
            self.base_df = self.base_df.with_columns(
                pl.col(self.factor_name).shift(self.fac_shift).over(self.symbol_col)
            )
        
        if self.bins < 3:
            raise ValueError(f"bins must be >= 3, got {self.bins}")
        
        if self.agg_freq:
            self.freq = self.agg_freq
        else:
            self.freq = FactorAnalyzer.map_freq(self.td)
    
    @classmethod
    def simple(
        cls,
        df,
        factor_name,
        trade_date_col="datetime",
        symbol_col="symbol",
        price_col="close",
        research_cfg=None,
        analysis_cfg=None,
    ):

        required_cols = [trade_date_col, symbol_col, price_col, factor_name]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return cls(
            base_df=df,
            trade_date_col=trade_date_col,
            symbol_col=symbol_col,
            price_col=price_col,
            factor_name=factor_name,
            research_cfg=research_cfg,
            analysis_cfg=analysis_cfg,
        )
    
    @staticmethod
    def map_freq(td: timedelta) -> str | None:
        seconds = td.total_seconds()

        s = 1
        m = 60
        d = 86400

        if 1*s <= seconds < 3*s:
            return "30s"
        elif 5*s <= seconds < 10*s:
            return "3m"
        elif 10*s <= seconds < 30*s:
            return "5m"
        elif 30*s <= seconds < 1*m:
            return "15m"
        elif 1*m <= seconds < 3*m:
            return "30m"
        elif 3*m <= seconds < 15*m:
            return "1h"
        elif 15*m <= seconds < 1*d:
            return "5d"
        elif 1*d <= seconds < 5*d:
            return "20d"
        elif 5*d <= seconds < 365*d:
            return "1y"
        else: 
            return None
    
    @staticmethod
    def add_subtitle(fig, text, row, y=1.15, Exposures=False):
        
        if Exposures == True:
            axis_index = 2 * row - 1
            yref = "y domain" if axis_index == 1 else f"y{axis_index} domain"
            
        else:
            yref = f"y{row if row > 1 else ''} domain"
        fig.add_annotation(
            text=text,
            xref="paper",
            yref=yref,
            x=0.475,
            y=y,
            xanchor="center",
            showarrow=False,
            font=dict(size=22, color="black", family="Arial")
        )
    
    @staticmethod
    def map_symbol_to_industry(df: pd.DataFrame, symbol_col: str, dummy_dict: dict, industry_col: str = "industry") -> pd.DataFrame:
        """
        Map a symbol column to industry based on a dummy dictionary.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing a symbol column.
        symbol_col : str
            Column in df containing the stock/asset symbols.
        dummy_dict : dict
            Dictionary mapping symbol -> industry.
        industry_col : str, default 'industry'
            Name of the new column for industry labels.

        Returns
        -------
        pd.DataFrame
            DataFrame with the new industry column.
        """
        df[industry_col] = df[symbol_col].map(dummy_dict)
        return df
    
    def _overnight(self, df: pl.DataFrame, period: int, col_names: list, type:str) -> pl.DataFrame:
        if type == "ic":
            if self.overnight == "off":
                df = df.with_columns(
                    pl.col(self.trade_date_col).max()
                    .over(pl.col(self.trade_date_col).dt.date())
                    .alias("day_last_dt"),

                    (pl.col(self.trade_date_col) + period * self.td).alias("target_dt")
                )

                df = df.with_columns(
                    (pl.col("target_dt") > pl.col("day_last_dt"))
                    .alias("is_overnight")
                )
                mask = ~pl.col("is_overnight")
                
                df = df.filter(mask).select(
                    col_names
                )

            elif self.overnight == "only":
                df = df.with_columns(
                    pl.col(self.trade_date_col).max()
                    .over(pl.col(self.trade_date_col).dt.date())
                    .alias("day_last_dt"),

                    (pl.col(self.trade_date_col) + period * self.td).alias("target_dt")
                )

                df = df.with_columns(
                    (pl.col("target_dt") > pl.col("day_last_dt"))
                    .alias("is_overnight")
                )
                mask = pl.col("is_overnight")
                
                df = df.filter(mask).select(
                    col_names
                )
        elif type == "autocorr":
            if self.overnight == "off":
                df = df.with_columns(
                    pl.col(self.trade_date_col).max()
                    .over(pl.col(self.trade_date_col).dt.date())
                    .alias("day_last_dt"),

                    (pl.col(self.trade_date_col) - period * self.td).alias("target_dt")
                )

                df = df.with_columns(
                    (pl.col("target_dt") > pl.col("day_last_dt"))
                    .alias("is_overnight")
                )
                mask = ~pl.col("is_overnight")
                
                df = df.filter(mask).select(
                    col_names
                )

            elif self.overnight == "only":
                df = df.with_columns(
                    pl.col(self.trade_date_col).max()
                    .over(pl.col(self.trade_date_col).dt.date())
                    .alias("day_last_dt"),

                    (pl.col(self.trade_date_col) - period * self.td).alias("target_dt")
                )

                df = df.with_columns(
                    (pl.col("target_dt") > pl.col("day_last_dt"))
                    .alias("is_overnight")
                )
                mask = pl.col("is_overnight")
                
                df = df.filter(mask).select(
                    col_names
                )
        return df
    
    def _aggregation(self, df: pl.DataFrame, type:str) -> pl.DataFrame:
        if type == "ic" or "rank_ic":
            if self.agg_freq:
                agg_df = (
                            df.with_columns(pl.col(self.trade_date_col).dt.truncate(self.agg_freq).alias("agg_date"))
                            .group_by("agg_date")
                        .agg(
                                pl.col(type).sum().alias("monthly_ic_sum")
                                ).select(["agg_date", "monthly_ic_sum"])
                            )
            else:
                if self.freq != "1y" and self.freq is not None and self.freq != "20d":
                    agg_df = (
                                df.with_columns(pl.col(self.trade_date_col).dt.truncate(self.freq).alias("agg_date"))
                                .group_by("agg_date")
                                .agg(
                                    pl.col(type).sum().alias("monthly_ic_sum")
                                    ).select(["agg_date", "monthly_ic_sum"])
                                )
                
                if self.freq == "20d":
                    agg_df = (
                                df.with_columns(pl.col(self.trade_date_col).dt.strftime("%Y-%m").alias("agg_date"))
                                .group_by("agg_date")
                                .agg(
                                    pl.col(type).sum().alias("monthly_ic_sum")
                                    ).select(["agg_date", "monthly_ic_sum"])
                                )    
                if self.freq == "1y" or self.freq == None:
                    agg_df = pl.Datetime()
        return agg_df
     
    def _rebalance_date(self, freq:str, df:pl.DataFrame):
        if freq == "W":
            rebalance_dates = (
                df.select([
                    pl.col(self.trade_date_col),
                    pl.col(self.trade_date_col).dt.strftime("%Y-%W").alias("period")
                ])
                .group_by("period")
                .agg(
                    pl.col(self.trade_date_col).max().alias(self.trade_date_col)
                )
                .select(self.trade_date_col)
                .to_series()
                .sort()
            )

        elif freq == "M":
            rebalance_dates = (
                df.select([
                    pl.col(self.trade_date_col),
                    pl.col(self.trade_date_col).dt.strftime("%Y-%m").alias("period")
                ])
                .group_by("period")
                .agg(
                    pl.col(self.trade_date_col).max().alias(self.trade_date_col)
                )
                .select(self.trade_date_col)
                .to_series()
                .sort()
            )

        elif freq == "Q":
            rebalance_dates = (
                df.select([
                    pl.col(self.trade_date_col),
                    (
                        pl.col(self.trade_date_col).dt.year().cast(pl.Utf8)
                        + "-Q"
                        + pl.col(self.trade_date_col).dt.quarter().cast(pl.Utf8)
                    ).alias("period")
                ])
                .group_by("period")
                .agg(
                    pl.col(self.trade_date_col).max().alias(self.trade_date_col)
                )
                .select(self.trade_date_col)
                .to_series()
                .sort()
            )

        elif freq == "H":
            rebalance_dates = (
                df.select([
                    pl.col(self.trade_date_col),
                    (
                        pl.col(self.trade_date_col).dt.year().cast(pl.Utf8)
                        + "-H"
                        + (
                            ((pl.col(self.trade_date_col).dt.month() - 1) // 6 + 1)
                            .cast(pl.Utf8)
                        )
                    ).alias("period")
                ])
                .group_by("period")
                .agg(
                    pl.col(self.trade_date_col).max().alias(self.trade_date_col)
                )
                .select(self.trade_date_col)
                .to_series()
                .sort()
            )

        elif freq == "Y":
            rebalance_dates = (
                df.select([
                    pl.col(self.trade_date_col),
                    pl.col(self.trade_date_col).dt.year().alias("period")
                ])
                .group_by("period")
                .agg(
                    pl.col(self.trade_date_col).max().alias(self.trade_date_col)
                )
                .select(self.trade_date_col)
                .to_series()
                .sort()
            )
        else:
            raise ValueError("invalid freq")
        
        return rebalance_dates
    
    def calc_stats_for_period(self,args):
        period, base_df_path = args
        
        global _worker_df

        if _worker_df is None:
            with pa.memory_map(base_df_path, "r") as source:
                _worker_df = pl.from_arrow(
                    pa.ipc.open_file(source).read_all()
                )
        ###########################################贴标签分箱
        df = _worker_df.clone()
        
        df = df.with_columns([
            pl.col(self.factor_name).rank("average", descending=False).over(self.trade_date_col).alias("rank"),
            pl.len().over(self.trade_date_col).alias("n_stocks"),
        ])

        df = df.sort([self.trade_date_col, "rank"])  
        df = df.with_columns([
            pl.arange(0, pl.len(), eager=False).over(self.trade_date_col).alias("pos_index"),
        ])

        df = df.with_columns([
            (
                ((pl.col("pos_index") * self.bins) / pl.col("n_stocks")).floor() + 1
            ).cast(pl.Int32).alias("quantile_temp")
        ])

        df = df.with_columns(
            pl.when(pl.col("quantile_temp") > self.bins)
            .then(self.bins)
            .otherwise(pl.col("quantile_temp"))
            .alias("quantile")
        ).drop("quantile_temp").drop("pos_index").filter(pl.col(self.factor_name).is_not_null())
        
        ####################分g和原始w
        df = df.with_columns([
            pl.when(pl.col(self.price_col) == 0).then(None).otherwise(pl.col(self.price_col)).alias(self.price_col),
            pl.when(pl.col(self.price_col).shift(-1).over(self.symbol_col) == 0)
            .then(None)
            .otherwise(pl.col(self.price_col).shift(-1).over(self.symbol_col))
            .alias("price_fut")
        ]).filter(pl.col("price_fut").is_not_null())
        df = df.with_columns(
            ((pl.col("price_fut") / pl.col(self.price_col)) - 1).alias("fut_ret_1")
        )
        df = df.with_columns(
            pl.when(pl.col("fut_ret_1") == -1).then(None).otherwise(pl.col("fut_ret_1")).alias("fut_ret_1")
        )
        
        if isinstance(period, int):
            all_dates = df[self.trade_date_col].unique().sort()
            rebalance_dates = all_dates[::period]
        else:
            rebalance_dates = self._rebalance_date(period,df)
        
        rebalance_mask = pl.col(self.trade_date_col).is_in(
            rebalance_dates.implode()
        )
   
        df = df.sort(by=[self.trade_date_col, "quantile"])

        df = df.with_columns(
            global_date_id = pl.col(self.trade_date_col).rank(method="dense") - 1
        )

        if isinstance(period, int):

            df_o = (
                df
                .with_columns(
                    g = pl.col("global_date_id") // period
                )
                .drop("global_date_id")
            )

        else:

            g_df = (
                df
                .select(self.trade_date_col)
                .unique()
                .sort(self.trade_date_col)
                .with_columns(
                    rebalance_flag = (
                        pl.col(self.trade_date_col)
                        .is_in(rebalance_dates.implode())
                        .cast(pl.Int32)
                    )
                )
                .with_columns(
                    g = (
                        pl.col("rebalance_flag")
                        .cum_sum()
                        - 1
                    )
                )
                .drop("rebalance_flag")
            )

            df_o = (
                df
                .join(g_df, on=self.trade_date_col, how="left")
                .drop("global_date_id")
            )

        rebalance_df = df_o.filter(rebalance_mask).select([
            self.trade_date_col,
            self.symbol_col,
            "quantile",
            "g"
        ])

        df = df_o.join(
            rebalance_df.select([self.symbol_col, "g", "quantile"]),
            on=[self.symbol_col, "g"],
            how="inner"  
        )
        df = df.with_columns(
            quantile = pl.col("quantile_right")
        ).drop(["quantile_right"])

        df = df.with_columns(
            w = pl.when(rebalance_mask).then(
                1.0 / pl.len().over([self.trade_date_col, "quantile"])
            ).otherwise(0.0)
        )
        rebalance_df = rebalance_df.sort(by=[self.trade_date_col, "quantile",self.symbol_col])
        ####################################buqi
        
        
        date_g = df.select([self.trade_date_col, "g"]).unique()

        holdings = rebalance_df.select([
            self.symbol_col,
            "g",
            "quantile"
        ])

        full = holdings.join(date_g, on="g", how="inner")

        df_small = df.select([
            self.symbol_col,
            self.trade_date_col,
            "g",
            "fut_ret_1",
            "w"
        ])

        full = full.join(
            df_small,
            on=[self.symbol_col, self.trade_date_col, "g"],
            how="left"
        )

        df = full.with_columns([
            pl.col("w").fill_null(0.0),
            pl.col("fut_ret_1").fill_null(0.0)
        ])
        ############################################
        df = df.sort([self.symbol_col, self.trade_date_col])

        df = df.with_columns(
            w0 = pl.when(pl.col("w") > 0)
                .then(pl.col("w"))
                .otherwise(None)
        )

        df = df.with_columns(
            w0 = pl.col("w0").forward_fill().over([self.symbol_col, "g"])
        )

        df = df.with_columns(
            cum_ret = (1 + pl.col("fut_ret_1")).cum_prod().over([self.symbol_col, "g"])
        )

        df = df.with_columns(
            cum_ret_lag = pl.col("cum_ret").shift(1).over([self.symbol_col, "g"])
        )

        df = df.with_columns(
            cum_ret_lag = pl.when(pl.col("cum_ret_lag").is_null())
                .then(1.0)
                .otherwise(pl.col("cum_ret_lag"))
        ).sort(["g",self.trade_date_col,"quantile", self.symbol_col])

        df = df.with_columns(
            raw_w = pl.col("w0") * pl.col("cum_ret_lag")
        )

        df = df.with_columns(
            norm = pl.col("raw_w").sum().over([self.trade_date_col, "quantile"])
        )

        df = df.with_columns(
            w = pl.col("raw_w") / pl.col("norm")
        )
        ###############################################################rebalance_return_to_nan
    
        g_info = (
            df.group_by("g")
            .agg([
                pl.col(self.trade_date_col).max().alias("g_last_date"),
                pl.col(self.trade_date_col).min().alias("g_first_date"),
            ])
            .sort("g_first_date")
            .with_columns(
                pl.col("g_first_date").shift(-1).alias("next_g_first_date")
            )
        )


        last_rows = df.join(
            g_info,
            on="g",
            how="inner"
        ).filter(
            pl.col(self.trade_date_col) == pl.col("g_last_date")
        ).filter(
            pl.col("next_g_first_date").is_not_null()
        )


        df_new = last_rows.with_columns([
            pl.col("next_g_first_date").alias(self.trade_date_col),
            pl.col("cum_ret").alias("cum_ret_lag"),
        ])

        df_new = df_new.with_columns(
            raw_w = pl.col("w0") * pl.col("cum_ret_lag")
        )

        df_new = df_new.with_columns(
            norm = pl.col("raw_w").sum().over([self.trade_date_col, "quantile"])
        )

        df_new = df_new.with_columns(
            w = pl.col("raw_w") / pl.col("norm")
        ).drop(["g_last_date", "g_first_date", "next_g_first_date"]).sort([ "g", self.trade_date_col,'quantile',self.symbol_col])
        
        df_new = (
            df_new
            .with_columns(
                pl.lit(None).alias("fut_ret_1")
            ))
        df = pl.concat([df, df_new]).drop(["w0","cum_ret","cum_ret_lag","raw_w","norm"])
        ##########################################################

        df_reb = df.filter(rebalance_mask)
        
        if self.tax_rate_direction == "both":
            fee = self.tax_rate + self.fee_rate + self.slippage_rate
        
        elif self.tax_rate_direction == "buy":
            fee_buy = self.tax_rate + self.fee_rate + self.slippage_rate
            fee_sell = self.fee_rate + self.slippage_rate
        
        elif self.tax_rate_direction == "sell":
            fee_buy = self.fee_rate + self.slippage_rate
            fee_sell = self.tax_rate + self.fee_rate + self.slippage_rate

        else:
            raise ValueError(f"tax_rate_direction can be set to 'buy', 'sell', or 'both', got {self.tax_rate_direction}")
        
        fee_buy = fee if self.tax_rate_direction == "both" else fee_buy
        fee_sell = fee if self.tax_rate_direction == "both" else fee_sell
        
        df_reb = df_reb.sort([self.trade_date_col, "quantile", "g", self.symbol_col])

        min_dt = df_reb.select(pl.col(self.trade_date_col).min()).item()

        df_init = df_reb.filter(
            (pl.col(self.trade_date_col) == min_dt) & (pl.col("g") == 0)
        ).with_columns([
            pl.col("w").alias("buy"),
            pl.lit(0.0).alias("sell"),
        ])
        
        df_init = df_init.with_columns([
            (pl.col("buy")).alias("turnover"),
            (pl.col("buy") * fee_buy).alias("cost")
        ])


        df_main = df_reb.filter(~(
            (pl.col(self.trade_date_col) == min_dt) & (pl.col("g") == 0)
        )).sort([ "g", self.trade_date_col,'quantile',self.symbol_col])

        
        g_pairs = (
            df_main.select([self.trade_date_col, "quantile", "g"])
            .unique()
            .sort([self.trade_date_col, "quantile", "g"])
            .with_columns([
                pl.col("g").shift(1).over([self.trade_date_col, "quantile"]).alias("g_prev"),
                pl.col("g").alias("g_next")
            ])
            .drop_nulls()
        )

        
        df_prev = df_main.select([
            self.trade_date_col, "quantile", "g", self.symbol_col, "w"
        ]).rename({
            "g": "g_prev",
            "w": "w_prev"
        })

        df_next = df_main.select([
            self.trade_date_col, "quantile", "g", self.symbol_col, "w"
        ]).rename({
            "g": "g_next",
            "w": "w_next"
        })

        df_prev = df_prev.join(
            g_pairs, on=[self.trade_date_col, "quantile", "g_prev"], how="inner"
        )

        df_next = df_next.join(
            g_pairs, on=[self.trade_date_col, "quantile", "g_next"], how="inner"
        )

        
        common = df_prev.join(
            df_next,
            on=[self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col],
            how="inner"
        )

        common = common.with_columns([
            (pl.col("w_next") - pl.col("w_prev")).alias("diff"),
        ])

        common = common.with_columns([
            pl.when(pl.col("diff") > 0).then(pl.col("diff")).otherwise(0).alias("buy"),
            pl.when(pl.col("diff") < 0).then(-pl.col("diff")).otherwise(0).alias("sell"),
        ])

        
        only_prev = df_prev.join(
            df_next.select([self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col]),
            on=[self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col],
            how="anti"
        )

        only_prev = only_prev.with_columns([
            pl.lit(0.0).alias("buy"),
            pl.col("w_prev").alias("sell"),
        ])

        
        only_next = df_next.join(
            df_prev.select([self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col]),
            on=[self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col],
            how="anti"
        )

        only_next = only_next.with_columns([
            pl.col("w_next").alias("buy"),
            pl.lit(0.0).alias("sell"),
        ])
        
       
        df_pair = pl.concat([common, only_prev, only_next], how="diagonal")

       
        buy_df = df_pair.select([
            self.trade_date_col, "quantile", "g_next", self.symbol_col, "buy"
        ]).rename({"g_next": "g"})

        sell_df = df_pair.select([
            self.trade_date_col, "quantile", "g_prev", self.symbol_col, "sell"
        ]).rename({"g_prev": "g"})

       
        df_main = df_main.join(buy_df, on=[self.trade_date_col, "quantile", "g", self.symbol_col], how="left")
        df_main = df_main.join(sell_df, on=[self.trade_date_col, "quantile", "g", self.symbol_col], how="left")

        df_main = df_main.with_columns([
            pl.col("buy").fill_null(0),
            pl.col("sell").fill_null(0),
        ])

        df_main = df_main.with_columns([
            (pl.col("buy") + pl.col("sell")).alias("turnover"),
            (pl.col("buy") * fee_buy + pl.col("sell") * fee_sell).alias("cost")
        ])


        df_reb = pl.concat([df_init, df_main])
        #####################################################################
        df_agg = df_reb.group_by([self.trade_date_col, "quantile"]).agg([
                pl.col("buy").sum().alias("buy"),
                pl.col("sell").sum().alias("sell"),
                pl.col("turnover").sum().alias("turnover"),
                pl.col("cost").sum().alias("cost")
            ])

        df_agg = (
            df_agg.pivot(
                on="quantile",                            
                index=self.trade_date_col,                
                values=["buy", "sell", "turnover", "cost"] 
            )
        )

        df_agg = df_agg.with_columns([
            ((pl.col("turnover_1") + pl.col(f"turnover_{self.bins}")) / 2).alias("turnover_ls"),
            ((pl.col("cost_1") + pl.col(f"cost_{self.bins}")) / 2).alias("cost_ls"),
            ((pl.col("buy_1") + pl.col(f"buy_{self.bins}")) / 2).alias("buy_ls"),
            ((pl.col("sell_1") + pl.col(f"sell_{self.bins}")) / 2).alias("sell_ls")
        ]).sort(self.trade_date_col)
        
        
        ########################
        df = df.join(
            df_reb.select([
                self.trade_date_col, "quantile", "g", self.symbol_col,
                "buy", "sell", "turnover", "cost"
            ]),
            on=[self.trade_date_col, "quantile", "g", self.symbol_col],
            how="left"
        ).sort([ "g", self.trade_date_col,'quantile',self.symbol_col]).with_columns([
            (pl.col("w") * pl.col("fut_ret_1")).alias("ret")
        ])
        
        if self.group_by:
            df_ind = df.with_columns([
                pl.col(self.symbol_col)
                .replace(self.group_by, default="Unlabeled")
                .alias("industry")
            ])
            
            df_indus_daily = (
                df_ind.filter(pl.col("quantile").is_in([1, self.bins]))
                .with_columns([
                    (-pl.col("ret")).alias("ret_s"),
                    pl.col("cost").fill_null(0.0).alias("cost_filled")
                ])
                .group_by([self.trade_date_col, "industry", "quantile"])
                .agg([
                    (pl.col("ret").sum() - pl.col("cost_filled").sum()).alias("ret_net_daily"),
                    (pl.col("ret_s").sum() - pl.col("cost_filled").sum()).alias("ret_net_s_daily")
                ])
            )
            
            df_indus_avg = (
                df_indus_daily.group_by(["industry", "quantile"])
                .agg([
                    pl.col("ret_net_daily").mean(),
                    pl.col("ret_net_s_daily").mean()
                ])
            )
            
            df_indus_qb = (
                df_indus_avg.filter(pl.col("quantile") == 1)
                .select([pl.col("industry"), pl.col("ret_net_s_daily").alias("ret_net_s_qb_mean")])
            )
            df_indus_qt = (
                df_indus_avg.filter(pl.col("quantile") == self.bins)
                .select([pl.col("industry"), pl.col("ret_net_daily").alias("ret_net_qt_mean")])
            )
            
            df_indus = (
                df_indus_qb.join(df_indus_qt, on="industry", how="left")
                .with_columns([
                    (pl.col("ret_net_qt_mean") - pl.col("ret_net_s_qb_mean")).alias("ret_net_ls_i_mean")
                ])
            )
        else:
            df_indus = pl.DataFrame()
        ##########################################################
        df_g = (
            df
            .group_by([self.trade_date_col, "g", "quantile"])
            .agg([
                pl.col("ret").sum().alias("ret"),
                pl.col("cost").sum().alias("cost")
            ])
        ).with_columns([(-pl.col("ret")).alias("ret_s")]).drop(["g"])


        ###################################################
        df_barly_q = (
            df_g.group_by([self.trade_date_col, "quantile"])
            .agg([
                pl.col("ret").sum().alias("ret"),
                pl.col("ret_s").sum().alias("ret_s"),
                pl.col("cost").sum().alias("cost"),
            ])
        )

        df_result = df_barly_q.pivot(
            on="quantile",
            index=self.trade_date_col,
            values=["ret", "ret_s", "cost"]
        )

        df_result = (
            df_result.with_columns([
                (pl.col(f"ret_{q}") - pl.col(f"cost_{q}")).alias(f"ret_net_{q}")
                for q in range(1, self.bins + 1)
            ] + [
                (pl.col(f"ret_s_{q}") - pl.col(f"cost_{q}")).alias(f"ret_net_s_{q}")
                for q in range(1, self.bins + 1)
            ])
            .sort(self.trade_date_col)
        )
        
        ############################################################
        cum_exprs = []

        for q in range(1, self.bins + 1):
            if q == self.bins:
                cum_exprs.append((1 + pl.col(f"ret_{q}")).cum_prod().alias(f"cum_ret_{q}"))
                
            elif q == 1:
                cum_exprs.append((1 + pl.col(f"ret_s_{q}")).cum_prod().alias(f"cum_ret_s_{q}"))
            
            cum_exprs.extend([
                (1 + pl.col(f"ret_net_{q}")).cum_prod().alias(f"cum_ret_net_{q}"),
                (1 + pl.col(f"ret_net_s_{q}")).cum_prod().alias(f"cum_ret_net_s_{q}")
            ])
        
        cum_exprs.append(((pl.col(f"ret_{self.bins}") - pl.col(f"ret_1")) / 2).alias("ret_ls"))
        
        df_result = (df_result.with_columns(cum_exprs)
                     .with_columns([(pl.col("ret_ls") - (pl.col("cost_1") + pl.col(f"cost_{self.bins}")) / 2).alias("ret_net_ls")])
                     .with_columns([
                        (1 + pl.col("ret_ls")).cum_prod().alias("cum_ret_ls"),
                        (1 + pl.col("ret_net_ls")).cum_prod().alias("cum_ret_net_ls")
                     ])
                     .drop("^cost.*$"))
        
        df_result = df_result.with_columns(
            pl.col(self.trade_date_col).shift(-1) 
        )
        
        if df_result.height == 0:
            raise
        n_periods = len(df_result)
        days_per_period = self.days / n_periods
        annual_factor = 365.25 / days_per_period
        
        avg_turnover_l = df_agg[f'turnover_{self.bins}'].drop_nans().mean()
        avg_turnover_s = df_agg[f'turnover_1'].drop_nans().mean()
        avg_turnover_ls = df_agg['turnover_ls'].drop_nans().mean()
        df_result = df_result.drop("^turnover.*$")
        ##############################################################################################ls
        cum_nv = df_result["cum_ret_net_ls"].to_numpy()
        mean_ret = df_result.select(pl.col("ret_net_ls").mean()).item()
        mean_loss = (df_result.filter(pl.col("ret_net_ls") < 0).select(pl.col("ret_net_ls").mean()).item())
        PL = mean_ret / mean_loss
        ann_ret = cum_nv[-1]** (365.25 / self.days) - 1

        vol = (df_result.select(pl.col("ret_net_ls").std(ddof=1)).item()) * np.sqrt(annual_factor)
        if np.isnan(vol):
            vol = (df_result.select(pl.col("ret_net_ls").std(ddof=0)).item()) * np.sqrt(annual_factor)

        excess_ret = ann_ret - self.base_rate
        sharpe = excess_ret / vol if vol != 0 and not np.isnan(vol) else np.nan
        
        running_max = np.maximum.accumulate(cum_nv)
        drawdown = cum_nv / running_max - 1
        max_dd = float(np.nanmin(drawdown)) if drawdown.size > 0 else np.nan

        win_rate = float(df_result.select((pl.col("ret_net_ls") > 0).mean()).item())
        pnl = float(cum_nv[-1] - 1) if cum_nv.size > 0 else np.nan
        downside_std = (df_result.filter(pl.col("ret_net_ls") < 0).select(pl.col("ret_net_ls").std(ddof=1)).item())
        if downside_std is not None and not np.isnan(downside_std):
            downside_vol = downside_std * np.sqrt(annual_factor)
            sortino = excess_ret / downside_vol if downside_vol != 0 else np.nan
        else:
            sortino = np.nan
        calmar = excess_ret / abs(max_dd) if max_dd != 0 else np.nan
        
            
        stats_ls = {
            "Ann. Return": ann_ret,
            "Ann. Std": vol,
            "Ann. Sharpe": sharpe,
            "Ann. Sortino": sortino,
            "Ann. Calmar" : calmar,
            "Mean Turnover" : avg_turnover_ls,
            "Max Drawdown": max_dd,
            "WinRate": win_rate,
            "P/L" : PL,
            "PnL": pnl
        }
        ####################################################################################################
        if self.agg_freq:
            df_monthly = (
                df_result
                .with_columns(
                    pl.col(self.trade_date_col)
                    .dt.truncate(self.agg_freq)
                    .alias("agg_date"),
                )
                .group_by("agg_date")
                .agg([
                    ((pl.col("ret_net_ls") + 1).product() - 1).alias("ret_net_ls_agg"),
                    ((pl.col(f"ret_net_{self.bins}") + 1).product() - 1).alias("ret_net_qt_agg"),
                    ((pl.col("ret_net_s_1") + 1).product() - 1).alias("ret_net_s_qb_agg")
                ])
            )
            
            monthly_ret_ls = df_monthly.select(["agg_date", "ret_net_ls_agg"])
            monthly_ret_l = df_monthly.select(["agg_date", "ret_net_qt_agg"])
            monthly_ret_s = df_monthly.select(["agg_date", "ret_net_s_qb_agg"])
        
        
        else:   
            if self.freq != "1y" and self.freq is not None and self.freq != "20d":
                df_monthly = (
                    df_result
                    .with_columns(
                        pl.col(self.trade_date_col)
                        .dt.truncate(self.agg_freq)
                        .alias("agg_date"),
                    )
                    .group_by("agg_date")
                    .agg([
                        ((pl.col("ret_net_ls") + 1).product() - 1).alias("ret_net_ls_agg"),
                        ((pl.col(f"ret_net_{self.bins}") + 1).product() - 1).alias("ret_net_qt_agg"),
                        ((pl.col("ret_net_s_1") + 1).product() - 1).alias("ret_net_s_qb_agg")
                    ])
                )
                monthly_ret_ls = df_monthly.select(["agg_date", "ret_net_ls_agg"])
                monthly_ret_l = df_monthly.select(["agg_date", "ret_net_qt_agg"])
                monthly_ret_s = df_monthly.select(["agg_date", "ret_net_s_qb_agg"])
            
            if self.freq == "20d":
                df_monthly = (
                    df_result
                    .with_columns(
                        pl.col(self.trade_date_col)
                        .dt.strftime("%Y-%m")
                        .alias("agg_date"),
                    )
                    .group_by("agg_date")
                    .agg([
                        ((pl.col("ret_net_ls") + 1).product() - 1).alias("ret_net_ls_agg"),
                        ((pl.col(f"ret_net_{self.bins}") + 1).product() - 1).alias("ret_net_qt_agg"),
                        ((pl.col("ret_net_s_1") + 1).product() - 1).alias("ret_net_s_qb_agg")
                    ])
                )
                monthly_ret_ls = df_monthly.select(["agg_date", "ret_net_ls_agg"])
                monthly_ret_l = df_monthly.select(["agg_date", "ret_net_qt_agg"])
                monthly_ret_s = df_monthly.select(["agg_date", "ret_net_s_qb_agg"])
                
            
            if self.freq == "1y" or self.freq == None:
                monthly_ret_ls = pl.DataFrame()
                monthly_ret_l = pl.DataFrame()
                monthly_ret_s = pl.DataFrame()
        ############################################################################################
        df_heatmap_daily_l = df_result.with_columns([
            (pl.col(f"ret_net_{self.bins}") - pl.col("ret_net_ls")).alias("excess_ret"),
            pl.col(self.trade_date_col).dt.year().alias("year"),
            pl.col(self.trade_date_col).dt.month().alias("month")
        ])
        
        df_heatmap_monthly_l = (
            df_heatmap_daily_l.group_by(["year", "month"])
            .agg([
                ((pl.col("excess_ret") + 1).product() - 1).alias("monthly_excess_cum")
            ])
            .sort(["year", "month"])
        )
        
        df_heatmap_calendar_l = (
            df_heatmap_monthly_l.to_pandas()
            .pivot(index="year", columns="month", values="monthly_excess_cum")
            .fillna(0.0)
        )
        df_heatmap_calendar_l = df_heatmap_calendar_l.reindex(columns=range(1, 13), fill_value=0.0)
        
        df_heatmap_daily_s = df_result.with_columns([
            (pl.col(f"ret_net_s_1") - pl.col("ret_net_ls")).alias("excess_ret"),
            pl.col(self.trade_date_col).dt.year().alias("year"),
            pl.col(self.trade_date_col).dt.month().alias("month")
        ])
        
        df_heatmap_monthly_s = (
            df_heatmap_daily_s.group_by(["year", "month"])
            .agg([
                ((pl.col("excess_ret") + 1).product() - 1).alias("monthly_excess_cum")
            ])
            .sort(["year", "month"])
        )
        
        df_heatmap_calendar_s = (
            df_heatmap_monthly_s.to_pandas()
            .pivot(index="year", columns="month", values="monthly_excess_cum")
            .fillna(0.0)
        )
        df_heatmap_calendar_s = df_heatmap_calendar_s.reindex(columns=range(1, 13), fill_value=0.0)
        ############################################################################################l       
        
        cum_nv = df_result[f"cum_ret_net_{self.bins}"].to_numpy()
        mean_ret_l = df_result.select(pl.col(f"ret_net_{self.bins}").mean()).item()
        mean_loss = (df_result.filter(pl.col(f"ret_net_{self.bins}") < 0).select(pl.col(f"ret_net_{self.bins}").mean()).item())
        PL = mean_ret_l / mean_loss
        ann_ret = cum_nv[-1]** (365.25 / self.days) - 1
        vol = (df_result.select(pl.col(f"ret_net_{self.bins}").std(ddof=1)).item()) * np.sqrt(annual_factor)
        if np.isnan(vol):
            vol = (df_result.select(pl.col(f"ret_{self.bins}").std(ddof=0)).item()) * np.sqrt(annual_factor)

        excess_ret = ann_ret - self.base_rate
        sharpe = excess_ret/ vol if vol != 0 and not np.isnan(vol) else np.nan
        
        running_max = np.maximum.accumulate(cum_nv)
        drawdown = cum_nv / running_max - 1
        max_dd = float(np.nanmin(drawdown)) if drawdown.size > 0 else np.nan

        win_rate = float(df_result.select((pl.col(f"ret_net_{self.bins}") > 0).mean()).item())
        pnl = float(cum_nv[-1] - 1) if cum_nv.size > 0 else np.nan
        downside_std = (df_result.filter(pl.col(f"ret_net_{self.bins}") < 0).select(pl.col(f"ret_net_{self.bins}").std(ddof=1)).item())
        if downside_std is not None and not np.isnan(downside_std):
            downside_vol = downside_std * np.sqrt(annual_factor)
            sortino = excess_ret / downside_vol if downside_vol != 0 else np.nan
        else:
            sortino = np.nan
        calmar = excess_ret / abs(max_dd) if max_dd != 0 else np.nan

        stats_l = {
            "Ann. Return": ann_ret,
            "Ann. Std": vol,
            "Ann. Sharpe": sharpe,
            "Ann. Sortino": sortino,
            "Ann. Calmar" : calmar,
            "Mean Turnover" : avg_turnover_l,
            "Max Drawdown": max_dd,
            "WinRate": win_rate,
            "P/L" : PL,
            "PnL": pnl
        }
        ######################################################s
        cum_nv = df_result["cum_ret_net_s_1"].to_numpy()
        mean_ret_s = df_result.select(pl.col("ret_net_s_1").mean()).item()
        mean_loss = (df_result.filter(pl.col("ret_net_s_1") < 0).select(pl.col("ret_net_s_1").mean()).item())
        PL = mean_ret_s / mean_loss
        ann_ret = cum_nv[-1]** (365.25 / self.days) - 1
        vol = (df_result.select(pl.col("ret_net_s_1").std(ddof=1)).item()) * np.sqrt(annual_factor)
        if np.isnan(vol):
            vol = (df_result.select(pl.col("ret_net_s_1").std(ddof=0)).item()) * np.sqrt(annual_factor)

        excess_ret = ann_ret - self.base_rate
        sharpe = excess_ret / vol if vol != 0 and not np.isnan(vol) else np.nan
        
        running_max = np.maximum.accumulate(cum_nv)
        drawdown = cum_nv / running_max - 1
        max_dd = float(np.nanmin(drawdown)) if drawdown.size > 0 else np.nan

        win_rate = float(df_result.select((pl.col("ret_net_s_1") > 0).mean()).item())
        pnl = float(cum_nv[-1] - 1) if cum_nv.size > 0 else np.nan
        downside_std = (df_result.filter(pl.col("ret_net_s_1") < 0).select(pl.col("ret_net_s_1").std(ddof=1)).item())
        if downside_std is not None and not np.isnan(downside_std):
            downside_vol = downside_std * np.sqrt(annual_factor)
            sortino = excess_ret / downside_vol if downside_vol != 0 else np.nan
        else:
            sortino = np.nan
        calmar = excess_ret / abs(max_dd) if max_dd != 0 else np.nan
        
        stats_s = {
            "Ann. Return": ann_ret,
            "Ann. Std": vol,
            "Ann. Sharpe": sharpe,
            "Ann. Sortino": sortino,
            "Ann. Calmar" : calmar,
            "Ann. Turnover" : avg_turnover_s,
            "Max Drawdown": max_dd,
            "WinRate": win_rate,
            "P/L" : PL,
            "PnL": pnl
        }
        
        if self.log_scale:
            log_exprs = [
                pl.col("cum_ret_ls").log().alias("cum_ret_ls"),
                pl.col("cum_ret_net_ls").log().alias("cum_ret_net_ls")
            ]
            
            log_exprs.append(pl.col(f"cum_ret_{self.bins}").log().alias(f"cum_ret_{self.bins}"))
            log_exprs.append(pl.col("cum_ret_s_1").log().alias("cum_ret_s_1"))
        
            for q in range(1, self.bins + 1):
                log_exprs.extend([
                    pl.col(f"cum_ret_net_{q}").log().alias(f"cum_ret_net_{q}"),
                    pl.col(f"cum_ret_net_s_{q}").log().alias(f"cum_ret_net_s_{q}")
                ])
            
            df_result = df_result.with_columns(log_exprs)
        
        return period, stats_ls, df_result, mean_ret, monthly_ret_ls, monthly_ret_l, stats_l, monthly_ret_s, stats_s, df_indus, avg_turnover_ls, df_agg, df_heatmap_calendar_l, df_heatmap_calendar_s
    
    def calc_stats_for_horizon(self,args):
        period, base_df_path = args
        global _worker_df

        if _worker_df is None:
            with pa.memory_map(base_df_path, "r") as source:
                _worker_df = pl.from_arrow(
                    pa.ipc.open_file(source).read_all()
                )

        df = _worker_df.clone()
        
        corr = df.with_columns([
            pl.col(self.factor_name).rank("dense").over(self.trade_date_col).alias("factor_rank")
        ])

        corr = corr.with_columns([
            pl.col("factor_rank").shift(period).over(self.symbol_col).alias("lag_rank")
        ])

        corr = (corr.group_by(self.trade_date_col)
                .agg(pl.corr("factor_rank", "lag_rank", method="pearson").alias("autocorr"))
                .sort(self.trade_date_col))
        
        if self.overnight == "off" or self.overnight == "only":
            corr = self._overnight(corr,period,[self.trade_date_col, "autocorr"],type='autocorr')

        mean_ic_autocorr =  corr.select(pl.col("autocorr").drop_nans().mean()).item()
        
        df = df.with_columns([
            pl.when(pl.col(self.price_col) == 0).then(None).otherwise(pl.col(self.price_col)).alias(self.price_col),
            pl.when(pl.col(self.price_col).shift(-period).over(self.symbol_col) == 0)
            .then(None)
            .otherwise(pl.col(self.price_col).shift(-period).over(self.symbol_col))
            .alias("price_fut")
        ])
        df = df.with_columns(
            ((pl.col("price_fut") / pl.col(self.price_col)) - 1).alias("fut_ret")
        )
        
        if self.group_by:
            df = df.with_columns([
                        pl.when(pl.col("fut_ret") == -1).then(None).otherwise(pl.col("fut_ret")).alias("fut_ret"),
                pl.col(self.symbol_col).map_elements(lambda x: self.group_by.get(x, "Unlabeled"), return_dtype=pl.Utf8).alias("industry")
            ])
        
        else:
            df = df.with_columns(
                        pl.when(pl.col("fut_ret") == -1).then(None).otherwise(pl.col("fut_ret")).alias("fut_ret"))
            
        if self.rank_ic:
            if self.group_by:    
                industry_ic = (
                    df.group_by([self.trade_date_col, "industry"])
                    .agg(pl.corr(self.factor_name, "fut_ret", method="spearman").alias("industry_ic"),
                        pl.count().alias("n_stocks"))  
                    .drop_nans()
                )
                
                if self.overnight == "off" or self.overnight == "only":
                    industry_ic = self._overnight(industry_ic,period,[self.trade_date_col,"n_stocks" ,"industry" , "industry_ic"],type='ic')

                industry_ic = industry_ic.with_columns([
                    (pl.col("n_stocks") / pl.col("n_stocks").sum().over(self.trade_date_col)).alias("weight")
                ])

                industry_ic = industry_ic.with_columns([
                    (pl.col("industry_ic") * pl.col("weight")).alias("industry_contrib")
                ])

                industry_contrib = (
                    industry_ic.group_by("industry")
                            .agg(pl.col("industry_contrib").mean().alias("contrib"))
                            .sort("contrib", descending=True)
                )
            else:
                industry_contrib = pl.DataFrame()
                
            df = (df.group_by(self.trade_date_col)
                .agg(pl.corr(self.factor_name, "fut_ret", method="spearman").alias("rank_ic"))
                .sort(self.trade_date_col)).drop_nans()
            
            if self.overnight == "off" or self.overnight == "only":
                df = self._overnight(df,period,[self.trade_date_col, "rank_ic"],type='ic')
            
            df = df.with_columns([
                pl.col("rank_ic").rolling_mean(window_size=self.horizon_rolling_period).alias("rank_ic_rolling"),
                pl.col("rank_ic").cum_sum().alias("rank_ic_cum")
                ])
            df = df.join(corr, on=self.trade_date_col, how="left")
        
            monthly_ic_cum = self._aggregation(df,type='rank_ic')
            
            mean_ic = df.select(pl.col("rank_ic").mean()).item()
            rank_ic_values = df['rank_ic'].to_numpy()
            skew_val = stats.skew(rank_ic_values, nan_policy="omit")
            kurt_val = stats.kurtosis(rank_ic_values, fisher=True, nan_policy="omit")  
            t_val, p_val = stats.ttest_1samp(rank_ic_values, 0.0, nan_policy="omit")
            std_val = np.nanstd(rank_ic_values, ddof=1)  
            ir_val = mean_ic / std_val if std_val != 0 else np.nan
            
            ic_panal =  {
            "Mean Rank IC": mean_ic,
            "Std" : std_val,
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
            "t-stat": t_val,
            "p-Value": p_val,
            "IR": ir_val
            }
            
        else:
            if self.group_by:
                industry_ic = (
                    df.group_by([self.trade_date_col, "industry"])
                    .agg(
                        pl.corr(self.factor_name, "fut_ret", method="spearman").alias("industry_ic"),
                        pl.count().alias("n_stocks")
                    )
                    .drop_nans()
                )
                
                if self.overnight == "off" or self.overnight == "only":
                    industry_ic = self._overnight(industry_ic,period,[self.trade_date_col, "industry", "n_stocks", "industry_ic"],type='ic')

                industry_ic = industry_ic.with_columns([
                    (pl.col("n_stocks") / pl.col("n_stocks").sum().over(self.trade_date_col)).alias("weight")
                ])

                industry_ic = industry_ic.with_columns([
                    (pl.col("industry_ic") * pl.col("weight")).alias("industry_contrib")
                ])

                industry_contrib = (
                    industry_ic.group_by("industry")
                            .agg(pl.col("industry_contrib").mean().alias("contrib"))
                            .sort("contrib", descending=True)
                )
            else:
                industry_contrib = pl.DataFrame()
                
            df = (df.group_by(self.trade_date_col)
                .agg(pl.corr(self.factor_name, "fut_ret", method="spearman").alias("ic"))
                .sort(self.trade_date_col)).drop_nans()
            
            if self.overnight == "off" or self.overnight == "only":
                df = self._overnight(df,period,[self.trade_date_col, "ic"],type='ic')
            
            df = df.with_columns([
                pl.col("ic").rolling_mean(window_size=self.horizon_rolling_period).alias("ic_rolling"),
                pl.col("ic").cum_sum().alias("ic_cum")
                ])
            df = df.join(corr, on=self.trade_date_col, how="left")
        
            monthly_ic_cum = self._aggregation(df,type='ic')
            
            mean_ic = df.select(pl.col("ic").mean()).item()
            ic_values = df['ic'].to_numpy()
            skew_val = stats.skew(ic_values, nan_policy="omit")
            kurt_val = stats.kurtosis(ic_values, fisher=True, nan_policy="omit")  
            t_val, p_val = stats.ttest_1samp(ic_values, 0.0, nan_policy="omit")
            std_val = np.nanstd(ic_values, ddof=1)  
            ir_val = mean_ic / std_val if std_val != 0 else np.nan
            
            ic_panal =  {
            "Mean IC": mean_ic,
            "Std" : std_val,
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
            "t-stat": t_val,
            "p-Value": p_val,
            "IR": ir_val
            }            
        
        return period, df, mean_ic, ic_panal, mean_ic_autocorr, monthly_ic_cum, industry_contrib

    def calc_stats_for_trace(self,period,date):
        df = self.base_df.clone()
        
        date = pl.Series([date]).str.strptime(pl.Datetime).item()
        
        df = df.with_columns([
            pl.col(self.factor_name).rank("average").over(self.trade_date_col).alias("rank"),
            pl.len().over(self.trade_date_col).alias("n_stocks"),
        ])

        df = df.sort([self.trade_date_col, "rank"])  
        df = df.with_columns([
            pl.arange(0, pl.len(), eager=False).over(self.trade_date_col).alias("pos_index"),
        ])

        df = df.with_columns([
            (
                ((pl.col("pos_index") * self.bins) / pl.col("n_stocks")).floor() + 1
            ).cast(pl.Int32).alias("quantile_temp")
        ])

        df = df.with_columns(
            pl.when(pl.col("quantile_temp") > self.bins)
            .then(self.bins)
            .otherwise(pl.col("quantile_temp"))
            .alias("quantile")
        ).drop("quantile_temp").drop("pos_index").filter(pl.col(self.factor_name).is_not_null())
        
        ####################分g和原始w
        df = df.with_columns([
            pl.when(pl.col(self.price_col) == 0).then(None).otherwise(pl.col(self.price_col)).alias(self.price_col),
            pl.when(pl.col(self.price_col).shift(-1).over(self.symbol_col) == 0)
            .then(None)
            .otherwise(pl.col(self.price_col).shift(-1).over(self.symbol_col))
            .alias("price_fut")
        ]).filter(pl.col("price_fut").is_not_null())
        df = df.with_columns(
            ((pl.col("price_fut") / pl.col(self.price_col)) - 1).alias("fut_ret_1")
        )
        df = df.with_columns(
            pl.when(pl.col("fut_ret_1") == -1).then(None).otherwise(pl.col("fut_ret_1")).alias("fut_ret_1")
        )
        
        all_dates = df[self.trade_date_col].unique().sort()
        if isinstance(period, int):
            rebalance_dates = all_dates[::period]
        else:
            rebalance_dates = self._rebalance_date(period,df)
        
        rebalance_mask = pl.col(self.trade_date_col).is_in(
            rebalance_dates.implode()
        )
   
        df = df.sort(by=[self.trade_date_col, "quantile"])

        df = df.with_columns(
            global_date_id = pl.col(self.trade_date_col).rank(method="dense") - 1
        )

        if isinstance(period, int):

            df_o = (
                df
                .with_columns(
                    g = pl.col("global_date_id") // period
                )
                .drop("global_date_id")
            )

        else:
            g_df = (
                df
                .select(self.trade_date_col)
                .unique()
                .sort(self.trade_date_col)
                .with_columns(
                    rebalance_flag = (
                        pl.col(self.trade_date_col)
                        .is_in(rebalance_dates.implode())
                        .cast(pl.Int32)
                    )
                )
                .with_columns(
                    g = (
                        pl.col("rebalance_flag")
                        .cum_sum()
                        - 1
                    )
                )
                .drop("rebalance_flag")
            )

            df_o = (
                df
                .join(g_df, on=self.trade_date_col, how="left")
                .drop("global_date_id")
            )

        rebalance_df = df_o.filter(rebalance_mask).select([
            self.trade_date_col,
            self.symbol_col,
            "quantile",
            "g"
        ])

        df = df_o.join(
            rebalance_df.select([self.symbol_col, "g", "quantile"]),
            on=[self.symbol_col, "g"],
            how="inner"  
        )
        df = df.with_columns(
            quantile = pl.col("quantile_right")
        ).drop(["quantile_right"])

        df = df.with_columns(
            w = pl.when(rebalance_mask).then(
                1.0 / pl.len().over([self.trade_date_col, "quantile"])
            ).otherwise(0.0)
        )
        rebalance_df = rebalance_df.sort(by=[self.trade_date_col, "quantile",self.symbol_col])
        ####################################buqi
        
        
        date_g = df.select([self.trade_date_col, "g"]).unique()

        holdings = rebalance_df.select([
            self.symbol_col,
            "g",
            "quantile"
        ])

        full = holdings.join(date_g, on="g", how="inner")

        df_small = df.select([
            self.symbol_col,
            self.trade_date_col,
            "g",
            "fut_ret_1",
            "w"
        ])

        full = full.join(
            df_small,
            on=[self.symbol_col, self.trade_date_col, "g"],
            how="left"
        )

        df = full.with_columns([
            pl.col("w").fill_null(0.0),
            pl.col("fut_ret_1").fill_null(0.0)
        ])
        ############################################
        df = df.sort([self.symbol_col, self.trade_date_col])

        df = df.with_columns(
            w0 = pl.when(pl.col("w") > 0)
                .then(pl.col("w"))
                .otherwise(None)
        )

        df = df.with_columns(
            w0 = pl.col("w0").forward_fill().over([self.symbol_col, "g"])
        )

        df = df.with_columns(
            cum_ret = (1 + pl.col("fut_ret_1")).cum_prod().over([self.symbol_col, "g"])
        )

        df = df.with_columns(
            cum_ret_lag = pl.col("cum_ret").shift(1).over([self.symbol_col, "g"])
        )

        df = df.with_columns(
            cum_ret_lag = pl.when(pl.col("cum_ret_lag").is_null())
                .then(1.0)
                .otherwise(pl.col("cum_ret_lag"))
        ).sort(["g",self.trade_date_col,"quantile", self.symbol_col])

        df = df.with_columns(
            raw_w = pl.col("w0") * pl.col("cum_ret_lag")
        )

        df = df.with_columns(
            norm = pl.col("raw_w").sum().over([self.trade_date_col, "quantile"])
        )

        df = df.with_columns(
            w = pl.col("raw_w") / pl.col("norm")
        )
        ###############################################################re
    
        g_info = (
            df.group_by("g")
            .agg([
                pl.col(self.trade_date_col).max().alias("g_last_date"),
                pl.col(self.trade_date_col).min().alias("g_first_date"),
            ])
            .sort("g_first_date")
            .with_columns(
                pl.col("g_first_date").shift(-1).alias("next_g_first_date")
            )
        )


        last_rows = df.join(
            g_info,
            on="g",
            how="inner"
        ).filter(
            pl.col(self.trade_date_col) == pl.col("g_last_date")
        ).filter(
            pl.col("next_g_first_date").is_not_null()
        )


        df_new = last_rows.with_columns([
            pl.col("next_g_first_date").alias(self.trade_date_col),
            pl.col("cum_ret").alias("cum_ret_lag"),
        ])

        df_new = df_new.with_columns(
            raw_w = pl.col("w0") * pl.col("cum_ret_lag")
        )

        df_new = df_new.with_columns(
            norm = pl.col("raw_w").sum().over([self.trade_date_col, "quantile"])
        )

        df_new = df_new.with_columns(
            w = pl.col("raw_w") / pl.col("norm")
        ).drop(["g_last_date", "g_first_date", "next_g_first_date"]).sort([ "g", self.trade_date_col,'quantile',self.symbol_col])
        
        df_new = (
            df_new
            .with_columns(
                pl.lit(None).alias("fut_ret_1")
            ))
        df = pl.concat([df, df_new]).drop(["w0","cum_ret","cum_ret_lag","raw_w","norm"])
        ##########################################################

        df_reb = df.filter(rebalance_mask)
        
        if self.tax_rate_direction == "both":
            fee = self.tax_rate + self.fee_rate + self.slippage_rate
        
        elif self.tax_rate_direction == "buy":
            fee_buy = self.tax_rate + self.fee_rate + self.slippage_rate
            fee_sell = self.fee_rate + self.slippage_rate
        
        elif self.tax_rate_direction == "sell":
            fee_buy = self.fee_rate + self.slippage_rate
            fee_sell = self.tax_rate + self.fee_rate + self.slippage_rate

        else:
            raise ValueError(f"tax_rate_direction can be set to 'buy', 'sell', or 'both', got {self.tax_rate_direction}")
        
        fee_buy = fee if self.tax_rate_direction == "both" else fee_buy
        fee_sell = fee if self.tax_rate_direction == "both" else fee_sell
        
        df_reb = df_reb.sort([self.trade_date_col, "quantile", "g", self.symbol_col])

        min_dt = df_reb.select(pl.col(self.trade_date_col).min()).item()

        df_init = df_reb.filter(
            (pl.col(self.trade_date_col) == min_dt) & (pl.col("g") == 0)
        ).with_columns([
            pl.col("w").alias("buy"),
            pl.lit(0.0).alias("sell"),
        ])
        
        df_init = df_init.with_columns([
            (pl.col("buy")).alias("turnover"),
            (pl.col("buy") * fee_buy).alias("cost")
        ])

        df_main = df_reb.filter(~(
            (pl.col(self.trade_date_col) == min_dt) & (pl.col("g") == 0)
        )).sort([ "g", self.trade_date_col,'quantile',self.symbol_col])

        g_pairs = (
            df_main.select([self.trade_date_col, "quantile", "g"])
            .unique()
            .sort([self.trade_date_col, "quantile", "g"])
            .with_columns([
                pl.col("g").shift(1).over([self.trade_date_col, "quantile"]).alias("g_prev"),
                pl.col("g").alias("g_next")
            ])
            .drop_nulls()
        )

        df_prev = df_main.select([
            self.trade_date_col, "quantile", "g", self.symbol_col, "w"
        ]).rename({
            "g": "g_prev",
            "w": "w_prev"
        })

        df_next = df_main.select([
            self.trade_date_col, "quantile", "g", self.symbol_col, "w"
        ]).rename({
            "g": "g_next",
            "w": "w_next"
        })

        df_prev = df_prev.join(
            g_pairs, on=[self.trade_date_col, "quantile", "g_prev"], how="inner"
        )

        df_next = df_next.join(
            g_pairs, on=[self.trade_date_col, "quantile", "g_next"], how="inner"
        )

        common = df_prev.join(
            df_next,
            on=[self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col],
            how="inner"
        )

        common = common.with_columns([
            (pl.col("w_next") - pl.col("w_prev")).alias("diff"),
        ])

        common = common.with_columns([
            pl.when(pl.col("diff") > 0).then(pl.col("diff")).otherwise(0).alias("buy"),
            pl.when(pl.col("diff") < 0).then(-pl.col("diff")).otherwise(0).alias("sell"),
        ])

        only_prev = df_prev.join(
            df_next.select([self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col]),
            on=[self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col],
            how="anti"
        )

        only_prev = only_prev.with_columns([
            pl.lit(0.0).alias("buy"),
            pl.col("w_prev").alias("sell"),
        ])

        only_next = df_next.join(
            df_prev.select([self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col]),
            on=[self.trade_date_col, "quantile", "g_prev", "g_next", self.symbol_col],
            how="anti"
        )

        only_next = only_next.with_columns([
            pl.col("w_next").alias("buy"),
            pl.lit(0.0).alias("sell"),
        ])

        
        df_pair = pl.concat([common, only_prev, only_next], how="diagonal")

        buy_df = df_pair.select([
            self.trade_date_col, "quantile", "g_next", self.symbol_col, "buy"
        ]).rename({"g_next": "g"})

        sell_df = df_pair.select([
            self.trade_date_col, "quantile", "g_prev", self.symbol_col, "sell"
        ]).rename({"g_prev": "g"})

        df_main = df_main.join(buy_df, on=[self.trade_date_col, "quantile", "g", self.symbol_col], how="left")
        df_main = df_main.join(sell_df, on=[self.trade_date_col, "quantile", "g", self.symbol_col], how="left")

        df_main = df_main.with_columns([
            pl.col("buy").cast(pl.Float64).fill_null(0.0).alias("buy"),
            pl.col("sell").cast(pl.Float64).fill_null(0.0).alias("sell"),
        ])

        df_main = df_main.with_columns([
            (pl.col("buy") + pl.col("sell")).alias("turnover"),
            (pl.col("buy") * fee_buy + pl.col("sell") * fee_sell).alias("cost")
        ])


        df_reb = pl.concat([df_init, df_main])
        df = df.join(
            df_reb.select([
                self.trade_date_col, "quantile", "g", self.symbol_col,
                "buy", "sell", "turnover", "cost"
            ]),
            on=[self.trade_date_col, "quantile", "g", self.symbol_col],
            how="left"
        ).sort([ "g", self.trade_date_col,'quantile',self.symbol_col])
        ################################################################

        
        df_g = df.with_columns([
            (pl.col("w") * pl.col("fut_ret_1")).alias("ret")
        ])
        #####################################################################
        df_g_1 = (
            df_g
            .group_by([self.trade_date_col, "g", "quantile"])
            .agg([
                pl.col("ret").sum().alias("ret"),
                pl.col("cost").sum().alias("cost")
            ])
        )

        dates = (
            df_g
            .select(self.trade_date_col)
            .unique()
            .sort(self.trade_date_col)
            .to_series()
        )
        rebal_dates = dates.filter(
            dates.is_in(rebalance_dates.slice(1).implode())
        )

        idx = dates.search_sorted(rebal_dates)

        prev_dates = dates.gather(idx - 1)

        target_dates = pl.concat([rebal_dates, prev_dates]).unique()

        df_g_2 = (
            df_g
            .filter(
                pl.col(self.trade_date_col).is_in(target_dates.implode()) 
            ).filter(
                pl.col("g") == pl.col("g").min().over(self.trade_date_col)
            ).with_columns(
            pl.when(pl.col(self.trade_date_col) != pl.col(self.trade_date_col).max().over("g"))
            .then(pl.col(self.trade_date_col).max().over("g"))
            .otherwise(None)
            .alias(self.trade_date_col)
        ).filter(pl.col(self.trade_date_col).is_not_null())
        )
        
       
        df_g_2 = (
            df_g_2
            .group_by([self.trade_date_col, "quantile"])
            .agg([
                pl.col("ret").sum().alias("ret")
            ])
        )
        df_g:pl.DataFrame
        df_g = df_g_1.join(df_g_2, on=[self.trade_date_col, "quantile"],how='left',suffix='_new').with_columns(
            pl.coalesce("ret_new","ret").alias("ret")
        ).drop("ret_new").sort([self.trade_date_col, "quantile"]).with_columns([
                (pl.col("ret") - pl.col("cost")).alias("ret_net"),
                (-pl.col("ret") - pl.col("cost")).alias("ret_net_s"),
                (-pl.col("ret")).alias("ret_s")
            ])
        ####################################################
        
        df_result = (
            df_g
            .drop(["g", "cost"])
            .group_by([self.trade_date_col, "quantile"])
            .agg([
                pl.col("ret").mean().alias("ret"),
                pl.col("ret_s").mean().alias("ret_s"),
                pl.col("ret_net").mean().alias("ret_net"),
                pl.col("ret_net_s").mean().alias("ret_net_s"),
            ])
            .unpivot(
                index=[self.trade_date_col, "quantile"],
                on=["ret", "ret_s", "ret_net", "ret_net_s"],
                variable_name="metric",
                value_name="value"
            )
            .with_columns([
                (
                    pl.col("metric") + "_q" + pl.col("quantile").cast(pl.Utf8)
                ).alias("metric_q")
            ])
            .pivot(
                values="value",
                index=self.trade_date_col,
                on="metric_q"
            )
            .sort(self.trade_date_col)
        )
    
        ###############################################取三个，删掉一个，但是要知道最后是哪一天
        idx = all_dates.search_sorted(date)

        if idx >= len(all_dates) or all_dates[idx] != date:
            raise ValueError("date not exists")

        n = len(all_dates)

        indices_4 = (
            pl.Series([0, 1]) if idx == 0 else
            pl.Series([n-3, n-2, n-1]) if idx == n-1 else
            pl.Series([idx-2, idx-1, idx, idx+1])
        )
        dates_4 = all_dates.gather(indices_4)
        dates_4_mask = pl.col(self.trade_date_col).is_in(
            dates_4.implode()
        )
        
        indices_3 = (
            pl.Series([0, 1]) if idx == 0 else
            pl.Series([n-2, n-1]) if idx == n-1 else
            pl.Series([idx-1, idx, idx+1])
        )
        dates_3 = all_dates.gather(indices_3)
        dates_3_mask = pl.col(self.trade_date_col).is_in(
            dates_3.implode()
        )
        
        
        ###################################################
        df_result = df_result.with_columns(
            pl.col(self.trade_date_col).shift(-1) 
        ).filter(pl.col(self.trade_date_col).is_not_null()).filter(dates_3_mask)
        
        if df_result.height == 0:
            raise
        return df.filter(dates_4_mask), df_result, dates_4, rebalance_dates
    
    def run_stats_parallel(self):
        def _dispatch(task):
            task_type, args = task

            if task_type == "ret":
                return ("ret", self.calc_stats_for_period(args))
            else:
                return ("ic", self.calc_stats_for_horizon(args))
        base_df = self.base_df.clone()
        total_tasks = len(self.rebalance_periods) + len(self.return_horizons)
        max_workers = min(total_tasks, mp.cpu_count() - 1) if self.max_workers == -1 else self.max_workers
        base_df = base_df.with_columns(pl.col(self.trade_date_col).cast(pl.Datetime))

        with tempfile.TemporaryDirectory() as tmp_dir:

            arrow_table = base_df.to_arrow()
            base_df_path = f"{tmp_dir}/base_df.arrow"
            with pa.OSFile(base_df_path, "wb") as sink:
                writer = pa.RecordBatchFileWriter(sink, arrow_table.schema)
                writer.write_table(arrow_table)
                writer.close()

            ret_args_list = [(p, base_df_path) for p in self.rebalance_periods]
            ic_args_list = [(p, base_df_path) for p in self.return_horizons]

            all_tasks = []

            for args in ret_args_list:
                all_tasks.append(("ret", args))

            for args in ic_args_list:
                all_tasks.append(("ic", args))
                
            all_results = Parallel(
                n_jobs=max_workers,
                backend="loky",
                mmap_mode="r"
            )(
                delayed(_dispatch)(task) for task in all_tasks
            )
            results = []
            ic_results = []

            for task_type, res in all_results:
                if task_type == "ret":
                    results.append(res)
                else:
                    ic_results.append(res)
            
        stats_df_ls = pd.DataFrame([{"period": p, **s} for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results])
        stats_df_l = pd.DataFrame([{"period": p, **stats_l} for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results])
        stats_df_s = pd.DataFrame([{"period": p, **stats_s} for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results])
        ls_rets = {p: ls.to_pandas() for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results}
        ls_turnovers = {p: avg_turnover_ls for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results}
        mean_rets = {p: mean_ret for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results}
        indus_rets = {p: ls_ret_i.to_pandas() for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results}
        agg_dfs = {p: df_agg.to_pandas() for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results}
        heatmap_calendar_dfs = {p: df_heatmap_calendar for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results}
        heatmap_calendar_s_dfs = {p: df_heatmap_calendar_s for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results}
        
        monthly_ls = pl.concat(
        [m.with_columns(pl.lit(p).alias("period")) for p, _, _, _, m, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results if m.height > 0]
    )
        monthly_l = pl.concat(
        [monthly_ret_l.with_columns(pl.lit(p).alias("period")) for p, _, _, _, m, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results if monthly_ret_l.height > 0]
    )
        monthly_s = pl.concat(
        [monthly_s.with_columns(pl.lit(p).alias("period")) for p, _, _, _, m, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls, df_agg, df_heatmap_calendar, df_heatmap_calendar_s in results if monthly_s.height > 0]
    )
        monthly_pivot_ls = (
        monthly_ls
        .to_pandas()
        .pivot(index="agg_date", columns="period", values="ret_net_ls_agg")
        .sort_index()
    )
        monthly_pivot_l = (
        monthly_l
        .to_pandas()
        .pivot(index="agg_date", columns="period", values="ret_net_qt_agg")
        .sort_index()
    )
        monthly_pivot_s = (
        monthly_s
        .to_pandas()
        .pivot(index="agg_date", columns="period", values="ret_net_s_qb_agg")
        .sort_index()
    )
        
        ic_dfs = {p: s.to_pandas() for p, s, mean_ic, ic_stats, mean_ic_autocorr, monthly_ic, indus_contrib in ic_results}
        mean_ics = {p: mean_ic for p, s, mean_ic, ic_stats, mean_ic_autocorr, monthly_ic, indus_contrib in ic_results}
        mean_ic_autocorrs = {p: mean_ic_autocorr for p, s, mean_ic, ic_stats, mean_ic_autocorr, monthly_ic, indus_contrib in ic_results}
        ic_panal = pd.DataFrame([{"period": p, **ic_stats} for p, s, mean_ic, ic_stats, mean_ic_autocorr, monthly_ic, indus_contrib in ic_results])
        monthly_ic =pl.concat([monthly_ic.with_columns(pl.lit(p).alias("period")) for p, s, mean_ic, ic_stats, mean_ic_autocorr, monthly_ic, indus_contrib in ic_results])
        ic_indus_contribs = {p: indus_contrib.to_pandas() for p, s, mean_ic, ic_stats, mean_ic_autocorr, monthly_ic, indus_contrib in ic_results}
        monthly_pivot_ic = (
        monthly_ic
        .to_pandas()
        .pivot(index="agg_date", columns="period", values="monthly_ic_sum")
        .sort_index()
    )
        return ls_rets, indus_rets, ls_turnovers, ic_dfs, stats_df_ls, stats_df_l, stats_df_s, ic_panal, monthly_pivot_ls.T, monthly_pivot_l.T, monthly_pivot_s.T, monthly_pivot_ic.T, mean_rets, mean_ics, mean_ic_autocorrs, ic_indus_contribs, agg_dfs, heatmap_calendar_dfs, heatmap_calendar_s_dfs
    
    def run(self):
        (   
            self.returns_dict,
            self.indus_returns_dict,
            self.ls_turnovers_dict,
            self.ics_dict,
            self.ls_stats_panel,
            self.l_stats_panel,
            self.s_stats_panel,
            self.ic_stats_panel,
            self.ls_monthly_panel,
            self.l_monthly_panel,
            self.s_monthly_panel,
            self.ic_monthly_panel,
            self.mean_returns_dict,
            self.mean_ics_dict, 
            self.mean_ic_autocorrs_dict,
            self.ic_indus_contribs_dict,
            self.agg_dfs_dict,
            self.heatmap_calendar_dfs_dict,
            self.heatmap_calendar_s_dfs_dict
            
        )  = self.run_stats_parallel() 
        
    def create_long_return_sheet(self, staticPlot:bool=False, return_fig:bool=False):
        r"""
        Generate an advanced enterprise-grade visual presentation sheet for multi-period long-only factor performance diagnostics.

        This method constructs a highly customized, vertically stacked Plotly multi-panel dashboard. 
        Unlike standard Plotly shared axes, this pipeline dynamically allocates spatial rendering budgets 
        for each metric group based on experimental rebalance cycles. It effectively resolves chart element 
        collisions by manually computing pixel heights and injecting customized localized tracking domains.


        Visual Diagnostic Panels Dispatched:
        1. Cumulative Long Returns Tier:
           - Plots sequential compound equity curves for top-quantile down to baseline-quantile buckets.
           - Dual tracking: Visualizes raw factor returns alongside multi-tier deduction net curves 
             (accounting for transaction slippage, fees, and execution stamp tax directional parameters).
           - Supports natural geometric progression or localized exponential translation via log-scaling flags.
        
        2. Asset Turnover Velocity Tier:
           - Generates cross-sectional scatter mapping of chronological turnover tracking variables 
             comparing the leading tail vs. trailing alpha portfolios.
           - Essential for assessing capacity thresholds and verifying signal decay attributes.
        
        3. Seasonal Alpha Distribution Tier:
           - Renders highly tailored distribution box plots mapped across a 12-month calendar horizon.
           - Empowers quantitative analysts to quickly spot systematic calendar anomalies, seasonal 
             macro regimes, or institutional window-dressing structural drifts.
        
        4. Industry Asset Pricing Attribution (Conditional):
           - Injected explicitly if `self.group_by` configurations are populated.
           - Maps horizontal cross-sectional ranking bars showing factor-induced alpha concentration 
             across different specific macroeconomic sectors.
        
        5. Aggregated Period Return Matrix & Descriptive Statistics Panel:
           - Heatmap layer mapping returns across sub-intervals to easily check alpha consistency over time.
           - Returns a complete summary diagnostic metric matrix tracking key performance criteria 
             (Annualized Sharp Ratio, Sortino Risk Limits, Calmar Drawdown levels, and Win Rates).

        Parameters
        ----------
        staticPlot : bool, default False
            If True, overrides default interactive rendering and loads a static asset image config layer. 
            Highly recommended for continuous integration automated reporting pipelines or static document publishing.
            If False, enables full interactive capabilities (vector panning, cross-hair tracking, and hover tools).
        
        return_fig : bool, default False
            If True, instructs the backend execution context to return the compiled `plotly.graph_objects.Figure` 
            instance back to the interpreter frame instead of instantly disposing of it post-rendering.

        Returns
        -------
        fig : plotly.graph_objects.Figure or None
            A fully structured Plotly multi-index figure container object, returned exclusively 
            if `return_fig=True` is verified.
        
        Notes
        -----
        - Subplot Figure Insulation: This method circumvents global Plotly canvas bugs where independent traces 
          bleed across disparate legends by dynamically creating virtual trace group indices mapped across 
          localized string coordinates (`legend`, `legend2`, ... `legendN`).
        """
        n_periods = len(self.rebalance_periods)
        n_rows = n_periods *4  + 2 if self.group_by else n_periods * 3 + 2
    
        specs = [[{"type": "xy"}] for _ in range(n_rows - 2)]
        specs += [[{"type": "heatmap"}], [{"type": "heatmap"}]]

        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                            vertical_spacing=0,
                            specs=specs)

        gap_px = 350              
        row_content_px = 550
        row_pixel_heights = [row_content_px] * n_rows
        total_height_px = sum(row_pixel_heights) + (n_rows - 1) * gap_px +200

        domains = []
        cursor = total_height_px
        for h in row_pixel_heights:
            top = cursor / total_height_px
            bottom = (cursor - h) / total_height_px
            domains.append([bottom, top])
            cursor -= (h + gap_px)

        for i, dom in enumerate(domains, start=1):
            axis_name = "yaxis" if i == 1 else f"yaxis{i}"
            if axis_name in fig.layout:
                fig.layout[axis_name].domain = dom
            else:
                fig.layout[axis_name] = dict(domain=dom)
        
        
        
        row = 1
        for period in self.rebalance_periods:
            df_period = self.returns_dict[period]
            date = df_period[self.trade_date_col].values
            self.add_subtitle(fig, f'Cumulative Long Returns (Rebalance Period = {period}, Log Scale = {self.log_scale})', row)
            
            for q in range(self.bins, 0, -1):
                is_top_q = (q == self.bins)
                
                
                if is_top_q:
                    fig.add_trace(
                        go.Scatter(
                            x=date,
                            y=df_period[f'cum_ret_{q}'], 
                            mode="lines",
                            name=f"Q{q} CR Before Cost",
                            line=dict(width=2, color="darkgreen", dash="dashdot"),
                            showlegend=True,
                        ),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=date,
                            y=df_period[f'cum_ret_net_{q}'], 
                            mode="lines",
                            name=f"Q{q} Net Cum. Return",
                            line=dict(width=2, color="darkgreen"),
                            showlegend=True,
                        ),
                        row=row, col=1
                    )
                elif q < self.bins:
                    fig.add_trace(
                        go.Scatter(
                            x=date,
                            y=df_period[f'cum_ret_net_{q}'], 
                            mode="lines",
                            name=f"Q{q} Net Cum. Return",
                            line=dict(width=2),
                            showlegend=True,
                        ),
                        row=row, col=1
                    )
                
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            fig.update_yaxes(title_text="Cumulative Return", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1
            
        for period in self.rebalance_periods:
            date = self.agg_dfs_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Turnover Rate (Rebalance Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.agg_dfs_dict[period][f'turnover_{self.bins}'],
                    mode="markers",
                    name=f"Q{self.bins} Turnover",
                    marker=dict(size=6),
                    line=dict(color="darkgreen", width=2),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.agg_dfs_dict[period]['turnover_1'],
                    mode="markers",
                    name="Q1 Turnover",
                    marker=dict(size=6),
                    line=dict(color="lightgreen", width=2),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            fig.update_yaxes(title_text="Turnover Ratio", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1
            
        for period in self.rebalance_periods:
            dom = domains[row - 1]
            y_center = (dom[0] + dom[1]) / 2
            
            df_calendar = self.heatmap_calendar_dfs_dict[period]
            
            z_data = df_calendar.values
            x_months = list(df_calendar.columns)
            
            self.add_subtitle(fig, f'Q-{self.bins} Seasonal Alpha Distribution (Rebalance Period = {period})', row)
            
            for m_idx, month in enumerate(x_months):
                m_data = z_data[:, m_idx] 
                if len(m_data) == 0:
                    continue
                
                fig.add_trace(
                    go.Box(
                        x=[f"{month}M"] * len(m_data),  
                        y=m_data * 100,                 
                        name=f"Q{self.bins}",
                        
                        line_color="#004d26",     
                        fillcolor="rgb(0, 60, 30)",   
                        opacity=0.85,                   
    
                        boxpoints="all",                
                        pointpos=0,                     
                        jitter=0.15,                    
                        
                        marker=dict(
                            size=4,                     
                            opacity=0.45,               
                            color="rgb(255, 215, 100)", 
                            line=dict(width=0)          
                        ),
                        
                        text=[f"Year: {y}" for y in df_calendar.index], 
                        hovertemplate="Month: %{x}<br>%{text}<br>Excess: %{y:.2f}%<extra></extra>",
                        showlegend=False              
                    ),
                    row=row, col=1
                )
            
            fig.update_xaxes(
                title_text="Month", row=row, col=1, 
                type="category", 
                title_font=dict(size=18), tickfont=dict(size=14)
            )
            fig.update_yaxes(
                title_text="Excess Return (%)", row=row, col=1, 
                tickformat=".1f%",                      
                title_font=dict(size=18), tickfont=dict(size=16)
            )
            fig.update_layout(
            boxgap=0.15,          
            boxgroupgap=0.0       
        )
            row += 1
        
        if self.group_by:
            for i ,period in enumerate(self.rebalance_periods):
                dom = domains[row - 1]
                y_center = (dom[0] + dom[1]) / 2
                df:pd.DataFrame = self.indus_returns_dict[period][['industry',"ret_net_qt_mean"]].sort_values(by='ret_net_qt_mean', ascending=True)
                self.add_subtitle(fig,f'Average Industry Q-{self.bins} Long Return (Rebalance Period = {period})',row)
                fig.add_trace(
                    go.Bar(
                        x=df["ret_net_qt_mean"],
                        y=df["industry"],
                        orientation='h',
                        marker=dict(
                            color=df["ret_net_qt_mean"],
                            colorscale="RdYlGn",
                    showscale=True, 
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=y_center,
                        len=(1 / n_rows)*0.7,
                        x=1.04,
                        outlinewidth=0,
                        titlefont=dict(size=16, family="Arial", color="black"), 
                        tickfont=dict(size=14, family="Arial", color="black"))
                        ),
                        showlegend=False,
                        ),
                    row=row, col=1
                )
                
                font_size = min(12, max(7, int( 150 / len(df))))
                fig.update_xaxes(title_text="Mean Return", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
                fig.update_yaxes(title_text="Industry", type="category", row=row, col=1, tickfont=dict(size=font_size), title_font=dict(size=18))
                row += 1
                

        try:
            monthly_panel = self.l_monthly_panel
            if monthly_panel is None or getattr(monthly_panel, "shape", (0,))[0] == 0:
                raise ValueError("l_monthly_panel empty")
            self.add_subtitle(fig,f'Aggregated Q-{self.bins} Long Return',row,y=1.1)
            z = monthly_panel.values
            x = list(monthly_panel.columns)
            y = list(monthly_panel.index)
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(
                        title="Return",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (row - 0.55) / n_rows,
                        len=(1 / n_rows)*0.7,
                        x=1.04),
                    text=np.round(z, 4),
                    texttemplate="%{text:.2%}",
                    hovertemplate="Period: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text=f"Per {self.freq}", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=8))
            fig.update_yaxes(title_text="Rebalance Period", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=16))
        except Exception:
            fig.add_annotation(text="No monthly panel available", row=row, col=1, showarrow=False)
        row += 1

        try:
            stats_df = self.l_stats_panel.set_index("period").T
            self.add_subtitle(fig,f'Q-{self.bins} Long Return Statistics',row,y=1.1)
            z_stats = stats_df.T.values
            z_stats = np.asarray(z_stats, dtype=np.float64)
            z_stats = np.nan_to_num(z_stats, nan=np.nan)
            text = np.round(z_stats, 4)
            x_stats = list(stats_df.index)
            y_stats = list(stats_df.columns)
            fig.add_trace(
                go.Heatmap(
                    z=z_stats,
                    x=x_stats,
                    y=y_stats,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (row - 0.55) / n_rows,  
                        len=(1 / n_rows)*0.7,                 
                        x=1.04                           
                    ),
                    text=text,
                    texttemplate="%{text}",
                    hovertemplate="Metric: %{x}<br>Period: %{y}<br>Value: %{z:.4f}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text="Metrics", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=9))
            fig.update_yaxes(title_text="Rebalance Period", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=16))
        except Exception as e:
            print(e)
            fig.add_annotation(text="No stats panel available", row=row, col=1, showarrow=False)
        row += 1
        height_per_row = 550
        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=120, b=80, l=80, r=200),
            title=dict(
                text=f"Long Return Sheet ({self.factor_name})",   
                font=dict(size=24, family="Arial", color="black")
            )
        )

        legend_name_for_row = lambda r: "legend" if r == 1 else f"legend{r}"
        traces_per_row = []
        for _ in range(n_periods):
            traces_per_row.append(self.bins+1)
        for _ in range(n_periods):
            traces_per_row.append(2)
        if self.group_by:
            for _ in range(n_periods):
                traces_per_row.append(1)
        for _ in range(n_periods):
            traces_per_row.append(2)

        tidx = 0
        for r_idx, cnt in enumerate(traces_per_row, start=1):
            legend_key = legend_name_for_row(r_idx)
            for _ in range(cnt):
                if tidx < len(fig.data):
                    fig.data[tidx].update(legend=legend_key)
                tidx += 1
            
        legend_layouts = {}
        for i in range(1, n_rows + 1):
            key = "legend" if i == 1 else f"legend{i}"
            y_fixed = 1 - (i - 0.85) / n_rows
            legend_layouts[key] = dict(
                x=1.03,
                y=y_fixed,
                xanchor="left",
                yanchor="middle",
                orientation="v",
                tracegroupgap=6,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0 ,
                font=dict(size=16, family="Arial", color="black")
            )
        layout_updates = base_layout.copy()
        layout_updates.update(legend_layouts)
        fig.update_layout(**layout_updates)
        if staticPlot:
            fig.show(
                config={
                    "staticPlot": True, 
                    "responsive": True
                }
            )
        else:
            fig.show(config={"responsive": True})
        if return_fig:
            return fig
    
    def create_long_short_return_sheet(self, staticPlot:bool=False, return_fig:bool=False):
        r"""
        Generate a comprehensive, high-dimensional multi-panel validation sheet for Long-Short spread alpha factor performance.

        This method orchestrates a complex, vertically stacked hierarchical visualization system utilizing Plotly templates. 
        It isolates and quantifies pure factor alpha spreads across disparate cross-sectional testing dimensions, 
        evaluating signal decay properties, empirical structural stability, trading frictions, and localized 
        macroeconomic market regime shifts.

        Mathematical & Spatial Allocation Schema
        ----------------------------------------
        Independent Legend Mapping Matrix:
        To resolve legend crowding and overlapping cross-panel elements in standard Plotly subplots, 
        this algorithm automatically intercepts the underlying graph structure trace elements. It establishes 
        discrete trace-to-subplot groups using explicit row indexing counters (`legend`, `legend2`, ... `legendN`) 
        to ensure visual boundaries between row tiers are clean and easy to scan.

        Visual Diagnostic Subplots Rendered:
        1. Cumulative Pure Alpha Spread Tier:
           - Constructs simultaneous compound time-series curves capturing gross portfolio alpha equity 
             as well as realistic deduction net-of-cost tracking curves.
           - Crucial for identifying edge decay, tracking asymmetric structural market regimes, 
             and testing structural signal persistence under customizable geometric scaling regimes.
        
        2. Portfolio Turnover Velocity & Cost Frictions Tier:
           - Maps discrete cross-sectional trading turnover observations against a centralized mathematical 
             mean convergence line.
           - Annotates absolute average volume parameters directly into the plot space, allowing 
             for rapid execution capacity verification and leverage scaling evaluations.
        
        3. Industry-Neutralized Attribution Bars (Conditional):
           - Deployed exclusively if conditional factor industry configurations are populated.
           - Maps horizontal cross-sectional ranking bar clusters to visualize sector-level alpha 
             concentration and ensure factor returns aren't secretly just accidental bets on a single industry.
        
        4. Cross-Sectional Return Matrix & Performance Descriptive Metrics:
           - Renders localized calendar or frequency matrices mapping periodic sub-interval net value drift.
           - Returns a full evaluation matrix of annualized portfolio returns, Standard Deviations, Sharpe ratios, 
             Sortino downside risk parameters, Calmar ratios, and Win Rate probabilities.

        Parameters
        ----------
        staticPlot : bool, default False
            If True, overrides reactive UI callbacks and delivers a flattened static vector graph layer. 
            Excellent for cron-tab automated reporting engines or publishing print-ready documentation.
            If False, retains full dynamic capabilities including vector panning, canvas zoom, and localized trace hover tools.
        
        return_fig : bool, default False
            If True, instructs the backend runtime context to yield the fully rendered `plotly.graph_objects.Figure` 
            data structure back to the execution stack for downstream modifications.

        Returns
        -------
        fig : plotly.graph_objects.Figure or None
            A highly optimized, multi-index Plotly visualization graph container object, 
            returned exclusively if `return_fig=True` is verified.
        """
        n_periods = len(self.rebalance_periods)
        n_rows = n_periods *3  + 2 if self.group_by else n_periods * 2 + 2
        
        specs = [[{"type": "xy"}] for _ in range(n_rows - 2)]
        specs += [[{"type": "heatmap"}], [{"type": "heatmap"}]]

        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                            vertical_spacing=0,
                            specs=specs)
        gap_px = 350              
        row_content_px = 550
        row_pixel_heights = [row_content_px] * n_rows
        total_height_px = sum(row_pixel_heights) + (n_rows - 1) * gap_px +200

        domains = []
        cursor = total_height_px
        for h in row_pixel_heights:
            top = cursor / total_height_px
            bottom = (cursor - h) / total_height_px
            domains.append([bottom, top])
            cursor -= (h + gap_px)

        for i, dom in enumerate(domains, start=1):
            axis_name = "yaxis" if i == 1 else f"yaxis{i}"
            if axis_name in fig.layout:
                fig.layout[axis_name].domain = dom
            else:
                fig.layout[axis_name] = dict(domain=dom)

        line_width = 2
        row = 1
        for period in self.rebalance_periods:
            date = self.returns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Cumulative Long-Short Return (Rebalance Period = {period}, Log Scale = {self.log_scale})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.returns_dict[period]['cum_ret_ls'],
                    mode="lines",
                    name="Return Before Cost",
                    line=dict(width=line_width, color='darkgreen', dash="dot"),
                    showlegend=True,
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.returns_dict[period]['cum_ret_net_ls'],
                    mode="lines",
                    name="Cumulative LS Return",
                    line=dict(width=line_width, color='darkgreen'),
                    showlegend=True,
                ),
                row=row, col=1
            )
            
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            fig.update_yaxes(title_text="Cumulative Return", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1
        if self.group_by:
            for i ,period in enumerate(self.rebalance_periods):
                dom = domains[row - 1]
                y_center = (dom[0] + dom[1]) / 2
                df:pd.DataFrame = self.indus_returns_dict[period][['industry',"ret_net_ls_i_mean"]].sort_values(by='ret_net_ls_i_mean', ascending=True)
                self.add_subtitle(fig,f'Average Industry Long–Short Return (Rebalance Period = {period})',row)
                fig.add_trace(
                    go.Bar(
                        x=df["ret_net_ls_i_mean"],
                        y=df["industry"],
                        orientation='h',
                        marker=dict(
                            color=df["ret_net_ls_i_mean"],
                            colorscale="RdYlGn",
                    showscale=True, 
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=y_center,
                        len=(1 / n_rows)*0.7,
                        x=1.04,
                        outlinewidth=0,
                        titlefont=dict(size=16, family="Arial", color="black"), 
                        tickfont=dict(size=14, family="Arial", color="black"))
                        ),
                        showlegend=False,
                        ),
                    row=row, col=1
                )
                
                font_size = min(12, max(7, int( 150 / len(df))))
                fig.update_xaxes(title_text="Mean Return", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
                fig.update_yaxes(title_text="Industry", type="category", row=row, col=1, tickfont=dict(size=font_size), title_font=dict(size=18))
                row += 1
        for period in self.rebalance_periods:
            date = self.agg_dfs_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Turnover Rate (Rebalance Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.agg_dfs_dict[period]['turnover_ls'],
                    mode="markers",
                    name="LS Portfolio Turnover",
                    marker=dict(size=6),
                    line=dict(color='darkgreen', width=line_width),
                    showlegend=True,
                ),
                row=row, col=1
            )
            mean_turnover = self.ls_turnovers_dict[period]
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=[mean_turnover] * len(date),
                    mode="lines",
                    name=f"Mean Turnover",
                    line=dict(color="red", width=3, dash="dot"),
                    showlegend=True
                ),
                row=row, col=1
            )
            fig.add_annotation(x=0.99, y=0.99, text=f"Mean = {mean_turnover:.4f}", 
                               showarrow=False, 
                               font=dict(color="red", size=18), 
                               xanchor="right",
                               yanchor="top",
                               xref=f"x{row if row > 1 else ''} domain",
                               yref=f"y{row if row > 1 else ''} domain")
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            fig.update_yaxes(title_text="Turnover Ratio", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1

        try:
            monthly_panel:pd.DataFrame = self.ls_monthly_panel
            if monthly_panel is None or getattr(monthly_panel, "shape", (0,))[0] == 0:
                raise ValueError("l_monthly_panel empty")
            self.add_subtitle(fig,'Aggregated Long-Short Return',row,y=1.1)
            z = monthly_panel.values
            x = list(monthly_panel.columns)
            y = list(monthly_panel.index)
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="RdYlGn",
                    showscale=True,  
                    colorbar=dict(
                        title="Return",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (row - 0.55) / n_rows, 
                        len=(1 / n_rows)*0.7,
                        x=1.04),
                    text=np.round(z, 4),
                    texttemplate="%{text:.2%}",
                    hovertemplate="Period: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text=f"Per {self.freq}", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=8))
            fig.update_yaxes(title_text="Rebalance Period", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=16))
        except Exception:
            fig.add_annotation(text="No monthly panel available", row=row, col=1, showarrow=False)
        row += 1

        try:
            stats_df = self.ls_stats_panel.set_index("period").T
            self.add_subtitle(fig,'Long-Short Return Statistics',row,y=1.1)
            z_stats = stats_df.T.values
            z_stats = np.asarray(z_stats, dtype=np.float64)
            z_stats = np.nan_to_num(z_stats, nan=np.nan)
            text = np.round(z_stats, 4)
            x_stats = list(stats_df.index)
            y_stats = list(stats_df.columns)
            fig.add_trace(
                go.Heatmap(
                    z=z_stats,
                    x=x_stats,
                    y=y_stats,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (row - 0.55) / n_rows,  
                        len=(1 / n_rows)*0.7,                 
                        x=1.04                         
                    ),
                    text=text,
                    texttemplate="%{text}",
                    hovertemplate="Metric: %{x}<br>Period: %{y}<br>Value: %{z:.4f}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text="Metrics", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=9))
            fig.update_yaxes(title_text="Rebalance Period", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=16))
        except Exception:
            fig.add_annotation(text="No stats panel available", row=row, col=1, showarrow=False)

        row += 1
        height_per_row = 550
        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=120, b=80, l=80, r=200),
            title=dict(
                text=f"Long-Short Return Sheet ({self.factor_name})" ,
                font=dict(size=24, family="Arial", color="black")
            )
        )
        

        legend_name_for_row = lambda r: "legend" if r == 1 else f"legend{r}"
        traces_per_row = []
        for _ in range(n_periods):
            traces_per_row.append(2)
        if self.group_by:
            for _ in range(n_periods):
                traces_per_row.append(1)
        for _ in range(n_periods):
            traces_per_row.append(2)

        tidx = 0
        for r_idx, cnt in enumerate(traces_per_row, start=1):
            legend_key = legend_name_for_row(r_idx)
            for _ in range(cnt):
                if tidx < len(fig.data):
                    fig.data[tidx].update(legend=legend_key)
                tidx += 1
            
        legend_layouts = {}
        for i in range(1, n_rows + 1):
            key = "legend" if i == 1 else f"legend{i}"
            y_fixed = 1 - (i - 0.85) / n_rows
            legend_layouts[key] = dict(
                x=1.03,
                y=y_fixed,
                xanchor="left",
                yanchor="middle",
                orientation="v",
                tracegroupgap=6,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0 ,
                font=dict(size=16, family="Arial", color="black")
            )
        layout_updates = base_layout.copy()
        layout_updates.update(legend_layouts)
        fig.update_layout(**layout_updates)
        if staticPlot:
            fig.show(
                config={
                    "staticPlot": True, 
                    "responsive": True
                }
            )
        else:
            fig.show(config={"responsive": True})
        if return_fig:
            return fig
           
    def create_single_fac_ic_sheet(self, staticPlot:bool=False, return_fig:bool=False):
        r"""
        Generate an enterprise-grade multi-panel diagnostic sheet for systematic Information Coefficient (IC) analysis.

        This method builds an advanced graphical analytics dashboard to evaluate the cross-sectional 
        predictive power and statistical persistence of an alpha factor across multiple forecasting horizons. 
        It dynamically adapts to standard Pearson IC or Spearman Rank IC profiles (`self.rank_ic`), orchestrating 
        time-series stability tracking, asset industry attribution, temporal aggregation profiles, and empirical 
        distribution deviations.

        Mathematical & Spatial Allocation Schema
        ----------------------------------------
        Subplot Legend Resolution Matrix:
        To bypass canvas tracing collisions where individual plots share or overwrite generic legend domains, 
        this routine captures the underlying graph stream and force-groups discrete geometric traces into isolated 
        row indexes (`legend`, `legend2`, ... `legendN`), keeping the right-hand panel organized and readable.

        Visual Diagnostic Subplots Rendered:
        1. Historical IC Trajectory & Signal Momentum:
           - Plots empirical periodic IC realizations paired with an aligned localized rolling moving average 
             (`self.horizon_rolling_period`).
           - Renders a horizontal mathematical mean benchmark annotation layer for quick factor decay validation.
        
        2. Cumulative IC Consistency Profile:
           - Computes and graphs the chronological cumulative integration curve ($\sum \text{IC}_t$).
           - Used to quickly identify structural regime shifts, regime drift variance, or periods of localized alpha decay.
        
        3. Sector Alpha Attribution (Conditional):
           - Deployed exclusively if specialized factor group dictionaries are loaded.
           - Maps horizontal cross-sectional ranking bar segments showing true information transmission performance 
             inside distinct industry buckets.
        
        4. IC Signal Autocorrelation & Memory Decay Tier:
           - Scatter-maps factor rank self-correlation across lagged windows to quantify memory structure 
             and decay boundaries.
        
        5. Empirical Q-Q Gaussian Deviation Diagnostic:
           - Pairs empirical sample quantiles against standard theoretical Gaussian normal structures ($Z$-scores).
           - Empowers quantitative researchers to instantly spot structural skewness, tail dependencies, 
             or systemic volatility burst probabilities hidden in the alpha engine.
        
        6. Intertemporal Correlation Matrix & Descriptive Statistics Heatmaps:
           - Generates localized frequency heatmaps mapping aggregated rolling multi-interval correlation performance.
           - Outputs an adjacent diagnostic summary matrix detailing strict empirical performance targets: 
             Factor Mean IC, variance standard deviation, Skewness, Kurtosis distribution scales, and 
             one-sample Student's t-test verification outcomes ($t$-statistic and $p$-value).

        Parameters
        ----------
        staticPlot : bool, default False
            If True, silences downstream JavaScript canvas workers and loads a optimized vector rendering layer. 
            Crucial for automated PDF document generation, pipeline testing, or static web interface compilation.
            If False, enables full canvas reactivity, including cross-hair tooltips, panning, and group zoom triggers.
        
        return_fig : bool, default False
            If True, prevents immediate garbage collection of the graph workspace and returns the underlying 
            `plotly.graph_objects.Figure` pointer to the calling application frame.

        Returns
        -------
        fig : plotly.graph_objects.Figure or None
            A highly optimized, standalone multi-index Plotly graph workspace object, 
            returned exclusively if `return_fig=True` is verified.
        """
        key_ic = 'rank_ic' if self.rank_ic else 'ic'
        KEY_IC = 'Rank IC' if self.rank_ic else 'IC'
        
        n_periods = len(self.return_horizons)
        n_rows = n_periods * 5 + 2 if self.group_by else n_periods * 4 + 2
        
        specs = [[{"type": "xy"}] for _ in range(n_rows - 2)]
        specs += [[{"type": "heatmap"}], [{"type": "heatmap"}]]

        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                            vertical_spacing=0,
                            specs=specs)
        gap_px = 350              
        row_content_px = 550
        row_pixel_heights = [row_content_px] * n_rows
        total_height_px = sum(row_pixel_heights) + (n_rows - 1) * gap_px +200

        domains = []
        cursor = total_height_px
        for h in row_pixel_heights:
            top = cursor / total_height_px
            bottom = (cursor - h) / total_height_px
            domains.append([bottom, top])
            cursor -= (h + gap_px)

        for i, dom in enumerate(domains, start=1):
            axis_name = "yaxis" if i == 1 else f"yaxis{i}"
            if axis_name in fig.layout:
                fig.layout[axis_name].domain = dom
            else:
                fig.layout[axis_name] = dict(domain=dom)


        line_width = 2
        row = 1
        for period in self.return_horizons:
            date = self.ics_dict[period][self.trade_date_col].values
            
            self.add_subtitle(fig,f'{KEY_IC} (Horizon Period = {period})',row)
            
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.ics_dict[period][key_ic],
                    mode="lines",
                    name=f'{KEY_IC}',
                    line=dict(width=line_width, color='lightgreen'),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.ics_dict[period][key_ic + '_rolling'],
                    mode="lines",
                    name=f"Rolling {self.horizon_rolling_period} Mean",
                    line=dict(width=line_width, color='darkgreen'),
                    showlegend=True,
                ),
                row=row, col=1
            )
            mean_ic = self.mean_ics_dict[period]
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=[mean_ic] * len(date),
                    mode="lines",
                    name=f"Mean {KEY_IC}",
                    line=dict(color="red", width=3, dash="dot"),
                    showlegend=True,
                ),
                row=row, col=1
            )
        
            fig.add_annotation(x=0.99, y=0.99, text=f"Mean = {mean_ic:.4f}", 
                               showarrow=False, 
                               font=dict(color="red", size=18), 
                               xanchor="right",
                               yanchor="top",
                               xref=f"x{row if row > 1 else ''} domain",
                               yref=f"y{row if row > 1 else ''} domain")
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            fig.update_yaxes(title_text=KEY_IC, row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1
        for period in self.return_horizons:
            date = self.ics_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Cumulative {KEY_IC} (Horizon Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.ics_dict[period][key_ic + '_cum'],
                    mode="lines",
                    name=f"Cumulative {KEY_IC}",
                    line=dict(width=line_width, color='darkgreen'),
                    showlegend=True,
                ),
                row=row, col=1
            )
            
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            fig.update_yaxes(title_text=KEY_IC, row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1
        if self.group_by:
            for i ,period in enumerate(self.return_horizons):
                dom = domains[row - 1]
                y_center = (dom[0] + dom[1]) / 2
                df:pd.DataFrame = self.ic_indus_contribs_dict[period].sort_values(by='contrib', ascending=True)
                self.add_subtitle(fig,f'Contributions of Industries (Horizon Period = {period})',row)
                fig.add_trace(
                    go.Bar(
                        x=df["contrib"],
                        y=df["industry"],
                        orientation='h',
                        marker=dict(
                            color=df["contrib"],
                            colorscale="RdYlGn",
                    showscale=True, 
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=y_center,
                        len=(1 / n_rows)*0.7,
                        x=1.04,
                        outlinewidth=0,
                        titlefont=dict(size=16, family="Arial", color="black"), 
                        tickfont=dict(size=14, family="Arial", color="black"))
                        ),
                        showlegend=False,
                        ),
                    row=row, col=1
                )
                
                font_size = min(12, max(7, int( 150 / len(df))))
                fig.update_xaxes(title_text=KEY_IC, row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
                fig.update_yaxes(title_text="Industry", type="category", row=row, col=1, tickfont=dict(size=font_size), title_font=dict(size=18))
                row += 1
            
        for period in self.return_horizons:
            date = self.ics_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'{KEY_IC} Self-Correlation (Horizon Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.ics_dict[period]['autocorr'],
                    mode="markers",
                    name="Self-Correlation",
                    marker=dict(size=6),
                    line=dict(color='darkgreen', width=line_width),
                    showlegend=True,
                ),
                row=row, col=1
            )
            mean_turnover = self.mean_ic_autocorrs_dict[period]
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=[mean_turnover] * len(date),
                    mode="lines",
                    name=f"Mean Self-Corr",
                    line=dict(color="red", width=3, dash="dot"),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_annotation(x=0.99, y=0.99, text=f"Mean = {mean_turnover:.4f}", 
                               showarrow=False, 
                               font=dict(color="red", size=18), 
                               xanchor="right",
                               yanchor="top",
                               xref=f"x{row if row > 1 else ''} domain",
                               yref=f"y{row if row > 1 else ''} domain")
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            fig.update_yaxes(title_text="Self-Correlation", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1
            
        for period in self.return_horizons:
            self.add_subtitle(fig,f'{KEY_IC} Q-Q Gragh (Horizon Period = {period})',row)
            ic_series = self.ics_dict[period][key_ic]
            ic_sorted = np.sort(ic_series)
            n = len(ic_sorted)
            theoretical_q = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
            sample_q = (ic_sorted - ic_sorted.mean()) / ic_sorted.std(ddof=1)

            fig.add_trace(
                go.Scatter(
                    x=theoretical_q,
                    y=sample_q,
                    mode="markers",
                    marker=dict(color="darkgreen", size=6),
                    name=KEY_IC,
                    showlegend=True
                ),
                row=row, col=1
            )
            min_q = min(theoretical_q.min(), sample_q.min())
            max_q = max(theoretical_q.max(), sample_q.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_q, max_q],
                    y=[min_q, max_q],
                    mode="lines",
                    line=dict(color="red", dash="dot", width=3),
                    name ='Base Line',
                    showlegend=True
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text="Theoretical Quantiles", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            fig.update_yaxes(title_text="Sample Quantiles", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))

            row += 1            
        try:
            monthly_panel:pd.DataFrame = self.ic_monthly_panel
            if monthly_panel is None or getattr(monthly_panel, "shape", (0,))[0] == 0:
                raise ValueError("l_monthly_panel empty")
            self.add_subtitle(fig,f'Aggregated {KEY_IC}',row,y=1.1)
            z = monthly_panel.values
            x = list(monthly_panel.columns)
            y = list(monthly_panel.index)
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (row - 0.55) / n_rows ,
                        len=(1 / n_rows)*0.7,
                        x=1.04,
                        titlefont=dict(size=16, family="Arial", color="black"), 
                        tickfont=dict(size=14, family="Arial", color="black")),
                    text=np.round(z, 4),
                    texttemplate="%{text}",
                    hovertemplate="Period: %{y}<br>Per " + self.freq + ": %{x}<br>Return: %{z}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text=f"Per {self.freq}", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=8))
            fig.update_yaxes(title_text="Horizon Period", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=16))
        except Exception:
            fig.add_annotation(text="No monthly panel available", row=row, col=1, showarrow=False)
        row += 1

        try:
            self.add_subtitle(fig,f'{KEY_IC} Statistics',row,y=1.1)
            stats_df = self.ic_stats_panel.set_index("period").T
            z_stats = stats_df.T.values
            x_stats = list(stats_df.index)
            y_stats = list(stats_df.columns)
            fig.add_trace(
                go.Heatmap(
                    z=z_stats,
                    x=x_stats,
                    y=y_stats,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (row - 0.55) / n_rows, 
                        len=(1 / n_rows)*0.7,                 
                        x=1.04,
                        titlefont=dict(size=16, family="Arial", color="black"),
                        tickfont=dict(size=14, family="Arial", color="black")
                    ),
                    text=np.round(z_stats, 4),
                    texttemplate="%{text}",
                    hovertemplate="Period: %{y}<br>Per " + self.freq + ": %{x}<br>Return: %{z}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text="Metrics", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            fig.update_yaxes(title_text="Horizon Period", row=row, col=1, type="category", title_font=dict(size=16), tickfont=dict(size=18))
        except Exception:
            fig.add_annotation(text="No stats panel available", row=row, col=1, showarrow=False)
        row += 1
        height_per_row = 550
        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=120, b=80, l=80, r=200),
            title=dict(
                text=f"{KEY_IC} Sheet ({self.factor_name})",
                font=dict(size=24, family="Arial", color="black")
            )
        )
        legend_name_for_row = lambda r: "legend" if r == 1 else f"legend{r}"
        traces_per_row = []
        for _ in range(n_periods):
            traces_per_row.append(3)
        for _ in range(n_periods):
            traces_per_row.append(1)
        if self.group_by:
            for _ in range(n_periods):
                traces_per_row.append(1)
        for _ in range(n_periods):
            traces_per_row.append(2)
        for _ in range(n_periods):
            traces_per_row.append(2)
        
        tidx = 0
        for r_idx, cnt in enumerate(traces_per_row, start=1):
            legend_key = legend_name_for_row(r_idx)
            for _ in range(cnt):
                if tidx < len(fig.data):
                    fig.data[tidx].update(legend=legend_key)
                tidx += 1

        legend_layouts = {}
        for i in range(1, n_rows + 1):
            key = "legend" if i == 1 else f"legend{i}"
            y_fixed = 1 - (i - 0.85) / n_rows
            legend_layouts[key] = dict(
                x=1.03,
                y=y_fixed,
                xanchor="left",
                yanchor="middle",
                orientation="v",
                tracegroupgap=6,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0 ,
                font=dict(size=16, family="Arial", color="black")
            )
        layout_updates = base_layout.copy()
        layout_updates.update(legend_layouts)
        fig.update_layout(**layout_updates)
        if staticPlot:
            fig.show(
                config={
                    "staticPlot": True, 
                    "responsive": True
                }
            )
        else:
            fig.show(config={"responsive": True})
        if return_fig:
            return fig
    
    def create_short_return_sheet(self, staticPlot:bool=False, return_fig:bool=False):
        r"""
        Generate a comprehensive, high-performance visual presentation sheet for multi-period short-only leg factor diagnostics.

        This method builds an advanced, vertically stacked Plotly analytical pipeline dedicated to isolating 
        and validating the performance of the short leg (the trailing cross-sectional quantile, typically Q1) 
        of a factor strategy. It treats the bottom quantile inverted as an independent alpha source, tracking 
        its temporal risk-return velocity, localized industry capacity, seasonal alpha decay vectors, and 
        transaction cost breakdown under realistic funding or capital constraints.

        Layout & Grid Architecture Specification
        ----------------------------------------
        Subplot Legend Resolution Matrix:
        To protect the chart canvas against coordinate crowding and trace bleed within identical axis grids, 
        the compiler loops through the underlying trace array and maps discrete geometric streams into standalone 
        subplot rows (`legend`, `legend2`, ... `legendN`). This enforces crisp group categorization across the 
        right-side unified rendering margin.

        Visual Diagnostic Subplots Rendered:
        1. Inverted Cumulative Short Returns Tier:
           - Plots cumulative raw returns alongside multi-tier deduction net asset curves for short-quantile portfolios.
           - Maps short leg returns relative to trading execution slippage, execution commissions, and short-direction 
             stamp taxes across customizable geometric scaling grids (`self.log_scale`).
        
        2. Short Leg Asset Turnover Velocity Tier:
           - Indispensable for evaluating short-selling liquidity constraints and estimating execution alpha degradation.
        
        3. Seasonal Short-Alpha Distribution Tier:
           - Dispatches customized geometric box-whisker plot lines segmented over a 12-month calendar horizon.
           - Empowers alternative alpha researchers to inspect seasonal systematic short patterns, capital squeezes, 
             or institutional window-dressing behavior.
        
        4. Sector Short Attribution Matrix (Conditional):
           - Injected explicitly if factor-to-sector map parameters are initialized.
           - Generates cross-sectional horizontal bars demonstrating alpha extraction success specifically 
             within short-quantile sector selections.
        
        5. Intertemporal Performance Summary Heatmaps:
           - Renders specialized sub-interval panels detailing sub-period rolling short return drift.
           - Outputs an adjacent descriptive metric matrix charting exact strategy criteria (Annualized Returns, 
             Standard Volatility, Sharpe limits, downside Sortino metrics, Calmar drawdown parameters, and Win Rate).

        Parameters
        ----------
        staticPlot : bool, default False
            If True, silences reactive UI callbacks and compiles a static vector layer layout. 
            Excellent for automated reporting scripts, continuous snapshot storage, or documentation printing.
            If False, retains full live chart features including cross-hair labels, coordinate panning, and individual box zoom.
        
        return_fig : bool, default False
            If True, instructs the backend execution stack to return the compiled `plotly.graph_objects.Figure` 
            data structure instead of instant disposal.

        Returns
        -------
        fig : plotly.graph_objects.Figure or None
            A highly optimized Plotly subplot layout graph container object, returned exclusively 
            if `return_fig=True` is verified.
        """
        n_periods = len(self.rebalance_periods)
        n_rows = n_periods *4  + 2 if self.group_by else n_periods * 3 + 2

        specs = [[{"type": "xy"}] for _ in range(n_rows - 2)]
        specs += [[{"type": "heatmap"}], [{"type": "heatmap"}]]

        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                            vertical_spacing=0,
                            specs=specs)
        gap_px = 350              
        row_content_px = 550
        row_pixel_heights = [row_content_px] * n_rows
        total_height_px = sum(row_pixel_heights) + (n_rows - 1) * gap_px +200

        domains = []
        cursor = total_height_px
        for h in row_pixel_heights:
            top = cursor / total_height_px
            bottom = (cursor - h) / total_height_px
            domains.append([bottom, top])
            cursor -= (h + gap_px)

        for i, dom in enumerate(domains, start=1):
            axis_name = "yaxis" if i == 1 else f"yaxis{i}"
            if axis_name in fig.layout:
                fig.layout[axis_name].domain = dom
            else:
                fig.layout[axis_name] = dict(domain=dom)
                
        row = 1
        for period in self.rebalance_periods:
            df_period = self.returns_dict[period]
            date = df_period[self.trade_date_col].values
            self.add_subtitle(fig, f'Cumulative Short Returns (Rebalance Period = {period}, Log Scale = {self.log_scale})', row)
            
            for q in range(1,self.bins+1):
                is_bottom_q = (q == 1)
                
                
                if is_bottom_q:
                    fig.add_trace(
                        go.Scatter(
                            x=date,
                            y=df_period[f'cum_ret_s_{q}'], 
                            mode="lines",
                            name=f"Q{q} CR Before Cost",
                            line=dict(width=2, color="darkgreen", dash="dashdot"),
                            showlegend=True,
                        ),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=date,
                            y=df_period[f'cum_ret_net_s_{q}'], 
                            mode="lines",
                            name=f"Q{q} Net Cum. Return",
                            line=dict(width=2, color="darkgreen"),
                            showlegend=True,
                        ),
                        row=row, col=1
                    )
                elif q > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=date,
                            y=df_period[f'cum_ret_net_s_{q}'], 
                            mode="lines",
                            name=f"Q{q} Net Cum. Return",
                            line=dict(width=2),
                            showlegend=True,
                        ),
                        row=row, col=1
                    )
                
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            fig.update_yaxes(title_text="Cumulative Return", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1
            
        for period in self.rebalance_periods:
            date = self.agg_dfs_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Turnover Rate (Rebalance Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.agg_dfs_dict[period]['turnover_1'],
                    mode="markers",
                    name="Q-1 Turnover",
                    marker=dict(size=6),
                    line=dict(color="darkgreen", width=2),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.agg_dfs_dict[period][f'turnover_{self.bins}'],
                    mode="markers",
                    name=f"Q-{self.bins} Turnover",
                    marker=dict(size=6),
                    line=dict(color="lightgreen", width=2),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            fig.update_yaxes(title_text="Turnover Ratio", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1

        for period in self.rebalance_periods:
            dom = domains[row - 1]
            y_center = (dom[0] + dom[1]) / 2
            
            df_calendar = self.heatmap_calendar_s_dfs_dict[period]
            
            z_data = df_calendar.values
            x_months = list(df_calendar.columns)
            
            self.add_subtitle(fig, f'Q-1 Seasonal Alpha Distribution (Rebalance Period = {period})', row)
            
            for m_idx, month in enumerate(x_months):
                m_data = z_data[:, m_idx] 
                if len(m_data) == 0:
                    continue
                
                fig.add_trace(
                    go.Box(
                        x=[f"{month}M"] * len(m_data),  
                        y=m_data * 100,                 
                        name=f"Q1",
                        
                        line_color="#004d26",     
                        fillcolor="rgb(0, 60, 30)",   
                        opacity=0.85,                   
    
                        boxpoints="all",                
                        pointpos=0,                     
                        jitter=0.15,                    
                        
                        marker=dict(
                            size=4,                     
                            opacity=0.45,               
                            color="rgb(255, 215, 100)", 
                            line=dict(width=0)          
                        ),
                        
                        text=[f"Year: {y}" for y in df_calendar.index], 
                        hovertemplate="Month: %{x}<br>%{text}<br>Excess: %{y:.2f}%<extra></extra>",
                        showlegend=False              
                    ),
                    row=row, col=1
                )
            
            fig.update_xaxes(
                title_text="Month", row=row, col=1, 
                type="category", 
                title_font=dict(size=18), tickfont=dict(size=14)
            )
            fig.update_yaxes(
                title_text="Excess Return (%)", row=row, col=1, 
                tickformat=".1f%",                      
                title_font=dict(size=18), tickfont=dict(size=16)
            )
            fig.update_layout(
            boxgap=0.15,          
            boxgroupgap=0.0       
        )
            row += 1
        
        if self.group_by:
            for i ,period in enumerate(self.rebalance_periods):
                dom = domains[row - 1]
                y_center = (dom[0] + dom[1]) / 2
                df:pd.DataFrame = self.indus_returns_dict[period][['industry',"ret_net_s_qb_mean"]].sort_values(by='ret_net_s_qb_mean', ascending=True)
                self.add_subtitle(fig,f'Average Industry Q-1 Short Return (Rebalance Period = {period})',row)
                fig.add_trace(
                    go.Bar(
                        x=df["ret_net_s_qb_mean"],
                        y=df["industry"],
                        orientation='h',
                        marker=dict(
                            color=df["ret_net_s_qb_mean"],
                            colorscale="RdYlGn",
                    showscale=True, 
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=y_center,
                        len=(1 / n_rows)*0.7,
                        x=1.04,
                        outlinewidth=0,
                        titlefont=dict(size=16, family="Arial", color="black"), 
                        tickfont=dict(size=14, family="Arial", color="black"))
                        ),
                        showlegend=False,
                        ),
                    row=row, col=1
                )
                
                font_size = min(12, max(7, int( 150 / len(df))))
                fig.update_xaxes(title_text="Mean Return", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
                fig.update_yaxes(title_text="Industry", type="category", row=row, col=1, tickfont=dict(size=font_size), title_font=dict(size=18))
                row += 1
        
        try:
            monthly_panel = self.s_monthly_panel
            if monthly_panel is None or getattr(monthly_panel, "shape", (0,))[0] == 0:
                raise ValueError("s_monthly_panel empty")
            self.add_subtitle(fig,'Aggregated Q-1 Short Return',row,y=1.1)
            z = monthly_panel.values
            x = list(monthly_panel.columns)
            y = list(monthly_panel.index)
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="RdYlGn",
                    showscale=True, 
                    colorbar=dict(
                        title="Return",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (row - 0.55) / n_rows,  
                        len=(1 / n_rows)*0.7,
                        x=1.04),
                    text=np.round(z, 4),
                    texttemplate="%{text:.2%}",
                    hovertemplate="Period: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text=f"Per {self.freq}", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=8))
            fig.update_yaxes(title_text="Rebalance Period", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=16))
        except Exception:
            fig.add_annotation(text="No stats panel available", row=row, col=1, showarrow=False)
        row += 1

        try:
            self.add_subtitle(fig,'Q-1 Short Return Statistics',row,y=1.1)
            stats_df = self.s_stats_panel.set_index("period").T
            z_stats = stats_df.T.values
            z_stats = np.asarray(z_stats, dtype=np.float64)
            z_stats = np.nan_to_num(z_stats, nan=np.nan)
            text = np.round(z_stats, 4)
            x_stats = list(stats_df.index)
            y_stats = list(stats_df.columns)
            fig.add_trace(
                go.Heatmap(
                    z=z_stats,
                    x=x_stats,
                    y=y_stats,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (row - 0.55) / n_rows,  
                        len=(1 / n_rows)*0.7,                 
                        x=1.04                           
                    ),
                    text=text,
                    texttemplate="%{text}",
                    hovertemplate="Metric: %{x}<br>Period: %{y}<br>Value: %{z:.4f}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text="Metrics", row=row, col=1)
            fig.update_yaxes(title_text="Rebalance Period", row=row, col=1, type="category")
        except Exception:
            fig.add_annotation(text="No stats panel available", row=row, col=1, showarrow=False)

        height_per_row = 550
        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=120, b=80, l=80, r=200),
            title=dict(
                text=f"Short Return Sheet ({self.factor_name})",
                font=dict(size=24, family="Arial", color="black")
            )
        )

        legend_name_for_row = lambda r: "legend" if r == 1 else f"legend{r}"
        traces_per_row = []
        for _ in range(n_periods):
            traces_per_row.append(self.bins+1)
        for _ in range(n_periods):
            traces_per_row.append(2)
        if self.group_by:
            for _ in range(n_periods):
                traces_per_row.append(1)
        for _ in range(n_periods):
            traces_per_row.append(2)

        tidx = 0
        for r_idx, cnt in enumerate(traces_per_row, start=1):
            legend_key = legend_name_for_row(r_idx)
            for _ in range(cnt):
                if tidx < len(fig.data):
                    fig.data[tidx].update(legend=legend_key)
                tidx += 1
            
        legend_layouts = {}
        for i in range(1, n_rows + 1):
            key = "legend" if i == 1 else f"legend{i}"
            y_fixed = 1 - (i - 0.85) / n_rows
            legend_layouts[key] = dict(
                x=1.03,
                y=y_fixed,
                xanchor="left",
                yanchor="middle",
                orientation="v",
                tracegroupgap=6,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0 ,
                font=dict(size=16, family="Arial", color="black")
            )
        layout_updates = base_layout.copy()
        layout_updates.update(legend_layouts)
        fig.update_layout(**layout_updates)
        if staticPlot:
            fig.show(
                config={
                    "staticPlot": True, 
                    "responsive": True
                }
            )
        else:
            fig.show(config={"responsive": True})
        if return_fig:
            return fig

    def create_single_fac_full_sheet(self, staticPlot:bool=False, return_fig:bool=False):
        self.create_single_fac_ic_sheet(staticPlot,return_fig)
        self.create_long_short_return_sheet(staticPlot,return_fig)
        self.create_long_return_sheet(staticPlot,return_fig)
        self.create_short_return_sheet(staticPlot,return_fig)
    
    def trace(self, rebalance_period:int|str, date:str, bins:list, position="l", staticPlot:bool=False, return_fig:bool=False, return_full_df:bool=False):
        r"""
        Execute a microscopic cross-sectional alpha tracer pipeline at a specific critical historical trading bar.

        This method operates as a high-fidelity continuous white-box diagnostic engine. It isolates 
        and extracts a structural sub-panel window surrounding a target execution date ($\pm 1$ Bar) 
        to diagnose the micro-properties of the factor alpha strategy. It visualizes cross-sectional return 
        heatmaps, asset-level price change scatter clusters, granular gross return contributions, and 
        dynamic portfolio rebalancing dumbbell curves simultaneously.

        Parameters
        ----------
        rebalance_period : int or str
            The target strategy rebalancing interval (e.g., integer trading days or custom calendaring codes like 'W', 'M').
        
        date : str
            The exact chronological execution target bar string (e.g., '2026-05-20 09:30:00') serving as the 
            central node for historical micro-panel slicing.
        
        bins : list[int], optional
            The specific quantile bin coordinates to visualize. Defaults dynamically to the edge legs and 
            the central quantile block: `[1, self.mid_q, self.bins]`.
        
        position : str, default 'l'
            The directional execution perspective modifier. 
            - 'l' maps long-leg performance tracking vectors (highest quantile groups match positive alpha).
            - 's' maps inverted short-leg metrics (lowest quantile scores invert capital assignment matrices).
        
        staticPlot : bool, default False
            If True, silences reactive UI JavaScript workers and outputs an immutable flattened vector graphic asset. 
            Crucial for continuous batch reporting scripts or generating static compliance documentation.
            If False, retains full live plotly features including interactive vector panning and hovering.
        
        return_fig : bool, default False
            If True, returns the compiled Plotly Figure container alongside the structured intermediate micro-panel dataframe. 
            If False, only the diagnostic dataframe is passed back to the interpreter frame.

        Returns
        -------
        If return_fig is True:
            fig : plotly.graph_objects.Figure
                The compiled high-density interactive micro-diagnostic diagnostic figure asset.
            df_ori : pd.DataFrame
                The sliced multi-index cross-sectional tracking dataset cast into a Pandas format.
        If return_fig is False:
            df_ori : pd.DataFrame
                The sliced multi-index cross-sectional tracking dataset cast into a Pandas format.
        """
        df_ori, df_result, Dates, rebalance_dates = self.calc_stats_for_trace(rebalance_period,date)
        bins = bins if bins else [q for q in range(self.bins, 0, -1)]
        bins.sort(reverse=True)
        
        date_map = dict(zip(sorted(Dates)[1:], sorted(Dates)[:-1]))
        dates_4_mask = pl.col(self.trade_date_col).is_in(
            Dates.implode()
        )
        if not return_full_df:
            df_ori = df_ori.filter(dates_4_mask)
            
        Dates = set(sorted(Dates)[1:])
        rebalance_dates = set(rebalance_dates)
        rebalence_bars_len = len(Dates & rebalance_dates)
        
        ###############################################################
        if position == "l":
            cols = [c for c in df_result.columns if c.startswith("ret_net_q")]
        
        elif position == "s":
            cols = [c for c in df_result.columns if c.startswith("ret_net_s_q")]
            df_ori = df_ori.with_columns(
                (-pl.col("w")).alias("w")  
            )
        else:
            raise ValueError("position must be l / s")

        cols = sorted(cols)

        n_periods = len(df_result)
        n_rows = n_periods * 2 * len(bins) + 1 + len(bins) * rebalence_bars_len
        specs = [[{"type": "heatmap"}]]
        specs += [[{"type": "xy"}] for _ in range(n_rows - 1)]
        
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                            vertical_spacing=0,
                            specs=specs)

        gap_px = 200           
        row_content_px = 550
        row_pixel_heights = [row_content_px] * n_rows
        total_height_px = sum(row_pixel_heights) + (n_rows - 1) * gap_px + 200

        domains = []
        cursor = total_height_px
        for h in row_pixel_heights:
            top = cursor / total_height_px
            bottom = (cursor - h) / total_height_px
            domains.append([bottom, top])
            cursor -= (h + gap_px)

        for i, dom in enumerate(domains, start=1):
            axis_name = "yaxis" if i == 1 else f"yaxis{i}"
            if axis_name in fig.layout:
                fig.layout[axis_name].domain = dom
            else:
                fig.layout[axis_name] = dict(domain=dom)
        
        row = 1
        try:
            df_long = (
                df_result.select([self.trade_date_col] + cols)
                .unpivot(
                    index=self.trade_date_col,
                    on=cols,
                    variable_name="quantile_label",  
                    value_name="return"
                )
                .with_columns(
                    ("Q" + pl.col("quantile_label").str.extract(r"q(\d+)$")).alias("quantile_label")
                )
                .sort([pl.col("quantile_label").str.extract(r"(\d+)").cast(pl.Int32), pl.col(self.trade_date_col)], descending=[False, False])
            )

            pivot = df_long.pivot(
                values="return",
                index="quantile_label",
                on=self.trade_date_col
            )
            
            x = sorted(pivot.columns[1:])
            pivot = pivot.select(["quantile_label"] + x)
            
            z = pivot.select(pl.exclude("quantile_label")).to_numpy()
            y = pivot["quantile_label"].to_list()
            
            self.add_subtitle(fig, 'Panel of Net Returns', row, y=1.1)
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (row - 0.65) / n_rows,
                        len=(1 / n_rows) * 0.7,                 
                        x=1.04                                   
                    ),
                    text=z,
                    texttemplate="%{text:.2%}",
                    hovertemplate="Metric: %{y}<br>Date: %{x}<br>Value:  %{z:.2%}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text="Date", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=9), type="category")
            fig.update_yaxes(title_text="Quantile", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=16))
        except Exception as e:
            print(f"Error drawing heatmap: {e}")
            fig.add_annotation(text="No returns panel available", row=row, col=1, showarrow=False)
        row += 1
        
        df_ori = df_ori.with_columns([
            pl.col("w").alias("weight_lag_1"),
            pl.col("fut_ret_1").alias("return"),
            (pl.col("w") * pl.col("fut_ret_1")).alias("contribution")
        ]).drop(["w", "fut_ret_1"])
        df_ori = df_ori.to_pandas()
        
        for bin in bins:
            for d in sorted(Dates):
                self.add_subtitle(fig,f"Weights and Returns (Date = {d}, bin = {bin})",row)
                df_bin = df_ori[(df_ori['quantile'] == bin) & (df_ori[self.trade_date_col] == date_map[d])]
                fig.add_trace(
                    go.Scatter(
                        x=df_bin["weight_lag_1"],
                        y=df_bin["return"],
                        mode="markers+text",
                        text=df_bin[self.symbol_col],
                        textposition="middle center",
                        marker=dict(
                            size=20,
                            color=df_bin["return"],
                            colorscale="RdYlGn",
                            colorbar=dict(
                                title="Return",
                                y=1 - (row - 0.7) / n_rows,
                                len=(1 / n_rows)*0.7,
                                bordercolor="rgba(0,0,0,0)",  
                                outlinewidth=0),
                                showscale=True
                            ),
                        customdata=df_bin[[self.symbol_col]],
                        hovertemplate=
                            "code: %{customdata[0]}<br>" +
                            "weight_lag_1: %{x:.2%}<br>" +
                            "ret: %{y:.2%}<extra></extra>"
                    ),
                    row=row, col=1   
                )

                fig.add_hline(y=0, line_dash="dash", line_color="black", row=row, col=1)

                fig.add_vline(
                    x=df_bin["weight_lag_1"].median(),
                    line_dash="dash",
                    line_color="gray",
                    row=row, col=1
                )
                fig.update_xaxes(
                    title_text="Weight",
                    tickformat=".1%",
                    row=row, col=1,
                    title_font=dict(size=18),
                    tickfont=dict(size=8)
                )
                fig.update_yaxes(title_text="Return", row=row, col=1, tickformat=".1%", title_font=dict(size=18), tickfont=dict(size=16))
                    
                row += 1
                
                df_bar = df_bin.sort_values("contribution", ascending=False).dropna(subset=["contribution"]).copy()
                self.add_subtitle(fig,f"Contributions of Gross Return (Date = {d}, bin = {bin})",row)
                fig.add_trace(
                    go.Bar(
                        x=df_bar["contribution"],
                        y=df_bar[self.symbol_col],
                        orientation="h",
                        marker=dict(
                            color=df_bar["contribution"],
                            colorscale="RdYlGn",
                            showscale=True,
                            colorbar=dict(
                                title="Return",
                                y=1 - (row - 0.7) / n_rows,
                                len=(1 / n_rows)*0.7,
                                bordercolor="rgba(0,0,0,0)",  
                                outlinewidth=0),
                            ),
                        hovertemplate=
                            "code: %{y}<br>" +
                            "contribution: %{x:.2%}<extra></extra>"
                    ),
                    row=row, col=1
                )
                fig.update_xaxes(
                    title_text="Contribution",
                    tickformat=".1%",
                    row=row, col=1,
                    title_font=dict(size=18),
                    tickfont=dict(size=8)
                )
                fig.update_yaxes(title_text="Symbol", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16),
                                 type="category",
                                 categoryorder="array",
                                 categoryarray=df_bar[self.symbol_col])    
                row += 1
                if d in rebalance_dates:
                    self.add_subtitle(
                        fig,
                        f"Weight Changes (Date = {d}, bin = {bin}, Rebalance = Ture)",
                        row
                    )
            
                    df_bin = df_ori[(df_ori['quantile'] == bin) & (df_ori[self.trade_date_col] == d)]
                    g_list = sorted(df_bin["g"].unique())
                    
                    if len(g_list) >= 2:

                        old_g = g_list[0]
                        new_g = g_list[1]

                        df_old = (
                            df_bin[df_bin["g"] == old_g]
                            [[self.symbol_col, "weight_lag_1"]]
                            .rename(columns={"weight_lag_1": "old_w"})
                        )

                        df_new = (
                            df_bin[df_bin["g"] == new_g]
                            [[self.symbol_col, "weight_lag_1"]]
                            .rename(columns={"weight_lag_1": "new_w"})
                        )

                        df_dumbbell = (
                            pd.merge(
                                df_old,
                                df_new,
                                on=self.symbol_col,
                                how="outer"
                            )
                            .fillna(0)
                            .sort_values("new_w")
                        )

                        for _, r in df_dumbbell.iterrows():

                            fig.add_trace(
                                go.Scatter(
                                    x=[r["old_w"], r["new_w"]],
                                    y=[r[self.symbol_col], r[self.symbol_col]],
                                    mode="lines",
                                    line=dict(
                                        color="gray",
                                        width=2
                                    ),
                                    showlegend=False,
                                    hoverinfo="skip"
                                ),
                                row=row,
                                col=1
                            )

                        fig.add_trace(
                            go.Scatter(
                                x=df_dumbbell["old_w"],
                                y=df_dumbbell[self.symbol_col],
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    symbol="circle"
                                ),
                                name="Old Weight",
                                customdata=df_dumbbell[[self.symbol_col]],
                                hovertemplate=
                                    "code: %{customdata[0]}<br>" +
                                    "old weight: %{x:.2%}<extra></extra>"
                            ),
                            row=row,
                            col=1
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=df_dumbbell["new_w"],
                                y=df_dumbbell[self.symbol_col],
                                mode="markers",
                                marker=dict(
                                    size=12,
                                    symbol="diamond"
                                ),
                                name="New Weight",
                                customdata=df_dumbbell[[self.symbol_col]],
                                hovertemplate=
                                    "{}: %{customdata[0]}<br>" +
                                    "new weight: %{x:.2%}<extra></extra>"
                            ),
                            row=row,
                            col=1
                        )

                        fig.update_xaxes(
                            title_text="Weight",
                            tickformat=".1%",
                            row=row,
                            col=1,
                            title_font=dict(size=18),
                            tickfont=dict(size=10)
                        )

                        fig.update_yaxes(
                            title_text="Symbol",
                            row=row,
                            col=1,
                            title_font=dict(size=18),
                            tickfont=dict(size=10),
                            type="category"
                        )

                        row += 1
        
        height_per_row = 550

        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            margin=dict(t=120, b=80, l=80, r=200),
            title=dict(
                text=f"Cross-sectional Snapshot ({date} ±1 Bar, Position = '{position}')",   
                font=dict(size=24, family="Arial", color="black")
            ),
            showlegend=False
        )
        fig.update_layout(base_layout)
        
        if staticPlot:
            fig.show(
                config={
                    "staticPlot": True, 
                    "responsive": True
                }
            )
        else:
            fig.show(config={"responsive": True})
        
        if return_fig:
            return fig, df_ori
        
        else:
            return df_ori


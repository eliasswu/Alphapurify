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

@dataclass
class ResearchConfig:
    rebalance_periods: list[int] = (1,5,10)
    return_rolling_period: int = 20
    return_horizons: list[int] = (1,5,10)
    horizon_rolling_period: int = 20
    bins: int = 5
    fac_shift: int | None = None
    base_rate: float = 0.02
    overnight: str = "on"

@dataclass
class AnalysisConfig:
    rank_ic: bool = True
    log_scale: bool = True
    agg_freq: str | None = None
    group_by: dict | None = None
    max_workers: int = -1

_worker_df = None

class FactorAnalyzer():
    """
    FactorAnalyzer

    A comprehensive factor research and evaluation tool for cross-sectional
    asset pricing factors. This class provides a full pipeline for factor
    performance analysis, including IC evaluation, quantile portfolio
    backtesting, turnover analysis, and statistical summaries.

    The analyzer supports parallel computation and is optimized for large
    datasets using Polars and Apache Arrow memory mapping.

    Main functionalities include:
    - Cross-sectional IC / Rank IC analysis
    - Quantile portfolio return analysis
    - Long-short strategy evaluation
    - Factor autocorrelation analysis
    - Turnover calculation
    - Industry grouped analysis
    - Rolling statistics and cumulative metrics
    - Parallel computation for multiple horizons

    Parameters
    ----------
    base_df : pd.DataFrame or pl.DataFrame
        Input dataset containing price and factor data.

    trade_date_col : str
        Column name representing the timestamp of observations.

    symbol_col : str
        Column name representing the asset identifier.

    price_col : str
        Column name representing the asset price used to compute returns.

    factor_name : str
        Column name of the factor to be analyzed.

    research_cfg : ResearchConfig or dict, optional
        Configuration controlling factor research parameters such as
        return horizons, rebalance periods, quantile bins, rolling windows,
        and overnight handling.

    analysis_cfg : AnalysisConfig or dict, optional
        Configuration controlling analysis settings such as aggregation
        frequency, IC type, logging scale, grouping options, and
        parallelization parameters.

    Attributes
    ----------
    base_df : pl.DataFrame
        Internal dataset converted to Polars format and sorted by
        (symbol, trade_date).

    rebalance_periods : list[int]
        Rebalancing intervals used for quantile portfolio backtests.

    return_horizons : list[int]
        Forecast horizons used for IC analysis.

    bins : int
        Number of quantile groups used in portfolio sorting.

    freq : str
        Automatically inferred aggregation frequency based on data interval.

    ls_stats_panel : pd.DataFrame
        Long-short portfolio performance statistics.

    l_stats_panel : pd.DataFrame
        Long portfolio performance statistics.

    s_stats_panel : pd.DataFrame
        Short portfolio performance statistics.

    ic_stats_panel : pd.DataFrame
        Information coefficient summary statistics.

    Notes
    -----
    - Factor values can optionally be shifted using `fac_shift` to prevent
      look-ahead bias.
    - Overnight returns can be included, excluded, or isolated via the
      `overnight` configuration.
    - Industry level analysis can be enabled using the `group_by` mapping.
    - Parallel computation uses joblib with Apache Arrow memory mapping
      to avoid large data duplication across processes.

    Typical Workflow
    ----------------
    >>> analyzer = FactorAnalyzer.simple(df, factor_name="alpha")
    >>> analyzer.run()
    >>> analyzer.create_single_fac_ic_sheet()
    >>> analyzer.create_long_short_return_sheet()

    Visualization utilities are also provided to inspect cumulative
    returns, IC time series, quantile spreads, and industry contributions.
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
        self.return_rolling_period = self.research_cfg.return_rolling_period
        self.horizon_rolling_period = self.research_cfg.horizon_rolling_period
        self.rank_ic = self.analysis_cfg.rank_ic
        self.log_scale = self.analysis_cfg.log_scale
        self.agg_freq = self.analysis_cfg.agg_freq
        self.base_rate =  self.research_cfg.base_rate
        self.group_by = self.analysis_cfg.group_by
        self.overnight = self.research_cfg.overnight
        self.bins = self.research_cfg.bins
        self.fac_shift = self.research_cfg.fac_shift
        self.max_workers = self.analysis_cfg.max_workers
        
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
        
        self.mid_q = (self.bins + 1) // 2
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
    
    def calc_stats_for_period(self,args):
        period, base_df_path = args
        global _worker_df

        if _worker_df is None:
            with pa.memory_map(base_df_path, "r") as source:
                _worker_df = pl.from_arrow(
                    pa.ipc.open_file(source).read_all()
                )

        df = _worker_df.clone()
            
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
        df = df.with_columns(
            pl.when(pl.col("fut_ret") == -1).then(None).otherwise(pl.col("fut_ret")).alias("fut_ret")
        )
        
        all_dates = df[self.trade_date_col].unique().sort()
        rebalance_dates = all_dates[::period]
        rebalance_mask = pl.col(self.trade_date_col).is_in(
            rebalance_dates.implode()
        )
        
        df = df.with_columns(
            pl.col(self.trade_date_col).max()
            .over([self.symbol_col, pl.col(self.trade_date_col).dt.date()])
            .alias("day_last_dt"),

            (pl.col(self.trade_date_col) + period * self.td).alias("target_dt")
        )

        df = df.with_columns(
            (pl.col("target_dt") > pl.col("day_last_dt"))
            .alias("is_overnight")
        )

        if self.overnight == "on":
            mask = rebalance_mask

        elif self.overnight == "off":
            mask = (~pl.col("is_overnight")) | rebalance_mask

        elif self.overnight == "only":
            mask = (pl.col("is_overnight")) | rebalance_mask

        df = df.filter(mask).select(
            [self.trade_date_col, self.symbol_col, self.factor_name, "fut_ret"]
        )           
        
        
        df = df.with_columns([
            pl.col(self.factor_name).rank("average", descending=True).over(self.trade_date_col).alias("rank"),
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
        ##############################################################################################################
        if self.group_by:
            df = df.with_columns([
                pl.col(self.symbol_col).map_elements(lambda x: self.group_by.get(x, "Unlabels"), return_dtype=pl.Utf8).alias("industry")
            ])
        
        long_df  = df.filter(pl.col("quantile") == 1)
        mid_df   = df.filter(pl.col("quantile") == self.mid_q)
        short_df = df.filter(pl.col("quantile") == self.bins)
        
        if self.group_by:
            long_ret_i = long_df.group_by([self.trade_date_col,'industry']).agg(pl.col("fut_ret").mean().alias("i_ret_q1")).sort(self.trade_date_col)
            short_ret_i = short_df.group_by([self.trade_date_col,'industry']).agg(pl.col("fut_ret").mean().alias("i_ret_qb")).sort(self.trade_date_col)
            
            ls_ret_i = long_ret_i.join(short_ret_i, on=[self.trade_date_col,'industry'], how="left")
            
            ls_ret_i = ls_ret_i.with_columns((pl.col("i_ret_q1") - pl.col("i_ret_qb")).alias("i_ls_ret"))
        
            ls_ret_i = (
                ls_ret_i
                .group_by("industry")
                .agg([
                    pl.col("i_ret_q1").mean().alias("i_l_ret_mean"),
                    pl.col("i_ret_qb").mean().alias("i_s_ret_mean"),
                    pl.col("i_ls_ret").mean().alias("i_ls_ret_mean")
                ])
            )
        
        else:
            ls_ret_i = pl.DataFrame()

        long_ret = long_df.group_by(self.trade_date_col).agg(pl.col("fut_ret").mean().alias("ret_q1")).sort(self.trade_date_col)
        mid_ret = mid_df.group_by(self.trade_date_col).agg(pl.col("fut_ret").mean().alias("ret_qm")).sort(self.trade_date_col)
        short_ret = short_df.group_by(self.trade_date_col).agg(pl.col("fut_ret").mean().alias("ret_qb")).sort(self.trade_date_col)
        
        ls_ret = long_ret.join(mid_ret, on=self.trade_date_col, how="left").join(short_ret, on=self.trade_date_col, how="left")
        ls_ret = ls_ret.with_columns((pl.col("ret_q1") - pl.col("ret_qb")).alias("ls_ret"))
        
        ls_ret = ls_ret.with_columns(
        [pl.col("ls_ret").rolling_mean(window_size=self.return_rolling_period).alias("ls_ret_rolling"),
        pl.col("ret_q1").rolling_mean(window_size=self.return_rolling_period).alias("ret_q1_rolling"),
        pl.col("ret_qm").rolling_mean(window_size=self.return_rolling_period).alias("ret_qm_rolling"),
        pl.col("ret_qb").rolling_mean(window_size=self.return_rolling_period).alias("ret_qb_rolling"),
        -pl.col("ret_q1").rolling_mean(window_size=self.return_rolling_period).alias("ret_q1_rolling_s"),
        -pl.col("ret_qm").rolling_mean(window_size=self.return_rolling_period).alias("ret_qm_rolling_s"),
        -pl.col("ret_qb").rolling_mean(window_size=self.return_rolling_period).alias("ret_qb_rolling_s"),
        (pl.col("ret_q1") + 1).cum_prod().alias("l_cum_nv_q1"),
        (pl.col("ret_qm") + 1).cum_prod().alias("l_cum_nv_qm"),
        (pl.col("ret_qb") + 1).cum_prod().alias("l_cum_nv_qb"),
        (-pl.col("ret_qb") + 1).cum_prod().alias("s_cum_nv_qb"),
        (-pl.col("ret_qm") + 1).cum_prod().alias("s_cum_nv_qm"),
        (-pl.col("ret_q1") + 1).cum_prod().alias("s_cum_nv_q1"),
        ])
        ls_ret = ls_ret.with_columns(
        (((pl.col("ls_ret") + 1).cum_prod()) - 1).alias("nv"),
        ((pl.col("ls_ret") + 1).cum_prod()).alias("cum_nv")
    )

        df_reb = df.select([self.trade_date_col, "quantile", self.symbol_col])

        q1_list = (
            df_reb.filter(pl.col("quantile") == 1)
                .group_by(self.trade_date_col)
                .agg(pl.col(self.symbol_col).alias("codes"))
                .sort(self.trade_date_col)
        )
        q5_list = (
            df_reb.filter(pl.col("quantile") == self.bins)
                .group_by(self.trade_date_col)
                .agg(pl.col(self.symbol_col).alias("codes"))
                .sort(self.trade_date_col)
        )

        q1_list = q1_list.with_columns(pl.col("codes").shift(1).alias("prev_codes"))
        q5_list = q5_list.with_columns(pl.col("codes").shift(1).alias("prev_codes"))

        q1_list = q1_list.with_columns(
        (1 - (pl.col("codes").list.set_intersection(pl.col("prev_codes")).list.len() 
            / pl.col("codes").list.len())).alias("turnover_q1")
    )
        q5_list = q5_list.with_columns(
        (1 - (pl.col("codes").list.set_intersection(pl.col("prev_codes")).list.len() 
            / pl.col("codes").list.len())).alias("turnover_qb")
    )
        ls_list = q1_list.join(q5_list, on=self.trade_date_col, how="inner",suffix="_right").with_columns([
            pl.concat_list([pl.col("codes"), pl.col("codes_right")]).alias("long_short_codes"),
            pl.concat_list([pl.col("prev_codes"), pl.col("prev_codes_right")]).alias("long_short_prev_codes")
        ])

        ls_list = ls_list.with_columns(
            (1 - (pl.col("long_short_codes").list.set_intersection(pl.col("long_short_prev_codes")).list.len()
                / pl.col("long_short_codes").list.len())).alias("turnover_ls")
        )
        turnover_df = q1_list.select([self.trade_date_col, "turnover_q1"]).join(
            q5_list.select([self.trade_date_col, "turnover_qb"]),
            on=self.trade_date_col, how="inner"
        ).join(ls_list.select([self.trade_date_col, "turnover_ls"]), on=self.trade_date_col, how="inner")
        
        
        
        ls_ret = ls_ret.join(turnover_df, on=self.trade_date_col, how="left")
        
        ls_ret_valid = ls_ret.filter(pl.col('ret_q1').is_not_null())
        
        if self.log_scale:
            ls_ret = ls_ret.with_columns([
                pl.col('cum_nv').log().alias('cum_nv'),
                pl.col("l_cum_nv_q1").log().alias("l_cum_nv_q1"),
                pl.col("l_cum_nv_qm").log().alias("l_cum_nv_qm"),
                pl.col("l_cum_nv_qb").log().alias("l_cum_nv_qb"),
                pl.col("s_cum_nv_qb").log().alias("s_cum_nv_qb"),
                pl.col("s_cum_nv_qm").log().alias("s_cum_nv_qm"),
                pl.col("s_cum_nv_q1").log().alias("s_cum_nv_q1"),
                
            ])
        
        if ls_ret_valid.height == 0:
            raise
        n_periods = len(ls_ret_valid)
        days_per_period = self.days / n_periods
        annual_factor = 365.25 / days_per_period
        
        avg_turnover_l = ls_ret['turnover_q1'].drop_nans().mean()
        avg_turnover_s = ls_ret['turnover_qb'].drop_nans().mean()
        avg_turnover_ls = ls_ret['turnover_ls'].drop_nans().mean()
        ##############################################################################################ls
        cum_nv = ls_ret_valid["cum_nv"].to_numpy()
        mean_ret = ls_ret_valid.select(pl.col("ls_ret").mean()).item()
        mean_loss = (ls_ret_valid.filter(pl.col("ls_ret") < 0).select(pl.col("ls_ret").mean()).item())
        PL = mean_ret / mean_loss
        ann_ret = cum_nv[-1]** (365.25 / self.days) - 1

        vol = (ls_ret_valid.select(pl.col("ls_ret").std(ddof=1)).item()) * np.sqrt(annual_factor)
        if np.isnan(vol):
            vol = (ls_ret_valid.select(pl.col("ls_ret").std(ddof=0)).item()) * np.sqrt(annual_factor)

        excess_ret = ann_ret - self.base_rate
        sharpe = excess_ret / vol if vol != 0 and not np.isnan(vol) else np.nan
        
        running_max = np.maximum.accumulate(cum_nv)
        drawdown = cum_nv / running_max - 1
        max_dd = float(np.nanmin(drawdown)) if drawdown.size > 0 else np.nan

        win_rate = float(ls_ret_valid.select((pl.col("ls_ret") > 0).mean()).item())
        pnl = float(cum_nv[-1] - 1) if cum_nv.size > 0 else np.nan
        downside_std = (ls_ret_valid.filter(pl.col("ls_ret") < 0).select(pl.col("ls_ret").std(ddof=1)).item())
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
            monthly_ret_ls = (
            ls_ret_valid
            .with_columns(
                pl.col(self.trade_date_col).dt.truncate(self.agg_freq).alias("agg_date"),
                (pl.col("ls_ret") + 1).alias("ls_ret_1p")
            )
            .group_by("agg_date")
            .agg(((pl.col("ls_ret") + 1).product() - 1).alias("ls_ret"))

            .select(["agg_date", "ls_ret"])
        )
            
            monthly_ret_l = (
            ls_ret_valid
            .with_columns(
                pl.col(self.trade_date_col).dt.truncate(self.agg_freq).alias("agg_date"),
                (pl.col("ret_q1") + 1).alias("ls_ret_1p")
            )
            .group_by("agg_date")
            .agg(((pl.col("ret_q1") + 1).product() - 1).alias("ret_q1"))

            .select(["agg_date", "ret_q1"])
        )
            monthly_ret_s = (
            ls_ret_valid
            .with_columns(
                pl.col(self.trade_date_col).dt.truncate(self.agg_freq).alias("agg_date"),
                (-pl.col("ret_qb") + 1).alias("ret_qb")
            )
            .group_by("agg_date")
            .agg(((-pl.col("ret_qb") + 1).product() - 1).alias("ret_qb"))

            .select(["agg_date", "ret_qb"])
        )
        
        else:   
            if self.freq != "1y" and self.freq is not None and self.freq != "20d":
                monthly_ret_ls = (
                ls_ret_valid
                .with_columns(
                    pl.col(self.trade_date_col).dt.truncate(self.freq).alias("agg_date"),
                    (pl.col("ls_ret") + 1).alias("ls_ret_1p")
                )
                .group_by("agg_date")
                .agg(((pl.col("ls_ret") + 1).product() - 1).alias("ls_ret"))

                .select(["agg_date", "ls_ret"])
            )
                
                monthly_ret_l = (
                ls_ret_valid
                .with_columns(
                    pl.col(self.trade_date_col).dt.truncate(self.freq).alias("agg_date"),
                    (pl.col("ret_q1") + 1).alias("ls_ret_1p")
                )
                .group_by("agg_date")
                .agg(((pl.col("ret_q1") + 1).product() - 1).alias("ret_q1"))

                .select(["agg_date", "ret_q1"])
            )
                monthly_ret_s = (
                ls_ret_valid
                .with_columns(
                    pl.col(self.trade_date_col).dt.truncate(self.freq).alias("agg_date"),
                    (-pl.col("ret_qb") + 1).alias("ret_qb")
                )
                .group_by("agg_date")
                .agg(((-pl.col("ret_qb") + 1).product() - 1).alias("ret_qb"))

                .select(["agg_date", "ret_qb"])
            )
            
            if self.freq == "20d":
                monthly_ret_ls = (
                ls_ret_valid
                .with_columns(
                    pl.col(self.trade_date_col).dt.strftime("%Y-%m").alias("agg_date"),
                    (pl.col("ls_ret") + 1).alias("ls_ret_1p")
                )
                .group_by("agg_date")
                .agg(((pl.col("ls_ret") + 1).product() - 1).alias("ls_ret"))

                .select(["agg_date", "ls_ret"])
            )
                
                monthly_ret_l = (
                ls_ret_valid
                .with_columns(
                    pl.col(self.trade_date_col).dt.strftime("%Y-%m").alias("agg_date"),
                    (pl.col("ret_q1") + 1).alias("ls_ret_1p")
                )
                .group_by("agg_date")
                .agg(((pl.col("ret_q1") + 1).product() - 1).alias("ret_q1"))

                .select(["agg_date", "ret_q1"])
            )
                monthly_ret_s = (
                ls_ret_valid
                .with_columns(
                    pl.col(self.trade_date_col).dt.strftime("%Y-%m").alias("agg_date"),
                    (-pl.col("ret_qb") + 1).alias("ret_qb")
                )
                .group_by("agg_date")
                .agg(((-pl.col("ret_qb") + 1).product() - 1).alias("ret_qb"))

                .select(["agg_date", "ret_qb"])
            )
            
            if self.freq == "1y" or self.freq == None:
                monthly_ret_ls = pl.DataFrame()
                monthly_ret_l = pl.DataFrame()
                monthly_ret_s = pl.DataFrame()
        ############################################################################################l       
        
        cum_nv = ls_ret_valid["l_cum_nv_q1"].to_numpy()
        mean_ret_l = ls_ret_valid.select(pl.col("ret_q1").mean()).item()
        mean_loss = (ls_ret_valid.filter(pl.col("ret_q1") < 0).select(pl.col("ret_q1").mean()).item())
        PL = mean_ret_l / mean_loss
        ann_ret = cum_nv[-1]** (365.25 / self.days) - 1
        vol = (ls_ret_valid.select(pl.col("ret_q1").std(ddof=1)).item()) * np.sqrt(annual_factor)
        if np.isnan(vol):
            vol = (ls_ret_valid.select(pl.col("ret_q1").std(ddof=0)).item()) * np.sqrt(annual_factor)

        excess_ret = ann_ret - self.base_rate
        sharpe = excess_ret/ vol if vol != 0 and not np.isnan(vol) else np.nan
        
        running_max = np.maximum.accumulate(cum_nv)
        drawdown = cum_nv / running_max - 1
        max_dd = float(np.nanmin(drawdown)) if drawdown.size > 0 else np.nan

        win_rate = float(ls_ret_valid.select((pl.col("ret_q1") > 0).mean()).item())
        pnl = float(cum_nv[-1] - 1) if cum_nv.size > 0 else np.nan
        downside_std = (ls_ret_valid.filter(pl.col("ret_q1") < 0).select(pl.col("ret_q1").std(ddof=1)).item())
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
        cum_nv = ls_ret_valid["s_cum_nv_qb"].to_numpy()
        mean_ret_s = ls_ret_valid.select(-pl.col("ret_qb").mean()).item()
        mean_loss = (ls_ret_valid.filter(-pl.col("ret_qb") < 0).select(pl.col("ret_qb").mean()).item())
        PL = mean_ret_s / mean_loss
        ann_ret = cum_nv[-1]** (365.25 / self.days) - 1
        vol = (ls_ret_valid.select(pl.col("ret_qb").std(ddof=1)).item()) * np.sqrt(annual_factor)
        if np.isnan(vol):
            vol = (ls_ret_valid.select(pl.col("ret_qb").std(ddof=0)).item()) * np.sqrt(annual_factor)

        excess_ret = ann_ret - self.base_rate
        sharpe = excess_ret / vol if vol != 0 and not np.isnan(vol) else np.nan
        
        running_max = np.maximum.accumulate(cum_nv)
        drawdown = cum_nv / running_max - 1
        max_dd = float(np.nanmin(drawdown)) if drawdown.size > 0 else np.nan

        win_rate = float(ls_ret_valid.select((-pl.col("ret_qb") > 0).mean()).item())
        pnl = float(cum_nv[-1] - 1) if cum_nv.size > 0 else np.nan
        downside_std = (ls_ret_valid.filter(-pl.col("ret_qb") < 0).select(-pl.col("ret_qb").std(ddof=1)).item())
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

        return period, stats_ls, ls_ret, mean_ret, monthly_ret_ls, monthly_ret_l, stats_l, monthly_ret_s, stats_s, ls_ret_i, avg_turnover_ls
        
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
                pl.col("code").map_elements(lambda x: self.group_by.get(x, "Unlabels"), return_dtype=pl.Utf8).alias("industry")
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
            
        stats_df_ls = pd.DataFrame([{"period": p, **s} for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results])
        stats_df_l = pd.DataFrame([{"period": p, **stats_l} for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results])
        stats_df_s = pd.DataFrame([{"period": p, **stats_s} for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results])
        ls_rets = {p: ls.to_pandas() for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results}
        ls_turnovers = {p: avg_turnover_ls for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results}
        mean_rets = {p: mean_ret for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results}
        indus_rets = {p: ls_ret_i.to_pandas() for p, s, ls, mean_ret, monthly, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results}
        
        monthly_ls = pl.concat(
        [m.with_columns(pl.lit(p).alias("period")) for p, _, _, _, m, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results if m.height > 0]
    )
        monthly_l = pl.concat(
        [monthly_ret_l.with_columns(pl.lit(p).alias("period")) for p, _, _, _, m, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results if monthly_ret_l.height > 0]
    )
        monthly_s = pl.concat(
        [monthly_s.with_columns(pl.lit(p).alias("period")) for p, _, _, _, m, monthly_ret_l, stats_l, monthly_s, stats_s, ls_ret_i, avg_turnover_ls in results if monthly_ret_l.height > 0]
    )
        monthly_pivot_ls = (
        monthly_ls
        .to_pandas()
        .pivot(index="agg_date", columns="period", values="ls_ret")
        .sort_index()
    )
        monthly_pivot_l = (
        monthly_l
        .to_pandas()
        .pivot(index="agg_date", columns="period", values="ret_q1")
        .sort_index()
    )
        monthly_pivot_s = (
        monthly_s
        .to_pandas()
        .pivot(index="agg_date", columns="period", values="ret_qb")
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
        return ls_rets, indus_rets, ls_turnovers, ic_dfs, stats_df_ls, stats_df_l, stats_df_s, ic_panal, monthly_pivot_ls.T, monthly_pivot_l.T, monthly_pivot_s.T, monthly_pivot_ic.T, mean_rets, mean_ics, mean_ic_autocorrs, ic_indus_contribs
    
    def run(self):
        (   
            self.retruns_dict,
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
            
        )  = self.run_stats_parallel() 
    
    def create_single_fac_full_sheet(self, staticPlot:bool=False, return_fig:bool=False):
        self.create_single_fac_ic_sheet(staticPlot,return_fig)
        self.create_long_short_return_sheet(staticPlot,return_fig)
        self.create_long_return_sheet(staticPlot,return_fig)
        self.create_short_return_sheet(staticPlot,return_fig)
            
    def create_long_return_sheet(self, staticPlot:bool=False, return_fig:bool=False):
        """
        Generate a comprehensive multi-panel visualization for long-only factor performance analysis.

        This function constructs a vertically stacked Plotly figure containing multiple analytical panels
        based on different rebalance periods. It visualizes rolling returns, cumulative returns, turnover,
        optional industry-level performance, and summary heatmaps.

        Parameters
        ----------
        staticPlot : bool, default False
            If True, renders the figure in static mode (non-interactive). Useful for exporting or reporting.
            If False, enables full interactivity (zoom, hover, etc.).

        return_fig : bool, default False
            If True, returns the Plotly Figure object instead of (or in addition to) displaying it.
            Useful for testing, further customization, or external saving.

        Panels Included
        ---------------
        1. Rolling Mean Returns
            - Displays rolling average returns for top (Q1), middle (Qm), and bottom (Qb) quantiles.
            - Helps identify short-term signal stability.

        2. Cumulative Returns
            - Shows cumulative net value curves for different quantiles.
            - Supports optional log scaling (controlled by self.log_scale).

        3. Industry Analysis (Optional)
            - Only included if self.group_by is True.
            - Horizontal bar chart of average returns by industry for top quantile.

        4. Turnover Rate
            - Scatter plot of turnover ratios for top and bottom quantiles.
            - Indicates trading frequency and strategy cost implications.

        5. Monthly Aggregated Return Heatmap
            - Heatmap of monthly returns for top quantile across rebalance periods.
            - Useful for seasonality and consistency analysis.

        6. Statistics Heatmap
            - Summary metrics (e.g., Sharpe, mean return) across rebalance periods.
            - Provides quick comparison of performance quality.

        Layout Design
        -------------
        - Dynamic number of rows based on:
            * number of rebalance periods
            * whether industry grouping is enabled
        - Custom vertical spacing using manual domain control for precise layout.
        - Independent legends per subplot row for clarity.
        - Right-side aligned legends and colorbars to avoid overlap.

        Returns
        -------
        fig : plotly.graph_objects.Figure, optional
            Returned only if return_fig=True.
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
        
        colors = {"qtop": "darkgreen", "qmid": "limegreen", "qbot": "lightgreen"}
        line_width = 2
        
        row = 1
        for period in self.rebalance_periods:
            date = self.retruns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Long Return Rolling Means (Rebalance Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['ret_q1_rolling'],
                    mode="lines",
                    name=f"Q-Top Rolling {self.return_rolling_period} Mean",
                    line=dict(width=line_width, color=colors["qtop"]),
                    showlegend=True
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['ret_qm_rolling'],
                    mode="lines",
                    name=f"Q-{self.mid_q} Rolling {self.return_rolling_period} Mean",
                    line=dict(width=line_width, color=colors["qmid"]),
                    showlegend=True
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['ret_qb_rolling'],
                    mode="lines",
                    name=f"Q-Bottom Rolling {self.return_rolling_period} Mean",
                    line=dict(width=line_width, color=colors["qbot"]),
                    showlegend=True
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
            fig.update_yaxes(title_text="Return", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1

        for period in self.rebalance_periods:
            date = self.retruns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Cumulative Long Returns (Rebalance Period = {period}, Log Scale = {self.log_scale})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['l_cum_nv_q1'],
                    mode="lines",
                    name="Q-Top Cumulative Return",
                    line=dict(width=line_width, color=colors["qtop"]),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['l_cum_nv_qm'],
                    mode="lines",
                    name=f"Q-{self.mid_q} Cumulative Return",
                    line=dict(width=line_width, color=colors["qmid"]),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['l_cum_nv_qb'],
                    mode="lines",
                    name="Q-Bottom Cumulative Return",
                    line=dict(width=line_width, color=colors["qbot"]),
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
                df:pd.DataFrame = self.indus_returns_dict[period][['industry',"i_l_ret_mean"]].sort_values(by='i_l_ret_mean', ascending=True)
                self.add_subtitle(fig,f'Average Industry Q-Top Long Return (Rebalance Period = {period})',row)
                fig.add_trace(
                    go.Bar(
                        x=df["i_l_ret_mean"],
                        y=df["industry"],
                        orientation='h',
                        marker=dict(
                            color=df["i_l_ret_mean"],
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
            date = self.retruns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Turnover Rate (Rebalance Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['turnover_q1'],
                    mode="markers",
                    name="Q-Top Turnover",
                    marker=dict(size=6),
                    line=dict(color=colors["qtop"], width=line_width),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['turnover_qb'],
                    mode="markers",
                    name="Q-Bottom Turnover",
                    marker=dict(size=6),
                    line=dict(color=colors["qbot"], width=line_width),
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

        try:
            monthly_panel = self.l_monthly_panel
            if monthly_panel is None or getattr(monthly_panel, "shape", (0,))[0] == 0:
                raise ValueError("l_monthly_panel empty")
            self.add_subtitle(fig,'Aggregated Q-Top Long Return',row,y=1.1)
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
                    texttemplate="%{text}",
                    hovertemplate="Period: %{y}<br>Month: %{x}<br>Return: %{z:.4f}<extra></extra>"
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
            self.add_subtitle(fig,'Q-Top Long Return Statistics',row,y=1.1)
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
                        x=1.04                           
                    ),
                    text=np.round(z_stats, 4),
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
                text=f"Long Return Sheet ({self.factor_name})",   
                font=dict(size=24, family="Arial", color="black")
            )
        )

        legend_name_for_row = lambda r: "legend" if r == 1 else f"legend{r}"
        traces_per_row = []
        for _ in range(n_periods):
            traces_per_row.append(3)
        for _ in range(n_periods):
            traces_per_row.append(3)
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
        """
        Generate a comprehensive multi-panel visualization for long-short factor performance analysis.

        This function builds a vertically stacked Plotly figure that summarizes the performance of a
        long-short strategy across multiple rebalance periods. It includes return dynamics, cumulative
        performance, turnover, optional industry attribution, and summary heatmaps.

        Parameters
        ----------
        staticPlot : bool, default False
            If True, renders the figure in static (non-interactive) mode.
            If False, enables full interactivity (zoom, hover, etc.).

        return_fig : bool, default False
            If True, returns the Plotly Figure object for further processing or testing.
            Otherwise, only displays the figure.

        Panels Included
        ---------------
        1. Long-Short Returns
            - Time series of long-short returns.
            - Includes rolling mean and horizontal mean return line.
            - Displays mean value annotation for quick reference.

        2. Cumulative Long-Short Returns
            - Net value curve of the long-short portfolio.
            - Supports log scale (controlled by self.log_scale).

        3. Industry Analysis (Optional)
            - Only shown if self.group_by is True.
            - Horizontal bar chart of average industry-level long-short returns.
            - Helps identify sector contribution to alpha.

        4. Turnover Rate
            - Scatter plot of portfolio turnover over time.
            - Includes mean turnover line and annotation.
            - Useful for understanding trading intensity and cost.

        5. Monthly Aggregated Return Heatmap
            - Heatmap of monthly long-short returns across rebalance periods.
            - Useful for identifying consistency and seasonality.

        6. Statistics Heatmap
            - Summary performance metrics (e.g., Sharpe ratio, mean return).
            - Enables quick comparison across rebalance periods.

        Layout Design
        -------------
        - Dynamic row count based on:
            * number of rebalance periods
            * whether industry grouping is enabled
        - Custom vertical spacing via manual domain control for precise layout.
        - Independent legends for each subplot row to improve readability.
        - Right-aligned legends and colorbars to avoid overlap with plots.

        Returns
        -------
        fig : plotly.graph_objects.Figure, optional
            Returned only if return_fig=True.
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

        line_width = 2
        row = 1
        for period in self.rebalance_periods:
            date = self.retruns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Long-Short Return (Rebalance Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['ls_ret'],
                    mode="lines",
                    name="Return",
                    line=dict(width=line_width, color='lightgreen'),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['ls_ret_rolling'],
                    mode="lines",
                    name=f"Rolling {self.return_rolling_period} Mean",
                    line=dict(width=line_width, color='darkgreen'),
                    showlegend=True,
                ),
                row=row, col=1
            )
            mean_ret = self.mean_returns_dict[period]
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=[mean_ret] * len(date),
                    mode="lines",
                    name=f"Return Rolling {self.return_rolling_period} Mean",
                    line=dict(color="red", width=3, dash="dot"),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_annotation(x=0.99, y=0.99, text=f"Mean = {mean_ret:.4f}", 
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
            fig.update_yaxes(title_text="Return", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1

        for period in self.rebalance_periods:
            date = self.retruns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Cumulative Long-Short Return (Rebalance Period = {period}, Log Scale = {self.log_scale})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['cum_nv'],
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
                df:pd.DataFrame = self.indus_returns_dict[period][['industry',"i_ls_ret_mean"]].sort_values(by='i_ls_ret_mean', ascending=True)
                self.add_subtitle(fig,f'Average Industry Long–Short Return (Rebalance Period = {period})',row)
                fig.add_trace(
                    go.Bar(
                        x=df["i_ls_ret_mean"],
                        y=df["industry"],
                        orientation='h',
                        marker=dict(
                            color=df["i_ls_ret_mean"],
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
            date = self.retruns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Turnover Rate (Rebalance Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['turnover_ls'],
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
                    texttemplate="%{text}",
                    hovertemplate="Period: %{y}<br>Month: %{x}<br>Return: %{z:.4f}<extra></extra>"
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
                    text=np.round(z_stats, 4),
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
            traces_per_row.append(3)
        for _ in range(n_periods):
            traces_per_row.append(1)
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
        """
        Generate a comprehensive multi-panel visualization for factor IC (Information Coefficient) analysis.

        This function constructs a vertically stacked Plotly figure to evaluate the predictive power of a factor
        across multiple return horizons. It supports both standard IC and Rank IC, and provides time-series,
        distributional, and structural diagnostics.

        Parameters
        ----------
        staticPlot : bool, default False
            If True, renders the figure in static (non-interactive) mode.
            If False, enables full interactivity (zoom, hover, etc.).

        return_fig : bool, default False
            If True, returns the Plotly Figure object for further processing or testing.
            Otherwise, only displays the figure.

        Panels Included
        ---------------
        1. IC Time Series
            - Displays IC (or Rank IC) over time.
            - Includes rolling mean and horizontal mean IC line.
            - Annotates average IC value.

        2. Cumulative IC
            - Cumulative sum of IC values over time.
            - Helps assess long-term predictive consistency.

        3. Industry Contribution (Optional)
            - Only included if self.group_by is True.
            - Horizontal bar chart of IC contributions by industry.
            - Identifies sector-level alpha sources.

        4. IC Self-Correlation
            - Measures autocorrelation of IC over time.
            - Indicates signal persistence.
            - Includes mean autocorrelation line and annotation.

        5. Q-Q Plot
            - Compares empirical IC distribution to theoretical normal distribution.
            - Used to assess normality and tail behavior.

        6. Monthly Aggregated IC Heatmap
            - Heatmap of IC aggregated by month and horizon.
            - Useful for stability and seasonality analysis.

        7. IC Statistics Heatmap
            - Summary metrics (e.g., mean IC, ICIR) across horizons.
            - Enables quick comparison of predictive strength.

        Layout Design
        -------------
        - Dynamic number of rows based on:
            * number of return horizons
            * whether industry grouping is enabled
        - Manual domain allocation for precise vertical spacing.
        - Independent legends per subplot row for clarity.
        - Right-aligned legends and colorbars to avoid overlap.

        Returns
        -------
        fig : plotly.graph_objects.Figure, optional
            Returned only if return_fig=True.
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
            self.add_subtitle(fig,f'Q-Q Gragh',row)
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
            fig.update_yaxes(title_text="Rebalance Period", row=row, col=1, type="category", title_font=dict(size=16), tickfont=dict(size=18))
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
        """
        Generate a comprehensive multi-panel visualization for short-only factor performance analysis.

        This function constructs a vertically stacked Plotly figure to evaluate the performance of the
        short leg (typically bottom quantile) of a factor strategy across multiple rebalance periods.
        It includes rolling returns, cumulative returns, turnover, optional industry attribution, and
        summary heatmaps.

        Parameters
        ----------
        staticPlot : bool, default False
            If True, renders the figure in static (non-interactive) mode.
            If False, enables full interactivity (zoom, hover, etc.).

        return_fig : bool, default False
            If True, returns the Plotly Figure object for further processing or testing.
            Otherwise, only displays the figure.

        Panels Included
        ---------------
        1. Rolling Short Returns
            - Displays rolling mean returns for different quantiles under short perspective.
            - Typically reflects inverted interpretation (bottom quantile as alpha source).

        2. Cumulative Short Returns
            - Net value curves for short portfolios across quantiles.
            - Helps evaluate profitability and drawdowns of short strategies.
            - Supports optional log scaling (controlled by self.log_scale).

        3. Industry Analysis (Optional)
            - Only included if self.group_by is True.
            - Horizontal bar chart of average industry-level short returns.
            - Identifies sector contributions to short alpha.

        4. Turnover Rate
            - Scatter plot of turnover ratios for short portfolios.
            - Indicates trading frequency and potential transaction cost impact.

        5. Monthly Aggregated Return Heatmap
            - Heatmap of monthly short returns across rebalance periods.
            - Useful for detecting seasonality and consistency.

        6. Statistics Heatmap
            - Summary metrics (e.g., mean return, Sharpe ratio) for short strategies.
            - Enables comparison across rebalance periods.

        Layout Design
        -------------
        - Dynamic number of rows based on:
            * number of rebalance periods
            * whether industry grouping is enabled
        - Manual domain allocation for precise vertical spacing.
        - Independent legends per subplot row for clarity.
        - Right-aligned legends and colorbars to avoid overlap.

        Returns
        -------
        fig : plotly.graph_objects.Figure, optional
            Returned only if return_fig=True.
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
                
        colors = {"qtop": "lightgreen", "qmid": "limegreen", "qbot": "darkgreen"}
        line_width = 2
        row = 1
        for period in self.rebalance_periods:
            date = self.retruns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Short Return (Rebalance Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['ret_q1_rolling_s'],
                    mode="lines",
                    name=f"Q-Bottom Rolling {self.return_rolling_period} Mean",
                    line=dict(width=line_width, color=colors["qbot"]),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['ret_qm_rolling_s'],
                    mode="lines",
                    name=f"Q-{self.mid_q} Rolling {self.return_rolling_period} Mean",
                    line=dict(width=line_width, color=colors["qmid"]),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['ret_qb_rolling_s'],
                    mode="lines",
                    name=f"Q-Top Rolling {self.return_rolling_period} Mean",
                    line=dict(width=line_width, color=colors["qtop"]),
                    showlegend=True
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
            fig.update_yaxes(title_text="Return", row=row, col=1, title_font=dict(size=18), tickfont=dict(size=16))
            row += 1

        for period in self.rebalance_periods:
            date = self.retruns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Cumulative Short Returns (Rebalance Period = {period}, Log Scale = {self.log_scale})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['s_cum_nv_q1'],
                    mode="lines",
                    name="Q-Top Cumulative Return",
                    line=dict(width=line_width, color=colors["qtop"]),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['s_cum_nv_qm'],
                    mode="lines",
                    name=f"Q-{self.mid_q} Cumulative Return",
                    line=dict(width=line_width, color=colors["qmid"]),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['s_cum_nv_qb'],
                    mode="lines",
                    name="Q-Bottom Cumulative Return",
                    line=dict(width=line_width, color=colors["qbot"]),
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
                df:pd.DataFrame = self.indus_returns_dict[period][['industry',"i_s_ret_mean"]].sort_values(by='i_s_ret_mean', ascending=True)
                self.add_subtitle(fig,f'Average Industry Q-Bottom Short Return (Rebalance Period = {period})',row)
                fig.add_trace(
                    go.Bar(
                        x=df["i_s_ret_mean"],
                        y=df["industry"],
                        orientation='h',
                        marker=dict(
                            color=df["i_s_ret_mean"],
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
            date = self.retruns_dict[period][self.trade_date_col].values
            self.add_subtitle(fig,f'Turnover Rate (Rebalance Period = {period})',row)
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['turnover_q1'],
                    mode="markers",
                    name="Q-Top Turnover",
                    marker=dict(size=6),
                    line=dict(color=colors["qtop"], width=line_width),
                    showlegend=True,
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.retruns_dict[period]['turnover_qb'],
                    mode="markers",
                    name="Q-Bottom Turnover",
                    marker=dict(size=6),
                    line=dict(color=colors["qbot"], width=line_width),
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

        try:
            monthly_panel = self.s_monthly_panel
            if monthly_panel is None or getattr(monthly_panel, "shape", (0,))[0] == 0:
                raise ValueError("s_monthly_panel empty")
            self.add_subtitle(fig,'Q-Bottom Short Return Statistics',row,y=1.1)
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
                    texttemplate="%{text}",
                    hovertemplate="Period: %{y}<br>Month: %{x}<br>Return: %{z:.4f}<extra></extra>"
                ),
                row=row, col=1
            )
            fig.update_xaxes(title_text=f"Per {self.freq}", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=8))
            fig.update_yaxes(title_text="Rebalance Period", row=row, col=1, type="category", title_font=dict(size=18), tickfont=dict(size=16))
        except Exception:
            fig.add_annotation(text="No stats panel available", row=row, col=1, showarrow=False)
        row += 1

        try:
            stats_df = self.s_stats_panel.set_index("period").T
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
                        x=1.04                           
                    ),
                    text=np.round(z_stats, 4),
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
            traces_per_row.append(3)
        for _ in range(n_periods):
            traces_per_row.append(3)
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


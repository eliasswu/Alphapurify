import pandas as pd
import polars as pl
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .FactorAnalyzer import FactorAnalyzer

class PortfolioExposures():
    """
    Portfolio_Exposures

    A portfolio-level exposure and return attribution analyzer designed
    for factor research workflows. This class evaluates how a factor-sorted
    portfolio is exposed to selected risk factors and decomposes portfolio
    returns into contributions from these exposures.

    The analyzer constructs quantile portfolios based on factor rankings,
    calculates forward returns, estimates cross-sectional regressions,
    and attributes portfolio performance to underlying risk exposures.

    Main functionalities include:
    - Quantile portfolio construction
    - Long / Short / Long-Short portfolio exposure analysis
    - Cross-sectional regression for factor attribution
    - Cumulative return decomposition
    - Visualization of exposure dynamics and attribution returns

    Parameters
    ----------
    base_df : pd.DataFrame or pl.DataFrame
        Input dataset containing price, factor, and exposure variables.

    trade_date_col : str
        Column name representing the timestamp.

    symbol_col : str
        Column name representing the asset identifier.

    price_col : str
        Column name used to compute forward returns.

    factor_name : str
        Factor column used for ranking assets into quantile portfolios.

    exposure_cols : list[str]
        List of exposure factor columns used in regression attribution.
        These typically represent risk factors such as:
        - size
        - value
        - momentum
        - volatility
        - industry factors

    rebalance_period : int, default 1
        Portfolio rebalancing interval measured in data periods.

    bins : int, default 5
        Number of quantile groups used in portfolio sorting.

    position : {"l", "s", "ls"}, default "l"
        Portfolio position type:
        - "l"  → long-only (top quantile)
        - "s"  → short-only (bottom quantile)
        - "ls" → long-short spread

    overnight : {"on", "off", "only"}, default "on"
        Controls how overnight returns are treated:
        - "on"   → include overnight returns
        - "off"  → exclude overnight returns
        - "only" → analyze overnight returns only

    fac_shift : int, optional
        Number of periods used to shift factor and exposure variables
        forward in time to prevent look-ahead bias.

    Attributes
    ----------
    base_df : pl.DataFrame
        Internal working dataset stored in Polars format.

    result_df : pl.DataFrame
        Final result dataset containing exposures, regression coefficients,
        attribution returns, and cumulative portfolio returns.

    exposure_cols : list[str]
        List of exposure factors used in regression analysis.

    bins : int
        Number of quantile portfolios.

    rebalance_period : int
        Rebalancing interval used in portfolio construction.

    Notes
    -----
    Portfolio Construction

    Assets are ranked cross-sectionally based on the specified factor
    and assigned into quantile portfolios. Depending on the `position`
    parameter, the portfolio can represent:

    - Top quantile (long)
    - Bottom quantile (short)
    - Long-short spread

    Return Attribution

    At each rebalance date, a cross-sectional OLS regression is estimated:

        future_return = intercept + β₁ * exposure₁ + ... + βₙ * exposureₙ

    The resulting coefficients are used to decompose portfolio returns
    into exposure-driven contributions.

    Visualization

    The class provides several plotting utilities:

    - plot_portfolio_exposures()
        Shows the evolution of portfolio exposures over time.

    - plot_portfolio_returns()
        Displays cumulative attribution returns for each exposure.

    - plot_portfolio_exposures_and_returns()
        Combined visualization of exposures and cumulative returns.

    Example
    -------
    >>> pe = PortfolioExposures(
    ...     base_df=data,
    ...     trade_date_col="datetime",
    ...     symbol_col="symbol",
    ...     price_col="close",
    ...     factor_name="momentum",
    ...     exposure_cols=["size", "value", "volatility"],
    ...     bins=5,
    ...     position="ls"
    ... )

    >>> pe.run()
    >>> pe.plot_portfolio_exposures()
    >>> pe.plot_portfolio_returns()

    This allows researchers to understand whether a factor portfolio's
    performance is driven by genuine alpha or unintended risk exposures.
    """
    def __init__(self,
                 base_df:pd.DataFrame,
                 trade_date_col:str,
                 symbol_col:str,
                 price_col:str,
                 factor_name:str,
                 exposure_cols:list,
                 rebalance_period:int = 1,
                 bins:int = 5,
                 position:str = 'l',
                 overnight: str = "on",
                 fac_shift: int | None = None):
        
        self.price_col = price_col 
        self.trade_date_col = trade_date_col
        self.symbol_col = symbol_col
        self.factor_name = factor_name
        self.overnight = overnight
        self.fac_shift = fac_shift
        self.exposure_cols = exposure_cols
        self.rebalance_period = rebalance_period
        self.bins = bins
        self.position = position
        
        if isinstance(base_df,pd.DataFrame):
            self.base_df= pl.from_pandas(base_df).drop_nulls([trade_date_col, symbol_col, price_col] + exposure_cols)
        else:
            self.base_df = base_df.drop_nulls([trade_date_col, symbol_col, price_col] + exposure_cols)
        self.base_df:pl.DataFrame = self.base_df.with_columns(pl.col(self.trade_date_col).cast(pl.Datetime)).sort([self.symbol_col,self.trade_date_col])
        self.td = self.base_df[self.trade_date_col][1] - self.base_df[self.trade_date_col][0]
        
        if self.fac_shift:
            self.base_df = self.base_df.with_columns(
                pl.col([self.factor_name] + exposure_cols).shift(self.fac_shift).over(self.symbol_col)
            )
        
        if self.bins < 2:
            raise ValueError(f"bins must be >= 2, got {self.bins}")
        
    def calc_stats(self):
        def _cross_section_ols(sub_df, factor_cols):
            y = sub_df["fut_ret"].to_numpy()
            X = sub_df.select(factor_cols).to_numpy()

            X = np.column_stack([np.ones(len(X)), X])

            XtX = X.T @ X
            XtX_inv = np.linalg.pinv(XtX)

            beta = XtX_inv @ (X.T @ y)

            return beta
        df = self.base_df.clone()
        
        df = df.with_columns([
            pl.when(pl.col(self.price_col) == 0).then(None).otherwise(pl.col(self.price_col)).alias(self.price_col),
            pl.when(pl.col(self.price_col).shift(self.rebalance_period).over(self.symbol_col) == 0)
            .then(None)
            .otherwise(pl.col(self.price_col).shift(self.rebalance_period).over(self.symbol_col))
            .alias("price_fut")
        ])
        df = df.with_columns(
            ((pl.col("price_fut") / pl.col(self.price_col)) - 1).alias("fut_ret")
        )
        df = df.with_columns(
            pl.when(pl.col("fut_ret") == -1).then(None).otherwise(pl.col("fut_ret")).alias("fut_ret")
        )
        
        all_dates = df[self.trade_date_col].unique().sort()
        rebalance_dates = all_dates[::self.rebalance_period]
        rebalance_mask = pl.col(self.trade_date_col).is_in(
            rebalance_dates.implode()
        )
        
        df = df.with_columns(
            pl.col(self.trade_date_col).max()
            .over([self.symbol_col, pl.col(self.trade_date_col).dt.date()])
            .alias("day_last_dt"),

            (pl.col(self.trade_date_col) + self.rebalance_period * self.td).alias("target_dt")
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
            [self.trade_date_col, self.symbol_col, self.factor_name, "fut_ret"] + self.exposure_cols
        )           
        
        df = df.with_columns([
            (
                (pl.col(f) - pl.col(f).mean().over(self.trade_date_col))
                / pl.col(f).std().over(self.trade_date_col)
            ).alias(f)
            for f in self.exposure_cols
        ]).drop_nulls([self.factor_name] + self.exposure_cols)
        
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
        df = df.drop_nulls(["fut_ret"])
        
        if self.position == 'ls':
            mean_df = df.filter((pl.col("quantile") == 1) | (pl.col("quantile") == self.bins))
            mean_df = mean_df.with_columns(
                    [
                        pl.when(pl.col("quantile") == self.bins)
                        .then(-pl.col(f))
                        .otherwise(pl.col(f))
                        .alias(f)
                        for f in self.exposure_cols
                    ]
                )
            mean_df = (
                mean_df
                .group_by(self.trade_date_col)
                .agg([
                    pl.col(f).mean().alias(f"{f}_expo")
                    for f in self.exposure_cols
                ])
                .sort(self.trade_date_col)
            )
        elif self.position == 'l':
            mean_df = df.filter((pl.col("quantile") == 1))
            mean_df = (
                mean_df
                .group_by(self.trade_date_col)
                .agg([
                    pl.col(f).mean().alias(f"{f}_expo")
                    for f in self.exposure_cols
                ])
                .sort(self.trade_date_col)
            )
        elif self.position == 's':
            mean_df = df.filter((pl.col("quantile") == self.bins))
            mean_df = mean_df.with_columns(
                    [
                        pl.when(pl.col("quantile") == self.bins)
                        .then(-pl.col(f))
                        for f in self.exposure_cols
                    ]
                )
            mean_df = (
                mean_df
                .group_by(self.trade_date_col)
                .agg([
                    pl.col(f).mean().alias(f"{f}_expo")
                    for f in self.exposure_cols
                ])
                .sort(self.trade_date_col)
            )
        else:
            raise 
            
        betas = []
        for dt, sub_df in df.group_by(self.trade_date_col):
            beta = _cross_section_ols(sub_df, self.exposure_cols)
            dt = sub_df[self.trade_date_col][0]
            betas.append(
                [dt] + beta.tolist()
            )

        beta_df = pl.DataFrame(
            betas,
            schema=[self.trade_date_col,  "intercept"] + self.exposure_cols,
            orient="row"
        )
        
        result_df = mean_df.join(
            beta_df,
            on=self.trade_date_col,
            how="inner"
        )
        result_df = result_df.fill_null(0)
        result_df = result_df.sort(self.trade_date_col)
        result_df = (
            result_df
            .sort(self.trade_date_col)
            .with_columns(
                [
                    (pl.col(f"{f}_expo") * pl.col(f)).alias(f"{f}_attr")
                    for f in self.exposure_cols
                ]
            )
            .with_columns(
                [
                    (
                        (1 + pl.col(f"{f}_attr"))
                        .cum_prod()
                        - 1
                    ).alias(f"{f}_cum_ret")
                    for f in self.exposure_cols
                ]
            )
            .with_columns(
                [
                    (
                        (1 + pl.col("intercept"))
                        .cum_prod()
                        - 1
                    ).alias("intercept_cum_ret")
                ]
            )
            .with_columns(
                (
                    pl.col("intercept") +
                    sum(pl.col(f"{f}_attr") for f in self.exposure_cols)
                ).alias("portfolio_ret")
            )
            .with_columns(
                (
                    (1 + pl.col("portfolio_ret"))
                    .cum_prod()
                    - 1
                ).alias("portfolio_cum_ret")
            )
        )
        return result_df

    def run(self):
        self.result_df = self.calc_stats()
        
    def plot_portfolio_exposures(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_expo")]
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    mode="lines",
                    name=col,
                    marker=dict(size=6),
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title=dict(
                text=f"Exposures Over Time ({self.factor_name})",
                font=dict(
                    size=24,          
                    family="Arial",  
                    color="black"    
                ),),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(
                font=dict(size=16)
            ),height=550
            
        )
        fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=self.result_df[self.trade_date_col],
                title_text="Date",
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig
    
    def plot_portfolio_returns(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_cum_ret")] 
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    line=dict(width=2),
                    mode="lines",
                    name=col
                )
            )

        fig.update_layout(
            title=dict(
                text=f"Cummulative Returns of Exposures over time ({self.factor_name})",
                font=dict(
                    size=24,          
                    family="Arial",  
                    color="black"    
                ),),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(
                font=dict(size=16)
            ),height=550
        )
        
        fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=self.result_df[self.trade_date_col],
                title_text="Date",
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))
        
        
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig
    
    def plot_portfolio_exposures_and_returns(self, staticPlot: bool = False, return_fig: bool = False):

        n_rows = len(self.exposure_cols)

        specs = [[{"secondary_y": True}] for _ in range(n_rows)]
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

            axis_index = 2 * i - 1

            yaxis_name = "yaxis" if axis_index == 1 else f"yaxis{axis_index}"
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"

            fig.layout[yaxis_name].update(domain=dom)

            anchor_name = "y" if axis_index == 1 else f"y{axis_index}"
            fig.layout[xaxis_name].anchor = anchor_name

        date = self.result_df[self.trade_date_col]

        for row, factor in enumerate(self.exposure_cols, start=1):
            FactorAnalyzer.add_subtitle(fig,f'{factor}', row, Exposures = True)
            expo_col = f"{factor}_expo"
            attr_col = f"{factor}_cum_ret"

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[expo_col],
                    fill="tozeroy",
                    name="Exposure" if row == 1 else None,
                    marker=dict(size=6),
                    line=dict(width=2,color='lightblue'),
                    showlegend=False
                ),
                row=row,
                col=1,
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[attr_col],
                    marker=dict(size=6),
                    name="Cumulative Return" if row == 1 else None,
                    line=dict(width=2,color='red'),
                    showlegend=False
                ),
                row=row,
                col=1,
                secondary_y=True
            )

            fig.update_yaxes(
                title_text=f"{factor} Exposure",
                row=row,
                col=1,
                secondary_y=False,
                title_font=dict(size=18), 
                tickfont=dict(size=16)
            )

            fig.update_yaxes(
                title_text="Cumulative Return",
                tickformat=".2f",
                row=row,
                col=1,
                secondary_y=True, 
                title_font=dict(size=18), 
                tickfont=dict(size=16)
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
        
        height_per_row = 550

        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=100, b=60, l=60, r=60),
            title=dict(
                text=f"Portfolio Returns & Exposures ({self.factor_name})",   
                font=dict(size=24, family="Arial", color="black")
            )
        )
        fig.update_layout(base_layout)
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig

class PureExposures():
    """
    Pure_Exposures

    A factor exposure decomposition tool that evaluates how a signal
    portfolio loads on predefined risk factors and attributes its
    returns into exposure-driven contributions and pure alpha.

    The class constructs a normalized factor-weighted portfolio,
    estimates cross-sectional regressions, and decomposes portfolio
    performance into risk-factor contributions and intercept alpha.

    It also provides diagnostics including factor correlations,
    exposure dynamics, and attribution-based cumulative returns.

    Parameters
    ----------
    base_df : pd.DataFrame or pl.DataFrame
        Input dataset containing asset prices, factor signals,
        and exposure variables.

    trade_date_col : str
        Column representing the timestamp of observations.

    symbol_col : str
        Column representing the asset identifier.

    price_col : str
        Column used to compute forward returns.

    factor_name : str
        Column name of the alpha signal whose pure exposure
        characteristics are analyzed.

    exposure_cols : list[str]
        List of risk factor columns used for exposure estimation
        and cross-sectional regression.

        Examples may include:
        - size
        - value
        - momentum
        - volatility
        - liquidity
        - industry exposures

    overnight : {"on", "off", "only"}, default "on"
        Controls how overnight returns are handled:

        - "on"   → include all returns
        - "off"  → exclude overnight returns
        - "only" → analyze overnight returns only

    fac_shift : int, optional
        Number of periods used to shift factor and exposure variables
        forward in time to prevent look-ahead bias.

    Attributes
    ----------
    base_df : pl.DataFrame
        Internal working dataset stored in Polars format.

    result_df : pl.DataFrame
        Final dataframe containing exposures, attribution returns,
        alpha contributions, and cumulative portfolio returns.

    corr_df : pd.DataFrame
        Time-series correlation between the alpha factor and each
        exposure factor.

    corr_matrix : pd.DataFrame
        Cross-sectional correlation matrix among factors and exposures.

    Notes
    -----
    Portfolio Construction

    The alpha factor is standardized cross-sectionally and converted
    into portfolio weights:

        weight_i = factor_i / Σ |factor_i|

    This produces a dollar-neutral factor portfolio whose exposures
    can be analyzed against predefined risk factors.

    Return Attribution

    At each timestamp, a cross-sectional regression is estimated:

        future_return = intercept + β₁ * exposure₁ + ... + βₙ * exposureₙ

    The regression coefficients are combined with the portfolio's
    factor exposures to decompose returns into:

    - exposure-driven contributions
    - intercept alpha

    Exposure Contribution

    Exposure attribution is calculated as:

        contribution_t = exposure_t × β_t

    Cumulative returns are then obtained via compounding.

    Diagnostics

    The class also computes:

    - time-series correlations between the alpha factor and exposures
    - average factor correlations
    - full factor correlation matrices
    - correlation distribution diagnostics

    Visualization

    The class provides several plotting utilities:

    - plot_pure_exposures()
        Displays exposure dynamics over time.

    - plot_pure_returns()
        Shows cumulative attribution returns of exposures and alpha.

    - plot_pure_exposures_and_returns()
        Combined visualization of exposures and cumulative returns.

    - plot_correlations()
        Diagnostic plots for factor correlations including
        time-series correlation, mean correlation, correlation
        matrix heatmap, and distribution histograms.

    Example
    -------
    >>> pe = PureExposures(
    ...     base_df=data,
    ...     trade_date_col="datetime",
    ...     symbol_col="symbol",
    ...     price_col="close",
    ...     factor_name="momentum",
    ...     exposure_cols=["size", "value", "volatility"]
    ... )

    >>> pe.run()

    >>> pe.plot_pure_exposures()
    >>> pe.plot_pure_returns()
    >>> pe.plot_correlations()

    This allows researchers to determine whether an alpha signal
    generates genuine alpha or derives its performance from
    unintended risk exposures.
    """
    def __init__(self,
                 base_df:pd.DataFrame,
                 trade_date_col:str,
                 symbol_col:str,
                 price_col:str,
                 factor_name:str,
                 exposure_cols:list,
                 overnight: str = "on",
                 fac_shift: int | None = None):
        
        self.price_col = price_col 
        self.trade_date_col = trade_date_col
        self.symbol_col = symbol_col
        self.factor_name = factor_name
        self.overnight = overnight
        self.fac_shift = fac_shift
        self.exposure_cols = exposure_cols
        
        if isinstance(base_df,pd.DataFrame):
            self.base_df= pl.from_pandas(base_df).drop_nulls([trade_date_col, symbol_col, price_col] + exposure_cols)
        else:
            self.base_df = base_df.drop_nulls([trade_date_col, symbol_col, price_col] + exposure_cols)
        self.base_df:pl.DataFrame = self.base_df.with_columns(pl.col(self.trade_date_col).cast(pl.Datetime)).sort([self.symbol_col,self.trade_date_col])
        self.td = self.base_df[self.trade_date_col][1] - self.base_df[self.trade_date_col][0]
        
        if self.fac_shift:
            self.base_df = self.base_df.with_columns(
                pl.col([self.factor_name] + exposure_cols).shift(self.fac_shift).over(self.symbol_col)
            )
        
    def calc_stats(self):
        def _cross_section_ols(sub_df):
            y = sub_df["fut_ret"].to_numpy()
            X = sub_df.select(self.exposure_cols).to_numpy()

            X = np.column_stack([np.ones(len(X)), X])

            XtX = X.T @ X
            XtX_inv = np.linalg.pinv(XtX)

            beta = XtX_inv @ (X.T @ y)

            return beta
        
        def _cross_section_corr(df:pl.DataFrame):
            df = df.to_pandas()
            corr_panel = (
                df
                .groupby(self.trade_date_col)
                .corr()
            )
            
            corr = (
                df.drop(columns=self.trade_date_col)
                .corr()
            )
            
            fac_corr = (
                corr_panel
                .xs(self.factor_name, level=1)
                .drop(columns=[self.factor_name])
            ).reset_index()
            
            return fac_corr, corr
        
        df = self.base_df.clone()
        
        df = df.with_columns([
            pl.when(pl.col(self.price_col) == 0).then(None).otherwise(pl.col(self.price_col)).alias(self.price_col),
            pl.when(pl.col(self.price_col).shift(1).over(self.symbol_col) == 0)
            .then(None)
            .otherwise(pl.col(self.price_col).shift(1).over(self.symbol_col))
            .alias("price_fut")
        ])
        df = df.with_columns(
            ((pl.col("price_fut") / pl.col(self.price_col)) - 1).alias("fut_ret")
        )
        df = df.with_columns(
            pl.when(pl.col("fut_ret") == -1).then(None).otherwise(pl.col("fut_ret")).alias("fut_ret")
        )
        
        df = df.with_columns(
            pl.col(self.trade_date_col).max()
            .over([self.symbol_col, pl.col(self.trade_date_col).dt.date()])
            .alias("day_last_dt"),

            (pl.col(self.trade_date_col) + self.td).alias("target_dt")
        )

        df = df.with_columns(
            (pl.col("target_dt") > pl.col("day_last_dt"))
            .alias("is_overnight")
        )

        if self.overnight == "off":
            mask = (~pl.col("is_overnight"))
            df = df.filter(mask).select(
                        [self.trade_date_col, self.symbol_col, self.factor_name, "fut_ret"] + self.exposure_cols
                    )  

        elif self.overnight == "only":
            mask = (pl.col("is_overnight"))       
            df = df.filter(mask).select(
                        [self.trade_date_col, self.symbol_col, self.factor_name, "fut_ret"] + self.exposure_cols
                    )     
        
        df = df.with_columns([
            (
                (pl.col(f) - pl.col(f).mean().over(self.trade_date_col))
                / pl.col(f).std().over(self.trade_date_col)
            ).alias(f)
            for f in self.exposure_cols + [self.factor_name]
        ]).drop_nulls([self.factor_name] + self.exposure_cols)
        ######################
        
        df = df.with_columns(
                (
                    pl.col("fac")
                    / pl.col("fac").abs().sum().over(self.trade_date_col)
                ).alias("weight")
                )
        ##########################
        mean_df = (
            df
            .with_columns(
                [
                    (pl.col("weight") * pl.col(f)).alias(f"{f}_w")
                    for f in self.exposure_cols
                ]
            )
            .group_by(self.trade_date_col)
            .agg(
                [
                    pl.sum(f"{f}_w").alias(f"{f}_expo")
                    for f in self.exposure_cols
                ] 
            )
            .sort(self.trade_date_col)
        )
               
        betas = []
        for dt, sub_df in df.group_by(self.trade_date_col):
            beta = _cross_section_ols(sub_df)
            dt = sub_df[self.trade_date_col][0]
            betas.append(
                [dt] + beta.tolist()
            )

        beta_df = pl.DataFrame(
            betas,
            schema=[self.trade_date_col, "intercept"] + self.exposure_cols,
            orient="row"
        )
        
        result_df = mean_df.join(
            beta_df,
            on=self.trade_date_col,
            how="inner"
        )
        
        result_df = result_df.fill_null(0)
        result_df = result_df.sort(self.trade_date_col)
        result_df = (
            result_df
            .sort(self.trade_date_col)
            .with_columns(
                [
                    (pl.col(f"{f}_expo") * pl.col(f)).alias(f"{f}_attr")
                    for f in self.exposure_cols
                ]
            )
            .with_columns(
                [
                    (
                        (1 + pl.col(f"{f}_attr"))
                        .cum_prod()
                        - 1
                    ).alias(f"{f}_cum_ret")
                    for f in self.exposure_cols
                ]
            )
            .with_columns(
                [
                    (
                        (1 + pl.col("intercept"))
                        .cum_prod()
                        - 1
                    ).alias("Alpha")
                ]
            )
            .with_columns(
                (
                    pl.col("intercept") +
                    sum(pl.col(f"{f}_attr") for f in self.exposure_cols)
                ).alias("portfolio_ret")
            )
            .with_columns(
                (
                    (1 + pl.col("portfolio_ret"))
                    .cum_prod()
                    - 1
                ).alias(f"{self.factor_name}_cum_ret")
            )
        )
        ##############################################
        corr_df = df.select([self.trade_date_col,self.factor_name]+self.exposure_cols)
        
        corr_df, corr = _cross_section_corr(corr_df)
        return result_df, corr_df, corr

    def run(self):
        self.result_df, self.corr_df, self.corr_matrix = self.calc_stats()
        
    def plot_pure_exposures(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_expo")]
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    mode="lines",
                    name=col,
                    marker=dict(size=6),
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title=dict(
                text=f"Exposures Over Time ({self.factor_name})",
                font=dict(
                    size=24,          
                    family="Arial",  
                    color="black"    
                ),),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(
                font=dict(size=16)
            ),height=550
            
        )
        fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=self.result_df[self.trade_date_col],
                title_text="Date",
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig
    
    def plot_pure_returns(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_cum_ret") or col.endswith("Alpha")] 
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    line=dict(width=2),
                    mode="lines",
                    name=col
                )
            )

        fig.update_layout(
            title=dict(
                text=f"Cummulative Returns of Exposures over time ({self.factor_name})",
                font=dict(
                    size=24,          
                    family="Arial",  
                    color="black"    
                ),),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(
                font=dict(size=16)
            ),height=550
        )
        
        fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=self.result_df[self.trade_date_col],
                title_text="Date",
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))
        
        
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig
    
    def plot_pure_exposures_and_returns(self, staticPlot: bool = False, return_fig: bool = False):

        n_rows = len(self.exposure_cols)

        specs = [[{"secondary_y": True}] for _ in range(n_rows)]
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

            axis_index = 2 * i - 1

            yaxis_name = "yaxis" if axis_index == 1 else f"yaxis{axis_index}"
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"

            fig.layout[yaxis_name].update(domain=dom)

            anchor_name = "y" if axis_index == 1 else f"y{axis_index}"
            fig.layout[xaxis_name].anchor = anchor_name

        date = self.result_df[self.trade_date_col]

        for row, factor in enumerate(self.exposure_cols, start=1):
            FactorAnalyzer.add_subtitle(fig,f'{factor}', row, Exposures = True)
            expo_col = f"{factor}_expo"
            attr_col = f"{factor}_cum_ret"

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[expo_col],
                    fill="tozeroy",
                    name="Exposure" if row == 1 else None,
                    marker=dict(size=6),
                    line=dict(width=2,color='lightblue'),
                    showlegend=False
                ),
                row=row,
                col=1,
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[attr_col],
                    marker=dict(size=6),
                    name="Cumulative Return" if row == 1 else None,
                    line=dict(width=2,color='red'),
                    showlegend=False
                ),
                row=row,
                col=1,
                secondary_y=True
            )

            fig.update_yaxes(
                title_text=f"{factor} Exposure",
                row=row,
                col=1,
                secondary_y=False,
                title_font=dict(size=18), 
                tickfont=dict(size=16)
            )

            fig.update_yaxes(
                title_text="Cumulative Return",
                tickformat=".2f",
                row=row,
                col=1,
                secondary_y=True, 
                title_font=dict(size=18), 
                tickfont=dict(size=16)
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
        
        height_per_row = 550

        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=100, b=60, l=60, r=60),
            title=dict(
                text=f"Pure Returns & Exposures ({self.factor_name})",   
                font=dict(size=24, family="Arial", color="black")
            )
        )
        fig.update_layout(base_layout)
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig

    def plot_correlations(self, staticPlot: bool = False, return_fig: bool = False):
        n_cols = len(self.exposure_cols)
        n_rows = n_cols + 3
        
        specs = [[{"type": "xy"}], [{"type": "xy"}], [{"type": "heatmap"}]]
        specs2 = [[{"type": "xy"}] for _ in range(n_cols)]
        specs += specs2
        
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
        
        date = self.corr_df[self.trade_date_col]
        FactorAnalyzer.add_subtitle(fig,'Factor Correlations Over Time',1)
        for factor in self.exposure_cols:
            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.corr_df[factor],
                    marker=dict(size=6),
                    name=factor,
                    line=dict(width=2),
                    showlegend=True
                ),
                row=1,
                col=1,
            )

            fig.update_yaxes(
                title_text="Value",
                row=1,
                col=1,
                secondary_y=False,
                title_font=dict(size=18), 
                tickfont=dict(size=16)
            )

            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=1, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
            
        s = self.corr_df.mean().drop('datetime').sort_values()
        FactorAnalyzer.add_subtitle(fig,'Mean Correlations',2)
        fig.add_trace(
            go.Bar(
                y=s.index,
                x=s.values,
                name="Mean Correlations",
                orientation="h",
                showlegend=False,
                marker=dict(
            color=s.values,          
            colorscale="RdYlGn",     
            showscale=False
        ),
            ),
            row=2,
            col=1
        )
        
        fig.update_xaxes(
                title_text="Value",
                row=2,
                col=1,
                title_font=dict(size=18), 
                tickfont=dict(size=16)
            )

        fig.update_yaxes(
                type="category",
                categoryorder="array",
                categoryarray=s.index,
                row=2, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=16)
            )
        
        FactorAnalyzer.add_subtitle(fig,'Factor Correlations Matrix',3)
        
        fig.add_trace(
            go.Heatmap(
                z=self.corr_matrix.values,
                x=self.corr_matrix.columns,
                y=self.corr_matrix.index,
                colorscale="RdYlGn",
                zmin=-1,
                zmid=0,
                zmax=1,
                showscale=True,
                colorbar=dict(
                        title="Value",       
                        title_side="top",
                        yanchor="middle",
                        y=1 - (3 - 0.625) / n_rows,  
                        len=(1 / n_rows)*0.7,                 
                        x=1.04                           
                    ),
                text=np.round(self.corr_matrix.values, 4),
                texttemplate="%{text}"
            ),
            row=3,
            col=1
        )
        
        fig.update_yaxes(
                type="category",
                categoryorder="array",
                categoryarray=self.corr_matrix.index,
                row=3, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=16)
            )

        fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=self.corr_matrix.columns,
                row=3, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=16)
            )
        
        for row, factor in enumerate(self.exposure_cols, start=4):
            FactorAnalyzer.add_subtitle(fig,f'Correlations Distribution of {factor}', row)
            fig.add_trace(
                go.Histogram(
                    x=self.corr_df[factor],
                    nbinsx=40,
                    name=factor,
                    showlegend=True
                ),
                row=row,
                col=1
            )
        
        
        height_per_row = 550
        base_layout = dict(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            margin=dict(t=120, b=80, l=80, r=200),
            title=dict(
                text=f"Factor Correlations ({self.factor_name})",   
                font=dict(size=24, family="Arial", color="black")
            )
        )
        legend_name_for_row = lambda r: "legend" if r == 1 else f"legend{r}"
        traces_per_row = []
        traces_per_row.append(n_cols)
        traces_per_row.append(1)
        traces_per_row.append(1)
        for _ in range(n_cols):
            traces_per_row.append(1)

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
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig
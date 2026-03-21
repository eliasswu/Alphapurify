import polars as pl
import pandas as pd


class AlphaPurifier():
    """
    AlphaPurifier

    A flexible factor preprocessing engine designed for quantitative
    research pipelines. This class provides a unified interface for
    cleaning, transforming, and neutralizing cross-sectional factor data
    before factor evaluation or model training.

    AlphaPurifier supports a wide range of statistical and machine
    learning-based preprocessing techniques, including outlier treatment,
    factor neutralization, and standardization.

    The class follows a method-chaining design, allowing multiple
    preprocessing steps to be applied sequentially in a concise workflow.

    Main functionalities include:
    - Outlier treatment (winsorization / robust compression)
    - Factor neutralization (linear models, robust regression, ML methods)
    - Cross-sectional factor standardization
    - Rolling normalization for time-series stability
    - Method discovery via registry inspection

    Parameters
    ----------
    base_df : pd.DataFrame or pl.DataFrame
        Input dataset containing factor values and identifier columns.

    factor_name : str
        Name of the factor column to be processed.

    trade_date_col : str
        Column name representing the timestamp.

    symbol_col : str
        Column name representing the asset identifier.

    Attributes
    ----------
    df : pl.DataFrame
        Internal working dataset stored in Polars format for
        high-performance transformations.

    cols : list[str]
        Original column ordering preserved for final output.

    factor_name : str
        Target factor column used in preprocessing operations.

    Notes
    -----
    Factor preprocessing usually follows the standard pipeline:

        1. Winsorization (remove extreme outliers)
        2. Neutralization (remove risk exposures such as size/industry)
        3. Standardization (normalize factor distribution)

    AlphaPurifier enables this pipeline through method chaining:

    >>> purifier = AlphaPurifier(df, factor_name="alpha", trade_date_col="datetime", symbol_col="symbol")
    >>> result = (
    ...     purifier
    ...     .winsorize("mad")
    ...     .neutralize("multiOLS", ["log_mktcap", "industry"])
    ...     .standardize("zscore")
    ...     .to_result()
    ... )

    Method Registry
    ---------------
    Available preprocessing methods can be inspected dynamically:

    >>> AlphaPurifier.get_methods()
    >>> AlphaPurifier.get_methods("winsorize")
    >>> AlphaPurifier.get_methods("neutralize", "multiOLS")

    This helps users explore available implementations and their parameters.
    """
    def __init__(self, base_df: pd.DataFrame, factor_name: str, trade_date_col:str, symbol_col:str):
        
        self.factor_name = factor_name
        self.trade_date_col = trade_date_col
        self.symbol_col = symbol_col
        
        if isinstance(base_df,pd.DataFrame):
            self.df= pl.from_pandas(base_df)
        else:
            self.df = base_df.with_columns(pl.col(self.trade_date_col).cast(pl.Datetime)).sort([self.symbol_col,self.trade_date_col])

        self.cols = self.df.columns
    @staticmethod
    def get_methods(method: str = None, sub_method: str = None):
        if not method and not sub_method:
            print("Available main categories of methods:\n")
            for cat in METHOD_REGISTRY.keys():
                print(f" - {cat} ({len(METHOD_REGISTRY[cat])} implementations)\n")
        elif method in METHOD_REGISTRY and sub_method is None:
            print(f"Available methods under [{method}]:\n")
            for m, info in METHOD_REGISTRY[method].items():
                print(f" - {m}: {info['description']}\n")
        elif method in METHOD_REGISTRY and sub_method in METHOD_REGISTRY[method]:
            info = METHOD_REGISTRY[method][sub_method]
            print(f"Method: {sub_method}\n")
            print(f"Description: {info['description']}\n")
            print("Parameters:\n")
            for p, d in info["params"].items():
                print(f"  - {p}: {d}\n")
        else:
            print("The specified category or method was not found.")
        
    def winsorize(self, method: str = 'mad', *args):
        if method == 'mad':
            self.df = mad_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'mean_std':
            self.df = mean_std_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'volatility':
            self.df = volatility_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'iqr':
            self.df = iqr_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'quantile':
            self.df = quantile_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'rolling_quantile':
            self.df = rolling_quantile_winsorize(self.df, self.trade_date_col, self.symbol_col, self.factor_name, *args)
        elif method == 'boxcox_compress':
            self.df = boxcox_compress_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'zscore':
            self.df = zscore_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'rankgauss':
            self.df = rankgauss_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'tanh':
            self.df = tanh_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'huber':
            self.df = huber_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        elif method == 'ransac':
            self.df = ransac_winsorize(self.df, self.trade_date_col, self.factor_name, *args)
        else:
            raise NotImplementedError(f"{method} not found")
        return self
    
    def neutralize(self, method: str = 'multiOLS', *args):
        if method == 'multiOLS':
            self.df = multiOLS_neutralize(self.df,self.trade_date_col,self.factor_name,*args)
        elif method == 'lasso':
            self.df = lasso_neutralize(self.df,self.trade_date_col,self.factor_name, *args)
        elif method == 'ridge':
            self.df = ridge_neutralize(self.df,self.trade_date_col,self.factor_name, *args)
        elif method == 'elasticnet':
            self.df = elasticnet_neutralize(self.df,self.trade_date_col,self.factor_name, *args)
        elif method == 'polynomial':
            self.df = polynomial_neutralize(self.df,self.trade_date_col,self.factor_name, *args)
        elif method == 'kernelridge':
            self.df = kernelridge_neutralize(self.df,self.trade_date_col,self.factor_name, *args)  
        elif method == 'huber':
            self.df = huber_neutralize(self.df,self.trade_date_col,self.factor_name, *args) 
        elif method == 'rank':
            self.df = rank_neutralize(self.df,self.trade_date_col,self.factor_name, *args)
        elif method == 'theilsen':
            self.df = theilsen_neutralize(self.df,self.trade_date_col,self.factor_name, *args)
        elif method == 'randomforest':
            self.df = randomforest_neutralize(self.df,self.trade_date_col,self.factor_name, *args)
        elif method == 'GBDT':
            self.df = GBDT_neutralize(self.df,self.trade_date_col,self.factor_name, *args)
        elif method == 'ICA':
            self.df = ICA_neutralize(self.df,self.trade_date_col,self.factor_name, *args)   
        elif method == 'PCA':
            self.df = PCA_neutralize(self.df,self.trade_date_col,self.factor_name, *args)    
        elif method == 'bayesianridge':
            self.df = bayesianridge_neutralize(self.df,self.trade_date_col,self.factor_name, *args)
        elif method == 'partialcorrelation':
            self.df = partialcorrelation_neutralize(self.df,self.trade_date_col,self.factor_name, *args) 
        else:
            raise NotImplementedError(f"{method} not found")   
        return self

    def standardize(self, method: str = 'zscore',*args):
        if method == 'zscore':
            self.df = zscore_standardize(self.df, self.trade_date_col, self.factor_name)
        elif method == 'robust_zscore':
            self.df = robust_zscore_standardize(self.df, self.trade_date_col, self.factor_name,*args)
        elif method == 'minmax':
            self.df = minmax_standardize(self.df, self.trade_date_col, self.factor_name,*args)
        elif method == 'rank':
            self.df = rank_standardize(self.df, self.trade_date_col, self.factor_name,*args)
        elif method == 'rank_gaussianize':
            self.df = rank_gaussianize_standardize(self.df, self.trade_date_col, self.factor_name,*args)
        elif method == 'rolling':
            self.df = rolling_standardize(self.df, self.trade_date_col, self.symbol_col, self.factor_name,*args)
        elif method == 'rolling_robust':
            self.df = rolling_robust_standardize(self.df, self.trade_date_col, self.symbol_col, self.factor_name,*args)
        elif method == 'rolling_minmax':
            self.df = rolling_minmax_standardize(self.df, self.trade_date_col, self.symbol_col, self.factor_name,*args)
        elif method == 'volatility_scaling':
            self.df = volatility_scaling_standardize(self.df, self.trade_date_col, self.symbol_col, self.factor_name,*args)
        elif method == 'EWMA':
            self.df = EWMA_standardize(self.df, self.trade_date_col, self.symbol_col, self.factor_name,*args)
        elif method == 'normal_scores':
            self.df = normal_scores_standardize(self.df, self.trade_date_col, self.factor_name,*args)
        elif method == 'quantile_binning':
            self.df = quantile_binning_standardize(self.df, self.trade_date_col, self.factor_name,*args)
        elif method == 'log_zscore':
            self.df = log_zscore_standardize(self.df, self.trade_date_col, self.factor_name,*args)
        elif method == 'boxcox':
            self.df = boxcox_standardize(self.df, self.trade_date_col, self.factor_name,*args)
        elif method == 'yeo_johnson':
            self.df = yeo_johnson_standardize(self.df, self.trade_date_col, self.factor_name,*args)
        else:
            raise NotImplementedError(f"{method} not found")
        return self

    def to_result(self,cols:list=None)-> pd.DataFrame:
        self.df = self.df.select(self.cols)
        if cols:
            return self.df.to_pandas()[cols]
        else:
             return self.df.to_pandas()


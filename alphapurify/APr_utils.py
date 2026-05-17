#Copyright 2025 @Elias Wu
import scipy.stats as stats
import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, norm
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso, Ridge, ElasticNet, HuberRegressor, RANSACRegressor, LinearRegression, TheilSenRegressor, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.kernel_ridge import KernelRidge

from datetime import timedelta

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

METHOD_REGISTRY = {
    'winsorize': {
        'ransac': {
            'description': "Apply RANSAC-based regression cleaning to remove or suppress extreme factor values within each cross-section.",
            'params': {
                'residual_threshold': ' Residual cutoff expressed in standard deviation units. Default is 2.5',
                'replace_with_fit': 'Whether to replace outliers with the RANSAC fitted values. If False, the extreme values are clipped to the allowable residual range.Default is True'
            }
        },
        'huber': {
            'description': "Apply Huber-based winsorization (Huber Regression Clipping) to robustly limit extreme factor values within each cross-section.",
            'params': {
                'c': 'Huber clipping threshold. Controls the strength of outlier suppression. default is 2.0'
            }
        },
        'tanh': {
            'description': 'Apply Tanh-based winsorization (Tanh Normalization) to reduce the impact of extreme factor values.',
            'params': {
                'scale': 'Scaling factor applied before tanh to adjust compression intensity. Default is 1.0'
            }
        },
        'zscore': {
            'description': 'Apply cross-sectional outlier compression using the Box–Cox (or log) transformation.',
            'params': {
                'k': 'Z-score clipping threshold. Default is 3.'
            }
        },
        'rankgauss': {
            'description': "Apply RankGauss (Quantile Normalization) to transform factor values within each cross-section to approximate a standard normal distribution, reducing the influence of outliers while preserving rank information.",
            'params': {
                'trade_date': "The column name used to define each cross-section (group).",
                'factor_col': "The name of the factor column to be transformed via RankGauss normalization."
            }
        },
        'boxcox_compress': {
            'description': 'Apply cross-sectional outlier compression using the Box–Cox (or log) transformation.',
            'params': {
                'λ': 'Box–Cox parameter. Common values range from 0.25 to 0.5 and λ = 0 corresponds to a log transform.'
            }
        },
        'rolling_quantile': {
            'description': "Apply time-series rolling quantile-based winsorization (Rolling Quantile Clipping) to robustly limit extreme factor values within each symbol over time. Quantile thresholds are computed using a rolling window along the time dimension, grouped by symbol.",
            'params': {
                'window': 'Rolling window size (number of past observations). Default is 252 (approximately one trading year).',
                'lower_q': 'Lower quantile threshold within the rolling window. Default is 0.01 (1%).',
                'upper_q': 'Upper quantile threshold within the rolling window. Default is 0.99 (99%).'
            }
        },
        'quantile': {
            'description': 'Apply cross-sectional quantile-based winsorization (Quantile Clipping).',
            'params': {
                'lower_q': 'Lower quantile threshold. Default is 0.01 (1%).',
                'upper_q': 'Upper quantile threshold. Default is 0.99 (99%).'
            }
        },
        'iqr': {
            'description': 'Apply cross-sectional winsorization using the Interquartile Range (IQR) method.',
            'params': {
                'k': 'Winsorization multiplier applied to the IQR. Common values are 1.5 or 3.'
            }
        },
        'volatility': {
            'description': 'Apply cross-sectional percentage-based winsorization using mean ± k * standard deviation.',
            'params': {
                'k': 'Winsorization threshold multiplier applied as mean ± k * std.'
            }
        },
        'mad': {
            'description': 'Apply cross-sectional winsorization to a factor column using the MAD method: Median ± n * MAD.',
            'params': {
                'n': 'Winsorization threshold multiplier. Typically 3 (Median ± 3 * MAD).'
            }
        },
        'mean_std': {
            'description': 'Perform cross-sectional winsorization on a factor column using Mean ± n * Std.',
            'params': {
                'n': 'Winsorization threshold multiplier. For example, n=3 applies mean ± 3σ clipping.'
            }
        }
    },
    "neutralize":{
        "multiOLS": {
            'description': "Perform cross-sectional multi-factor OLS neutralization on a factor column.",
            'params': {
                "neutralizer_cols : list of str" : "Continuous control variables used in the OLS regression.",
                "dummy_cols : list of str, optional" : "Categorical columns from which dummy variables are created. These are included as regressors in the OLS.",
                "scale_X : bool, default False": "Whether to globally standardize continuous neutralizer variables before regression.",
                "n_jobs : int, default -1": "Number of parallel processes used for cross-sectional regressions. -1 uses all available CPU cores."
            }
        },
        "lasso": {
            'description': "Perform cross-sectional factor neutralization using Lasso regression.",
            'params': {
                "neutralizer_cols : list of str" : "Continuous control variables used in the OLS regression.",
                "dummy_cols : list of str, optional" : "Categorical columns from which dummy variables are created. These are included as regressors in the OLS.",
                "scale_X : bool, default False": "Whether to globally standardize continuous neutralizer variables before regression.",
                "alpha : float, default 0.01": "Regularization strength for Lasso (L1 penalty). Higher values yield sparser models.",
                "fit_intercept : bool, default True": "Whether Lasso should fit an intercept term.",
                "max_iter : int, default 1000": "Maximum number of iterations for Lasso optimization.",
                "tol : float, default 1e-4": "Convergence tolerance for Lasso optimization.",
                "warm_start : bool, default False": "Whether to reuse the solution of the previous fit to speed up computation.",
                "random_state : int, optional": "Random seed for reproducibility."
            }
        },
        "ridge": {
            'description': "Perform cross-sectional factor neutralization using Ridge regression.",
            'params': {
                "neutralizer_cols : list of str" : "Continuous control variables used in the OLS regression.",
                "dummy_cols : list of str, optional" : "Categorical columns from which dummy variables are created. These are included as regressors in the OLS.",
                "scale_X : bool, default False": "Whether to globally standardize continuous neutralizer variables before regression.",
                "alpha : float, default 0.01": "Regularization strength for Ridge (L1 penalty). Higher values yield sparser models.",
                "fit_intercept : bool, default True": "Whether Ridge should fit an intercept term.",
                "max_iter : int, default 1000": "Maximum number of iterations for Ridge optimization.",
                "tol : float, default 1e-4": "Convergence tolerance for Ridge optimization.",
                "warm_start : bool, default False": "Whether to reuse the solution of the previous fit to speed up computation.",
                "random_state : int, optional": "Random seed for reproducibility."
            }
        },
        "elasticnet": {
            'description': "Perform cross-sectional factor neutralization using Elastic Net regression.",
            'params': {
                "neutralizer_cols : list of str" : "Continuous control variables used in the OLS regression.",
                "dummy_cols : list of str, optional" : "Categorical columns from which dummy variables are created. These are included as regressors in the OLS.",
                "scale_X : bool, default False": "Whether to globally standardize continuous neutralizer variables before regression.",
                "alpha : float, default 1.0": "Overall regularization strength applied to the Elastic Net model.",
                "l1_ratio : float, default 0.5": "The proportion of L1 (Lasso) vs. L2 (Ridge) penalty (l1_ratio = 1 → Lasso, l1_ratio = 0 → Ridge, 0 < l1_ratio < 1 → Elastic Net)",
                "fit_intercept : bool, default True": "Whether to fit an intercept term in the regression.",
                "max_iter : int, default 1000": "Maximum number of optimization iterations.",
                "tol : float, default 1e-4": "Convergence tolerance for the solver.",
                "warm_start : bool, default False": "Whether to reuse the solution of the previous fit to speed up computation.",
                "random_state : int, optional": "Random seed for reproducibility."
            }
        },
        "polynomial": {
                    'description': "Perform cross-sectional factor neutralization using Polynomial Regression.",
                    'params': {
                        "neutralizer_cols : list of str" : "Continuous control variables used in the OLS regression.",
                        "dummy_cols : list of str, optional" : "Categorical columns from which dummy variables are created. These are included as regressors in the OLS.",
                        "scale_X : bool, default False": "Whether to globally standardize continuous neutralizer variables before regression.",
                        "degree : int, default 2": "Degree of the polynomial expansion. (degree = 1 → linear model, degree = 2 → squares and pairwise interactions, degree = 3 → cubic terms, etc.)",
                        "interaction_only : bool, default False": "If True, only interaction terms are generated (no squared terms).",
                        "include_bias : bool, default True": "Whether to include a bias (constant) column in the polynomial features."
            }
        },
        "kernelridge": {
                    'description': "Perform cross-sectional factor neutralization using Kernel Ridge Regression (KRR).",
                    'params': {
                        "neutralizer_cols : list of str" : "Continuous control variables used in the OLS regression.",
                        "dummy_cols : list of str, optional" : "Categorical columns from which dummy variables are created. These are included as regressors in the OLS.",
                        "scale_X : bool, default False": "Whether to globally standardize continuous neutralizer variables before regression.",
                        "kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default 'rbf'": "Kernel type used by Kernel Ridge Regression.",
                        "alpha : float, default 1.0": "Regularization strength. Larger values enforce stronger shrinkage and reduce overfitting.",
                        "gamma : float, optional": "Kernel width parameter for RBF, polynomial, and sigmoid kernels. If None, scikit-learn defaults to 1 / n_features.",
                        "degree : int, default 3": "Polynomial degree used when kernel='poly'",
                        "coef0 : float, default 1.0": "Independent term in polynomial and sigmoid kernels."
            }
        },
        "huber": {
                    'description': "Perform robust cross-sectional factor neutralization using Huber Regression.",
                    'params': {
                        "neutralizer_cols : list of str" : "Continuous control variables used in the OLS regression.",
                        "dummy_cols : list of str, optional" : "Categorical columns from which dummy variables are created. These are included as regressors in the OLS.",
                        "scale_X : bool, default False": "Whether to globally standardize continuous neutralizer variables before regression.",
                        "epsilon : float, default 1.35": "Threshold parameter for the Huber loss. Larger values approach OLS; smaller values increase robustness to outliers.",
                        "max_iter : int, default 100": "Maximum number of optimization iterations.",
                        "alpha : float, default 0.0": "L2 regularization strength (Ridge penalty).",
                        "warm_start : bool, default False": "Whether to reuse the previous solution as initialization.",
                        "tol : float, default 1e-5": "Numerical tolerance for convergence."
            }
        },
        "random_forest": {
            "description": "Perform nonlinear cross-sectional factor neutralization using Random Forest regression.",
            "params": {
                "neutralizer_cols : list of str": "Continuous control variables used as regressors in the Random Forest model.",
                "dummy_cols : list of str, optional": "Categorical columns from which dummy variables are created and included as model inputs.",
                "scale_X : bool, default False": "Whether to globally standardize continuous neutralizer variables before model fitting.",
                "n_estimators : int, default 150": "Number of trees in the forest.",
                "max_depth : int or None, optional": "Maximum depth of each decision tree. If None, nodes are expanded until pure or below min_samples_split.",
                "min_samples_split : int or float, default 2": "Minimum number of samples required to split an internal node.",
                "min_samples_leaf : int or float, default 1": "Minimum number of samples required to be at a leaf node.",
                "min_weight_fraction_leaf : float, default 0": "Minimum weighted fraction of total sample weight required at a leaf node.",
                "max_features : int, float, str or None, default 1": "Number of features to consider when searching for the best split.",
                "max_leaf_nodes : int or None, optional": "Maximum number of leaf nodes per tree.",
                "min_impurity_decrease : float, default 0": "Minimum impurity decrease required to perform a split.",
                "bootstrap : bool, default True": "Whether bootstrap samples are used when building trees.",
                "oob_score : bool, default False": "Whether to use out-of-bag samples to estimate generalization performance.",
                "random_state : int, optional": "Random seed for reproducibility.",
                "warm_start : bool, default False": "Whether to reuse the solution of the previous fit and add more estimators.",
                "ccp_alpha : float, default 0": "Complexity parameter used for Minimal Cost-Complexity Pruning.",
                "max_samples : int, float or None, optional": "If bootstrap=True, number of samples to draw from X to train each tree."
            }
        },
    "rank": {
    "description": "Perform robust cross-sectional factor neutralization using RANSAC Regression.",
    "params": {
        "neutralizer_cols : list of str": "List of continuous columns used as neutralizers.",
        "dummy_cols : list of str, optional": "List of categorical columns to be one-hot encoded. Default is None.",
        "scale_X : bool, default False": "Whether to standardize the neutralizer variables.",
        "residual_threshold : float, default 2.5": "Maximum allowed residual for a sample to be classified as an inlier in RANSAC.",
        "max_trials : int, default 100": "Maximum number of random iterations for model estimation.",
        "min_samples : float or int, default 0.5": "Minimum number (or proportion) of samples required to estimate the model.",
        "random_state : int, optional": "Seed for reproducibility."
    }
},
    "theilsen": {
    "description": "Perform robust cross-sectional factor neutralization using the Theil–Sen Regressor.",
    "params": {
        "neutralizer_cols : list of str": "List of continuous columns used as neutralizers.",
        "dummy_cols : list of str, optional": "List of categorical columns to be one-hot encoded. Default is None.",
        "scale_X : bool, default False": "Whether to standardize the neutralizer variables before fitting.",
        "fit_intercept : bool, default True": "Whether to fit an intercept in the regression model.",
        "max_subpopulation : int or float, default 10000": "Maximum subpopulation size used internally by Theil–Sen estimator to limit computational cost.",
        "n_subsamples : int or None, optional": "Number of samples used per iteration. If None, scikit-learn chooses a robust default based on the number of predictors.",
        "max_iter : int, default 300": "Maximum number of iterations in the estimator.",
        "tol : float, default 0.001": "Convergence tolerance for the iterative procedure.",
        "random_state : int, optional": "Random seed for reproducibility (affects subpopulation sampling)."
    }
},
    "GBDT": {
    "description": "Perform nonlinear cross-sectional factor neutralization using Gradient Boosting Decision Trees (GBDT).",
    "params": {
        "neutralizer_cols : list of str": "List of continuous columns used as neutralizers.",
        "dummy_cols : list of str, optional": "List of categorical columns to be one-hot encoded. Default is None.",
        "scale_X : bool, default False": "Whether to standardize the neutralizer variables before fitting.",
        "learning_rate : float, default 0.1": "Shrinkage factor applied to each tree.",
        "n_estimators : int, default 100": "Number of boosting stages (trees).",
        "subsample : float, default 1.0": "Fraction of samples used per boosting stage. Values < 1.0 improve generalization by introducing randomness.",
        "min_samples_split : int, default 2": "Minimum number of samples required to split an internal node.",
        "min_samples_leaf : int, default 1": "Minimum samples required to be at a leaf node.",
        "min_weight_fraction_leaf : float, default 0.0": "Minimum weighted fraction of samples in a leaf node.",
        "max_depth : int, default 3": "Maximum depth of each regression tree.",
        "min_impurity_decrease : float, default 0.0": "Minimum impurity decrease required to perform a split.",
        "init : estimator or None, optional": "Initial estimator for boosting. If None, use the mean estimator.",
        "random_state : int, optional": "Random seed for reproducibility.",
        "alpha : float, default 0.9": "Quantile used in loss functions such as 'quantile' (if enabled).",
        "max_leaf_nodes : int or None, optional": "Maximum number of leaf nodes per tree.",
        "warm_start : bool, default False": "Whether to reuse the fitted solution for additional boosting stages.",
        "validation_fraction : float, default 0.1": "Fraction of data used for early stopping when n_iter_no_change is set.",
        "n_iter_no_change : int or None, optional": "Stops training if validation loss does not improve for this many iterations.",
        "tol : float, default 1e-4": "Tolerance for early stopping.",
        "ccp_alpha : float, default 0.0": "Complexity parameter used for minimal cost-complexity pruning."
    }
},
    "ICA": {
    "description": "Perform cross-sectional factor neutralization using Independent Component Analysis (ICA).",
    "params": {
        "neutralizer_cols : list of str": "List of numeric columns used as neutralizers.",
        "dummy_cols : list of str, optional": "List of categorical columns to be one-hot encoded. Default is None.",
        "scale_X : bool, default False": "Whether to standardize neutralizer variables before ICA.",
        "n_components : int or None, optional": "Number of ICA components. If None, defaults to min(n_features, n_samples).",
        "max_iter : int, default 200": "Maximum number of iterations for the FastICA solver.",
        "tol : float, default 1e-4": "Tolerance for the FastICA convergence condition.",
        "random_state : int or None, optional": "Random seed for reproducibility.",
        "return_components : bool, default False": "If True, the output will include the extracted ICA components (columns 'ICA_1', 'ICA_2', ...)."
    }
},
    "PCA": {
    "description": "Perform cross-sectional factor neutralization using Principal Component Analysis (PCA). Neutralizer variables (continuous and optionally dummy-encoded categorical variables) are transformed into orthogonal principal components, and the factor is regressed onto these components within each cross-section (grouped by trade_date_col). The residuals represent the PCA-neutralized factor.",
    "params": {
        "neutralizer_cols : list of str": "List of numeric columns used as neutralizers.",
        "dummy_cols : list of str, optional": "List of categorical columns to be one-hot encoded. Default is None.",
        "scale_X : bool, default True": "Whether to standardize the neutralizer variables before applying PCA.",
        "n_components : int or None, optional": "Number of principal components to retain. If None, automatically uses min(10, n_features, n_samples).",
        "tol : float, default 0": "Tolerance parameter for randomized SVD solver (if applicable).",
        "n_oversamples : int, default 10": "Additional number of random vectors used in randomized SVD to improve approximation accuracy.",
        "random_state : int or None, optional": "Random seed for reproducibility."
    }
},
    "bayesianridge": {
    "description": "Perform factor neutralization using Bayesian Ridge Regression. Bayesian Ridge introduces precision priors on both coefficients and noise, providing a more stable, regularized regression compared to OLS or standard ridge. The residuals of the regression are returned as the neutralized factor.",
    "params": {
        "neutralizer_cols : list of str": "List of numeric columns used as neutralizing variables.",
        "dummy_cols : list of str, optional": "List of categorical variables to be one-hot encoded. Default is None.",
        "scale_X : bool, default False": "Whether to standardize the neutralizer features.",
        "n_iter : int, default 300": "Maximum number of iterations for BayesianRidge.",
        "tol : float, default 0.001": "Convergence tolerance of the optimization.",
        "alpha_1, alpha_2 : float, default 1e-6": "Hyperparameters for the Gamma prior over the noise precision.",
        "lambda_1, lambda_2 : float, default 1e-6": "Hyperparameters for the Gamma prior over the weight precision.",
        "alpha_init : float, optional": "Initial value for the noise precision.",
        "lambda_init : float, optional": "Initial value for the weight precision.",
        "fit_intercept : bool, default True": "Whether the BayesianRidge model should estimate an intercept.",
        "compute_score : bool, default False": "Whether to compute the log-marginal-likelihood score during training.",
        "return_coef : bool, default False": "If True, output an additional column 'coef_mean' representing the mean of the posterior coefficient estimates for diagnostic purposes."
    }
},
   
    "partialcorrelation": {
    "description": "Perform cross-sectional factor neutralization using Partial Correlation. This method removes the linear effects of neutralizer variables from the factor, and optionally computes the partial correlation between the factor and a target variable after controlling for the neutralizers.",
    "params": {
        "neutralizer_cols : list of str": "Numeric columns used for neutralization.",
        "dummy_cols : list of str, optional": "Categorical variables to be automatically one-hot encoded.",
        "scale_X : bool, default False": "Whether to standardize the neutralizer columns before processing.",
        "target_col : str, optional": "If provided, compute the partial correlation between the factor and this target variable after controlling for neutralizers.",
        "method : {'pearson', 'spearman'}, default 'pearson'": "Correlation method used when computing partial correlation.",
        "return_type : {'neutralized_factor', 'partial_corr'}, default 'neutralized_factor'": "- 'neutralized_factor': return cross-sectionally neutralized factor values for each row.\n- 'partial_corr': return one partial correlation value per cross-section."
    }
}
    },
    "standardize":{
        "zscore": {
            "description": "Perform cross-sectional Z-score standardization on a factor column.",
            "params": {}
        },
        "robust_zscore": {
            "description": "Perform cross-sectional robust Z-score standardization using median and MAD.",
            "params": {
                "c : float, default 1.4826": "Consistency factor applied to MAD to make it comparable to standard deviation under normality."
            }
        },
        "minmax": {
            "description": "Perform cross-sectional Min–Max (0–1) standardization on a factor column.",
            "params": {
                "clip : bool, default True": "Whether to clamp scaled values to the [0, 1] interval after Min–Max normalization."
            }
        },
        "rank": {
            "description": "Perform cross-sectional rank-based percentile standardization on a factor column.",
            "params": {
                "method : str, default 'average'": "Ranking method used to handle ties (e.g., 'average', 'min', 'max', 'dense', 'ordinal')."
            }
        },
        "rank_gaussianize": {
            "description": "Perform cross-sectional rank-based Gaussianization using percentile ranks and inverse normal CDF.",
            "params": {
                "method : str, default 'average'": "Ranking method used to handle ties before percentile-to-Gaussian transformation."
            }
        },
        "rolling": {
            "description": "Perform rolling time-series Z-score standardization grouped by stock code.",
            "params": {
                "window : int, default 20": "Rolling window size used to compute trailing mean and standard deviation.",
                "min_periods : int | None, default None": "Minimum number of observations required within the window to produce a valid standardized value."
            }
        },
        "rolling_robust": {
            "description": "Perform rolling robust Z-score standardization using median and MAD, grouped by stock code.",
            "params": {
                "window : int, default 20": "Rolling window size used to compute median and MAD.",
                "min_periods : int | None, default None": "Minimum number of observations required within the window to compute a valid robust rolling Z-score."
            }
        },
        "rolling_minmax": {
            "description": "Perform rolling Min–Max normalization on a factor column, grouped by stock code.",
            "params": {
                "window : int, default 20": "Rolling window size used to compute rolling minimum and maximum.",
                "min_periods : int | None, default None": "Minimum number of observations required within the window to compute a valid rolling Min–Max value."
            }
        },
        "volatility_scaling": {
            "description": "Perform volatility-scaling standardization on a factor column using rolling standard deviation, optionally shifted to avoid look-ahead bias.",
            "params": {
                "window : int, default 20": "Rolling window size used to compute the rolling standard deviation (volatility).",
                "min_periods : int | None, default None": "Minimum number of observations required within the window to compute a valid rolling volatility.",
                "shift_vol : bool, default True": "Whether to shift the rolling volatility by one period (σ(t-1)) to prevent look-ahead bias."
            }
        },
        "EWMA": {
            "description": "Perform EWMA-based volatility standardization on a factor column, scaling by exponentially weighted moving average of past squared values.",
            "params": {
                "lambda_ : float, default 0.94": "Exponential decay parameter for EWMA. Higher values place more weight on recent observations and produce smoother volatility estimates.",
                "eps : float, default 1e-12": "Small constant added to the denominator to prevent division by zero when volatility is very small."
            }
        },
        "normal_scores": {
            "description": "Perform rank-based normal score transformation on a factor column, converting cross-sectional ranks to standard normal quantiles.",
            "params": {
                "eps : float, default 1e-9": "Small clipping constant to avoid probabilities 0 or 1 entering the inverse normal CDF, preventing ±∞ values."
            }
        },
        "quantile_binning": {
            "description": "Perform cross-sectional quantile binning on a factor column, assigning each observation to one of q bins within each datetime slice.",
            "params": {
                "q : int, default 5": "Number of quantile bins. Determines into how many discrete buckets the factor values are divided (e.g., q=5 → quintiles)."
            }
        },
        "log_zscore": {
            "description": "Apply Log-Zscore standardization to a factor column (log transform → cross-sectional Z-score), computed independently for each datetime slice.",
            "params": {
                "eps : float, default 1e-9": "Small positive constant added before log and during Z-score normalization to avoid log(0) and division by zero."
            }
        },
        "yeo_johnson": {
            "description": "Apply a Yeo–Johnson power transform followed by cross-sectional Z-score standardization for each datetime slice, handling both positive and negative values.",
            "params": {
                "lambda_ : float, default 0.0": "Power transform parameter controlling the nonlinearity of the Yeo–Johnson transform. Lambda=0 gives log-like transform for x>=0, lambda=2 gives log-like transform for x<0, other values follow standard formula.",
                "eps : float, default 1e-9": "Small constant added during Z-score standardization to avoid division by zero and ensure numerical stability."
            }
        },
        "boxcox": {
            "description": "Apply a Box–Cox power transform followed by cross-sectional Z-score standardization for each datetime slice. Automatically shifts the factor to ensure positivity, reduces skewness, and stabilizes variance.",
            "params": {
                "lambda_ : float, default 0.0": "Power transform parameter. Lambda=0 → log transform, else → y = (x^λ − 1)/λ.",
                "eps : float, default 1e-9": "Small constant added to avoid numerical issues when shifting factor values or dividing by near-zero standard deviations."
            }
        }
    }
}

def mean_std_winsorize(base_df: pl.DataFrame, trade_date_col, factor_col: str, n: float = 3.0) -> pl.DataFrame:
    """
    Perform cross-sectional winsorization on a factor column using Mean ± n * Std.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date_col : str
    factor_col : str
        Name of the factor column to be winsorized.
    n : float
        Winsorization threshold multiplier. For example, n=3 applies mean ± 3σ clipping.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor column
        has been winsorized by clipping values outside Mean ± n * Std within each
        cross-section.
    """
    fac = pl.col(factor_col)
    
    result = (
        base_df.join(
            base_df.group_by(trade_date_col)
            .agg([
                fac.mean().alias("mean"),
                fac.std().alias("std")
            ]),
            on=trade_date_col,
            how="left"
        )
        .with_columns([
            (fac.clip(pl.col("mean") - n * pl.col("std"),
                      pl.col("mean") + n * pl.col("std")))
            .alias(factor_col)
        ])
    )
    return result

def mad_winsorize(base_df: pl.DataFrame, trade_date_col, factor_col: str, n: float = 3.0) -> pl.DataFrame:
    """
    Apply cross-sectional winsorization to a factor column using the MAD method:
    Median ± n * MAD.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date_col :str
    factor_col : str
        Name of the factor column to be winsorized.
    n : float
        Winsorization threshold multiplier. Typically 3 (Median ± 3 * MAD).

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor
        column has been winsorized by clipping values outside
        Median ± n * MAD within each cross-section.
    """
    fac = pl.col(factor_col)

    stats = (
        base_df.group_by(trade_date_col)
        .agg([
            fac.median().alias("median"),
            ((fac - fac.median()).abs().median()).alias("mad")
        ])
    )

    result = (
        base_df.join(stats, on=trade_date_col, how="left")
        .with_columns([
            fac.clip(
                pl.col("median") - n * pl.col("mad"),
                pl.col("median") + n * pl.col("mad")
            ).alias(factor_col)
        ])
    )

    return result

def volatility_winsorize(base_df: pl.DataFrame, trade_date_col, factor_col: str, k: float = 2.0) -> pl.DataFrame:
    """
    Apply cross-sectional percentage-based winsorization using mean ± k * standard deviation.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date_col : str
    factor_col : str
        Name of the factor column to be winsorized.
    k : float
        Winsorization threshold multiplier applied as mean ± k * std.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor
        column has been winsorized by clipping values exceeding mean ± k * std
        within each cross-section.
    """
    stats = (
        base_df.group_by(trade_date_col)
        .agg([
            pl.mean(factor_col).alias("mean"),
            pl.std(factor_col).alias("std")
        ])
    )

    df = base_df.join(stats, on=trade_date_col, how="left")

    df = df.with_columns([
        pl.when(df[factor_col] > df["mean"] + k * df["std"])
          .then(df["mean"] + k * df["std"])
          .when(df[factor_col] < df["mean"] - k * df["std"])
          .then(df["mean"] - k * df["std"])
          .otherwise(df[factor_col])
          .alias(factor_col)
    ])

    return df

def iqr_winsorize(base_df: pl.DataFrame, trade_date_col, factor_col: str, k: float = 1.5) -> pl.DataFrame:
    """
    Apply cross-sectional winsorization using the Interquartile Range (IQR) method.

    For each cross-section:
    - Q1 and Q3 are the 25th and 75th percentiles.
    - IQR = Q3 - Q1
    - Lower bound  = Q1 - k * IQR
    - Upper bound  = Q3 + k * IQR
    Values outside these bounds are clipped accordingly.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date_col : str
    factor_col : str
        Name of the factor column to be winsorized.
    k : float
        Winsorization multiplier applied to the IQR. Common values are 1.5 or 3.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor
        column has been winsorized using the IQR-based clipping within each
        cross-section (grouped by 'datetime').
    """
    df = base_df.with_columns([
        pl.when(pl.col(factor_col).is_finite())
          .then(pl.col(factor_col))
          .otherwise(None)
          .alias(factor_col)
    ])

    stats = (
        df.group_by(trade_date_col, maintain_order=True)
          .agg([
              pl.col(factor_col).quantile(0.25, interpolation="linear").alias("q1"),
              pl.col(factor_col).quantile(0.75, interpolation="linear").alias("q3"),
          ])
          .with_columns([
              (pl.col("q3") - pl.col("q1")).alias("iqr")
          ])
    )

    df = df.join(stats, on=trade_date_col, how="left")

    lower = pl.col("q1") - k * pl.col("iqr")
    upper = pl.col("q3") + k * pl.col("iqr")

    df = df.with_columns([
        pl.when(pl.col(factor_col) < lower).then(lower)
         .when(pl.col(factor_col) > upper).then(upper)
         .otherwise(pl.col(factor_col))
         .alias(factor_col)
    ])

    return df

def quantile_winsorize(base_df: pl.DataFrame, trade_date_col, factor_col: str, lower_q: float = 0.01, upper_q: float = 0.99) -> pl.DataFrame:
    """
    Apply cross-sectional quantile-based winsorization (Quantile Clipping).

    For each cross-section (grouped by 'datetime'):
    - Compute the lower and upper quantiles (e.g., 1% and 99%).
    - Clip values below the lower quantile to the lower boundary.
    - Clip values above the upper quantile to the upper boundary.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date_col : str
    factor_col : str
        Name of the factor column to be winsorized.
    lower_q : float
        Lower quantile threshold. Default is 0.01 (1%).
    upper_q : float
        Upper quantile threshold. Default is 0.99 (99%).

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor
        column has been winsorized using cross-sectional quantile clipping.
    """
    df = base_df.with_columns([
        pl.when(pl.col(factor_col).is_finite())
          .then(pl.col(factor_col))
          .otherwise(None)
          .alias(factor_col)
    ])

    stats = (
        df.group_by(trade_date_col, maintain_order=True)
          .agg([
              pl.col(factor_col).quantile(lower_q, interpolation="linear").alias("q_low"),
              pl.col(factor_col).quantile(upper_q, interpolation="linear").alias("q_high"),
          ])
    )

    df = df.join(stats, on=trade_date_col, how="left")

    df = df.with_columns([
        pl.when(pl.col(factor_col) < pl.col("q_low")).then(pl.col("q_low"))
         .when(pl.col(factor_col) > pl.col("q_high")).then(pl.col("q_high"))
         .otherwise(pl.col(factor_col))
         .alias(factor_col)
    ])

    return df

def rolling_quantile_winsorize(
    df: pl.DataFrame,
    trade_date_col,
    symbol_col,
    factor_col: str,
    window: int = 252,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pl.DataFrame:
    """
    Apply time-series rolling quantile-based winsorization.

    For each individual symbol (grouped by `symbol_col`):
    - Sort observations by trade date.
    - Compute rolling lower and upper quantiles over a specified window
      (e.g., past 252 observations).
    - Clip values below the rolling lower quantile to the lower boundary.
    - Clip values above the rolling upper quantile to the upper boundary.

    This method performs winsorization along the time dimension
    for each symbol independently, rather than across the cross-section.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing time-series factor data.
    trade_date_col : str
        Name of the trade date column used for time ordering.
    symbol_col : str
        Name of the symbol (e.g., stock code) column used for grouping.
    factor_col : str
        Name of the factor column to be winsorized.
    window : int
        Rolling window size (number of past observations).
        Default is 252 (approximately one trading year).
    lower_q : float
        Lower quantile threshold within the rolling window.
        Default is 0.01 (1%).
    upper_q : float
        Upper quantile threshold within the rolling window.
        Default is 0.99 (99%).

    Returns
    -------
    pl.DataFrame
        A DataFrame where the factor column has been winsorized
        using rolling time-series quantile clipping for each symbol.
        """
    df = df.sort([symbol_col, trade_date_col])

    df = df.with_columns([
        pl.col(factor_col)
        .rolling_quantile(
            quantile=lower_q,
            window_size=window,
            min_periods=1
        )
        .over(symbol_col)
        .alias("q_low"),

        pl.col(factor_col)
        .rolling_quantile(
            quantile=upper_q,
            window_size=window,
            min_periods=1
        )
        .over(symbol_col)
        .alias("q_high"),
    ])

    df = df.with_columns(
        pl.when(pl.col(factor_col) < pl.col("q_low"))
        .then(pl.col("q_low"))
        .when(pl.col(factor_col) > pl.col("q_high"))
        .then(pl.col("q_high"))
        .otherwise(pl.col(factor_col))
        .alias(factor_col)
    )

    df = df.drop(["q_low", "q_high"])

    return df

def boxcox_compress_winsorize(base_df: pl.DataFrame, trade_date, factor_col: str, λ: float = 0.5) -> pl.DataFrame:
    """
    Apply cross-sectional outlier compression using the Box–Cox (or log) transformation.

    For each cross-section:
    - Shift the factor values so they are strictly positive.
    - Apply the Box–Cox transform with parameter λ.
    - When λ = 0, the transformation reduces to a log transform.
    - When λ ≠ 0, use ((x^λ - 1) / λ).

    This method compresses extreme values rather than clipping them.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date : str
    factor_col : str
        Name of the factor column to transform.
    λ : float
        Box–Cox parameter. Common values range from 0.25 to 0.5.
        λ = 0 corresponds to a log transform.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor
        column has been transformed using a cross-sectional Box–Cox (or log)
        compression method to reduce outlier influence.
    """
    df = base_df.clone()

    df = df.with_columns(
        (pl.col(factor_col) - pl.col(factor_col).min().over(trade_date) + 1e-9).alias("_shifted")
    )

    if λ == 0:
        df = df.with_columns(pl.col("_shifted").log().alias(factor_col))
    else:
        df = df.with_columns(
            (((pl.col("_shifted") ** λ) - 1) / λ).alias(factor_col)
        )

    return df

def zscore_winsorize(base_df: pl.DataFrame, trade_date, factor_col: str, k: float = 3.0) -> pl.DataFrame:
    """
    Apply cross-sectional Z-score winsorization (standard deviation clipping).

    For each cross-section:
    - Compute the mean and standard deviation of the factor values.
    - Convert values to Z-scores.
    - Clip values outside the range ±k standard deviations:
        lower bound = mean - k * std
        upper bound = mean + k * std

    This method suppresses extreme outliers based on standardized deviation.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date : str
    factor_col : str
        Name of the factor column to be winsorized.
    k : float
        Z-score clipping threshold. Default is 3.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor
        column has been winsorized using cross-sectional Z-score clipping.
    """
    df = base_df.clone()

    df = df.with_columns([
        pl.col(factor_col).mean().over(trade_date).alias("_mean"),
        pl.col(factor_col).std().over(trade_date).alias("_std")
    ])

    df = df.with_columns(
        pl.when(pl.col("_std") == 0).then(1e-9).otherwise(pl.col("_std")).alias("_std")
    )

    df = df.with_columns(
        ((pl.col(factor_col) - pl.col("_mean")) / pl.col("_std")).alias("_z")
    )

    df = df.with_columns(
        pl.when(pl.col("_z") > k).then(pl.col("_mean") + k * pl.col("_std"))
        .when(pl.col("_z") < -k).then(pl.col("_mean") - k * pl.col("_std"))
        .otherwise(pl.col(factor_col))
        .alias(factor_col)
    )

    # 返回三列
    return df

def rankgauss_winsorize(
    base_df: pl.DataFrame,
    trade_date,
    factor_col: str
) -> pl.DataFrame:
    """
    Apply RankGauss (Quantile Normalization) winsorization.

    For each cross-section:
    - Compute the factor's rank within the group.
    - Convert ranks to empirical quantiles using (rank − 0.5) / n.
    - Map these quantiles to the corresponding values of a standard normal
    distribution via the inverse CDF (Gaussian PPF).

    This transforms the factor into a distribution that approximates N(0, 1),
    reducing the influence of extreme values while preserving rank information.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date : str
    factor_col : str
        Name of the factor column to be normalized via RankGauss.

    Returns
    -------
    pl.DataFrame
        A DataFram, where the factor
        has been transformed by cross-sectional RankGauss normalization.
    """

    df = (
        base_df
        .with_columns(
            pl.col(factor_col)
            .rank(method="average")
            .over(trade_date)
            .alias("rank")
        )
        .with_columns(
            pl.len().over(trade_date).alias("n")
        )
        .with_columns(
            (((pl.col("rank") - 0.5) / pl.col("n")).clip(1e-6, 1 - 1e-6)).alias("quantile")
        )
    )

    df = df.with_columns(
        pl.col("quantile").map_batches(lambda q: stats.norm.ppf(q.to_numpy())).alias(factor_col)
    )

    result = df
    return result

def tanh_winsorize(
    base_df: pl.DataFrame,
    trade_date : str,
    factor_col: str,
    scale: float = 1.0
) -> pl.DataFrame:
    """
    Apply Tanh-based winsorization (Tanh Normalization) to reduce the impact of
    extreme factor values.

    For each cross-section:
    - Standardize the factor using z-score: (x − mean) / std.
    - Apply the hyperbolic tangent transformation tanh(z / scale), which
    smoothly compresses large deviations while preserving relative ordering.

    The parameter `scale` controls the strength of compression:
    - Smaller scale → stronger compression of extremes.
    - Larger scale → behavior approaches the original z-score.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date : str
    factor_col : str
        Name of the factor column to be compressed using tanh normalization.
    scale : float, default 1.0
        Scaling factor applied before tanh to adjust compression intensity.

    Returns
    -------
    pl.DataFrame
        A DataFrame with ['datetime', 'code', factor_col'], where the factor has
        been cross-sectionally normalized and compressed via tanh.
    """

    df = (
        base_df
        .with_columns([
            pl.col(factor_col).mean().over(trade_date).alias("mean"),
            pl.col(factor_col).std().over(trade_date).alias("std")
        ])
        .with_columns(
            ((pl.col(factor_col) - pl.col("mean")) / (pl.col("std") + 1e-12)).alias("zscore")
        )
        .with_columns(
            pl.col("zscore").map_batches(lambda x: np.tanh(x.to_numpy() / scale)).alias(factor_col)
        )
    )
    return df

def huber_winsorize(
    base_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    c: float = 2.0
) -> pl.DataFrame:
    """
    Apply Huber-based winsorization (Huber Regression Clipping) to robustly
    limit extreme factor values within each cross-section .

    This method standardizes the factor using z-scores and then applies the
    Huber clipping rule:

        z_clipped = sign(z) * min(|z|, c)

    The threshold `c` determines how strongly outliers are compressed:
    - Smaller c → stronger shrinkage of extreme deviations.
    - Typical values range from 1.5 to 3.

    After clipping in z-score space, the values are mapped back to the original
    factor scale.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date : str
    factor_col : str
        Name of the factor column to be processed using Huber clipping.
    c : float, default 2.0
        Huber clipping threshold. Controls the strength of outlier suppression.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor values
        have been robustly winsorized using the Huber transformation.
    """

    df = (
        base_df
        .with_columns([
            pl.col(factor_col).mean().over(trade_date).alias("mean"),
            pl.col(factor_col).std().over(trade_date).alias("std")
        ])
        .with_columns(
            ((pl.col(factor_col) - pl.col("mean")) / (pl.col("std") + 1e-12)).alias("zscore")
        )
    )

    df = df.with_columns(
        pl.col("zscore").map_batches(
            lambda z: np.sign(z.to_numpy()) * np.minimum(np.abs(z.to_numpy()), c)
        ).alias("z_clipped")
    )

    df = df.with_columns(
        (pl.col("z_clipped") * pl.col("std") + pl.col("mean")).alias(factor_col)
    )

    return df

def ransac_winsorize(
    base_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    residual_threshold: float = 2.5,
    replace_with_fit: bool = True,
) -> pl.DataFrame:
    """
    Apply RANSAC-based regression cleaning to remove or suppress extreme factor
    values within each cross-section.

    RANSAC (Random Sample Consensus) fits a robust linear model that identifies
    inliers based on residuals. Outliers—defined as points whose residuals exceed
    `residual_threshold` times the residual standard deviation—are treated in one
    of two ways:

    1. replace_with_fit = True  
    Outliers are replaced directly by the RANSAC-fitted values (robust smoothing).

    2. replace_with_fit = False  
    Outliers are clipped to the boundary:
        y_pred ± residual_threshold * std(residual)

    This method is useful when the factor shows structural patterns or local
    trends within each group, and you want to preserve the trend while removing
    erratic spikes.

    Parameters
    ----------
    base_df : pl.DataFrame
    trade_date : str
    factor_col : str
        The name of the factor column to be cleaned using RANSAC regression.
    residual_threshold : float, default 2.5
        Residual cutoff expressed in standard deviation units.
    replace_with_fit : bool, default True
        Whether to replace outliers with the RANSAC fitted values. If False, the
        extreme values are clipped to the allowable residual range.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the
        factor values have been robustly cleaned using RANSAC regression.
    """

    def _ransac_clean(df: pl.DataFrame) -> pl.DataFrame:
        y = df[factor_col].to_numpy().reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)
        if len(y) < 5:
            return df
        
        try:
            model = RANSACRegressor(base_estimator=LinearRegression(), random_state=0)
            model.fit(x, y)
            y_pred = model.predict(x)
            resid = y - y_pred
            std = np.std(resid)
            mask = np.abs(resid) <= residual_threshold * std

            if replace_with_fit:
                y_new = np.where(mask, y, y_pred)
            else:
                low, high = y_pred - residual_threshold * std, y_pred + residual_threshold * std
                y_new = np.clip(y, low, high)

            df = df.with_columns(pl.Series(factor_col, y_new.flatten()))
            return df
        except Exception:
            return df

    result = (
        base_df
        .group_by(trade_date, maintain_order=True)
        .map_groups(_ransac_clean)
    )
    return result


def preprocess_for_neutralization(
    pl_df: pl.DataFrame, 
    factor_col, 
    neutralizer_cols, 
    dummy_cols = None,
    scale_X: bool = False
):
    """
    Prepare inputs for cross-sectional factor neutralization.

    This function cleans the input DataFrame, optionally standardizes neutralizer
    variables, generates dummy variables, and returns a pandas DataFrame ready to
    be used for regression-based neutralization (e.g., OLS).

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing at least ['datetime', 'code', factor_col] and 
        all specified neutralizer and dummy columns.
    factor_col : str
        Name of the factor column that will later be neutralized.
    neutralizer_cols : list of str
        Continuous control variables used in neutralization (e.g., size, beta, volatility).
    dummy_cols : list of str, optional
        Categorical columns from which dummy variables will be created. 
        If None, no categorical encoding is performed.
    scale_X : bool, default False
        Whether to standardize neutralizer columns globally (mean-center or z-score).

    Returns
    -------
    Tuple[pd.DataFrame, list]
        - A pandas DataFrame containing the cleaned dataset with dummy variables added.
        - A list of column names representing all regressors used in neutralization,
        including continuous neutralizers and generated dummy variables.

    Notes
    -----
    - Rows with null values in any required columns are removed.
    - Dummy variables are generated using pandas.get_dummies(), with first-category drop.
    - Standardization (if enabled) is applied only to continuous neutralizer columns.
    """
    dummy_cols = dummy_cols or []

    df = pl_df.clone()

    if scale_X:
        df = df.with_columns([
            pl.when(pl.col(c).std() == 0)
            .then(pl.col(c) - pl.col(c).mean())           
            .otherwise((pl.col(c) - pl.col(c).mean()) / pl.col(c).std())
            .alias(c)
            for c in neutralizer_cols
        ])

    if dummy_cols:
        df_dummy = pd.get_dummies(df.select(dummy_cols).to_pandas(), drop_first=True, dtype=float)
        df = pl.concat([df, pl.from_pandas(df_dummy)], how="horizontal")
        dummy_cols_final = df_dummy.columns.to_list()
    else:
        dummy_cols_final = []
    
    X_cols = neutralizer_cols + dummy_cols_final
    df_pd = df.to_pandas().dropna(subset=[factor_col]+X_cols)
    for col in X_cols:
        df_pd[col] = pd.to_numeric(df_pd[col], errors="coerce")

    df_pd[factor_col] = pd.to_numeric(df_pd[factor_col], errors="coerce")
    
    return df_pd, X_cols
    
def multiOLS_neutralize(
    pl_df: pl.DataFrame, 
    trade_date_col,
    factor_col, 
    neutralizer_cols, 
    dummy_cols = None,
    scale_X: bool = False,
    ):
    """
    Perform cross-sectional multi-factor OLS neutralization on a factor column.

    This function neutralizes a factor by regressing it against a set of continuous
    neutralizer variables and optional dummy variables within each cross-section
    (grouped by 'datetime'). Regression is performed in parallel for efficiency,
    and the residuals from OLS are returned as the neutralized factor values.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] and all
        neutralizer / dummy columns required for regression.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list of str
        Continuous control variables used in the OLS regression.
    dummy_cols : list of str, optional
        Categorical columns from which dummy variables are created.  
        These are included as regressors in the OLS.  
        If None, no categorical encoding is applied.
    scale_X : bool, default False
        Whether to globally standardize continuous neutralizer variables before regression.
    

    Returns
    -------
    pl.DataFrame
        A DataFrame containing [trade_date_col, symbol_col, factor_col], where the factor
        column has been replaced by OLS residuals (i.e., the neutralized factor).

    Notes
    -----
    - Cross-sectional OLS is performed independently for each 'datetime'.
    - Dummy variable generation and preprocessing are handled by
    `preprocess_for_neutralization()`.
    - The OLS solution uses the pseudoinverse (pinv) to ensure numerical stability.
    - The intercept term is automatically added for all regressions.
    """
    df_pd, X_cols = preprocess_for_neutralization(pl_df,factor_col,neutralizer_cols,dummy_cols,scale_X)
        
    def _neutralize_group(g: pd.DataFrame):
        y = g[factor_col].to_numpy()

        if X_cols:
            X = g[X_cols].to_numpy()
            X = np.column_stack([np.ones(len(g)), X])  
        else:
            X = np.ones((len(g), 1))

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta

        g[factor_col] = y - y_pred

        return g

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def lasso_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    alpha: float = 0.01,
    fit_intercept=True,
    max_iter: int = 1000,
    tol: float = 0.0001,
    warm_start: bool = False,
    random_state = None
) -> pl.DataFrame:
    """
    Perform cross-sectional factor neutralization using Lasso regression.

    This function applies L1-regularized regression (Lasso) to neutralize a factor
    column against a set of continuous neutralizers and optional dummy variables,
    within each cross-section grouped by 'datetime'. The residuals from the Lasso
    model are returned as the neutralized factor values.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] along with
        specified neutralizer and dummy-variable columns.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list of str
        Continuous numerical variables to use as regressors for neutralization.
    dummy_cols : list of str, optional
        Categorical columns from which dummy variables will be generated.
        If None, no categorical encoding is performed.
    scale_X : bool, default False
        Whether to globally standardize continuous neutralizers before regression.
    alpha : float, default 0.01
        Regularization strength for Lasso (L1 penalty). Higher values yield sparser models.
    fit_intercept : bool, default True
        Whether Lasso should fit an intercept term.
    max_iter : int, default 1000
        Maximum number of iterations for Lasso optimization.
    tol : float, default 1e-4
        Convergence tolerance for Lasso optimization.
    warm_start : bool, default False
        Whether to reuse the solution of the previous fit to speed up computation.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        A DataFrame with [trade_date_col, symbol_col, factor_col], where the factor column
        has been replaced with Lasso residuals (i.e., the neutralized factor).

    Notes
    -----
    - Preprocessing (null removal, optional scaling, dummy creation) is handled
    by `preprocess_for_neutralization()`.
    - Lasso is applied independently within each datetime cross-section.
    - L1 regularization allows feature selection, which can be useful when many
    dummy variables or collinear neutralizers are present.
    - If no regressors are provided (X has zero columns), the factor is returned unchanged.
    - Parallelism is implemented via joblib to accelerate large universes.
    """
    
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, warm_start=warm_start, random_state=random_state)
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values

        if X.shape[1] > 0:
            model.fit(X, y)
            y_pred = model.predict(X)
            group[factor_col] = y - y_pred
        return group

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def ridge_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    alpha: float = 1.0,
    fit_intercept=True,
    max_iter: int = 1000,
    tol: float = 0.0001,
    random_state = None
) -> pl.DataFrame:
    """
    Perform cross-sectional factor neutralization using Lasso regression.

    This function applies L1-regularized linear regression (Lasso) to remove the
    influence of specified neutralizer variables—both continuous and dummy-encoded—
    from a target factor. Neutralization is performed independently within each
    cross-section grouped by 'datetime'. The returned factor values represent the
    residuals from the Lasso model, effectively yielding a cross-sectionally
    neutralized factor.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] together with
        all continuous neutralizers and categorical columns (if any).
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list of str
        Continuous numerical columns used as neutralization regressors.
    dummy_cols : list of str, optional
        Categorical variables to be one-hot encoded. If None, no dummy variables
        are generated.
    scale_X : bool, default False
        Whether to standardize continuous neutralizers globally before regression.
    alpha : float, default 0.01
        L1 regularization strength for Lasso. Larger values produce sparser models.
    fit_intercept : bool, default True
        Whether the Ridge model includes an intercept term.
    max_iter : int, default 1000
        Maximum number of optimization iterations for Lasso.
    tol : float, default 1e-4
        Convergence tolerance for the optimization solver.
    warm_start : bool, default False
        If True, reuse previous model coefficients to speed up fitting.
    random_state : int, optional
        Seed for random number generation, ensuring reproducibility.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing [trade_date_col, symbol_col, factor_col], where
        the factor values have been replaced with Lasso regression residuals,
        representing the neutralized factor.

    Notes
    -----
    - Preprocessing—including null removal, optional scaling, and dummy-variable
    generation—is handled by `preprocess_for_neutralization()`.
    - Each datetime slice is neutralized independently using a separate Lasso model.
    - Lasso performs automatic feature selection, beneficial when dummy encoding
    produces large sparse design matrices.
    - If no regressors are available, the factor is returned unchanged.
    - Parallel execution via joblib significantly speeds up processing on large universes.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    reg = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, random_state = random_state)
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values

        if X.shape[1] > 0:
            reg.fit(X, y)
            y_pred = reg.predict(X)
            group[factor_col] = y - y_pred
        return group

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def elasticnet_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    alpha: float = 1.0,
    l1_ratio= 0.5,
    fit_intercept=True,
    max_iter: int = 300,
    tol: float = 0.0001,
    warm_start = False,
    random_state = None
) -> pl.DataFrame:
    """
    Perform cross-sectional factor neutralization using Elastic Net regression.

    This function applies Elastic Net—combining both L1 (Lasso) and L2 (Ridge)
    regularization—to neutralize a factor column against specified continuous
    neutralizers and optional dummy variables. Neutralization is performed
    independently within each cross-section grouped by 'datetime'. The output
    factor values correspond to regression residuals, representing the
    neutralized factor after removing linear exposure to the given covariates.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] together with
        all continuous neutralizers and categorical columns.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list of str
        Continuous numerical columns used as regressors.
    dummy_cols : list of str, optional
        Categorical variables to be one-hot encoded. If None, no dummy variables
        are created.
    scale_X : bool, default False
        Whether to standardize continuous neutralizer columns globally before regression.
    alpha : float, default 1.0
        Overall regularization strength applied to the Elastic Net model.
    l1_ratio : float, default 0.5
        The proportion of L1 (Lasso) vs. L2 (Ridge) penalty:
        - l1_ratio = 1 → Lasso
        - l1_ratio = 0 → Ridge
        - 0 < l1_ratio < 1 → Elastic Net
    fit_intercept : bool, default True
        Whether to fit an intercept term in the regression.
    max_iter : int, default 1000
        Maximum number of optimization iterations.
    tol : float, default 1e-4
        Convergence tolerance for the solver.
    warm_start : bool, default False
        If True, reuse previous coefficients as initialization.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing [trade_date_col, symbol_col, factor_col], where the
        factor column has been replaced by residuals from the Elastic Net model,
        representing the neutralized factor.

    Notes
    -----
    - Preprocessing (null dropping, dummy encoding, optional scaling) is handled
    via `preprocess_for_neutralization()`.
    - Neutralization is performed independently for each datetime cross-section.
    - Elastic Net is useful when both sparsity (L1) and stability (L2) are desired,
    especially in high-dimensional settings with many correlated predictors.
    - Parallelism via joblib provides substantial speedup for large datasets.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            random_state=random_state
        )
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values
        if X.shape[1] > 0:
            model.fit(X, y)
            y_pred = model.predict(X)
            group[factor_col] = y - y_pred
        return group

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def polynomial_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = True,
) -> pl.DataFrame:
    """
    Perform cross-sectional factor neutralization using Polynomial Regression.

    This function extends the neutralization framework by incorporating nonlinear
    effects through polynomial feature expansion. Within each datetime cross-section,
    the factor column is regressed onto polynomial-transformed neutralizer variables.
    The returned neutralized factor values are the residuals after removing both
    linear and nonlinear exposures.

    Polynomial regression is particularly useful when the relationship between the
    factor and neutralizers exhibits curvature or interactions that linear models
    cannot capture.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] along with the
        specified continuous and categorical neutralizer columns.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list of str
        Continuous numerical columns used as base variables for polynomial expansion.
    dummy_cols : list of str, optional
        Categorical columns to be one-hot encoded. If None, no dummy variables are added.
    scale_X : bool, default False
        Whether to standardize continuous neutralizers globally prior to constructing
        polynomial features.
    degree : int, default 2
        Degree of the polynomial expansion. For example:
        - degree = 1 → linear model
        - degree = 2 → squares and pairwise interactions
        - degree = 3 → cubic terms, etc.
    interaction_only : bool, default False
        If True, only interaction terms are generated (no squared terms).
    include_bias : bool, default True
        Whether to include a bias (constant) column in the polynomial features.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing [trade_date_col, symbol_col, factor_col], where the
        factor column is replaced with residuals from polynomial regression.

    Notes
    -----
    - Preprocessing (null dropping, dummy encoding, standardization) is performed
    by `preprocess_for_neutralization()`.
    - Polynomial features are generated using scikit-learn's `PolynomialFeatures`.
    - Neutralization is implemented via an orthogonal projection:
        residual = y - X_poly @ (X_poly⁺ @ y)
    where X_poly⁺ denotes the Moore–Penrose pseudoinverse.
    - Including higher-order terms significantly increases feature dimensionality,
    so memory and performance considerations are important for large universes.
    - Parallelization via joblib accelerates per-date neutralization.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )

    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )

    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        y = group[factor_col].values
        X = group[X_cols].values
        

        
        if X.shape[1] > 0:
            X_poly = poly.fit_transform(X)

            XtX_inv = np.linalg.pinv(X_poly.T @ X_poly)
            P = X_poly @ XtX_inv @ X_poly.T
            y_pred = P @ y
            group[factor_col] = y - y_pred

            return group

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def kernelridge_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    kernel: str = "rbf",
    alpha: float = 1.0,
    gamma: float = None,
    degree: int = 3,
    coef0: float = 1.0,
) -> pl.DataFrame:
    """
    Perform cross-sectional factor neutralization using Kernel Ridge Regression (KRR).

    This method removes both linear and nonlinear exposure of a factor to a set of
    neutralizer variables by fitting a kernelized ridge regression model within each
    datetime cross-section. The residuals (actual minus predicted values) represent
    the neutralized factor.

    Kernel Ridge combines ridge regularization with the flexibility of kernel
    methods, allowing it to capture rich nonlinear relationships such as curvature,
    interaction patterns, and complex manifolds that cannot be modeled by linear
    regression or polynomial expansion alone.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] along with
        specified numerical neutralizers and optional dummy variables.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list of str
        Continuous numerical variables used as regressors.
    dummy_cols : list of str, optional
        Categorical columns to be one-hot encoded. If None, no dummy variables
        are constructed.
    scale_X : bool, default False
        Whether to standardize continuous neutralizers globally before kernel
        regression. Standardization is recommended for RBF and polynomial kernels.
    n_jobs : int, default -1
        Number of parallel jobs. -1 uses all available CPU cores.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default 'rbf'
        Kernel type used by Kernel Ridge Regression.
    alpha : float, default 1.0
        Regularization strength. Larger values enforce stronger shrinkage and reduce
        overfitting.
    gamma : float, optional
        Kernel width parameter for RBF, polynomial, and sigmoid kernels.
        If None, scikit-learn defaults to 1 / n_features.
    degree : int, default 3
        Polynomial degree used when kernel='poly'.
    coef0 : float, default 1.0
        Independent term in polynomial and sigmoid kernels.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing [trade_date_col, symbol_col, factor_col], where the factor
        column has been replaced by residuals from the KRR model within each slice.

    Notes
    -----
    - Preprocessing (null filtering, optional scaling, dummy encoding) is performed
    by `preprocess_for_neutralization()`.
    - Kernel Ridge Regression fits models of the form:
        f(x) = Σ αᵢ K(x, xᵢ)
    allowing it to capture nonlinear structure in the feature space.
    - Computation scales as O(n²) per cross-section; thus, large universes should
    consider using a linear kernel or dimensionality reduction beforehand.
    - Parallelization via joblib accelerates processing across multiple dates.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    model = KernelRidge(
                kernel=kernel, alpha=alpha, gamma=gamma,
                degree=degree, coef0=coef0
            )
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values
        
        
        if X.shape[1] > 0:
            model.fit(X, y)
            y_pred = model.predict(X)

            group[factor_col] = y - y_pred
            return group

    groups = Parallel(n_jobs=-1)( delayed(_neutralize_group)(g) for _, g in df_pd.groupby(trade_date_col, sort=False) )

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def huber_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = True,
    epsilon: float = 1.35,
    max_iter: int = 100,
    alpha: float = 0.0,
    warm_start: bool = False,
    fit_intercept: bool = True,
    tol = 0.00001
) -> pl.DataFrame:
    """
    Perform robust cross-sectional factor neutralization using Huber Regression.

    Huber Regression provides robustness against outliers by combining the
    advantages of OLS and LAD, making it suitable for cross-sectional factor
    distributions with heavy tails.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col'] and the
        neutralizer columns.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list
        List of numeric columns used as neutralizers.
    dummy_cols : list, optional
        List of categorical columns to be one-hot encoded. Default is None.
    scale_X : bool, default True
        Whether to standardize the neutralizer variables before regression.
    epsilon : float, default 1.35
        Threshold parameter for the Huber loss. Larger values approach OLS;
        smaller values increase robustness to outliers.
    max_iter : int, default 100
        Maximum number of optimization iterations.
    alpha : float, default 0.0
        L2 regularization strength (Ridge penalty).
    warm_start : bool, default False
        Whether to reuse the previous solution as initialization.
    fit_intercept : bool, default True
        Whether to include an intercept term in the regression.
    tol : float, default 1e-5
        Numerical tolerance for convergence.

    Returns
    -------
    pl.DataFrame
        A DataFrame with [trade_date_col, factor_col'], where the factor column
        has been neutralized by taking the residuals (original factor minus predicted
        values) within each cross-sectional group.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    model = HuberRegressor(
                epsilon=epsilon, alpha=alpha, max_iter=max_iter, warm_start=warm_start, fit_intercept=fit_intercept, tol = tol
            )
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values
        
        if X.shape[1] > 0:
            model.fit(X, y)
            y_pred = model.predict(X)

            group[factor_col] = y - y_pred
            return group

    groups = Parallel(n_jobs=-1)( delayed(_neutralize_group)(g) for _, g in df_pd.groupby(trade_date_col, sort=False) )

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def rank_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    residual_threshold: float = 2.5,
    max_trials: int = 100,
    min_samples: float = 0.5,
    random_state=None
) -> pl.DataFrame:
    """
    Perform robust cross-sectional factor neutralization using RANSAC Regression

    RANSAC is highly resistant to outliers in both predictors and the factor
    distribution, making it useful when cross-sectional data contains extreme
    values, abrupt jumps, or structural breaks.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] and the
        neutralizer variables.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list
        List of continuous columns used as neutralizers.
    dummy_cols : list, optional
        List of categorical columns to be one-hot encoded. Default is None.
    scale_X : bool, default False
        Whether to standardize the neutralizer variables.
    residual_threshold : float, default 2.5
        Maximum allowed residual for a sample to be classified as an inlier in RANSAC.
    max_trials : int, default 100
        Maximum number of random iterations for model estimation.
    min_samples : float or int, default 0.5
        Minimum number (or proportion) of samples required to estimate the model.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        A DataFrame with [trade_date_col, symbol_col, factor_col], where the factor column
        contains the neutralized values computed as residuals from the RANSAC
        regression within each cross-section.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    estimator = LinearRegression(fit_intercept=True)
    ransac = RANSACRegressor(
                estimator=estimator,
                residual_threshold=residual_threshold,
                max_trials=max_trials,
                min_samples=min_samples,
                random_state=random_state,
            )
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values
        if X.shape[1] > 0:
            ransac.fit(X, y)
            y_pred = ransac.predict(X)
            group[factor_col] = y - y_pred
        return group

    groups = Parallel(n_jobs=-1)( delayed(_neutralize_group)(g) for _, g in df_pd.groupby(trade_date_col, sort=False) )

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def theilsen_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    fit_intercept=True,
    max_subpopulation: int | float = 1000,
    n_subsamples: int | None = None,
    max_iter: int = 300,
    tol: float = 0.001,
    random_state = None
) -> pl.DataFrame:
    """
    Perform robust cross-sectional factor neutralization using the Theil–Sen

    Theil–Sen regression is highly resistant to outliers in both predictors
    and target values, making it suitable for noisy or heavy-tailed
    cross-sectional financial data. Compared to RANSAC, it produces a fully
    deterministic estimator without random subset selection.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] and the
        neutralizer variables.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list
        List of continuous columns used as neutralizers.
    dummy_cols : list, optional
        List of categorical columns to be one-hot encoded. Default is None.
    scale_X : bool, default False
        Whether to standardize the neutralizer variables before fitting.
    n_jobs : int, default -1
        Number of parallel jobs. -1 uses all available CPU cores.
    fit_intercept : bool, default True
        Whether to fit an intercept in the regression model.
    max_subpopulation : int or float, default 10000
        Maximum subpopulation size used internally by Theil–Sen estimator
        to limit computational cost.
    n_subsamples : int or None, optional
        Number of samples used per iteration. If None, scikit-learn chooses a
        robust default based on the number of predictors.
    max_iter : int, default 300
        Maximum number of iterations in the estimator.
    tol : float, default 0.001
        Convergence tolerance for the iterative procedure.
    random_state : int, optional
        Random seed for reproducibility (affects subpopulation sampling).

    Returns
    -------
    pl.DataFrame
        A DataFrame containing [trade_date_col, symbol_col, factor_col], where the
        factor column holds the neutralized values computed as residuals
        from the Theil–Sen regression within each cross-sectional slice.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    model = TheilSenRegressor(
                fit_intercept=fit_intercept,
                max_subpopulation=max_subpopulation,  
                n_subsamples=n_subsamples,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state
            )
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values
        
        if X.shape[1] > 0:
            model.fit(X, y)
            y_pred = model.predict(X)
            group[factor_col] = y - y_pred
        return group

    groups = Parallel(n_jobs=-1)( delayed(_neutralize_group)(g) for _, g in df_pd.groupby(trade_date_col, sort=False) )
    
    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def randomforest_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    n_estimators: int = 50,
    max_depth: int | None = None,
    min_samples_split: float = 2,
    min_samples_leaf: float = 1,
    min_weight_fraction_leaf: float = 0,
    max_features: float  = 1,
    max_leaf_nodes: int | None = None,
    min_impurity_decrease: float = 0,
    bootstrap: bool = True,
    oob_score: bool = False,
    random_state: int | None = None,
    warm_start: bool = False,
    ccp_alpha: float = 0,
    max_samples: float | None = None
) -> pl.DataFrame:
    """
    Perform nonlinear cross-sectional factor neutralization using Random Forest
    regression, processed independently for each 'trade_date_col' group with
    optional parallelization.

    Random Forest captures nonlinear relationships and higher-order interactions
    between neutralizer variables through an ensemble of decision trees. The
    neutralized factor values are computed as residuals from the fitted model
    within each cross-sectional slice.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col]
        and the neutralizer variables.
    trade_date_col : str
        Column name representing the cross-sectional grouping key (e.g. datetime).
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list
        List of continuous columns used as neutralizers.
    dummy_cols : list, optional
        List of categorical columns to be one-hot encoded. Default is None.
    scale_X : bool, default False
        Whether to standardize the neutralizer variables before fitting.
    n_jobs : int, default -1
        Number of parallel jobs for cross-sectional processing. -1 uses all
        available CPU cores.
    n_estimators : int, default 150
        Number of trees in the forest.
    max_depth : int or None, optional
        Maximum depth of each tree. If None, nodes are expanded until all
        leaves are pure or contain fewer samples than min_samples_split.
    min_samples_split : int or float, default 2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int or float, default 1
        Minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float, default 0
        Minimum weighted fraction of the sum total of weights required
        to be at a leaf node.
    max_features : int, float, str or None, default 1
        Number of features to consider when looking for the best split.
    max_leaf_nodes : int or None, optional
        Grow trees with at most this number of leaf nodes.
    min_impurity_decrease : float, default 0
        A node will be split if this split induces a decrease of impurity
        greater than or equal to this value.
    bootstrap : bool, default True
        Whether bootstrap samples are used when building trees.
    oob_score : bool, default False
        Whether to use out-of-bag samples to estimate the generalization score.
    random_state : int or None, optional
        Controls randomness of the estimator.
    warm_start : bool, default False
        Whether to reuse the solution of the previous call to fit and
        add more estimators to the ensemble.
    ccp_alpha : float, default 0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    max_samples : int or float or None, optional
        If bootstrap is True, the number of samples to draw from X to train
        each base estimator.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing [trade_date_col, symbol_col, factor_col], where
        the factor column holds the neutralized values computed as residuals
        from the Random Forest model within each cross-sectional slice.
    """

    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=1,
                min_samples_split=min_samples_split,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                warm_start=warm_start,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples
            )
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values

        if X.shape[1] > 0:
            rf.fit(X, y)
            y_pred = rf.predict(X)
            group[factor_col] = y - y_pred  # 残差部分 = 中性化结果
        return group

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def GBDT_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    learning_rate = 0.1,
    n_estimators = 50,
    subsample = 1,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0,
    max_depth = 3,
    min_impurity_decrease = 0,
    init = None,
    random_state = None,
    alpha = 0.9,
    max_leaf_nodes = None,
    warm_start = False,
    validation_fraction = 0.1,
    n_iter_no_change = None,
    tol = 0.0001,
    ccp_alpha = 0
) -> pl.DataFrame:
    """
    Perform nonlinear cross-sectional factor neutralization using Gradient
    Boosting Decision Trees (GBDT), processed independently for each
    'datetime' group with optional parallelization.

    GBDT captures nonlinear and interaction effects between neutralizer
    variables, providing a flexible alternative to linear or robust
    regressions for residual-based factor neutralization.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] and the
        neutralizer variables.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list
        List of continuous columns used as neutralizers.
    dummy_cols : list, optional
        List of categorical columns to be one-hot encoded. Default is None.
    scale_X : bool, default False
        Whether to standardize the neutralizer variables before fitting.
    n_jobs : int, default -1
        Number of parallel jobs. -1 uses all available CPU cores.
    learning_rate : float, default 0.1
        Shrinkage factor applied to each tree.
    n_estimators : int, default 100
        Number of boosting stages (trees).
    subsample : float, default 1.0
        Fraction of samples used per boosting stage. Values < 1.0 improve
        generalization by introducing randomness.
    min_samples_split : int, default 2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default 1
        Minimum samples required to be at a leaf node.
    min_weight_fraction_leaf : float, default 0.0
        Minimum weighted fraction of samples in a leaf node.
    max_depth : int, default 3
        Maximum depth of each regression tree.
    min_impurity_decrease : float, default 0.0
        Minimum impurity decrease required to perform a split.
    init : estimator or None, optional
        Initial estimator for boosting. If None, use the mean estimator.
    random_state : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.9
        Quantile used in loss functions such as 'quantile' (if enabled).
    max_leaf_nodes : int or None, optional
        Maximum number of leaf nodes per tree.
    warm_start : bool, default False
        Whether to reuse the fitted solution for additional boosting stages.
    validation_fraction : float, default 0.1
        Fraction of data used for early stopping when n_iter_no_change is set.
    n_iter_no_change : int or None, optional
        Stops training if validation loss does not improve for this many iterations.
    tol : float, default 1e-4
        Tolerance for early stopping.
    ccp_alpha : float, default 0.0
        Complexity parameter used for minimal cost-complexity pruning.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing [trade_date_col, symbol_col, factor_col], where the
        factor column holds the neutralized values computed as residuals from
        the GBDT model within each cross-sectional slice.
    """

    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    model = GradientBoostingRegressor(
                learning_rate = learning_rate,
                n_estimators = n_estimators,
                subsample = subsample,
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf,
                min_weight_fraction_leaf = min_weight_fraction_leaf,
                max_depth = max_depth,
                min_impurity_decrease = min_impurity_decrease,
                init = init,
                random_state = random_state,
                alpha = alpha,
                max_leaf_nodes = max_leaf_nodes,
                warm_start = warm_start,
                validation_fraction = validation_fraction,
                n_iter_no_change = n_iter_no_change,
                tol = tol,
                ccp_alpha = ccp_alpha
            )
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values
        
        if X.shape[1] > 0:
            model.fit(X, y)
            y_pred = model.predict(X)
            group[factor_col] = y - y_pred
        return group

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def PCA_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = True,
    n_components = None,
    tol= 0,
    n_oversamples = 10,
    random_state= None
) -> pl.DataFrame:
    """
    Perform cross-sectional factor neutralization using Principal Component
    Analysis (PCA). 

    PCA is used to extract orthogonal components from the neutralizer
    variables (continuous and dummy-encoded), and the factor is regressed
    onto these principal components. The residuals (y - y_pred) form the
    neutralized factor, representing the portion unexplained by the PCA
    components.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] and all
        neutralizer variables.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list
        List of numeric columns used as neutralizers.
    dummy_cols : list, optional
        List of categorical columns to be one-hot encoded. Default is None.
    scale_X : bool, default True
        Whether to standardize neutralizer variables before PCA.
    n_components : int or None, optional
        Number of principal components to use. If None, automatically uses:
        min(10, n_features, n_samples).
    tol : float, default 0
        Tolerance for randomized SVD (if applicable).
    n_oversamples : int, default 10
        Additional number of random vectors used in randomized SVD to improve
        accuracy.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing [trade_date_col, symbol_col, factor_col], where the
        factor column holds the PCA-neutralized values (i.e., residuals from
        regressing the factor on PCA components within each cross-section).
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values

        if X.shape[1] > 0:
            n_comp = n_components or min(10, X.shape[1], X.shape[0])
            pca = PCA(
                n_components = n_comp,
                tol= tol,
                n_oversamples = n_oversamples,
                random_state= random_state
                      )
            comps = pca.fit_transform(X)
            beta = np.linalg.lstsq(comps, y, rcond=None)[0]
            y_pred = comps @ beta

            group[factor_col] = y - y_pred 
            return group

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def ICA_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    n_jobs: int = -1,
    n_components: int = None,
    max_iter = 200,
    tol = 0.0001,
    random_state = None,
    return_components: bool = False
) -> pl.DataFrame:
    """
    Perform cross-sectional factor neutralization using Independent Component
    Analysis (ICA).
    ICA decomposes the neutralizer variables into statistically independent
    components. The factor is then regressed on these independent components,
    and the residuals (y - y_pred) represent the ICA-neutralized factor.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] and all
        neutralizer variables.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list
        List of numeric columns used as neutralizers.
    dummy_cols : list, optional
        List of categorical columns to be one-hot encoded. Default is None.
    scale_X : bool, default False
        Whether to standardize neutralizer variables before ICA.
    n_components : int or None, optional
        Number of ICA components. If None, defaults to:
        min(n_features, n_samples).
    max_iter : int, default 200
        Maximum number of iterations for the FastICA solver.
    tol : float, default 1e-4
        Tolerance for the FastICA convergence condition.
    random_state : int or None, optional
        Random seed for reproducibility.
    return_components : bool, default False
        If True, the output will include the extracted ICA components
        (columns 'ICA_1', 'ICA_2', ...).

    Returns
    -------
    pl.DataFrame
        A DataFrame containing [trade_date_col, symbol_col, factor_col], where the
        factor column holds the ICA-neutralized values (i.e., residuals from
        regressing the factor on ICA components within each cross-section).

        If `return_components=True`, additional columns 'ICA_k' are included,
        representing the independent components extracted for each group.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values

        n_comp = n_components or min(X.shape[1], X.shape[0])
        if X.shape[1] > 0:
            ica = FastICA(
                n_components=n_comp, 
                max_iter = max_iter,
                tol = tol,
                random_state = random_state)
            S = ica.fit_transform(X) 

            X_ica = np.column_stack([np.ones(len(S)), S])
            beta = np.linalg.pinv(X_ica.T @ X_ica) @ X_ica.T @ y
            y_pred = X_ica @ beta
            resid = y - y_pred
            group[factor_col] = resid

        if return_components:
            for i in range(S.shape[1]):
                group[f"ICA_{i+1}"] = S[:, i]
            return group
        else:
            return group

    
    groups = Parallel(n_jobs=n_jobs)(
        delayed(_neutralize_group)(g)
        for _, g in df_pd.groupby(trade_date_col, sort=False)
    )

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def bayesianridge_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    n_iter = 300,
    tol = 0.001,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    alpha_init = None,
    lambda_init = None,
    fit_intercept=True,
    compute_score=False,
    return_coef: bool = False,
) -> pl.DataFrame:
    """
    Perform factor neutralization using Bayesian Ridge Regression. Bayesian Ridge introduces
    precision priors on both coefficients and noise, providing a more stable,
    regularized regression compared to OLS or standard ridge. The residuals
    of the regression are returned as the neutralized factor.

    Workflow
    --------
    1. For each cross-section, extract neutralizer_cols and dummy_cols.
    2. Fit a BayesianRidge model to predict the factor from the neutralizers.
    3. Compute residuals (factor - prediction).
    4. Optionally compute the mean coefficient for diagnostics.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] and all
        neutralizer variables.
    factor_col : str
        The factor column to be neutralized.
    neutralizer_cols : list
        List of numeric columns used as neutralizing variables.
    dummy_cols : list, optional
        List of categorical variables to be one-hot encoded. Default is None.
    scale_X : bool, default False
        Whether to standardize the neutralizer features.
    n_iter : int, default 300
        Maximum number of iterations for BayesianRidge.
    tol : float, default 0.001
        Convergence tolerance of the optimization.
    alpha_1, alpha_2 : float, default 1e-6
        Hyperparameters for the Gamma prior over the noise precision.
    lambda_1, lambda_2 : float, default 1e-6
        Hyperparameters for the Gamma prior over the weight precision.
    alpha_init : float, optional
        Initial value for the noise precision.
    lambda_init : float, optional
        Initial value for the weight precision.
    fit_intercept : bool, default True
        Whether the BayesianRidge model should estimate an intercept.
    compute_score : bool, default False
        Whether to compute the log-marginal-likelihood score during training.
    return_coef : bool, default False
        If True, output an additional column 'coef_mean' representing the
        mean of the posterior coefficient estimates for diagnostic purposes.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing ['datetime', 'code', factor_col], where the
        factor column holds the neutralized factor (residuals).

        If `return_coef=True`, an additional column 'coef_mean' is included,
        representing the mean posterior regression coefficient for each
        cross-section.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    model = BayesianRidge(
                max_iter = n_iter,
                tol = tol,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                alpha_init = alpha_init,
                lambda_init = lambda_init,
                fit_intercept=fit_intercept,
                compute_score=compute_score,
            )
    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values

        if X.shape[1] > 0:
            model.fit(X, y)
            y_pred = model.predict(X)
            group[factor_col] = y - y_pred
        
        if return_coef:
            group["coef_mean"] = np.mean(model.coef_)
            return group
        else:
            return group

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))

def partialcorrelation_neutralize(
    pl_df: pl.DataFrame,
    trade_date_col,
    factor_col: str,
    neutralizer_cols: list,
    dummy_cols: list = None,
    scale_X: bool = False,
    target_col: str = None,
    method: str = "pearson",
    return_type: str = "neutralized_factor",
) -> pl.DataFrame:
    """
    Perform cross-sectional factor neutralization using Partial Correlation.
    This method removes the linear effects of neutralizer variables from the
    factor, and optionally computes the partial correlation between the factor
    and a target variable after controlling for the neutralizers.

    Workflow
    --------
    1. For each cross-section, regress the factor on the neutralizer variables.
    2. Extract residuals as the neutralized factor.
    3. If `target_col` is provided and `return_type='partial_corr'`, compute the
    partial correlation between the factor and the target after removing
    the influence of the neutralizers.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Input DataFrame containing [trade_date_col, symbol_col, factor_col] and all
        neutralizer variables.
    factor_col : str
        Name of the factor column to be neutralized.
    neutralizer_cols : list of str
        Numeric columns used for neutralization.
    dummy_cols : list of str, optional
        Categorical variables to be automatically one-hot encoded.
    scale_X : bool, default False
        Whether to standardize the neutralizer columns before processing.
    target_col : str, optional
        If provided, compute the partial correlation between the factor and
        this target variable after controlling for neutralizers.
    method : {'pearson', 'spearman'}, default 'pearson'
        Correlation method used when computing partial correlation.
    return_type : {'neutralized_factor', 'partial_corr'}, default 'neutralized_factor'
        - 'neutralized_factor': return cross-sectionally neutralized factor
        values for each row.
        - 'partial_corr': return one partial correlation value per cross-section.

    Returns
    -------
    pl.DataFrame
        If return_type='neutralized_factor':
            A DataFrame containing [trade_date_col, symbol_col, factor_col], where
            the factor column contains residuals after removing the influence
            of the neutralizer variables.

        If return_type='partial_corr':
            A DataFrame with one row per datetime, containing:
            [trade_date_col, factor_col]  
            where factor_col stores the partial correlation coefficient.
    """
    df_pd, X_cols = preprocess_for_neutralization(
                            pl_df, factor_col, neutralizer_cols, dummy_cols, scale_X
                        )
    def _partial_corr(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        if method == "spearman":
            return spearmanr(x, y)[0]
        else:
            return pearsonr(x, y)[0]

    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        X = group[X_cols].values
        y = group[factor_col].values

        if X.shape[1] > 0:
            P = X @ np.linalg.pinv(X.T @ X) @ X.T
            y_resid = y - P @ y
        else:
            y_resid = y.copy()

        if target_col is None or return_type == "neutralized_factor":
            group[factor_col] = y_resid
            return group

        else:
            r = group[target_col].values
            if X.shape[1] > 0:
                r_resid = r - P @ r
            else:
                r_resid = r.copy()
            pcorr = _partial_corr(y_resid, r_resid)
            return pd.DataFrame({
                "trade_date_col": [group["trade_date_col"].iloc[0]],
                factor_col: [pcorr]
            })

    groups = []
    for _, g in df_pd.groupby(trade_date_col, sort=False):
        groups.append(_neutralize_group(g))

    return pl.from_pandas(pd.concat(groups, ignore_index=True))


def zscore_standardize(
    pl_df: pl.DataFrame, 
    trade_date,
    factor_col: str
    ) -> pl.DataFrame:
    """
    Perform cross-sectional Z-score standardization on a factor column.

    This function standardizes a factor within each cross-section using the Z-score transformation:

        z = (x - mean_t) / std_t

    where mean_t and std_t are computed over all securities at the same
    section. The output replaces the factor values with their standardized
    counterparts.

    Parameters
    ----------
    pl_df : pl.DataFrame
        The factor column will be standardized
        cross-sectionally.
    trade_date : str
    factor_col : str
        Name of the factor column to be Z-score standardized.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the
        factor column has been transformed into cross-sectional Z-scores.

    Notes
    -----
    - Standardization is applied independently for each section.
    - The result preserves the original shape and ordering of the input rows.
    - If the cross-sectional standard deviation is zero, Polars will return
      null values for that group.
    """
    return (
        pl_df
        .with_columns([
            (
                (pl.col(factor_col) - pl.col(factor_col).mean().over(trade_date))
                / pl.col(factor_col).std().over(trade_date)
            ).alias(factor_col)
        ])
    )

def robust_zscore_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    c: float = 1.4826
) -> pl.DataFrame:
    """
    Perform cross-sectional robust Z-score standardization on a factor column.

    This function applies a robust alternative to the traditional Z-score by
    replacing the mean and standard deviation with the cross-sectional median
    and MAD (Median Absolute Deviation). This approach reduces sensitivity to
    outliers and is commonly used for factors that exhibit heavy tails or
    extreme values.

    The transformation is defined as:

        robust_z = (x - median_t) / (MAD_t * c)

    where:
        - median_t is the cross-sectional median at each section
        - MAD_t = median(|x - median_t|)
        - c is a consistency constant (default 1.4826) that makes MAD comparable
          to the standard deviation under normality.

    Parameters
    ----------
    pl_df : pl.DataFrame
        The factor column will be robustly standardized within each cross-section.
    trade_date : str
    factor_col : str
        Name of the factor column to be standardized robustly.
    c : float, default 1.4826
        Consistency factor applied to the MAD. The default value scales MAD to be
        approximately equal to standard deviation under Gaussian distribution.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the
        factor column has been replaced by its robust Z-score standardized
        values.

    Notes
    -----
    - Median and MAD are computed independently for each section.
    - More resistant to extreme values than traditional Z-score standardization.
    - If MAD equals zero within a cross-section, resulting values will be null
      for that group.
    - This method is particularly suitable for noisy, heavy-tailed, or
      microstructure-influenced factors.
    """
    med = (
        pl_df
        .group_by(trade_date)
        .agg(pl.col(factor_col).median().alias("median"))
    )

    mad = (
        pl_df
        .join(med, on=trade_date)
        .with_columns(
            (pl.col(factor_col) - pl.col("median")).abs().alias("abs_dev")
        )
        .group_by(trade_date)
        .agg(pl.col("abs_dev").median().alias("mad"))
    )

    df = (
        pl_df
        .join(med, on=trade_date)
        .join(mad, on=trade_date)
        .with_columns(
            ((pl.col(factor_col) - pl.col("median")) / (pl.col("mad") * c))
            .alias(factor_col)
        )
    )

    return df

def minmax_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    clip: bool = True, 
) -> pl.DataFrame:
    """
    Perform cross-sectional Min–Max (0–1) standardization on a factor column.

    This function rescales factor values within each cross-section to the [0, 1] range using the Min–Max transformation:

        scaled = (x - min_t) / (max_t - min_t)

    where min_t and max_t are computed from all securities at the same
    section. Optionally, values can be clipped to ensure they remain within
    the [0, 1] interval.

    Parameters
    ----------
    pl_df : pl.DataFrame
        The factor column will be Min–Max scaled
        cross-sectionally.
    trade_date : str
    factor_col : str
        Name of the factor column to be Min–Max standardized.
    clip : bool, default True
        Whether to clamp the resulting scaled values to the [0, 1] interval.
        - True ensures outputs strictly lie within [0, 1].
        - False allows values outside the range if division instability occurs.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the
        factor column has been transformed using cross-sectional Min–Max scaling.

    Notes
    -----
    - Standardization is performed independently for each section.
    - If max_t equals min_t for a given date, resulting values will be null
      for that group.
    - Min–Max scaling preserves relative ordering but not distribution shape.
    """

    stats = (
        pl_df
        .group_by(trade_date)
        .agg([
            pl.col(factor_col).min().alias("min"),
            pl.col(factor_col).max().alias("max"),
        ])
    )

    df = (
        pl_df
        .join(stats, on=trade_date)
        .with_columns(
            ((pl.col(factor_col) - pl.col("min")) / (pl.col("max") - pl.col("min")))
            .alias(factor_col)
        )
    )

    if clip:
        df = df.with_columns(
            pl.col(factor_col).clip(0, 1)
        )

    return df

def rank_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    method: str = "average"   
) -> pl.DataFrame:
    """
    Perform cross-sectional rank-based percentile standardization on a factor column.

    This function converts raw factor values into percentile scores within each
    cross-section using ranking. After ranking the factor
    values for each date, the ranks are divided by the total number of securities
    in that cross-section to produce a normalized percentile in the range (0, 1].

    The ranking behavior can be controlled via the `method` parameter, which
    determines how ties are handled (e.g., "average", "min", "max", "dense",
    "ordinal").

    Parameters
    ----------
    pl_df : pl.DataFrame
        The factor column will be transformed into cross-sectional percentile
        ranks.
    factor_col : str
        Name of the factor column to be rank-standardized.
    method : str, default "average"
        Ranking method used for handling ties. Supported options include:
        - "average": average rank for ties
        - "min": lowest rank assigned to all ties
        - "max": highest rank assigned to all ties
        - "dense": like "min" but ranks increase by 1
        - "ordinal": assigns unique increasing ranks with no ties preserved

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the
        factor column has been converted to a cross-sectional percentile
        representation.

    Notes
    -----
    - Percentile ranks are computed independently for each section.
    - The resulting values fall within (0, 1], with larger values representing
      higher-ranked factor observations.
    - Ranking-based standardization is robust to outliers and preserves only the
      relative ordering of observations.
    """
    df = (
        pl_df
        .with_columns([
            pl.col(factor_col)
            .rank(method=method)
            .over(trade_date)
            .alias("rank_val"),

            pl.count()
            .over(trade_date)
            .alias("count_val"),
        ])
        .with_columns([
            (pl.col("rank_val") / pl.col("count_val")).alias(factor_col)
        ])
    )

    return df

def rank_gaussianize_standardize(
    pl_df: pl.DataFrame, 
    trade_date,
    factor_col: str, 
    method: str = "average"
    ) -> pl.DataFrame:
    """
    Perform cross-sectional rank-based Gaussianization on a factor column.

    This function first converts raw factor values into percentile ranks within
    each cross-section, then applies an inverse normal
    CDF (Gaussian quantile) transformation to map the percentiles onto an
    approximately standard normal distribution.

    The rank → percentile → Gaussianization pipeline is commonly used to
    transform highly skewed, heavy-tailed, or non-linear factors into a more
    Gaussian shape, improving stability for models and cross-sectional
    regressions.

    Steps
    -----
    1) Within each section, compute the rank of each observation using the
       specified ranking method, and convert ranks to percentiles via:
           pct = rank / (count + 1)
       The +1 ensures percentiles stay strictly between (0,1).

    2) Apply `norm.ppf` to the percentiles to obtain Gaussian scores:
           z = Φ⁻¹(pct)
       Percentiles are clipped to [1e-6, 1−1e-6] to avoid infinities.

    Parameters
    ----------
    pl_df : pl.DataFrame
        The factor column will be Gaussianized cross-sectionally.
    trade_date :str
    factor_col : str
        Name of the factor column to be transformed.
    method : str, default "average"
        Ranking method determining how ties are treated. Options include:
        "average", "min", "max", "dense", "ordinal".

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the
        factor column has been transformed to approximately follow a standard
        normal distribution within each cross-section.

    Notes
    -----
    - Each 'datetime' is processed independently.
    - The transformation preserves only the ordinal information of the factor.
    - Gaussianization is useful as a robust normalization method when the
      factor distribution is irregular, long-tailed, or contains extreme values.
    - Clipping avoids numerical instability from percentiles equal to exactly
      0 or 1.
    """
    df = (
        pl_df
        .with_columns([
            pl.col(factor_col).rank(method=method).over(trade_date).cast(pl.Float64).alias("rank_val"),
            pl.count().over(trade_date).cast(pl.Float64).alias("count_val"),
        ])
        .with_columns(
            (pl.col("rank_val") / (pl.col("count_val") + 1)).alias("pct")
        )
    )

    pct = df["pct"].to_numpy()
    zscores = norm.ppf(np.clip(pct, 1e-6, 1 - 1e-6)) 
    df = df.with_columns(pl.Series(factor_col, zscores))

    return df

def rolling_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    symbol_col,
    factor_col: str,
    window: int = 20,
    min_periods: int | None = None,
) -> pl.DataFrame:
    """
    Perform rolling Z-score standardization on a factor column, grouped by stock code.

    This function computes a time-series Z-score for each security independently.
    For each section pair, the factor is standardized using the
    rolling mean and rolling standard deviation over a trailing window of
    length `window`. The operation transforms the factor into:

        z_t = (x_t - mean_t) / std_t

    where mean_t and std_t are computed from the previous `window` observations
    for the same stock.

    Parameters
    ----------
    pl_df : pl.DataFrame
        The factor
        values should be chronologically sorted or at least sortable by section.
    factor_col : str
        Name of the factor column on which the rolling standardization is applied.
    window : int, default 20
        Rolling window size (number of past observations) used to compute the
        rolling mean and standard deviation.
    min_periods : int or None, default None
        Minimum number of observations required to compute a valid rolling mean
        and standard deviation. If None, defaults to `window`. If fewer than
        `min_periods` observations exist for a given window, the standardized
        value will be null.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the
        factor column contains the time-series rolling Z-scores for each stock.

    Notes
    -----
    - Rolling calculations are performed independently for each stock code.
    - The DataFrame is sorted by section to ensure correct window
      formation.
    - If the rolling standard deviation is zero within a window, the resulting
      Z-score will be null.
    - This method is suitable for time-series normalization of momentum,
      volatility, or other serially dependent factors.
    """
    if min_periods is None:
        min_periods = window

    df = (
        pl_df
        .sort([symbol_col, trade_date])
        .with_columns([
            pl.col(factor_col)
            .rolling_mean(window_size=window, min_samples=min_periods)
            .over(symbol_col)
            .alias("mean"),

            pl.col(factor_col)
            .rolling_std(window_size=window, min_samples=min_periods)
            .over(symbol_col)
            .alias("std"),
        ])
        .with_columns([
            ((pl.col(factor_col) - pl.col("mean")) / pl.col("std"))
            .alias(factor_col)
        ])
    )

    return df

def rolling_robust_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    symbol_col,
    factor_col: str,
    window: int = 20,
    min_periods: int | None = None,
):
    """
    Perform rolling robust standardization on a factor column using median and MAD.

    This function computes a robust time-series Z-score for each security
    independently. Unlike the classical rolling Z-score that relies on mean and
    standard deviation, this method uses the rolling median and rolling MAD
    (Median Absolute Deviation), making it significantly more resistant to
    outliers and heavy-tailed noise in the factor values.

    The robust rolling Z-score is defined as:

        z_t = (x_t - median_t) / (1.4826 * MAD_t)

    where:
      - median_t is the rolling median over the past `window` observations,
      - MAD_t is the rolling median of |x - median| over the same window,
      - 1.4826 is the consistency constant to make MAD comparable to std.

    Parameters
    ----------
    pl_df : pl.DataFrame
        The factor column will be transformed using robust rolling normalization.
    trade_date:str
    symbol_col:str
    factor_col : str
        Name of the factor column to be standardized.
    window : int, default 20
        Rolling window size used to compute both the rolling median and rolling MAD.
    min_periods : int or None, default None
        Minimum number of observations required to compute valid rolling median
        and MAD. If None, defaults to `window`. When fewer observations exist,
        the resulting values will be null.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the
        factor column contains robust rolling Z-scores for each security.

    Notes
    -----
    - Rolling calculations are applied independently for each stock.
    - This method is highly robust to large spikes or irregular values, making it
      suitable for volume factors, sentiment factors, or microstructure noise.
    - If MAD_t is zero within a window, the resulting Z-score will be null.
    - Using median + MAD provides greater stability in environments with
      non-Gaussian or heavy-tailed factor distributions.
    """
    if min_periods is None:
        min_periods = window

    df = (
        pl_df
        .sort([symbol_col, trade_date])
        .with_columns([
            pl.col(factor_col)
            .rolling_median(window_size=window, min_samples=min_periods)
            .over(symbol_col)
            .alias("med")
        ])
        .with_columns([
            (pl.col(factor_col) - pl.col("med")).abs().alias("abs_dev")
        ])
        .with_columns([
            pl.col("abs_dev")
            .rolling_median(window_size=window, min_samples=min_periods)
            .over(symbol_col)
            .alias("mad")
        ])
        .with_columns([
            ((pl.col(factor_col) - pl.col("med")) / (1.4826 * pl.col("mad")))
            .alias(factor_col)
        ])
    )

    return df

def rolling_minmax_standardize(
    pl_df: pl.DataFrame,
    factor_col: str,
    trade_date,
    symbol_col,
    window: int = 20,
    min_periods: int | None = None,
):
    """
    Perform rolling Min–Max normalization on a factor column, grouped by stock code.

    This function computes a sliding-window Min–Max normalization for each
    security independently. For each pair, the factor is
    scaled into the [0, 1] range using the rolling minimum and maximum from the
    past `window` observations:

        norm_t = (x_t - min_t) / (max_t - min_t)

    This method preserves relative ordering within the window and is useful for
    turning raw factor levels into bounded indicators.

    Parameters
    ----------
    pl_df : pl.DataFrame
        The factor column will be transformed using rolling Min–Max scaling.
    trade_date : str
    symbol_col : str 
    factor_col : str
        Name of the factor column to be normalized.
    window : int, default 20
        Rolling window size used to compute the minimum and maximum.
    min_periods : int or None, default None
        Minimum number of observations required to compute a valid rolling min
        and max. If None, defaults to `window`. When fewer than `min_periods`
        values exist, the corresponding standardized value will be null.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor
        column contains the rolling Min–Max scaled values.

    Notes
    -----
    - Computation is done independently for each security .
    - If roll_max equals roll_min within a window, the resulting value will be null
      due to division by zero.
    - Rolling Min–Max is particularly useful for bounded factor indicators, such
      as oscillators, volatility ratios, or signals requiring normalization into
      a fixed range.
    """
    if min_periods is None:
        min_periods = window

    df = (
        pl_df
        .sort([symbol_col, trade_date])
        .with_columns([
            pl.col(factor_col)
            .rolling_min(window_size=window, min_samples=min_periods)
            .over(symbol_col)
            .alias("roll_min")
        ])
        .with_columns([
            pl.col(factor_col)
            .rolling_max(window_size=window, min_samples=min_periods)
            .over(symbol_col)
            .alias("roll_max")
        ])
        .with_columns([
            ((pl.col(factor_col) - pl.col("roll_min"))
             / (pl.col("roll_max") - pl.col("roll_min")))
            .alias(factor_col)
        ])
    )

    return df

def volatility_scaling_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    symbol_col,
    factor_col: str,
    window: int = 20,
    min_periods: int | None = None,
    shift_vol: bool = True,   
):
    """
    Perform volatility-scaling standardization on a factor column, grouped by stock code.

    This method rescales factor values using their rolling volatility (standard
    deviation). For each security, the factor is divided by its recent rolling
    standard deviation:

        scaled_t = x_t / σ_t

    Optionally, the rolling volatility can be shifted by one period to avoid
    look-ahead bias, a common practice in quantitative finance.

    Parameters
    ----------
    pl_df : pl.DataFrame
        The factor column will be volatility-scaled.
    factor_col : str
        Name of the factor column to be standardized.
    trade_date : str
    symbol_col : str
    window : int, default 20
        Rolling window size used to compute volatility.
    min_periods : int or None, default None
        Minimum number of observations required for volatility calculation.
        If None, defaults to `window`.  
        When fewer than `min_periods` values exist, volatility will be null.
    shift_vol : bool, default True
        Whether to shift the rolling volatility by one period (σ(t−1)), which is
        commonly used to avoid look-ahead bias in backtesting.

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the factor
        column contains the volatility-scaled factor values.

    Notes
    -----
    - Computation is performed independently for each security.
    - If rolling volatility is zero or null, the resulting scaled value will be null.
    - Volatility scaling normalizes factor magnitudes and is often used when
      combining signals with different inherent variance levels.
    """

    if min_periods is None:
        min_periods = window

    df = (
        pl_df
        .sort([symbol_col, trade_date])
        .with_columns([
            pl.col(factor_col)
            .rolling_std(window_size=window, min_samples=min_periods)
            .over(symbol_col)
            .alias("vol")
        ])
        .with_columns([
            pl.col("vol").shift(1).over(symbol_col).alias("vol") if shift_vol else pl.col("vol")
        ])
        .with_columns([
            (pl.col(factor_col) / pl.col("vol")).alias(factor_col)
        ])
    )

    return df

def EWMA_standardize(
    pl_df: pl.DataFrame,
    trade_date: str,
    symbol_col:str,
    factor_col: str,
    lambda_: float = 0.94,
    eps: float = 1e-12):
    """
    Perform EWMA-based volatility standardization on a factor column.

    This method rescales factor values using volatility estimated via an
    Exponentially Weighted Moving Average (EWMA), a widely used estimator in
    quantitative finance and risk modeling (e.g., RiskMetrics). The EWMA assigns
    exponentially decaying weights to historical squared returns (or factor
    values), producing a responsive yet smooth volatility estimate:

        σ²_t = (1 - λ) * Σ_{k=0..t} λ^k * x²_{t-k}

    Once the EWMA variance is computed for each security, the factor is scaled by
    dividing the raw value by the EWMA standard deviation:

        scaled_t = x_t / σ_t

    Parameters
    ----------
    pl_df : pl.DataFrame
        Rolling EWMA calculations are performed independently for each stock.
    factor_col : str
        Name of the factor column to be EWMA-standardized.
    lambda_ : float, default 0.94
        The exponential decay parameter λ. Higher λ places more weight on
        recent observations and produces a smoother volatility estimate.
        Typical finance values: 0.94 (daily), 0.97 (monthly).
    eps : float, default 1e-12
        A small constant added to the denominator to avoid division-by-zero
        when volatility is extremely small.

    Returns
    -------
    pl.DataFrame
        A DataFrame where the factor
        column has been scaled using EWMA-estimated volatility.

    Notes
    -----
    - Computation is grouped by security; each ticker's history is
      processed independently.
    - The function constructs EWMA variance via reversed accumulation of
      weighted squared values, enabling efficient vectorized computation.
    - EWMA volatility reacts faster to regime shifts compared to rolling-window
      standard deviation, making it suitable for high-frequency factors or
      volatile assets.
    - If the EWMA variance is zero or null, the resulting scaled value will be
      null.
    - The output preserves the original time order of observations.
    """
    df = (
        pl_df
        .sort([symbol_col, trade_date])
        .with_columns([
            pl.col(factor_col).alias("x"),
            (pl.col(factor_col) ** 2).alias("x2"),
        ])
    )

    df = df.with_columns([
        pl.arange(0, pl.count()).over(symbol_col).alias("t")
    ])

    df = df.with_columns([
        ((1 - lambda_) * (lambda_ ** pl.col("t"))).alias("weight")
    ])

    df = (
        df
        .with_columns([
            pl.col("x2").reverse().over(symbol_col).alias("rev_x2"),
            pl.col("weight").reverse().over(symbol_col).alias("rev_weight"),
        ])
        .with_columns([
            (pl.col("rev_x2") * pl.col("rev_weight"))
            .cum_sum()
            .over(symbol_col)
            .alias("rev_ewma"),
        ])
        .with_columns([
            pl.col("rev_ewma").reverse().over(symbol_col).alias("ewma_var")
        ])
    )
    df = df.with_columns([
        pl.col("ewma_var").sqrt().alias("ewma_vol")
    ])
    df = df.with_columns([
        (pl.col("x") / (pl.col("ewma_vol") + eps)).alias(factor_col)
    ])

    return df

def normal_scores_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    eps: float = 1e-9          
):
    """
    Perform rank-based normal score transformation on a factor column.

    This method transforms raw factor values into approximately standard normal
    scores using an empirical cumulative distribution function (ECDF).  
    For each cross-sectional slice, the factor values
    are ranked and converted into uniform probabilities:

        u_i = (rank_i - 0.5) / n

    where n is the cross-sectional sample size.  
    These uniform scores are then mapped to standard normal quantiles via the
    inverse CDF (probit transformation):

        z_i = Φ⁻¹(u_i)

    This procedure removes the effect of heavy tails and extreme outliers,
    producing a distribution closer to N(0, 1), which is often desirable for
    cross-sectional factor modeling and regression-based strategies.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Cross-sectional normalization is performed independently for each
        timestamp .
    trade_date : str
    factor_col : str
        Name of the factor column to be normal-score transformed.
    eps : float, default 1e-9
        A small clipping constant to avoid probabilities 0 or 1 entering the
        inverse normal CDF, which would result in ±∞.

    Returns
    -------
    pl.DataFrame
        A DataFrame where the factor
        column has been transformed into approximately standard normal scores.

    Notes
    -----
    - The method operates cross-sectionally: each slice is ranked
      independently.
    - Resulting values follow an approximate N(0, 1) distribution regardless
      of the original factor distribution shape (skewness/fat tails).
    - Normal-score scaling is widely used before cross-sectional regressions,
      portfolio construction, and factor combination to stabilize factor behavior.
    - The transformation is monotonic: it preserves the ordinal relationship
      between factor values.
    - Extreme ranks are clipped via `eps` to ensure numerical stability.
    """

    df = (
        pl_df
        .sort([trade_date, factor_col])
        .with_columns([
            pl.arange(1, pl.len() + 1).over(trade_date).alias("rank"),
            pl.len().over(trade_date).alias("n")
        ])
        .with_columns([
            ((pl.col("rank") - 0.5) / pl.col("n")).alias("cdf")
        ])
    )

    cdf_values = df["cdf"].to_numpy()
    z_values = norm.ppf(np.clip(cdf_values, eps, 1-eps))
    df = df.with_columns([
    pl.Series(name=factor_col, values=z_values)
])

    return df

def quantile_binning_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    q: int = 5,          
):
    """
    Perform cross-sectional quantile binning on a factor column.

    This method assigns each factor value into one of `q` quantile bins
    within each cross-sectional slice.  
    The factor values are first ranked, and the rank percentiles are mapped
    into integer bin labels:

        bin_i = floor( (rank_i - 1) * q / n )

    where n is the number of observations in the cross-section.  
    The resulting bin indices range from 0 to q−1.

    Quantile binning is often used to discretize factors for portfolio
    construction, regime classification, or for converting continuous signals
    into ordinal categories that are more robust to outliers.

    Parameters
    ----------
    pl_df : pl.DataFrame
        Quantile binning is applied independently to each timestamp.
    factor_col : str
        Name of the factor column to be discretized into quantile bins.
    q : int, default 5
        Number of quantile buckets.  
        For example, q=5 corresponds to quintile binning (0–4).

    Returns
    -------
    pl.DataFrame
        A DataFrame, where the
        factor column contains integer bin labels in the range [0, q−1].

    Notes
    -----
    - Ties inherit deterministic ordering due to sorting by section.
    - All computations are performed cross-sectionally per 'datetime'.
    - Quantile bins contain approximately equal number of samples, except in
      the presence of repeated values.
    - Useful for long-short portfolio grouping, signal discretization,
      cross-sectional tests, or when factor magnitudes carry less information
      than their ordinal ordering.
    """
    df = (
        pl_df
        .sort([trade_date, factor_col])
        .with_columns([
            pl.arange(1, pl.len() + 1).over(trade_date).alias("rank"),
            pl.len().over(trade_date).alias("n")
        ])
        .with_columns([
            (
                ((pl.col("rank") - 1) * q) / pl.col("n")
            )
            .floor()
            .clip(0, q - 1)        
            .cast(pl.Int32)
            .alias(factor_col)
        ])
    )

    return df

def log_zscore_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    eps: float = 1e-9,       
):
    """
    Apply Log-Zscore standardization to a factor column (log transform →
    cross-sectional Z-score), computed independently for each datetime
    cross-section.

    This transformation is useful for factors with heavy right tails or
    multiplicative scaling effects. The log transform reduces skewness,
    and the subsequent Z-score ensures cross-sectional comparability.

    Parameters
    ----------
    pl_df : pl.DataFrame
    
    trade_date : str
    
    factor_col : str
        Name of the factor column to be transformed.
    
    eps : float, optional (default = 1e-9)
        Small positive constant added before log to avoid log(0) and
        stabilize division during Z-score normalization.

    Returns
    -------
    pl.DataFrame

    Notes
    -----
    Steps performed:
    1. Apply log(x + eps) transform to reduce skewness.
    2. For each datetime cross-section:
       - Compute mean and standard deviation of the log-transformed values.
    3. Apply Z-score normalization:
         z = (log_x - mu) / (sigma + eps)
       Adding eps ensures numerical stability for cases where sigma is zero.
    """

    df = (
        pl_df
        .with_columns([
            (pl.col(factor_col) + eps).log().alias("log_x")
        ])
        .with_columns([
            pl.col("log_x").mean().over(trade_date).alias("mu"),
            pl.col("log_x").std().over(trade_date).alias("sigma"),
        ])
        .with_columns([
            ((pl.col("log_x") - pl.col("mu")) / (pl.col("sigma") + eps))
            .alias(factor_col)
        ])
    )

    return df

def yeo_johnson_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    lambda_: float = 0.0,    
    eps: float = 1e-9
):
    """
    Apply a Yeo–Johnson power transform followed by cross-sectional Z-score
    standardization for each datetime slice.

    The Yeo–Johnson transform (an extension of the Box–Cox transform) handles
    both positive and negative values, making it suitable for factors that
    exhibit heavy tails, skewness, or values around zero. After the nonlinear
    transformation, a Z-score is computed within each datetime cross-section
    to ensure cross-sectional comparability.

    Parameters
    ----------
    pl_df : pl.DataFrame

    trade_date : str

    factor_col : str
        Name of the factor column to be transformed.

    lambda_ : float, optional (default = 0.0)
        Power transform parameter controlling the degree of nonlinearity.
        - lambda_ = 0   → log-like transform for x ≥ 0
        - lambda_ = 2   → log-like transform for x < 0
        - Other values follow the standard Yeo–Johnson formula.

    eps : float, optional (default = 1e-9)
        Small constant added to avoid numerical instability when dividing
        by standard deviation.

    Returns
    -------
    pl.DataFrame

    Notes
    -----
    Steps performed:
    1. Apply vectorized Yeo–Johnson transform:
        For x ≥ 0:
            - If λ = 0:        y = log(x + 1)
            - Else:            y = ((x + 1)^λ - 1) / λ

        For x < 0:
            - If λ = 2:        y = -log(1 - x)
            - Else:            y = -(((1 - x)^(2 - λ) - 1) / (2 - λ))

    2. Compute cross-sectional mean and standard deviation within each datetime.

    3. Apply Z-score normalization:
           z = (pt - mu) / (sigma + eps)
       Adding eps ensures stability when sigma is zero or extremely small.
    """

    x = pl.col(factor_col)

    yj = (
        pl.when(x >= 0)
        .then(
            pl.when(lambda_ == 0)
                .then((x + 1).log())
                .otherwise(((x + 1) ** lambda_ - 1) / lambda_)
        )
        .otherwise(
            pl.when(lambda_ == 2)
                .then(-( (-x + 1).log() ))
                .otherwise(-(((-x + 1) ** (2 - lambda_) - 1) / (2 - lambda_)))
        )
    ).alias("pt")

    df = (
        pl_df
        .with_columns([yj])
        .with_columns([
            pl.col("pt").mean().over(trade_date).alias("mu"),
            pl.col("pt").std().over(trade_date).alias("sigma"),
        ])
        .with_columns([
            ((pl.col("pt") - pl.col("mu")) / (pl.col("sigma") + eps))
            .alias(factor_col)
        ])
    )

    return df

def boxcox_standardize(
    pl_df: pl.DataFrame,
    trade_date,
    factor_col: str,
    lambda_: float = 0.0,
    eps: float = 1e-9
):
    """
    Apply a Box–Cox power transform followed by cross-sectional Z-score
    standardization for each datetime slice.

    The Box–Cox transform requires strictly positive inputs. Therefore,
    the factor is automatically shifted so that its minimum becomes > 0.
    This transform is useful for reducing skewness, compressing heavy tails,
    and stabilizing variance before applying cross-sectional normalization.

    Parameters
    ----------
    pl_df : pl.DataFrame

    trade_date : str
    
    factor_col : str
        Name of the factor column to be transformed.

    lambda_ : float, optional (default = 0.0)
        Power transform parameter:
        - λ = 0   → log transform:  y = log(x)
        - else    → power transform: y = (x^λ − 1) / λ

    eps : float, optional (default = 1e-9)
        Small constant added to avoid numerical issues in:
        - shifting values to ensure x > 0
        - dividing by very small standard deviations

    Returns
    -------
    pl.DataFrame

    Notes
    -----
    Steps performed:
    1. Compute the global minimum of the factor. If min ≤ 0, shift all values:
           shift = -min + eps
       so that (x + shift) > 0 for all rows.

    2. Apply Box–Cox transform:
         If λ = 0:
             pt = log(x)
         Else:
             pt = (x^λ − 1) / λ

    3. Compute cross-sectional mean and standard deviation within each section.

    4. Apply Z-score normalization:
           z = (pt - mu) / (sigma + eps)

       Adding eps prevents division by zero when sigma is extremely small.
    """

    min_val = pl_df[factor_col].min()
    shift = float(-min_val + eps) if min_val <= 0 else 0.0

    x = (pl.col(factor_col) + shift)

    pt = (
        pl.when(lambda_ == 0)
        .then(x.log())
        .otherwise(((x ** lambda_) - 1) / lambda_)
    ).alias("pt")

    df = (
        pl_df
        .with_columns([pt])
        .with_columns([
            pl.col("pt").mean().over(trade_date).alias("mu"),
            pl.col("pt").std().over(trade_date).alias("sigma"),
        ])
        .with_columns([
            ((pl.col("pt") - pl.col("mu")) / (pl.col("sigma") + eps))
            .alias(factor_col)
        ])
    )

    return df

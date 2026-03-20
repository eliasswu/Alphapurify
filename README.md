[![Python version](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat)](https://pypi.python.org/pypi/quantstats)
[![PyPi version](https://img.shields.io/pypi/v/alphapurify.svg?maxAge=60)](https://pypi.python.org/pypi/alphapurify)
[![PyPi status](https://img.shields.io/pypi/status/alphapurify.svg?maxAge=60)](https://pypi.python.org/pypi/alphapurify)
[![License](https://img.shields.io/pypi/l/alphapurify)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/alphapurify/#files)

# AlphaPurify: Factor research for quants

**AlphaPurify** Python library for factor construction, preprocessing, backtesting, and factor return attribution to help quants rapidly validate ideas.

---

![IC](assets/logo.jpg)

---

### AlphaPurify is comprised of 4 main modules:

1.  **`alphapurify.FactorAnalyzer`** — for IC testing and quantile portfolio analysis to evaluate factor predictive ability.
2.  **`alphapurify.AlphaPurifier`** — for factor preprocessing, including 40+ Winsorization, Neutralization, and Standardization methods.
3.  **`alphapurify.Database`** — for reading, writing, and aggregating financial and factor datasets.
4.  **`alphapurify.Exposures`** — for factor correlation analysis and factor-based return attribution.

---

##  Quick Start

### 1.Install with pip
Users can easily install ``AlphaPurify`` by pip according to the following command.

```bash
pip install alphapurify
```
**Note**: pip will install the latest stable ``AlphaPurify``. However, the main branch of AlphaPurify is in active development. If you want to test the latest scripts or functions in the main branch. Please install ``AlphaPurify`` with clone.

---

### 2.Load your DataFrame
| datetime           | symbol | close | volume | factor | momentum_12_1 | vol_60 | beta_252 |
|:-------------------|:------|------:|------:|------:|--------------:|------:|--------:|
| 2024-01-01 09:30   | AAPL  | 189.9 | 120034 | 0.42 | 0.15 | 0.21 | 1.08 |
| 2024-01-01 09:31   | AAPL  | 190.0 | 98321  | 0.38 | 0.16 | 0.22 | 1.07 |
| 2024-01-01 09:32   | AAPL  | 190.4 | 101245 | 0.41 | 0.17 | 0.23 | 1.06 |
| 2024-01-01 09:30   | MSFT  | 378.5 | 84211  | -0.15 | -0.05 | 0.18 | 0.95 |
| 2024-01-01 09:31   | MSFT  | 378.9 | 90122  | -0.12 | -0.04 | 0.19 | 0.96 |
| 2024-01-01 09:32   | MSFT  | 379.1 | 95433  | -0.08 | -0.03 | 0.20 | 0.97 |

**p.s. Your DataFrame must include a time column, an asset identifier column, a price column, and your factor column to ensure proper usage.**

---

### 3.Creating backtesting report
```bash
from alphapurify import AlphaPurifier, FactorAnalyzer, Pure_Exposures

# preprocess
df = (
    AlphaPurifier(df, factor_col="alpha_003")
    .winsorize(method="mad")
    .standardize(method="zscore")
    .to_result()
)

#backtest
FA = FactorAnalyzer(base_df=df,
                    trade_date_col='datetime',
                    symbol_col='symbol',
                    price_col='close',
                    factor_name='alpha_003')
FA.run()
FA.create_long_return_sheet()
FA.create_long_short_return_sheet()
FA.create_short_return_sheet()
FA.create_single_fac_ic_sheet()

#contributions of other factors
Ex = Pure_Exposures(
    base_df=df,
    trade_date_col='datetime',
    symbol_col='symbol',
    price_col='close',
    factor_name='alpha_003',
    exposure_cols=['momentum_12_1', 'vol_60', 'beta_252'],
)

Ex.run()
Ex.plot_pure_exposures()
Ex.plot_pure_returns()
Ex.plot_pure_exposures_and_returns()
Ex.plot_correlations()
```

---

## Examples of Backtesting Reports
### Portfolio for long positions only:
![IC](assets/newplot2.png)
### Return attributions of other factors:
![IC2](assets/newplot3.png)
![IC2](assets/newplot4.png)
![IC2](assets/newplot5.png)

---

## Why AlphaPurify?

For whole pipeline backtesting of your factors,**you merely just need a Dataframe!**

**• Optimized for single-machine research**
 
AlphaPurify is designed with optimized caching, vectorized computation, and multiprocessing wherever possible.

For example, a **15-year daily dataset of the CSI 300 universe** can complete full factor evaluation — including **long-only, long-short, short portfolios and IC analysis** — in **around 30 seconds** on a typical laptop.


**• Adaptive to arbitrary bar frequency**

AlphaPurify works with **any bar frequency** (daily, hourly, minute-level, etc.).  
Return aggregation automatically adapts to the data frequency, while allowing users to explicitly specify the horizon if needed.

**• Professional factor preprocessing toolkit**

AlphaPurify provides **40+ built-in preprocessing methods** for factor research, including common operations such as:

- winsorize
- neutralize
- standardize
 

This allows researchers to rapidly experiment with different factor cleaning pipelines.

**• Lightweight high-performance data backend**

AlphaPurify integrates a fast **Parquet + DuckDB** data layer for factor storage and aggregation.

This avoids the need for configuring complex database systems while still providing **high-performance querying and fast factor construction workflows**.

---

More detailed documentation and examples will be released soon.

Leave me a feedback after using it please!
---

**Elias Wu**




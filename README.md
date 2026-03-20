[![Python version](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat)](https://pypi.python.org/pypi/quantstats)
[![PyPi version](https://img.shields.io/pypi/v/alphapurify.svg?maxAge=60)](https://pypi.python.org/pypi/alphapurify)
[![PyPi status](https://img.shields.io/pypi/status/alphapurify.svg?maxAge=60)](https://pypi.python.org/pypi/alphapurify)
[![License](https://img.shields.io/pypi/l/alphapurify)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/alphapurify/#files)

# AlphaPurify: Factor research for quants

**AlphaPurify** Python library for factor construction, preprocessing, backtesting, and factor return attributions to help quants rapidly validate ideas.

---

![IC](assets/logo.jpg)

---

### AlphaPurify is comprised of 4 main modules:

1.  **`alphapurify.FactorAnalyzer`** — for IC testing and quantile portfolio analysis to evaluate factor predictive ability.
2.  **`alphapurify.AlphaPurifier`** — for factor preprocessing, including 40+ Winsorization, Neutralization, and Standardization methods.
3.  **`alphapurify.Database`** — for reading, writing, and aggregating financial and factor datasets.
4.  **`alphapurify.Exposures`** — for factor correlation analysis and factor-based return attribution.

---

## AlphaPurify vs Other Quant Libraries

| Feature / Library | AlphaPurify | Qlib | Backtrader | Alphalens | QuantStats | Pyfolio |
|:------------------|:------------|:--------|:------------|:------------|:-------------|:-------------|
| Computation Speed | 🚀 Very Fast (vectorized + multiprocessing) | ❌ Slow (heavy infrastructure) | ⚠️ Medium | ✅ Fast | no backtest | no backtest |
| Factor Preprocessing (40+) | ✅ Built-in | ⚠️ Limited | ❌ No | ❌ No | ❌ No | ❌ No |
| IC Analysis | ✅ Native | ✅ Yes | ❌ No | ✅Yes | ❌ No | ❌ No |
| Long / Short / Long-Short Rebalancing Quantile Backtest | ✅ Native | ⚠️ Indirect | ⚠️ Indirect | ❌ No | ❌ No | ❌ No |
| Factor Return Attribution | ✅ Native | ⚠️ Indirect | ❌ No | ❌ No | ❌ No | ❌ No |
| Multi-Frequency Support | ✅ Any (min → daily) | ⚠️ Mostly daily | ⚠️ Mostly daily | ⚠️ Mostly daily | ❌ No | ❌ No |
| Setup Complexity | 🟢 Low | 🔴 High | 🟡 Medium | 🟢 Low | 🟢 Low | 🟢 Low |
| Data Backend Support | ✅ Parquet + DuckDB | ⚠️ Custom infra | ❌ None | ❌ None | ❌ None | ❌ None |

While ``AlphaPurify`` may look similar to ``Alphalens`` at first glance, it goes far beyond IC analysis and simple graphs.
It natively supports long, short, and long-short rebalancing backtests and delivers a new generation of interactive visualizations by Plotly.

``AlphaPurify`` is different from libraries like ``QuantStats`` and ``Pyfolio``, which primarily focus on analyzing return curves and portfolio performance. Compared to tools like ``Qlib`` and ``Backtrader``, ``AlphaPurify`` directly provides a lightweight, fast factor-driven rebalancing backtesting framework — eliminating the need for users to build custom pipelines or infrastructure in these libraries.

In short, ``AlphaPurify`` provide quants with a whole factor testing pipeline and beautiful interactive reports.

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

### 3.Creating backtesting reports
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


More detailed documentation and examples will be released soon.

Leave me a feedback after using it please!
---

**Elias Wu**




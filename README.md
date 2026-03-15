# AlphaPurify: Factor analytics for quants

**AlphaPurify** Python library for financial data aggregation, factor construction, IC testing, factor return attribution, full-pipeline backtesting, and large-scale experimentation to help quants rapidly validate ideas.

---

![IC](assets/logo.jpg)

---

### AlphaPurify is comprised of 4 main modules:

1.  **`alphapurify.FactorAnalyzer`** — for IC testing and quantile portfolio analysis to evaluate factor predictive ability.
2.  **`alphapurify.AlphaPurifier`** — for factor preprocessing and method discovery, including cleaning, transformation, and factor inspection utilities.
3.  **`alphapurify.Database`** — for reading, writing, and aggregating financial and factor datasets.
4.  **`alphapurify.Exposures`** — for factor correlation analysis and factor-based return attribution.

---

##  Quick Start

### 1.Install with pip
Users can easily install ``AlphaPurify`` by pip according to the following command.

```bash
pip install alphapurify
```
**Note**: pip will install the latest stable ``AlphaPurify``. However, the main branch of qlib is in active development. If you want to test the latest scripts or functions in the main branch. Please install ``AlphaPurify`` with clone.

---

### 2.Load your DataFrame
| datetime           | symbol | open  | high  | low   | close | volume | factor | momentum_12_1 | vol_60 | beta_252 |
|:-------------------|:------|-----:|-----:|-----:|-----:|------:|------:|--------------:|------:|--------:|
| 2024-01-01 09:30   | AAPL  | 189.2 | 190.1 | 188.7 | 189.9 | 120034 | 0.42 | 0.15 | 0.21 | 1.08 |
| 2024-01-01 09:31   | AAPL  | 189.9 | 190.3 | 189.5 | 190.0 | 98321  | 0.38 | 0.16 | 0.22 | 1.07 |
| 2024-01-01 09:32   | AAPL  | 190.0 | 190.6 | 189.8 | 190.4 | 101245 | 0.41 | 0.17 | 0.23 | 1.06 |
| 2024-01-01 09:30   | MSFT  | 378.1 | 379.0 | 377.6 | 378.5 | 84211  | -0.15 | -0.05 | 0.18 | 0.95 |
| 2024-01-01 09:31   | MSFT  | 378.5 | 379.2 | 378.0 | 378.9 | 90122  | -0.12 | -0.04 | 0.19 | 0.96 |
| 2024-01-01 09:32   | MSFT  | 378.9 | 379.5 | 378.4 | 379.1 | 95433  | -0.08 | -0.03 | 0.20 | 0.97 |
---

### 3.Creating reports
```bash
# preprocess
df = (
    AlphaPurifier(df, factor_col="alpha")
    .winsorize(method="mad")
    .standardize(method="zscore")
    .to_result()
)

#backtest
FA = FactorAnalyzer(base_df=df,
                    trade_date_col='datetime',
                    symbol_col='synbol',
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

## Exemples of Outputs

![IC](assets/newplot2.png)
![IC2](assets/newplot3.png)
![IC2](assets/newplot4.png)
![IC2](assets/newplot5.png)






## 📊 Example Workflow

from alphapurify import AlphaPurifier, FactorAnalyzer


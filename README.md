# AlphaPurify: Factor analytics for quants

**AlphaPurify** Python library for financial data aggregation, factor construction, IC testing, factor return attribution, full-pipeline backtesting, and large-scale experimentation to help quants rapidly validate ideas.


### AlphaPurify is comprised of 4 main modules:

1.  **`alphapurify.FactorAnalyzer`** — for IC testing and quantile portfolio analysis to evaluate factor predictive ability.
2.  **`alphapurify.AlphaPurifier`** — for factor preprocessing and method discovery, including cleaning, transformation, and factor inspection utilities.
3.  **`alphapurify.Database`** — for reading, writing, and aggregating financial and factor datasets.
4.  **`alphapurify.Exposures`** — for factor correlation analysis and factor-based return attribution.

## Factor Performance

![IC](assets/newplot.png)

## Installation

```bash
pip install alphapurify

---

## Quick Start: You merely just need a Dataframe!

## 📊 Example Workflow

from alphapurify import AlphaPurifier, FactorAnalyzer

# Load your DataFrame
df = ...

# Clean factor
cleaned = (
    AlphaPurifier(df, factor_col="alpha")
    .winsorize(method="mad")
    .neutralize(neutralizer_cols=["size", "industry"])
    .standardize(method="zscore")
    .to_result()
)

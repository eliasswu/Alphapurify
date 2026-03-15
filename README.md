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

### Install with pip
Users can easily install ``AlphaPurify`` by pip according to the following command.

```bash
pip install alphapurify
```
**Note**: pip will install the latest stable ``AlphaPurify``. However, the main branch of qlib is in active development. If you want to test the latest scripts or functions in the main branch. Please install ``AlphaPurify`` with clone.

---

## Exemples of Outputs

![IC](assets/newplot.png1)
![IC](assets/newplot.png2)
![IC](assets/newplot.png3)
![IC](assets/newplot.png4)
![IC](assets/newplot.png5)




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

![IC](assets/logo.jpg)
<p align="center">
  <a href="https://pypi.python.org/pypi/alphapurify">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=flat">
  </a>
  <a href="https://pypi.python.org/pypi/alphapurify">
    <img src="https://img.shields.io/pypi/v/alphapurify.svg?maxAge=60">
  </a>
  <a href="https://pypi.python.org/pypi/alphapurify">
    <img src="https://img.shields.io/pypi/status/alphapurify.svg?maxAge=60">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/pypi/l/alphapurify">
  </a>
  <a href="https://pypi.org/project/alphapurify/#files">
    <img src="https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey">
  </a>
</p>

<p align="center">
  <a href="README.md">English README</a> | <a href="README_CN.md">简体中文 README</a> | <a href="README_CN2.md">繁体中文 README</a>
</p>

# AlphaPurify：面向量化研究员的因子分析库

**AlphaPurify** 库用于因子构建、因子预处理、因子回测和因子收益率归因，帮助量化研究员快速验证想法。

---

## 4个主要功能:

  1.**`alphapurify.FactorAnalyzer`** — 用于IC/ Rank IC测试和多头，空头，多空分层测试。

  2.**`alphapurify.AlphaPurifier`** — 用于因子预处理，包括40+的去极值、中性化和标准化方法(e.g.，ridge回归，lasso回归，PCA分解，etc.).

  3.**`alphapurify.Database`** — 用于金融数据聚合、因子构建和因子入库。

  4.**`alphapurify.Exposures`** — 用于因子相关性分析和因子收益率归因分析。

---

## Pipeline Overview
![Pipeline](assets/pipeline.jpg)
---

### 完整文档及示例: **[English docs](./examples)**

---

## 主要优点:
### - 极致高性能 — 在标准 i7 CPU 上，可在 25 秒内处理 400 万+ 行数据（15 年沪深 300），包括多头/空头、多空以及 IC 回测，并生成 4 个交互式报告。
### - 大规模稳定性 — 可靠支持千万级数据（如分钟级数据），并通过内存优化设计防止溢出。
### - 40+ 因子预处理方法 — 内置专业的因子清洗工具，支持从超高频到低频的数据处理需求。
### - 灵活周期支持 — 支持同时运行无限数量的调仓周期和 IC 观测窗口，实现丰富的多维因子分析。

---

## AlphaPurify vs 其他量化库

| 领域 / 库 | AlphaPurify | Qlib | Backtrader | Alphalens | QuantStats | Pyfolio |
|:------------------|:------------|:--------|:------------|:------------|:-------------|:-------------|
| 计算速度 | 🚀 极速 (Rust向量化 + 多进程) | ❌ 慢 (重基础设施) | ⚠️ 中等 | ✅ 快 | 无回测 | 无回测 |
| 因子预处理 (40+) | ✅ 原生内置 | ⚠️ 有限制 | ❌ 无 | ❌ 无 | ❌ 无 | ❌ 无 |
| IC 分析 | ✅ 原生内置 | ✅ 支持 | ❌ 无 | ✅ 支持 | ❌ 无 | ❌ 无 |
| 多头 / 空头 / 多空再平衡分层回测 | ✅ 原生内置 | ✅ 支持 | ⚠️ 不直接支持 | ❌ 无 | ❌ 无 | ❌ 无 |
| 因子收益率归因 | ✅ 支持 | ⚠️ 不直接支持 | ❌ 无 | ❌ 无 | ❌ 无 | ❌ 无 |
| 多频率bar支持 | ✅ 任意 (微秒 → 年频率) | ⚠️ 有限制 | ⚠️ 日频 | ⚠️ 日频 | ⚠️ 有限制 | ⚠️ 有限制 |
| 复杂度 | 🟢 低 | 🔴 高 | 🟡 中等 | 🟢 低 | 🟢 低 | 🟢 低 |
| 数据库支持 | ✅ Parquet + DuckDB | ⚠️ 传统架构 | ❌ 无 | ❌ 无 | ❌ 无 | ❌ 无 |

``AlphaPurify``虽然看起来``Alphalens``，但它远远超出了IC分析和简单图表的范畴。它支持多头，空头，多空的再平衡回测、因子清洗、因子收益率归因，并由 Plotly 提供新一代交互式可视化图表。

``AlphaPurify``跟``QuantStats``和``Pyfolio``这些主要关注收益率曲线和投资组合表现分析而非回测的库不同。但是相比于``Qlib``和``Backtrader``等工具，``AlphaPurify``直接提供了轻量级、快速的因子驱动再平衡回测框架——用户无需在这些库中自己写复杂的流水线或回测流程。

简而言之，``AlphaPurify``为量化研究员提供全因子测试流程和精美的互动报告，以快速验证想法。

---

##  快速开始

### 1.使用pip安装
用户可以通过以下命令轻松通过 pip 安装``AlphaPurify``。

```bash
pip install alphapurify
```
注意：pip 会安装最新的稳定版``AlphaPurify``。不过，AlphaPurify的主分支正在积极开发中。如果你想测试主分支中最新的脚本或函数，请用克隆安装。

---

### 2.加载你的DataFrame
| datetime           | symbol | close | volume | alpha_003 | momentum_12_1 | vol_60 | beta_252 |
|:-------------------|:------|------:|------:|------:|--------------:|------:|--------:|
| 2024-01-01 09:30   | AAPL  | 189.9 | 120034 | 0.42 | 0.15 | 0.21 | 1.08 |
| 2024-01-01 09:31   | AAPL  | 190.0 | 98321  | 0.38 | 0.16 | 0.22 | 1.07 |
| 2024-01-01 09:32   | AAPL  | 190.4 | 101245 | 0.41 | 0.17 | 0.23 | 1.06 |
| 2024-01-01 09:30   | MSFT  | 378.5 | 84211  | -0.15 | -0.05 | 0.18 | 0.95 |
| 2024-01-01 09:31   | MSFT  | 378.9 | 90122  | -0.12 | -0.04 | 0.19 | 0.96 |
| 2024-01-01 09:32   | MSFT  | 379.1 | 95433  | -0.08 | -0.03 | 0.20 | 0.97 |

**p.s. 你的数据框架必须包含时间栏、标的标识栏、价格栏和因子栏，以确保正确使用。**

---

### 3.创建回测报告
```bash
from alphapurify import AlphaPurifier, FactorAnalyzer

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
Ex = PureExposures(
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

## 回测报告示例
### 仅限多头头寸的回测组合：
![IC](assets/newplot2.png)
### 因子值和收益率的归因分析:
![IC2](assets/newplot3.png)
![IC2](assets/newplot4.png)
![IC2](assets/newplot5.png)

---

### 如果你喜欢``AlphaPurify``， 请为这个项目star并fork以支持开发！

---

**Elias Wu 吴一鸣**




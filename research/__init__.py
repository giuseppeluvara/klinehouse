"""klinehouse research layer — turn the collected 1s order-flow data into
honest, reproducible microstructure analytics.

Modules
-------
db          gap-aware data loading from TimescaleDB (or CSV)
features    causal microstructure feature engineering (order-flow imbalance, …)
evaluate    information coefficient, decile monotonicity, cost-aware backtest
volatility  short-horizon volatility forecasting + out-of-sample evaluation
execution   execution alpha (signal-tilted vs TWAP) — value without fees
report      CLI that runs the full study and writes a markdown report + plots
"""

__all__ = ["db", "features", "evaluate", "volatility", "execution", "report"]

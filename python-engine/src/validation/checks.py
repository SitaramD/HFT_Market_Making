import os

IC_MIN        = float(os.getenv("IC_MIN_THRESHOLD", 0.02))
SHARPE_MIN    = float(os.getenv("SHARPE_MIN_THRESHOLD", 1.5))
MAX_DRAWDOWN  = float(os.getenv("MAX_DRAWDOWN_PCT", 5.0))
FILL_RATE_MIN = float(os.getenv("FILL_RATE_MIN", 0.60))

def validate_all(metrics: dict) -> dict:
    results = {}
    results["ic_ok"]        = metrics.get("ic_200", 0)    >= IC_MIN
    results["sharpe_ok"]    = metrics.get("sharpe", 0)    >= SHARPE_MIN
    results["fillrate_ok"]  = metrics.get("fill_rate", 0) >= FILL_RATE_MIN
    results["inventory_ok"] = abs(metrics.get("inventory", 0)) < float(os.getenv("MAX_INVENTORY_BTC", 0.5))
    results["all_pass"]     = all(results.values())
    return results

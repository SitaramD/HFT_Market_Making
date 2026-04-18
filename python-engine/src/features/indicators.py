import numpy as np

def information_coefficient(predictions, actuals):
    if len(predictions) < 2:
        return 0.0
    p = np.array(predictions)
    a = np.array(actuals)
    std_p = p.std()
    std_a = a.std()
    if std_p == 0 or std_a == 0:
        return 0.0
    return float(np.corrcoef(p, a)[0, 1])

def rolling_sharpe(returns, window=60):
    if len(returns) < window:
        return 0.0
    r = np.array(returns[-window:])
    mean = r.mean()
    std  = r.std()
    return float(mean / std * np.sqrt(252 * 1440)) if std > 0 else 0.0

def adverse_selection_rate(fills, mid_moves):
    if not fills:
        return 0.0
    adverse = sum(1 for f, m in zip(fills, mid_moves) if f * m < 0)
    return adverse / len(fills)

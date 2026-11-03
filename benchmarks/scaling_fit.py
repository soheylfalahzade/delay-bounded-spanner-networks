"""Log-log fit used by the scaling study (numpy only, so that the README renderer and the figures stay light)."""
import numpy as np


def fit_exponent(edges, seconds, n_boot=2000, seed=0):
    """Slope of log(time) on log(edges) with a percentile bootstrap interval over the measured networks."""
    x, y = np.log(np.asarray(edges, float)), np.log(np.asarray(seconds, float))
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) >= 2:
            boots.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return slope, float(lo), float(hi)

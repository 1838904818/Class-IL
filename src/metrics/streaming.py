"""Streaming-evaluation metrics (Area-Under-Time / AUT, prequential).

Stub for D13 — to be implemented in Week 2.

Reference: P26 SOUL 2024 (Amalapuram et al.) introduced AUT for class-IL.
"""

def aut(acc_over_time):
    """Trapezoidal Area-Under-Time over the per-step accuracy series.

    Args:
        acc_over_time: 1-D array of per-step accuracies (length = #eval steps).
    Returns:
        AUT in [0, 1].
    """
    import numpy as np
    a = np.asarray(acc_over_time, dtype=float)
    if a.size < 2:
        return float(a.mean()) if a.size else 0.0
    # np.trapz was renamed to np.trapezoid in NumPy 2.0
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trapz(a, dx=1.0) / (a.size - 1))

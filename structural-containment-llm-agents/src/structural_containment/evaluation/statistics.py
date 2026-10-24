"""Statistical helpers for analysing containment experiment results."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def bootstrap_confidence_interval(
    data: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int | None = 42,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Estimate a bootstrap confidence interval for a sample statistic.

    Args:
        data: Observed sample values.
        confidence: Confidence level in (0, 1). Default 0.95.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        statistic: Statistic to estimate: "mean" or "median".

    Returns:
        A (lower, upper) tuple of confidence interval bounds.

    Raises:
        TypeError: If arguments have unsupported types.
        ValueError: If data is empty or arguments are out of range.
    """
    if isinstance(data, (str, bytes)):
        raise TypeError("data must be a sequence of floats.")

    try:
        sample = [float(x) for x in data]
    except (TypeError, ValueError) as exc:
        raise TypeError("data must be a sequence of numeric values.") from exc

    if not sample:
        raise ValueError("data must not be empty.")

    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be a real number.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")

    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None.")

    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'.")

    rng = random.Random(seed)
    n = len(sample)

    def _stat(vals: list[float]) -> float:
        if statistic == "mean":
            return sum(vals) / len(vals)

        sorted_vals = sorted(vals)
        mid = len(sorted_vals) // 2

        if len(sorted_vals) % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

        return float(sorted_vals[mid])

    boot_stats = [_stat(rng.choices(sample, k=n)) for _ in range(n_resamples)]
    boot_stats.sort()

    alpha = 1.0 - confidence
    lower_idx = math.floor(alpha / 2 * n_resamples)
    upper_idx = math.ceil((1.0 - alpha / 2) * n_resamples) - 1

    lower_idx = max(0, lower_idx)
    upper_idx = min(n_resamples - 1, upper_idx)

    return boot_stats[lower_idx], boot_stats[upper_idx]


def cohen_d(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> float:
    """Compute Cohen's d effect size between two independent groups.

    Positive values indicate group A has a larger mean.

    Returns:
        Cohen's d. Returns 0.0 when the groups have equal means and
        zero pooled variance. Returns signed infinity when their means
        differ despite zero pooled variance.

    Raises:
        TypeError: If arguments have unsupported types.
        ValueError: If either group is empty.
    """
    for name, group in (("group_a", group_a), ("group_b", group_b)):
        if isinstance(group, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of floats.")

        try:
            iter(group)
        except TypeError as exc:
            raise TypeError(f"{name} must be a sequence of floats.") from exc

    try:
        a = [float(x) for x in group_a]
        b = [float(x) for x in group_b]
    except (TypeError, ValueError) as exc:
        raise TypeError("Groups must contain numeric values.") from exc

    if not a:
        raise ValueError("group_a must not be empty.")
    if not b:
        raise ValueError("group_b must not be empty.")

    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)

    var_a = sum((x - mean_a) ** 2 for x in a) / len(a) if len(a) > 1 else 0.0
    var_b = sum((x - mean_b) ** 2 for x in b) / len(b) if len(b) > 1 else 0.0

    pooled_var = (var_a * len(a) + var_b * len(b)) / (len(a) + len(b))
    pooled_sd = math.sqrt(pooled_var)

    if pooled_sd == 0.0:
        if mean_a == mean_b:
            return 0.0
        return math.copysign(math.inf, mean_a - mean_b)

    return (mean_a - mean_b) / pooled_sd


def welch_t_statistic(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> float:
    """Compute the Welch t-statistic for two independent samples.

    Returns 0.0 if the standard error is zero.

    Raises:
        TypeError: If arguments have unsupported types.
        ValueError: If either group has fewer than two observations.
    """
    for name, group in (("group_a", group_a), ("group_b", group_b)):
        if isinstance(group, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of floats.")

    try:
        a = [float(x) for x in group_a]
        b = [float(x) for x in group_b]
    except (TypeError, ValueError) as exc:
        raise TypeError("Groups must contain numeric values.") from exc

    if len(a) < 2:
        raise ValueError("group_a must have at least 2 observations.")
    if len(b) < 2:
        raise ValueError("group_b must have at least 2 observations.")

    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)

    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)

    standard_error = math.sqrt(var_a / len(a) + var_b / len(b))

    if standard_error == 0.0:
        return 0.0

    return (mean_a - mean_b) / standard_error

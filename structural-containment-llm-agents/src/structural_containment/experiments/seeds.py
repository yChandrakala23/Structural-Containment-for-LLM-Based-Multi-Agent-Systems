"""Seed generation utilities for reproducible experiments."""

from __future__ import annotations

import hashlib


def make_seed_sequence(
    base_seed: int,
    count: int,
    *,
    offset: int = 0,
) -> tuple[int, ...]:
    """Generate a deterministic sequence of non-overlapping seeds.

    Starting from ``base_seed + offset``, each subsequent seed is
    ``base_seed + offset + i`` for ``i`` in ``range(count)``.  For
    experiments that need fully independent streams, use
    :func:`derive_seed` instead.

    Args:
        base_seed: Root seed from which the sequence is derived.
        count: Number of seeds to generate.
        offset: Starting offset added to each seed index.

    Returns:
        Tuple of ``count`` integer seeds.

    Raises:
        TypeError: If arguments have unsupported types.
        ValueError: If ``count`` or ``offset`` is negative.
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer.")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer.")
    if count < 0:
        raise ValueError("count must be non-negative.")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("offset must be an integer.")
    if offset < 0:
        raise ValueError("offset must be non-negative.")

    return tuple(base_seed + offset + i for i in range(count))


def derive_seed(base_seed: int, *labels: str | int) -> int:
    """Deterministically derive a child seed from a base seed and labels.

    Uses SHA-256 to combine the base seed and an arbitrary number of string
    or integer labels into a single reproducible 32-bit integer seed.  Useful
    for independently seeding sub-experiments by name.

    Args:
        base_seed: Root integer seed.
        *labels: Arbitrary string or integer labels to mix in.

    Returns:
        A non-negative 32-bit integer derived seed.

    Raises:
        TypeError: If ``base_seed`` is not an integer.

    Examples:
        >>> derive_seed(42, "policy", "random", 1)
        <some reproducible int>
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer.")

    parts = [str(base_seed)]
    for label in labels:
        if not isinstance(label, (str, int)) or isinstance(label, bool):
            raise TypeError("Labels must be strings or integers.")
        parts.append(str(label))

    key = ":".join(parts)
    digest = hashlib.sha256(key.encode()).digest()
    # Use the first 4 bytes as an unsigned 32-bit int.
    return int.from_bytes(digest[:4], byteorder="big")


def latin_hypercube_seeds(
    n_samples: int,
    *,
    base_seed: int = 0,
) -> tuple[int, ...]:
    """Generate seeds using a simple Latin Hypercube Sampling (LHS) scheme.

    LHS divides the seed space ``[0, 2**31)`` into ``n_samples`` equal strata
    and picks one seed uniformly from each stratum, ensuring better coverage
    of the seed space than purely sequential seeds.

    Args:
        n_samples: Number of seeds to generate.
        base_seed: Seed for the random sampler (for reproducibility).

    Returns:
        Tuple of ``n_samples`` distinct non-negative integer seeds.

    Raises:
        TypeError: If arguments have unsupported types.
        ValueError: If ``n_samples`` is non-positive.
    """
    import random

    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer.")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer.")

    max_seed = 2**31
    stratum_size = max_seed // n_samples
    rng = random.Random(base_seed)

    seeds: list[int] = []
    for i in range(n_samples):
        low = i * stratum_size
        high = low + stratum_size - 1
        seeds.append(rng.randint(low, high))

    return tuple(seeds)

"""Validated configuration for reproducible containment experiments."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_ATTACKS = frozenset({"random", "highest_out_degree", "targeted"})
SUPPORTED_POLICIES = frozenset({"random", "highest_out_degree"})


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Immutable settings for a family of containment experiments."""

    initial_compromised: tuple[str, ...]
    attack_strategy: str = "targeted"
    policies: tuple[str, ...] = ("random", "highest_out_degree")
    budgets: tuple[int, ...] = (1,)
    propagation_probability: float = 0.2
    max_steps: int = 100
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55)

    def __post_init__(self) -> None:
        if not isinstance(self.initial_compromised, tuple):
            raise TypeError("initial_compromised must be a tuple.")
        if any(not isinstance(node, str) or not node for node in self.initial_compromised):
            raise ValueError("Initial compromised agent IDs must be non-empty strings.")
        if len(set(self.initial_compromised)) != len(self.initial_compromised):
            raise ValueError("initial_compromised must not contain duplicates.")

        if self.attack_strategy not in SUPPORTED_ATTACKS:
            raise ValueError(f"Unsupported attack strategy: {self.attack_strategy!r}.")

        if not isinstance(self.policies, tuple) or not self.policies:
            raise ValueError("policies must be a non-empty tuple.")
        if any(policy not in SUPPORTED_POLICIES for policy in self.policies):
            raise ValueError("policies contains an unsupported containment policy.")
        if len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must not contain duplicates.")

        if not isinstance(self.budgets, tuple) or not self.budgets:
            raise ValueError("budgets must be a non-empty tuple.")
        if any(
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
            for budget in self.budgets
        ):
            raise ValueError("Every budget must be a non-negative integer.")

        if (
            isinstance(self.propagation_probability, bool)
            or not isinstance(self.propagation_probability, (int, float))
            or not 0.0 <= self.propagation_probability <= 1.0
        ):
            raise ValueError("propagation_probability must be between 0 and 1.")

        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer.")

        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise ValueError("seeds must be a non-empty tuple.")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise TypeError("Every seed must be an integer.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates.")

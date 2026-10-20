import pytest

from structural_containment.experiments.config import ExperimentConfig


def test_default_configuration_is_valid():
    config = ExperimentConfig(initial_compromised=("agent_a",))

    assert config.attack_strategy == "targeted"
    assert config.policies == ("random", "highest_out_degree")
    assert config.budgets == (1,)
    assert config.propagation_probability == 0.2
    assert config.max_steps == 100
    assert config.seeds == (11, 22, 33, 44, 55)


def test_custom_configuration_is_preserved():
    config = ExperimentConfig(
        initial_compromised=("a", "b"),
        attack_strategy="random",
        policies=("highest_out_degree",),
        budgets=(0, 1, 2),
        propagation_probability=0.75,
        max_steps=20,
        seeds=(7, 42),
    )

    assert config.initial_compromised == ("a", "b")
    assert config.budgets == (0, 1, 2)
    assert config.seeds == (7, 42)


def test_configuration_is_immutable():
    config = ExperimentConfig(initial_compromised=("a",))

    with pytest.raises(AttributeError):
        config.max_steps = 10


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), True])
def test_invalid_probability_is_rejected(probability):
    with pytest.raises(ValueError):
        ExperimentConfig(
            initial_compromised=("a",),
            propagation_probability=probability,
        )


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        ExperimentConfig(initial_compromised=("a",), budgets=(budget,))


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_max_steps_is_rejected(steps):
    with pytest.raises(ValueError):
        ExperimentConfig(initial_compromised=("a",), max_steps=steps)


def test_empty_initial_attack_is_allowed():
    config = ExperimentConfig(initial_compromised=())

    assert config.initial_compromised == ()


def test_duplicate_initial_agents_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        ExperimentConfig(initial_compromised=("a", "a"))


def test_unknown_attack_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unsupported attack"):
        ExperimentConfig(
            initial_compromised=("a",),
            attack_strategy="unknown",
        )


def test_unknown_containment_policy_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        ExperimentConfig(
            initial_compromised=("a",),
            policies=("unknown",),
        )


def test_duplicate_seeds_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        ExperimentConfig(initial_compromised=("a",), seeds=(42, 42))


def test_boolean_seed_is_rejected():
    with pytest.raises(TypeError, match="integer"):
        ExperimentConfig(initial_compromised=("a",), seeds=(True,))

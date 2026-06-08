"""
Tests for Plackett-Luce model

This test file verifies that the Plackett-Luce model implementation
works correctly for permutation-based EDAs.
"""

import sys
import numpy as np

from pl_learning import LearnPlackettLuce, learn_plackett_luce
from pl_sampling import SamplePlackettLuce, sample_plackett_luce


# Common constants for tests
N_VARS = 5
CARDINALITY = np.arange(N_VARS)
DEFAULT_POP = np.array([
    [0, 1, 2, 3, 4],
    [0, 2, 1, 3, 4],
    [1, 0, 2, 3, 4],
    [0, 1, 3, 2, 4],
    [1, 2, 0, 3, 4],
])
DEFAULT_FITNESS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def _kendall_distance(p, q):
    """Kendall tau distance between two permutations."""
    inv_q = np.argsort(q)
    p2 = inv_q[p]
    dist = 0
    for i in range(len(p2)):
        for j in range(i + 1, len(p2)):
            if p2[i] > p2[j]:
                dist += 1
    return dist


def _learn_model(pop=DEFAULT_POP, fitness=DEFAULT_FITNESS, n_vars=N_VARS, **kwargs):
    """Helper to instantiate LearnPlackettLuce and learn a model."""
    learner = LearnPlackettLuce()
    return learner(
        generation=0,
        n_vars=n_vars,
        cardinality=np.arange(n_vars),
        selected_pop=pop,
        selected_fitness=fitness,
        **kwargs
    )


def _sample_model(model, sample_size, n_vars=N_VARS, population=None, fitness=None, **kwargs):
    """Helper to instantiate SamplePlackettLuce and sample permutations."""
    sampler = SamplePlackettLuce()
    return sampler(
        n_vars=n_vars,
        model=model,
        cardinality=np.arange(n_vars),
        population=population if population is not None else np.array([]),
        fitness=fitness if fitness is not None else np.array([]),
        sample_size=sample_size,
        **kwargs
    )


def _assert_valid_permutations(pop, n_vars):
    """Assert that all rows in pop are valid permutations of length n_vars."""
    expected_set = set(range(n_vars))
    for i, perm in enumerate(pop):
        assert len(perm) == n_vars, f"Sample {i} has incorrect length: {len(perm)}"
        assert len(set(perm)) == n_vars, f"Sample {i} is not a valid permutation: {perm}"
        assert set(perm) == expected_set, f"Sample {i} has wrong elements: {perm}"


def test_plackett_luce_learning():
    """Test Plackett-Luce model learning."""
    print("Testing Plackett-Luce learning...")

    model = _learn_model(max_iter=100, tol=1e-6)

    print(f"  Learned model type: {model['model_type']}")
    print(f"  Weights: {np.round(model['weights'], 4)}")
    print(f"  Weights sum: {np.sum(model['weights']):.6f}")

    assert model['model_type'] == 'plackett_luce'
    assert 'weights' in model
    assert len(model['weights']) == N_VARS
    assert np.all(model['weights'] > 0), "All weights must be positive"
    assert abs(np.sum(model['weights']) - 1.0) < 1e-6, "Weights must sum to 1"

    # Item 0 appears first most often -> should have the highest weight
    assert np.argmax(model['weights']) == 0, \
        f"Expected item 0 to have highest weight, got item {np.argmax(model['weights'])}"

    print("Plackett-Luce learning test passed!")


def test_plackett_luce_learning_uniform():
    """Test that random permutations produce approximately uniform weights."""
    print("\nTesting Plackett-Luce learning with uniform data...")

    rng = np.random.default_rng(0)
    k = 2000
    pop = np.array([rng.permutation(N_VARS) for _ in range(k)])
    fitness = np.zeros(k)

    model = _learn_model(pop=pop, fitness=fitness)

    print(f"  Weights: {np.round(model['weights'], 4)}")
    print(f"  Expected uniform weight: {1.0/N_VARS:.4f}")

    expected = 1.0 / N_VARS
    assert np.allclose(model['weights'], expected, atol=0.05), \
        f"Weights not close to uniform: {model['weights']}"

    print("Uniform data test passed!")


def test_plackett_luce_convenience_function():
    """Test that the convenience function matches the class."""
    print("\nTesting convenience function learn_plackett_luce...")

    pop = np.array([
        [0, 1, 2, 3, 4],
        [1, 0, 2, 3, 4],
        [0, 2, 1, 3, 4],
    ])
    fitness = np.zeros(3)

    model_class = _learn_model(pop=pop, fitness=fitness)
    model_func = learn_plackett_luce(
        generation=0, n_vars=N_VARS, cardinality=CARDINALITY,
        selected_pop=pop, selected_fitness=fitness,
    )

    np.testing.assert_array_almost_equal(
        model_class['weights'], model_func['weights'],
        err_msg="Class and convenience function must return the same weights",
    )

    print(f"  Weights (class):    {np.round(model_class['weights'], 4)}")
    print(f"  Weights (function): {np.round(model_func['weights'], 4)}")
    print("Convenience function test passed!")


def test_plackett_luce_sampling():
    """Test Plackett-Luce model sampling."""
    print("\nTesting Plackett-Luce sampling...")

    model = _learn_model()
    new_pop = _sample_model(
        model,
        sample_size=10,
        population=np.array([[0, 1, 2, 3, 4]]),
        fitness=np.array([1.0]),
        rng=np.random.default_rng(42)
    )

    print(f"  Sampled {len(new_pop)} permutations")
    print(f"  Sample shape: {new_pop.shape}")
    print(f"  First 3 samples:")
    for i in range(min(3, len(new_pop))):
        print(f"    {new_pop[i]}")

    assert new_pop.shape == (10, N_VARS), f"Expected shape (10, 5), got {new_pop.shape}"
    _assert_valid_permutations(new_pop, N_VARS)

    print(" Plackett-Luce sampling test passed!")


def test_plackett_luce_sampling_biased():
    """Test that a heavily biased weight vector places item 0 first most often."""
    print("\nTesting Plackett-Luce sampling with biased weights...")

    weights = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    weights /= weights.sum()
    model = {'weights': weights, 'model_type': 'plackett_luce'}

    new_pop = _sample_model(model, sample_size=500, rng=np.random.default_rng(0))

    fraction_first = np.mean(new_pop[:, 0] == 0)
    print(f"  Fraction of times item 0 appears first: {fraction_first:.3f}")
    assert fraction_first > 0.90, \
        f"Expected item 0 first >90% of the time, got {fraction_first:.3f}"

    print("Biased sampling test passed!")


def test_plackett_luce_sampling_convenience_function():
    """Test that the convenience function sample_plackett_luce works correctly."""
    print("\nTesting convenience function sample_plackett_luce...")

    model = {'weights': np.ones(N_VARS) / N_VARS, 'model_type': 'plackett_luce'}
    new_pop = sample_plackett_luce(
        n_vars=N_VARS,
        model=model,
        cardinality=CARDINALITY,
        population=np.array([]),
        fitness=np.array([]),
        sample_size=20,
    )

    print(f"  Sample shape: {new_pop.shape}")

    assert new_pop.shape == (20, N_VARS)
    _assert_valid_permutations(new_pop, N_VARS)

    print("Convenience function sampling test passed!")


def test_full_eda_cycle():
    """Test a complete EDA cycle with Plackett-Luce."""
    print("\nTesting full EDA cycle with Plackett-Luce...")

    rng = np.random.default_rng(42)
    pop_size = 20
    n_vars = 6

    population = np.array([rng.permutation(n_vars) for _ in range(pop_size)])

    # Simple fitness: minimize Kendall distance to target permutation
    target = np.arange(n_vars)
    fitness = np.array([_kendall_distance(perm, target) for perm in population])

    # Select best half
    n_select = pop_size // 2
    best_indices = np.argsort(fitness)[:n_select]
    selected_pop = population[best_indices]
    selected_fitness = fitness[best_indices]

    print(f"  Population size: {pop_size}")
    print(f"  Selected size: {n_select}")
    print(f"  Best fitness in selected: {np.min(selected_fitness):.2f}")
    print(f"  Worst fitness in selected: {np.max(selected_fitness):.2f}")

    # Learn model
    model = _learn_model(pop=selected_pop, fitness=selected_fitness, n_vars=n_vars)

    print(f"  Learned weights: {np.round(model['weights'], 4)}")

    # Sample new population
    new_pop = _sample_model(
        model,
        sample_size=pop_size,
        n_vars=n_vars,
        population=population,
        fitness=fitness,
        rng=np.random.default_rng(0)
    )

    new_fitness = np.array([_kendall_distance(perm, target) for perm in new_pop])

    print(f"  Original mean fitness:    {np.mean(fitness):.2f}")
    print(f"  New population best fitness: {np.min(new_fitness):.2f}")
    print(f"  New population mean fitness: {np.mean(new_fitness):.2f}")

    _assert_valid_permutations(new_pop, n_vars)

    print("Full EDA cycle test passed!")


def test_round_trip_weights():
    """Test that learn -> sample -> learn recovers the original weight ranking."""
    print("\nTesting round-trip weight recovery...")

    true_weights = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
    model_true = {'weights': true_weights, 'model_type': 'plackett_luce'}

    pop = _sample_model(model_true, sample_size=3000, rng=np.random.default_rng(42))
    model_recovered = _learn_model(pop=pop, fitness=np.zeros(3000))

    print(f"  True weights:      {np.round(true_weights, 4)}")
    print(f"  Recovered weights: {np.round(model_recovered['weights'], 4)}")

    np.testing.assert_allclose(
        model_recovered['weights'], true_weights, atol=0.05,
        err_msg="Recovered weights too far from true weights",
    )

    true_ranking = np.argsort(true_weights)[::-1]
    recovered_ranking = np.argsort(model_recovered['weights'])[::-1]
    assert np.array_equal(true_ranking, recovered_ranking), \
        f"Rankings differ: true={true_ranking}, recovered={recovered_ranking}"

    print(f"  True ranking:      {true_ranking}")
    print(f"  Recovered ranking: {recovered_ranking}")
    print("Round-trip weight recovery test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PLACKETT-LUCE MODEL IMPLEMENTATION")
    print("=" * 60)

    test_functions = [
        test_plackett_luce_learning,
        test_plackett_luce_learning_uniform,
        test_plackett_luce_convenience_function,
        test_plackett_luce_sampling,
        test_plackett_luce_sampling_biased,
        test_plackett_luce_sampling_convenience_function,
        test_full_eda_cycle,
        test_round_trip_weights,
    ]

    try:
        for test_fn in test_functions:
            test_fn()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

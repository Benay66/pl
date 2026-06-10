"""
Plackett-Luce Model Sampling for Permutation-based EDAs

This module implements the sampling method for the Plackett-Luce model.
Permutations are generated sequentially: at each step, the next item is
drawn from the remaining items with probability proportional to its weight.

References:
    [1] R.D. Luce: Individual Choice Behavior: A Theoretical Analysis. Wiley, 1959
    [2] R.L. Plackett: The analysis of permutations. Applied Statistics, 1975
    [3] J. Ceberio, A. Mendiburu, J.A. Lozano: The Plackett-Luce ranking model
        on permutation-based optimization problems. CEC 2013
"""

import numpy as np
from typing import Dict, Any, Optional


class SamplePlackettLuce:
    """Sample permutations from a Plackett-Luce model.

    At each step, an item is drawn from the remaining items with probability
    proportional to its weight, producing a full ranking of all n items.
    """

    def sample(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray = None,
        fitness: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """Sample method to match EDA interface. Calls __call__ internally."""
        if population is None:
            population = np.array([])
        if fitness is None:
            fitness = np.array([])

        sample_size = kwargs.get("sample_size", 100)
        rng = kwargs.get("rng", None)

        return self.__call__(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
            sample_size=sample_size,
            rng=rng,
        )

    def __call__(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample permutations from the Plackett-Luce model.

        At each position, an item is chosen from the remaining items with
        probability proportional to its weight.

        Args:
            n_vars: Number of variables (permutation length)
            model: Model dictionary from learning phase containing:
                   - weights: Weight vector of length n_vars
            cardinality: Not used for permutations
            population: Current population (not used)
            fitness: Fitness values (not used)
            sample_size: Number of permutations to sample
            rng: Random number generator (optional)

        Returns:
            Array of sampled permutations, shape (sample_size, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        weights = model["weights"]
        new_pop = np.zeros((sample_size, n_vars), dtype=int)

        for s in range(sample_size):
            new_pop[s] = self._sample_one(weights, n_vars, rng)

        return new_pop

    def _sample_one(
        self,
        weights: np.ndarray,
        n_vars: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a single permutation from the Plackett-Luce model."""
        perm = np.zeros(n_vars, dtype=int)
        available = list(range(n_vars))

        for i in range(n_vars - 1):
            remaining_weights = weights[available]
            total = np.sum(remaining_weights)

            if total <= 0:
                probs = np.ones(len(available), dtype=float) / len(available)
            else:
                probs = remaining_weights / total
                # Re-normalize to guard against floating-point drift
                probs /= np.sum(probs)

            chosen_idx = rng.choice(len(available), p=probs)
            perm[i] = available.pop(chosen_idx)

        # Last remaining item fills the final position
        perm[n_vars - 1] = available[0]

        return perm


def sample_plackett_luce(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """
    Convenience function to sample from Plackett-Luce model.

    See SamplePlackettLuce for parameter details.
    """
    sampler = SamplePlackettLuce()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size)

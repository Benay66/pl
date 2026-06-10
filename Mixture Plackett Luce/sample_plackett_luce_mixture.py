"""
Sampling module for Mixture of Plackett-Luce Models.

Generates new rankings (permutations) based on a learned mixture model.
Uses the Gumbel-max trick for efficient parallel sampling of Plackett-Luce distributions.
"""

import numpy as np
from typing import Dict, Any


class SamplePlackettLuceMixture:
    """
    Sample permutations from a learned Mixture of Plackett-Luce models.
    """

    def sample(
        self,
        model: Dict[str, Any],
        n_samples: int,
        random_state: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """Alias for __call__ to match some EDA framework interfaces."""
        return self.__call__(
            model=model, 
            n_samples=n_samples, 
            random_state=random_state, 
            **kwargs
        )

    def __call__(
        self,
        model: Dict[str, Any],
        n_samples: int,
        random_state: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate new permutations from the learned mixture model.

        Args:
            model:        Dictionary returned by LearnPlackettLuceMixture
            n_samples:    Number of rankings to generate
            random_state: Random seed for reproducibility
            **kwargs:     Extra arguments (ignored, for compatibility)

        Returns:
            samples: (n_samples, n_vars) array of generated permutations
        """
        # 1. Extraer los parámetros del modelo entrenado
        beta = model["mixing_weights"]
        weights_per_comp = model["weights_per_component"]
        K = model["n_components"]
        n_vars = len(weights_per_comp[0])

        rng = np.random.default_rng(random_state)
        samples = np.zeros((n_samples, n_vars), dtype=int)

        # 2. Elegir a qué clúster va a pertenecer cada nueva muestra
        # Z es un array de tamaño (n_samples,) con valores entre 0 y K-1
        Z = rng.choice(K, size=n_samples, p=beta)

        # 3. Generar los rankings usando el Gumbel-max trick
        for i in range(n_samples):
            k = Z[i]
            w = weights_per_comp[k]

            # Evitar logaritmo de cero
            safe_w = np.maximum(w, 1e-12)
            log_w = np.log(safe_w)

            # Generar ruido Gumbel(0, 1)
            # Gumbel = -log(-log(Uniform(0,1)))
            u = rng.uniform(0, 1, size=n_vars)
            gumbel_noise = -np.log(-np.log(u + 1e-12))

            # Sumar log-utilidad y ruido
            scores = log_w + gumbel_noise

            # El ranking es simplemente ordenar los scores de mayor a menor
            samples[i] = np.argsort(scores)[::-1]

        return samples


def sample_plackett_luce_mixture(
    model: Dict[str, Any],
    n_samples: int,
    **params,
) -> np.ndarray:
    """
    Convenience function to sample from a mixture of Plackett-Luce models.
    """
    sampler = SamplePlackettLuceMixture()
    return sampler(model, n_samples, **params)
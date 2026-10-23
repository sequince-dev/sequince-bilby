"""Fit a Gaussian location with Bilby 3 and NumPy or JAX particles.

Run with ``pixi run -e bilby-main example-backends --backend jax``.

Note: you must currently install bilby 3 from source to use sequince with
different array backends.
"""

import argparse
import math
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import bilby
from orng import RandomGenerator
from sequince.array_api import namespace, to_numpy


class GaussianLocationLikelihood(bilby.Likelihood):
    """Gaussian observations with unknown location and unit standard deviation.

    Parameters
    ----------
    data : array
        One-dimensional observations on the selected array backend.
    """

    def __init__(self, data: Any) -> None:
        super().__init__()
        self.data = data

    def log_likelihood(self, parameters: dict[str, Any]) -> Any:
        """Return a scalar log likelihood using Bilby 3's explicit parameters."""
        xp = namespace(self.data)
        residual = self.data - parameters["mu"]
        return -0.5 * xp.sum(residual**2 + math.log(2 * math.pi))


def main() -> None:
    """Run either Sequince sampler and compare to the analytic posterior mean."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "jax"], default="jax")
    parser.add_argument(
        "--sampler",
        choices=["sequince-smc", "sequince-ps"],
        default="sequince-smc",
    )
    parser.add_argument("--n-particles", type=int, default=256)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    # The optional Pixi environment enables the array API and JAX float64
    # before these libraries are imported. ORNG resolves the backend by name.
    data_rng = RandomGenerator(backend=args.backend, seed=1234)
    data = 1.2 + data_rng.normal(size=32)
    likelihood = GaussianLocationLikelihood(data)
    priors = bilby.core.prior.PriorDict(
        {"mu": bilby.core.prior.Uniform(-5, 5, name="mu")}
    )
    # Keep repeated example runs separate, including Sequince's HDF5 result.
    outdir = args.outdir or Path(mkdtemp(prefix="sequince-bilby3-"))
    outdir.mkdir(parents=True, exist_ok=True)
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=args.sampler,
        outdir=str(outdir),
        label=f"gaussian_{args.backend}",
        n_particles=args.n_particles,
        mutation="differential-evolution",
        mutation_n_steps=2,
        sampler_backend=args.backend,
        sampling_seed=5678,
        npool=1,
        checkpoint=False,
        resume=False,
        plot=False,
    )
    # The bounded uniform prior is effectively flat across this posterior.
    analytic_mean = float(to_numpy(data).mean())
    sampled_mean = result.posterior["mu"].mean()
    print(f"Bilby {bilby.__version__}; backend: {args.backend}")
    print(f"Posterior mean: {sampled_mean:.4f}; analytic: {analytic_mean:.4f}")
    print(f"Log evidence: {result.log_evidence:.4f}")
    print(f"Results: {outdir}")


if __name__ == "__main__":
    main()

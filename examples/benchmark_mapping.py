"""Compare JAX mapping for Gaussian likelihoods, optionally using JIT.

Run in the bilby-main environment with BILBY_ARRAY_API=1, SCIPY_ARRAY_API=1,
and JAX_ENABLE_X64=1. Timings include device synchronization. Sampler timings
include initialization and Sequince result writing, but exclude imports.
Use --jit to compile each sampler's vectorized likelihood callback once.
Each sampler timing creates a new sampler and includes its compilation cost.
"""

import argparse
import logging
from collections.abc import Callable
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import bilby
import jax
import jax.numpy as jnp
import numpy as np
import sequince
from bilby_array_backends import GaussianLocationLikelihood

from sequince_bilby.plugin import SequinceSMC
from sequince_bilby.utils import get_sequince_functions


def measure(
    function: Callable[[], Any], warmup: int, repeats: int
) -> tuple[float, float]:
    """Return first-call and median warmed seconds, synchronizing each call."""

    def timed_call() -> float:
        start = perf_counter()
        jax.block_until_ready(function())
        return perf_counter() - start

    first = timed_call()
    for _ in range(warmup):
        timed_call()
    return first, median(timed_call() for _ in range(repeats))


def main() -> None:
    """Benchmark identical seeded runs with small and larger data arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-particles", type=int, default=64)
    parser.add_argument("--n-data", type=int, nargs="+", default=[32, 4096])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--jit", action="store_true", help="Compile the JAX vmap callback."
    )
    parser.add_argument(
        "--scope", choices=["callback", "sampler", "both"], default="both"
    )
    args = parser.parse_args()
    if (
        min(args.n_particles, *args.n_data, args.repeats) < 1
        or args.warmup < 0
    ):
        parser.error("Counts must be positive and warmup must be nonnegative")
    logging.getLogger("bilby").setLevel(logging.ERROR)
    print(
        f"JAX {jax.__version__}; devices={jax.devices()}; "
        f"x64={jax.config.x64_enabled}; Bilby {bilby.__version__}; "
        f"Sequince {sequince.__version__}",
        flush=True,
    )
    print(
        f"particles={args.n_particles}; repeats={args.repeats}; "
        f"warmup={args.warmup} (in addition to first call); "
        "mutation=differential-evolution, steps=2; seeds=1234/5678",
        flush=True,
    )
    print("scope,n_data,mapping,first_s,warm_median_s", flush=True)
    for n_data in args.n_data:
        reference = None
        data = jnp.asarray(
            1.2 + np.random.default_rng(1234).normal(size=n_data)
        )
        likelihood = GaussianLocationLikelihood(data)
        priors = bilby.core.prior.PriorDict(
            {"mu": bilby.core.prior.Uniform(-5, 5, name="mu")}
        )
        for mapping in ("map", "vmap"):
            jit = args.jit and mapping == "vmap"
            mapping_label = f"{mapping}+jit" if jit else mapping
            with TemporaryDirectory(
                prefix="sequince-mapping-benchmark-"
            ) as outdir:
                options = dict(
                    likelihood=likelihood,
                    priors=priors,
                    outdir=outdir,
                    sampler_backend="jax",
                    map_fn=mapping,
                    jit=jit,
                    n_particles=args.n_particles,
                    rng=5678,
                    mutation="differential-evolution",
                    mutation_n_steps=2,
                    checkpoint=False,
                    resume=False,
                    logging_frequency=0,
                    sequince_log_level="ERROR",
                    soft_init=True,
                )
                if args.scope in ("callback", "both"):
                    sampler = SequinceSMC(**options)
                    sampler._setup_pool()
                    functions = get_sequince_functions(
                        map_fn=mapping, backend="jax", jit=jit
                    )
                    particles = jnp.linspace(-4, 4, args.n_particles)[:, None]

                    def evaluate() -> Any:
                        return functions.log_likelihood(particles)

                    first, warmed = measure(
                        evaluate, args.warmup, args.repeats
                    )
                    residual = np.asarray(data)[None, :] - np.asarray(
                        particles
                    )
                    expected = -0.5 * np.sum(
                        residual**2 + np.log(2 * np.pi), axis=1
                    )
                    np.testing.assert_allclose(evaluate(), expected)
                    print(
                        f"callback,{n_data},{mapping_label},{first:.6f},{warmed:.6f}",
                        flush=True,
                    )
                if args.scope in ("sampler", "both"):
                    iteration = 0
                    last_samples = None
                    last_evidence = None

                    def run() -> np.ndarray:
                        nonlocal iteration, last_samples, last_evidence
                        iteration += 1
                        result = SequinceSMC(
                            **options, label=f"run_{iteration}"
                        ).run_sampler()
                        last_samples = result.nested_samples.to_numpy()
                        last_evidence = result.log_evidence
                        return last_samples

                    first, warmed = measure(run, args.warmup, args.repeats)
                    if reference is None:
                        reference = (last_samples, last_evidence)
                    else:
                        np.testing.assert_allclose(last_samples, reference[0])
                        np.testing.assert_allclose(last_evidence, reference[1])
                    print(
                        f"sampler,{n_data},{mapping_label},{first:.6f},{warmed:.6f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()

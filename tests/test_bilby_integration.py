from unittest.mock import patch

import bilby
import numpy as np
import pytest


def model(x, m, c):
    return m * x + c


def conversion_func(parameters):
    # d = |m| + |c|
    parameters["d"] = abs(parameters["m"]) + abs(parameters["c"])
    return parameters


@pytest.fixture()
def bilby_likelihood(rng):
    x = np.linspace(0, 10, 100)
    injection_parameters = dict(m=0.5, c=0.2)
    sigma = 1.0
    y = model(x, **injection_parameters) + rng.normal(0.0, sigma, len(x))
    likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma)
    return likelihood


@pytest.fixture()
def bilby_priors():
    priors = bilby.core.prior.PriorDict(conversion_function=conversion_func)
    priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
    priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
    priors["d"] = bilby.core.prior.Constraint(name="d", minimum=0, maximum=5)
    return priors


@pytest.fixture()
def sampler_kwargs():
    return dict(
        n_particles=10,
        mutation_n_steps=5,
    )


@pytest.fixture()
def ps_sampler_kwargs():
    return dict(
        n_particles=10,
        target_ess=20,
        mutation_n_steps=5,
    )


def test_run_smc_sampler(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    sampler_kwargs,
):
    outdir = tmp_path / "test_run_sampler"
    outdir.mkdir(parents=True, exist_ok=True)

    bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        sampler="sequince-smc",
        outdir=outdir,
        **sampler_kwargs,
    )


def test_run_sampler_pool(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    sampler_kwargs,
):
    from multiprocessing.dummy import Pool

    outdir = tmp_path / "test_run_sampler_pool"
    outdir.mkdir(parents=True, exist_ok=True)
    with patch("multiprocessing.Pool", new=Pool):
        bilby.run_sampler(
            likelihood=bilby_likelihood,
            priors=bilby_priors,
            sampler="sequince-smc",
            outdir=outdir,
            npool=2,
            **sampler_kwargs,
        )


def test_run_persistent_sampler(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    ps_sampler_kwargs,
):
    outdir = tmp_path / "test_run_ps_sampler"
    outdir.mkdir(parents=True, exist_ok=True)

    bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        sampler="sequince-ps",
        outdir=outdir,
        **ps_sampler_kwargs,
    )

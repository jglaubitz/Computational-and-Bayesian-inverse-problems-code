"""Smoke tests of the modules of the lecture notes (course_code/): imports and tiny runs of the key functions."""
import glob
import importlib
import os

import numpy as np
import pytest

from conftest import COURSE_CODE

import running_examples as re_
import solvers
import priors
import samplers
import approximations

SCRIPT_SUFFIXES = ("_figures.py", "_landscape.py")
MODULES = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(COURSE_CODE, "*.py"))
                 if not p.endswith(SCRIPT_SUFFIXES))


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


# ---------------------------------------------------------------- running examples and Chapter 2
def test_deconvolution_operator(deconv):
    A, t, s, _ = deconv
    assert A.shape == (64, 128) and t.shape == (128,) and s.shape == (64,)
    sv = np.linalg.svd(A, compute_uv=False)
    assert sv[0] / sv[-1] == pytest.approx(3.5e7, rel=0.05)          # kappa(A) = 3.5e7 in the notes


@pytest.mark.parametrize("use_L", [False, True])
def test_tikhonov_matches_normal_equations(deconv, use_L):
    A, t, s, y = deconv
    n, lam = A.shape[1], 1e-3
    L = solvers.difference_matrix(n) if use_L else None
    x = solvers.tikhonov(A, y["smooth"], lam, L)
    Lf = np.eye(n) if L is None else L
    x_ne = np.linalg.solve(A.T @ A + lam * Lf.T @ Lf, A.T @ y["smooth"])
    assert np.linalg.norm(x - x_ne) <= 1e-6 * np.linalg.norm(x_ne)


def test_cgls_residuals_decrease(deconv):
    A, t, s, y = deconv
    its = solvers.cgls(A, y["smooth"], 10)
    res = [np.linalg.norm(A @ x - y["smooth"]) for x in its]
    assert len(its) == 11 and np.all(np.diff(res) <= 1e-12)


def test_l1_regularization_runs(deconv):
    A, t, s, y = deconv
    x, hist = solvers.l1_regularization(A, y["piecewise"], 1e-3, transform="differences", n_iter=200)
    assert x.shape == (128,) and np.all(np.isfinite(x))
    assert hist[-1] < 0.5 * np.sum(y["piecewise"] ** 2)


def test_gaussian_posterior_mean_is_tikhonov(deconv):
    A, t, s, y = deconv
    sigma, gamma = 0.05, 0.1
    mean, cov = solvers.gaussian_posterior(A, y["smooth"], sigma**2, np.eye(128) / gamma**2)
    assert np.allclose(mean, solvers.tikhonov(A, y["smooth"], sigma**2 / gamma**2), atol=1e-8)
    assert np.allclose(cov, cov.T) and np.all(np.linalg.eigvalsh(cov) > 0)


def test_ias_objective_decreases(deconv):
    A, t, s, y = deconv
    x, theta, hist = solvers.ias(A, y["piecewise"], 0.05, 1.0, 1e-3, transform="differences", n_iter=20)
    assert x.shape == (128,) and theta.shape == (127,) and np.all(theta > 0)
    assert np.all(np.diff(hist) <= 1e-8 * np.abs(hist[:-1]))         # block-coordinate descent is monotone


# ---------------------------------------------------------------- priors (Chapter 3)
@pytest.mark.parametrize("transform", ["identity", "differences", "second-differences"])
def test_prior_samplers(transform):
    rng = np.random.default_rng(1)
    for X in (priors.sample_gaussian(50, 0.1, transform, size=3, rng=rng),
              priors.sample_laplace(50, 0.1, transform, size=3, rng=rng),
              priors.sample_gsbl(50, 1.0, 1e-3, transform, size=3, rng=rng)):
        assert X.shape == (3, 50) and np.all(np.isfinite(X))
    np.linalg.cholesky(priors.gaussian_precision(50, 0.1, transform))   # positive definite


def test_log_invgamma_no_overflow():
    lt = priors.sample_log_invgamma(1 / 128, 1e-3, 1000, np.random.default_rng(2))
    assert np.all(np.isfinite(lt))


# ---------------------------------------------------------------- samplers (Chapter 4)
def test_rw_metropolis():
    chain, acc = samplers.rw_metropolis(lambda x: -0.5 * x @ x, np.zeros(2), 200, 1.0, rng=np.random.default_rng(3))
    assert chain.shape == (201, 2) and 0.2 < acc < 0.9


def test_perturb_optimize_matches_posterior():
    rng = np.random.default_rng(4)
    A, y, sigma = rng.standard_normal((3, 5)), rng.standard_normal(3), 0.5
    B = np.eye(5) / 0.8                                               # prior N(0, 0.8^2 I)
    X = samplers.perturb_optimize(A, y, sigma, B, 20000, rng)
    mean, cov = solvers.gaussian_posterior(A, y, sigma**2, B.T @ B)
    assert np.allclose(X.mean(axis=0), mean, atol=0.03)
    assert np.allclose(np.cov(X.T), cov, atol=0.03)


def test_one_gibbs_sweep_hierarchical(deconv):
    A, t, s, y = deconv
    xs, thetas = samplers.gibbs_hierarchical(A, y["piecewise"], 0.05, 1.0, 1e-3, 1, rng=np.random.default_rng(5))
    assert xs.shape == (1, 128) and thetas.shape == (1, 127)
    assert np.all(np.isfinite(xs)) and np.all(thetas > 0)


def test_one_gibbs_sweep_laplace(deconv):
    A, t, s, y = deconv
    xs, taus = samplers.gibbs_laplace(A, y["piecewise"], 0.05, 0.1, 1, rng=np.random.default_rng(6))
    assert xs.shape == (1, 128) and taus.shape == (1, 127) and np.all(taus > 0)


def test_hmc_numpy():
    chain, acc = samplers.hmc(lambda x: 0.5 * x @ x, lambda x: x, np.zeros(3), 50, 0.3, 5,
                              rng=np.random.default_rng(7))
    assert chain.shape == (51, 3) and acc > 0.7


def test_hmc_jax_few_steps():
    import jax
    import jax.numpy as jnp
    step = samplers.make_hmc_jax(lambda x: 0.5 * jnp.sum(x**2), 0.3, 5, np.eye(4))
    key, x, n_acc = jax.random.PRNGKey(0), jnp.zeros(4), 0
    for _ in range(10):
        key, sub = jax.random.split(key)
        x, acc, dH = step(sub, x)
        n_acc += int(acc)
    assert np.all(np.isfinite(np.asarray(x))) and n_acc >= 7


def test_elliptical_slice():
    chain, n_eval = samplers.elliptical_slice(lambda w: -np.sum((w - 1.0) ** 2), np.zeros(3), 50,
                                              rng=np.random.default_rng(8))
    assert chain.shape == (51, 3) and n_eval >= 1


def test_diagnostics_on_iid_draws():
    rng = np.random.default_rng(9)
    chains = rng.standard_normal((4, 2000))
    assert samplers.autocorrelation(chains[0])[0] == pytest.approx(1.0)
    assert 0.6 * 2000 < samplers.ess(chains[0]) < 1.5 * 2000
    assert samplers.split_rhat(chains) < 1.02 and samplers.rank_rhat(chains) < 1.02


# ---------------------------------------------------------------- approximations (Chapter 5)
def test_lowrank_laplace_full_rank_is_exact(deconv):
    A, t, s, y = deconv
    sigma, gamma = 0.05, 0.1
    R = gamma * np.eye(128)
    lam, V = approximations.lowrank_laplace(A, R, sigma)
    _, cov = solvers.gaussian_posterior(A, y["smooth"], sigma**2, np.eye(128) / gamma**2)
    assert np.allclose(approximations.lowrank_std(lam, V, R, 128), np.sqrt(np.diag(cov)), rtol=1e-6)
    X = approximations.lowrank_sample(np.zeros(128), lam, V, R, 10, 5, np.random.default_rng(10))
    assert X.shape == (5, 128)


def test_gaussian_vi_two_dimensional():
    import jax.numpy as jnp
    mu = jnp.array([1.0, -1.0])
    U = lambda x: 0.5 * jnp.sum((x - mu) ** 2 / jnp.array([1.0, 0.25]))
    m, C, rec = approximations.gaussian_vi(U, np.zeros(2), family="full", n_iter=400, lr=0.05, lr_final=1e-3, seed=0)
    assert np.allclose(m, [1.0, -1.0], atol=0.2)
    assert rec["elbo"][-1] > rec["elbo"][0]


def test_importance_weights():
    from scipy.stats import norm
    rng = np.random.default_rng(11)
    x = rng.standard_normal(4000)                                     # draws of q = N(0, 1)
    assert approximations.pareto_khat(norm.logpdf(x, 0, 0.8) - norm.logpdf(x)) < 0.5     # light-tailed weights
    assert approximations.pareto_khat(norm.logpdf(x, 0, 3.0) - norm.logpdf(x)) > 0.7     # q too narrow
    est, ess = approximations.importance_estimate(x, np.zeros_like(x))
    assert est == pytest.approx(x.mean()) and ess == pytest.approx(len(x))


def test_cavi_hierarchical(deconv):
    A, t, s, y = deconv
    mean, cov, (a, b), n_it = approximations.cavi_hierarchical(A, y["piecewise"], 0.05, 1.0, 1e-3, n_iter=50)
    assert mean.shape == (128,) and cov.shape == (128, 128) and np.allclose(cov, cov.T)
    assert np.all(np.diag(cov) > 0) and a == 1.5 and np.all(b > 0) and 1 <= n_it <= 50


def _linear_ensemble_problem():
    rng = np.random.default_rng(12)
    n, m, J = 10, 5, 200
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    sigma = 0.1
    y = A @ x_true + sigma * rng.standard_normal(m)
    return A, y, sigma, rng.standard_normal((J, n)), rng


def test_ensemble_kalman_step_reduces_misfit():
    A, y, sigma, X0, rng = _linear_ensemble_problem()
    X1 = approximations.ensemble_kalman_step(X0, X0 @ A.T, y, sigma, rng)
    assert X1.shape == X0.shape
    assert np.linalg.norm(A @ X1.mean(axis=0) - y) < 0.5 * np.linalg.norm(A @ X0.mean(axis=0) - y)


def test_es_mda_and_eki():
    A, y, sigma, X0, rng = _linear_ensemble_problem()
    F = lambda x: A @ x
    X, n_fwd, halv = approximations.es_mda(F, X0, y, sigma, 3, rng)
    assert X.shape == X0.shape and np.all(np.isfinite(X)) and halv == 0
    X, recs, n_fwd = approximations.eki(F, X0, y, sigma, 3, rng, prior_factor=np.eye(10),
                                        record=lambda X, FX: {"misfit": np.linalg.norm(FX.mean(0) - y)})
    assert len(recs) == 3 and np.all(np.isfinite(X))


# ---------------------------------------------------------------- the bathymetry forward model (NumPy and JAX)
def test_swe_jax_matches_numpy_and_jacobian():
    import jax
    import swe_jax
    n, m = 200, 20
    x, dx = re_.make_grid(n)
    x_obs = np.linspace(0.5, re_.L - 0.5, m)
    b = re_.bottom_smooth(x)[:-1]                                       # unknowns: b_1..b_{n-1}; b_n = 0 is fixed
    F = swe_jax.make_forward(x, x_obs)
    eta_jax = np.asarray(F(b))
    eta_np = re_.surface_at_sensors(np.r_[b, 0.0], x, x_obs)
    assert eta_jax.shape == (m,) and np.allclose(eta_jax, eta_np, atol=1e-10)
    J = np.asarray(jax.jacfwd(F)(b))
    J_fd = re_.jacobian_fd(lambda bb: re_.surface_at_sensors(np.r_[bb, 0.0], x, x_obs), b)
    assert J.shape == (m, n - 1) and np.allclose(J, J_fd, atol=1e-6)


def test_gauss_newton_bathymetry():
    import jax
    import swe_jax
    n, m = 100, 20
    x, dx = re_.make_grid(n)
    x_obs = np.linspace(0.5, re_.L - 0.5, m)
    F = swe_jax.make_forward(x, x_obs)
    jac = jax.jit(jax.jacfwd(F))
    b_true = re_.bottom_smooth(x)[:-1]
    y = np.asarray(F(b_true)) + 0.01 * np.random.default_rng(13).standard_normal(m)
    L = solvers.difference_matrix(n - 1)
    b, hist = solvers.gauss_newton(lambda v: np.asarray(F(v)), lambda v: np.asarray(jac(v)), y, np.zeros(n - 1),
                                   lam=1e-2, L=L, n_iter=5)
    assert hist[-1] < hist[0] and np.all(np.isfinite(b))

"""Smoke tests of the starter code of the default projects (projects/project_*/).

Each project is a folder of plain modules (project C has its own priors.py), so every check runs in a separate
Python process in the project folder. The data generators run in a temporary copy, and the regenerated data are
compared with the shipped files: the shipped data are reproducible from the code with the default seed."""
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from conftest import PROJECTS


def run(cwd, code="", args=None, timeout=300):
    cmd = [sys.executable] + (args if args is not None else ["-c", textwrap.dedent(code)])
    r = subprocess.run(cmd, cwd=cwd, env=dict(os.environ, MPLBACKEND="Agg"), capture_output=True, text=True,
                       timeout=timeout)
    assert r.returncode == 0, f"{' '.join(cmd[:3])} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def project(name):
    path = os.path.join(PROJECTS, name)
    if not os.path.isdir(path):
        pytest.skip(f"{name} not present")
    return path


def copy_project(name, tmp_path):
    dst = tmp_path / name
    shutil.copytree(project(name), dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def assert_same_npz(f_new, f_old, rtol=1e-6, atol=1e-10):
    a, b = np.load(f_new), np.load(f_old)
    assert sorted(a.files) == sorted(b.files), f"{os.path.basename(f_old)}: different keys"
    for k in a.files:
        if a[k].dtype.kind in "fc":
            assert np.allclose(a[k], b[k], rtol=rtol, atol=atol, equal_nan=True), f"{os.path.basename(f_old)}: {k}"
        else:
            assert np.array_equal(a[k], b[k]), f"{os.path.basename(f_old)}: {k}"


# ---------------------------------------------------------------- project A
def test_project_A_model():
    run(project("project_A"), """
        import numpy as np, model, hyperpriors, burgers
        d = model.load_deconvolution()
        n = d["A"].shape[1]
        rng = np.random.default_rng(0)
        for name, hp in hyperpriors.DEFAULTS.items():
            theta = hp.rvs(n - 1, rng)
            assert np.all(theta > 0), name
            tau = hp.tau_from_theta(theta)
            assert np.allclose(hp.theta_from_tau(tau), theta, rtol=1e-6), name
            w = np.r_[0.0, np.sqrt(theta) * rng.standard_normal(n - 1)]
            assert np.isfinite(model.log_joint(w, theta, d["y"], d["G"], d["sigma"], hp)), name
            v = rng.standard_normal(2 * n - 1)
            assert np.isfinite(model.normalized_log_likelihood(v, d["y"], d["G"], d["sigma"], hp)), name
        b = model.load_burgers()
        assert b["G"](b["x_true"]).shape == b["y"].shape
    """)


def test_project_A_generate_data(tmp_path):
    p = copy_project("project_A", tmp_path)
    run(p, args=["generate_data.py"])
    for f in ("deconvolution.npz", "burgers.npz"):
        assert_same_npz(p / "data" / f, os.path.join(project("project_A"), "data", f))


# ---------------------------------------------------------------- project B
def test_project_B_model():
    run(project("project_B"), """
        import numpy as np, acrnn
        d = np.load("data/trimerization_20exp_5pct.npz")
        B, key, dep = acrnn.atom_conservation_basis(d["molecular_matrix"])
        assert np.allclose(B, d["B"]) and np.allclose(d["molecular_matrix"] @ B, 0.0)
        c = acrnn.simulate(d["params_true"], d["c0"][0], d["t_eval"], B, 5, 4)
        assert np.allclose(c, d["c_clean"][0], rtol=1e-5, atol=1e-8)
    """)


def test_project_B_generate_data(tmp_path):
    p = copy_project("project_B", tmp_path)
    run(p, args=["generate_data.py"])
    for f in ("trimerization_20exp_5pct.npz", "trimerization_5exp_5pct.npz", "trimerization_20exp_20pct.npz"):
        assert_same_npz(p / "data" / f, os.path.join(project("project_B"), "data", f), rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------- project C
def test_project_C_steady_and_priors():
    run(project("project_C"), """
        import numpy as np, swe, priors
        d = np.load("data/steady_smooth.npz")
        x, dx = swe.make_grid(200)
        eta = swe.surface_at_sensors(d["b_true_200"], x, d["x_obs"])
        assert np.sqrt(np.mean((eta - d["eta_obs"]) ** 2)) < 2 * float(d["noise_std"])
        gg = priors.GeneralizedGamma(*priors.GSBL_HYPERPRIOR)
        rng = np.random.default_rng(0)
        z, theta = priors.map_inverse(rng.standard_normal(10), rng.standard_normal(10), gg)
        assert priors.ias_theta_update(z, gg).shape == (10,)
        assert np.allclose(swe.bottom_to_increments(swe.increments_to_bottom(z, "left")), z)
        assert priors.baseline_proposal_chol(64).shape == (64, 64)
    """)


def test_project_C_flume_numpy_vs_jax():
    run(project("project_C"), """
        import numpy as np, flume
        N = 64                                  # coarse grid: one solver step per sample, 1000 steps
        x, dx = flume.cells(N)
        b = flume.bottom_hill(x)
        dt, k, n_samples = flume.time_grid(N)
        eta_left = flume.eta_left_synthetic(dt * np.arange(k * n_samples + 1))
        rec = flume.forward(b, eta_left, N)
        rec_jax = np.asarray(flume.forward_jax(N)(b, eta_left))
        assert rec.shape == (n_samples, 3) and np.allclose(rec, rec_jax, atol=1e-10)
        # (jax.grad of forward_jax is not tested here: compiling it takes about 20 s)
    """)


def test_project_C_generate_steady_data(tmp_path):
    p = copy_project("project_C", tmp_path)
    run(p, args=["generate_data.py", "--no-flume"])
    for f in ("steady_smooth.npz", "steady_piecewise.npz", "steady_smooth_sparse.npz", "steady_piecewise_sparse.npz"):
        assert_same_npz(p / "data" / f, os.path.join(project("project_C"), "data", f))

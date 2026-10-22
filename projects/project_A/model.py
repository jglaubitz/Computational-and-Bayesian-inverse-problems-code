"""
Project A -- the hierarchical model on increments for a linear or nonlinear forward map (NumPy).
No inference code: that is your job (start from the course code, see the README).

    y = G(x) + e,  e ~ N(0, sigma^2 I),      x = c 1 + S z   (x_1 = c, x_{j+1} - x_j = z_j),
    c ~ N(0, gamma0^2),  z_i | theta_i ~ N(0, theta_i),  theta_i ~ p(theta),  i = 1, ..., n - 1,

as in samplers.gibbs_hierarchical and approximations.cavi_hierarchical of the course code (level c with its own
Gaussian prior, hierarchical prior on the n - 1 increments). w = (c, z) collects the Gaussian variables, x = T w.

    load_deconvolution(), load_burgers()   the data sets of generate_data.py
    increment_maps(n)                      S (n x n-1) and T = [1, S] (n x n)
    log_likelihood(x, y, G, sigma)         with its normalizing constant
    log_joint(w, theta, y, G, sigma, hp)   log p(y, c, z, theta), all normalizing constants included
    log_joint_logtheta(w, zeta, ...)       the same in (c, z, zeta = log theta), with the Jacobian
    normalized_log_likelihood(v, ...)      log-likelihood in prior-normalized coordinates v = (c, w, tau), whose prior
                                           is N(0, I) (Glaubitz & Marzouk): the target of elliptical slice sampling
    normalized_to_x(v, hp)                 v = (c, w, tau) -> (x, theta)
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG2PI = np.log(2.0 * np.pi)


def load_deconvolution(name="deconvolution.npz"):
    d = np.load(os.path.join(HERE, "data", name))
    A = d["A"]
    return {"A": A, "G": (lambda x: A @ x), "y": d["y"], "x_true": d["x_true"], "sigma": float(d["sigma"]), "t": d["t"]}


def load_burgers(name="burgers.npz"):
    import burgers
    d = np.load(os.path.join(HERE, "data", name))
    obs = d["obs_cells"]
    return {"G": (lambda u0: burgers.forward(u0, obs)), "y": d["y"], "x_true": d["u0_true"], "sigma": float(d["sigma"]),
            "t": d["x"], "obs_cells": obs}


def increment_maps(n):
    S = np.tril(np.ones((n, n)), -1)[:, :-1]
    return S, np.hstack([np.ones((n, 1)), S])


def log_likelihood(x, y, G, sigma):
    r = G(x) - y
    return -0.5 * np.dot(r, r) / sigma**2 - 0.5 * len(y) * (LOG2PI + 2.0 * np.log(sigma))


def log_joint(w, theta, y, G, sigma, hp, gamma0=1.0):
    """log p(y, c, z, theta) for w = (c, z), theta > 0, hyper-prior hp (see hyperpriors.py)."""
    theta = np.asarray(theta, dtype=float)
    if np.any(theta <= 0):
        return -np.inf
    n = len(w)
    _, T = increment_maps(n)
    c, z = w[0], w[1:]
    lc = -0.5 * (LOG2PI + 2.0 * np.log(gamma0)) - 0.5 * c**2 / gamma0**2
    lz = np.sum(-0.5 * (LOG2PI + np.log(theta)) - 0.5 * z**2 / theta)
    return log_likelihood(T @ w, y, G, sigma) + lc + lz + np.sum(hp.logpdf(theta))


def log_joint_logtheta(w, zeta, y, G, sigma, hp, gamma0=1.0):
    """log density in (c, z, zeta) with zeta = log theta: log_joint + sum(zeta) (Jacobian d theta / d zeta = theta)."""
    return log_joint(w, np.exp(zeta), y, G, sigma, hp, gamma0) + np.sum(zeta)


def normalized_to_x(v, hp, gamma0=1.0):
    """Prior-normalized coordinates v = (c', w_1..w_{n-1}, tau_1..tau_{n-1}), all N(0, 1) a priori:
    c = gamma0 c', theta_i = P^{-1}(Phi(tau_i)), z_i = sqrt(theta_i) w_i. Returns (x, theta)."""
    v = np.asarray(v, dtype=float)
    k = (len(v) - 1) // 2
    theta = hp.theta_from_tau(v[1 + k:])
    z = v[1:1 + k] * np.sqrt(theta)
    x = gamma0 * v[0] + np.concatenate([[0.0], np.cumsum(z)])
    return x, theta


def normalized_log_likelihood(v, y, G, sigma, hp, gamma0=1.0):
    """log L(v) = log p(y | x(v)); the prior of v is N(0, I), so p(v | y) ∝ L(v) N(v; 0, I)."""
    x, theta = normalized_to_x(v, hp, gamma0)
    if not np.all(np.isfinite(theta)) or np.any(theta <= 0):
        return -np.inf
    return log_likelihood(x, y, G, sigma)

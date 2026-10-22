"""Figures of Chapter 5 (beyond MCMC: approximate Bayesian inference). Output: ../figures/ch05/*.pdf

    python ch05_figures.py                  all figures (about three minutes)
    python ch05_figures.py kl viridge       selected figures only

Figures: laplaceidea, laplacefail, laplacebathy, kl, viridge, vimf, vbhier, enkidea, enkbathy; vibathy prints the numbers of
Gaussian variational inference on the bathymetry problem (no figure). The opening cartoon is figures/ch05/landscape.tikz,
written by ch05_landscape.py.
"""
import os
import sys
import time
import functools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
from scipy.stats import norm

from running_examples import (deconvolution_operator, signal_smooth, signal_piecewise, make_grid, bottom_smooth,
                              N_CELLS, M_OBS, NOISE_STD, L as LEN)
from solvers import difference_matrix, gaussian_posterior, gauss_newton, ias
from priors import gaussian_precision
from samplers import perturb_optimize, gibbs_laplace, gibbs_hierarchical, make_hmc_jax, ess, iat
from approximations import (lowrank_laplace, lowrank_std, gaussian_vi, pareto_khat, importance_estimate,
                            cavi_hierarchical, ensemble_kalman_step, es_mda, eki)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "ch05")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "lines.linewidth": 1.1})
W = 6.3
C = {"truth": "k", "rec": "tab:red", "a": "tab:blue", "b": "tab:orange", "c": "tab:green", "d": "tab:purple"}
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(60)          # Gauss-Hermite rule for expectations under N(0, 1)
GH_W = GH_W / GH_W.sum()


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), **kw)
    plt.close(fig)
    print("wrote", name)


def ellipse(ax, mean, cov, r, **kw):
    """Contour {x : (x - mean)^T cov^{-1} (x - mean) = r^2} of a Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mean, 2 * r * np.sqrt(vals[1]), 2 * r * np.sqrt(vals[0]), angle=ang, fill=False, **kw))


# ====================================================================== the bent ridge of Example 4.3
Y_B, SIG_B = 1.0, 0.3


def U_banana(x):
    r = Y_B - x[..., 0] - 0.5 * x[..., 1] ** 2
    return 0.5 * (x[..., 0] ** 2 + x[..., 1] ** 2) + r**2 / (2 * SIG_B**2)


GB = np.linspace(-4, 4, 1201)
GX1, GX2 = np.meshgrid(GB, GB)
UB = U_banana(np.stack([GX1, GX2], -1))
PB = np.exp(-(UB - UB.min()))
LOGZ_B = np.log(PB.sum() * (GB[1] - GB[0]) ** 2) - UB.min()     # log of the normalizing constant of exp(-U)
PB /= PB.sum()
E_X1 = float((GX1 * PB).sum())
SD_X1 = float(np.sqrt(((GX1 - E_X1) ** 2 * PB).sum()))
SD_X2 = float(np.sqrt((GX2**2 * PB).sum()))
X_MAP_B = np.array([Y_B / (1 + SIG_B**2), 0.0])
H_B = np.diag([1 + 1 / SIG_B**2, 1 - (Y_B - X_MAP_B[0]) / SIG_B**2])   # exact Hessian of U at the MAP estimate
H_GN_B = np.diag([1 + 1 / SIG_B**2, 1.0])                          # Gauss-Newton Hessian: J = (1, x2) = (1, 0) there
LEV = np.exp(-np.array([3.0, 1.5, 0.4]))                        # contour levels relative to the maximum
RAD = np.sqrt(2 * np.array([3.0, 1.5, 0.4]))                   # the same levels for a Gaussian: Mahalanobis radii


def banana_contours(ax, **kw):
    ax.contour(GX1, GX2, PB / PB.max(), levels=sorted(LEV), colors=kw.pop("colors", "0.65"), linewidths=0.7, **kw)


def elbo_banana(m, Cf):
    """ELBO of N(m, Cf Cf^T) for the bent ridge, by tensor Gauss-Hermite quadrature."""
    E1, E2 = np.meshgrid(GH_X, GH_X, indexing="ij")
    X = m + np.stack([E1, E2], -1) @ Cf.T
    return -(np.outer(GH_W, GH_W) * U_banana(X)).sum() + np.log(abs(np.linalg.det(Cf))) + (1 + np.log(2 * np.pi))


# ====================================================================== deconvolution data (as in Chapters 2 to 4)
n, m, sigma = N_CELLS, M_OBS, NOISE_STD
A, t, s = deconvolution_operator(n, m)
rng_data = np.random.default_rng(1)
x_s, x_p = signal_smooth(t), signal_piecewise(t)
y_s = A @ x_s + sigma * rng_data.standard_normal(m)
y_p = A @ x_p + sigma * rng_data.standard_normal(m)
Ln = difference_matrix(n)
jumps = np.nonzero(Ln @ x_p)[0]
GAMMA_SMOOTH = 0.0294
GAMMA_PRIOR, BETA, VARTHETA = 0.1, 1.0, 1e-3


# ====================================================================== bathymetry with the prior of Example 4.19(b)
SIG_W, GAMMA_W = 0.01, 0.01                                     # 1 cm noise, increments of 1 cm per cell


@functools.lru_cache(None)
def bathy(MM=20):
    """Data, forward map, potential, MAP estimate, and Gauss-Newton Hessian of the bathymetry problem with MM sensors."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import swe_jax
    xg, dx = make_grid(200)
    b_true = bottom_smooth(xg)
    xo = np.linspace(0.5, LEN - 0.5, MM)
    fwd = swe_jax.make_forward(xg, xo)
    y = np.asarray(fwd(jnp.asarray(b_true[:-1]))) + SIG_W * np.random.default_rng(4).standard_normal(MM)
    yj = jnp.asarray(y)
    jac = jax.jit(jax.jacrev(fwd))
    count = {"F": 0, "J": 0}

    def F(b):
        count["F"] += 1
        return np.asarray(fwd(jnp.asarray(b)))

    def Jf(b):
        count["J"] += 1
        return np.asarray(jac(jnp.asarray(b)))
    L1 = difference_matrix(200)[:, :-1]                         # increments of (b, 0): the outflow bottom is known
    Lw = L1 / GAMMA_W
    R = np.linalg.inv(Lw)                                       # prior N(0, R R^T): b = R w with w ~ N(0, I)
    Lwj = jnp.asarray(Lw)

    def U(b):
        r = fwd(b) - yj
        z = Lwj @ b
        return 0.5 * r @ r / SIG_W**2 + 0.5 * z @ z
    b_map, hist = gauss_newton(F, Jf, y, np.zeros(199), lam=SIG_W**2, L=Lw, n_iter=60)
    cost_gn = dict(count)
    Jm = Jf(b_map)
    H = Jm.T @ Jm / SIG_W**2 + Lw.T @ Lw
    return dict(xg=xg, b_true=b_true, xo=xo, y=y, fwd=fwd, F=F, Jf=Jf, U=U, Lw=Lw, R=R, b_map=b_map,
                n_gn=len(hist) - 1, cost_gn=cost_gn, J=Jm, H=H, count=count)


@functools.lru_cache(None)
def bathy_hmc():
    """HMC reference for the bathymetry posterior with 20 sensors (mass matrix: the Gauss-Newton Hessian)."""
    import jax
    import jax.numpy as jnp
    d = bathy(20)
    Cm = np.linalg.cholesky(np.linalg.inv(d["H"]))
    step = make_hmc_jax(d["U"], 0.3, 5, Cm)
    key = jax.random.PRNGKey(0)
    xb = jnp.asarray(d["b_map"])
    xs, acc = [], 0
    t0 = time.time()
    for k in range(5500):
        key, sub = jax.random.split(key)
        xb, a, dH = step(sub, xb)
        acc += bool(a)
        xs.append(np.asarray(xb))
    X = np.array(xs)[500:]
    ess_min = min(ess(X[:, i]) for i in range(199))
    print(f"bathymetry HMC reference: acceptance {acc / 5500:.2f}, min ESS {ess_min:.0f} of {len(X)}, "
          f"{len(X) * 5 / ess_min:.1f} gradients per effective sample, {time.time() - t0:.0f}s")
    return X


def fd_jacobian(F, x, eps=1e-6):
    """Jacobian by one-sided finite differences: n + 1 forward solves."""
    f0 = F(x)
    Jm = np.empty((len(f0), len(x)))
    for j in range(len(x)):
        e = np.zeros_like(x); e[j] = eps
        Jm[:, j] = (F(x + e) - f0) / eps
    return Jm


def prior_ensemble(d, J, rng):
    """J draws from the prior for which the forward model can be evaluated; returns them and the number of draws."""
    X, tries = [], 0
    while len(X) < J:
        b = d["R"] @ rng.standard_normal(199)
        tries += 1
        if np.all(np.isfinite(d["F"](b))):
            X.append(b)
    return np.array(X), tries


# ====================================================================== the Laplace approximation in one dimension
def fig_laplaceidea():
    s_u = 0.6                                                   # u = log(kappa) ~ N(0, 0.6^2), Remark 3.16
    k_map = np.exp(-s_u**2)
    sd_l = s_u * k_map                                          # U''(kappa_map) = 1 / (s_u kappa_map)^2
    k = np.linspace(1e-3, 3.6, 2000)
    Uk = np.log(k) + np.log(k) ** 2 / (2 * s_u**2)
    Uk -= Uk.min()
    P_neg = norm.cdf(-k_map / sd_l)
    print(f"Laplace idea: kappa MAP {k_map:.3f}, Laplace std {sd_l:.3f}, mass of the Laplace approximation below 0: {P_neg:.3f}; "
          f"posterior mean {np.exp(s_u**2 / 2):.3f}, std {np.sqrt((np.exp(s_u**2) - 1) * np.exp(s_u**2)):.3f}")
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.95), constrained_layout=True)
    ax = axs[0]
    kk = np.linspace(-0.6, 3.6, 400)
    ax.plot(k, Uk, color="k", label="$U(\\kappa)$")
    ax.plot(kk, 0.5 * (kk - k_map) ** 2 / sd_l**2, "--", color=C["b"], label="quadratic at the MAP")
    ax.plot(k_map, 0, "*", color="k", ms=7)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 6); ax.set_xlabel("$\\kappa$")
    ax.set_title("(a) potential of $\\kappa$"); ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax = axs[1]
    u = np.linspace(-2.4, 1.6, 400)
    ax.plot(u, u**2 / (2 * s_u**2), color="k", label="$U(u)$")
    ax.plot(u, u**2 / (2 * s_u**2), "--", color=C["b"], dashes=(3, 3), label="quadratic at the MAP")
    ax.plot(0, 0, "*", color="k", ms=7)
    ax.set_xlim(-2.4, 1.6); ax.set_ylim(0, 6); ax.set_xlabel("$u = \\log\\kappa$")
    ax.set_title("(b) potential of $u = \\log\\kappa$"); ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax = axs[2]
    kp = np.linspace(-0.9, 3.6, 1000)
    pk = np.where(kp > 0, norm.pdf(np.log(np.maximum(kp, 1e-12)), 0, s_u) / np.maximum(kp, 1e-12), 0.0)
    ql = norm.pdf(kp, k_map, sd_l)
    ax.fill_between(kp, 0, pk, color=C["a"], alpha=0.25, lw=0, label="posterior")
    ax.plot(kp, ql, color=C["b"], label="Laplace in $\\kappa$")
    ax.fill_between(kp, 0, ql, where=kp < 0, color=C["rec"], alpha=0.5, lw=0)
    ax.plot(kp, pk, "--", color="k", lw=0.9, label="Laplace in $u$, mapped back")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlim(-0.9, 3.6); ax.set_ylim(0, 1.5); ax.set_xlabel("$\\kappa$")
    ax.set_title("(c) densities of $\\kappa$"); ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    save(fig, "laplaceidea.pdf")


def fig_laplacefail():
    sd_l = 1 / np.sqrt(np.diag(H_B))
    sd_gn = 1 / np.sqrt(np.diag(H_GN_B))
    print(f"bent ridge: MAP {X_MAP_B.round(4)}, Hessian diag {np.diag(H_B).round(4)}, Laplace std {sd_l.round(3)}; "
          f"Gauss-Newton Hessian diag {np.diag(H_GN_B).round(4)}, std {sd_gn.round(3)}; "
          f"posterior mean ({E_X1:.3f}, 0), std ({SD_X1:.3f}, {SD_X2:.3f}), log Z {LOGZ_B:.3f}; "
          f"conditional std of x1 given x2 {1 / np.sqrt(1 + 1 / SIG_B**2):.3f}")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.05), constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]})
    ax = axs[0]
    banana_contours(ax)
    for r in RAD:
        ellipse(ax, X_MAP_B, np.diag(sd_l**2), r, color=C["b"], lw=0.9, ls="--")
        ellipse(ax, X_MAP_B, np.diag(sd_gn**2), r, color=C["c"], lw=1.1, ls=":")
    ax.plot([], [], "--", color=C["b"], lw=0.9, label="Laplace")
    ax.plot([], [], ":", color=C["c"], lw=1.1, label="Laplace, GN")
    ax.plot(*X_MAP_B, "*", color="k", ms=6, label="MAP")
    ax.plot(E_X1, 0, "o", color=C["rec"], ms=3.5, label="posterior mean")
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-3.2, 3.2); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("(a) the bent ridge"); ax.legend(frameon=False, fontsize=5.8, loc="center left", handletextpad=0.3, handlelength=1.5)
    ax = axs[1]
    x = np.linspace(-2.0, 2.2, 3000)
    U2 = (1 - x**2) ** 2 / (2 * 0.09) + (x - 0.4) ** 2 / 2
    p2 = np.exp(-(U2 - U2.min())); p2 /= np.trapezoid(p2, x)
    x_map = x[np.argmin(U2)]
    h = 1e-4
    U2f = lambda z: (1 - z**2) ** 2 / (2 * 0.09) + (z - 0.4) ** 2 / 2
    x_map = minimize(lambda z: U2f(z[0]), [1.0]).x[0]
    curv = (U2f(x_map + h) - 2 * U2f(x_map) + U2f(x_map - h)) / h**2
    mass_left = np.trapezoid(p2 * (x < 0), x)
    print(f"two peaks: MAP {x_map:.4f}, Laplace std {1 / np.sqrt(curv):.3f}, posterior mass of the other peak {mass_left:.3f}, "
          f"posterior mean {np.trapezoid(x * p2, x):.3f}")
    ax.fill_between(x, 0, p2, color=C["a"], alpha=0.25, lw=0, label="posterior")
    ax.plot(x, norm.pdf(x, x_map, 1 / np.sqrt(curv)), color=C["b"], label="Laplace")
    ax.text(-0.97, 0.75, f"{100 * mass_left:.0f}%\nof the mass", ha="center", fontsize=6.5)
    ax.set_xlim(-2.0, 2.2); ax.set_ylim(0, 2.9); ax.set_xlabel("$x$")
    ax.set_title("(b) two peaks"); ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax = axs[2]
    x = np.linspace(-3.0, 3.5, 3000)
    Uk = (x - 0.8) ** 2 / 2 + np.abs(x)
    pk = np.exp(-(Uk - Uk.min())); pk /= np.trapezoid(pk, x)
    print(f"kink: posterior mean {np.trapezoid(x * pk, x):.3f}, std {np.sqrt(np.trapezoid((x - np.trapezoid(x * pk, x)) ** 2 * pk, x)):.3f}, "
          f"P(x < 0) {np.trapezoid(pk * (x < 0), x):.3f}")
    ax.fill_between(x, 0, pk, color=C["a"], alpha=0.25, lw=0, label="posterior")
    ax.plot(x, norm.pdf(x, 0, 1), "--", color=C["b"], label="Gaussian with the\ncurvature beside the kink")
    ax.plot(0, pk.max(), "*", color="k", ms=6)
    ax.set_xlim(-3.0, 3.5); ax.set_ylim(0, 0.95); ax.set_xlabel("$x$")
    ax.set_title("(c) a kink at the MAP estimate"); ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    save(fig, "laplacefail.pdf")


# ====================================================================== the Laplace approximation of the bathymetry posterior
def fig_laplacebathy():
    d20, d200 = bathy(20), bathy(200)
    lam20, V20 = lowrank_laplace(d20["J"], d20["R"], SIG_W)
    lam200, _ = lowrank_laplace(d200["J"], d200["R"], SIG_W)
    for name, lam in (("20 sensors", lam20), ("200 sensors", lam200)):
        print(f"low rank, {name}: {np.sum(lam > 1)} eigenvalues above 1, {np.sum(lam > 0.1)} above 0.1, largest {lam[:3].round(1)}")
    Xh = bathy_hmc()
    mh, sd_h = Xh.mean(0), Xh.std(0)
    lo_h, hi_h = np.percentile(Xh, [2.5, 97.5], axis=0)
    ess_h = np.array([ess(Xh[:, i]) for i in range(199)])
    xg, bt = d20["xg"][:-1], d20["b_true"][:-1]
    sd_prior = np.sqrt(np.sum(d20["R"] ** 2, axis=1))
    sd_full = np.sqrt(np.diag(np.linalg.inv(d20["H"])))
    c = d20["cost_gn"]
    cover = lambda lo, hi: 100 * np.mean((bt >= lo) & (bt <= hi))
    print(f"   Gauss-Newton: {d20['n_gn']} iterations, {c['J']} Jacobians and {c['F']} forward solves; Laplace/HMC std "
          f"{np.min(sd_full / sd_h):.3f}-{np.max(sd_full / sd_h):.3f}; max |HMC mean - MAP| / HMC std {np.max(np.abs(mh - d20['b_map']) / sd_h):.2f}; "
          f"posterior std {100 * sd_h.min():.2f}-{100 * sd_h.max():.2f} cm, prior std {100 * sd_prior.min():.2f}-{100 * sd_prior.max():.1f} cm; "
          f"HMC ESS {ess_h.min():.0f}-{ess_h.max():.0f}; truth inside the band: Laplace {cover(d20['b_map'] - 1.96 * sd_full, d20['b_map'] + 1.96 * sd_full):.0f}%, "
          f"HMC {cover(lo_h, hi_h):.0f}%")
    for r in (2, 5, 8, 20):
        sd_r = lowrank_std(lam20, V20, d20["R"], r)
        print(f"   rank {r}: largest relative error of the std against the full Laplace approximation {np.max(np.abs(sd_r / sd_full - 1)):.3f}")
    import jax
    import jax.numpy as jnp
    Ub = jax.jit(jax.vmap(d20["U"]))
    Cl = np.linalg.cholesky(np.linalg.inv(d20["H"]))
    rng = np.random.default_rng(13)
    for rep in range(3):
        Xi = rng.standard_normal((4000, 199))
        Xs = d20["b_map"] + Xi @ Cl.T
        lw = -np.asarray(Ub(jnp.asarray(Xs))) + 0.5 * np.sum(Xi**2, axis=1)
        e, ess_is = importance_estimate(np.c_[Xs, Xs**2], lw)
        sd_is = np.sqrt(np.maximum(e[199:] - e[:199] ** 2, 0.0))
        print(f"   importance sampling with the Laplace approximation (4000 draws): k-hat {pareto_khat(lw):.2f}, ESS {ess_is:.0f}; "
              f"corrected std / HMC std {np.min(sd_is / sd_h):.3f}-{np.max(sd_is / sd_h):.3f}")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.05), constrained_layout=True)
    ax = axs[0]
    ax.fill_between(xg, lo_h, hi_h, color=C["a"], alpha=0.3, lw=0, label="HMC, 95% band")
    ax.plot(xg, d20["b_map"] - 1.96 * sd_full, color=C["b"], lw=0.8, label="Laplace, 95% band")
    ax.plot(xg, d20["b_map"] + 1.96 * sd_full, color=C["b"], lw=0.8)
    ax.plot(d20["xg"], d20["b_true"], color="k", lw=0.8, label="truth")
    ax.plot(d20["xo"], np.full(20, -0.105), "|", color="k", ms=3)
    ax.set_ylim(-0.12, 0.34); ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$b$ [m]")
    ax.set_title("(a) bands"); ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    ax = axs[1]
    ax.plot(xg, 100 * sd_prior, color="0.6", lw=0.9, label="prior")
    ax.plot(xg, 100 * sd_full, color=C["b"], lw=1.0, label="Laplace")
    ax.plot(xg[::6], 100 * sd_h[::6], "o", ms=2.2, color=C["a"], label="HMC")
    ax.plot(d20["xo"], np.full(20, 0.62), "|", color="k", ms=3)
    ax.set_yscale("log"); ax.set_ylim(0.55, 90); ax.set_xlabel("$x$ [m]"); ax.set_ylabel("std [cm]")
    ax.set_title("(b) standard deviations"); ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    ax = axs[2]
    band = 1.96 / np.sqrt(2 * ess_h)                           # Monte Carlo error of the HMC standard deviations
    ax.fill_between(xg, 1 - band, 1 + band, color="0.85", lw=0, label="Monte Carlo error of HMC")
    ax.plot(xg, sd_full / sd_h, color=C["b"], lw=0.9, label="Laplace / HMC")
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.plot(d20["xo"], np.full(20, 0.915), "|", color="k", ms=3)
    ax.set_ylim(0.9, 1.1); ax.set_xlabel("$x$ [m]")
    ax.set_title("(c) std of Laplace / std of HMC"); ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    save(fig, "laplacebathy.pdf")


# ====================================================================== variational inference: the two directions of KL
def fig_kl():
    x = np.linspace(-2.2, 2.4, 4000)
    U2f = lambda z: (1 - z**2) ** 2 / (2 * 0.09) + (z - 0.4) ** 2 / 2
    U2 = U2f(x)
    p2 = np.exp(-(U2 - U2.min()))
    logZ = np.log(np.trapezoid(p2, x)) - U2.min()
    p2 /= np.trapezoid(p2, x)
    mean2 = np.trapezoid(x * p2, x)
    sd2 = np.sqrt(np.trapezoid((x - mean2) ** 2 * p2, x))

    def neg_elbo(par):
        mu, ls = par
        return (GH_W * U2f(mu + np.exp(ls) * GH_X)).sum() - ls - 0.5 * (1 + np.log(2 * np.pi))
    opts = []
    for start in (1.0, -1.0):
        r = minimize(neg_elbo, [start, np.log(0.2)], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
        opts.append((r.x[0], np.exp(r.x[1]), logZ + r.fun))
        print(f"KL directions: reverse KL from {start:+.0f}: mean {r.x[0]:.3f}, std {np.exp(r.x[1]):.3f}, KL(q||p) = {logZ + r.fun:.3f}")
    print(f"   forward KL (moment matching): mean {mean2:.3f}, std {sd2:.3f}")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.15), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axs[0]
    ax.fill_between(x, 0, p2, color=C["a"], alpha=0.25, lw=0, label="posterior")
    (m1, s1, k1), (m2_, s2_, k2) = opts
    ax.plot(x, norm.pdf(x, m1, s1), color=C["a"], label="$\\min_q \\mathrm{KL}(q \\,\\|\\, p)$")
    ax.plot(x, norm.pdf(x, m2_, s2_), ":", color=C["a"], lw=1.2, label="a second local minimum")
    ax.plot(x, norm.pdf(x, mean2, sd2), "--", color=C["b"], label="$\\min_q \\mathrm{KL}(p \\,\\|\\, q)$")
    ax.text(m1 + 0.22, 2.0, f"KL $= {k1:.2f}$", fontsize=6.5, color=C["a"])
    ax.text(m2_ + 0.22, 2.0, f"KL $= {k2:.2f}$", fontsize=6.5, color=C["a"])
    ax.set_xlim(-2.2, 2.4); ax.set_ylim(0, 4.3); ax.set_xlabel("$x$")
    ax.set_title("(a) two peaks: Gaussian approximations"); ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax = axs[1]
    rho = 0.9
    Sig = np.array([[1.0, rho], [rho, 1.0]])
    for r in (1.0, 2.0):
        ellipse(ax, (0, 0), Sig, r, color="0.55", lw=0.9)
        ellipse(ax, (0, 0), (1 - rho**2) * np.eye(2), r, color=C["a"], lw=1.1)
        ellipse(ax, (0, 0), np.eye(2), r, color=C["b"], lw=1.0, ls="--")
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.7, 2.7); ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"(b) correlation {rho}: mean field")
    save(fig, "kl.pdf")


# ====================================================================== Gaussian VI on the bent ridge
def fig_viridge():
    import jax.numpy as jnp

    def U_j(x):
        r = Y_B - x[0] - 0.5 * x[1] ** 2
        return 0.5 * (x[0] ** 2 + x[1] ** 2) + r**2 / (2 * SIG_B**2)

    def neg(p):
        mm, Cf = p[:2], np.array([[np.exp(p[2]), 0.0], [p[4], np.exp(p[3])]])
        return -elbo_banana(mm, Cf)
    best = minimize(neg, [0.9, 0.2, np.log(0.3), np.log(1.0), 0.1], method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000})
    m_opt = best.x[:2]
    C_opt = np.array([[np.exp(best.x[2]), 0.0], [best.x[4], np.exp(best.x[3])]])
    S_opt = C_opt @ C_opt.T
    print(f"Gaussian VI optimum on the bent ridge (quadrature): mean {m_opt.round(3)}, std {np.sqrt(np.diag(S_opt)).round(3)}, "
          f"ELBO {-best.fun:.4f}, log Z {LOGZ_B:.4f}, KL {LOGZ_B + best.fun:.4f}")
    t0 = time.time()
    mv, Cv, rec = gaussian_vi(U_j, np.zeros(2), family="full", n_iter=6000, n_samples=16, lr=1e-2, lr_final=1e-3, seed=2,
                              record=20, keep_params=True)
    Sv = Cv @ Cv.T
    elbos = np.array([elbo_banana(mk, Ck) for mk, Ck in zip(rec["m"], rec["C"])])
    print(f"   ADVI: {time.time() - t0:.1f}s, {rec['gradients']} gradients; mean {mv.round(3)}, std {np.sqrt(np.diag(Sv)).round(3)}, "
          f"correlation {Sv[0, 1] / np.sqrt(Sv[0, 0] * Sv[1, 1]):.3f}; exact ELBO of the result {elbos[-1]:.4f}; "
          f"ELBO within 0.01 of the optimum after {rec['iter'][np.argmax(elbos > -best.fun - 0.01)]} iterations")
    sd_l = 1 / np.sqrt(np.diag(H_B))
    sd_gn = 1 / np.sqrt(np.diag(H_GN_B))
    elbo_lap = elbo_banana(X_MAP_B, np.diag(sd_l))
    elbo_gn = elbo_banana(X_MAP_B, np.diag(sd_gn))
    print(f"   ELBO of the Laplace approximation {elbo_lap:.1f} (KL {LOGZ_B - elbo_lap:.1f}); with the Gauss-Newton Hessian "
          f"{elbo_gn:.3f} (KL {LOGZ_B - elbo_gn:.2f}); ADVI result: KL {LOGZ_B - elbos[-1]:.3f}")
    rng = np.random.default_rng(3)
    for name, mu, Cq in (("Laplace", X_MAP_B, np.diag(sd_l)), ("Gauss-Newton Laplace", X_MAP_B, np.diag(sd_gn)), ("VI", mv, Cv)):
        ks, est = [], []
        for rep in range(5):
            X = mu + rng.standard_normal((4000, 2)) @ Cq.T
            lq = -0.5 * np.sum(np.linalg.solve(Cq, (X - mu).T) ** 2, axis=0) - np.log(abs(np.linalg.det(Cq)))
            lw = -U_banana(X) - lq
            ks.append(pareto_khat(lw))
            e, _ = importance_estimate(np.c_[X, X**2], lw)
            est.append([e[0], np.sqrt(e[3] - e[1] ** 2)])
        est = np.array(est)
        print(f"   importance sampling with the {name} approximation, 5 x 4000 draws: k-hat {np.round(ks, 2).tolist()}; "
              f"corrected mean of x1 {est[:, 0].round(2).tolist()}, std of x2 {est[:, 1].round(2).tolist()}")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.05), constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.2, 1.2]})
    ax = axs[0]
    banana_contours(ax)
    for r in RAD:
        ellipse(ax, X_MAP_B, np.diag(sd_l**2), r, color=C["b"], lw=0.8, ls="--")
        ellipse(ax, X_MAP_B, np.diag(sd_gn**2), r, color=C["c"], lw=1.1, ls=":")
        ellipse(ax, mv, Sv, r, color=C["a"], lw=1.0)
    ax.plot([], [], "--", color=C["b"], label="Laplace")
    ax.plot([], [], ":", color=C["c"], lw=1.1, label="Laplace, GN")
    ax.plot([], [], color=C["a"], label="Gaussian VI")
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-3.2, 3.2); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("(a) three Gaussians"); ax.legend(frameon=False, fontsize=5.8, loc="center left")
    ax = axs[1]
    ax.semilogx(rec["iter"][1:] + 1, rec["elbo"][1:], color=C["a"], alpha=0.18, lw=0.6, label="estimate ($S = 16$)")
    ax.semilogx(rec["iter"] + 1, elbos, color=C["a"], lw=1.1, label="exact")
    ax.axhline(LOGZ_B, color="k", ls="--", lw=0.8, label="$\\log p(\\mathbf{y})$ + const")
    ax.axhline(-best.fun, color="0.5", ls=":", lw=0.8, label="best Gaussian")
    ax.set_ylim(-1.6, 0.6); ax.set_xlabel("iteration"); ax.set_ylabel("ELBO")
    ax.set_title("(b) ELBO of the iterates"); ax.legend(frameon=False, fontsize=5.6, loc="lower right")
    ax = axs[2]
    marg = PB.sum(axis=1) / (GB[1] - GB[0])
    ax.fill_between(GB, 0, marg, color=C["a"], alpha=0.25, lw=0, label="posterior")
    ax.plot(GB, norm.pdf(GB, mv[1], np.sqrt(Sv[1, 1])), color=C["a"], label="Gaussian VI")
    ax.plot(GB, norm.pdf(GB, 0, sd_l[1]), "--", color=C["b"], label="Laplace")
    ax.plot(GB, norm.pdf(GB, 0, sd_gn[1]), ":", color=C["c"], lw=1.3, label="Laplace, Gauss-Newton")
    ax.set_xlim(-4, 4); ax.set_ylim(0, 0.95); ax.set_xlabel("$x_2$")
    ax.set_title("(c) marginal density of $x_2$"); ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    save(fig, "viridge.pdf")


# ====================================================================== mean-field versus full-rank VI for deconvolution
def fig_vimf():
    import jax.numpy as jnp
    P_s = gaussian_precision(n, GAMMA_SMOOTH, "differences", gamma0=1.0)
    mean_s, cov_s = gaussian_posterior(A, y_s, sigma**2, P_s)
    sd = np.sqrt(np.diag(cov_s))
    H = A.T @ A / sigma**2 + P_s
    b = A.T @ y_s / sigma**2
    sd_mf = 1 / np.sqrt(np.diag(H))
    logZ = 0.5 * mean_s @ H @ mean_s + 0.5 * n * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(H)[1]   # U = x^T H x / 2 - b^T x

    def elbo(mq, Cq):
        Sq = Cq @ Cq.T
        return -(0.5 * np.sum(H * Sq) + 0.5 * mq @ H @ mq - b @ mq) + np.sum(np.log(np.abs(np.diag(Cq)))) + 0.5 * n * (1 + np.log(2 * np.pi))
    Hj, bj = jnp.asarray(H), jnp.asarray(b)
    U = lambda x: 0.5 * x @ Hj @ x - bj @ x
    res = {}
    for fam, nit, lr in (("mean-field", 4000, 2e-2), ("full", 10000, 1e-2)):
        t0 = time.time()
        mq, Cq, rec = gaussian_vi(U, np.zeros(n), C0=0.1 * np.eye(n), family=fam, n_iter=nit, n_samples=8, lr=lr,
                                  lr_final=lr / 1000, seed=0, record=max(nit // 200, 1), keep_params=True)
        kl = np.array([logZ - elbo(mk, Ck) for mk, Ck in zip(rec["m"], rec["C"])])
        sq = np.sqrt(np.sum(Cq**2, axis=1))
        res[fam] = (mq, sq, rec["iter"], kl)
        print(f"VI on deconvolution, {fam}: {time.time() - t0:.1f}s, {rec['gradients']} gradients; largest error of the mean "
              f"{np.max(np.abs(mq - mean_s) / sd):.3f} posterior std; std / exact std {np.min(sq / sd):.3f}-{np.max(sq / sd):.3f}; "
              f"std / (1/sqrt(H_ii)) {np.min(sq / sd_mf):.3f}-{np.max(sq / sd_mf):.3f}; final KL {kl[-1]:.3f}")
    kl_mf = 0.5 * (np.sum(np.log(np.diag(H))) - np.linalg.slogdet(H)[1])
    Cp = np.linalg.cholesky(np.linalg.inv(P_s))
    for J in (100, 1000, 10000):
        X0 = np.random.default_rng(40 + J).standard_normal((J, n)) @ Cp.T
        X1 = ensemble_kalman_step(X0, X0 @ A.T, y_s, sigma, np.random.default_rng(50 + J))
        print(f"   one ensemble Kalman update, J = {J}: largest error of the mean {np.max(np.abs(X1.mean(0) - mean_s) / sd):.2f} posterior std; "
              f"std / exact std {np.min(X1.std(0, ddof=1) / sd):.2f}-{np.max(X1.std(0, ddof=1) / sd):.2f}")
    cover = lambda lo, hi: 100 * np.mean((x_s >= lo) & (x_s <= hi))
    print(f"   mean-field theory: std ratio {np.min(sd_mf / sd):.3f}-{np.max(sd_mf / sd):.3f} (median {np.median(sd_mf / sd):.3f}); "
          f"KL of the best mean-field Gaussian {kl_mf:.2f}; coverage exact band {cover(mean_s - 1.96 * sd, mean_s + 1.96 * sd):.0f}%, "
          f"mean-field band {cover(mean_s - 1.96 * sd_mf, mean_s + 1.96 * sd_mf):.0f}%; correlation of neighbors "
          f"{cov_s[64, 65] / (sd[64] * sd[65]):.2f}")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
    ax = axs[0]
    ax.plot(t, sd, color="k", lw=1.0, label="posterior")
    ax.plot(t[::4], res["full"][1][::4], "o", ms=2.2, color=C["a"], label="VI, full rank")
    ax.plot(t, res["mean-field"][1], color=C["b"], lw=1.0, label="VI, mean field")
    ax.plot(t, sd_mf, ":", color="k", lw=0.8, label="$1/\\sqrt{H_{ii}}$")
    ax.set_ylim(0, 0.075); ax.set_xlabel("$t$"); ax.set_ylabel("standard deviation")
    ax.set_title("(a) standard deviations"); ax.legend(frameon=False, fontsize=5.6, loc="upper center", ncol=2)
    ax = axs[1]
    ax.fill_between(t, mean_s - 1.96 * sd, mean_s + 1.96 * sd, color=C["a"], alpha=0.3, lw=0, label="95% band")
    ax.plot(t, res["mean-field"][0] - 1.96 * res["mean-field"][1], color=C["b"], lw=0.8, label="mean-field band")
    ax.plot(t, res["mean-field"][0] + 1.96 * res["mean-field"][1], color=C["b"], lw=0.8)
    ax.plot(t, x_s, color="k", lw=0.8, label="truth")
    ax.set_xlim(0.15, 0.45); ax.set_ylim(0.3, 1.12); ax.set_xlabel("$t$")
    ax.set_title("(b) bands at the first bump"); ax.legend(frameon=False, fontsize=5.6, loc="lower center")
    ax = axs[2]
    for fam, col in (("full", C["a"]), ("mean-field", C["b"])):
        it, kl = res[fam][2], res[fam][3]
        ax.loglog(it + 1, kl, color=col, lw=1.0, label="full rank" if fam == "full" else "mean field")
    ax.axhline(kl_mf, color=C["b"], ls=":", lw=0.8)
    ax.text(1.5, kl_mf * 1.35, f"best mean field: {kl_mf:.1f}", fontsize=6, color=C["b"])
    ax.set_ylim(0.05, 3e4); ax.set_xlabel("iteration"); ax.set_ylabel("$\\mathrm{KL}(q \\,\\|\\, p)$")
    ax.set_title("(c) KL divergence of the iterates"); ax.legend(frameon=False, fontsize=6, loc="upper right")
    save(fig, "vimf.pdf")


# ====================================================================== variational Bayes for the hierarchical prior
def fig_vbhier():
    rng = np.random.default_rng(5)                          # the same chains as in Example 4.14
    N_GIBBS, BURN = 20000, 2000
    t0 = time.time()
    P_g = gaussian_precision(n, GAMMA_PRIOR, "differences", gamma0=1.0)
    perturb_optimize(A, y_p, sigma, np.linalg.cholesky(P_g).T, N_GIBBS - BURN, rng)
    gibbs_laplace(A, y_p, sigma, GAMMA_PRIOR, N_GIBBS, rng=rng)
    t1 = time.time()
    X_h, th_h = gibbs_hierarchical(A, y_p, sigma, BETA, VARTHETA, N_GIBBS, rng=rng)
    t_gibbs = time.time() - t1
    X_h = X_h[BURN:]
    mu_g = X_h.mean(0); sd_g = X_h.std(0); lo_g, hi_g = np.percentile(X_h, [2.5, 97.5], axis=0)
    loud_g = np.mean(np.abs(X_h @ Ln.T) > 0.1, axis=0)
    t0 = time.time()
    mu_v, cov_v, (a_v, b_v), its = cavi_hierarchical(A, y_p, sigma, BETA, VARTHETA)
    t_vb = time.time() - t0
    sd_v = np.sqrt(np.diag(cov_v)); lo_v, hi_v = mu_v - 1.96 * sd_v, mu_v + 1.96 * sd_v
    mz, sz = Ln @ mu_v, np.sqrt(np.diag(Ln @ cov_v @ Ln.T))
    loud_v = norm.sf((0.1 - mz) / sz) + norm.cdf((-0.1 - mz) / sz)
    x_ias, _, _ = ias(A, y_p, sigma, BETA, VARTHETA, transform="differences")
    cover = lambda lo, hi: 100 * np.mean((x_p >= lo) & (x_p <= hi))
    print(f"variational Bayes, hierarchical prior: {its} iterations in {t_vb:.2f}s (Gibbs: {N_GIBBS} sweeps in {t_gibbs:.0f}s); "
          f"error of the mean VB {np.linalg.norm(mu_v - x_p):.2f}, Gibbs {np.linalg.norm(mu_g - x_p):.2f}, IAS {np.linalg.norm(x_ias - x_p):.2f}; "
          f"max |VB mean - IAS| {np.max(np.abs(mu_v - x_ias)):.3f}; coverage VB {cover(lo_v, hi_v):.0f}%, Gibbs {cover(lo_g, hi_g):.0f}%; "
          f"band width at the jumps VB {np.round(hi_v[jumps] - lo_v[jumps], 2).tolist()}, Gibbs {np.round(hi_g[jumps] - lo_g[jumps], 2).tolist()}; "
          f"on the first plateau VB {np.mean(hi_v[25:45] - lo_v[25:45]):.3f}, Gibbs {np.mean(hi_g[25:45] - lo_g[25:45]):.3f}; "
          f"std ratio VB/Gibbs {np.min(sd_v / sd_g):.2f}-{np.max(sd_v / sd_g):.2f} (median {np.median(sd_v / sd_g):.2f}); "
          f"max |VB mean - Gibbs mean| / Gibbs std {np.max(np.abs(mu_v - mu_g) / sd_g):.2f}")
    print(f"   P(|increment| > 0.1): VB at the four large jumps {np.round([loud_v[j - 2:j + 3].max() for j in jumps[[0, 1, 2, 4]]], 2).tolist()}, "
          f"Gibbs {np.round([loud_g[j - 2:j + 3].max() for j in jumps[[0, 1, 2, 4]]], 2).tolist()}; expected number of loud increments VB {loud_v.sum():.1f}, "
          f"Gibbs {loud_g.sum():.1f}; small step: VB {loud_v[78:88].max():.3f}, Gibbs {loud_g[78:88].max():.3f}")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
    ax = axs[0]
    ax.fill_between(t, lo_g, hi_g, color=C["c"], alpha=0.3, lw=0, label="Gibbs, 95% band")
    ax.plot(t, lo_v, color=C["b"], lw=0.8, label="VB, 95% band")
    ax.plot(t, hi_v, color=C["b"], lw=0.8)
    ax.plot(t, x_p, color="k", lw=0.8, label="truth")
    ax.set_ylim(-0.3, 1.65); ax.set_xlabel("$t$")
    ax.set_title("(a) credible bands"); ax.legend(frameon=False, fontsize=5.6, loc="upper right", ncol=2, columnspacing=0.8)
    ax = axs[1]
    for jj in jumps:
        ax.axvline(t[jj], color="0.85", lw=0.7)
    ax.bar(t[:-1], loud_g, width=1.0 / n, color=C["c"], label="Gibbs")
    ax.plot(t[:-1], loud_v, "-", color=C["b"], lw=0.9, label="VB")
    ax.set_ylim(0, 1.05); ax.set_xlabel("$t$")
    ax.set_title("(b) $\\mathbb{P}(|(\\mathbf{L}\\mathbf{x})_i| > 0.1 \\mid \\mathbf{y})$"); ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax = axs[2]
    ax.plot(t, mu_g, color=C["c"], lw=1.8, alpha=0.75, label="Gibbs, mean")
    ax.plot(t, mu_v, color=C["b"], lw=0.9, label="VB, mean")
    ax.plot(t, x_p, color="k", lw=0.7, label="truth")
    ax.set_ylim(-0.3, 1.65); ax.set_xlabel("$t$")
    ax.set_title("(c) posterior means"); ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    save(fig, "vbhier.pdf")



# ====================================================================== ensemble Kalman updates on the bent ridge
def fig_enkidea():
    Fr = lambda x: np.array([x[0] + 0.5 * x[1] ** 2])
    yb = np.array([Y_B])
    rng = np.random.default_rng(2)
    X0 = rng.standard_normal((300, 2))
    F0 = np.array([Fr(x) for x in X0])
    X1 = ensemble_kalman_step(X0, F0, yb, SIG_B, np.random.default_rng(3))
    X8, _, _ = es_mda(Fr, X0, yb, SIG_B, 8, np.random.default_rng(4))
    big = np.random.default_rng(5).standard_normal((20000, 2))
    Fb = np.array([Fr(x) for x in big])
    dX, dF = big - big.mean(0), Fb - Fb.mean(0)
    K = (dX.T @ dF / (len(big) - 1)) / (dF.var(ddof=1) + SIG_B**2)
    B1 = ensemble_kalman_step(big, Fb, yb, SIG_B, np.random.default_rng(6))
    B8, _, _ = es_mda(Fr, big, yb, SIG_B, 8, np.random.default_rng(7))
    Bk, _, _ = es_mda(Fr, big, yb, SIG_B, 64, np.random.default_rng(8))
    for name, Xb in (("one step", B1), ("ES-MDA, 8 steps", B8), ("ES-MDA, 64 steps", Bk)):
        print(f"ensemble Kalman on the bent ridge, {name} (J = 20000): mean {Xb.mean(0).round(3)}, std {Xb.std(0).round(3)} "
              f"(posterior: mean ({E_X1:.3f}, 0), std ({SD_X1:.3f}, {SD_X2:.3f}))")
    print(f"   gain of the first step (J = 20000): {K.ravel().round(3)}; conditional std of x1 given x2 after one step "
          f"{np.sqrt((1 - K[0, 0]) ** 2 + K[0, 0] ** 2 * SIG_B**2):.3f} (posterior {1 / np.sqrt(1 + 1 / SIG_B**2):.3f})")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.15), constrained_layout=True)
    for ax in axs:
        banana_contours(ax)
        ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 3.0); ax.set_aspect("equal"); ax.set_xlabel("$x_1$")
    ax = axs[0]
    ax.plot(X0[:, 0], X0[:, 1], ".", color="0.6", ms=2)
    for j in range(0, 300, 5):
        ax.annotate("", xy=X1[j], xytext=X0[j], arrowprops=dict(arrowstyle="-|>", color=C["a"], lw=0.5, mutation_scale=5))
    ax.set_ylabel("$x_2$"); ax.set_title("(a) one update")
    ax = axs[1]
    ax.plot(X1[:, 0], X1[:, 1], ".", color=C["a"], ms=2)
    ax.set_title("(b) after one update")
    ax = axs[2]
    ax.plot(X8[:, 0], X8[:, 1], ".", color=C["a"], ms=2)
    ax.set_title("(c) after eight inflated updates")
    save(fig, "enkidea.pdf")


# ====================================================================== ensemble Kalman methods on the bathymetry problem
def fig_enkbathy():
    d = bathy(20)
    Xh = bathy_hmc()
    mh, sh = Xh.mean(0), Xh.std(0)
    lo_h, hi_h = np.percentile(Xh, [2.5, 97.5], axis=0)
    bt, xg = d["b_true"][:-1], d["xg"][:-1]
    rng = np.random.default_rng(11)
    X0s, rej = {}, 0
    for J in (50, 100, 200, 400):
        X0s[J], tries = prior_ensemble(d, J, rng)
        rej += tries - J
    print(f"ensembles on the bathymetry problem: {rej} of {sum(X0s) + rej} prior draws rejected (no subcritical steady state)")
    for gam in (GAMMA_W, 0.03):                             # the priors of Examples 4.19(b) and 4.18
        probe, bad = np.random.default_rng(12), 0
        Rg = d["R"] * (gam / GAMMA_W)
        for k in range(10000):
            bad += not np.all(np.isfinite(d["F"](Rg @ probe.standard_normal(199))))
        print(f"   fraction of prior draws without a subcritical steady state: {bad / 10000:.4f} "
              f"(+- {np.sqrt(bad / 10000 * (1 - bad / 10000) / 10000):.4f}; {100 * gam:.0f} cm per cell, 10000 draws)")
    res = {}
    for J, Na in ((400, 1), (50, 8), (100, 8), (200, 8), (400, 8)):
        t0 = time.time()
        X, nf, hv = es_mda(d["F"], X0s[J], d["y"], SIG_W, Na, np.random.default_rng(J + Na))
        me, se = X.mean(0), X.std(0, ddof=1)
        res[(J, Na)] = (me, se)
        cov = 100 * np.mean((bt >= me - 1.96 * se) & (bt <= me + 1.96 * se))
        print(f"   ES-MDA J = {J}, {Na} steps: {nf} forward solves, {hv} halvings, {time.time() - t0:.2f}s; |mean - HMC mean| / HMC std: "
              f"max {np.max(np.abs(me - mh) / sh):.2f}, median {np.median(np.abs(me - mh) / sh):.2f}; std / HMC std {np.min(se / sh):.2f}-{np.max(se / sh):.2f} "
              f"(median {np.median(se / sh):.2f}); truth inside the band at {cov:.0f}% of the cells")
    print(f"   HMC: truth inside the 95% band at {100 * np.mean((bt >= lo_h) & (bt <= hi_h)):.0f}% of the cells")
    # Tikhonov EKI against Gauss-Newton with a finite-difference Jacobian, both counted in forward solves
    rec = lambda X, FX: {"d": 100 * np.sqrt(np.mean((X.mean(0) - d["b_map"]) ** 2)), "s": 100 * np.sqrt(np.mean(X.var(0)))}
    teki = {}
    for J in (50, 100, 200):
        _, recs, nf = eki(d["F"], X0s[J], d["y"], SIG_W, 40, np.random.default_rng(20 + J), prior_factor=d["Lw"], record=rec)
        teki[J] = (J * (np.arange(1, 41) + 1), np.array([r["d"] for r in recs]), np.array([r["s"] for r in recs]))
        print(f"   Tikhonov EKI J = {J}: distance of the mean to the MAP after 10, 20, 40 iterations "
              f"{teki[J][1][[9, 19, 39]].round(2).tolist()} cm; spread {teki[J][2][[9, 19, 39]].round(3).tolist()} cm; {nf} forward solves")
    for J in (50, 100, 200):                                # the best point that Tikhonov EKI can reach (Lemma 5.15)
        xb = X0s[J].mean(0); D = (X0s[J] - xb).T
        c = np.zeros(J)
        for it in range(30):
            xs = xb + D @ c
            r = np.r_[(d["F"](xs) - d["y"]) / SIG_W, d["Lw"] @ xs]
            Js = np.vstack([d["Jf"](xs) @ D / SIG_W, d["Lw"] @ D])
            step = np.linalg.lstsq(Js, -r, rcond=None)[0]
            c = c + step
            if np.linalg.norm(step) < 1e-10:
                break
        print(f"   J = {J}: the minimizer of U on the span of the initial ensemble is "
              f"{100 * np.sqrt(np.mean((xb + D @ c - d['b_map']) ** 2)):.2f} cm from the MAP estimate")
    cnt = {"F": 0}

    def Fc(b):
        cnt["F"] += 1
        return d["F"](b)
    gn_cost, gn_dist = [0], [100 * np.sqrt(np.mean(d["b_map"] ** 2))]
    for k in range(1, 7):
        cnt["F"] = 0
        bk, _ = gauss_newton(Fc, lambda b: fd_jacobian(Fc, b), d["y"], np.zeros(199), lam=SIG_W**2, L=d["Lw"], n_iter=k, tol=0.0)
        gn_cost.append(cnt["F"]); gn_dist.append(100 * np.sqrt(np.mean((bk - d["b_map"]) ** 2)))
    print(f"   Gauss-Newton with finite differences: forward solves {gn_cost}, distance to the MAP {np.round(gn_dist, 4).tolist()} cm")
    _, recs, _ = eki(d["F"], X0s[200], d["y"], SIG_W, 30, np.random.default_rng(31),
                     record=lambda X, FX: {"mis": np.linalg.norm(d["F"](X.mean(0)) - d["y"])})
    mis = np.array([r["mis"] for r in recs])
    print(f"   plain EKI J = 200: misfit of the mean after 1, 2, 3, 5, 10, 30 iterations {mis[[0, 1, 2, 4, 9, 29]].round(4).tolist()} "
          f"(noise level {np.sqrt(20) * SIG_W:.4f}); first below the noise level after {int(np.argmax(mis < np.sqrt(20) * SIG_W)) + 1} iterations")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.05), constrained_layout=True)
    ax = axs[0]
    me, se = res[(200, 8)]
    ax.fill_between(xg, lo_h, hi_h, color=C["a"], alpha=0.3, lw=0, label="HMC, 95% band")
    ax.plot(xg, me - 1.96 * se, color=C["b"], lw=0.8, label="ES-MDA, $J = 200$")
    ax.plot(xg, me + 1.96 * se, color=C["b"], lw=0.8)
    ax.plot(d["xg"], d["b_true"], color="k", lw=0.8, label="truth")
    ax.plot(d["xo"], np.full(20, -0.105), "|", color="k", ms=3)
    ax.set_ylim(-0.12, 0.34); ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$b$ [m]")
    ax.set_title("(a) bands"); ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    ax = axs[1]
    for (J, Na), col, ls in (((400, 1), C["d"], ":"), ((50, 8), C["c"], "-"), ((400, 8), C["b"], "-")):
        ax.plot(xg, res[(J, Na)][1] / sh, ls, color=col, lw=0.9,
                label=f"$J = {J}$, " + ("one update" if Na == 1 else f"{Na} updates"))
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_ylim(0.4, 2.4); ax.set_xlabel("$x$ [m]")
    ax.set_title("(b) std / std of HMC"); ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    ax = axs[2]
    for J, col in ((50, C["c"]), (100, C["d"]), (200, C["b"])):
        ax.loglog(teki[J][0], teki[J][1], color=col, lw=0.9, label=f"EKI, $J = {J}$")
    ax.loglog(teki[200][0], teki[200][2], ":", color=C["b"], lw=0.9, label="spread, $J = 200$")
    keep = np.array(gn_dist[1:]) > 1e-3
    ax.loglog(np.array(gn_cost[1:])[keep], np.array(gn_dist[1:])[keep], "s-", color="k", ms=2.5, lw=0.8, label="Gauss-Newton, FD")
    ax.set_ylim(1e-3, 30); ax.set_xlabel("forward solves"); ax.set_ylabel("distance to the MAP [cm]")
    ax.set_title("(c) Tikhonov EKI as an optimizer"); ax.legend(frameon=False, fontsize=5.4, loc="lower right")
    save(fig, "enkbathy.pdf")


# ====================================================================== Gaussian VI for the bathymetry problem (numbers only)
def fig_vibathy():
    import jax
    import jax.numpy as jnp
    d = bathy(20)
    Xh = bathy_hmc()
    sh, mh = Xh.std(0), Xh.mean(0)
    Hinv = np.linalg.inv(d["H"])
    Cl = np.linalg.cholesky(Hinv)
    sd_l = np.sqrt(np.diag(Hinv))
    Clj, bmj = jnp.asarray(Cl), jnp.asarray(d["b_map"])
    Uw = lambda w: d["U"](bmj + Clj @ w)                       # coordinates of the Laplace approximation: x = x_MAP + C_L w
    t0 = time.time()
    mw, Cw, rec = gaussian_vi(Uw, np.zeros(199), C0=np.eye(199), family="full", n_iter=4000, n_samples=8, lr=1e-2,
                              lr_final=1e-3, seed=0, record=50, average=0.5)
    mq, Cq = d["b_map"] + Cl @ mw, Cl @ Cw
    sq = np.sqrt(np.sum(Cq**2, axis=1))
    Ub = jax.jit(jax.vmap(d["U"]))
    eps = np.random.default_rng(14).standard_normal((20000, 199))
    elbo = lambda mm, Cc: -np.asarray(Ub(jnp.asarray(mm + eps @ Cc.T))) + np.sum(np.log(np.abs(np.diag(Cc)))) + 99.5 * (1 + np.log(2 * np.pi))
    e_l, e_v = elbo(d["b_map"], Cl), elbo(mq, Cq)
    print(f"Gaussian VI on the bathymetry posterior, full rank, in the coordinates of the Laplace approximation, started there: "
          f"{time.time() - t0:.0f}s, {rec['gradients']} gradients, average of the last half of the iterates; "
          f"ELBO (20000 common draws) Laplace {e_l.mean():.2f}, VI {e_v.mean():.2f}, difference {np.mean(e_v - e_l):.2f} +- {np.std(e_v - e_l) / np.sqrt(len(eps)):.2f}; "
          f"std / Laplace std {np.min(sq / sd_l):.3f}-{np.max(sq / sd_l):.3f}; std / HMC std {np.min(sq / sh):.3f}-{np.max(sq / sh):.3f}; "
          f"|mean - HMC mean| / HMC std max {np.max(np.abs(mq - mh) / sh):.2f} (MAP: {np.max(np.abs(d['b_map'] - mh) / sh):.2f}); "
          f"Monte Carlo error of the HMC reference: mean {1 / np.sqrt(2551):.3f} posterior std, std {1 / np.sqrt(2 * 2551):.3f} relative (per component, at the smallest ESS)")


FIGURES = {"laplaceidea": fig_laplaceidea, "laplacefail": fig_laplacefail, "laplacebathy": fig_laplacebathy, "kl": fig_kl,
           "viridge": fig_viridge, "vimf": fig_vimf, "vbhier": fig_vbhier, "enkidea": fig_enkidea, "enkbathy": fig_enkbathy,
           "vibathy": fig_vibathy}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGURES):
        FIGURES[name]()

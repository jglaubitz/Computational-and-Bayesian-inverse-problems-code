"""Figures of Chapter 4 (uncertainty quantification by sampling). Output: ../figures/ch04/*.pdf

    python ch04_figures.py                  all figures (about five minutes; NUTS needs blackjax)
    python ch04_figures.py mhidea hmcidea   selected figures only

Figures: walkers (with mcerror), mcmcidea, mhidea, rwmh, ptoidea, pto (with the truncated-CG experiment), gibbsidea,
threepriors, funnel, diagnostics, hmcidea, essidea, essref, hmc (with bathyprior).
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import log_ndtr

from running_examples import (deconvolution_operator, signal_smooth, signal_piecewise, make_grid, bottom_smooth,
                              N_CELLS, M_OBS, NOISE_STD, L as LEN)
from solvers import difference_matrix, gaussian_posterior, gauss_newton, ias
from priors import gaussian_precision
from samplers import (rw_metropolis, perturb_optimize, gibbs_hierarchical, gibbs_laplace, make_hmc_jax,
                      elliptical_slice, autocorrelation, iat, ess, rank_rhat)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "ch04")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "lines.linewidth": 1.1})
W = 6.3
C = {"truth": "k", "rec": "tab:red", "a": "tab:blue", "b": "tab:orange", "c": "tab:green", "d": "tab:purple"}
COLS = [C["a"], C["b"], C["c"]]


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), **kw)
    plt.close(fig)
    print("wrote", name)


# ====================================================================== the bent ridge (banana) of Figure 3.2(b)
Y_B, SIG_B = 1.0, 0.3                                     # prior N(0, I), y = x1 + x2^2/2 + eps, eps ~ N(0, 0.3^2), y = 1


def U_banana(x):
    """Potential -log p(x | y) for points x of shape (..., 2)."""
    r = Y_B - x[..., 0] - 0.5 * x[..., 1] ** 2
    return 0.5 * (x[..., 0] ** 2 + x[..., 1] ** 2) + r**2 / (2 * SIG_B**2)


def grad_banana(x):
    r = Y_B - x[..., 0] - 0.5 * x[..., 1] ** 2
    return np.stack([x[..., 0] - r / SIG_B**2, x[..., 1] - r * x[..., 1] / SIG_B**2], axis=-1)


GB = np.linspace(-4, 4, 1601)
GX1, GX2 = np.meshgrid(GB, GB)
PB = np.exp(-(U_banana(np.stack([GX1, GX2], -1)) - U_banana(np.stack([GX1, GX2], -1)).min()))
PB /= PB.sum()
E_X1 = float((GX1 * PB).sum())
SD_X1 = float(np.sqrt(((GX1 - E_X1) ** 2 * PB).sum()))
SD_X2 = float(np.sqrt((GX2**2 * PB).sum()))
X_MAP_B = np.array([0.92, 0.0])
LEV = np.exp(-np.array([3.0, 1.5, 0.4]))                  # contour levels relative to the maximum


def banana_contours(ax):
    ax.contour(GX1, GX2, PB / PB.max(), levels=sorted(LEV), colors="0.7", linewidths=0.7)


def exact_banana(size, rng):
    """Independent draws from the banana: cells of the quadrature grid drawn by probability, jittered within the cell."""
    idx = rng.choice(PB.size, size=size, p=PB.ravel())
    h = GB[1] - GB[0]
    return np.stack([GX1.ravel()[idx], GX2.ravel()[idx]], -1) + h * (rng.random((size, 2)) - 0.5)


def rw_banana(x0, n, step, rng):
    """Random-walk Metropolis for R parallel chains, x0 of shape (R, 2). Returns (n + 1, R, 2)."""
    x = x0.copy(); u = U_banana(x); out = np.empty((n + 1,) + x.shape); out[0] = x
    for k in range(n):
        prop = x + step * rng.standard_normal(x.shape); up = U_banana(prop)
        acc = np.log(rng.random(len(x))) < u - up
        x[acc], u[acc] = prop[acc], up[acc]; out[k + 1] = x
    return out


def cond_x2(x1):
    """Unnormalized conditional density p(x2 | x1, y) on the grid GB."""
    lg = -0.5 * GB**2 - (Y_B - x1 - 0.5 * GB**2) ** 2 / (2 * SIG_B**2)
    return np.exp(lg - lg.max())


PREC_X1 = 1 + 1 / SIG_B**2                                 # x1 | x2 is Gaussian with this precision


def mean_x1(x2):
    return ((Y_B - 0.5 * x2**2) / SIG_B**2) / PREC_X1


def gibbs_banana(x0, n, rng, record_half_steps=False):
    """Gibbs sampler for R parallel chains: x1 | x2 is Gaussian, x2 | x1 is drawn by inversion on the grid GB."""
    x = x0.copy()
    out = [x.copy()]
    for k in range(n):
        x[:, 0] = mean_x1(x[:, 1]) + rng.standard_normal(len(x)) / np.sqrt(PREC_X1)
        if record_half_steps:
            out.append(x.copy())
        lg = -0.5 * GB[None, :] ** 2 - (Y_B - x[:, :1] - 0.5 * GB[None, :] ** 2) ** 2 / (2 * SIG_B**2)
        w = np.exp(lg - lg.max(axis=1, keepdims=True)); cdf = np.cumsum(w, axis=1); cdf /= cdf[:, -1:]
        u = rng.random(len(x))
        x[:, 1] = [np.interp(u[i], cdf[i], GB) for i in range(len(x))]
        out.append(x.copy())
    return np.array(out)


def hmc_banana(x0, n, eps, n_leap, rng, record_paths=False, jitter=0.2):
    """HMC with identity mass matrix for R parallel chains; optionally records the leapfrog paths of chain 0."""
    x = x0.copy(); out = np.empty((n + 1,) + x.shape); out[0] = x; paths = []
    for k in range(n):
        p = rng.standard_normal(x.shape); H0 = U_banana(x) + 0.5 * (p**2).sum(-1)
        h = eps * (1 + jitter * (2 * rng.random() - 1))
        xn, pn = x.copy(), p - 0.5 * h * grad_banana(x); path = [xn[0].copy()]
        for j in range(n_leap):
            xn = xn + h * pn
            if j < n_leap - 1:
                pn = pn - h * grad_banana(xn)
            path.append(xn[0].copy())
        pn = pn - 0.5 * h * grad_banana(xn)
        H1 = U_banana(xn) + 0.5 * (pn**2).sum(-1)
        acc = np.log(rng.random(len(x))) < H0 - H1
        x[acc] = xn[acc]; out[k + 1] = x
        if record_paths:
            paths.append((np.array(path), bool(acc[0])))
    return (out, paths) if record_paths else out


RW_STEP_B, HMC_EPS_B, HMC_L_B = 0.8, 0.15, 15

# ====================================================================== a one-dimensional target with two modes
MIX = [(0.65, -1.0, 0.55), (0.35, 1.5, 0.6)]              # weights, means, standard deviations
X1D = np.linspace(-4.0, 5.0, 901)


def p1d(x):
    return sum(w * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)) for w, m, s in MIX)


def U1d(x):
    return -np.log(p1d(x))


def dU1d(x, h=1e-5):
    return (U1d(x + h) - U1d(x - h)) / (2 * h)


# ====================================================================== deconvolution data (as in Chapters 2 and 3)
n, m, sigma = N_CELLS, M_OBS, NOISE_STD
A, t, s = deconvolution_operator(n, m)
rng_data = np.random.default_rng(1)
x_s, x_p = signal_smooth(t), signal_piecewise(t)
y_s = A @ x_s + sigma * rng_data.standard_normal(m)
y_p = A @ x_p + sigma * rng_data.standard_normal(m)
Ln = difference_matrix(n)
jumps = np.nonzero(Ln @ x_p)[0]
GAMMA_SMOOTH = 0.0294                                      # Example 3.12: lambda from the discrepancy principle
P_s = gaussian_precision(n, GAMMA_SMOOTH, "differences", gamma0=1.0)
mean_s, cov_s = gaussian_posterior(A, y_s, sigma**2, P_s)
sd_s = np.sqrt(np.diag(cov_s))
i30 = int(np.argmin(np.abs(t - 0.3)))
i60 = int(np.argmin(np.abs(t - 0.6)))
GAMMA_PRIOR, BETA, VARTHETA = 0.1, 1.0, 1e-3


# ---------------------------------------------------------------- three walkers on the banana, and their Monte Carlo errors
def fig_walkers():
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.3), constrained_layout=True, sharex=True, sharey=True)
    for ax in axs:
        banana_contours(ax)
        ax.set_xlim(-2.1, 2.0); ax.set_ylim(-2.7, 2.7); ax.set_xlabel("$x_1$")
    axs[0].set_ylabel("$x_2$")
    ch = rw_banana(X_MAP_B[None, :].copy(), 300, RW_STEP_B, rng)[:, 0]
    n_moves = int(np.sum(np.any(np.diff(ch, axis=0) != 0, axis=1)))
    axs[0].plot(ch[:, 0], ch[:, 1], "-", color=C["a"], lw=0.5, alpha=0.8)
    axs[0].plot(ch[:, 0], ch[:, 1], ".", color=C["a"], ms=2)
    axs[0].set_title(f"(a) random walk: 300 steps, {n_moves} moves")
    chg = gibbs_banana(X_MAP_B[None, :].copy(), 60, rng, record_half_steps=True)[:, 0]
    axs[1].plot(chg[:, 0], chg[:, 1], "-", color=C["b"], lw=0.6)
    axs[1].plot(chg[::2, 0], chg[::2, 1], ".", color=C["b"], ms=2.5)
    axs[1].set_title("(b) Gibbs: 60 sweeps")
    chh, paths = hmc_banana(X_MAP_B[None, :].copy(), 8, HMC_EPS_B, HMC_L_B, rng, record_paths=True)
    for path, acc in paths:
        axs[2].plot(path[:, 0], path[:, 1], "-", color=C["c"] if acc else "0.6", lw=0.6)
    axs[2].plot(chh[:, 0, 0], chh[:, 0, 1], "o", color=C["c"], ms=3)
    axs[2].set_title("(c) Hamiltonian: 8 trajectories")
    save(fig, "walkers.pdf")
    print(f"banana: MAP {X_MAP_B}, E[x1|y] = {E_X1:.4f}, sd(x1) = {SD_X1:.3f}, sd(x2) = {SD_X2:.3f}")
    # Monte Carlo error against N (the same random stream continues)
    R = 200
    Ns = np.unique(np.logspace(1, 4, 13).astype(int))
    x0 = exact_banana(R, rng)                                   # stationary starts
    est = {}
    X_ind = exact_banana(R * Ns[-1], rng).reshape(R, Ns[-1], 2)
    est["independent"] = np.cumsum(X_ind[:, :, 0], axis=1)
    est["random walk"] = np.cumsum(rw_banana(x0.copy(), Ns[-1], RW_STEP_B, rng)[1:, :, 0].T, axis=1)
    est["Gibbs"] = np.cumsum(gibbs_banana(x0.copy(), Ns[-1], rng)[1:, :, 0].T, axis=1)
    est["Hamiltonian"] = np.cumsum(hmc_banana(x0.copy(), Ns[-1], HMC_EPS_B, HMC_L_B, rng)[1:, :, 0].T, axis=1)
    taus = {}
    fig, ax = plt.subplots(1, 1, figsize=(W * 0.6, 2.4), constrained_layout=True)
    for (name, cs), col in zip(est.items(), ["k", C["a"], C["b"], C["c"]]):
        rmse = np.sqrt(np.mean((cs[:, Ns - 1] / Ns - E_X1) ** 2, axis=0))
        chains = np.diff(np.concatenate([np.zeros((R, 1)), cs], axis=1), axis=1)
        taus[name] = np.mean([iat(chains[i]) for i in range(20)])
        ax.loglog(Ns, rmse, "o-", color=col, ms=3, label=f"{name} ($\\tau = {taus[name]:.0f}$)")
        ax.loglog(Ns, SD_X1 * np.sqrt(max(taus[name], 1.0) / Ns), ":", color=col, lw=0.8)
    ax.set_xlabel("number of samples $N$"); ax.set_ylabel("RMS error of the estimate of $\\mathbb{E}[x_1 \\mid y]$")
    ax.legend(frameon=False, fontsize=6.5)
    print("   integrated autocorrelation times " + ", ".join(f"{k} {v:.1f}" for k, v in taus.items()))
    save(fig, "mcerror.pdf")


# ---------------------------------------------------------------- the idea of MCMC: a transition density and the distributions it produces
def fig_mcmcidea():
    x = X1D; dx = x[1] - x[0]; p = p1d(x); s = 1.0
    Q = np.exp(-0.5 * ((x[:, None] - x[None, :]) / s) ** 2) / (s * np.sqrt(2 * np.pi)) * dx   # Q[i, j] = q(x_i | x_j) dx
    alpha = np.minimum(1.0, p[:, None] / p[None, :])
    K = Q * alpha
    np.fill_diagonal(K, 0.0)
    K[np.diag_indices_from(K)] = 1.0 - K.sum(axis=0)                                          # column j: law of the next state
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
    ax = axs[0]
    j = int(np.argmin(np.abs(x - 0.9)))
    cont = (Q[:, j] * alpha[:, j]) / dx; cont[j] = 0.0
    r = K[j, j]
    ax.plot(x, p, color="k", lw=1.0, label="target $p$")
    ax.plot(x, Q[:, j] / dx, "--", color="0.5", lw=0.8, label="proposal $q(\\cdot \\mid x)$")
    ax.fill_between(x, 0, cont, color=C["a"], alpha=0.35, lw=0, label="move: $q\\,\\alpha$")
    ax.annotate("", xy=(x[j], 0.52), xytext=(x[j], 0.0), arrowprops=dict(arrowstyle="-|>", color=C["a"], lw=1.2))
    ax.text(x[j] + 0.12, 0.5, f"stay: {r:.2f}", color=C["a"], fontsize=6.5, va="top")
    ax.set_xlim(-3.5, 4.0); ax.set_ylim(0, 0.86); ax.set_yticks([]); ax.set_xlabel("$x'$")
    ax.set_title(f"(a) $K(x' \\mid x)$ for $x = {x[j]:.1f}$"); ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax = axs[1]
    p0 = np.exp(-0.5 * ((x - 3.2) / 0.3) ** 2); p0 /= p0.sum()
    pk = p0.copy(); dist = [0.5 * np.abs(pk - p * dx).sum()]
    show = {1: None, 3: None, 10: None, 30: None, 100: None}
    for k in range(1, 201):
        pk = K @ pk
        dist.append(0.5 * np.abs(pk - p * dx).sum())
        if k in show:
            show[k] = pk.copy()
    cmap = plt.get_cmap("Blues")
    for i, (k, pkk) in enumerate(show.items()):
        ax.plot(x, pkk / dx, color=cmap(0.35 + 0.15 * i), lw=0.9, label=f"$k = {k}$")
    ax.plot(x, p, color="k", lw=1.3, label="target")
    ax.set_xlim(-3.5, 4.5); ax.set_ylim(0, 1.0); ax.set_yticks([]); ax.set_xlabel("$x$")
    ax.set_title("(b) density of $x^{(k)}$, started near $x = 3.2$"); ax.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=2)
    ax = axs[2]
    ax.semilogy(np.arange(len(dist)), dist, color=C["a"])
    ax.set_xlabel("step $k$"); ax.set_title("(c) total variation distance to $p$"); ax.set_ylim(1e-4, 1.5)
    print(f"mcmc idea: stay probability at x = {x[j]:.2f}: {r:.3f}; TV distance after 10, 30, 100 steps: {dist[10]:.3f}, {dist[30]:.4f}, {dist[100]:.5f}")
    save(fig, "mcmcidea.pdf")


# ---------------------------------------------------------------- the idea of Metropolis-Hastings, and its first steps on the banana
def fig_mhidea():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.45), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axs[0]
    x = np.linspace(-3.2, 3.6, 600)
    pn = lambda z: np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)                 # a standard normal target
    xk, xa, xb = 0.8, 0.25, 1.9
    pk, pa, pb = pn(xk), pn(xa), pn(xb)
    ax.plot(x, pn(x), color="k", lw=1.0)
    ax.plot(x, 0.16 * np.exp(-0.5 * ((x - xk) / 0.8) ** 2), "--", color="0.5", lw=0.8)
    ax.annotate("proposal $q(\\cdot \\mid x^{(k)})$", xy=(-0.55, 0.038), xytext=(-3.1, 0.105), color="0.4", fontsize=6.3,
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))
    for xx, pp, col in [(xk, pk, "k"), (xa, pa, C["c"]), (xb, pb, C["rec"])]:
        ax.plot([xx, xx], [0, pp], color=col, lw=0.9)
        ax.plot(xx, pp, "o", color=col, ms=3.5)
    ax.plot([xk, 3.4], [pk, pk], color="0.6", lw=0.6, ls=":")
    ax.text(xk, -0.035, "$x^{(k)}$", ha="center", fontsize=7)
    ax.text(xa, -0.035, "$x'_1$", ha="center", fontsize=7, color=C["c"])
    ax.text(xb, -0.035, "$x'_2$", ha="center", fontsize=7, color=C["rec"])
    ax.text(xa - 0.1, pa + 0.02, "higher: accept, $\\alpha = 1$", color=C["c"], fontsize=6.3, ha="right", va="bottom")
    ax.text(xb + 0.12, pb + 0.012, f"lower: accept with\n$\\alpha = p(x'_2)/p(x^{{(k)}}) = {pb / pk:.2f}$", color=C["rec"], fontsize=6.3,
            va="bottom")
    ax.set_xlim(-3.2, 3.6); ax.set_ylim(-0.05, 0.47); ax.set_yticks([]); ax.set_xticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("(a) propose, then accept or reject")
    ax = axs[1]
    banana_contours(ax)
    rng = np.random.default_rng(21)
    x = np.array([-1.3, -1.6]); u_x = U_banana(x); rows = []
    for k in range(10):
        prop = x + RW_STEP_B * rng.standard_normal(2); u_p = U_banana(prop)
        alpha = min(1.0, np.exp(u_x - u_p)); u = rng.random(); acc = u < alpha
        rows.append((k, x.copy(), prop.copy(), u_x, u_p, alpha, u, acc))
        if acc:
            ax.annotate("", xy=prop, xytext=x, arrowprops=dict(arrowstyle="-|>", color=C["a"], lw=0.9, shrinkA=1, shrinkB=1))
            ax.plot(*prop, "o", color=C["a"], ms=3.5)
        else:
            ax.annotate("", xy=prop, xytext=x, arrowprops=dict(arrowstyle="-", color="0.55", lw=0.7, ls="--", shrinkA=1, shrinkB=1))
            ax.plot(*prop, "x", color="0.45", ms=4)
        ax.text(prop[0] + 0.07, prop[1] + 0.07, str(k + 1), fontsize=6, color=C["a"] if acc else "0.45")
        if acc:
            x, u_x = prop, u_p
    ax.plot(-1.3, -1.6, "s", color="k", ms=3.5); ax.text(-1.95, -1.8, "$\\mathbf{x}^{(0)}$", fontsize=7)
    ax.set_xlim(-2.1, 2.0); ax.set_ylim(-2.7, 2.7); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("(b) the first ten proposals on the bent ridge")
    save(fig, "mhidea.pdf")
    print(f"MH idea: p(x_k) = {pk:.3f}, p(x'_1) = {pa:.3f}, p(x'_2) = {pb:.3f}, ratio {pb / pk:.3f}")
    print("MH on the banana, step 0.8, start (-1.3, -1.6):  k & x^(k) & x' & U(x^(k)) & U(x') & alpha & u & decision")
    for (k, xc, xp, ux, up, al, u, acc) in rows:
        print(f"   {k} & $({xc[0]:.2f}, {xc[1]:.2f})$ & $({xp[0]:.2f}, {xp[1]:.2f})$ & ${ux:.2f}$ & ${up:.2f}$ & ${al:.2f}$ & ${u:.2f}$ & {'accept' if acc else 'reject'} \\\\")


# ---------------------------------------------------------------- random-walk Metropolis-Hastings in n = 128 dimensions
def fig_rwmh():
    def lp_s(x):
        r = A @ x - y_s
        return -0.5 * r @ r / sigma**2 - 0.5 * x @ P_s @ x
    rng = np.random.default_rng(2)
    N_RW = 200000
    t0 = time.time()
    ch_iso, acc_iso = rw_metropolis(lp_s, mean_s.copy(), N_RW, 0.004, rng=rng)
    ch_pre, acc_pre = rw_metropolis(lp_s, mean_s.copy(), N_RW, 2.38 / np.sqrt(n), chol=np.linalg.cholesky(cov_s), rng=rng)
    tau_iso, tau_pre = iat(ch_iso[20000:, i30]), iat(ch_pre[20000:, i30])
    print(f"RW-MH on the deconvolution posterior ({time.time() - t0:.0f}s): isotropic step 0.004, acceptance {acc_iso:.2f}, tau {tau_iso:.0f}, "
          f"ESS {ess(ch_iso[20000:, i30]):.0f}; preconditioned step 2.38/sqrt(n) = {2.38 / np.sqrt(n):.2f}, acceptance {acc_pre:.2f}, "
          f"tau {tau_pre:.0f} (= {tau_pre / n:.1f} n), ESS {ess(ch_pre[20000:, i30]):.0f} of {N_RW - 20000}")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
    kk = np.arange(0, N_RW + 1, 100)
    for ax, ch, title, col in [(axs[0], ch_iso, "(a) isotropic proposal", C["a"]), (axs[1], ch_pre, "(b) proposal shaped like the posterior", C["b"])]:
        ax.plot(kk / 1000, ch[kk, i30], color=col, lw=0.5)
        for v in (mean_s[i30] - 1.96 * sd_s[i30], mean_s[i30], mean_s[i30] + 1.96 * sd_s[i30]):
            ax.axhline(v, color="k", lw=0.7, ls="--" if v != mean_s[i30] else "-")
        ax.set_xlabel("iteration [thousands]"); ax.set_title(title); ax.set_ylim(0.65, 1.05)
    axs[0].set_ylabel("$x(0.3)$")
    lags = np.unique(np.logspace(0, 4.7, 60).astype(int))
    for ch, col, name, tau in [(ch_iso, C["a"], "isotropic", tau_iso), (ch_pre, C["b"], "shaped", tau_pre)]:
        rho = autocorrelation(ch[20000:, i30])
        axs[2].semilogx(lags, rho[lags], color=col, label=f"{name}, $\\tau \\approx {tau:.0f}$")
    axs[2].axhline(0, color="0.6", lw=0.6)
    axs[2].set_xlabel("lag"); axs[2].set_title("(c) autocorrelation of $x(0.3)$"); axs[2].legend(frameon=False)
    save(fig, "rwmh.pdf")


# ---------------------------------------------------------------- the idea of perturb-then-optimize: the bowl moves, its shape does not
def fig_ptoidea():
    x_true2 = np.array([1.0, 0.5]); sig2 = 0.3
    A2 = np.array([[1.0, 0.8], [0.8, 1.0]])
    y2 = A2 @ x_true2 + sig2 * np.random.default_rng(7).standard_normal(2)     # the data of Figure 3.3(a)
    H2 = A2.T @ A2 / sig2**2 + np.eye(2)
    mu2 = np.linalg.solve(H2, A2.T @ y2 / sig2**2)
    g = np.linspace(-1.5, 2.5, 401); G1, G2 = np.meshgrid(g, g); P = np.stack([G1, G2], -1)

    def J(yv, eta):
        R = P @ A2.T - yv
        return 0.5 * (R**2).sum(-1) / sig2**2 + 0.5 * ((P - eta) ** 2).sum(-1)
    J0 = J(y2, np.zeros(2)); J0 -= J0.min()
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.2), constrained_layout=True, sharex=True, sharey=True)
    ax = axs[0]
    ax.contour(G1, G2, J0, levels=[0.5, 2.0, 4.5], colors=C["a"], linewidths=0.8)
    ax.plot(*mu2, "*", color="k", ms=7)
    ax.set_title("(a) the unperturbed objective"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax = axs[1]
    ax.contour(G1, G2, J0, levels=[0.5, 2.0], colors="0.75", linewidths=0.7)
    rng = np.random.default_rng(12)
    for col in [C["b"], C["c"], C["d"]]:
        xi, eta = rng.standard_normal(2), rng.standard_normal(2)
        Jk = J(y2 + sig2 * xi, eta); Jk -= Jk.min()
        ax.contour(G1, G2, Jk, levels=[0.5, 2.0], colors=col, linewidths=0.8)
        xk = np.linalg.solve(H2, A2.T @ (y2 + sig2 * xi) / sig2**2 + eta)
        ax.plot(*xk, "o", color=col, ms=3.5)
    ax.plot(*mu2, "*", color="k", ms=7)
    ax.set_title("(b) three perturbed objectives"); ax.set_xlabel("$x_1$")
    ax = axs[2]
    Xs = perturb_optimize(A2, y2, sig2, np.eye(2), 500, np.random.default_rng(13))
    ax.plot(Xs[:, 0], Xs[:, 1], ".", color="0.45", ms=1.8)
    ax.contour(G1, G2, J0, levels=[0.5 * 5.991], colors="k", linewidths=0.9)              # the exact 95 % ellipse
    ax.plot(*mu2, "*", color="k", ms=7)
    ax.set_title("(c) 500 minimizers, exact 95% ellipse"); ax.set_xlabel("$x_1$")
    inside = np.mean(np.einsum("ni,ij,nj->n", Xs - mu2, H2, Xs - mu2) <= 5.991)
    print(f"pto idea: posterior mean {mu2.round(3)}, fraction of the 500 minimizers inside the 95% ellipse {inside:.3f}")
    save(fig, "ptoidea.pdf")


# ---------------------------------------------------------------- perturb-then-optimize for the smooth signal, and truncated solves
def cg_solve(Hm, b, k, x0=None):
    """k steps of conjugate gradients for Hm x = b, started at x0 (default 0)."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - Hm @ x; d = r.copy(); rr = r @ r
    for _ in range(k):
        Hd = Hm @ d; a = rr / (d @ Hd)
        x = x + a * d; r = r - a * Hd
        rr_new = r @ r
        if rr_new < 1e-30:
            break
        d = r + (rr_new / rr) * d; rr = rr_new
    return x


def fig_pto():
    B_s = np.linalg.cholesky(P_s).T                             # P = B^T B
    rng = np.random.default_rng(3)
    t0 = time.time()
    X_pto = perturb_optimize(A, y_s, sigma, B_s, 4000, rng)
    print(f"perturb-then-optimize: 4000 exact samples in {time.time() - t0:.2f}s")
    Nc = np.unique(np.logspace(1, np.log10(4000), 12).astype(int))
    err_mean = [np.max(np.abs(X_pto[:k].mean(0) - mean_s) / sd_s) for k in Nc]
    err_sd = [np.max(np.abs(X_pto[:k].std(0) - sd_s) / sd_s) for k in Nc]
    p_exc = np.mean(X_pto > 0.5, axis=0)
    peak = X_pto[:, t < 0.5].max(axis=1)
    lo, hi = np.percentile(X_pto, [2.5, 97.5], axis=0)
    print(f"   max error of the mean after 4000 samples {err_mean[-1]:.3f} posterior std; P(x(t) > 0.5) between {p_exc.min():.2f} and {p_exc.max():.2f}; "
          f"peak height: posterior mean {peak.mean():.3f}, 95% interval [{np.percentile(peak, 2.5):.3f}, {np.percentile(peak, 97.5):.3f}], "
          f"peak of the posterior mean {mean_s[t < 0.5].max():.3f}, true peak {x_s[t < 0.5].max():.3f}; truth inside the sample band at "
          f"{100 * np.mean((x_s >= lo) & (x_s <= hi)):.0f}% of the points; largest difference of the band edges to the closed form "
          f"{np.max(np.abs(np.r_[lo - (mean_s - 1.96 * sd_s), hi - (mean_s + 1.96 * sd_s)])):.3f}")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
    ax = axs[0]
    ax.loglog(Nc, err_mean, "o-", color=C["a"], ms=3, label="mean")
    ax.loglog(Nc, err_sd, "s-", color=C["b"], ms=3, label="standard deviation")
    ax.loglog(Nc, 3.0 / np.sqrt(Nc), ":", color="0.4", lw=0.8, label="$\\propto N^{-1/2}$")
    ax.set_xlabel("number of samples $N$"); ax.set_title("(a) error against the closed form"); ax.legend(frameon=False, fontsize=6.3)
    ax = axs[1]
    se = np.sqrt(p_exc * (1 - p_exc) / len(X_pto))
    ax.fill_between(t, p_exc - 2 * se, p_exc + 2 * se, color=C["a"], alpha=0.3, lw=0)
    ax.plot(t, p_exc, color=C["a"])
    ax.fill_between(t, 0, 1, where=x_s > 0.5, color="0.85", lw=0, label="truth $> 0.5$")
    ax.set_ylim(0, 1.02); ax.set_xlabel("$t$"); ax.set_title("(b) $\\mathbb{P}(x(t) > 0.5 \\mid \\mathbf{y})$")
    ax.legend(frameon=False, loc="center", fontsize=6.3)
    ax = axs[2]
    ax.fill_between(t, lo, hi, color=C["a"], alpha=0.3, lw=0, label="95% band")
    ax.plot(t, mean_s - 1.96 * sd_s, "--", color="k", lw=0.6, label="closed form")
    ax.plot(t, mean_s + 1.96 * sd_s, "--", color="k", lw=0.6)
    for k, idx in enumerate([0, 1, 2]):
        ax.plot(t, X_pto[idx], color=[C["b"], C["c"], C["d"]][k], lw=0.6)
    ax.plot(t, x_s, color=C["truth"], lw=0.9, label="truth")
    ax.plot(t, X_pto.mean(0), color=C["rec"], lw=1.0, label="sample mean")
    ax.set_ylim(-0.35, 1.6); ax.set_xlabel("$t$"); ax.set_title("(c) mean, band, and samples")
    ax.legend(frameon=False, fontsize=5.6, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.4)
    save(fig, "pto.pdf")
    # truncated conjugate gradients instead of exact solves
    Hm = A.T @ A / sigma**2 + P_s
    rng = np.random.default_rng(14)
    Ns = 2000
    rhs = [A.T @ (y_s + sigma * rng.standard_normal(m)) / sigma**2 + B_s.T @ rng.standard_normal(n) for _ in range(Ns)]
    print(f"   truncated CG (condition number of the precision {np.linalg.cond(Hm):.1e}):")
    for k in (5, 10, 20, 40, 80):
        for start, x0 in (("zero", None), ("MAP", mean_s)):
            Xk = np.array([cg_solve(Hm, b, k, x0) for b in rhs])
            rat = Xk.std(0) / sd_s
            print(f"      k = {k:3d}, start {start}: max error of the mean {np.max(np.abs(Xk.mean(0) - mean_s) / sd_s):.2f} posterior std, "
                  f"std ratio median {np.median(rat):.2f}, min {rat.min():.2f}; truth inside the sample band at "
                  f"{100 * np.mean((x_s >= np.percentile(Xk, 2.5, 0)) & (x_s <= np.percentile(Xk, 97.5, 0))):.0f}%")


# ---------------------------------------------------------------- the idea of Gibbs sampling
def fig_gibbsidea():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), constrained_layout=True)
    ax = axs[0]
    banana_contours(ax)
    x2k = -1.2
    rng = np.random.default_rng(31)
    m1, s1 = mean_x1(x2k), 1 / np.sqrt(PREC_X1)
    x1n = m1 + s1 * 0.75                                             # a draw from p(x1 | x2), fixed for the picture
    g1 = np.linspace(-2.1, 2.0, 400)
    d1 = np.exp(-0.5 * ((g1 - m1) / s1) ** 2)
    ax.plot([-2.1, 2.0], [x2k, x2k], color=C["b"], lw=0.6, ls=":")
    ax.fill_between(g1, x2k, x2k + 0.75 * d1, color=C["b"], alpha=0.35, lw=0)
    ax.text(m1 + 0.55, x2k + 0.45, "$p(x_1 \\mid x_2, y)$", color=C["b"], fontsize=6.5)
    d2 = cond_x2(x1n)
    x2n = 1.05                                                        # a draw from the upper mode of p(x2 | x1)
    ax.plot([x1n, x1n], [-2.7, 2.7], color=C["c"], lw=0.6, ls=":")
    ax.fill_betweenx(GB, x1n, x1n + 0.6 * d2, color=C["c"], alpha=0.35, lw=0)
    ax.text(x1n + 0.62, 0.0, "$p(x_2 \\mid x_1, y)$", color=C["c"], fontsize=6.5, va="center")
    x0 = (m1 - 0.55, x2k)
    ax.plot(*x0, "s", color="k", ms=3.5); ax.text(x0[0] - 0.35, x0[1] - 0.35, "$\\mathbf{x}^{(k)}$", fontsize=7)
    ax.annotate("", xy=(x1n, x2k), xytext=x0, arrowprops=dict(arrowstyle="-|>", color=C["b"], lw=1.0, shrinkA=2, shrinkB=1))
    ax.annotate("", xy=(x1n, x2n), xytext=(x1n, x2k), arrowprops=dict(arrowstyle="-|>", color=C["c"], lw=1.0, shrinkA=1, shrinkB=2))
    ax.plot(x1n, x2n, "o", color="k", ms=3.5); ax.text(x1n - 0.8, x2n + 0.18, "$\\mathbf{x}^{(k+1)}$", fontsize=7)
    ax.set_xlim(-2.1, 2.0); ax.set_ylim(-2.7, 2.7); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("(a) one sweep: two exact one-dimensional draws")
    ax = axs[1]
    rho = 0.98
    g = np.linspace(-3, 3, 301); G1, G2 = np.meshgrid(g, g)
    Q = (G1**2 - 2 * rho * G1 * G2 + G2**2) / (1 - rho**2)
    ax.contour(G1, G2, Q, levels=[1, 4, 9], colors="0.7", linewidths=0.7)
    x = np.array([-2.0, -1.2]); path = [x.copy()]
    for k in range(25):
        x[0] = rho * x[1] + np.sqrt(1 - rho**2) * rng.standard_normal(); path.append(x.copy())
        x[1] = rho * x[0] + np.sqrt(1 - rho**2) * rng.standard_normal(); path.append(x.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "-", color=C["b"], lw=0.7)
    ax.plot(path[::2, 0], path[::2, 1], ".", color=C["b"], ms=3)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"(b) correlation {rho}: 25 sweeps, small steps")
    save(fig, "gibbsidea.pdf")


# ---------------------------------------------------------------- posteriors under the three priors on increments
def fig_threepriors():
    rng = np.random.default_rng(5)
    N_GIBBS, BURN = 20000, 2000
    t0 = time.time()
    P_g = gaussian_precision(n, GAMMA_PRIOR, "differences", gamma0=1.0)
    X_g = perturb_optimize(A, y_p, sigma, np.linalg.cholesky(P_g).T, N_GIBBS - BURN, rng)
    X_l, tau_l = gibbs_laplace(A, y_p, sigma, GAMMA_PRIOR, N_GIBBS, rng=rng)
    X_h, th_h = gibbs_hierarchical(A, y_p, sigma, BETA, VARTHETA, N_GIBBS, rng=rng)
    X_l, X_h, th_h = X_l[BURN:], X_h[BURN:], th_h[BURN:]
    print(f"three priors: {N_GIBBS - BURN} exact Gaussian samples and 2 x {N_GIBBS} Gibbs sweeps in {time.time() - t0:.0f}s")
    fig, axs = plt.subplots(3, 3, figsize=(W, 5.0), constrained_layout=True, sharex=True)
    names = [f"Gaussian, $\\gamma = {GAMMA_PRIOR:g}$", f"Laplace, $\\gamma = {GAMMA_PRIOR:g}$", "hierarchical, $\\vartheta = 10^{-3}$"]
    rng_show = np.random.default_rng(7)
    for j, (X, name) in enumerate(zip([X_g, X_l, X_h], names)):
        mu = X.mean(0); lo, hi = np.percentile(X, [2.5, 97.5], axis=0)
        Z = X @ Ln.T
        loud = np.mean(np.abs(Z) > 0.1, axis=0)
        cnt = np.sum(np.abs(Z) > 0.1, axis=1)
        inside = (x_p >= lo) & (x_p <= hi)
        print(f"   {name}: error of the posterior mean {np.linalg.norm(mu - x_p):.3f}; truth inside the 95% band at {100 * np.mean(inside):.0f}% of the points "
              f"(misses at {np.nonzero(~inside)[0].tolist()}); band width {np.mean(hi - lo):.3f}, on the plateau {np.mean(hi[25:45] - lo[25:45]):.3f}, at the jumps "
              f"{np.round([hi[jj] - lo[jj] for jj in jumps], 2).tolist()}; increments above 0.1 per sample {cnt.mean():.1f}; total variation of a sample "
              f"{np.abs(Z).sum(1).mean():.2f} (truth {np.abs(Ln @ x_p).sum():.2f}); P(loud) plateau {loud[25:45].mean():.3f}, near the small step "
              f"max {loud[78:88].max():.3f}, at the large jumps {np.round([loud[jj - 2:jj + 3].max() for jj in jumps[[0, 1, 2, 4]]], 2).tolist()}; "
              f"median |increment| on the first plateau {np.median(np.abs(Z[:, 25:45])):.3f}; tau of x(0.6) {iat(X[:, i60]):.1f}")
        ax = axs[0, j]
        ax.fill_between(t, lo, hi, color=COLS[j], alpha=0.3, lw=0, label="95% band")
        ax.plot(t, x_p, color=C["truth"], lw=0.9, label="truth")
        ax.plot(t, mu, color=C["rec"], lw=1.0, label="posterior mean")
        ax.set_title(f"({'abc'[j]}) {name}"); ax.set_ylim(-0.3, 1.45)
        ax.text(0.03, 0.9, f"error of the mean {np.linalg.norm(mu - x_p):.2f}", transform=ax.transAxes, fontsize=6.5)
        ax = axs[1, j]
        for k, idx in enumerate(rng_show.choice(len(X), 3, replace=False)):
            ax.plot(t, X[idx], color=COLS[k], lw=0.7)
        ax.plot(t, x_p, color=C["truth"], lw=0.9)
        ax.set_title(f"({'def'[j]}) three posterior samples"); ax.set_ylim(-0.3, 1.3)
        ax = axs[2, j]
        for jj in jumps:
            ax.axvline(t[jj], color="0.8", lw=0.7)
        ax.bar(t[:-1], loud, width=1.0 / n, color=COLS[j])
        ax.set_ylim(0, 1); ax.set_xlabel("$t$")
        ax.set_title(f"({'ghi'[j]}) $\\mathbb{{P}}(|(\\mathbf{{L}}\\mathbf{{x}})_i| > 0.1 \\mid \\mathbf{{y}})$")
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(os.path.join(OUT, "threepriors.pdf"), bbox_inches="tight"); plt.close(fig); print("wrote threepriors.pdf")


# ---------------------------------------------------------------- the funnel: two increments, one datum of their sum
Y_F, S_F = 0.5, 0.05                                         # a rise of 0.5 in one of two cells, measured with noise 0.05


def theta_of_v(v):
    """Inverse-gamma quantile map for beta = 1: theta = F^{-1}(Phi(v)) with F(theta) = exp(-vartheta / theta)."""
    return -VARTHETA / log_ndtr(v)


def fig_funnel():
    rng = np.random.default_rng(1)
    N = 20000
    z, th = np.array([0.5, 0.0]), np.array([0.25, 5e-4]); one = np.ones(2)
    Zg, THg = np.empty((N, 2)), np.empty((N, 2))
    for k in range(N):
        Pm = np.diag(1 / th) + np.outer(one, one) / S_F**2               # z | theta, y is Gaussian
        z = np.linalg.solve(Pm, one * Y_F / S_F**2) + np.linalg.solve(np.linalg.cholesky(Pm).T, rng.standard_normal(2))
        th = (VARTHETA + 0.5 * z**2) / rng.gamma(BETA + 0.5, 1.0, size=2)
        Zg[k], THg[k] = z, th

    def loglik(w):                                                      # w = (u1, u2, v1, v2), standard normal under the prior
        zz = np.sqrt(theta_of_v(w[2:])) * w[:2]
        return -0.5 * (Y_F - zz.sum()) ** 2 / S_F**2
    Wn, ne = elliptical_slice(loglik, np.array([1.0, 0.0, 2.65, 0.0]), N, rng=np.random.default_rng(2))
    Wn = Wn[1:]; Ze = np.sqrt(theta_of_v(Wn[:, 2:])) * Wn[:, :2]

    def switches(z1):
        state = np.where(z1 > 0.4, 1, np.where(z1 < 0.1, 0, -1)); s_ = state[state >= 0]
        return int(np.sum(np.abs(np.diff(s_)) > 0))
    print(f"funnel toy: Gibbs: {switches(Zg[:, 0])} switches of the jump in {N} sweeps, tau of z1 {iat(Zg[:, 0]):.0f}, P(z1 > 0.25) {np.mean(Zg[:, 0] > 0.25):.2f}; "
          f"elliptical slice in normalized coordinates: {switches(Ze[:, 0])} switches, tau of z1 {iat(Ze[:, 0]):.0f}, {ne:.1f} likelihood evaluations per iteration")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.2), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25, 1]})
    ax = axs[0]
    zg = np.linspace(-0.8, 0.8, 401); lg = np.linspace(-5, 1, 401); Zm, Lm = np.meshgrid(zg, lg)
    thm = 10.0**Lm
    logd = -0.5 * Zm**2 / thm - 0.5 * np.log(thm) - (BETA + 1) * np.log(thm) - VARTHETA / thm + np.log(thm)   # density of (z, log theta)
    logd -= logd.max()
    ax.contour(Zm, Lm, logd, levels=[-12, -8, -5, -3, -1.5], colors="0.6", linewidths=0.6, linestyles="solid")
    kk = np.arange(0, N, 10)
    ax.plot(Zg[kk, 0], np.log10(THg[kk, 0]), ".", color=C["a"], ms=1.2, alpha=0.6)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$\\log_{10}\\theta_1$"); ax.set_title("(a) the funnel and Gibbs samples")
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-5, 1)
    ax = axs[1]
    kk = np.arange(0, 4000)
    ax.plot(kk, Zg[kk, 0], color=C["a"], lw=0.5)
    ax.set_xlabel("sweep"); ax.set_ylabel("$z_1$"); ax.set_ylim(-0.15, 0.7)
    ax.set_title(f"(b) Gibbs: where is the jump? ($\\tau \\approx {iat(Zg[:, 0]):.0f}$)")
    ax.text(3950, 0.6, "cell 1", ha="right", fontsize=6.5); ax.text(3950, -0.11, "cell 2", ha="right", fontsize=6.5)
    ax = axs[2]
    g = np.linspace(-4.0, 4.0, 201); U1, V1 = np.meshgrid(g, g)
    ax.contour(U1, V1, U1**2 + V1**2, levels=[1, 4, 9], colors="0.7", linewidths=0.7)
    kk = np.arange(0, N, 10)
    ax.plot(Wn[kk, 0], Wn[kk, 2], ".", color=C["c"], ms=1.2, alpha=0.6)
    ax.set_aspect("equal"); ax.set_xlim(-4.0, 4.0); ax.set_ylim(-4.0, 4.0)
    ax.set_xlabel("$u_1$"); ax.set_ylabel("$v_1$"); ax.set_title("(c) normalized coordinates")
    save(fig, "funnel.pdf")


# ---------------------------------------------------------------- diagnostics of the hierarchical Gibbs sampler
def fig_diagnostics():
    rng = np.random.default_rng(11)
    starts = [None, 1.0, 1e-6, 1e-2]                              # theta0 of the four chains (None: hyper-prior mode)
    N_CH, BURN_CH = 6000, 1000
    chains_x, chains_th = [], []
    t0 = time.time()
    for th0 in starts:
        Xc, Tc = gibbs_hierarchical(A, y_p, sigma, BETA, VARTHETA, N_CH, theta0=th0, rng=rng)
        chains_x.append(Xc); chains_th.append(Tc)
    chains_x, chains_th = np.array(chains_x), np.array(chains_th)
    post_x, post_th = chains_x[:, BURN_CH:], chains_th[:, BURN_CH:]
    rh_x = np.array([rank_rhat(post_x[:, :, i]) for i in range(n)])
    rh_th = np.array([rank_rhat(np.log(post_th[:, :, i])) for i in range(n - 1)])
    ess_x = np.array([ess(post_x[:, :, i]) for i in range(n)])
    ess_th = np.array([ess(np.log(post_th[:, :, i])) for i in range(n - 1)])
    I_JUMP, I_PLAT = 18, 30
    print(f"diagnostics: 4 chains x {N_CH} sweeps ({time.time() - t0:.0f}s), burn-in {BURN_CH}; rank-Rhat of x: max {rh_x.max():.3f}; "
          f"of log theta: max {rh_th.max():.3f} (at increment {int(np.argmax(rh_th))}); ESS of x: min {ess_x.min():.0f}, median {np.median(ess_x):.0f}; "
          f"ESS of log theta: min {ess_th.min():.0f} (at {int(np.argmin(ess_th))}), median {np.median(ess_th):.0f} of {4 * (N_CH - BURN_CH)}; "
          f"tau of log theta at the jump {iat(np.log(post_th[0, :, I_JUMP])):.0f}, on the plateau {iat(np.log(post_th[0, :, I_PLAT])):.1f}")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
    ax = axs[0]
    kk = np.arange(0, 3000, 3)
    for c in range(2):
        ax.semilogy(kk, chains_th[c, kk, I_JUMP], color=COLS[c], lw=0.5, label=f"at the jump, chain {c + 1}")
    ax.semilogy(kk, chains_th[0, kk, I_PLAT], color="0.45", lw=0.5, label="on the plateau")
    ax.axvline(BURN_CH, color="k", lw=0.6, ls="--")
    ax.set_ylim(1e-5, 30)
    ax.set_xlabel("sweep"); ax.set_title("(a) traces of $\\theta_i$ (first 3000 sweeps)")
    ax = axs[1]
    lags = np.unique(np.logspace(0, 3, 40).astype(int))
    for series, col, lab in [(np.log(post_th[0, :, I_JUMP]), C["a"], "$\\log\\theta$, jump"), (np.log(post_th[0, :, I_PLAT]), "0.5", "$\\log\\theta$, plateau"),
                             (post_x[0, :, i60], C["rec"], "$x(0.6)$")]:
        ax.semilogx(lags, autocorrelation(series)[lags], color=col, label=lab)
    ax.axhline(0, color="0.6", lw=0.6); ax.set_xlabel("lag"); ax.set_title("(b) autocorrelation"); ax.legend(frameon=False, fontsize=6, loc="center right")
    ax = axs[2]
    ax.plot(t, rh_x, ".", color=C["rec"], ms=3, label="$x_i$")
    ax.plot(t[:-1], rh_th, ".", color=C["a"], ms=3, label="$\\log\\theta_i$")
    ax.axhline(1.01, color="k", lw=0.7, ls="--")
    ax.set_xlabel("$t$"); ax.set_title("(c) $\\widehat{R}$ of all components"); ax.legend(frameon=False, fontsize=6)
    save(fig, "diagnostics.pdf")


# ---------------------------------------------------------------- the idea of HMC: a sled on the landscape, and phase space
def leapfrog_1d(x, r, h, L):
    """L leapfrog steps (4.14) for the one-dimensional potential U1d with M = 1; returns all positions and momenta."""
    xs, rs = [x], [r]
    for _ in range(L):
        r_half = r - 0.5 * h * dU1d(x)
        x = x + h * r_half
        r = r_half - 0.5 * h * dU1d(x)
        xs.append(x); rs.append(r)
    return np.array(xs), np.array(rs)


def fig_hmcidea():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.4), constrained_layout=True)
    x = X1D; U = U1d(x)
    ax = axs[0]
    ax.plot(x, U, color="k", lw=1.0)
    x0, r0 = -1.0, 2.1
    xs, rs = leapfrog_1d(x0, r0, 0.05, 80)
    Hval = U1d(x0) + 0.5 * r0**2
    ax.axhline(Hval, color="0.5", lw=0.7, ls="--")
    ax.text(-3.4, Hval + 0.1, "total energy $H$", fontsize=6.5, color="0.4")
    cm = plt.get_cmap("viridis")
    for i in range(0, len(xs), 4):
        ax.plot(xs[i], U1d(xs[i]) + 0.12, "o", color=cm(i / len(xs)), ms=3.2)
    ax.annotate("", xy=(x0 + 0.6, U1d(x0) + 0.45), xytext=(x0, U1d(x0) + 0.45), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9))
    ax.text(x0 - 0.1, U1d(x0) + 0.62, "push", fontsize=6.5)
    ax.set_xlim(-3.5, 4.2); ax.set_ylim(0.3, 7.0); ax.set_xlabel("$x$"); ax.set_ylabel("$U(x)$")
    ax.set_title("(a) the sled: one trajectory, colored by time")
    ax = axs[1]
    g = np.linspace(-3.5, 4.2, 400); rg = np.linspace(-3.2, 3.2, 300); Xm, Rm = np.meshgrid(g, rg)
    Hm = U1d(Xm) + 0.5 * Rm**2
    ax.contour(Xm, Rm, Hm, levels=np.arange(1.0, 7.0, 0.6), colors="0.8", linewidths=0.6)
    rng = np.random.default_rng(41)
    xk = -1.0; r_end = 0.0
    for it, r_new in enumerate([0.9, -1.1, 2.3, -0.7, 1.2, -1.3]):     # six kicks; the third is large enough to cross the barrier
        ax.plot([xk, xk], [r_end, r_new], "--", color=C["rec"], lw=0.8)
        xs, rs = leapfrog_1d(xk, r_new, 0.12, 12)
        ax.plot(xs, rs, "-", color=C["a"], lw=0.8); ax.plot(xs, rs, ".", color=C["a"], ms=2.2)
        H0 = U1d(xk) + 0.5 * r_new**2; H1 = U1d(xs[-1]) + 0.5 * rs[-1] ** 2
        if np.log(rng.random()) < H0 - H1:
            xk, r_end = xs[-1], rs[-1]
        else:
            r_end = r_new
        ax.plot(xk, -3.0, "|", color="k", ms=7)
    ax.set_xlim(-3.5, 4.2); ax.set_ylim(-3.2, 3.2); ax.set_xlabel("position $x$"); ax.set_ylabel("momentum $r$")
    ax.set_title("(b) phase space: kicks (red) and flows (blue)")
    save(fig, "hmcidea.pdf")
    print(f"hmc idea: barrier U(0.4) - U(-1) = {U1d(0.4) - U1d(-1.0):.2f}; push r0 = {r0} gives kinetic energy {0.5 * r0**2:.2f}")



# ---------------------------------------------------------------- the idea of elliptical slice sampling, and the choice of the reference
def ess_step_record(log_lik, w, rng):
    """One iteration of Algorithm 4.17 that records the ellipse, the threshold, all proposals, and the brackets."""
    nu = rng.standard_normal(w.size)
    thr = log_lik(w) + np.log(rng.random())
    phi = rng.uniform(0.0, 2.0 * np.pi); lo, hi = phi - 2.0 * np.pi, phi
    props, brackets = [], [(lo, hi)]
    while True:
        wp = w * np.cos(phi) + nu * np.sin(phi); llp = log_lik(wp)
        acc = llp > thr
        props.append((phi, wp, llp, acc))
        if acc:
            return nu, thr, props, brackets
        if phi < 0.0:
            lo = phi
        else:
            hi = phi
        brackets.append((lo, hi))
        phi = rng.uniform(lo, hi)


def fig_essidea():
    loglik = lambda w: -0.5 * (Y_B - w[..., 0] - 0.5 * w[..., 1] ** 2) ** 2 / SIG_B**2     # the prior of the bent ridge is N(0, I)
    w = np.array([0.3, 1.2])
    ph_all = np.linspace(-np.pi, np.pi, 721)
    for seed in range(500):                                     # an iteration with a few rejections and two arcs, for the picture
        nu, thr, props, brackets = ess_step_record(loglik, w, np.random.default_rng(seed))
        ins = loglik(np.outer(np.cos(ph_all), w) + np.outer(np.sin(ph_all), nu)) > thr
        n_arcs = int(np.sum(np.diff(ins.astype(int)) == 1))
        if 3 <= len(props) <= 5 and abs(props[-1][0]) > 0.4 and thr < loglik(w) - 1.0 and np.linalg.norm(nu) > 1.3 and n_arcs >= 2:
            break
    print(f"ess idea: seed {seed}, {len(props)} proposals, accepted angle {props[-1][0]:.2f}, threshold {thr:.2f}, log L(w) = {loglik(w):.2f}")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.55), constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axs[0]
    banana_contours(ax)
    ph = np.linspace(-np.pi, np.pi, 721)
    E = np.outer(np.cos(ph), w) + np.outer(np.sin(ph), nu)
    inside = loglik(E) > thr
    ax.plot(E[:, 0], E[:, 1], "--", color="0.55", lw=0.7)
    Ein = np.where(inside[:, None], E, np.nan)
    ax.plot(Ein[:, 0], Ein[:, 1], "-", color=C["a"], lw=2.2, alpha=0.6)
    ax.plot(*w, "s", color="k", ms=4); ax.text(w[0] + 0.08, w[1] - 0.25, "$\\mathbf{w}$", fontsize=7)
    ax.plot(*nu, "^", color=C["d"], ms=4.5); ax.text(nu[0] + 0.08, nu[1] + 0.08, "$\\boldsymbol{\\nu}$", fontsize=7, color=C["d"])
    for i, (phi, wp, llp, acc) in enumerate(props):
        ax.plot(*wp, "o" if acc else "x", color=C["a"] if acc else "0.35", ms=4)
        ax.text(wp[0] + 0.07, wp[1] + 0.07, str(i + 1), fontsize=6.5, color=C["a"] if acc else "0.35")
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title("(a) the ellipse through $\\mathbf{w}$ and $\\boldsymbol{\\nu}$")
    ax = axs[1]
    lo0, hi0 = brackets[0]
    phs = np.linspace(lo0, hi0, 1200)
    Es = np.outer(np.cos(phs), w) + np.outer(np.sin(phs), nu)
    ll = loglik(Es)
    ax.plot(phs, ll, color="k", lw=0.9)
    ax.axhline(thr, color=C["a"], lw=0.8, ls="--"); ax.text(lo0 + 0.05, thr + 0.6, "threshold $\\ell$", color=C["a"], fontsize=6.5)
    ax.fill_between(phs, -40, 2, where=ll > thr, color=C["a"], alpha=0.12, lw=0)
    base = -26
    for i, (lo, hi) in enumerate(brackets):
        yb = base - 3.0 * i
        ax.plot([lo, hi], [yb, yb], "-", color="0.4", lw=1.0); ax.plot([lo, hi], [yb, yb], "|", color="0.4", ms=5)
        phi, wp, llp, acc = props[i]
        ax.plot(phi, yb, "o" if acc else "x", color=C["a"] if acc else "0.35", ms=4)
        ax.plot([phi, phi], [yb, max(llp, -24.5)], ":", color="0.6", lw=0.6)
        ax.text(phi + 0.08, yb + 0.6, str(i + 1), fontsize=6.5, color=C["a"] if acc else "0.35")
    ax.axvline(0, color="k", lw=0.5, ls=":"); ax.text(0.06, -21.5, "current\nstate", fontsize=6.5, va="top")
    ax.set_ylim(base - 3.0 * len(brackets), 2); ax.set_xlim(lo0 - 0.15, hi0 + 0.35)
    ax.set_xlabel("angle $\\varphi$"); ax.set_ylabel("$\\log\\mathcal{L}$ on the ellipse")
    ax.set_title("(b) log-likelihood along the ellipse, and the brackets")
    save(fig, "essidea.pdf")


def fig_essref():
    SIG = 0.05; NLIN = 0.3

    def make(nd, seed):
        rng = np.random.default_rng(seed)
        xt = np.array([0.8, -0.5]) if nd == 2 else rng.standard_normal(nd)
        Fm = lambda x: x + NLIN * np.roll(x, -1, axis=-1) ** 2
        def Jm(x):
            J = np.eye(nd); J[np.arange(nd), (np.arange(nd) + 1) % nd] += 2 * NLIN * np.roll(x, -1); return J
        y = Fm(xt) + SIG * rng.standard_normal(nd)
        U = lambda x: 0.5 * np.sum((Fm(x) - y) ** 2, axis=-1) / SIG**2 + 0.5 * np.sum(x**2, axis=-1)
        x = np.zeros(nd)
        for _ in range(100):                                    # MAP by Gauss-Newton
            J = Jm(x); x = x - np.linalg.solve(J.T @ J / SIG**2 + np.eye(nd), J.T @ (Fm(x) - y) / SIG**2 + x)
        J = Jm(x); Hm = J.T @ J / SIG**2 + np.eye(nd)
        return Fm, y, U, x, np.linalg.cholesky(np.linalg.inv(Hm))
    Fm, y2, U2, m2, C2 = make(2, 3)
    llp = lambda w: -0.5 * np.sum((Fm(w) - y2) ** 2, axis=-1) / SIG**2              # prior N(0, I) as reference
    lll = lambda w: -U2(m2 + w @ C2.T) + 0.5 * np.sum(w**2, axis=-1)                # Laplace approximation as reference
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.3), constrained_layout=True)
    g = np.linspace(-2.7, 2.7, 541); G1, G2 = np.meshgrid(g, g); PP = np.stack([G1, G2], -1)
    ax = axs[0]
    ax.contour(G1, G2, G1**2 + G2**2, levels=[1, 4], colors="0.75", linewidths=0.7)
    Ug = U2(PP); ax.contour(G1, G2, Ug - Ug.min(), levels=[0.5, 2.0], colors=C["a"], linewidths=0.8)
    w0 = m2 + np.array([0.03, 0.02])
    nu, thr, props, br = ess_step_record(llp, w0, np.random.default_rng(5))
    ph = np.linspace(-np.pi, np.pi, 721); E = np.outer(np.cos(ph), w0) + np.outer(np.sin(ph), nu)
    ax.plot(E[:, 0], E[:, 1], "--", color="0.55", lw=0.7)
    ax.plot(*w0, "s", color="k", ms=3.5)
    for i, (phi, wp, llv, acc) in enumerate(props):
        ax.plot(*wp, "o" if acc else "x", color=C["a"] if acc else "0.35", ms=3.0 if acc else 3.5)
    ax.set_aspect("equal"); ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.7, 2.7); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"(a) prior reference: {len(props)} proposals")
    ax = axs[1]
    gz = np.linspace(-0.22, 0.22, 401); Z1, Z2 = np.meshgrid(m2[0] + gz, m2[1] + gz); PZ = np.stack([Z1, Z2], -1)
    Uz = U2(PZ); ax.contour(Z1, Z2, Uz - Uz.min(), levels=[0.5, 2.0], colors=C["a"], linewidths=0.8)
    Hinv = C2 @ C2.T; D = PZ - m2
    Q = np.einsum("...i,ij,...j->...", D, np.linalg.inv(Hinv), D)
    ax.contour(Z1, Z2, 0.5 * Q, levels=[0.5, 2.0], colors=C["b"], linewidths=0.8, linestyles="dashed")
    v0 = np.linalg.solve(C2, w0 - m2)
    nu2, thr2, props2, br2 = ess_step_record(lll, v0, np.random.default_rng(6))
    E2 = m2 + (np.outer(np.cos(ph), v0) + np.outer(np.sin(ph), nu2)) @ C2.T
    ax.plot(E2[:, 0], E2[:, 1], "--", color="0.55", lw=0.7)
    ax.plot(E[:, 0], E[:, 1], "-", color="0.8", lw=0.7)
    for i, (phi, vp, llv, acc) in enumerate(props2):
        xp = m2 + C2 @ vp
        ax.plot(*xp, "o" if acc else "x", color=C["a"] if acc else "0.35", ms=3.5)
    ax.plot(*w0, "s", color="k", ms=3.5)
    ax.set_aspect("equal"); ax.set_xlim(m2[0] - 0.22, m2[0] + 0.22); ax.set_ylim(m2[1] - 0.22, m2[1] + 0.22)
    ax.set_xlabel("$x_1$"); ax.set_title(f"(b) Laplace reference: {len(props2)} proposal{'s' if len(props2) > 1 else ''}")
    print(f"ess reference: 2-D toy MAP {m2.round(3)}, posterior std {np.sqrt(np.diag(Hinv)).round(3)}; one iteration needs {len(props)} proposals "
          f"with the prior and {len(props2)} with the Laplace approximation as reference")
    ax = axs[2]
    ns = [2, 4, 8, 16, 32, 64]
    cost_p, cost_l = [], []
    for nd in ns:
        cp_, cl_ = [], []
        for seed in (nd, nd + 100):
            Fm_, y_, U_, m_, C_ = make(nd, seed)
            Nit = 20000 if nd <= 16 else 40000
            chp, nep = elliptical_slice(lambda w: -0.5 * np.sum((Fm_(w) - y_) ** 2) / SIG**2, m_.copy(), Nit, rng=np.random.default_rng(1))
            chl, nel = elliptical_slice(lambda w: -U_(m_ + C_ @ w) + 0.5 * w @ w, np.zeros(nd), Nit, rng=np.random.default_rng(2))
            Xl = m_ + chl @ C_.T
            cp_.append((Nit - 1000) * nep / min(ess(chp[1000:, i]) for i in range(nd)))
            cl_.append((Nit - 1000) * nel / min(ess(Xl[1000:, i]) for i in range(nd)))
        cost_p.append(np.exp(np.mean(np.log(cp_)))); cost_l.append(np.exp(np.mean(np.log(cl_))))
        print(f"   n = {nd:3d}: forward solves per effective sample, prior as reference {cost_p[-1]:.0f}, Laplace approximation as reference {cost_l[-1]:.1f}")
    ax.loglog(ns, cost_p, "o-", color=C["d"], ms=3.5, label="prior as reference")
    ax.loglog(ns, cost_l, "s-", color=C["c"], ms=3.5, label="Laplace as reference")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("number of unknowns $n$"); ax.set_ylabel("forward solves")
    ax.set_title("(c) cost per effective sample"); ax.legend(frameon=False, fontsize=6)
    save(fig, "essref.pdf")


# ====================================================================== HMC, elliptical slice sampling, and priors for the bathymetry problem
def fig_hmc():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import swe_jax
    xg, dx = make_grid(200)
    b_true = bottom_smooth(xg)
    MM, SIG_W, GAMMA_B = 20, 0.01, 0.03
    xo = np.linspace(0.5, LEN - 0.5, MM)
    fwd = swe_jax.make_forward(xg, xo)
    y_b = np.asarray(fwd(jnp.asarray(b_true[:-1]))) + SIG_W * np.random.default_rng(4).standard_normal(MM)
    ybj = jnp.asarray(y_b)
    jac = jax.jit(jax.jacfwd(fwd))
    F = lambda b: np.asarray(fwd(jnp.asarray(b)))
    Jf = lambda b: np.asarray(jac(jnp.asarray(b)))
    L1 = difference_matrix(200)[:, :-1]                     # increments of (b, 0): the outflow bottom b_200 = 0 is known
    D2 = np.diff(np.eye(200), n=2, axis=0)[:, :-1]            # second differences of (b, 0)
    i9 = int(np.argmin(np.abs(xg[:-1] - 9.0)))

    def posterior(Lw):
        """Weighted prior matrix Lw (prior N(0, (Lw^T Lw)^{-1})): potential, MAP, Gauss-Newton Hessian."""
        Lwj = jnp.asarray(Lw)

        def U(b):
            r = fwd(b) - ybj
            z = Lwj @ b
            return 0.5 * r @ r / SIG_W**2 + 0.5 * z @ z
        b_map, hist = gauss_newton(F, Jf, y_b, np.zeros(199), lam=SIG_W**2, L=Lw, n_iter=60)
        Jm = Jf(b_map)
        Hm = Jm.T @ Jm / SIG_W**2 + Lw.T @ Lw
        return U, b_map, len(hist) - 1, Hm

    def run_hmc(U, b_map, Hm, n_iter=3000, burn=500):
        Cm = np.linalg.cholesky(np.linalg.inv(Hm))                 # mass matrix M = H
        step = make_hmc_jax(U, 0.3, 5, Cm)
        key = jax.random.PRNGKey(0); xb = jnp.asarray(b_map); xs, accs = [], 0
        t0 = time.time()
        for k in range(n_iter):
            key, sub = jax.random.split(key)
            xb, a, dH = step(sub, xb)
            accs += bool(a); xs.append(np.asarray(xb))
        return np.array(xs)[burn:], accs / n_iter, time.time() - t0, Cm

    # --- first differences, gamma = 3 cm (Examples 3.12 and 2.9)
    Lw1 = L1 / GAMMA_B
    U_b, b_map, n_gn, Hm = posterior(Lw1)
    sd_lap = np.sqrt(np.diag(np.linalg.inv(Hm)))
    Xb, acc_hmc, t_hmc, Cm = run_hmc(U_b, b_map, Hm)
    ess_hmc = min(ess(Xb[:, i]) for i in range(199))
    U_np = jax.jit(U_b)
    lp_b = lambda b: -float(U_np(jnp.asarray(b)))
    t0 = time.time()
    ch_rw, acc_rw = rw_metropolis(lp_b, b_map.copy(), 20000, 2.38 / np.sqrt(199), chol=Cm, rng=np.random.default_rng(8))
    t_rw = time.time() - t0
    Xrw = ch_rw[2000:]
    ess_rw = min(ess(Xrw[:, i]) for i in range(199))
    # cost of a gradient relative to a forward evaluation, timed inside jitted loops (no Python overhead)
    loop_U = jax.jit(lambda b, k: jax.lax.fori_loop(0, k, lambda i, a: a + U_b(b + 1e-12 * a), 0.0), static_argnums=1)
    loop_G = jax.jit(lambda b, k: jax.lax.fori_loop(0, k, lambda i, a: a + jnp.sum(jax.grad(U_b)(b + 1e-12 * a)), 0.0),
                     static_argnums=1)
    bj = jnp.asarray(b_map)
    loop_U(bj, 2000).block_until_ready(); loop_G(bj, 2000).block_until_ready()
    t_fs, t_gs = [], []
    for _ in range(5):
        t0 = time.time(); loop_U(bj, 2000).block_until_ready(); t_fs.append((time.time() - t0) / 2000)
        t0 = time.time(); loop_G(bj, 2000).block_until_ready(); t_gs.append((time.time() - t0) / 2000)
    t_f, t_g = np.median(t_fs), np.median(t_gs)
    lo, hi = np.percentile(Xb, [2.5, 97.5], axis=0)
    print(f"bathymetry: MAP after {n_gn} Gauss-Newton iterations, misfit {np.linalg.norm(F(b_map) - y_b):.4f} (noise level {np.sqrt(MM) * SIG_W:.4f}), "
          f"rms error {100 * np.sqrt(np.mean((b_map - b_true[:-1]) ** 2)):.2f} cm; "
          f"HMC eps 0.3, 5 leapfrog steps: acceptance {acc_hmc:.2f}, tau of b(9 m) {iat(Xb[:, i9]):.1f}, min ESS {ess_hmc:.0f} of {len(Xb)} "
          f"({len(Xb) * 5 / ess_hmc:.1f} gradients per effective sample, {t_hmc:.0f}s); "
          f"RW-MH with the same preconditioner: acceptance {acc_rw:.2f}, tau of b(9 m) {iat(Xrw[:, i9]):.0f}, min ESS {ess_rw:.0f} of 18000 "
          f"({18000 / ess_rw:.0f} forward solves per effective sample, {t_rw:.0f}s); a gradient costs {t_g / t_f:.1f} forward evaluations; "
          f"posterior std / Laplace std between {np.min(Xb.std(0) / sd_lap):.2f} and {np.max(Xb.std(0) / sd_lap):.2f}; "
          f"posterior std between {100 * Xb.std(0).min():.1f} and {100 * Xb.std(0).max():.1f} cm; max |mean - MAP| {100 * np.max(np.abs(Xb.mean(0) - b_map)):.2f} cm; "
          f"truth inside the band at {100 * np.mean((b_true[:-1] >= lo) & (b_true[:-1] <= hi)):.0f}% of the cells")
    # elliptical slice sampling with the prior as reference (whitened: b = Lw^{-1} w) and with the Laplace approximation as reference
    Linv = jnp.asarray(np.linalg.inv(Lw1))
    llp = jax.jit(lambda w: -0.5 * jnp.sum((fwd(Linv @ w) - ybj) ** 2) / SIG_W**2)
    t0 = time.time()
    chp, nep = elliptical_slice(lambda w: float(llp(jnp.asarray(w))), Lw1 @ b_map, 20000, rng=np.random.default_rng(9))
    t_essp = time.time() - t0
    Xep = np.asarray((Linv @ jnp.asarray(chp[2000:].T)).T)
    ess_ep = min(ess(Xep[:, i]) for i in range(199))
    Cj, mj = jnp.asarray(Cm), jnp.asarray(b_map)
    lll = jax.jit(lambda w: -U_b(mj + Cj @ w) + 0.5 * w @ w)        # posterior / reference, reference N(b_MAP, H^{-1})
    t0 = time.time()
    chl, nel = elliptical_slice(lambda w: float(lll(jnp.asarray(w))), np.zeros(199), 6000, rng=np.random.default_rng(10))
    t_essl = time.time() - t0
    Xel = b_map + chl[500:] @ Cm.T
    ess_el = min(ess(Xel[:, i]) for i in range(199))
    print(f"   elliptical slice, prior as reference: {nep:.1f} likelihood evaluations per iteration, tau of b(9 m) {iat(Xep[:, i9]):.0f}, min ESS {ess_ep:.0f} of {len(Xep)} "
          f"({len(Xep) * nep / ess_ep:.0f} forward solves per effective sample, {t_essp:.0f}s)")
    print(f"   elliptical slice, Laplace approximation as reference: {nel:.2f} evaluations per iteration, tau of b(9 m) {iat(Xel[:, i9]):.2f}, min ESS {ess_el:.0f} of {len(Xel)} "
          f"({len(Xel) * nel / ess_el:.1f} forward solves per effective sample, {t_essl:.0f}s); std ratio to HMC {np.min(Xel.std(0) / Xb.std(0)):.2f}-{np.max(Xel.std(0) / Xb.std(0)):.2f}")
    try:
        import blackjax
        logdensity = lambda b: -U_b(b)
        for diag in (True, False):
            warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, is_mass_matrix_diagonal=diag)
            (state, params), _ = warmup.run(jax.random.PRNGKey(1), jnp.asarray(b_map), num_steps=500)
            kernel = jax.jit(blackjax.nuts(logdensity, **params).step)

            def one(state, k):
                state, info = kernel(k, state)
                return state, (state.position, info.num_integration_steps, info.is_divergent)
            _, (pos, nst, div) = jax.lax.scan(one, state, jax.random.split(jax.random.PRNGKey(2), 2000))
            pos, nst = np.asarray(pos), np.asarray(nst)
            ess_nuts = min(ess(pos[:, i]) for i in range(199))
            print(f"   NUTS (BlackJAX, window adaptation, {'diagonal' if diag else 'dense'} mass matrix): {np.mean(nst):.0f} leapfrog steps per iteration, "
                  f"{int(np.sum(div))} divergences, min ESS {ess_nuts:.0f} of 2000, {np.sum(nst) / ess_nuts:.0f} gradients per effective sample")
    except ImportError:
        print("   blackjax not available: NUTS cross-check skipped")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
    ax = axs[0]
    ax.fill_between(xg[:-1], lo, hi, color=C["a"], alpha=0.3, lw=0, label="95% band")
    ax.plot(xg, b_true, color=C["truth"], lw=0.9, label="truth")
    ax.plot(xg[:-1], Xb.mean(0), color=C["rec"], lw=1.0, label="posterior mean")
    ax.plot(xo, np.full(MM, -0.105), "|", color="k", ms=3)
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$b$ [m]"); ax.set_ylim(-0.12, 0.32)
    ax.set_title("(a) posterior from HMC"); ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax = axs[1]
    ax.plot(xg[:-1], 100 * Xb.std(0), color=C["a"], label="HMC")
    ax.plot(xg[:-1], 100 * Xel.std(0), ":", color=C["c"], lw=1.2, label="elliptical slice")
    ax.plot(xg[:-1], 100 * sd_lap, "--", color="k", lw=0.8, label="linearized at the MAP")
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("posterior std [cm]"); ax.set_ylim(0, 8.5)
    ax.set_title("(b) posterior standard deviation"); ax.legend(frameon=False, fontsize=5.8, loc="upper right", ncol=1)
    ax = axs[2]
    lags = np.unique(np.logspace(0, 3.6, 50).astype(int))
    for X_, col, lab in [(Xb, C["a"], "HMC"), (Xel, C["c"], "ell. slice, Laplace"), (Xrw, C["b"], "random walk"), (Xep, C["d"], "ell. slice, prior")]:
        ll_ = lags[lags < len(X_) // 2]
        ax.semilogx(ll_, autocorrelation(X_[:, i9])[ll_], "-", color=col, lw=0.9, label=lab)
    ax.axhline(0, color="0.6", lw=0.6); ax.set_xlabel("lag"); ax.set_title("(c) autocorrelation of $b(9\\,\\mathrm{m})$")
    ax.legend(frameon=False, fontsize=5.6, loc="center left", bbox_to_anchor=(0.0, 0.52), handlelength=1.4)
    save(fig, "hmc.pdf")

    # --- three priors for the same data: what makes the posterior mean look right
    cases = [("(a) increments, $\\gamma = 3$ cm", L1 / 0.03), ("(b) increments, $\\gamma = 1$ cm", L1 / 0.01),
             ("(c) second differences, $\\gamma_2 = 0.2$ cm", np.vstack([D2 / 2e-3, L1[-1:] / 0.03]))]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True, sharey=True)
    for ax, (title, Lw) in zip(axs, cases):
        U_c, bm, ngn, Hc = posterior(Lw)
        Xc, accc, tc, _ = run_hmc(U_c, bm, Hc)
        loc, hic = np.percentile(Xc, [2.5, 97.5], axis=0); mc = Xc.mean(0)
        print(f"   prior {title}: misfit of the MAP {np.linalg.norm(F(bm) - y_b):.4f}, rms error of the mean {100 * np.sqrt(np.mean((mc - b_true[:-1]) ** 2)):.2f} cm, "
              f"max error {100 * np.max(np.abs(mc - b_true[:-1])):.1f} cm, peak of the mean {mc[(xg[:-1] > 6) & (xg[:-1] < 12)].max():.3f} (truth 0.200), "
              f"truth inside the band at {100 * np.mean((b_true[:-1] >= loc) & (b_true[:-1] <= hic)):.0f}%, std {100 * Xc.std(0).min():.1f}-{100 * Xc.std(0).max():.1f} cm, "
              f"rms second difference of the mean {np.sqrt(np.mean(np.diff(np.r_[mc, 0.0], 2) ** 2)):.1e} (truth {np.sqrt(np.mean(np.diff(b_true, 2) ** 2)):.1e}), "
              f"HMC acceptance {accc:.2f}, tau of b(9 m) {iat(Xc[:, i9]):.1f}")
        ax.fill_between(xg[:-1], loc, hic, color=C["a"], alpha=0.3, lw=0)
        ax.plot(xg, b_true, color=C["truth"], lw=0.9)
        ax.plot(xg[:-1], mc, color=C["rec"], lw=1.0)
        ax.plot(xo, np.full(MM, -0.105), "|", color="k", ms=3)
        ax.set_xlabel("$x$ [m]"); ax.set_title(title); ax.set_ylim(-0.12, 0.32)
    axs[0].set_ylabel("$b$ [m]")
    save(fig, "bathyprior.pdf")


FIGURES = {"walkers": fig_walkers, "mcmcidea": fig_mcmcidea, "mhidea": fig_mhidea, "rwmh": fig_rwmh, "ptoidea": fig_ptoidea,
           "pto": fig_pto, "gibbsidea": fig_gibbsidea, "threepriors": fig_threepriors, "funnel": fig_funnel,
           "diagnostics": fig_diagnostics, "hmcidea": fig_hmcidea, "essidea": fig_essidea, "essref": fig_essref, "hmc": fig_hmc}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGURES):
        FIGURES[name]()

"""Figures of Chapter 3 (Bayesian inverse problems and MAP estimation). Output: ../figures/ch03/*.pdf

    python ch03_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from running_examples import (deconvolution_operator, signal_smooth, signal_piecewise, make_grid, bottom_smooth,
                              N_CELLS, M_OBS, NOISE_STD, L as LEN)
from solvers import difference_matrix, tikhonov, l1_regularization, gaussian_posterior, ias
from priors import (gaussian_precision, sample_gaussian, sample_laplace, sample_gsbl, sample_invgamma,
                    se_covariance, sample_gaussian_cov, cumsum_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "ch03")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "lines.linewidth": 1.1})
W = 6.3
C = {"truth": "k", "rec": "tab:red", "a": "tab:blue", "b": "tab:orange", "c": "tab:green", "d": "tab:purple"}
rng = np.random.default_rng(1)

n, m, sigma = N_CELLS, M_OBS, NOISE_STD
A, t, s = deconvolution_operator(n, m)
x_s, x_p = signal_smooth(t), signal_piecewise(t)
eps_s, eps_p = sigma * rng.standard_normal(m), sigma * rng.standard_normal(m)   # same realizations as Chapter 2
y_s, y_p = A @ x_s + eps_s, A @ x_p + eps_p
Ln = difference_matrix(n)


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), **kw)
    plt.close(fig)
    print("wrote", name)


def contour(ax, X1, X2, Z, color, ls="-", levels=(0.05, 0.3, 0.7), lw=0.9):
    Z = Z / Z.max()
    ax.contour(X1, X2, Z, levels=list(levels), colors=color, linestyles=ls, linewidths=lw)


# ---------------------------------------------------------------- Fig: prior x likelihood = posterior
from matplotlib.lines import Line2D
fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.1]})
ax = axs[0]
xx = np.linspace(-3.5, 3.5, 700)
prior = np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi)
y1, s1 = 1.5, 0.5
lik = np.exp(-(y1 - xx) ** 2 / (2 * s1**2))                    # as a function of x, not normalized
post_var = 1 / (1 + 1 / s1**2); post_mean = post_var * y1 / s1**2
post = np.exp(-(xx - post_mean) ** 2 / (2 * post_var)) / np.sqrt(2 * np.pi * post_var)
ax.plot(xx, prior, color="0.5")
ax.plot(xx, lik * prior.max() / lik.max(), color=C["b"], ls="--")
ax.plot(xx, post, color=C["a"])
ax.axvline(y1, color=C["b"], lw=0.6, ls=":")
ax.text(y1 + 0.1, 0.62, "data $y$", color=C["b"], fontsize=7)
ax.plot(post_mean, 0, "*", color=C["rec"], ms=11, zorder=5, clip_on=False)
ax.plot(post_mean, 0, "o", color=C["c"], ms=4, zorder=6, clip_on=False)
ax.set_xlabel("$x$"); ax.set_yticks([]); ax.set_ylim(0, 0.95); ax.set_title("(a) one unknown, one datum")
ax = axs[1]
g1 = np.linspace(-3.6, 3.6, 361); g2 = np.linspace(-3, 3, 301)
X1, X2 = np.meshgrid(g1, g2)
y2, s2 = 1.0, 0.3
G = X1 + 0.5 * X2**2
lp = -(X1**2 + X2**2) / 2
ll = -(y2 - G) ** 2 / (2 * s2**2)
lpost = lp + ll
post2 = np.exp(lpost - lpost.max())
i_map = np.unravel_index(np.argmax(post2), post2.shape)
mean2 = (np.sum(X1 * post2) / np.sum(post2), np.sum(X2 * post2) / np.sum(post2))
contour(ax, X1, X2, np.exp(lp), "0.6")
contour(ax, X1, X2, np.exp(ll), C["b"], "--")
contour(ax, X1, X2, post2, C["a"])
ax.plot(X1[i_map], X2[i_map], "*", color=C["rec"], ms=11, zorder=5)
ax.plot(*mean2, "o", color=C["c"], ms=4, zorder=6)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("(b) two unknowns, one datum $y = x_1 + x_2^2/2 + \\varepsilon$")
handles = [Line2D([], [], color="0.5", label="prior $p(\\mathbf{x})$"),
           Line2D([], [], color=C["b"], ls="--", label="likelihood $p(\\mathbf{y} \\mid \\mathbf{x})$"),
           Line2D([], [], color=C["a"], label="posterior $p(\\mathbf{x} \\mid \\mathbf{y})$"),
           Line2D([], [], marker="*", color=C["rec"], ls="", ms=10, label="MAP estimate"),
           Line2D([], [], marker="o", color=C["c"], ls="", ms=4, label="posterior mean")]
fig.legend(handles=handles, frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.09), columnspacing=1.4, handlelength=1.6)
print(f"2-D cartoon: MAP ({X1[i_map]:.2f}, {X2[i_map]:.2f}), mean ({mean2[0]:.2f}, {mean2[1]:.2f}); 1-D posterior mean {post_mean:.2f}, std {np.sqrt(post_var):.2f}")
fig.savefig(os.path.join(OUT, "posterior.pdf"), bbox_inches="tight"); plt.close(fig); print("wrote posterior.pdf")


# ---------------------------------------------------------------- Fig: the likelihood as a function of x in 2-D
fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), constrained_layout=True)
g = np.linspace(-3, 3, 361)
X1, X2 = np.meshgrid(g, g)
x_true2 = np.array([1.0, 0.5])
rng2 = np.random.default_rng(7)
sig2 = 0.3
A2 = np.array([[1.0, 0.8], [0.8, 1.0]])                      # n = m = 2, condition number 9
y2v = A2 @ x_true2 + sig2 * rng2.standard_normal(2)
R = (A2[0, 0] * X1 + A2[0, 1] * X2 - y2v[0]) ** 2 + (A2[1, 0] * X1 + A2[1, 1] * X2 - y2v[1]) ** 2
ll2 = -R / (2 * sig2**2)
x_ml = np.linalg.solve(A2, y2v)
ax = axs[0]
contour(ax, X1, X2, np.exp(ll2 - ll2.max()), C["b"], "-", levels=(0.01, 0.1, 0.5, 0.9))
ax.plot(*x_true2, "k+", ms=8, mew=1.3, label="truth")
ax.plot(*x_ml, "s", color=C["b"], ms=5, label="maximum likelihood")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
ax.set_title("(a) $n = m = 2$, $\\kappa(\\mathbf{A}) = 9$")
ax.legend(frameon=False, loc="upper left")
ax = axs[1]
a1 = np.array([1.0, 1.0])                                    # n = 2, m = 1
y1v = a1 @ x_true2 + sig2 * rng2.standard_normal()
ll1 = -(a1[0] * X1 + a1[1] * X2 - y1v) ** 2 / (2 * sig2**2)
contour(ax, X1, X2, np.exp(ll1 - ll1.max()), C["b"], "-", levels=(0.01, 0.1, 0.5, 0.9))
ax.plot(*x_true2, "k+", ms=8, mew=1.3, label="truth")
ax.annotate("", xy=(2.0, -0.8), xytext=(0.6, 0.6), arrowprops=dict(arrowstyle="->", color="0.4"))
ax.annotate("", xy=(-0.8, 2.0), xytext=(0.6, 0.6), arrowprops=dict(arrowstyle="->", color="0.4"))
ax.text(1.5, 1.3, "null space of $\\mathbf{A}$", color="0.3", fontsize=7, rotation=-45, ha="center")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
ax.set_title("(b) $n = 2$, $m = 1$: $\\mathbf{A} = (1 \\;\\; 1)$")
ax.legend(frameon=False, loc="upper left")
print(f"likelihood figure: truth {x_true2}, ML estimate {x_ml.round(2)}, singular values of A2 {np.linalg.svd(A2, compute_uv=False).round(2)}")
save(fig, "likelihood.pdf")


# ---------------------------------------------------------------- Fig: samples from the priors, 2 x 3: columns Gaussian,
# Laplace, hierarchical; top row the i.i.d. draws z (the prior applied to x itself), bottom row the signal with
# increments z (the prior applied to L x), started at x_1 = 0.
cols = [C["a"], C["b"], C["c"]]
HIER_BETA, HIER_VT = 1.0, 1e-3                                  # the hierarchical prior used throughout Chapter 3


def hier_draws(r):
    theta = sample_invgamma(HIER_BETA, HIER_VT, (3, n - 1), r)
    return np.sqrt(theta) * r.standard_normal(theta.shape)


prior_columns = [("$\\mathcal{N}(0, \\gamma^2)$", "$\\gamma = 0.1$", lambda r: 0.1 * r.standard_normal((3, n - 1))),
                 ("$\\mathrm{Laplace}(0, \\gamma)$", "$\\gamma = 0.1$", lambda r: r.laplace(0.0, 0.1, (3, n - 1))),
                 ("$\\mathcal{N}(0, \\theta_i)$", "$\\vartheta = 10^{-3}$", hier_draws)]
fig, axs = plt.subplots(2, 3, figsize=(W, 3.4), constrained_layout=True)
for j, (label, par, draws) in enumerate(prior_columns):
    z = draws(np.random.default_rng(3))
    x = z @ cumsum_matrix(n).T
    for k in range(3):
        axs[0, j].plot(t[:-1], z[k], color=cols[k], lw=0.8)
        axs[1, j].plot(t, x[k], color=cols[k], lw=0.8)
    axs[0, j].set_title(f"({'abc'[j]}) $x_i \\sim$ {label}, {par}")
    axs[1, j].set_title(f"({'def'[j]}) $(\\mathbf{{L}}\\mathbf{{x}})_i \\sim$ {label}, {par}")
    axs[1, j].set_xlabel("$t$")
save(fig, "priorsamples.pdf")


# ---------------------------------------------------------------- Fig: how the hierarchical prior works (beta = 1, vartheta = 1e-3)
beta_h, vt_h = HIER_BETA, HIER_VT
from scipy.special import gammaln
fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
ax = axs[0]
th = np.logspace(-6, 1, 500)
logp = beta_h * np.log(vt_h) - gammaln(beta_h) - (beta_h + 1) * np.log(th) - vt_h / th
dens = np.exp(logp) * th * np.log(10)                           # density with respect to log10(theta)
ax.semilogx(th, dens / dens.max(), "k")
theta_pick = np.array([2e-4, 1e-3, 2e-1])
for th_i, col in zip(theta_pick, cols):
    ax.axvline(th_i, color=col, lw=1.0, ls="--")
ax.set_xlabel("$\\theta$"); ax.set_yticks([])
ax.set_title("(a) hyper-prior density, three draws")
ax = axs[1]
ZMAX = 1.9
zz = np.linspace(-ZMAX, ZMAX, 2401)


def sci(v):
    mant, e = f"{v:.0e}".split("e")
    return f"${mant}\\cdot 10^{{{int(e)}}}$"


rng_b = np.random.default_rng(5)
for k, (th_i, col) in enumerate(zip(theta_pick, cols)):
    zs = np.sqrt(th_i) * rng_b.standard_normal(40)
    ax.plot(zz, k + 0.8 * np.exp(-zz**2 / (2 * th_i)), color=col, lw=0.8)      # normalized conditional density
    ax.plot(zs, k + 0.12 * (rng_b.random(40) - 0.5), ".", color=col, ms=3, alpha=0.8)
    ax.text(-0.97 * ZMAX, k + 0.45, f"$\\theta = $ {sci(th_i)}", color=col, fontsize=6.5)
ax.set_xlim(-ZMAX, ZMAX); ax.set_ylim(-0.25, 3.0); ax.set_yticks([])
ax.set_xlabel("$z$")
ax.set_title("(b) $\\mathcal{N}(z; 0, \\theta)$ and 40 draws for each $\\theta$")
rng_h = np.random.default_rng(175)
theta_i = sample_invgamma(beta_h, vt_h, n - 1, rng_h)
z_i = np.sqrt(theta_i) * rng_h.standard_normal(n - 1)
x_i = cumsum_matrix(n) @ z_i
ax = axs[2]
ax.plot(t[:-1], z_i, ".", color=C["b"], ms=2.5, label="increments $z_i$")
ax.plot(t, x_i, color=C["a"], lw=1.0, label="signal $x$")
ax.set_xlabel("$t$"); ax.set_title("(c) one draw: increments and signal")
ax.legend(frameon=False, loc="lower left", fontsize=6.5)
print(f"hierarchical mechanism (beta = 1, vartheta = {vt_h:.0e}): theta > 1e-2 at {np.sum(theta_i > 1e-2)} and theta > 1e-1 at {np.sum(theta_i > 1e-1)} of {n - 1} increments; mode {vt_h / (beta_h + 1):.1e}")
save(fig, "hiermechanism.pdf")


# ---------------------------------------------------------------- Fig: linear-Gaussian posteriors
# (a) deconvolution, smooth signal, smoothness prior with gamma from the discrepancy principle
lams = np.logspace(-7, 2, 200)
disc = np.array([np.linalg.norm(A @ tikhonov(A, y_s, lam, Ln) - y_s) for lam in lams])
errs = np.array([np.linalg.norm(tikhonov(A, y_s, lam, Ln) - x_s) for lam in lams])
i_disc = int(np.argmin(np.abs(disc - np.sqrt(m) * sigma))); i_opt = int(np.argmin(errs))
lam_disc = lams[i_disc]
gamma_s = sigma / np.sqrt(lam_disc)
P_s = gaussian_precision(n, gamma_s, "differences", gamma0=1.0)
mean_s, cov_s = gaussian_posterior(A, y_s, sigma**2, P_s)
std_s = np.sqrt(np.diag(cov_s))
chol = np.linalg.cholesky(cov_s + 1e-14 * np.eye(n))
samples_s = mean_s[None, :] + (chol @ rng.standard_normal((n, 3))).T
print(f"deconvolution, smoothness prior: discrepancy lambda {lam_disc:.3g} (optimal {lams[i_opt]:.3g}, error {errs[i_opt]:.3f}), "
      f"gamma = {gamma_s:.4f}, posterior mean error {np.linalg.norm(mean_s - x_s):.3f}, "
      f"std between {std_s.min():.3f} and {std_s.max():.3f}, truth inside the 95% band at {100 * np.mean(np.abs(mean_s - x_s) < 1.96 * std_s):.0f}% of the points")
cover = {"smooth": [], "piecewise": []}
for k in range(100):
    for name, x_true in [("smooth", x_s), ("piecewise", x_p)]:
        yy = A @ x_true + sigma * rng.standard_normal(m)
        mu, Cv = gaussian_posterior(A, yy, sigma**2, P_s)
        cover[name].append(np.mean(np.abs(mu - x_true) < 1.96 * np.sqrt(np.diag(Cv))))
print(f"coverage of the 95% band over 100 noise realizations: smooth {100 * np.mean(cover['smooth']):.0f}%, piecewise {100 * np.mean(cover['piecewise']):.0f}%")

# (b) bathymetry linearized at the truth, sparse sensors
bands = {}
try:
    import jax
    import jax.numpy as jnp
    import swe_jax
    xg, dx = make_grid(200)
    b_true = bottom_smooth(xg)
    mm = 20
    xo = np.linspace(0.5, LEN - 0.5, mm)
    fwd = swe_jax.make_forward(xg, xo)
    J = np.asarray(jax.jacfwd(fwd)(jnp.asarray(b_true[:-1])))
    d = 0.01 * rng.standard_normal(mm)                       # data minus F(b_true): pure noise
    Lf = difference_matrix(200)
    P_full = Lf.T @ Lf                                        # b_200 = 0 is known: the last increment is -b_199
    idx_s = np.array([np.argmin(np.abs(xg[:-1] - xs)) for xs in xo])
    mid = (idx_s[:-1] + idx_s[1:]) // 2
    for gam in (0.03, 0.01):
        mean_b, cov_b = gaussian_posterior(J, d, 0.01**2, P_full[:-1, :-1] / gam**2)
        std_b = np.sqrt(np.diag(cov_b))
        bands[gam] = (b_true[:-1] + mean_b, std_b)
        print(f"bathymetry m = 20, gamma = {gam}: posterior std {100 * std_b[idx_s].mean():.1f} cm at the sensors, "
              f"{100 * std_b[mid].mean():.1f} cm midway, maximum {100 * std_b.max():.1f} cm")
except ImportError:
    print("jax not available: panel (b) skipped")

fig, axs = plt.subplots(1, 2, figsize=(W, 2.1), constrained_layout=True)
ax = axs[0]
ax.fill_between(t, mean_s - 1.96 * std_s, mean_s + 1.96 * std_s, color=C["a"], alpha=0.2, lw=0, label="95% band")
for k in range(3):
    ax.plot(t, samples_s[k], color=C["a"], lw=0.5, alpha=0.7, label="posterior samples" if k == 0 else None)
ax.plot(t, x_s, color=C["truth"], label="truth")
ax.plot(t, mean_s, color=C["rec"], lw=1.0, label="posterior mean")
ax.set_xlabel("$t$"); ax.set_ylim(-0.5, 1.3); ax.set_title("(a) deconvolution, smoothness prior")
ax.legend(frameon=False, loc="lower center", ncol=2, fontsize=6.3)
ax = axs[1]
if bands:
    for gam, col, lab in [(0.03, C["a"], "$\\gamma = 3$ cm"), (0.01, C["d"], "$\\gamma = 1$ cm")]:
        mb, sb = bands[gam]
        ax.fill_between(xg[:-1], mb - 1.96 * sb, mb + 1.96 * sb, color=col, alpha=0.25, lw=0, label=f"95% band, {lab}")
    ax.plot(xg, b_true, color=C["truth"], label="truth")
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$b$ [m]"); ax.set_ylim(-0.15, 0.4)
    ax.set_title("(b) bathymetry, $m = 20$ sensors, linearized")
    ax.legend(frameon=False, loc="upper right", fontsize=6.3)
save(fig, "lineargaussian.pdf")


# ---------------------------------------------------------------- Fig: IAS on the piecewise-constant signal (beta = 1)
IAS_BETA, IAS_VT = HIER_BETA, HIER_VT
x_ias, theta, hist, xs_it, th_it = ias(A, y_p, sigma, IAS_BETA, IAS_VT, "differences", n_iter=200, return_iterates=True)
x_tv, _ = l1_regularization(A, y_p, 0.1, transform="differences", n_iter=4000)
jumps = np.nonzero(Ln @ x_p)[0]
K = len(hist)
rows = [(1, "after one iteration"), (K // 2, f"after {K // 2} iterations"), (K, f"after {K} iterations (converged)")]
print(f"IAS (beta = {IAS_BETA:g}, vartheta = {IAS_VT:.0e}, start at the mode {IAS_VT / (IAS_BETA + 1):.1e}): {K} iterations, "
      f"error {np.linalg.norm(x_ias - x_p):.3f} (TV {np.linalg.norm(x_tv - x_p):.3f}), discrepancy {np.linalg.norm(A @ x_ias - y_p):.3f} "
      f"(sqrt(m) sigma {np.sqrt(m) * sigma:.3f}), theta > 1e-2 at {np.sum(theta > 1e-2)} increments; "
      f"errors after iterations 1, {K // 2}: {np.linalg.norm(xs_it[0] - x_p):.3f}, {np.linalg.norm(xs_it[K // 2 - 1] - x_p):.3f}; "
      f"objective {hist[0]:.1f}, {hist[K // 2 - 1]:.1f}, {hist[-1]:.1f}")
x_long, th_long, h_long, xs_long, _ = ias(A, y_p, sigma, IAS_BETA, IAS_VT, "differences", n_iter=200, tol=0.0, return_iterates=True)
th_true = (0.5 * (Ln @ x_p) ** 2 + IAS_VT) / (IAS_BETA + 1.5)        # theta-step evaluated at the true signal
x_tr, th_tr, h_tr = ias(A, y_p, sigma, IAS_BETA, IAS_VT, "differences", n_iter=200, tol=0.0, theta0=th_true)
print(f"IAS run to 200 iterations: total change of x after iteration {K}: {np.linalg.norm(xs_long[-1] - xs_long[K - 1]):.1e}, "
      f"objective {h_long[-1]:.3f}; started from the variances of the true signal: error {np.linalg.norm(x_tr - x_p):.3f}, "
      f"objective {h_tr[-1]:.3f} (difference {h_tr[-1] - h_long[-1]:.2f}, density ratio {np.exp(h_tr[-1] - h_long[-1]):.1f}), "
      f"jumps at {np.nonzero(th_tr > 1e-2)[0].tolist()} vs {np.nonzero(theta > 1e-2)[0].tolist()} (true {jumps.tolist()})")
fig, axs = plt.subplots(3, 2, figsize=(W, 4.6), constrained_layout=True, sharex=True)
for r, (k, lab) in enumerate(rows):
    ax = axs[r, 0]
    ax.plot(t, x_p, color=C["truth"], label="truth")
    ax.plot(t, xs_it[k - 1], color=C["rec"], lw=1.0, label="$\\mathbf{x}_k$")
    ax.set_ylim(-0.3, 1.3); ax.set_title(f"({'ace'[r]}) reconstruction {lab}")
    ax.text(0.03, 0.85, f"error {np.linalg.norm(xs_it[k - 1] - x_p):.2f}", transform=ax.transAxes, fontsize=7)
    ax = axs[r, 1]
    for j in jumps:
        ax.axvline(t[j], color="0.8", lw=0.7)
    ax.semilogy(t[:-1], th_it[k - 1], ".-", color=C["rec"], ms=2, lw=0.5)
    ax.axhline(IAS_VT / (IAS_BETA + 1.5), color="0.4", ls="--", lw=0.7)
    ax.set_ylim(1e-4, 1); ax.set_title(f"({'bdf'[r]}) local variances $\\theta_i$ {lab}")
axs[0, 0].legend(frameon=False, loc="upper right", fontsize=6.5)
axs[2, 0].set_xlabel("$t$"); axs[2, 1].set_xlabel("$t$")
save(fig, "ias.pdf")


# ---------------------------------------------------------------- Fig: MAP estimates under the three priors on increments
# (the parameters of Figure 3.4: gamma = 0.1 for the Gaussian and the Laplace prior, beta = 1, vartheta = 1e-3)
GAMMA_PRIOR = 0.1
x_gauss = tikhonov(A, y_p, sigma**2 / GAMMA_PRIOR**2, Ln)                       # Gaussian increments: lambda = sigma^2 / gamma^2
x_lap, _ = l1_regularization(A, y_p, sigma**2 / GAMMA_PRIOR, transform="differences", n_iter=4000)   # Laplace: lambda = sigma^2 / gamma
maps = [(x_gauss, f"(a) Gaussian increments, $\\gamma = {GAMMA_PRIOR:g}$"),
        (x_lap, f"(b) Laplace increments, $\\gamma = {GAMMA_PRIOR:g}$"),
        (x_ias, "(c) hierarchical increments, $\\vartheta = 10^{-3}$")]
fig, axs = plt.subplots(1, 3, figsize=(W, 1.9), constrained_layout=True, sharey=True)
for ax, (xm, title) in zip(axs, maps):
    ax.plot(t, x_p, color=C["truth"], label="truth")
    ax.plot(t, xm, color=C["rec"], lw=1.0, label="MAP estimate")
    ax.text(0.03, 0.88, f"error {np.linalg.norm(xm - x_p):.2f}", transform=ax.transAxes, fontsize=7)
    ax.set_title(title); ax.set_xlabel("$t$")
axs[0].set_ylim(-0.3, 1.3)
axs[0].legend(frameon=False, loc="upper right", fontsize=6.3)
print("MAP comparison: " + "; ".join(f"{title.split(')')[1].strip()}: error {np.linalg.norm(xm - x_p):.3f}, discrepancy {np.linalg.norm(A @ xm - y_p):.3f}"
                                     for xm, title in maps) + f"; sqrt(m) sigma = {np.sqrt(m) * sigma:.3f}")
save(fig, "mapcompare.pdf")


# ---------------------------------------------------------------- Fig: three things to know about MAP estimates
fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
# (a) parametrization: kappa > 0 with a log-normal posterior; u = log kappa has a Gaussian posterior
ax = axs[0]
mu_u, s_u = 0.0, 0.6
kap = np.linspace(0.01, 3.5, 1000)
dens_k = np.exp(-(np.log(kap) - mu_u) ** 2 / (2 * s_u**2)) / (kap * s_u * np.sqrt(2 * np.pi))
map_k = np.exp(mu_u - s_u**2); map_u = np.exp(mu_u); mean_k = np.exp(mu_u + s_u**2 / 2)
ax.plot(kap, dens_k, "k", label="posterior of $\\kappa$")
ax.axvline(map_k, color=C["rec"], lw=1.0, label=f"MAP of $\\kappa$: {map_k:.2f}")
ax.axvline(map_u, color=C["b"], lw=1.0, ls="--", label=f"$\\exp$(MAP of $\\log\\kappa$): {map_u:.2f}")
ax.axvline(mean_k, color=C["c"], lw=1.0, ls=":", label=f"posterior mean: {mean_k:.2f}")
ax.set_xlabel("$\\kappa$"); ax.set_yticks([]); ax.set_title("(a) parametrization matters")
ax.legend(frameon=False, fontsize=5.8, loc="upper right")
# (b) sparsity: scalar denoising y = x + eps, Laplace prior; MAP is soft thresholding, the mean is not sparse
ax = axs[1]
sig_b, gam_b = 1.0, 1.0
yy = np.linspace(-4, 4, 161)
xx = np.linspace(-12, 12, 4001)
cm = []
for yv in yy:
    w = np.exp(-(yv - xx) ** 2 / (2 * sig_b**2) - np.abs(xx) / gam_b)
    cm.append(np.sum(xx * w) / np.sum(w))
cm = np.array(cm)
map_b = np.sign(yy) * np.maximum(np.abs(yy) - sig_b**2 / gam_b, 0.0)
ax.plot(yy, yy, color="0.7", lw=0.7, ls=":", label="$x = y$")
ax.plot(yy, map_b, color=C["rec"], label="MAP (soft threshold)")
ax.plot(yy, cm, color=C["c"], ls="--", label="posterior mean")
ax.set_xlabel("datum $y$"); ax.set_ylabel("estimate of $x$"); ax.set_title("(b) the mean is not sparse")
ax.legend(frameon=False, fontsize=6, loc="upper left")
# (c) nonlinear forward map: y = x^2 + eps gives a bimodal posterior
ax = axs[2]
y_c, s_c, mu_c, sp_c = 1.0, 0.3, 0.4, 1.0
xx = np.linspace(-2.5, 2.5, 2001)
lp_c = -(y_c - xx**2) ** 2 / (2 * s_c**2) - (xx - mu_c) ** 2 / (2 * sp_c**2)
p_c = np.exp(lp_c - lp_c.max()); p_c /= np.trapezoid(p_c, xx)
mean_c = np.trapezoid(xx * p_c, xx)
i1 = np.argmax(np.where(xx > 0, p_c, -1)); i2 = np.argmax(np.where(xx < 0, p_c, -1))
ax.plot(xx, p_c, "k", label="posterior")
ax.plot(xx[i1], p_c[i1], "*", color=C["rec"], ms=9, label=f"MAP: {xx[i1]:.2f}")
ax.plot(xx[i2], p_c[i2], "*", color=C["b"], ms=8, label=f"local maximum: {xx[i2]:.2f}")
ax.plot(mean_c, 0, "o", color=C["c"], ms=5, label=f"posterior mean: {mean_c:.2f}")
ax.annotate("", xy=(xx[i1] - 0.08, 0.3), xytext=(0.25, 0.3), arrowprops=dict(arrowstyle="->", color="0.4"))
ax.annotate("", xy=(xx[i2] + 0.08, 0.3), xytext=(-0.25, 0.3), arrowprops=dict(arrowstyle="->", color="0.4"))
ax.text(0, 0.36, "starts", color="0.3", fontsize=6, ha="center")
ax.set_xlabel("$x$"); ax.set_yticks([]); ax.set_title("(c) $y = x^2 + \\varepsilon$: two valleys")
ax.legend(frameon=False, fontsize=5.8, loc="upper left")
print(f"caveats: (a) MAP kappa {map_k:.2f}, exp(MAP log kappa) {map_u:.2f}, mean {mean_k:.2f}; (c) modes {xx[i1]:.2f}, {xx[i2]:.2f}, mean {mean_c:.2f}, "
      f"density ratio {p_c[i1] / p_c[i2]:.2f}")
save(fig, "mapcaveats.pdf")


# ---------------------------------------------------------------- numbers: evidence maximization (Section 3.6)
def log_evidence(y, gamma, sig, gamma0=1.0):
    P = gaussian_precision(n, gamma, "differences", gamma0)
    Cy = A @ np.linalg.inv(P) @ A.T + sig**2 * np.eye(m)
    return -0.5 * (y @ np.linalg.solve(Cy, y)) - 0.5 * np.linalg.slogdet(Cy)[1] - 0.5 * m * np.log(2 * np.pi)

gammas, sigs = np.logspace(-3, 0, 61), np.logspace(-3, 0, 61)
E = np.array([[log_evidence(y_s, g, sg) for sg in sigs] for g in gammas])
i, j = np.unravel_index(np.argmax(E), E.shape)
g_hat, s_hat = gammas[i], sigs[j]
mean_ev, _ = gaussian_posterior(A, y_s, s_hat**2, gaussian_precision(n, g_hat, "differences"))
print(f"evidence maximization (smooth signal): sigma {s_hat:.3f} (true {sigma}), gamma {g_hat:.3f}, lambda = sigma^2/gamma^2 = {s_hat**2 / g_hat**2:.2g}, "
      f"error of the posterior mean {np.linalg.norm(mean_ev - x_s):.3f} (discrepancy principle: {np.linalg.norm(mean_s - x_s):.3f})")

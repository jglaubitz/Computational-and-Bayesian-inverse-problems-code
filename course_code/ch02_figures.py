"""Figures of Chapter 2 (Deterministic regularization). Output: ../figures/ch02/*.pdf

    python ch02_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from running_examples import (deconvolution_operator, kernel, signal_smooth, signal_piecewise, noisy_data,
                              make_grid, bottom_smooth, GAMMA, N_CELLS, M_OBS, NOISE_STD, L)
from solvers import difference_matrix, tikhonov, l1_regularization, gauss_newton

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "ch02")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "lines.linewidth": 1.1})
W = 6.3
C = {"truth": "k", "rec": "tab:red", "a": "tab:blue", "b": "tab:orange", "c": "tab:green", "d": "tab:purple", "data": "tab:red"}
rng = np.random.default_rng(1)

n, m, sigma = N_CELLS, M_OBS, NOISE_STD
A, t, s = deconvolution_operator(n, m)
x_s, x_p = signal_smooth(t), signal_piecewise(t)
eps_s, eps_p = sigma * rng.standard_normal(m), sigma * rng.standard_normal(m)
y_s, y_p = A @ x_s + eps_s, A @ x_p + eps_p
Ln = difference_matrix(n)


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), **kw)
    plt.close(fig)
    print("wrote", name)


def fmt(v):
    mant, e = f"{v:.0e}".split("e")
    e = int(e)
    if e == 0:
        return f"${mant}$"
    return f"${mant}\\cdot 10^{{{e}}}$" if mant != "1" else f"$10^{{{e}}}$"


# ---------------------------------------------------------------- Fig: what each penalty considers cheap
fig, axs = plt.subplots(2, 2, figsize=(W * 0.85, 3.6), constrained_layout=True)
axs = axs.ravel()
bump = np.exp(-((t - 0.5) / 0.12) ** 2)
jag = bump + 0.15 * np.sin(2 * np.pi * 25 * t)
spikes = np.zeros(n); spikes[[20, 61, 100]] = [1.0, -0.7, 0.8]
box = np.where((t > 0.3) & (t < 0.65), 0.8, 0.0)
osc = 0.4 + 0.4 * np.sin(2 * np.pi * 6 * t)
panels = [
    ("(a) small: $\\frac{1}{2}\\|\\mathbf{x}\\|_2^2$", 0.25 * bump, bump, lambda v: 0.5 * np.sum(v**2)),
    ("(b) smooth: $\\frac{1}{2}\\|\\mathbf{L}\\mathbf{x}\\|_2^2$", bump, jag, lambda v: 0.5 * np.sum((Ln @ v) ** 2)),
    ("(c) sparse: $\\|\\mathbf{x}\\|_1$", spikes, bump, lambda v: np.sum(np.abs(v))),
    ("(d) piecewise const.: $\\|\\mathbf{L}\\mathbf{x}\\|_1$", box, osc, lambda v: np.sum(np.abs(Ln @ v))),
]
for ax, (title, cheap, dear, R) in zip(axs, panels):
    ax.plot(t, dear, color=C["rec"], lw=0.9, label="expensive")
    ax.plot(t, cheap, color=C["a"], lw=1.3, label="cheap")
    ax.text(0.03, 0.9, f"$R = {R(cheap):.2g}$", color=C["a"], transform=ax.transAxes, fontsize=7)
    ax.text(0.03, 0.78, f"$R = {R(dear):.2g}$", color=C["rec"], transform=ax.transAxes, fontsize=7)
    ax.set_title(title); ax.set_ylim(-1.1, 1.35)
for ax in axs[2:]:
    ax.set_xlabel("$t$")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
fig.savefig(os.path.join(OUT, "grammars.pdf"), bbox_inches="tight"); plt.close(fig); print("wrote grammars.pdf")


# ---------------------------------------------------------------- Tikhonov with L = I: error curve, rules
lams = np.logspace(-7, 1.5, 120)
U, sv, Vt = np.linalg.svd(A, full_matrices=False)          # only used to evaluate the GCV trace and the bias/noise split
x_lam, errs, disc, norms, gcv, bias, noise = [], [], [], [], [], [], []
for lam in lams:
    xl = tikhonov(A, y_s, lam)
    x_lam.append(xl)
    errs.append(np.linalg.norm(xl - x_s)); disc.append(np.linalg.norm(A @ xl - y_s)); norms.append(np.linalg.norm(xl))
    filt = sv**2 / (sv**2 + lam)
    gcv.append(m * disc[-1] ** 2 / (m - np.sum(filt)) ** 2)
    x_bias = tikhonov(A, A @ x_s, lam)                       # regularized solution for exact data
    bias.append(np.linalg.norm(x_bias - x_s)); noise.append(np.linalg.norm(xl - x_bias))
errs, disc, norms, gcv, bias, noise = map(np.array, (errs, disc, norms, gcv, bias, noise))
i_opt = int(np.argmin(errs))
i_disc = int(np.argmin(np.abs(disc - np.sqrt(m) * sigma)))
i_gcv = int(np.argmin(gcv))
# L-curve corner: maximum curvature of (log residual, log norm) parametrized by log lambda
lr, ln_ = np.log(disc), np.log(norms)
u = np.log(lams)
d1r, d1n = np.gradient(lr, u), np.gradient(ln_, u)
d2r, d2n = np.gradient(d1r, u), np.gradient(d1n, u)
curv = (d1r * d2n - d2r * d1n) / (d1r**2 + d1n**2) ** 1.5
i_lc = int(np.argmax(curv[10:-10])) + 10
print(f"lambda: optimal {lams[i_opt]:.2e} (err {errs[i_opt]:.3f}), discrepancy {lams[i_disc]:.2e} (err {errs[i_disc]:.3f}), "
      f"L-curve {lams[i_lc]:.2e} (err {errs[i_lc]:.3f}), GCV {lams[i_gcv]:.2e} (err {errs[i_gcv]:.3f}); ||x|| = {np.linalg.norm(x_s):.2f}")

# ---------------------------------------------------------------- Fig: the Tikhonov filter in the frequency domain
fig, ax = plt.subplots(1, 1, figsize=(W * 0.62, 2.0), constrained_layout=True)
nu = np.linspace(0, 40, 400)
khat = np.exp(-2 * np.pi**2 * GAMMA**2 * nu**2)
ax.semilogy(nu, khat, "k", lw=1.2, label="attenuation $\\widehat{k}(\\nu)$")
for lam, col, lab in [(1e-5, C["c"], "$\\lambda = 10^{-5}$"), (lams[i_disc], C["a"], f"$\\lambda = {lams[i_disc]:.2g}$ (discrepancy)"), (1.0, C["b"], "$\\lambda = 1$")]:
    ax.semilogy(nu, khat**2 / (khat**2 + lam), color=col, label=f"filter $\\varphi(\\nu)$, {lab}")
ax.axhline(sigma, color="0.5", ls="--", lw=0.8)
ax.text(30, sigma * 1.4, "noise level $\\sigma$", color="0.4", fontsize=7)
ax.set_ylim(1e-6, 2); ax.set_xlabel("frequency $\\nu$"); ax.set_title("What Tikhonov regularization does to each frequency")
ax.legend(frameon=False, loc="lower left")
save(fig, "filter.pdf")

# ---------------------------------------------------------------- Fig: Tikhonov reconstructions for three lambdas
fig, axs = plt.subplots(2, 3, figsize=(W, 3.4), constrained_layout=True, sharex=True)
choices = [(1e-5, "too small"), (lams[i_disc], "discrepancy principle"), (1.0, "too large")]
for row, (x_true, y, lab) in enumerate([(x_s, y_s, "smooth signal"), (x_p, y_p, "piecewise-constant signal")]):
    for col, (lam, what) in enumerate(choices):
        ax = axs[row, col]
        xl = tikhonov(A, y, lam)
        ax.plot(t, x_true, color=C["truth"], label="truth")
        ax.plot(t, xl, color=C["rec"], lw=0.9, label="$\\mathbf{x}_\\lambda$")
        ax.set_title(f"({'abcdef'[3 * row + col]}) $\\lambda = $ {fmt(lam) if lam in (1e-5, 1.0) else f'{lam:.2g}'} ({what})", fontsize=7.5)
        if col == 0:
            ax.set_ylabel(lab)
        err = np.linalg.norm(xl - x_true)
        ax.text(0.03, 0.88, f"error {err:.2f}", transform=ax.transAxes, fontsize=7)
        if col == 0:
            ax.set_ylim(-1.2, 2.2)
        else:
            ax.set_ylim(-0.3, 1.3)
for ax in axs[1]:
    ax.set_xlabel("$t$")
axs[0, 1].legend(frameon=False, loc="upper right")
save(fig, "tikhonov.pdf")

# ---------------------------------------------------------------- Fig: parameter choice
fig, axs = plt.subplots(2, 2, figsize=(W * 0.85, 3.9), constrained_layout=True)
axs = axs.ravel()
ax = axs[0]
ax.loglog(lams, errs, "k", label="error $\\|\\mathbf{x}_\\lambda - \\mathbf{x}^\\dagger\\|_2$")
ax.loglog(lams, bias, "--", color=C["b"], label="bias (exact data)")
ax.loglog(lams, noise, "--", color=C["c"], label="amplified noise")
for i, col, lab in [(i_disc, C["a"], "discrepancy"), (i_lc, C["d"], "L-curve"), (i_gcv, C["rec"], "GCV")]:
    ax.plot(lams[i], errs[i], "o", color=col, ms=4, label=lab)
ax.set_xlabel("$\\lambda$"); ax.set_title("(a) error and its two parts"); ax.set_ylim(2e-3, 2e4)
ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2, handlelength=1.5, columnspacing=1.0)
ax = axs[1]
ax.loglog(lams, disc, "k")
ax.axhline(np.sqrt(m) * sigma, color=C["a"], ls="--", lw=0.8)
ax.plot(lams[i_disc], disc[i_disc], "o", color=C["a"], ms=4)
ax.text(2e-7, np.sqrt(m) * sigma * 1.25, "$\\sqrt{m}\\,\\sigma$", color=C["a"], fontsize=7)
ax.set_xlabel("$\\lambda$"); ax.set_title("(b) discrepancy $\\|\\mathbf{A}\\mathbf{x}_\\lambda - \\mathbf{y}\\|_2$")
ax.set_yticks([0.2, 0.5, 1, 2]); ax.set_yticklabels(["0.2", "0.5", "1", "2"]); ax.minorticks_off()
ax = axs[2]
ax.loglog(disc, norms, "k")
ax.plot(disc[i_lc], norms[i_lc], "o", color=C["d"], ms=4)
for i, lab in [(i_lc, "corner")]:
    ax.annotate(lab, (disc[i], norms[i]), textcoords="offset points", xytext=(8, 4), fontsize=7, color=C["d"])
ax.set_xlabel("$\\|\\mathbf{A}\\mathbf{x}_\\lambda - \\mathbf{y}\\|_2$"); ax.set_ylabel("$\\|\\mathbf{x}_\\lambda\\|_2$"); ax.set_title("(c) L-curve")
ax.set_xticks([0.3, 1, 3]); ax.set_xticklabels(["0.3", "1", "3"]); ax.minorticks_off()
ax = axs[3]
ax.loglog(lams, gcv, "k")
ax.plot(lams[i_gcv], gcv[i_gcv], "o", color=C["rec"], ms=4)
ax.set_xlabel("$\\lambda$"); ax.set_title("(d) GCV function")
save(fig, "parameter.pdf")

# ---------------------------------------------------------------- Fig: Tikhonov versus TV
def discrepancy_lambda_tv(y, transform, grid):
    best = None
    for lam in grid:
        x, _ = l1_regularization(A, y, lam, transform=transform, n_iter=4000)
        d = np.linalg.norm(A @ x - y)
        if best is None or abs(d - np.sqrt(m) * sigma) < best[0]:
            best = (abs(d - np.sqrt(m) * sigma), lam, x, d)
    return best[1], best[2], best[3]

grid = np.logspace(-3, 0.5, 15)
lam_tv_p, x_tv_p, d_p = discrepancy_lambda_tv(y_p, "differences", grid)
lam_tv_s, x_tv_s, d_s = discrepancy_lambda_tv(y_s, "differences", grid)
disc_p = [np.linalg.norm(A @ tikhonov(A, y_p, lam) - y_p) for lam in lams]
lam_tik_p = lams[int(np.argmin(np.abs(np.array(disc_p) - np.sqrt(m) * sigma)))]
x_tik_p = tikhonov(A, y_p, lam_tik_p)
print(f"piecewise: Tikhonov lam {lam_tik_p:.2g} err {np.linalg.norm(x_tik_p - x_p):.2f}; TV lam {lam_tv_p:.2g} err {np.linalg.norm(x_tv_p - x_p):.2f} (disc {d_p:.3f}); "
      f"smooth: TV lam {lam_tv_s:.2g} err {np.linalg.norm(x_tv_s - x_s):.2f}, Tikhonov err {errs[i_disc]:.2f}")
fig, axs = plt.subplots(1, 3, figsize=(W, 1.9), constrained_layout=True)
for ax, x_true, xr, lab in [(axs[0], x_p, x_tik_p, f"(a) Tikhonov, $\\lambda = {lam_tik_p:.2g}$"),
                            (axs[1], x_p, x_tv_p, f"(b) total variation, $\\lambda = {lam_tv_p:.2g}$"),
                            (axs[2], x_s, x_tv_s, f"(c) total variation, $\\lambda = {lam_tv_s:.2g}$")]:
    ax.plot(t, x_true, color=C["truth"], label="truth")
    ax.plot(t, xr, color=C["rec"], lw=0.9, label="$\\mathbf{x}_\\lambda$")
    ax.set_title(lab); ax.set_xlabel("$t$"); ax.set_ylim(-0.3, 1.3)
    ax.text(0.03, 0.88, f"error {np.linalg.norm(xr - x_true):.2f}", transform=ax.transAxes, fontsize=7)
axs[0].legend(frameon=False, loc="upper right")
save(fig, "tv.pdf")

# ---------------------------------------------------------------- Fig: Gauss-Newton on the bathymetry example
try:
    import jax
    import jax.numpy as jnp
    import swe_jax
    xg, dx = make_grid(200)
    b_true = bottom_smooth(xg)
    Lb = difference_matrix(199)
    results = {}
    for mm in (200, 20):
        xo = np.linspace(0.5, L - 0.5, mm)
        fwd = swe_jax.make_forward(xg, xo)
        jac = jax.jit(jax.jacfwd(fwd))
        yb = np.asarray(fwd(jnp.asarray(b_true[:-1]))) + 0.01 * rng.standard_normal(mm)
        F = lambda b, fwd=fwd: np.asarray(fwd(jnp.asarray(b)))
        Jf = lambda b, jac=jac: np.asarray(jac(jnp.asarray(b)))
        best = None
        for lam in np.logspace(-2, 1, 13):
            b_hat, hist = gauss_newton(F, Jf, yb, np.zeros(199), lam=lam, L=Lb, n_iter=40)
            d = np.linalg.norm(F(b_hat) - yb)
            if best is None or abs(d - 0.01 * np.sqrt(mm)) < best[0]:
                best = (abs(d - 0.01 * np.sqrt(mm)), lam, b_hat, hist)
        _, lam, b_hat, hist = best
        b0, hist0 = gauss_newton(F, Jf, yb, np.zeros(199), lam=0.0, L=Lb, n_iter=40)
        results[mm] = dict(lam=lam, b=b_hat, hist=hist, b0=b0, xo=xo, yb=yb)
        print(f"Gauss-Newton m={mm}: lambda {lam:.2g}, {len(hist) - 1} iterations, rms error {np.sqrt(np.mean((b_hat - b_true[:-1])**2)):.4f} m, "
              f"without penalty rms error {np.sqrt(np.mean((b0 - b_true[:-1])**2)):.4f} m")
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.9), constrained_layout=True)
    ax = axs[0]
    for mm, col in [(200, C["a"]), (20, C["rec"])]:
        h = results[mm]["hist"]
        ax.semilogy(np.arange(len(h)), h, "o-", ms=3, color=col, label=f"$m = {mm}$, $\\lambda = {results[mm]['lam']:.2g}$")
    ax.set_xlabel("Gauss–Newton iteration"); ax.set_ylabel("$J(\\mathbf{b}_k)$"); ax.set_title("(a) convergence")
    ax.legend(frameon=False)
    for ax, mm, lab in [(axs[1], 200, "(b) dense sensors, $m = 200$"), (axs[2], 20, "(c) sparse sensors, $m = 20$")]:
        r = results[mm]
        ax.fill_between(xg, 0, b_true, color="0.8", lw=0, label="truth")
        if mm == 20:
            ax.plot(xg[:-1], r["b0"], ":", color="0.4", lw=0.9, label="$\\lambda = 0$")
        ax.plot(xg[:-1], r["b"], color=C["rec"], lw=1.0, label="$\\mathbf{b}_\\lambda$")
        ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$b$ [m]"); ax.set_ylim(-0.1, 0.3); ax.set_title(lab)
        ax.legend(frameon=False, loc="upper right", fontsize=6.5, ncol=2)
    save(fig, "gaussnewton.pdf")
except ImportError:
    print("jax not available: skipping gaussnewton.pdf")

"""Figures of Chapter 1 (Inverse problems: the basics). Output: ../figures/ch01/*.pdf

    python ch01_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from running_examples import (deconvolution_operator, kernel, signal_smooth, signal_piecewise, noisy_data,
                              make_grid, steady_depth, surface_at_sensors, froude, bottom_smooth, jacobian_fd,
                              GAMMA, N_CELLS, M_OBS, NOISE_STD, L, Q, H_OUT, G)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "ch01")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "lines.linewidth": 1.1})
W = 6.3                                   # text width in inches
C = {"truth": "k", "blur": "tab:blue", "data": "tab:red", "a": "tab:blue", "b": "tab:orange", "c": "tab:green", "d": "tab:purple"}
rng = np.random.default_rng(1)

n, m, sigma = N_CELLS, M_OBS, NOISE_STD
A, t, s = deconvolution_operator(n, m)
x_s, x_p = signal_smooth(t), signal_piecewise(t)
y_s = noisy_data(A, x_s, sigma, rng)
y_p = noisy_data(A, x_p, sigma, rng)
U, sv, Vt = np.linalg.svd(A, full_matrices=False)   # U (m x m), sv (m,), Vt (m x n): the first m right singular vectors
t_fine = np.linspace(0, 1, 1000)
A_fine, _, _ = deconvolution_operator(n, 1000)        # the blurred signal on a fine grid, for plotting


def sci(v):
    mant, e = f"{v:.0e}".split("e")
    return f"${mant} \\cdot 10^{{{int(e)}}}$"


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), **kw)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig: the deconvolution example
fig, axs = plt.subplots(1, 3, figsize=(W, 1.9), constrained_layout=True)
tt = np.linspace(-0.15, 0.15, 400)
axs[0].plot(tt, kernel(tt), color=C["a"])
axs[0].fill_between(tt, 0, kernel(tt), color=C["a"], alpha=0.15, lw=0)
axs[0].set_xlabel("$t$"); axs[0].set_title(f"(a) kernel $k$, width $\\gamma = {GAMMA}$")
axs[0].set_yticks([0, 5, 10])
for ax, x, y, lab in [(axs[1], x_s, y_s, "(b) smooth signal"), (axs[2], x_p, y_p, "(c) piecewise-constant signal")]:
    ax.plot(t, x, color=C["truth"], label="unknown signal $x$")
    ax.plot(t_fine, A_fine @ x, color=C["blur"], label="blurred signal $k \\ast x$")
    ax.plot(s, y, ".", color=C["data"], ms=3.5, label=f"data $\\mathbf{{y}}$ ($m = {m}$ points, noise $\\sigma = {sigma}$)")
    ax.set_xlabel("$t$"); ax.set_title(lab); ax.set_ylim(-0.2, 1.2)
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
save(fig, "deconvolution.pdf", bbox_inches="tight")

# ---------------------------------------------------------------- Fig: X-ray CT
try:
    from skimage.data import shepp_logan_phantom
    from skimage.transform import radon, iradon, resize
    img = resize(shepp_logan_phantom(), (256, 256), anti_aliasing=True)
    theta = np.linspace(0.0, 180.0, 180, endpoint=False)
    sino = radon(img, theta=theta)
    n_ang, noise_ct = 20, 0.05
    theta_few = np.linspace(0.0, 180.0, n_ang, endpoint=False)
    sino_few = radon(img, theta=theta_few)
    sino_few_noisy = sino_few + noise_ct * sino.max() * rng.standard_normal(sino_few.shape)
    rec = iradon(sino_few_noisy, theta=theta_few, filter_name="ramp")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.25), constrained_layout=True)
    axs[0].imshow(img, cmap="gray", extent=[-1, 1, -1, 1])
    for ang, col in [(30, C["b"]), (110, C["c"])]:
        a = np.deg2rad(ang)
        for off in np.linspace(-0.8, 0.8, 5):
            p0 = off * np.array([np.cos(a), np.sin(a)]) - 1.3 * np.array([-np.sin(a), np.cos(a)])
            p1 = off * np.array([np.cos(a), np.sin(a)]) + 1.3 * np.array([-np.sin(a), np.cos(a)])
            axs[0].plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=0.7, alpha=0.9)
    axs[0].set_xlim(-1, 1); axs[0].set_ylim(-1, 1); axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].set_title("(a) attenuation $\\mu$ and\ntwo ray directions")
    axs[1].imshow(sino.T, cmap="gray", aspect="auto", extent=[-1, 1, 180, 0])
    axs[1].axhline(30, color=C["b"], lw=0.8); axs[1].axhline(110, color=C["c"], lw=0.8)
    axs[1].set_xlabel("detector position $s$"); axs[1].set_ylabel("angle $\\theta$ [deg]"); axs[1].set_title("(b) sinogram (the data)")
    axs[2].imshow(np.clip(rec, 0, 1), cmap="gray", extent=[-1, 1, -1, 1]); axs[2].set_xticks([]); axs[2].set_yticks([])
    axs[2].set_title(f"(c) naive inversion from {n_ang} angles,\nnoise $\\sigma = $ {int(100*noise_ct)} % of the maximum")
    save(fig, "ct.pdf")
except ImportError:
    print("scikit-image not available: skipping ct.pdf")

# ---------------------------------------------------------------- Fig: bathymetry example
x, dx = make_grid(200)
b = bottom_smooth(x)
h = steady_depth(b, Q, H_OUT, 0.0)
x_obs = np.linspace(0.5, L - 0.5, 20)
eta_obs = surface_at_sensors(b, x, x_obs) + 0.01 * rng.standard_normal(len(x_obs))
fig, axs = plt.subplots(1, 2, figsize=(W, 2.0), constrained_layout=True)
ax = axs[0]
ax.fill_between(x, 0, b, color="0.6", lw=0, label="bottom $b$ (unknown)")
ax.fill_between(x, b, h + b, color=C["a"], alpha=0.18, lw=0, label="water, depth $h$")
ax.plot(x, h + b, color=C["a"], lw=1.2, label="free surface $\\eta = h + b$")
ax.plot(x_obs, eta_obs, "v", color=C["data"], ms=4, label="sensors")
ax.annotate("", xy=(6.5, 1.0), xytext=(2.5, 1.0), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(2.5, 1.08, "flow, $q = 4.42\\,\\mathrm{m^2/s}$", fontsize=7)
ax.set_xlabel("$x$ [m]"); ax.set_ylabel("elevation [m]"); ax.set_ylim(0, 2.6)
ax.set_title("(a) steady flow over an unknown bottom")
ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.6), ncol=2, fontsize=6.5)
ax = axs[1]
ax.plot(x, (h + b - H_OUT) * 100, color=C["a"], label="$\\eta - h_{\\mathrm{out}}$, exact")
ax.plot(x_obs, (eta_obs - H_OUT) * 100, "v", color=C["data"], ms=4, label="sensor readings (noise $\\sigma = 1$ cm)")
ax.axhline(0, color="0.6", lw=0.6)
ax.set_xlabel("$x$ [m]"); ax.set_ylabel("surface deviation [cm]")
ax.set_title("(b) the data: surface at $m = 20$ sensors")
ax.legend(frameon=False, loc="lower right")
save(fig, "bathymetry.pdf")

# ---------------------------------------------------------------- Fig: the naive solution (least-squares / pseudo-inverse)
fig, axs = plt.subplots(1, 3, figsize=(W, 1.9), constrained_layout=True)
levels = [(0.0, "(a) exact data"), (1e-3, "(b) noise $\\sigma = 10^{-3}$"), (sigma, f"(c) noise $\\sigma = {sigma}$")]
for ax, (lev, lab) in zip(axs, levels):
    y = A @ x_s + lev * rng.standard_normal(m)
    xn = np.linalg.lstsq(A, y, rcond=None)[0]
    ax.plot(t, x_s, color=C["truth"], label="truth $\\mathbf{x}$")
    ax.plot(t, xn, color=C["data"], lw=0.8, label="$\\mathbf{A}^\\dagger\\mathbf{y}$")
    ax.set_title(lab); ax.set_xlabel("$t$")
    ax.text(0.03, 0.9, "max $|x_i| \\approx$ " + sci(np.max(np.abs(xn))), transform=ax.transAxes, fontsize=7)
axs[0].legend(frameon=False, loc="upper right")
axs[0].set_ylim(-0.2, 1.3)
save(fig, "naive.pdf")

# ---------------------------------------------------------------- Fig: condition number versus n (m = n/2 observations)
fig, axs = plt.subplots(1, 2, figsize=(W, 2.0), constrained_layout=True)
ns = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
for gam, col, lab in [(GAMMA, C["a"], f"$\\gamma = {GAMMA}$"), (GAMMA / 2, C["b"], f"$\\gamma = {GAMMA/2}$"), (2 * GAMMA, C["c"], f"$\\gamma = {2*GAMMA}$")]:
    kap = [np.linalg.cond(deconvolution_operator(nn, nn // 2, gamma=gam)[0]) for nn in ns]
    axs[0].semilogy(ns, kap, "o-", ms=3, color=col, label=lab)
axs[0].axhline(1 / np.finfo(float).eps, color="0.5", ls="--", lw=0.8)
axs[0].text(20, 2.5e16, "$1/\\epsilon_{\\mathrm{mach}}$", color="0.4", fontsize=7)
axs[0].set_xlabel("number of cells $n$ ($m = n/2$ observations)"); axs[0].set_ylabel("$\\kappa(\\mathbf{A}_n)$")
axs[0].set_title("(a) condition number of the discretized operator")
axs[0].legend(frameon=False, loc="lower right")
for nn, col in [(64, C["c"]), (128, C["a"]), (256, C["b"]), (512, C["d"])]:
    An = deconvolution_operator(nn, nn // 2)[0]
    svn = np.linalg.svd(An, compute_uv=False)
    axs[1].semilogy(np.arange(1, len(svn) + 1), svn, ".", ms=2.5, color=col, label=f"$n = {nn}$")
axs[1].set_ylim(1e-9, 3); axs[1].set_xlim(0, 70)
axs[1].set_xlabel("index $i$"); axs[1].set_ylabel("$\\sigma_i$")
axs[1].set_title("(b) singular values for several $n$")
axs[1].legend(frameon=False, loc="lower left")
save(fig, "condition.pdf")

# ---------------------------------------------------------------- Fig: SVD of A
fig = plt.figure(figsize=(W, 1.9), constrained_layout=True)
gs = fig.add_gridspec(1, 5, width_ratios=[1.6, 1, 1, 1, 1])
ax = fig.add_subplot(gs[0, 0])
ax.semilogy(np.arange(1, m + 1), sv, ".", ms=2.5, color=C["a"])
shown = [1, 5, 15, 40]
for i in shown:
    ax.plot(i, sv[i - 1], "o", ms=5, mfc="none", color=C["data"])
ax.set_xlabel("$i$"); ax.set_title("(a) singular values $\\sigma_i$"); ax.set_ylim(1e-9, 3)
for k, i in enumerate(shown):
    ax = fig.add_subplot(gs[0, k + 1])
    ax.plot(t, Vt[i - 1], color=C["a"], lw=0.9)
    ax.set_title(f"({'bcde'[k]}) $\\mathbf{{v}}_{{{i}}}$")
    ax.set_yticks([]); ax.set_xlabel("$t$"); ax.set_ylim(-0.25, 0.25)
    ax.text(0.5, 0.88, f"$\\sigma_{{{i}}} \\approx {sv[i-1]:.2g}$", transform=ax.transAxes, ha="center", fontsize=7)
save(fig, "svd.pdf")

# ---------------------------------------------------------------- Fig: Picard plot
fig, axs = plt.subplots(1, 2, figsize=(W, 2.1), constrained_layout=True)
idx = np.arange(1, m + 1)
y_exact = A @ x_s
beta_exact = np.abs(U.T @ y_exact)
axs[0].semilogy(idx, sv, "-", color="k", label="$\\sigma_i$")
axs[0].semilogy(idx, beta_exact, ".", ms=3, color=C["a"], label="$|\\mathbf{u}_i^\\top \\mathbf{y}|$, exact data")
for lev, col, lab in [(1e-3, C["c"], "$10^{-3}$"), (sigma, C["data"], f"${sigma}$")]:
    y = y_exact + lev * rng.standard_normal(m)
    axs[0].semilogy(idx, np.abs(U.T @ y), ".", ms=3, color=col, label=f"$|\\mathbf{{u}}_i^\\top \\mathbf{{y}}|$, noise $\\sigma =$ {lab}")
    axs[1].semilogy(idx, np.abs(U.T @ y) / sv, ".", ms=3, color=col, label=f"noise $\\sigma =$ {lab}")
axs[1].semilogy(idx, beta_exact / sv, ".", ms=3, color=C["a"], label="exact data")
axs[0].set_ylim(1e-10, 10); axs[0].set_xlabel("index $i$"); axs[0].set_title("(a) Picard plot")
axs[0].legend(frameon=False, loc="lower left", ncol=1)
axs[1].set_ylim(1e-4, 1e8); axs[1].set_xlabel("index $i$"); axs[1].set_title("(b) solution coefficients $|\\mathbf{u}_i^\\top \\mathbf{y}| / \\sigma_i$")
axs[1].legend(frameon=False, loc="upper left")
save(fig, "picard.pdf")

# ---------------------------------------------------------------- Fig: sensitivity and Jacobian of the shallow-water map
x, dx = make_grid(200)
b = bottom_smooth(x)
fig, axs = plt.subplots(1, 3, figsize=(W, 1.9), constrained_layout=True)
ax = axs[0]
Fr = froude(b)
ax.plot(x, -Fr**2 / (1 - Fr**2), color=C["a"])
ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$\\partial\\eta / \\partial b$")
ax.set_title("(a) sensitivity $-\\mathrm{Fr}^2/(1 - \\mathrm{Fr}^2)$")
ax.set_ylim(-0.75, 0)
for mm, col, lab in [(200, C["a"], "dense, $m = 200$"), (20, C["data"], "sparse, $m = 20$")]:
    xo = np.linspace(0.5, L - 0.5, mm)
    J = jacobian_fd(lambda bb: surface_at_sensors(bb, x, xo), b)
    svJ = np.linalg.svd(J, compute_uv=False)
    svJ[svJ < 1e-8 * svJ[0]] = 1e-16                  # numerically zero (finite differences): drawn at 1e-16
    print(f"Jacobian m={mm}: sigma_1 {svJ[0]:.2f}, rank {np.sum(svJ > 1e-16)}, smallest nonzero {svJ[svJ > 1e-16].min():.3f}")
    axs[1].semilogy(np.arange(1, len(svJ) + 1), svJ, ".", ms=3, color=col, label=lab)
axs[1].set_ylim(3e-17, 80); axs[1].set_xlabel("index $i$"); axs[1].set_ylabel("$\\sigma_i(\\mathbf{J})$")
axs[1].set_title("(b) singular values of the Jacobian")
axs[1].legend(frameon=False, loc="lower left")
xo = np.linspace(0.5, L - 0.5, 20)
F = lambda bb: surface_at_sensors(bb, x, xo)
lo, hi = xo[9], xo[10]
db = 0.10 * np.exp(-((x - 0.5 * (lo + hi)) / (0.18 * (hi - lo))) ** 2) * ((x > lo + dx) & (x < hi - dx))
print("invisible bump: max |F(b + db) - F(b)| =", np.max(np.abs(F(b + db) - F(b))))
ax = axs[2]
ax.fill_between(x, 0, b, color="0.6", lw=0, label="bottom $b$")
ax.fill_between(x, b, b + db, color=C["data"], alpha=0.5, lw=0, label="invisible change $\\delta b$")
ax.plot(xo, np.zeros(20) - 0.015, "v", color=C["data"], ms=4, clip_on=False, label="sensors")
ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$b$ [m]"); ax.set_ylim(0, 0.32)
ax.set_title("(c) invisible change, $m = 20$")
ax.legend(frameon=False, loc="upper right", fontsize=6.5)
save(fig, "jacobian.pdf")

# ---------------------------------------------------------------- numbers quoted in the text
print(f"kappa(A), n={n}, m={m}, gamma={GAMMA}:", np.linalg.cond(A))
print("sigma_1, sigma_m:", sv[0], sv[-1])
for nu in [5, 10, 20, 30, 40]:
    print(f"attenuation factor at frequency {nu}: {np.exp(-2 * np.pi**2 * GAMMA**2 * nu**2):.2e}")
for lev in [1e-3, sigma]:
    print(f"noise {lev}: exact coefficients above the noise level: {int(np.sum(beta_exact > lev))}; "
          f"index where sigma_i < noise: {int(np.argmax(sv < lev)) + 1}; "
          f"formula sqrt(2 ln(1/s))/(pi gamma): {np.sqrt(2 * np.log(1 / lev)) / (np.pi * GAMMA):.0f}; "
          f"max amplification noise/sigma_m: {lev / sv[-1]:.1e}")
x_ls = np.linalg.lstsq(A, A @ x_p, rcond=None)[0]
print("exact-data pseudo-inverse, piecewise signal: max error", np.max(np.abs(x_ls - x_p)))
Fr = froude(bottom_smooth(make_grid(200)[0]))
print("Froude range:", Fr.min(), Fr.max(), " sensitivity range:", (-Fr**2 / (1 - Fr**2)).min(), (-Fr**2 / (1 - Fr**2)).max())

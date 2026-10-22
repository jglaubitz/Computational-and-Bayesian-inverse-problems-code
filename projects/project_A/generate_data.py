"""
Project A -- data generator.

    data/deconvolution.npz   LINEAR: y = A x + e; the deconvolution example of the lecture notes (n = 128, m = 64,
                             Gaussian kernel of width 0.03, sigma = 0.05, seed 1): exactly the data y_p of
                             code/ch05_figures.py (piecewise-constant signal); the smooth signal and its data y_s are
                             included for completeness.
    data/burgers.npz         NONLINEAR: y = G(u0) + e; inviscid Burgers equation solved to T = 0.25 on n = 100 cells
                             and sampled at m = 20 cells, sigma = 0.01 (the data of the previous project A, seed 0).
The overview figure is written to ../figures/data_A.pdf.

Usage:  python generate_data.py                      # the default data sets
        python generate_data.py --m 32 --sigma 0.1   # a less informative deconvolution data set (other file name)
        python generate_data.py --sigma_burgers 0.05 # a less informative Burgers data set (other file name)
"""
import argparse
import os

import numpy as np

import burgers

GAMMA = 0.03


def deconvolution_operator(n, m, gamma=GAMMA):
    """Midpoint discretization of y(s) = int_0^1 k(s - t) x(t) dt with a Gaussian kernel k of width gamma
    (as code/running_examples.py of the lecture notes). Returns A (m x n), cells t (n,), sensors s (m,)."""
    t = (np.arange(n) + 0.5) / n
    s = (np.arange(m) + 0.5) / m
    A = np.exp(-(s[:, None] - t[None, :]) ** 2 / (2 * gamma**2)) / (gamma * np.sqrt(2 * np.pi)) / n
    return A, t, s


def signal_smooth(t):
    return np.exp(-((t - 0.3) / 0.08) ** 2) + 0.6 * np.exp(-((t - 0.7) / 0.12) ** 2)


def signal_piecewise(t):
    x = np.zeros_like(t)
    x[(t > 0.15) & (t <= 0.4)] = 1.0
    x[(t > 0.55) & (t <= 0.65)] = 0.5
    x[(t > 0.65) & (t <= 0.85)] = 0.8
    return x


def write_burgers(here, sigma, name):
    """Burgers data as in the previous project A (its rng first drew 22 normals for its own deconvolution data)."""
    rng = np.random.default_rng(0)
    rng.standard_normal(22)
    x, dx = burgers.grid()
    u0_true = burgers.true_initial_condition(x)
    obs = burgers.observation_cells(len(x), 20)
    uT_true = burgers.solve(u0_true)
    y_nl = uT_true[obs] + sigma * rng.standard_normal(len(obs))
    f2 = os.path.join(here, "data", name)
    np.savez(f2, x=x, obs_cells=obs, x_obs=x[obs], y=y_nl, u0_true=u0_true, uT_true=uT_true, sigma=sigma,
             T=burgers.T_FINAL, cfl=burgers.CFL, umax=burgers.UMAX)
    print(f"wrote {f2}: n={len(x)}, m={len(obs)}, sigma={sigma}, T={burgers.T_FINAL}")
    return x, obs, u0_true, uT_true, y_nl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma_burgers", type=float, default=0.01, help="Burgers noise level (other values: other file name)")
    a = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "data"), exist_ok=True)
    default = (a.n, a.m, a.sigma, a.seed) == (128, 64, 0.05, 1)
    if a.sigma_burgers != 0.01:                     # only a Burgers data set with another noise level
        write_burgers(here, a.sigma_burgers, f"burgers_sigma{a.sigma_burgers:g}.npz")
        return

    # ---- linear: deconvolution (same order of random draws as the lecture notes: y_s, then y_p)
    A, t, s = deconvolution_operator(a.n, a.m)
    rng = np.random.default_rng(a.seed)
    x_s, x_p = signal_smooth(t), signal_piecewise(t)
    y_s = A @ x_s + a.sigma * rng.standard_normal(a.m)
    y_p = A @ x_p + a.sigma * rng.standard_normal(a.m)
    name = "deconvolution.npz" if default else f"deconvolution_m{a.m}_sigma{a.sigma:g}_seed{a.seed}.npz"
    f1 = os.path.join(here, "data", name)
    np.savez(f1, t=t, s=s, A=A, y=y_p, x_true=x_p, y_smooth=y_s, x_smooth=x_s, sigma=a.sigma, gamma=GAMMA)
    print(f"wrote {f1}: n={a.n}, m={a.m}, sigma={a.sigma}, seed={a.seed}")
    if not default:
        return

    # ---- nonlinear: Burgers
    x, obs, u0_true, uT_true, y_nl = write_burgers(here, 0.01, "burgers.npz")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 8})
        fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.2), constrained_layout=True)
        ax = axs[0]
        ax.plot(t, x_p, "k-", lw=1.0, label="$x$ (unknown)")
        ax.plot(s, y_p, "o", ms=2.5, color="tab:red", label="data $y$")
        ax.set_ylim(-0.3, 1.3); ax.set_xlabel("$t$")
        ax.set_title(f"(a) deconvolution (linear), $m = {a.m}$, $n = {a.n}$")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax = axs[1]
        ax.step(x, u0_true, "k-", lw=1.0, where="mid", label="$u_0$ (unknown)")
        ax.plot(x, uT_true, "-", color="tab:blue", lw=1.0, label=f"$u(\\cdot, T)$, $T = {burgers.T_FINAL}$")
        ax.plot(x[obs], y_nl, "o", ms=2.5, color="tab:red", label="data $y$")
        ax.set_ylim(0, 1.3); ax.set_xlabel("$x$")
        ax.set_title(f"(b) Burgers (nonlinear), $m = {len(obs)}$, $n = {len(x)}$")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        figdir = os.path.join(here, "..", "figures")
        os.makedirs(figdir, exist_ok=True)
        fig.savefig(os.path.join(figdir, "data_A.pdf"))
        print("wrote figures/data_A.pdf")
    except ImportError:
        pass


if __name__ == "__main__":
    main()

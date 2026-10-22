"""
Project C -- data generator.

Warm-up (steady channel flow, algebraic forward model), two topographies and two sensor layouts:
    data/steady_smooth.npz, data/steady_piecewise.npz                 dense:  m = 200 sensors
    data/steady_smooth_sparse.npz, data/steady_piecewise_sparse.npz   sparse: m = 40 sensors
Topography (1) is smooth (two Gaussian bumps), topography (2) is piecewise constant (four jumps).
Steady subcritical flow (q = 4.42 m^2/s, outflow depth 2 m); the free-surface elevation eta = h + b is
measured at m equispaced sensors with noise 0.01 m. The data are computed from the continuous truth and the
exact Bernoulli relation at the sensor positions (no grid), so that an inversion on n = 100 or n = 200 cells
commits no inverse crime. The truth is stored on a fine grid and on the n = 100 and n = 200 grids.

Target (time-dependent flume flow, finite-volume forward model of flume.py), two bathymetries:
    data/flume_hill.npz    smooth Gaussian hill (height 0.2 m at x = 4 m), the shape of the TUHH experiment
    data/flume_boxes.npz   piecewise constant: two boxes (0.15 m on (3.6, 4.8] m, 0.08 m on (6.1, 7.3] m)
Synthetic wave maker eta(1.5 m, t) = H0 + 0.02 sin(2 pi t / 2.5 s) (switched on over one period), still-water
depth H0 = 0.4 m, friction kappa = 0.2, sensors at x = 3.5, 5.5, 7.5 m sampled at 100 Hz for 10 s (3000 data),
noise 5 % of the wave amplitude (1 mm), as in Stietz et al. (2026). The data are generated on N = 2160 cells
(discretization error about 0.06 mm rms); an inversion on N = 270 cells (0.5 mm rms) or N = 540 (0.2 mm rms)
commits no inverse crime. Stored: the three sensor records (noisy and clean), the wave-maker record at 1.5 m
(clean and with the same noise, at 100 Hz), and the truth on the fine grid and on n = 64 and n = 128 nodes.

Overview figure: ../figures/data_C.pdf.

Usage: python generate_data.py [--seed 0] [--noise 0.01] [--no-flume]
"""
import argparse
import os
import numpy as np

from swe import (G, L, Q, H_OUT, TOPOGRAPHIES, make_grid, steady_depth)
import flume


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.01, help="measurement noise std of the steady data [m]")
    p.add_argument("--noise-flume", type=float, default=0.05 * flume.A_WAVE,
                   help="measurement noise std of the flume data [m] (default: 5 %% of the wave amplitude)")
    p.add_argument("--no-flume", action="store_true", help="skip the (slow, about 30 s) flume data sets")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "data"), exist_ok=True)

    # ---------------- warm-up: steady channel
    x_fine, _ = make_grid(2000)
    x_100, _ = make_grid(100)
    x_200, _ = make_grid(200)
    steady = {}
    for name, bottom in TOPOGRAPHIES.items():
        b_fine = bottom(x_fine)
        h_fine = steady_depth(b_fine, Q, H_OUT, b_out=0.0)
        eta_fine = h_fine + b_fine
        for m, suffix in [(200, ""), (40, "_sparse")]:
            x_obs = np.linspace(0.5, L - 0.5, m)
            b_obs = bottom(x_obs)                               # exact truth at the sensors
            eta_exact = steady_depth(b_obs, Q, H_OUT, b_out=0.0) + b_obs
            eta_obs = eta_exact + args.noise * rng.standard_normal(m)
            f = os.path.join(here, "data", f"steady_{name}{suffix}.npz")
            np.savez(f, x_obs=x_obs, eta_obs=eta_obs, noise_std=args.noise, q=Q, h_out=H_OUT, L=L, g=G,
                     x_fine=x_fine, b_true_fine=b_fine, eta_true_fine=eta_fine,
                     x_100=x_100, b_true_100=bottom(x_100), x_200=x_200, b_true_200=bottom(x_200))
            if suffix == "":
                steady[name] = (b_fine, eta_fine, x_obs, eta_obs)
            print(f"wrote {f}: {m} sensors, noise {args.noise} m, max Froude {np.max(Q / h_fine / np.sqrt(G * h_fine)):.2f}")

    # ---------------- target: time-dependent flume
    flume_rec = {}
    if not args.no_flume:
        N = 2160
        x, dx = flume.cells(N)
        dt, k, n_samples = flume.time_grid(N)
        times = np.arange(1, n_samples + 1) / flume.FS
        t_left = np.arange(0, n_samples + 1) / flume.FS
        eta_left_true = flume.eta_left_synthetic(t_left)
        x_64, x_128 = flume.nodes(64), flume.nodes(128)
        flume_rec["flat"] = flume.forward(flume.bottom_flat(x), flume.eta_left_synthetic, N)
        for name, bottom in flume.BATHYMETRIES.items():
            b = bottom(x)
            eta_clean = flume.forward(b, flume.eta_left_synthetic, N)
            eta_obs = eta_clean + args.noise_flume * rng.standard_normal(eta_clean.shape)
            eta_left_obs = eta_left_true + args.noise_flume * rng.standard_normal(eta_left_true.shape)
            f = os.path.join(here, "data", f"flume_{name}.npz")
            np.savez(f, times=times, x_sensors=flume.SENSORS, eta_obs=eta_obs, eta_clean=eta_clean,
                     noise_std=args.noise_flume, t_left=t_left, eta_left_obs=eta_left_obs, eta_left_true=eta_left_true,
                     H0=flume.H0, kappa=flume.KAPPA, g=G, x_L=flume.X_L, x_R=flume.X_R, fs=flume.FS,
                     x_fine=x, b_true_fine=b, x_64=x_64, b_true_64=bottom(x_64), x_128=x_128, b_true_128=bottom(x_128))
            flume_rec[name] = eta_clean
            sig = np.sqrt(np.mean((eta_clean - flume_rec["flat"]) ** 2, axis=0))
            print(f"wrote {f}: {eta_obs.shape[0]} samples x {eta_obs.shape[1]} sensors, noise {args.noise_flume * 1e3:.2f} mm, "
                  f"rms signal of the bathymetry at the sensors {np.round(sig * 1e3, 2)} mm")

    # ---------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 8})
        nrows = 2 if flume_rec else 1
        fig, axs = plt.subplots(nrows, 2, figsize=(6.6, 2.3 * nrows), constrained_layout=True)
        axs = np.atleast_2d(axs)
        for ax, (name, (b_fine, eta_fine, x_obs, eta_obs)), lab in zip(axs[0], steady.items(), ["(a) smooth", "(b) piecewise-constant"]):
            ax.fill_between(x_fine, 0, b_fine, color="0.75", lw=0, label="bottom $b$ (unknown)")
            ax.plot(x_fine, eta_fine, "-", color="tab:blue", lw=1.2, label="free surface $\\eta$")
            ax.plot(x_obs, eta_obs, ".", color="tab:red", ms=2.0, label=f"measurements ($m={len(x_obs)}$)")
            ax.set_xlabel("$x$ [m]")
            ax.set_ylim(0, 2.5)
            ax.set_title(f"{lab} bottom (steady channel)" if name == "smooth" else f"{lab} bottom (channel)")
        axs[0, 0].set_ylabel("elevation [m]")
        axs[0, 1].set_ylabel("elevation [m]")
        axs[0, 0].legend(frameon=False, fontsize=7, loc="center right")
        if flume_rec:
            ax = axs[1, 0]
            xf = np.linspace(flume.X_L, 9.0, 1500)
            ax.fill_between(xf, 0, flume.bottom_hill(xf), color="0.75", lw=0, label="hill")
            ax.plot(xf, flume.bottom_boxes(xf), "-", color="k", lw=1.0, label="two boxes")
            ax.axhline(flume.H0, color="tab:blue", lw=1.2, label="still water $H_0$")
            for xs, lab in zip(np.concatenate([[flume.X_L], flume.SENSORS]), ["$s_1$ (BC)", "$s_2$", "$s_3$", "$s_4$"]):
                ax.plot([xs, xs], [0, flume.H0 + 0.05], ":", color="tab:red", lw=1.0)
                ax.text(xs, flume.H0 + 0.065, lab, color="tab:red", fontsize=6.5, ha="center")
            ax.set_xlim(1.0, 9.0)
            ax.set_ylim(0, flume.H0 + 0.2)
            ax.set_xlabel("$x$ [m]")
            ax.set_ylabel("elevation [m]")
            ax.set_title("(c) flume: bathymetries and sensors")
            ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=3, columnspacing=1.0, handlelength=1.6)
            ax = axs[1, 1]
            tt = times
            sel = (tt >= 3.0) & (tt <= 6.5)
            j = 2                                           # sensor at 7.5 m
            d = np.load(os.path.join(here, "data", "flume_hill.npz"))
            ax.plot(tt[sel][::3], (d["eta_obs"][sel, j][::3] - flume.H0) * 1e3, ".", color="tab:red", ms=2.0, label="noisy data (hill)")
            ax.plot(tt[sel], (flume_rec["flat"][sel, j] - flume.H0) * 1e3, "-", color="0.6", lw=1.0, label="flat floor")
            ax.plot(tt[sel], (flume_rec["hill"][sel, j] - flume.H0) * 1e3, "-", color="tab:blue", lw=1.2, label="hill")
            ax.plot(tt[sel], (flume_rec["boxes"][sel, j] - flume.H0) * 1e3, "--", color="k", lw=1.0, label="two boxes")
            ax.set_xlabel("$t$ [s]")
            ax.set_ylabel("$\\eta - H_0$ at $x = 7.5$ m [mm]")
            ax.set_ylim(-16, 22)
            ax.set_title("(d) flume: records at the fourth sensor")
            ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=4, columnspacing=0.8, handlelength=1.6)
        figdir = os.path.join(here, "..", "figures")
        os.makedirs(figdir, exist_ok=True)
        fig.savefig(os.path.join(figdir, "data_C.pdf"))
        print("wrote figures/data_C.pdf")
    except ImportError:
        pass


if __name__ == "__main__":
    main()

"""
Project B -- data generator: synthetic isothermal reactor experiments for the trimerization mechanism
(test case 1 of Ji & Deng (2021) and Doeppel & Votsmeier (2023)).

Each experiment: random initial concentrations c0(A), c0(A2) ~ U(0.2, 1.2), the other species 0;
the true mechanism is integrated to t = 40 and all five species are "measured" at n_t equidistant
times with additive Gaussian noise whose standard deviation is `noise` times the range of each
species over the whole data set (the normalization used in the AC-CRNN paper).

Default files (run with no arguments):
    data/trimerization_20exp_5pct.npz     20 experiments, 10 time points, 5 % noise   (standard)
    data/trimerization_5exp_5pct.npz       5 experiments, 10 time points, 5 % noise   (scarce data)
    data/trimerization_20exp_20pct.npz    20 experiments, 10 time points, 20 % noise  (noisy data)
and the overview figure ../figures/data_B.pdf.

Usage: python generate_data.py                      (the three default files)
       python generate_data.py --n_exp 10 --noise 0.1 --seed 3 --out data/my_data.npz
"""
import argparse
import os
import numpy as np

from acrnn import (MOLECULAR_MATRIX_TRIMERIZATION, SPECIES_TRIMERIZATION, atom_conservation_basis,
                   true_params_trimerization, simulate_all)


def make_dataset(n_exp, n_t, noise, seed, t_final=40.0):
    rng = np.random.default_rng(seed)
    N = MOLECULAR_MATRIX_TRIMERIZATION
    n_s, n_r = N.shape[1], 4
    B, key, dep = atom_conservation_basis(N)
    params_true, NU_true, a_true, k_true = true_params_trimerization(B)

    c0 = np.zeros((n_exp, n_s))
    c0[:, 0] = rng.uniform(0.2, 1.2, size=n_exp)      # A
    c0[:, 1] = rng.uniform(0.2, 1.2, size=n_exp)      # A2
    t_eval = np.linspace(0.0, t_final, n_t + 1)[1:]   # n_t equidistant times, excluding t = 0

    c_clean = simulate_all(params_true, c0, t_eval, B, n_s, n_r)       # (n_exp, n_t, n_s)
    scale = c_clean.reshape(-1, n_s).max(axis=0) - c_clean.reshape(-1, n_s).min(axis=0)
    c_obs = c_clean + noise * scale * rng.standard_normal(c_clean.shape)

    return dict(t_eval=t_eval, c0=c0, c_obs=c_obs, c_clean=c_clean, noise_level=noise,
                noise_std=noise * scale, species=np.array(SPECIES_TRIMERIZATION),
                molecular_matrix=N, B=B, key_species=np.array(key),
                NU_true=NU_true, a_true=a_true, k_true=k_true, params_true=params_true)


def save(d, fname):
    np.savez(fname, **d)
    print(f"wrote {fname}: {d['c0'].shape[0]} experiments x {len(d['t_eval'])} times x "
          f"{len(d['species'])} species, noise {100*d['noise_level']:.0f} % of range")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_exp", type=int, default=None)
    p.add_argument("--n_t", type=int, default=10)
    p.add_argument("--noise", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "data"), exist_ok=True)

    if args.n_exp is not None or args.noise is not None:
        d = make_dataset(args.n_exp or 20, args.n_t, args.noise if args.noise is not None else 0.05, args.seed)
        save(d, args.out or os.path.join(here, "data", f"trimerization_{d['c0'].shape[0]}exp_{int(100*d['noise_level'])}pct.npz"))
        return

    std = make_dataset(20, args.n_t, 0.05, args.seed)
    save(std, os.path.join(here, "data", "trimerization_20exp_5pct.npz"))
    save(make_dataset(5, args.n_t, 0.05, args.seed + 1), os.path.join(here, "data", "trimerization_5exp_5pct.npz"))
    save(make_dataset(20, args.n_t, 0.20, args.seed + 2), os.path.join(here, "data", "trimerization_20exp_20pct.npz"))

    # overview figure: three experiments of the standard data set
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 8})
        B, n_s, n_r = std["B"], len(std["species"]), 4
        t_fine = np.linspace(0, 40, 200)
        c_fine = simulate_all(std["params_true"], std["c0"][:3], t_fine, B, n_s, n_r)
        fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35), constrained_layout=True, sharey=True)
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
        markers = ["o", "s", "^", "D", "v"]
        from matplotlib.lines import Line2D
        for e, ax in enumerate(axs):
            for i, sp in enumerate(std["species"]):
                ax.plot(t_fine, c_fine[e, :, i], linestyle=styles[i], color=colors[i], lw=1.0)
                ax.plot(std["t_eval"], std["c_obs"][e, :, i], linestyle="none", marker=markers[i], color=colors[i], ms=3.0, mfc="white", mew=0.9)
            ax.set_title(f"exp. {e+1}: $c_0(A)={std['c0'][e,0]:.2f}$, $c_0(A_2)={std['c0'][e,1]:.2f}$", fontsize=7.5)
            ax.set_xlabel("$t$")
        axs[0].set_ylabel("concentration")
        handles = [Line2D([0], [0], color=colors[i], lw=1.2, linestyle=styles[i], marker=markers[i], ms=3.5, mfc="white", mew=0.9, label=sp)
                   for i, sp in enumerate(std["species"])]
        fig.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.13))
        figdir = os.path.join(here, "..", "figures")
        os.makedirs(figdir, exist_ok=True)
        fig.savefig(os.path.join(figdir, "data_B.pdf"), bbox_inches="tight")
        print("wrote figures/data_B.pdf")
    except ImportError:
        pass


if __name__ == "__main__":
    main()

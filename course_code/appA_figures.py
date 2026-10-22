"""Figure of Appendix A (probability refresher). Output: ../figures/appA/distributions.pdf

    python appA_figures.py

Densities of the normal, Laplace, Student-t (two degrees of freedom), and Cauchy distributions with scale one, their
tails on a logarithmic axis, and running means of independent draws: the law of large numbers at the rate 1/sqrt(N)
for the normal distribution, a slower convergence for the Student-t distribution with infinite variance, and no
convergence at all for the Cauchy distribution, which has no mean.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "appA")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "lines.linewidth": 1.1})
W = 6.3
DISTS = [("normal", stats.norm(), "tab:blue", "-"),
         ("Laplace", stats.laplace(), "tab:orange", "--"),
         ("Student-$t$, 2 dof", stats.t(df=2), "tab:green", "-."),
         ("Cauchy", stats.cauchy(), "tab:purple", ":")]


def fig_distributions():
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), constrained_layout=True)
    x = np.linspace(-5, 5, 1001)
    for name, d, col, ls in DISTS:
        axs[0].plot(x, d.pdf(x), ls, color=col, label=name)
    axs[0].set_xlim(-5, 5); axs[0].set_ylim(0, 0.55); axs[0].set_xlabel("$x$")
    axs[0].set_title("(a) densities, scale one")
    x = np.linspace(0, 30, 1201)
    for name, d, col, ls in DISTS:
        axs[1].semilogy(x, d.pdf(x), ls, color=col, label=name)
    axs[1].set_xlim(0, 30); axs[1].set_ylim(1e-8, 10); axs[1].set_xlabel("$x$")
    axs[1].set_title("(b) the same on a log scale"); axs[1].legend(frameon=False, fontsize=5.6, loc="upper right", handlelength=1.8)
    rng = np.random.default_rng(3)
    N = np.arange(1, 10**5 + 1)
    for name, d, col, ls in (DISTS[0], DISTS[2], DISTS[3]):
        draws = d.rvs(size=N[-1], random_state=rng)
        axs[2].semilogx(N, np.cumsum(draws) / N, ls if ls != ":" else "-", color=col, lw=1.0, label=name)
        tail = np.cumsum(draws) / N
        print(f"{name}: running mean after 1e3, 1e4, 1e5 draws {tail[[999, 9999, 99999]].round(4).tolist()}")
    axs[2].fill_between(N, -2 / np.sqrt(N), 2 / np.sqrt(N), color="0.85", lw=0, label="$\\pm 2/\\sqrt{N}$")
    axs[2].axhline(0, color="k", lw=0.6)
    axs[2].set_xlim(1, N[-1]); axs[2].set_ylim(-2.5, 2.5); axs[2].set_xlabel("number of draws $N$")
    axs[2].set_title("(c) running means of $N$ draws"); axs[2].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=5.6, loc="lower right", handlelength=1.8)
    fig.savefig(os.path.join(OUT, "distributions.pdf"))
    plt.close(fig)
    print("wrote distributions.pdf")
    # numbers quoted in Appendix A
    t2 = stats.t(df=2, scale=np.sqrt(1e-3))
    print(f"P(|z| > 0.1) for the marginal of the hierarchical prior (beta = 1, vartheta = 1e-3): {2 * t2.sf(0.1):.4f}")
    print(f"P(|x| > 3) normal {2 * stats.norm.sf(3):.4f}, Laplace {2 * stats.laplace.sf(3):.4f}, "
          f"t2 {2 * stats.t(df=2).sf(3):.4f}, Cauchy {2 * stats.cauchy.sf(3):.4f}")


if __name__ == "__main__":
    fig_distributions()

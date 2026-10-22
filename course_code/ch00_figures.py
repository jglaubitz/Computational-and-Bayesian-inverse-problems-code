"""Figures of the preliminary chapter (notation and running examples). Output: ../figures/ch00/

    python ch00_figures.py

    deconvmatrix.pdf   what the deconvolution matrix A does: the matrix, one row, one column
    channel.tikz       schematic of the channel of running example 2 (coordinates computed from the smooth bottom)
    bathydata.pdf      the two test bottoms, their free surfaces, and data from 200 and 20 sensors
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from running_examples import (deconvolution_operator, kernel, signal_piecewise, make_grid, steady_depth,
                              surface_at_sensors, froude, BOTTOMS, GAMMA, N_CELLS, M_OBS, L, Q, H_OUT, G)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "ch00")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "lines.linewidth": 1.1})
W = 6.3                                   # text width in inches
C = {"truth": "k", "a": "tab:blue", "b": "tab:orange", "c": "tab:green", "d": "tab:purple", "data": "tab:red"}
SIGMA_B = 0.01                            # noise level of the bathymetry data: 1 cm


def save(fig, name, **kw):
    fig.savefig(os.path.join(OUT, name), **kw)
    plt.close(fig)
    print("wrote", name)


# ====================================================================== Example 1: what the matrix A does
n, m = N_CELLS, M_OBS
A, t, s = deconvolution_operator(n, m)
x_p = signal_piecewise(t)
i_row, j_col = 24, 89                     # s_24 = 0.383 (next to the edge at 0.4), t_89 = 0.699
print(f"row {i_row}: s_i = {s[i_row]:.4f}, sum of weights {A[i_row].sum():.3f}; "
      f"row 0: s_0 = {s[0]:.4f}, sum {A[0].sum():.3f}; column {j_col}: t_j = {t[j_col]:.4f}")
print(f"largest entry {A.max():.4f}; entries with |s_i - t_j| > 4 gamma are below {kernel(4 * GAMMA) / n:.1e}; "
      f"kappa(A) = {np.linalg.cond(A):.2e}")

fig, axs = plt.subplots(1, 3, figsize=(W, 1.95), constrained_layout=True)
ax = axs[0]
im = ax.imshow(A, cmap="Greys", aspect="auto", extent=[0, 1, 1, 0], interpolation="nearest")
ax.axhline(s[i_row], color=C["b"], lw=1.0)
ax.axvline(t[j_col], color=C["c"], lw=1.0)
ax.text(0.97, s[i_row] - 0.03, f"row $i = {i_row + 1}$", color=C["b"], ha="right", va="bottom", fontsize=7)
ax.text(t[j_col] + 0.02, 0.06, f"column\n$j = {j_col + 1}$", color=C["c"], ha="left", va="top", fontsize=7)
ax.set_xlabel("cell center $t_j$"); ax.set_ylabel("observation point $s_i$")
ax.set_xticks([0, 0.5, 1])
ax.set_title(f"(a) the matrix $\\mathbf{{A}}$ (${m} \\times {n}$)")
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, ticks=[0, 0.05, 0.1])
cb.ax.tick_params(labelsize=7)

ax = axs[1]
ax.fill_between(t, 0, 0.11 * x_p, step="mid", color="0.85", lw=0, label="signal $x$ (scaled)")
for i, col, lab in [(i_row, C["b"], f"row {i_row + 1}"), (0, C["a"], "row 1")]:
    keep = A[i] > 1e-4                   # entries below 1e-4 are not drawn
    ml, sl, bl = ax.stem(t[keep], A[i, keep], linefmt="-", markerfmt="o", basefmt=" ")
    plt.setp(sl, color=col, lw=0.6); plt.setp(ml, color=col, ms=2.0)
    ml.set_label(f"{lab}, weights sum to {A[i].sum():.2f}")
ax.set_xlim(-0.02, 0.6); ax.set_ylim(0, 0.19)
ax.set_xlabel("$t_j$"); ax.set_ylabel("$A_{ij}$")
ax.set_title("(b) rows: the weights of one datum")
ax.set_xticks([0, 0.2, 0.4, 0.6])
ax.legend(frameon=False, loc="upper right", fontsize=6.5, handlelength=1.2)

ax = axs[2]
tt = np.linspace(0.5, 0.9, 400)
ax.plot(tt, kernel(tt - t[j_col]) / n, color="0.6", lw=0.8, ls="--", label="$k(s - t_j)/n$")
keep = A[:, j_col] > 1e-4
ml, sl, bl = ax.stem(s[keep], A[keep, j_col], linefmt="-", markerfmt="o", basefmt=" ")
plt.setp(sl, color=C["c"], lw=0.6); plt.setp(ml, color=C["c"], ms=2.0)
ml.set_label(f"column {j_col + 1}")
ax.set_xlim(0.5, 0.9); ax.set_ylim(0, 0.19)
ax.set_xlabel("$s_i$"); ax.set_xticks([0.6, 0.7, 0.8])
ax.set_title("(c) a column: spike data")
ax.legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.2)
ax.set_yticklabels([])
save(fig, "deconvmatrix.pdf")

# ====================================================================== Example 2: schematic of the channel (TikZ)
x, dx = make_grid(200)
b = BOTTOMS["smooth"](x)
h = steady_depth(b, b_out=b[-1])
eta = h + b
x_obs20 = np.linspace(0.5, L - 0.5, 20)
SX = 0.48                                 # cm per meter along the channel
SD = 1.4                                  # cm per meter of mean depth
SE = 7.0                                  # cm per meter for the bottom and the surface deviation (exaggerated)
yb = lambda bb: SE * bb
ye = lambda ee: SD * H_OUT + SE * (ee - H_OUT)
xs = np.concatenate([[0.0], x, [L]])
bs = np.interp(xs, x, b)
es = np.interp(xs, x, eta)
bot = " ".join(f"({xx:.3f},{yb(bb):.3f})" for xx, bb in zip(xs[::2], bs[::2])) + f" ({L:.3f},{yb(bs[-1]):.3f})"
sur = " ".join(f"({xx:.3f},{ye(ee):.3f})" for xx, ee in zip(xs[::2], es[::2])) + f" ({L:.3f},{ye(es[-1]):.3f})"
sur_rev = " ".join(f"({xx:.3f},{ye(ee):.3f})" for xx, ee in zip(xs[::-2], es[::-2]))
sensors = ", ".join(f"{xo:.3f}/{ye(np.interp(xo, x, eta)):.3f}" for xo in x_obs20)
i9 = np.argmin(np.abs(x - 9.0))
x9, b9, e9 = x[i9], yb(b[i9]), ye(eta[i9])
x3, e3 = 3.0, ye(np.interp(3.0, x, eta))
i22 = np.argmin(np.abs(x - 22.0))
x22, b22, e22 = x[i22], yb(b[i22]), ye(eta[i22])
cells = ", ".join(f"{k * dx:.4f}" for k in range(201))
tikz = rf"""% Running example 2: the channel. Generated by code/ch00_figures.py from the smooth bottom of code/running_examples.py.
% Horizontal: {SX} cm per meter. Vertical: {SD} cm per meter of mean depth, but {SE} cm per meter for the bottom and the
% deviation of the surface from the outflow level (exaggerated {SE / SD:.1f} times).
\begin{{tikzpicture}}[x={SX}cm, y=1cm, line cap=round, line join=round, every node/.style={{font=\footnotesize}}]
  \definecolor{{tabblue}}{{RGB}}{{31,119,180}}
  \definecolor{{tabred}}{{RGB}}{{214,39,40}}
  % water and ground
  \fill[BoxBlue] plot coordinates {{{bot}}} -- plot coordinates {{{sur_rev}}} -- cycle;
  \fill[BoxGray-dark] (0,-0.35) -- plot coordinates {{{bot}}} -- ({L},-0.35) -- cycle;
  \draw[thick] plot coordinates {{{bot}}};
  \draw[tabblue, very thick] plot coordinates {{{sur}}};
  % outflow level
  \draw[tabblue, densely dashed, thin] (0,{ye(H_OUT):.3f}) -- ({L},{ye(H_OUT):.3f});
  % sensors
  \foreach \xo/\ye in {{{sensors}}} {{
    \draw[tabred, thin] (\xo,\ye) -- (\xo,{ye(H_OUT) + 0.42:.3f});
    \fill[tabred] (\xo,{ye(H_OUT) + 0.42:.3f}) -- ++(-0.13,0.17) -- ++(0.26,0) -- cycle;
  }}
  \node[tabred, anchor=south east, inner sep=1pt] at ({L},{ye(H_OUT) + 0.62:.3f}) {{sensors at $x_{{\mathrm{{obs}}}}$, here $m = 20$}};
  % inflow
  \foreach \yy in {{0.5,1.1,1.7,2.3}} \draw[->, thick, AccentDark] (-2.8,\yy) -- (-0.4,\yy);
  \node[AccentDark, align=left, anchor=south west, inner sep=1pt] at (-2.8,{ye(H_OUT) + 0.62:.3f}) {{inflow, discharge\\$q = {Q}\,\mathrm{{m^2/s}}$}};
  % flow direction
  \draw[->, thick] ({x3 + 0.5:.3f},{0.55 * e3:.3f}) -- ({x3 + 4.5:.3f},{0.55 * e3:.3f});
  \node[anchor=south] at ({x3 + 2.5:.3f},{0.55 * e3:.3f}) {{flow}};
  % bottom at the first bump, depth near the outflow, surface
  \draw[<->, thin] ({x9:.3f},0) -- node[pos=0.3, left, inner sep=1.5pt] {{$b(x)$}} ({x9:.3f},{b9:.3f});
  \draw[<->, thin] ({x22:.3f},{b22:.3f}) -- node[right, inner sep=1.5pt] {{$h(x)$}} ({x22:.3f},{e22:.3f});
  \node[tabblue, anchor=north west, inner sep=1pt] at (0.8,{ye(H_OUT) - 0.08:.3f}) {{$\eta(x) = h(x) + b(x)$}};
  % outflow
  \draw[<->, thin] ({L + 0.7:.3f},0) -- node[right, align=left, inner sep=2pt] {{outflow\\depth\\$h_{{\mathrm{{out}}}} = {H_OUT:g}$\,m\\bottom\\$b_n = 0$}} ({L + 0.7:.3f},{ye(H_OUT):.3f});
  % axis with the cell boundaries
  \draw[thin] (0,-0.35) -- ({L},-0.35);
  \foreach \xc in {{{cells}}} \draw[very thin, black!55] (\xc,-0.35) -- (\xc,-0.42);
  \foreach \xx in {{0,5,10,15,20,25}} \draw (\xx,-0.35) -- (\xx,-0.5) node[below, inner sep=1pt] {{\scriptsize $\xx$}};
  \node[anchor=north] at ({L / 2:.3f},-0.78) {{$x$ [m], $n = 200$ cells of width $\Delta x = {100 * dx:g}$\,cm}};
  % Bernoulli
  \node[draw, fill=white, rounded corners=2pt, inner sep=3pt, align=center] at (13.2,1.3)
    {{in every cell\\$\tfrac{{q^2}}{{2h^2}} + g\,(h + b) = E$}};
\end{{tikzpicture}}
"""
with open(os.path.join(OUT, "channel.tikz"), "w") as f:
    f.write(tikz)
print("wrote channel.tikz")

# ====================================================================== Example 2: the two bottoms and their data
rng = np.random.default_rng(2)
E = Q**2 / (2 * H_OUT**2) + G * H_OUT
h_c = (Q**2 / G) ** (1 / 3)
print(f"E = {E:.4f} m^2/s^2, critical depth {h_c:.3f} m, a flat bottom above {E / G - 1.5 * h_c:.3f} m has no "
      f"subcritical steady state; Froude number at the outflow {Q / H_OUT / np.sqrt(G * H_OUT):.3f}")
fig, axs = plt.subplots(2, 2, figsize=(W, 2.65), sharex=True, constrained_layout=True,
                        gridspec_kw={"height_ratios": [1, 1.35]})
for k, (name, title) in enumerate([("smooth", "smooth bottom"), ("piecewise", "piecewise-constant bottom")]):
    b = BOTTOMS[name](x)
    h = steady_depth(b, b_out=b[-1])
    eta = h + b
    Fr = froude(b)
    print(f"{name}: max b = {100 * b.max():.1f} cm, surface deviation in [{100 * (eta - H_OUT).min():.2f}, "
          f"{100 * (eta - H_OUT).max():.2f}] cm, Froude number in [{Fr.min():.3f}, {Fr.max():.3f}]")
    ax = axs[0, k]
    ax.fill_between(x, 0, 100 * b, color="0.8", lw=0, step=None)
    ax.plot(x, 100 * b, color=C["truth"], label="bottom $b$")
    ax.plot(x_obs20, np.full(20, -2.5), "v", color=C["data"], ms=3, clip_on=False, label="sparse sensors ($m = 20$)")
    ax.set_ylim(-4, 24); ax.set_yticks([0, 10, 20])
    ax.set_title(f"({'ab'[k]}) {title}")
    if k == 0:
        ax.set_ylabel("$b$ [cm]")
    ax = axs[1, k]
    for mm, sty in [(200, dict(marker=".", ls="none", color=C["b"], ms=2.5, alpha=0.8)),
                    (20, dict(marker="v", ls="none", color=C["data"], ms=3.5))]:
        xo = np.linspace(0.5, L - 0.5, mm)
        y = surface_at_sensors(b, x, xo) + SIGMA_B * rng.standard_normal(mm)
        ax.plot(xo, 100 * (y - H_OUT), label=f"data, $m = {mm}$", **sty)
    ax.plot(x, 100 * (eta - H_OUT), color=C["a"], label="$\\eta - h_{\\mathrm{out}}$, noise-free")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylim(-13, 4.5); ax.set_xlim(0, L)
    ax.set_title(f"({'cd'[k]}) surface and data, noise $\\sigma = 1$ cm")
    ax.set_xlabel("$x$ [m]")
    if k == 0:
        ax.set_ylabel("$\\eta - h_{\\mathrm{out}}$ [cm]")
h0, l0 = axs[0, 0].get_legend_handles_labels()
h1, l1 = axs[1, 0].get_legend_handles_labels()
fig.legend(h0 + h1, l0 + l1, frameon=False, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
save(fig, "bathydata.pdf", bbox_inches="tight")

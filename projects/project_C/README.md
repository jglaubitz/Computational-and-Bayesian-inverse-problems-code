# Project C — starter code

| File | Content |
|---|---|
| `swe.py` | Shared numerics. Increment parametrization (`bottom_to_increments`, `increments_to_bottom(z, reference="right"/"left")`), the vectorized subcritical steady state `steady_depth` and the forward operator `surface_at_sensors(b, x, x_obs)` of the warm-up (≈ 0.5 ms for n = 200), the two warm-up topographies, and the well-balanced finite-volume scheme (`fv_rhs`, `rk2_step`: hydrostatic reconstruction, HLL flux, first or second order (MUSCL/minmod + SSP-RK2), linear friction; exact for lakes at rest). `simulate` is the adaptive-step channel solver for cross-checks. All of it is written for `xp = numpy` or `xp = jax.numpy`. |
| `flume.py` | The target scenario: configuration of the wave flume of Angel et al. (2024) / Stietz et al. (2026) (domain [1.5, 15] m, sensors at 3.5, 5.5, 7.5 m, 100 Hz, 10 s, κ = 0.2, left boundary driven by the surface record at 1.5 m, wall on the right), the two bathymetries `bottom_hill`, `bottom_box`, the synthetic wave maker, nodes/cells and the interpolation `to_cells`, and the forward model `forward(b_cells, eta_left, N)` (NumPy, ≈ 0.7 s for N = 270) and `forward_jax(N)` (jitted, differentiable, ≈ 60 ms per evaluation, ≈ 0.6 s per gradient for N = 270; needs `pip install jax`). |
| `priors.py` | Gaussian prior on the increments (log-density, whitening), GSBL prior with the generalized gamma hyper-prior (`GSBL_HYPERPRIOR = (r, beta, vartheta) = (-1, 1, 1e-4)`), the hierarchical transport map `map_forward`/`map_inverse`, the prior-normalized log-posterior, prior samplers, the closed-form IAS θ-update, and the baseline prior of Stietz et al. on the nodal values (`baseline_log_prior`, `baseline_proposal_chol`, constants in `BASELINE`). |
| `generate_data.py` | Reproduces all data sets in `data/` from fixed seeds (bit-for-bit with NumPy 2.5 and SciPy 1.18; other versions may differ in the last digits) and the overview figure `../figures/data_C.pdf` (about 30 s). `--no-flume` skips the finite-volume runs; `--seed`, `--noise`, `--noise-flume` create replicates. |

Data files (provided in `data/`):

* Warm-up (steady channel): `steady_smooth.npz`, `steady_piecewise.npz` (dense, m = 200 sensors) and `steady_smooth_sparse.npz`, `steady_piecewise_sparse.npz` (m = 40). Each contains `x_obs`, `eta_obs`, `noise_std` (0.01 m), `q`, `h_out`, `L`, `g`, and the truth for verification only: `x_fine`, `b_true_fine`, `eta_true_fine`, `x_100`, `b_true_100`, `x_200`, `b_true_200`. The data were computed from the continuous truth at the sensor positions (no grid): reconstructions on n = 100 or n = 200 commit no inverse crime.
* Target (flume): `flume_hill.npz`, `flume_box.npz`. Each contains the records of the three data sensors `eta_obs` (1000 × 3, noise std 1 mm = 5 % of the wave amplitude) and `eta_clean`, `times`, `x_sensors`, the wave-maker record at 1.5 m at 100 Hz (`t_left`, `eta_left_true`, `eta_left_obs`), the constants `H0`, `kappa`, `g`, `x_L`, `x_R`, `fs`, and the truth for verification only: `x_fine`, `b_true_fine` (2160 cells, the generating grid), `x_64`, `b_true_64`, `x_128`, `b_true_128`. Inversions on N = 270 or 540 solver cells commit no inverse crime (discretization error 0.5 and 0.2 mm rms).
* The measured flume data of Angel et al. are not included; download `Data_Sensors.zip` and `experiment.pdf` from <https://doi.org/10.15480/882.9403> (CC BY-NC 4.0).

Minimal usage (flume, GSBL model in prior-normalized coordinates):

```python
import numpy as np, swe, flume, priors
d = np.load("data/flume_hill.npz")
N, n = 270, 64
x_cells, dx = flume.cells(N); x_nodes = flume.nodes(n)
dt, k, n_samples = flume.time_grid(N)
eta_left = np.interp(dt * np.arange(k * n_samples + 1), d["t_left"], d["eta_left_true"])
def F(z):                                   # increments of the nodal values -> sensor records (1000, 3)
    b_nodes = swe.increments_to_bottom(z, reference="left")
    return flume.forward(flume.to_cells(b_nodes, x_nodes, x_cells), eta_left, N)
gg = priors.GeneralizedGamma(*priors.GSBL_HYPERPRIOR)
# priors.log_posterior_gsbl_normalized(w, tau, d["eta_obs"], F, float(d["noise_std"]), gg)
# JAX: f = flume.forward_jax(N); f(b_cells, eta_left) is jitted and differentiable (jax.grad, jax.vmap).
```

Requirements: `numpy`, `scipy`, `matplotlib`; optional `jax` for `forward_jax`.

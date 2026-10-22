# Project B — starter code

| File | Content |
|---|---|
| `acrnn.py` | AC-CRNN forward model: atom-conservation basis `atom_conservation_basis(N)`, mass-action right-hand side, and the ODE solve (SciPy LSODA) in two parametrizations. **Tied orders (default model):** `pack_tied`/`unpack_tied`, `simulate_tied`/`simulate_all_tied`, `true_params_trimerization_tied(B)` (20 parameters with the atom-balance layer, 24 with `B = np.eye(5)`, i.e., without it). **Free orders:** `pack`/`unpack`, `simulate`/`simulate_all`, `true_params_trimerization(B)` (40 parameters; has the scaling symmetry explained in the module docstring). |
| `generate_data.py` | The data generator. It reproduces the three data sets in `data/` from fixed seeds (bit-for-bit with NumPy 2.5 and SciPy 1.18; other versions may differ in the last digits) and the overview figure `../figures/data_B.pdf`. Custom sets: `python generate_data.py --n_exp 10 --noise 0.1 --seed 3`. |
| `data/` | The three data sets (provided): `trimerization_20exp_5pct.npz` (standard), `trimerization_5exp_5pct.npz` (scarce), `trimerization_20exp_20pct.npz` (noisy). |

Data files contain: `t_eval` (measurement times), `c0` (initial concentrations, n_exp×5), `c_obs` (noisy measurements, n_exp×n_t×5), `c_clean`, `noise_std` (per species), `noise_level`, `species`, `molecular_matrix`, `B` (atom-conservation basis), `key_species`, and the truth (`NU_true`, `a_true`, `k_true`, `params_true`) for verification only.
The data were generated with the orders equal to the reactant stoichiometry, `a_true = max(-NU_true, 0)`, so they are consistent with the tied model.
`params_true` is in the free-order parametrization; for the tied model use `acrnn.true_params_trimerization_tied(B)`.

One likelihood evaluation (20 experiments) costs about 0.05 s with SciPy. For gradient-based inference (NUTS, variational inference) use a differentiable ODE solver, e.g., `jax` + `diffrax` with `numpyro`, or `torchdiffeq` with Pyro; the NumPy model here is the reference to validate against.

```python
import numpy as np, acrnn
d = np.load("data/trimerization_20exp_5pct.npz")
B = d["B"]
p = acrnn.true_params_trimerization_tied(B)                   # 20 parameters
c = acrnn.simulate_all_tied(p, d["c0"], d["t_eval"], B, 5, 4)  # equals d["c_clean"]
```

```bash
python generate_data.py          # reproduces data/*.npz and ../figures/data_B.pdf
```

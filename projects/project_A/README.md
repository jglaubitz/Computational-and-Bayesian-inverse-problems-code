# Project A — starter code

| File | Content |
|---|---|
| `generate_data.py` | Writes `data/deconvolution.npz` (the deconvolution example of the lecture notes, identical to `y_p` in `ch05_figures.py`) and `data/burgers.npz` (the Burgers problem), and `../figures/data_A.pdf`. Options `--m`, `--sigma`, `--seed` write further deconvolution data sets, and `--sigma_burgers` a Burgers data set with another noise level, under other file names. |
| `hyperpriors.py` | The four hyper-priors of the project (`DEFAULTS`: `IG`, `G`, `GGm`, `HC`), with log-density, cdf/sf/ppf/isf, sampler, the conditionals theta_i given z_i for Gibbs samplers (exact for `IG` and `G`, the two-level conditionals for `HC`, a grid sampler otherwise), and the maps of the hierarchical prior normalization. No variational code. |
| `model.py` | Data loaders, the increment maps `S` and `T` (x = T w, w = (c, z)), the log-likelihood and the log-joint density with all normalizing constants (also in log theta), and the log-likelihood in prior-normalized coordinates (c, w, tau) for elliptical slice sampling. |
| `burgers.py` | NumPy Godunov solver for the inviscid Burgers equation (fixed number of time steps) and the observation operator. |
| `burgers_jax.py` | The same solver in JAX (`lax.scan`), with a jitted forward map and Jacobian; `python burgers_jax.py` checks it against `burgers.py`. |

Data files:

* `deconvolution.npz`: `t` (cells, n = 128), `s` (sensors, m = 64), `A` (m x n), `y`, `x_true` (piecewise-constant signal), `y_smooth`, `x_smooth`, `sigma = 0.05`, `gamma = 0.03` (kernel width).
* `burgers.npz`: `x` (cells, n = 100), `obs_cells`, `x_obs`, `y` (m = 20), `u0_true`, `uT_true`, `sigma = 0.01`, `T = 0.25`, `cfl`, `umax`.

The model is that of the course code (`samplers.gibbs_hierarchical`, `approximations.cavi_hierarchical`): x = c 1 + S z with c ~ N(0, 1) and the hierarchical prior on the n - 1 increments z. Start your inference code from the course code (`approximations.py`, `samplers.py`); add the course folder `code/` to your `sys.path`.

Requirements: `numpy`, `scipy`, `matplotlib`; `jax` for `burgers_jax.py` and for gradient-based variational inference. If small dense linear algebra (n = 128) is unexpectedly slow, set `OPENBLAS_NUM_THREADS=1` before starting Python.

```bash
python generate_data.py                               # data/*.npz and ../figures/data_A.pdf
python generate_data.py --m 32 --sigma 0.1 --seed 2   # a less informative deconvolution data set
python generate_data.py --sigma_burgers 0.05          # a less informative Burgers data set
python burgers_jax.py                                 # check the JAX solver
```

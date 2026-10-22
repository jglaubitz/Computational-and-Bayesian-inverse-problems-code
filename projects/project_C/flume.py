"""
Project C -- the time-dependent flume scenario (the target of the project).

Setting of Angel et al. (2024) and Stietz et al. (2026): a wave flume with a wave maker at the left, a bathymetry
on an otherwise flat floor, and free-surface sensors at x = 1.5, 3.5, 5.5, 7.5 m sampled at 100 Hz. As in both
papers the record of the first sensor drives the left boundary of the one-dimensional shallow-water model with
linear bottom friction (kappa = 0.2), the computational domain [1.5, 15] m is longer than the flume and closed by a
wall u = 0 on the right, the water is at rest at t = 0, and the records of the other three sensors are the data.

    forward(b_cells, eta_left, N, ...)   bottom on the solver cells -> surface elevation at the three sensors
                                         at the 100 Hz sample times  (shape (n_times, 3))
    forward_jax(...)                     the same in JAX (jit, grad, vmap); install jax to use it

The bottom is represented by its values on n equidistant nodes on [X_L, X_R] (`nodes`), interpolated linearly to
the solver cells (`to_cells`), with b = 0 fixed at the first node (the floor at the wave-maker sensor is flat, which
is also needed to convert the measured surface at x = 1.5 m into the depth imposed at the boundary). The priors
act on the increments of the nodal values (see priors.py and swe.increments_to_bottom(..., reference="left")).

The synthetic configuration (still-water depth, wave amplitude and period, hill geometry) is chosen to resemble
the TUHH experiment; the actual values are documented in experiment.pdf of the data artefact (doi:10.15480/882.9403)
and should be taken from there when you work with the measured data.
"""
import numpy as np
import swe

X_L, X_R = 1.5, 15.0            # computational domain [m]: first sensor to (artificial) right wall
H0 = 0.40                       # still-water depth [m]            (synthetic configuration)
KAPPA = 0.2                     # linear friction coefficient      (Angel et al. 2024)
SENSORS = np.array([3.5, 5.5, 7.5])   # data sensors [m]; the sensor at X_L = 1.5 m drives the boundary
FS = 100.0                      # sampling frequency [Hz]
T_END = 10.0                    # record length [s]  (3 sensors x 1000 samples = 3000 data, as in Stietz et al.)
A_WAVE, T_WAVE = 0.02, 2.5      # synthetic wave maker: amplitude [m] and period [s] of the surface at X_L
CFL = 0.45


# ---------------------------------------------------------------- bathymetries and wave maker
def bottom_flat(x):
    return np.zeros_like(x, dtype=float)


def bottom_hill(x):
    """Smooth Gaussian hill, height 0.2 m, centred at x = 4 m (roughly the TUHH hill, cf. Stietz et al.)."""
    return 0.20 * np.exp(-((x - 4.0) ** 2) / 0.05)


def bottom_box(x):
    """Piecewise constant: a box of the same height and cross-section as the hill (0.2 m on (3.8, 4.2]) and a lower,
    wider step (0.06 m on (6.0, 6.8]) between the third and the fourth sensor."""
    b = np.zeros_like(x, dtype=float)
    b[(x > 3.8) & (x <= 4.2)] = 0.20
    b[(x > 6.0) & (x <= 6.8)] = 0.06
    return b


BATHYMETRIES = {"hill": bottom_hill, "box": bottom_box}


def eta_left_synthetic(t):
    """Surface elevation imposed at X_L: a sinusoidal wave train switched on smoothly over one period."""
    t = np.asarray(t, dtype=float)
    ramp = np.minimum(t / T_WAVE, 1.0)
    return H0 + A_WAVE * ramp * np.sin(2 * np.pi * t / T_WAVE)


# ---------------------------------------------------------------- grids
def cells(N):
    """Cell centres and width of the solver grid on [X_L, X_R]."""
    dx = (X_R - X_L) / N
    return X_L + (np.arange(N) + 0.5) * dx, dx


def nodes(n):
    """n equidistant nodes on [X_L, X_R] carrying the unknown bottom values (first node fixed to 0)."""
    return np.linspace(X_L, X_R, n)


def to_cells(b_nodes, x_nodes, x_cells, xp=np):
    """Linear interpolation of nodal bottom values to the solver cells (a linear map of the nodal values)."""
    return xp.interp(x_cells, x_nodes, b_nodes)


def sensor_matrix(N, sensors=SENSORS):
    """P (len(sensors), N) with P @ eta = linear interpolation of the cell values eta to the sensor positions."""
    x, _ = cells(N)
    P = np.zeros((len(sensors), N))
    for k, xs in enumerate(sensors):
        j = np.searchsorted(x, xs) - 1
        w = (xs - x[j]) / (x[j + 1] - x[j])
        P[k, j], P[k, j + 1] = 1.0 - w, w
    return P


def time_grid(N, fs=FS, T=T_END, c_max=None):
    """Fixed time step dt = 1/(fs * k) with the smallest integer k such that CFL <= `CFL` for wave speeds up to
    c_max (default: sqrt(g (H0 + 3 A_WAVE)) + 0.5 m/s). Returns dt, k (sub-steps per sample), n_samples."""
    _, dx = cells(N)
    if c_max is None:
        c_max = np.sqrt(swe.G * (H0 + 3 * A_WAVE)) + 0.5
    k = int(np.ceil(c_max / (CFL * dx * fs)))
    return 1.0 / (fs * k), k, int(round(T * fs))


# ---------------------------------------------------------------- forward model (NumPy)
def _ghosts_flume(xp):
    def ghosts(h, q, b, eta_left_stage):
        hl = eta_left_stage - b[0]                                   # depth imposed at the left boundary
        ql = hl * swe._velocity(xp, h[0], q[0])                      # velocity extrapolated from the first cell
        return (hl, ql, b[0]), (h[-1], -q[-1], b[-1])                # right: reflecting wall
    return ghosts


def forward(b_cells, eta_left, N, order=2, kappa=KAPPA, fs=FS, T=T_END, sensors=SENSORS, check_cfl=True,
            return_state=False):
    """Surface elevation at the sensors at times (1/fs, 2/fs, ..., T) for the bottom b_cells (N,).

    eta_left : callable t -> surface elevation at X_L, or an array of length n_steps + 1 with the values at the
               solver time levels (t_j = j dt; use `time_grid(N)` for dt). A measured record at 100 Hz has to be
               interpolated to these levels (linear interpolation is adequate).
    """
    x, dx = cells(N)
    dt, k, n_samples = time_grid(N, fs, T)
    n_steps = k * n_samples
    if callable(eta_left):
        eta_series = eta_left(dt * np.arange(n_steps + 1))
    else:
        eta_series = np.asarray(eta_left, dtype=float)
        assert len(eta_series) == n_steps + 1, f"eta_left needs {n_steps + 1} values at the solver time levels"
    P = sensor_matrix(N, sensors)
    b = np.asarray(b_cells, dtype=float)
    h = np.maximum(eta_series[0] - b, 0.0)
    q = np.zeros(N)
    ghosts = _ghosts_flume(np)
    rec = np.empty((n_samples, len(sensors)))
    smax_all = 0.0
    for j in range(n_steps):
        gh = lambda hh, qq, stage, j=j: ghosts(hh, qq, b, eta_series[j + stage])
        h, q, smax = swe.rk2_step(h, q, b, dx, dt, gh, kappa, order, np)
        smax_all = max(smax_all, float(smax))
        if (j + 1) % k == 0:
            rec[(j + 1) // k - 1] = P @ (h + b)
    if check_cfl and smax_all * dt / dx > 1.0:
        raise RuntimeError(f"CFL number {smax_all * dt / dx:.2f} > 1: decrease dt (pass a larger c_max to time_grid)")
    return (rec, h, q) if return_state else rec


# ---------------------------------------------------------------- forward model (JAX)
def forward_jax(N, order=2, kappa=KAPPA, fs=FS, T=T_END, sensors=SENSORS):
    """Returns a jitted function f(b_cells, eta_series) -> (n_samples, len(sensors)) with the same semantics as
    `forward` (eta_series: array of length n_steps + 1 at the solver time levels). Differentiable in b_cells
    (jax.grad, jax.jacfwd) and vectorizable (jax.vmap). Enables 64-bit floats in JAX; on a laptop CPU one
    evaluation takes about 60 ms for N = 270 and a reverse-mode gradient about ten times as much."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    x, dx = cells(N)
    dt, k, n_samples = time_grid(N, fs, T)
    n_steps = k * n_samples
    P = jnp.asarray(sensor_matrix(N, sensors))
    ghosts = _ghosts_flume(jnp)

    def step(carry, eta_pair):
        h, q, b = carry
        gh = lambda hh, qq, stage: ghosts(hh, qq, b, eta_pair[stage])
        h, q, _ = swe.rk2_step(h, q, b, dx, dt, gh, kappa, order, jnp)
        return (h, q, b), P @ (h + b)

    @jax.jit
    def f(b_cells, eta_series):
        b = jnp.asarray(b_cells)
        h0 = jnp.maximum(eta_series[0] - b, 0.0)
        pairs = jnp.stack([eta_series[:-1], eta_series[1:]], axis=1)
        _, eta_all = jax.lax.scan(step, (h0, jnp.zeros(N), b), pairs)
        return eta_all[k - 1::k]

    f.dt, f.n_steps, f.n_samples = dt, n_steps, n_samples
    return f

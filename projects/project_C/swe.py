"""
Project C -- Bayesian bathymetry inference from shallow-water surface data
Shared numerics in NumPy (and, through the `xp` argument, JAX): (1) algebraic steady states for the warm-up
scenario, (2) a well-balanced finite-volume solver (first or second order) used by the time-dependent flume
scenario in `flume.py`.  No inference code: that is your job.

One-dimensional shallow-water equations with bottom topography b(x) and linear bottom friction:
        h_t + (h u)_x = 0,
        (h u)_t + (h u^2 + g h^2 / 2)_x = -g h b_x - kappa h u,
h = water depth, u = velocity, q = h u = discharge, eta = h + b = free-surface elevation.

(1) Smooth steady states (kappa = 0) satisfy q = const and the Bernoulli relation
        q^2 / (2 h^2) + g (h + b) = E = const,
    i.e. for given (q, E) the depth in each cell is a root of  g h^3 + (g b - E) h^2 + q^2 / 2 = 0.
    `steady_depth` returns the subcritical root (h > h_c = (q^2/g)^(1/3)) by a vectorized Newton
    iteration; `surface_at_sensors` is the complete forward operator  b -> eta(x_obs) of the warm-up.

(2) `fv_rhs` is the semi-discrete finite-volume scheme with hydrostatic reconstruction (Audusse et al.,
    2004) and the HLL flux; order 2 uses MUSCL reconstruction of (h, eta, q) with the minmod limiter and the
    centred source term of Audusse et al., and `rk2_step` integrates it with SSP-RK2 (order 1: forward Euler).
    The scheme preserves lake-at-rest states exactly on both orders. `simulate` (adaptive time step, channel
    boundary conditions) converges to the moving-water steady state of (1) and is kept for cross-checks;
    the flume forward model with fixed time step lives in `flume.py`.

Parametrization of the unknown: the bottom is represented by its values on n equidistant nodes, one of
which is fixed to 0 as reference level, and the priors are placed on the increments z_i = b_{i+1} - b_i
(see `bottom_to_increments`, `increments_to_bottom`).  In the steady channel the reference is the outflow
node b_n = 0, in the flume it is the node at the wave-maker sensor, b_1 = 0.
"""
import numpy as np

G = 9.81
L = 25.0           # channel length [m]              (steady warm-up scenario)
Q = 4.42           # discharge [m^2/s]
H_OUT = 2.0        # outflow depth [m]
H_MIN = 1e-9       # depth below which a cell is treated as dry


# ---------------------------------------------------------------- grid and parametrization
def make_grid(n, L=L):
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    return x, dx


def bottom_to_increments(b):
    """z_i = b_{i+1} - b_i  (length n-1)."""
    return np.diff(b)


def increments_to_bottom(z, reference="right", xp=np):
    """b from increments (length n).
    reference="right":  b_n = 0,  b_i = -sum_{j >= i} z_j   (steady channel: outflow node as reference)
    reference="left" :  b_1 = 0,  b_i =  sum_{j <  i} z_j   (flume: node at the wave-maker sensor as reference)
    """
    z = xp.asarray(z)
    zero = xp.zeros((1,), dtype=z.dtype)
    if reference == "right":
        return -xp.concatenate([xp.cumsum(z[::-1])[::-1], zero])
    if reference == "left":
        return xp.concatenate([zero, xp.cumsum(z)])
    raise ValueError("reference must be 'right' or 'left'")


# ---------------------------------------------------------------- algebraic steady state (warm-up)
def steady_depth(b, q=Q, h_out=H_OUT, b_out=0.0, n_iter=40):
    """Subcritical steady depth h(x) for bottom b (array), discharge q and outflow depth h_out at b_out.

    Vectorized Newton iteration on f(h) = g h^3 + (g b - E) h^2 + q^2/2, started to the right of the
    largest root (monotone convergence). Returns NaN where no subcritical steady state exists.
    """
    b = np.asarray(b, dtype=float)
    E = q**2 / (2 * h_out**2) + G * (h_out + b_out)
    h = np.full_like(b, E / G - min(b.min(), 0.0) + 1.0)
    for _ in range(n_iter):
        f = G * h**3 + (G * b - E) * h**2 + 0.5 * q**2
        fp = 3 * G * h**2 + 2 * (G * b - E) * h
        h = h - f / fp
    h_c = (q**2 / G) ** (1.0 / 3.0)
    ok = (h > h_c) & np.isfinite(h) & (np.abs(G * h**3 + (G * b - E) * h**2 + 0.5 * q**2) < 1e-8)
    return np.where(ok, h, np.nan)


def surface_at_sensors(b, x, x_obs, q=Q, h_out=H_OUT):
    """Forward operator of the steady scenario: free-surface elevation eta = h + b at the sensor positions
    (linear interpolation from the cell centres x). Returns NaN if the flow is not subcritical everywhere."""
    h = steady_depth(b, q, h_out, b_out=b[-1])
    if np.isnan(h).any():
        return np.full(len(x_obs), np.nan)
    return np.interp(x_obs, x, h + b)


# ---------------------------------------------------------------- finite-volume scheme (xp = numpy or jax.numpy)
def _minmod(xp, a, b):
    return xp.where(a * b > 0.0, xp.sign(a) * xp.minimum(xp.abs(a), xp.abs(b)), 0.0)


def _velocity(xp, h, q):
    return xp.where(h > H_MIN, q / xp.maximum(h, H_MIN), 0.0)


def hll_flux(hL, qL, hR, qR, xp=np):
    """HLL numerical flux for the homogeneous shallow-water system, vectorized over interfaces.
    Returns the flux (2, M) and the largest wave speed per interface."""
    uL, uR = _velocity(xp, hL, qL), _velocity(xp, hR, qR)
    cL, cR = xp.sqrt(G * hL), xp.sqrt(G * hR)
    sL = xp.minimum(uL - cL, uR - cR)
    sR = xp.maximum(uL + cL, uR + cR)
    FL = xp.stack([qL, qL * uL + 0.5 * G * hL**2])
    FR = xp.stack([qR, qR * uR + 0.5 * G * hR**2])
    UL = xp.stack([hL, qL])
    UR = xp.stack([hR, qR])
    F = xp.where(sL >= 0, FL, xp.where(sR <= 0, FR,
                 (sR * FL - sL * FR + sL * sR * (UR - UL)) / xp.maximum(sR - sL, 1e-12)))
    return F, xp.maximum(xp.abs(sL), xp.abs(sR))


def fv_rhs(h, q, b, dx, bc_left, bc_right, kappa=0.0, order=2, xp=np):
    """Semi-discrete well-balanced scheme: returns (dh/dt, dq/dt, max wave speed).

    h, q, b   : cell values (N,)
    bc_left   : ghost-cell state (h, q, b) to the left of cell 1;  bc_right: the same to the right of cell N
    order     : 1 (piecewise constant) or 2 (MUSCL/minmod reconstruction of h, eta = h + b and q, with the
                hydrostatic reconstruction at the interfaces and the centred source term of Audusse et al. 2004)
    kappa     : linear friction coefficient, source -kappa q in the momentum equation
    """
    def ext(v, vl, vr):
        return xp.concatenate([xp.reshape(xp.asarray(vl, dtype=v.dtype), (1,)), v,
                               xp.reshape(xp.asarray(vr, dtype=v.dtype), (1,))])

    he = ext(h, bc_left[0], bc_right[0])
    qe = ext(q, bc_left[1], bc_right[1])
    be = ext(b, bc_left[2], bc_right[2])
    etae = he + be

    def slopes(v):
        if order == 1:
            return xp.zeros_like(v)
        s = _minmod(xp, v[1:-1] - v[:-2], v[2:] - v[1:-1])
        z = xp.zeros((1,), dtype=v.dtype)                      # ghost cells: no reconstruction
        return xp.concatenate([z, s, z])

    sh, se, sq = slopes(he), slopes(etae), slopes(qe)
    # states at the interfaces 1/2, ..., N+1/2:  "m" from the cell on the left, "p" from the cell on the right
    h_m, eta_m, q_m = he[:-1] + 0.5 * sh[:-1], etae[:-1] + 0.5 * se[:-1], qe[:-1] + 0.5 * sq[:-1]
    h_p, eta_p, q_p = he[1:] - 0.5 * sh[1:], etae[1:] - 0.5 * se[1:], qe[1:] - 0.5 * sq[1:]
    b_m, b_p = eta_m - h_m, eta_p - h_p
    b_star = xp.maximum(b_m, b_p)
    h_ms = xp.maximum(h_m + b_m - b_star, 0.0)
    h_ps = xp.maximum(h_p + b_p - b_star, 0.0)
    F, smax = hll_flux(h_ms, h_ms * _velocity(xp, h_m, q_m), h_ps, h_ps * _velocity(xp, h_p, q_p), xp=xp)

    corr_r = 0.5 * G * (h_m[1:]**2 - h_ms[1:]**2)              # cell i, interface i+1/2
    corr_l = 0.5 * G * (h_p[:-1]**2 - h_ps[:-1]**2)            # cell i, interface i-1/2
    fc = -G * 0.5 * (h_p[:-1] + h_m[1:]) * (b_m[1:] - b_p[:-1])   # centred source (zero for order 1)
    dh = -(F[0, 1:] - F[0, :-1]) / dx
    dq = -((F[1, 1:] + corr_r) - (F[1, :-1] + corr_l) - fc) / dx - kappa * q
    return dh, dq, xp.max(smax)


def rk2_step(h, q, b, dx, dt, ghosts, kappa=0.0, order=2, xp=np):
    """One time step. ghosts(h, q, stage) -> (bc_left, bc_right) supplies the ghost states for stage 0 (time t)
    and stage 1 (time t + dt). Order 2: SSP-RK2 (Heun); order 1: forward Euler. Returns h, q, max wave speed."""
    bl, br = ghosts(h, q, 0)
    dh, dq, smax = fv_rhs(h, q, b, dx, bl, br, kappa, order, xp)
    h1, q1 = xp.maximum(h + dt * dh, 0.0), q + dt * dq
    if order == 1:
        return h1, q1, smax
    bl, br = ghosts(h1, q1, 1)
    dh, dq, smax1 = fv_rhs(h1, q1, b, dx, bl, br, kappa, order, xp)
    h2 = xp.maximum(0.5 * h + 0.5 * (h1 + dt * dh), 0.0)
    q2 = 0.5 * q + 0.5 * (q1 + dt * dq)
    return h2, q2, xp.maximum(smax, smax1)


# ---------------------------------------------------------------- channel solver (adaptive step, cross-checks)
def simulate(b, q_in, h_out, T, dx, h0=None, q0=None, cfl=0.45, order=1, kappa=0.0, record=None):
    """Channel flow on cells with bottom b (N,) up to time T: discharge q_in(t) imposed at the left boundary,
    depth h_out imposed at the right (subcritical boundary conditions), adaptive time step.

    q_in   : float or callable t -> discharge at the left boundary
    h0, q0 : initial depth and discharge (default: lake at rest with surface at b[-1] + h_out, q = 0)
    record : None, or (times, gauge_cells) -> also returns eta[len(times), len(gauge_cells)]
    """
    N = len(b)
    h = np.maximum(b[-1] + h_out - b, 1e-3) if h0 is None else h0.copy()
    q = np.zeros(N) if q0 is None else q0.copy()
    qin = q_in if callable(q_in) else (lambda t, v=q_in: v)

    rec = None
    if record is not None:
        rec_times, rec_cells = np.asarray(record[0], float), np.asarray(record[1], int)
        rec = np.full((len(rec_times), len(rec_cells)), np.nan)
        k_rec = 0

    t = 0.0
    while t < T - 1e-12:
        c = np.sqrt(G * np.maximum(h, H_MIN))
        u = _velocity(np, h, q)
        dt = cfl * dx / np.max(np.abs(u) + c)
        dt = min(dt, T - t)
        if rec is not None and k_rec < len(rec_times):
            dt = min(dt, max(rec_times[k_rec] - t, 1e-12))
        t_stage = (t, t + dt)

        def ghosts(hh, qq, stage):
            return (hh[0], qin(t_stage[stage]), b[0]), (h_out, qq[-1], b[-1])

        h, q, _ = rk2_step(h, q, b, dx, dt, ghosts, kappa, order, np)
        t += dt
        if rec is not None and k_rec < len(rec_times) and t >= rec_times[k_rec] - 1e-9:
            rec[k_rec] = (h + b)[rec_cells]
            k_rec += 1

    return (h, q) if rec is None else (h, q, rec)


def interp_bottom(b_coarse, x_coarse, x):
    """Piecewise-linear interpolation of a bottom given on one grid onto another grid."""
    return np.interp(x, x_coarse, b_coarse)


# ---------------------------------------------------------------- the two test topographies of the steady warm-up
def bottom_smooth(x):
    """Topography (1): smooth, two Gaussian bumps."""
    return 0.20 * np.exp(-((x - 9.0) / 1.6) ** 2) + 0.12 * np.exp(-((x - 17.5) / 2.4) ** 2)


def bottom_piecewise(x):
    """Topography (2): piecewise constant with four jumps."""
    b = np.zeros_like(x, dtype=float)
    b[(x > 6.0) & (x <= 11.0)] = 0.15
    b[(x > 11.0) & (x <= 14.0)] = 0.06
    b[(x > 17.0) & (x <= 20.5)] = 0.20
    return b


TOPOGRAPHIES = {"smooth": bottom_smooth, "piecewise": bottom_piecewise}

"""
Project A -- nonlinear forward operator: the inviscid Burgers equation.

    u_t + (u^2 / 2)_x = 0   on the periodic domain [0, 1),   u(x, 0) = u0(x),

solved with a first-order Godunov finite-volume scheme (exact Riemann solver for the convex
flux) with a fixed time step, from t = 0 to t = T.  The observation operator samples u(., T)
at m equispaced cells.  The inverse problem is to recover u0 from noisy observations of u(., T).

Why Burgers: the solution develops shocks (compressive parts of u0) and rarefactions (expansive
parts); information about u0 that is swallowed by a shock before time T is lost, so the forward
map is nonlinear, non-injective, and non-smooth (the Godunov flux involves max/min).  This is a
genuinely different geometry from the linear deconvolution problem.

The number of time steps is fixed from a global bound UMAX on |u| (|u| does not increase in
time for entropy solutions), so that the forward map has a fixed computational graph.  This
matters if you port the solver to JAX (a `lax.scan` over the steps) to get gradients.
"""
import numpy as np

N_CELLS = 100      # grid cells on [0, 1)
T_FINAL = 0.25     # observation time
UMAX = 2.5         # global bound on |u| used to fix the time step (data satisfy |u| <= 1.2)
CFL = 0.5


def grid(n=N_CELLS):
    dx = 1.0 / n
    return (np.arange(n) + 0.5) * dx, dx


def godunov_flux(uL, uR):
    """Exact Godunov flux for f(u) = u^2/2 (vectorized)."""
    fL, fR = 0.5 * uL**2, 0.5 * uR**2
    # shock or rarefaction not containing u=0: take the appropriate one-sided flux
    flux = np.where(uL <= uR,
                    np.minimum(fL, fR) * (~((uL < 0) & (uR > 0))),   # rarefaction: min flux, 0 if sonic
                    np.maximum(fL, fR))                                 # shock: max flux
    return flux


def solve(u0, T=T_FINAL, cfl=CFL, umax=UMAX):
    """Godunov scheme on the periodic grid; returns u(., T) on the same cells as u0."""
    u = np.asarray(u0, dtype=float).copy()
    n = u.size
    dx = 1.0 / n
    dt = cfl * dx / umax
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps
    for _ in range(n_steps):
        uL = u                       # left state of interface i+1/2 is cell i
        uR = np.roll(u, -1)          # right state is cell i+1 (periodic)
        F = godunov_flux(uL, uR)     # flux through interface i+1/2
        u = u - dt / dx * (F - np.roll(F, 1))
    return u


def observation_cells(n=N_CELLS, m=20):
    """m equispaced observation cells."""
    return np.linspace(0, n, m, endpoint=False).astype(int) + n // (2 * m)


def forward(u0, obs_cells, T=T_FINAL):
    """Nonlinear forward operator G(u0) = u(., T) at the observation cells."""
    return solve(u0, T)[obs_cells]


def true_initial_condition(x):
    """Piecewise-constant initial data: background 0.2, two blocks of height 1.0 and 0.6."""
    u0 = np.full_like(x, 0.2)
    u0[(x > 0.15) & (x <= 0.45)] = 1.0
    u0[(x > 0.60) & (x <= 0.80)] = 0.6
    return u0

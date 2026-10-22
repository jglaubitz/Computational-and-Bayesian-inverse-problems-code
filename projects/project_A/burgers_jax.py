"""
Project A -- JAX version of burgers.py (same Godunov scheme, same fixed number of time steps), for Jacobians
(linearized variational Bayes, Laplace approximations) and fast batched likelihoods (elliptical slice sampling).

    G, J = make_forward(obs_cells)    # G(u0) -> u(., T)[obs_cells],  J(u0) -> dG/du0 (m x n), both jitted
    Gb = jax.jit(jax.vmap(G))          # batched forward solves, e.g., for several ESS chains at once

The Godunov flux is piecewise smooth (min/max), so the Jacobian exists almost everywhere; at a kink JAX returns
one of the one-sided derivatives. Validate against burgers.py with check() before you rely on it.
"""
import numpy as np
import jax
import jax.numpy as jnp

import burgers

jax.config.update("jax_enable_x64", True)


def make_solver(n=burgers.N_CELLS, T=burgers.T_FINAL, cfl=burgers.CFL, umax=burgers.UMAX):
    dx = 1.0 / n
    n_steps = int(np.ceil(T / (cfl * dx / umax)))
    dt = T / n_steps

    def flux(uL, uR):
        fL, fR = 0.5 * uL**2, 0.5 * uR**2
        sonic = (uL < 0) & (uR > 0)
        return jnp.where(uL <= uR, jnp.where(sonic, 0.0, jnp.minimum(fL, fR)), jnp.maximum(fL, fR))

    def step(u, _):
        F = flux(u, jnp.roll(u, -1))
        return u - dt / dx * (F - jnp.roll(F, 1)), None

    def solve(u0):
        u, _ = jax.lax.scan(step, u0, None, length=n_steps)
        return u
    return solve


def make_forward(obs_cells, n=burgers.N_CELLS):
    solve = make_solver(n)
    obs = jnp.asarray(obs_cells)
    G = jax.jit(lambda u0: solve(u0)[obs])
    J = jax.jit(jax.jacfwd(lambda u0: solve(u0)[obs]))
    return G, J


def check(seed=0):
    """Compare with the NumPy solver at the true initial condition and at a random perturbation of it."""
    x, _ = burgers.grid()
    u0 = burgers.true_initial_condition(x)
    solve = jax.jit(make_solver(len(x)))
    rng = np.random.default_rng(seed)
    for u in (u0, u0 + 0.1 * rng.standard_normal(len(x))):
        print("max |JAX - NumPy| =", float(np.max(np.abs(np.asarray(solve(jnp.asarray(u))) - burgers.solve(u)))))


if __name__ == "__main__":
    check()

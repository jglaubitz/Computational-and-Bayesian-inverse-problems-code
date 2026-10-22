"""
Running example 2 in JAX: the steady shallow-water forward map b -> eta(x_obs), differentiable.

    forward = make_forward(x_cells, x_obs)      jitted function b_cells -> surface elevation at the sensors
                                                (all NaN if some cell has no subcritical steady state)
    jac = jax.jacfwd(forward)                   its Jacobian (m x n), by forward-mode automatic differentiation
    jax.grad(lambda b: 0.5 * jnp.sum((forward(b) - y) ** 2))(b)     gradient of a misfit (reverse mode)

Same model and constants as running_examples.py: Bernoulli's relation solved per cell by a fixed number of
Newton steps (a differentiable program), then linear interpolation to the sensors.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from running_examples import G, Q, H_OUT


def steady_depth(b, q=Q, h_out=H_OUT, b_out=0.0, n_iter=40):
    """Subcritical steady depth h for bottom b: largest root of g h^3 + (g b - E) h^2 + q^2/2 = 0, 40 Newton steps."""
    E = q**2 / (2 * h_out**2) + G * (h_out + b_out)
    h0 = E / G - jnp.minimum(jnp.min(b), 0.0) + 1.0
    h = jnp.full_like(b, h0)

    def newton(h, _):
        f = G * h**3 + (G * b - E) * h**2 + 0.5 * q**2
        fp = 3 * G * h**2 + 2 * (G * b - E) * h
        return h - f / fp, None

    h, _ = jax.lax.scan(newton, h, None, length=n_iter)
    h_c = (q**2 / G) ** (1.0 / 3.0)
    residual = G * h**3 + (G * b - E) * h**2 + 0.5 * q**2
    ok = (h > h_c) & jnp.isfinite(h) & (jnp.abs(residual) < 1e-8)
    return jnp.where(ok, h, jnp.nan)                         # NaN where no subcritical steady state exists


def make_forward(x_cells, x_obs, q=Q, h_out=H_OUT, fixed_outflow=True):
    """Returns the jitted forward map F: b -> eta at the sensors (m,).

    fixed_outflow=True : the bottom at the outflow is the known datum b_n = 0; the unknown is b[:-1] in R^{n-1}.
    fixed_outflow=False: all n bottom values are unknown and the outflow value sets the energy level E
                         (this leaves a nearly invisible direction, see Chapter 2).
    """
    x_cells = jnp.asarray(x_cells)
    x_obs = jnp.asarray(x_obs)

    @jax.jit
    def forward(b):
        if fixed_outflow:
            b = jnp.concatenate([b, jnp.zeros(1)])
        h = steady_depth(b, q, h_out, b_out=b[-1])
        eta = jnp.interp(x_obs, x_cells, h + b)
        # as surface_at_sensors in running_examples.py: all data are NaN if any cell has no subcritical steady state
        # (for feasible bottoms the where selects eta, so values and derivatives are unchanged)
        return jnp.where(jnp.all(jnp.isfinite(h)), eta, jnp.nan)

    return forward

"""
Running examples of the lecture notes "Computational and Bayesian Inverse Problems" (LiU, 2026).

Example 1 -- one-dimensional deconvolution (linear):
    y(s) = int_0^1 k(s - t) x(t) dt + noise,   k(t) = exp(-t^2 / (2 gamma^2)) / (gamma sqrt(2 pi)),
    discretized with the midpoint rule on n cells:  y = A x + e,  A in R^{m x n},  A_ij = (1/n) k(s_i - t_j).
    Default configuration of the notes: n = 128 unknowns, m = 64 observation points, gamma = 0.03, white noise with
    standard deviation 0.05 (five percent of the signal maximum); kappa(A) = 3.5e7.
    Two test signals: `signal_smooth` (two bumps) and `signal_piecewise` (a box and a step).

Example 2 -- bathymetry from steady shallow-water flow (nonlinear):
    steady subcritical flow in a channel of length L with discharge q and outflow depth h_out satisfies the
    Bernoulli relation  q^2 / (2 h^2) + g (h + b) = E,  so the depth in every cell is the subcritical root
    of a cubic; the data are the free-surface elevation eta = h + b at m sensors.  `surface_at_sensors` is
    the forward operator b -> eta(x_obs), `bottom_smooth` and `bottom_piecewise` are the two test bottoms
    (the same as in default project C).

Everything is NumPy; the exercises of Chapters 2 and 4 use a JAX version of Example 2 (code/swe_jax.py).
"""
import numpy as np

# ====================================================================== Example 1: deconvolution
GAMMA = 0.03          # kernel width (standard deviation of the Gaussian kernel)
N_CELLS, M_OBS = 128, 64   # unknowns and observation points
NOISE_STD = 0.05           # noise level of the default data sets


def kernel(t, gamma=GAMMA):
    return np.exp(-t**2 / (2 * gamma**2)) / (gamma * np.sqrt(2 * np.pi))


def deconvolution_operator(n=N_CELLS, m=M_OBS, gamma=GAMMA):
    """Midpoint discretization of the convolution on [0, 1]. Returns A (m x n), cell centres t (n,), sensors s (m,)."""
    t = (np.arange(n) + 0.5) / n
    s = (np.arange(m) + 0.5) / m
    A = kernel(s[:, None] - t[None, :], gamma) / n
    return A, t, s


def signal_smooth(t):
    return np.exp(-((t - 0.3) / 0.08) ** 2) + 0.6 * np.exp(-((t - 0.7) / 0.12) ** 2)


def signal_piecewise(t):
    x = np.zeros_like(t)
    x[(t > 0.15) & (t <= 0.4)] = 1.0
    x[(t > 0.55) & (t <= 0.65)] = 0.5
    x[(t > 0.65) & (t <= 0.85)] = 0.8
    return x


SIGNALS = {"smooth": signal_smooth, "piecewise": signal_piecewise}


def noisy_data(A, x, noise_std, rng):
    """y = A x + e with e ~ N(0, noise_std^2 I)."""
    return A @ x + noise_std * rng.standard_normal(A.shape[0])


# ====================================================================== Example 2: steady shallow water
G = 9.81
L = 25.0          # channel length [m]
Q = 4.42          # discharge [m^2/s]
H_OUT = 2.0       # outflow depth [m]


def make_grid(n, L=L):
    dx = L / n
    return (np.arange(n) + 0.5) * dx, dx


def steady_depth(b, q=Q, h_out=H_OUT, b_out=0.0, n_iter=40):
    """Subcritical steady depth for bottom b: largest root of g h^3 + (g b - E) h^2 + q^2/2 = 0 by Newton's method.
    Returns NaN where no subcritical steady state exists."""
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
    """Forward operator F: bottom values b on the cell centres x -> free-surface elevation at the sensors x_obs."""
    h = steady_depth(b, q, h_out, b_out=b[-1])
    if np.isnan(h).any():
        return np.full(len(x_obs), np.nan)
    return np.interp(x_obs, x, h + b)


def froude(b, q=Q, h_out=H_OUT):
    h = steady_depth(b, q, h_out, b_out=b[-1])
    return q / h / np.sqrt(G * h)


def bottom_smooth(x):
    return 0.20 * np.exp(-((x - 9.0) / 1.6) ** 2) + 0.12 * np.exp(-((x - 17.5) / 2.4) ** 2)


def bottom_piecewise(x):
    b = np.zeros_like(x, dtype=float)
    b[(x > 6.0) & (x <= 11.0)] = 0.15
    b[(x > 11.0) & (x <= 14.0)] = 0.06
    b[(x > 17.0) & (x <= 20.5)] = 0.20
    return b


BOTTOMS = {"smooth": bottom_smooth, "piecewise": bottom_piecewise}


# ====================================================================== tools
def jacobian_fd(F, x, eps=1e-6):
    """Jacobian of F at x by central finite differences (columns = partial derivatives)."""
    x = np.asarray(x, dtype=float)
    f0 = F(x)
    J = np.empty((len(f0), len(x)))
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = eps
        J[:, j] = (F(x + e) - F(x - e)) / (2 * eps)
    return J

"""
Solvers used in Chapter 2 (deterministic regularization) and later.

    difference_matrix(n)                  first differences L in R^{(n-1) x n}
    tikhonov(A, y, lam, L=None)           minimizer of 1/2 ||A x - y||^2 + lam/2 ||L x||^2 (stacked least squares)
    cgls(A, y, n_iter)                    conjugate gradients on the normal equations; returns all iterates
    l1_regularization(A, y, lam, ...)     minimizer of 1/2 ||A x - y||^2 + lam ||L x||_1 by FISTA (L = I or differences)
    gauss_newton(F, jac, y, x0, lam, L)   Gauss-Newton with backtracking line search for nonlinear F
    gaussian_posterior(A, y, noise_var, prior_prec, prior_mean)
                                          mean and covariance of the linear-Gaussian posterior (Chapter 3)
    ias(A, y, noise_std, beta, vartheta, transform)
                                          iterative alternating scheme: joint MAP for the hierarchical prior (Chapter 3)

All routines are NumPy; `gauss_newton` takes any callables F and jac (e.g. from JAX).
"""
import numpy as np


def difference_matrix(n):
    """L in R^{(n-1) x n} with (L x)_i = x_{i+1} - x_i."""
    return np.diff(np.eye(n), axis=0)


# ---------------------------------------------------------------- Tikhonov
def tikhonov(A, y, lam, L=None):
    """Minimizer of 1/2 ||A x - y||_2^2 + lam/2 ||L x||_2^2 (L = identity if None), computed from the stacked
    least-squares problem [A; sqrt(lam) L] x = [y; 0] with a QR-based solver. Solving the normal equations
    (A^T A + lam L^T L) x = A^T y instead is equally fine unless lam is tiny (the stacked matrix M has
    condition number sqrt((s_1^2 + lam)/(s_n^2 + lam)) for L = I, the normal-equation matrix M^T M its square)."""
    n = A.shape[1]
    L = np.eye(n) if L is None else L
    A_stack = np.vstack([A, np.sqrt(lam) * L])
    y_stack = np.concatenate([y, np.zeros(L.shape[0])])
    return np.linalg.lstsq(A_stack, y_stack, rcond=None)[0]


def cgls(A, y, n_iter, x0=None):
    """CGLS: conjugate gradients applied to the normal equations A^T A x = A^T y without forming A^T A.
    Returns the list of iterates x_0, ..., x_{n_iter} (early stopping = regularization)."""
    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    r = y - A @ x
    s = A.T @ r
    p = s.copy()
    gamma = s @ s
    iterates = [x.copy()]
    for _ in range(n_iter):
        q = A @ p
        alpha = gamma / (q @ q)
        x = x + alpha * p
        r = r - alpha * q
        s = A.T @ r
        gamma_new = s @ s
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
        iterates.append(x.copy())
    return iterates


# ---------------------------------------------------------------- l1 / total variation by FISTA
def _soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def l1_regularization(A, y, lam, transform="identity", n_iter=2000, tol=1e-10):
    """Minimizer of 1/2 ||A x - y||^2 + lam ||L x||_1 by FISTA (Beck & Teboulle 2009).

    transform="identity":    L = I, sparsity of x itself.
    transform="differences": L = first differences (total variation); solved in the variables
                             x = c 1 + S z with z = L x the increments (penalized) and the offset c (not penalized),
                             where S is the cumulative-sum matrix, so that the proximal step is a soft threshold.
    Returns x (and the objective history as second output).
    """
    m, n = A.shape
    if transform == "identity":
        B, pen = A, np.ones(n, dtype=bool)
    elif transform == "differences":
        S = np.tril(np.ones((n, n)), -1)[:, :-1]            # x = c 1 + S z,  z in R^{n-1}
        B = np.hstack([A @ np.ones((n, 1)), A @ S])          # unknowns (c, z): A x = c A 1 + A S z
        pen = np.concatenate([[False], np.ones(n - 1, dtype=bool)])
    else:
        raise ValueError("transform must be 'identity' or 'differences'")
    step = 1.0 / np.linalg.norm(B, 2) ** 2                  # 1 / Lipschitz constant of the gradient
    w = np.zeros(B.shape[1]); v = w.copy(); t = 1.0
    hist = []
    for k in range(n_iter):
        grad = B.T @ (B @ v - y)
        w_new = v - step * grad
        w_new[pen] = _soft(w_new[pen], step * lam)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        v = w_new + (t - 1) / t_new * (w_new - w)
        hist.append(0.5 * np.sum((B @ w_new - y) ** 2) + lam * np.sum(np.abs(w_new[pen])))
        if np.linalg.norm(w_new - w) < tol * max(1.0, np.linalg.norm(w_new)):
            w = w_new
            break
        w, t = w_new, t_new
    if transform == "identity":
        x = w
    else:
        x = w[0] + S @ w[1:]
    return x, np.array(hist)


# ---------------------------------------------------------------- Gauss-Newton for nonlinear least squares
def gauss_newton(F, jac, y, x0, lam=0.0, L=None, n_iter=30, tol=1e-8, verbose=False):
    """Minimize J(x) = 1/2 ||F(x) - y||^2 + lam/2 ||L x||^2 by Gauss-Newton with backtracking (Armijo) line search.

    F, jac : callables x -> F(x) in R^m and x -> Jacobian in R^{m x n} (finite differences, adjoints, or JAX)
    Returns x and the history of objective values (one per iteration, including the start).
    """
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    L = np.eye(n) if L is None else L
    sqrt_lam_L = np.sqrt(lam) * L

    def objective(z):
        r = F(z) - y
        val = 0.5 * r @ r + 0.5 * lam * np.sum((L @ z) ** 2)
        return val if np.isfinite(val) else np.inf          # an infeasible candidate (NaN forward map) is rejected

    hist = [objective(x)]
    for k in range(n_iter):
        r = F(x) - y
        J = jac(x)
        # Gauss-Newton step: minimize 1/2 ||J d + r||^2 + lam/2 ||L (x + d)||^2 over d (a Tikhonov problem)
        A_stack = np.vstack([J, sqrt_lam_L])
        b_stack = np.concatenate([-r, -sqrt_lam_L @ x])
        d = np.linalg.lstsq(A_stack, b_stack, rcond=None)[0]
        grad = J.T @ r + lam * L.T @ (L @ x)
        slope = grad @ d
        alpha = 1.0
        while not objective(x + alpha * d) <= hist[-1] + 1e-4 * alpha * slope and alpha > 1e-8:
            alpha *= 0.5
        x = x + alpha * d
        hist.append(objective(x))
        if verbose:
            print(f"iter {k + 1}: J = {hist[-1]:.3e}, step {alpha:.2e}")
        if abs(hist[-2] - hist[-1]) < tol * max(1.0, abs(hist[-1])):
            break
    return x, np.array(hist)


# ---------------------------------------------------------------- linear-Gaussian posterior (Chapter 3)
def gaussian_posterior(A, y, noise_var, prior_prec, prior_mean=None):
    """Posterior of x for y = A x + e, e ~ N(0, noise_var I), x ~ N(prior_mean, prior_prec^{-1}):
        cov  = (A^T A / noise_var + prior_prec)^{-1},
        mean = cov (A^T y / noise_var + prior_prec prior_mean).
    noise_var may be a scalar or a vector of per-observation variances. Returns (mean, cov), both dense."""
    n = A.shape[1]
    w = 1.0 / np.broadcast_to(np.asarray(noise_var, dtype=float), (A.shape[0],))
    H = A.T @ (w[:, None] * A) + prior_prec
    rhs = A.T @ (w * y)
    if prior_mean is not None:
        rhs = rhs + prior_prec @ prior_mean
    cov = np.linalg.inv(H)
    mean = np.linalg.solve(H, rhs)
    return mean, cov


# ---------------------------------------------------------------- iterative alternating scheme (Chapter 3)
def ias(A, y, noise_std, beta, vartheta, transform="identity", n_iter=100, gamma0=1.0, theta0=None, tol=1e-10,
        return_iterates=False):
    """Joint MAP estimate of (x, theta) for y = A x + e, e ~ N(0, noise_std^2 I), and the hierarchical prior
        z_i | theta_i ~ N(0, theta_i),   theta_i ~ InvGamma(beta, vartheta),
    where z = x (transform="identity") or z = L x, the increments (transform="differences"; the level x_1 then
    has the prior N(0, gamma0^2)). Block-coordinate descent on the negative log posterior
        ||A x - y||^2 / (2 sigma^2) + sum_i [ z_i^2 / (2 theta_i) + vartheta / theta_i + (beta + 3/2) log theta_i ]:
    the x-step is a weighted Tikhonov problem (stacked least squares), the theta-step is explicit,
        theta_i = (z_i^2 / 2 + vartheta) / (beta + 3/2).
    theta0 is the initial value of all local variances (default: the mode vartheta / (beta + 1) of the hyper-prior).
    Returns x, theta, and the history of the objective (one value per iteration); with return_iterates=True the
    lists of all iterates x_k and theta_k are returned as well."""
    m, n = A.shape
    if transform == "identity":
        B = A
        pen = np.ones(n, dtype=bool)
        to_x = lambda w: w
    elif transform == "differences":
        S = np.tril(np.ones((n, n)), -1)[:, :-1]               # x = c 1 + S z
        B = np.hstack([A @ np.ones((n, 1)), A @ S])             # unknowns w = (c, z): A x = c A 1 + A S z
        pen = np.concatenate([[False], np.ones(n - 1, dtype=bool)])
        to_x = lambda w: w[0] + S @ w[1:]
    else:
        raise ValueError("transform must be 'identity' or 'differences'")
    k = B.shape[1]
    theta = np.full(pen.sum(), vartheta / (beta + 1.0) if theta0 is None else theta0, dtype=float)
    w = np.zeros(k)
    xs, thetas = [], []

    def objective(w, theta):
        z = w[pen]
        val = 0.5 * np.sum((B @ w - y) ** 2) / noise_std**2 + 0.5 * np.sum(z**2 / theta) \
            + np.sum(vartheta / theta + (beta + 1.5) * np.log(theta))
        if transform == "differences":
            val += 0.5 * w[0] ** 2 / gamma0**2
        return val

    hist = []
    for it in range(n_iter):
        # x-step: minimize ||B w - y||^2 / (2 sigma^2) + sum z_i^2 / (2 theta_i) (+ c^2 / (2 gamma0^2))
        weights = np.empty(k)
        weights[pen] = 1.0 / np.sqrt(theta)
        if transform == "differences":
            weights[0] = 1.0 / gamma0
        B_stack = np.vstack([B / noise_std, np.diag(weights)])
        y_stack = np.concatenate([y / noise_std, np.zeros(k)])
        w = np.linalg.lstsq(B_stack, y_stack, rcond=None)[0]
        # theta-step
        theta = (0.5 * w[pen] ** 2 + vartheta) / (beta + 1.5)
        hist.append(objective(w, theta))
        if return_iterates:
            xs.append(to_x(w)); thetas.append(theta.copy())
        if it > 0 and abs(hist[-2] - hist[-1]) < tol * max(1.0, abs(hist[-1])):
            break
    if return_iterates:
        return to_x(w), theta, np.array(hist), xs, thetas
    return to_x(w), theta, np.array(hist)

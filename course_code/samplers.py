"""
Samplers and diagnostics of Chapter 4 (uncertainty quantification by sampling).

    rw_metropolis(log_post, x0, n_iter, step, chol, rng)       random-walk Metropolis-Hastings
    perturb_optimize(A, y, sigma, B, n_samples, rng)            exact samples of a linear-Gaussian posterior
    gibbs_hierarchical(A, y, sigma, beta, vartheta, ...)        Gibbs sampler for the hierarchical prior (Chapter 3)
    gibbs_laplace(A, y, sigma, gamma, ...)                      Gibbs sampler for the Laplace prior as a scale mixture
    hmc(U, grad_U, x0, n_iter, eps, n_leap, Minv_chol, rng)     Hamiltonian Monte Carlo (NumPy, any callables)
    make_hmc_jax(U, eps, n_leap, Minv_chol)                     jitted HMC step for a JAX potential U
    autocorrelation(x), iat(x), ess(x)                          autocorrelation, integrated autocorrelation time, ESS
    split_rhat(chains), rank_rhat(chains)                       potential scale reduction factors

Conventions: a posterior is given by its negative logarithm U(x) = -log p(x | y) + const ("potential");
priors on increments use transform="differences" exactly as in solvers.ias and priors.py.
"""
import numpy as np
from scipy.stats import norm, rankdata


# ---------------------------------------------------------------- random-walk Metropolis-Hastings
def rw_metropolis(log_post, x0, n_iter, step, chol=None, rng=None):
    """Random-walk Metropolis-Hastings with proposal x' = x + step * C xi, xi ~ N(0, I).

    log_post : callable x -> log p(x | y) up to a constant (may return -inf or nan outside the support)
    chol     : optional matrix C (proposal covariance step^2 C C^T); identity if None
    Returns the chain (n_iter + 1, n) including x0 and the acceptance rate."""
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    lp = log_post(x)
    chain = np.empty((n_iter + 1, n))
    chain[0] = x
    accepted = 0
    for k in range(n_iter):
        xi = rng.standard_normal(n)
        prop = x + step * (xi if chol is None else chol @ xi)
        lp_prop = log_post(prop)
        if np.isfinite(lp_prop) and np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop
            accepted += 1
        chain[k + 1] = x
    return chain, accepted / n_iter


# ---------------------------------------------------------------- exact samples of a linear-Gaussian posterior
def perturb_optimize(A, y, sigma, B, n_samples, rng=None, b=None):
    """Exact samples of the posterior of y = A x + e, e ~ N(0, sigma^2 I), with the Gaussian prior
    p(x) ~ exp(-1/2 ||B x - b||^2) (B square root of the prior precision, b = B mu; b = 0 if None):
        x_k = argmin 1/2 ||(A x - (y + sigma xi_k)) / sigma||^2 + 1/2 ||B x - (b + eta_k)||^2,
    with xi_k ~ N(0, I_m), eta_k ~ N(0, I_p) independent. One stacked least-squares solve per sample
    (here with one QR factorization shared by all samples). Returns an array (n_samples, n)."""
    rng = np.random.default_rng() if rng is None else rng
    m, n = A.shape
    p = B.shape[0]
    b = np.zeros(p) if b is None else b
    M = np.vstack([A / sigma, B])
    Q, R = np.linalg.qr(M)
    rhs = np.concatenate([np.broadcast_to(y / sigma, (n_samples, m)) + rng.standard_normal((n_samples, m)),
                          np.broadcast_to(b, (n_samples, p)) + rng.standard_normal((n_samples, p))], axis=1)
    return np.linalg.solve(R, Q.T @ rhs.T).T


# ---------------------------------------------------------------- Gibbs samplers for priors on increments
def _increment_setup(A, transform):
    """Variables w = (c, z) with x = c 1 + S z for priors on increments z = L x, or w = x for transform="identity"."""
    m, n = A.shape
    if transform == "identity":
        return A, np.ones(n, dtype=bool), (lambda w: w)
    if transform == "differences":
        S = np.tril(np.ones((n, n)), -1)[:, :-1]
        B = np.hstack([A @ np.ones((n, 1)), A @ S])        # A x = c A 1 + A S z
        pen = np.concatenate([[False], np.ones(n - 1, dtype=bool)])
        return B, pen, (lambda w: w[..., :1] + w[..., 1:] @ S.T)
    raise ValueError("transform must be 'identity' or 'differences'")


def _conditional_gaussian_draw(Bm, y, sigma, prec_diag, rng):
    """One exact draw of w ~ p(w | variances) for y = Bm w + e and independent N(0, 1/prec_diag) priors on w,
    by perturb-then-optimize (Cholesky of the n x n precision)."""
    H = Bm.T @ Bm / sigma**2 + np.diag(prec_diag)
    Lc = np.linalg.cholesky(H)
    rhs = Bm.T @ (y + sigma * rng.standard_normal(len(y))) / sigma**2 + np.sqrt(prec_diag) * rng.standard_normal(len(prec_diag))
    return np.linalg.solve(Lc.T, np.linalg.solve(Lc, rhs))


def gibbs_hierarchical(A, y, sigma, beta, vartheta, n_iter, transform="differences", gamma0=1.0, theta0=None, rng=None):
    """Gibbs sampler for y = A x + e, e ~ N(0, sigma^2 I), and the hierarchical prior
        z_i | theta_i ~ N(0, theta_i), theta_i ~ InvGamma(beta, vartheta),  z = x or z = L x (increments),
    (the level of x has the prior N(0, gamma0^2) for increments). Alternates
        x | theta, y   exact Gaussian draw (perturb-then-optimize),
        theta_i | x    ~ InvGamma(beta + 1/2, vartheta + z_i^2 / 2).
    Replacing both draws by the modes of the conditionals gives the iterative alternating scheme (solvers.ias).
    Returns the chains of x (n_iter, n) and theta (n_iter, k), started from theta0 (default: the hyper-prior mode)."""
    rng = np.random.default_rng() if rng is None else rng
    Bm, pen, to_x = _increment_setup(A, transform)
    k = int(pen.sum())
    theta = np.full(k, vartheta / (beta + 1.0) if theta0 is None else theta0, dtype=float)
    xs, thetas = np.empty((n_iter, A.shape[1])), np.empty((n_iter, k))
    prec = np.empty(Bm.shape[1])
    for it in range(n_iter):
        prec[pen] = 1.0 / theta
        prec[~pen] = 1.0 / gamma0**2
        w = _conditional_gaussian_draw(Bm, y, sigma, prec, rng)
        z = w[pen]
        theta = (vartheta + 0.5 * z**2) / rng.gamma(beta + 0.5, 1.0, k)      # InvGamma(a, b) = b / Gamma(a, 1)
        xs[it], thetas[it] = to_x(w), theta
    return xs, thetas


def gibbs_laplace(A, y, sigma, gamma, n_iter, transform="differences", gamma0=1.0, rng=None):
    """Gibbs sampler for the Laplace prior p(z_i) = exp(-|z_i| / gamma) / (2 gamma), written as the scale mixture
        z_i | tau_i ~ N(0, tau_i), tau_i ~ Exp(rate 1 / (2 gamma^2))   (Andrews & Mallows; Park & Casella).
    Alternates x | tau, y (exact Gaussian draw) and 1 / tau_i | z_i ~ InverseGaussian(1 / (gamma |z_i|), 1 / gamma^2).
    Returns the chains of x (n_iter, n) and tau (n_iter, k)."""
    rng = np.random.default_rng() if rng is None else rng
    Bm, pen, to_x = _increment_setup(A, transform)
    k = int(pen.sum())
    tau = np.full(k, 2.0 * gamma**2)                                      # prior mean of tau
    xs, taus = np.empty((n_iter, A.shape[1])), np.empty((n_iter, k))
    prec = np.empty(Bm.shape[1])
    for it in range(n_iter):
        prec[pen] = 1.0 / tau
        prec[~pen] = 1.0 / gamma0**2
        w = _conditional_gaussian_draw(Bm, y, sigma, prec, rng)
        z = np.maximum(np.abs(w[pen]), 1e-300)
        tau = 1.0 / rng.wald(1.0 / (gamma * z), 1.0 / gamma**2)          # numpy's wald = inverse Gaussian
        xs[it], taus[it] = to_x(w), tau
    return xs, taus


# ---------------------------------------------------------------- Hamiltonian Monte Carlo
def hmc(U, grad_U, x0, n_iter, eps, n_leap, Minv_chol=None, rng=None, jitter=0.2):
    """Hamiltonian Monte Carlo for the density exp(-U(x)) with mass matrix M = (C C^T)^{-1}, C = Minv_chol
    (identity if None): momentum p ~ N(0, M), n_leap leapfrog steps of size eps, Metropolis correction.
    The step size is drawn uniformly from [(1 - jitter) eps, (1 + jitter) eps] in every iteration, which avoids
    periodic trajectories (Neal 2011). Non-finite energies are rejected.
    Returns the chain (n_iter + 1, n) and the acceptance rate."""
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    C = np.eye(n) if Minv_chol is None else Minv_chol
    Ux, gx = U(x), grad_U(x)
    chain = np.empty((n_iter + 1, n)); chain[0] = x
    accepted = 0
    for k in range(n_iter):
        # momentum p = M^{1/2} xi; in the variables q = C^{-1} x the mass matrix is the identity
        r = rng.standard_normal(n)                                       # r = C^T p
        H0 = Ux + 0.5 * r @ r
        h = eps * (1.0 + jitter * (2.0 * rng.random() - 1.0))
        xn, gn, rn = x.copy(), gx.copy(), r.copy()
        rn = rn - 0.5 * h * (C.T @ gn)
        for j in range(n_leap):
            xn = xn + h * (C @ rn)
            gn = grad_U(xn)
            if not np.all(np.isfinite(gn)):
                break
            if j < n_leap - 1:
                rn = rn - h * (C.T @ gn)
        rn = rn - 0.5 * h * (C.T @ gn)
        Un = U(xn)
        H1 = Un + 0.5 * rn @ rn
        if np.isfinite(H1) and np.log(rng.random()) < H0 - H1:
            x, Ux, gx = xn, Un, gn
            accepted += 1
        chain[k + 1] = x
    return chain, accepted / n_iter


def make_hmc_jax(U, eps, n_leap, Minv_chol, jitter=0.2):
    """Jitted HMC transition for a JAX potential U (jnp array -> scalar), mass matrix M = (C C^T)^{-1}, step size
    jittered as in hmc(). Returns step(key, x) -> (x_new, accepted, energy_error). Gradients by jax.grad."""
    import jax
    import jax.numpy as jnp
    C = jnp.asarray(Minv_chol)
    grad_U = jax.grad(U)

    def leapfrog(x, r, h):
        r = r - 0.5 * h * (C.T @ grad_U(x))

        def body(_, xr):
            x, r = xr
            x = x + h * (C @ r)
            r = r - h * (C.T @ grad_U(x))
            return (x, r)

        x, r = jax.lax.fori_loop(0, n_leap - 1, body, (x, r))
        x = x + h * (C @ r)
        r = r - 0.5 * h * (C.T @ grad_U(x))
        return x, r

    @jax.jit
    def step(key, x):
        k1, k2, k3 = jax.random.split(key, 3)
        r0 = jax.random.normal(k1, x.shape)
        h = eps * (1.0 + jitter * (2.0 * jax.random.uniform(k3) - 1.0))
        H0 = U(x) + 0.5 * r0 @ r0
        xn, rn = leapfrog(x, r0, h)
        H1 = U(xn) + 0.5 * rn @ rn
        dH = jnp.where(jnp.isfinite(H1), H1 - H0, jnp.inf)
        accept = jnp.log(jax.random.uniform(k2)) < -dH
        return jnp.where(accept, xn, x), accept, dH

    return step


# ---------------------------------------------------------------- diagnostics
def autocorrelation(x, max_lag=None):
    """Normalized autocorrelation rho_0, rho_1, ... of a scalar chain (FFT based)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    N = len(x)
    f = np.fft.rfft(x, n=2 * N)
    acf = np.fft.irfft(f * np.conj(f))[:N]
    acf /= acf[0] if acf[0] > 0 else 1.0
    return acf if max_lag is None else acf[:max_lag + 1]


def iat(x):
    """Integrated autocorrelation time tau = 1 + 2 sum_k rho_k, truncated with Geyer's initial positive sequence
    (sums of consecutive pairs of autocorrelations are summed while they stay positive)."""
    rho = autocorrelation(x)
    tau = -1.0
    for k in range(0, len(rho) - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
    return max(tau, 1.0 / len(x))


def ess(x):
    """Effective sample size N / tau of a scalar chain (or the sum over several chains of shape (C, N)).
    Antithetic chains can have tau < 1; as in Stan and ArviZ the ESS of a chain is capped at N log10(N)."""
    x = np.atleast_2d(x)
    return float(sum(min(len(c) / iat(c), len(c) * np.log10(len(c))) for c in x))


def split_rhat(chains):
    """Split-Rhat (Gelman et al.) of a scalar quantity from chains of shape (C, N)."""
    chains = np.asarray(chains, dtype=float)
    C, N = chains.shape
    half = N // 2
    s = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    n = s.shape[1]
    W = s.var(axis=1, ddof=1).mean()
    B = n * s.mean(axis=1).var(ddof=1)
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def rank_rhat(chains):
    """Rank-normalized split-Rhat of Vehtari et al. (2021): the larger of the split-Rhat of the rank-normalized
    draws and of the rank-normalized folded draws |x - median|."""
    chains = np.asarray(chains, dtype=float)

    def z(a):
        r = rankdata(a, method="average").reshape(a.shape)
        return norm.ppf((r - 3.0 / 8.0) / (a.size + 1.0 / 4.0))

    folded = np.abs(chains - np.median(chains))
    return max(split_rhat(z(chains)), split_rhat(z(folded)))


# ---------------------------------------------------------------- elliptical slice sampling (Chapter 4)
def elliptical_slice(log_lik, w0, n_iter, rng=None):
    """Elliptical slice sampling (Murray, Adams, MacKay 2010) for p(w | y) ~ L(w) N(w; 0, I).

    log_lik : callable w -> log L(w) (may return -inf or NaN for infeasible w, which is then rejected)
    Returns the chain (n_iter + 1, n) and the average number of likelihood evaluations per iteration."""
    rng = np.random.default_rng() if rng is None else rng
    w = np.asarray(w0, dtype=float).copy()
    ll = log_lik(w)
    chain = np.empty((n_iter + 1, w.size)); chain[0] = w
    n_eval = 0
    for k in range(n_iter):
        nu = rng.standard_normal(w.size)                    # the ellipse through w and a fresh prior draw
        threshold = ll + np.log(rng.random())
        phi = rng.uniform(0.0, 2.0 * np.pi)
        lo, hi = phi - 2.0 * np.pi, phi
        while True:
            w_new = w * np.cos(phi) + nu * np.sin(phi)
            ll_new = log_lik(w_new); n_eval += 1
            if ll_new > threshold:                          # False for NaN
                w, ll = w_new, ll_new
                break
            if phi < 0.0:                                   # shrink the bracket towards phi = 0 (the current state)
                lo = phi
            else:
                hi = phi
            phi = rng.uniform(lo, hi)
        chain[k + 1] = w
    return chain, n_eval / n_iter

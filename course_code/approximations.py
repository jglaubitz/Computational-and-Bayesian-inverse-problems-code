"""
Approximate posteriors of Chapter 5 (beyond MCMC: approximate Bayesian inference).

    lowrank_laplace(J, R, sigma)               eigenpairs of the prior-preconditioned data-misfit Hessian (Section 5.2)
    lowrank_std(lam, V, R, r)                  posterior standard deviations of the rank-r Laplace approximation
    lowrank_sample(x_map, lam, V, R, r, ...)   samples of the rank-r Laplace approximation
    gaussian_vi(U, m0, ...)                    Gaussian variational inference (mean-field or full-rank) by the
                                               reparametrization trick and Adam, for a JAX potential U (Section 5.3)
    pareto_khat(log_w)                         Pareto k-hat of importance weights (PSIS diagnostic)
    importance_estimate(f, X, log_w)           self-normalized importance-sampling estimate and its ESS
    cavi_hierarchical(A, y, sigma, beta, ...)  mean-field variational Bayes for the hierarchical prior of Chapter 3
    ensemble_kalman_step(X, FX, y, sigma, ..)  one ensemble Kalman update with perturbed observations (Section 5.4)
    es_mda(F, X0, y, sigma, n_steps, ...)      ensemble smoother with multiple data assimilation (inflated noise)
    eki(F, X0, y, sigma, n_iter, ...)          iterated ensemble Kalman inversion, optionally with the prior as data

Conventions as in samplers.py: a posterior is given by its potential U(x) = -log p(x | y) + const; the noise is
N(0, sigma^2 I); a Gaussian prior N(mu, R R^T) is written through a factor R ("whitening": x = mu + R w, w ~ N(0, I)).
Forward maps F for the ensemble methods take one unknown (n,) and return the data (m,), or NaN where the model
cannot be evaluated (e.g., no subcritical steady state in the bathymetry problem).
"""
import numpy as np


# ====================================================================== the Laplace approximation and its low-rank form
def lowrank_laplace(J, R, sigma):
    """Eigenpairs of the prior-preconditioned data-misfit Hessian  H_tilde = R^T J^T J R / sigma^2.

    With the Gauss-Newton Hessian H = J^T J / sigma^2 + (R R^T)^{-1} at the MAP estimate,
        H^{-1} = R (I + H_tilde)^{-1} R^T = R R^T - R V diag(lam / (1 + lam)) V^T R^T,
    so the posterior covariance is the prior covariance minus a correction in the directions R v_i in which the data
    dominate the prior (lam_i > 1). H_tilde has rank <= m. Dense eigendecomposition here; for large problems one uses
    Lanczos or a randomized method that needs only products with J and J^T. Returns lam (descending) and V."""
    JR = J @ R / sigma
    lam, V = np.linalg.eigh(JR.T @ JR)
    order = np.argsort(lam)[::-1]
    return np.maximum(lam[order], 0.0), V[:, order]


def lowrank_std(lam, V, R, r):
    """Posterior standard deviations of the Laplace approximation with the r leading eigenpairs."""
    RV = R @ V[:, :r]
    var = np.sum(R**2, axis=1) - np.sum(RV**2 * (lam[:r] / (1.0 + lam[:r])), axis=1)
    return np.sqrt(np.maximum(var, 0.0))


def lowrank_sample(x_map, lam, V, R, r, size, rng):
    """Samples x = x_map + R (I + V_r ((1 + lam_r)^{-1/2} - 1) V_r^T) xi of the rank-r Laplace approximation."""
    xi = rng.standard_normal((size, R.shape[1]))
    Vr = V[:, :r]
    w = xi + ((xi @ Vr) * (1.0 / np.sqrt(1.0 + lam[:r]) - 1.0)) @ Vr.T
    return x_map + w @ R.T


# ====================================================================== Gaussian variational inference (JAX)
def gaussian_vi(U, m0, C0=None, family="full", n_iter=3000, n_samples=8, lr=1e-2, lr_final=1e-4, seed=0, record=10,
                keep_params=False, average=0.0):
    """Maximize the ELBO  -E_q[U(x)] + H[q]  over Gaussians q = N(m, C C^T) by stochastic gradient ascent.

    U       : JAX-differentiable potential of one point x (n,)
    family  : "full" (C lower triangular with positive diagonal) or "mean-field" (C diagonal)
    Reparametrization trick: x = m + C eps with eps ~ N(0, I), so that the ELBO is an expectation over eps and
    its gradient is estimated without bias from n_samples draws; H[q] = sum_i log C_ii + n (1 + log 2 pi) / 2.
    Adam with a learning rate that decays geometrically from lr to lr_final; with average > 0, the returned parameters
    are the average of the iterates over the last fraction `average` of the iterations (Polyak-Ruppert averaging),
    which removes most of the noise of the stochastic gradients. Returns m, C, and the record (iterations, ELBO
    estimates, number of gradient evaluations of U, and with keep_params=True the iterates m, C)."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    n = len(m0)
    tril = np.tril_indices(n, -1)
    C0 = np.eye(n) if C0 is None else np.asarray(C0)
    params = {"m": jnp.asarray(m0, dtype=float), "logd": jnp.log(jnp.asarray(np.diag(C0), dtype=float))}
    if family == "full":
        params["off"] = jnp.asarray(C0[tril], dtype=float)

    def factor(p):
        C = jnp.diag(jnp.exp(p["logd"]))
        if family == "full":
            C = C.at[tril].set(p["off"])
        return C

    def neg_elbo(p, eps):
        x = p["m"] + eps @ factor(p).T                              # reparametrization
        return jnp.mean(jax.vmap(U)(x)) - jnp.sum(p["logd"]) - 0.5 * n * (1.0 + jnp.log(2.0 * jnp.pi))

    vg = jax.jit(jax.value_and_grad(neg_elbo))
    key = jax.random.PRNGKey(seed)
    mom = jax.tree_util.tree_map(jnp.zeros_like, params)
    vel = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, tiny = 0.9, 0.999, 1e-8
    decay = (lr_final / lr) ** (1.0 / max(n_iter - 1, 1))
    rec_it, rec_elbo, rec_m, rec_C = [], [], [], []
    k_avg = int((1.0 - average) * n_iter) if average > 0 else n_iter
    avg, n_avg = None, 0
    for k in range(n_iter):
        key, sub = jax.random.split(key)
        val, g = vg(params, jax.random.normal(sub, (n_samples, n)))
        if k % record == 0 or k == n_iter - 1:                  # the ELBO estimate belongs to the current parameters
            rec_it.append(k); rec_elbo.append(-float(val))
            if keep_params:
                rec_m.append(np.asarray(params["m"])); rec_C.append(np.asarray(factor(params)))
        mom = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, mom, g)
        vel = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, vel, g)
        step = lr * decay**k
        params = jax.tree_util.tree_map(
            lambda p, a, b: p - step * (a / (1 - b1 ** (k + 1))) / (jnp.sqrt(b / (1 - b2 ** (k + 1))) + tiny),
            params, mom, vel)
        if k >= k_avg:
            n_avg += 1
            avg = params if avg is None else jax.tree_util.tree_map(lambda a, b: a + (b - a) / n_avg, avg, params)
    if avg is not None:
        params = avg
    record_out = {"iter": np.array(rec_it), "elbo": np.array(rec_elbo), "gradients": n_iter * n_samples}
    if keep_params:
        record_out["m"], record_out["C"] = np.array(rec_m), np.array(rec_C)
    return np.asarray(params["m"]), np.asarray(factor(params)), record_out


# ====================================================================== importance weights and their diagnostic
def _gpd_fit(x):
    """Shape k and scale of a generalized Pareto distribution fitted to the sorted exceedances x > 0, by the
    method of Zhang and Stephens (2009) with the weakly informative prior on k used in PSIS (Vehtari et al.)."""
    n = len(x)
    mq = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(mq / (np.arange(1, mq + 1) - 0.5))
    bs = bs / (3.0 * x[int(n / 4 + 0.5) - 1]) + 1.0 / x[-1]
    ks = np.mean(np.log1p(-bs[:, None] * x[None, :]), axis=1)
    Lk = n * (np.log(-bs / ks) - ks - 1.0)
    w = 1.0 / np.sum(np.exp(Lk[None, :] - Lk[:, None]), axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, bs = w[keep] / w[keep].sum(), bs[keep]
    b = np.sum(bs * w)
    k = np.mean(np.log1p(-b * x))
    sigma = -k / b
    k = (n * k + 10 * 0.5) / (n + 10)
    return k, sigma


def pareto_khat(log_w):
    """Pareto k-hat of importance weights w = p_tilde / q of draws from q: the shape of a generalized Pareto
    distribution fitted to the M = min(S/5, 3 sqrt(S)) largest weights (Vehtari et al. 2024). k-hat < 0.5: the weights
    have a finite variance; k-hat < 0.7: Pareto-smoothed importance sampling is reliable; k-hat > 0.7: it is not.
    The check sees only where the draws of q land. NaN log weights (infeasible draws) count as weight zero."""
    lw = np.asarray(log_w, dtype=float)
    lw = np.sort(np.where(np.isnan(lw), -np.inf, lw))
    lw = lw[np.isfinite(lw)] - lw[np.isfinite(lw)].max()
    S = len(lw)
    M = int(np.ceil(min(0.2 * S, 3.0 * np.sqrt(S))))
    cutoff = lw[S - M - 1]
    x = np.exp(lw[S - M:]) - np.exp(cutoff)
    if x[-1] <= 0.0 or np.ptp(x) == 0.0:                       # constant weights in the tail: no tail at all
        return -np.inf
    return _gpd_fit(x)[0]


def importance_estimate(fX, log_w):
    """Self-normalized importance-sampling estimate sum_k w_k f(x_k) / sum_k w_k for values fX (S, ...) of draws
    x_k ~ q with log weights log_w, and the effective sample size (sum w)^2 / sum w^2. NaN log weights count as zero."""
    log_w = np.where(np.isnan(log_w), -np.inf, np.asarray(log_w, dtype=float))
    w = np.exp(log_w - np.max(log_w))
    w /= w.sum()
    return np.tensordot(w, fX, axes=1), 1.0 / np.sum(w**2)


# ====================================================================== variational Bayes for the hierarchical prior
def cavi_hierarchical(A, y, sigma, beta, vartheta, gamma0=1.0, n_iter=500, tol=1e-10):
    """Mean-field variational Bayes q(w) q(theta) for y = A x + e, e ~ N(0, sigma^2 I), and the hierarchical prior
    on increments  z_i | theta_i ~ N(0, theta_i), theta_i ~ InvGamma(beta, vartheta),  with x = c 1 + S z and the
    level c ~ N(0, gamma0^2) (as in samplers.gibbs_hierarchical). Coordinate ascent:
        q(theta_i) = InvGamma(beta + 1/2, vartheta + E_q[z_i^2] / 2),   E_q[z_i^2] = m_{z_i}^2 + Sigma_{z_i z_i},
        q(w) = N(m, Sigma),  Sigma = (B^T B / sigma^2 + diag(E_q[1 / theta]))^{-1},  m = Sigma B^T y / sigma^2,
    with E_q[1 / theta_i] = (beta + 1/2) / (vartheta + E_q[z_i^2] / 2). Compare the IAS (solvers.ias), whose
    theta-step is the mode (vartheta + z_i^2 / 2) / (beta + 3/2) of the conditional of theta_i.
    Returns the mean and covariance of x under q, the parameters (a, b) of q(theta), and the number of iterations."""
    m_obs, n = A.shape
    S = np.tril(np.ones((n, n)), -1)[:, :-1]
    B = np.hstack([A @ np.ones((n, 1)), A @ S])                     # A x = c A 1 + A S z
    T = np.hstack([np.ones((n, 1)), S])                             # x = T w
    prec = np.r_[1.0 / gamma0**2, np.full(n - 1, (beta + 1.0) / vartheta)]     # start: theta at the hyper-prior mode
    BtB, Bty = B.T @ B / sigma**2, B.T @ y / sigma**2
    mw_old = np.zeros(n)
    for it in range(n_iter):
        Sigma = np.linalg.inv(BtB + np.diag(prec))
        mw = Sigma @ Bty
        Ez2 = mw[1:] ** 2 + np.diag(Sigma)[1:]
        a, b = beta + 0.5, vartheta + 0.5 * Ez2
        prec[1:] = a / b
        if np.max(np.abs(mw - mw_old)) < tol:
            break
        mw_old = mw
    return T @ mw, T @ Sigma @ T.T, (a, b), it + 1


# ====================================================================== ensemble Kalman methods
def _forward_all(F, X):
    return np.array([F(x) for x in X])


def ensemble_kalman_step(X, FX, y, sigma, rng, alpha=1.0, perturb=True):
    """One ensemble Kalman update of the members X (J, n) with forward values FX (J, m):
        x_j <- x_j + C_xF (C_FF + alpha sigma^2 I)^{-1} (y + eta_j - F(x_j)),   eta_j ~ N(0, alpha sigma^2 I),
    with the ensemble covariances C_xF, C_FF. Only forward values are used: no derivatives. Every update is a
    linear combination of the deviations x_j - mean(X), so the ensemble stays in its initial affine span."""
    Jn = X.shape[0]
    dX = X - X.mean(axis=0)
    dF = FX - FX.mean(axis=0)
    Cxf = dX.T @ dF / (Jn - 1)
    Cff = dF.T @ dF / (Jn - 1)
    eta = np.sqrt(alpha) * sigma * rng.standard_normal(FX.shape) if perturb else 0.0
    K = np.linalg.solve(Cff + alpha * sigma**2 * np.eye(Cff.shape[0]), Cxf.T).T      # Kalman gain (n, m)
    return X + (y + eta - FX) @ K.T


def _safeguard(F, X_old, X_new, max_halvings=30):
    """Forward values at X_new; members at which F fails (NaN) are moved back halfway to their old position until
    F can be evaluated. Returns the members, their forward values, and the number of halvings."""
    FX = _forward_all(F, X_new)
    halvings = 0
    for j in np.nonzero(~np.all(np.isfinite(FX), axis=1))[0]:
        for _ in range(max_halvings):
            X_new[j] = 0.5 * (X_new[j] + X_old[j]); halvings += 1
            FX[j] = F(X_new[j])
            if np.all(np.isfinite(FX[j])):
                break
    return X_new, FX, halvings


def es_mda(F, X0, y, sigma, n_steps, rng, FX0=None):
    """Ensemble smoother with multiple data assimilation (Emerick and Reynolds 2013): n_steps ensemble Kalman
    updates with the noise variance inflated by alpha = n_steps, i.e., each step assimilates the likelihood to the
    power 1 / n_steps. Exact for a linear F and a Gaussian prior ensemble as the ensemble size grows; n_steps = 1
    is the single ensemble Kalman update. Returns the final ensemble, the number of forward evaluations, and the
    number of safeguard halvings."""
    X = np.array(X0, dtype=float)
    FX = _forward_all(F, X) if FX0 is None else FX0
    n_fwd, halv = (0 if FX0 is not None else len(X)), 0
    for k in range(n_steps):
        X_new = ensemble_kalman_step(X, FX, y, sigma, rng, alpha=float(n_steps))
        if k < n_steps - 1:
            X, FX, h = _safeguard(F, X, X_new); n_fwd += len(X) + h; halv += h
        else:
            X = X_new
    return X, n_fwd, halv


def eki(F, X0, y, sigma, n_iter, rng, prior_factor=None, record=None):
    """Iterated ensemble Kalman inversion (Iglesias, Law, and Stuart 2013) with perturbed observations.

    prior_factor : None for plain EKI (the prior enters only through the initial ensemble; the iteration reduces the
                   data misfit within the span of the initial ensemble and must be stopped early), or a matrix Lw with
                   prior N(0, (Lw^T Lw)^{-1}), which is appended as the extra data Lw x = 0 with unit noise
                   (Tikhonov EKI, Chada, Stuart, and Tong 2020); the iteration then approaches the MAP estimate.
    record       : callable(X, FX) -> dict, evaluated after every iteration.
    Returns the final ensemble, the list of records, and the number of forward evaluations."""
    X = np.array(X0, dtype=float)
    FX = _forward_all(F, X)
    n_fwd, recs = len(X), []
    m = len(y)
    for k in range(n_iter):
        if prior_factor is None:
            X_new = ensemble_kalman_step(X, FX, y, sigma, rng)
        else:
            Z = np.hstack([FX / sigma, X @ prior_factor.T])          # whitened augmented data (F(x)/sigma, Lw x)
            yz = np.r_[y / sigma, np.zeros(prior_factor.shape[0])]
            X_new = ensemble_kalman_step(X, Z, yz, 1.0, rng)
        X, FX, h = _safeguard(F, X, X_new); n_fwd += len(X) + h
        if record is not None:
            recs.append(record(X, FX))
    return X, recs, n_fwd

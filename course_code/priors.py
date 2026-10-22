"""
Priors used in Chapters 3 and 4, on the unknown itself (transform="identity"), on its increments
(transform="differences", z = L x with the first-difference matrix L; samples then start at x_1 = 0), or on its
second differences (transform="second-differences"; samples start with x_1 = x_2 = 0).

    gaussian_precision(n, gamma, transform, gamma0)     precision matrix of the Gaussian prior
    sample_gaussian(n, gamma, transform, size, rng)     x_i ~ N(0, gamma^2)                (or the increments)
    se_covariance(t, gamma, ell)                        squared-exponential covariance (Gaussian process prior)
    sample_gaussian_cov(cov, size, rng)                 samples from N(0, cov)
    sample_laplace(n, gamma, transform, size, rng)      x_i ~ Laplace(0, gamma)            (or the increments)
    sample_gsbl(n, beta, vartheta, transform, size, rng)
                                                        x_i | theta_i ~ N(0, theta_i), theta_i ~ InvGamma(beta, vartheta)
    neg_log_invgamma(theta, beta, vartheta)             -log density of the inverse-gamma hyper-prior
    sample_log_invgamma(beta, vartheta, size, rng)      log theta for theta ~ InvGamma(beta, vartheta), safe for tiny beta

All samplers return an array of shape (size, n).
"""
import numpy as np

from solvers import difference_matrix


def cumsum_matrix(n):
    """S in R^{n x (n-1)} with x = S z the signal with x_1 = 0 and increments z, i.e., L S = I."""
    return np.tril(np.ones((n, n)), -1)[:, :-1]


def _assemble(z, transform):
    if transform == "identity":
        return z
    if transform == "differences":
        return z @ cumsum_matrix(z.shape[-1] + 1).T
    if transform == "second-differences":
        x = z @ cumsum_matrix(z.shape[-1] + 1).T               # first integration (increments from 0)
        return x @ cumsum_matrix(x.shape[-1] + 1).T            # second integration
    raise ValueError("transform must be 'identity', 'differences', or 'second-differences'")


def _dim(n, transform):
    return {"identity": n, "differences": n - 1, "second-differences": n - 2}[transform]


def se_covariance(t, gamma, ell):
    """Squared-exponential covariance gamma^2 exp(-(t_i - t_j)^2 / (2 ell^2)) on the points t."""
    return gamma**2 * np.exp(-(t[:, None] - t[None, :]) ** 2 / (2 * ell**2))


def sample_gaussian_cov(cov, size=1, rng=None):
    """Samples from N(0, cov) by a Cholesky factor (a small ridge keeps the factorization stable)."""
    rng = np.random.default_rng() if rng is None else rng
    R = np.linalg.cholesky(cov + 1e-10 * np.trace(cov) / len(cov) * np.eye(len(cov)))
    return (R @ rng.standard_normal((len(cov), size))).T


# ---------------------------------------------------------------- Gaussian
def gaussian_precision(n, gamma, transform="identity", gamma0=1.0):
    """Precision matrix P of the Gaussian prior p(x) ~ exp(-x^T P x / 2):
    identity:    P = I / gamma^2
    differences: P = L^T L / gamma^2 + e_1 e_1^T / gamma0^2   (the level x_1 gets the weak prior N(0, gamma0^2),
                 because L x does not see constant shifts)."""
    if transform == "identity":
        return np.eye(n) / gamma**2
    if transform == "differences":
        L = difference_matrix(n)
        P = L.T @ L / gamma**2
        P[0, 0] += 1.0 / gamma0**2
        return P
    if transform == "second-differences":
        L2 = difference_matrix(n - 1) @ difference_matrix(n)
        P = L2.T @ L2 / gamma**2
        P[0, 0] += 1.0 / gamma0**2
        P[1, 1] += 1.0 / gamma0**2
        return P
    raise ValueError("transform must be 'identity', 'differences', or 'second-differences'")


def sample_gaussian(n, gamma, transform="identity", size=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    z = gamma * rng.standard_normal((size, _dim(n, transform)))
    return _assemble(z, transform)


# ---------------------------------------------------------------- Laplace
def sample_laplace(n, gamma, transform="identity", size=1, rng=None):
    """Density exp(-|z|/gamma) / (2 gamma) per component."""
    rng = np.random.default_rng() if rng is None else rng
    z = rng.laplace(0.0, gamma, (size, _dim(n, transform)))
    return _assemble(z, transform)


# ---------------------------------------------------------------- hierarchical (GSBL with inverse-gamma hyper-prior)
def sample_invgamma(beta, vartheta, size, rng):
    """theta ~ InvGamma(beta, vartheta): density vartheta^beta / Gamma(beta) theta^(-beta-1) exp(-vartheta / theta)."""
    return vartheta / rng.gamma(beta, 1.0, size)


def sample_gsbl(n, beta, vartheta, transform="identity", size=1, rng=None, return_theta=False):
    """z_i | theta_i ~ N(0, theta_i) with theta_i ~ InvGamma(beta, vartheta); marginally z_i is Student-t with
    2 beta degrees of freedom and scale sqrt(vartheta / beta) (beta = 1/2: Cauchy with scale sqrt(2 vartheta))."""
    rng = np.random.default_rng() if rng is None else rng
    theta = sample_invgamma(beta, vartheta, (size, _dim(n, transform)), rng)
    z = np.sqrt(theta) * rng.standard_normal(theta.shape)
    x = _assemble(z, transform)
    return (x, theta) if return_theta else x


def neg_log_invgamma(theta, beta, vartheta):
    """-log p(theta) for theta ~ InvGamma(beta, vartheta), up to the constant."""
    return (beta + 1) * np.log(theta) + vartheta / theta


def sample_log_invgamma(beta, vartheta, size, rng):
    """log(theta) for theta ~ InvGamma(beta, vartheta), without overflow for small beta.
    theta = vartheta / g with g ~ Gamma(beta, 1), and g = G U^(1/beta) with G ~ Gamma(beta + 1, 1), U ~ U(0, 1]:
    for beta = 1/128 the theta_i of one draw span hundreds of orders of magnitude and often exceed the largest double."""
    G = rng.gamma(beta + 1.0, 1.0, size)
    U = 1.0 - rng.random(size)
    return np.log(vartheta) - np.log(G) - np.log(U) / beta

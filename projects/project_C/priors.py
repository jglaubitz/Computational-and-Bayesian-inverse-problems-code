"""
Project C -- priors.

Two priors on the increments z = (b_{i+1} - b_i)_{i=1}^{n-1} of the nodal bottom values:

(1) Gaussian prior on the differences (smoothness prior):
        z_i ~ N(0, lambda^{-1}),  i.e.  p(z) ∝ exp(-lambda/2 ||z||^2),
    with a scale parameter lambda > 0 that you must choose (prior predictive checks) or infer.
    Whitened coordinates: z = lambda^{-1/2} w,  w ~ N(0, I).

(2) Generalized sparse Bayesian learning (GSBL) prior (Glaubitz, Gelb & Song 2023; Glaubitz & Marzouk 2026):
        z_i | theta_i ~ N(0, theta_i),   theta_i ~ GG(r, beta, vartheta),
    with the generalized gamma hyper-prior
        p(theta) = |r| / (Gamma(beta) vartheta) (theta/vartheta)^(r beta - 1) exp(-(theta/vartheta)^r).
    The project uses r = -1, beta = 1, vartheta = 1e-4 (inverse gamma with shape 1 and scale 1e-4).
    Prior-normalized coordinates (hierarchical transport map):
        tau_i = Phi^{-1}(P_theta(theta_i)),   w_i = z_i / sqrt(theta_i),
    under which the GSBL prior becomes a standard normal in (w, tau).

The baseline of Stietz et al. (2026) on the nodal values b themselves (not on increments):

(3) independent Cauchy densities C(0, s) with s = 0.01 on every node ("sparse" prior, isolated features),
    multiplied by a Gaussian "smoothness" density N(0, Sigma) with the squared-exponential covariance
        Sigma_ij = sigma2 exp(-(i - j)^2 / ell^2),   sigma2 = 0.005, ell = 2  (in index units),
    sampled with a random-walk Metropolis-Hastings proposal with covariance Sigma(1e-6, 2); see `baseline_*`.

All priors are written for a general forward map F (steady warm-up: swe.surface_at_sensors(increments_to_bottom(z)),
flume: flume.forward(flume.to_cells(b_nodes, ...), ...)). No inference code is provided: implement MAP estimation
and sampling yourself.
"""
import numpy as np
from scipy.stats import gengamma, norm

GSBL_HYPERPRIOR = (-1.0, 1.0, 1.0e-4)     # (r, beta, vartheta)
BASELINE = dict(cauchy_scale=0.01, sigma2=0.005, ell=2.0, proposal_sigma2=1.0e-6)   # Stietz et al. (2026), Sect. 3


class GeneralizedGamma:
    """GG(r, beta, vartheta) via scipy.stats.gengamma(a=beta, c=r, scale=vartheta)."""

    def __init__(self, r, beta, vartheta):
        self.r, self.beta, self.vartheta = float(r), float(beta), float(vartheta)
        self._d = gengamma(a=self.beta, c=self.r, scale=self.vartheta)

    def logpdf(self, theta):
        return self._d.logpdf(theta)

    def cdf(self, theta):
        return self._d.cdf(theta)

    def ppf(self, p):
        return self._d.ppf(p)

    def sf(self, theta):
        return self._d.sf(theta)

    def isf(self, p):
        return self._d.isf(p)

    def theta_from_tau(self, tau):
        """theta = P^{-1}(Phi(tau)), accurate in both tails (upper tail via survival functions).
        For r = -1, beta = 1 this equals -vartheta / log Phi(tau) (closed form)."""
        tau = np.asarray(tau, dtype=float)
        return np.where(tau < 0, self._d.ppf(norm.cdf(tau)), self._d.isf(norm.sf(tau)))

    def tau_from_theta(self, theta):
        """tau = Phi^{-1}(P(theta)), accurate in both tails."""
        theta = np.asarray(theta, dtype=float)
        c = self._d.cdf(theta)
        return np.where(c < 0.5, norm.ppf(c), norm.isf(self._d.sf(theta)))

    def rvs(self, size, rng):
        return self._d.rvs(size=size, random_state=rng)


# ---------------------------------------------------------------- likelihood
def log_likelihood(z, y, forward, sigma):
    """Gaussian log-likelihood -||F(z) - y||^2 / (2 sigma^2) for data y of any shape (vector or time x sensors)."""
    r = np.ravel(forward(z) - y)
    if np.any(~np.isfinite(r)):
        return -np.inf
    return -0.5 * np.dot(r, r) / sigma**2


# ---------------------------------------------------------------- (1) Gaussian prior on the differences
def log_prior_gaussian(z, lam):
    return -0.5 * lam * np.dot(z, z)


def log_posterior_gaussian(z, y, forward, sigma, lam):
    return log_likelihood(z, y, forward, sigma) + log_prior_gaussian(z, lam)


def sample_prior_gaussian(n_inc, lam, rng):
    return rng.standard_normal(n_inc) / np.sqrt(lam)


# ---------------------------------------------------------------- (2) GSBL prior
def log_prior_gsbl(z, theta, gg):
    return np.sum(-0.5 * np.log(theta) - 0.5 * z**2 / theta) + np.sum(gg.logpdf(theta))


def log_posterior_gsbl(z, theta, y, forward, sigma, gg):
    if np.any(theta <= 0):
        return -np.inf
    return log_likelihood(z, y, forward, sigma) + log_prior_gsbl(z, theta, gg)


def map_forward(z, theta, gg):
    """S: (z, theta) -> (w, tau); pushes the GSBL prior to a standard normal."""
    tau = gg.tau_from_theta(theta)   # accurate in both tails
    w = z / np.sqrt(theta)
    return w, tau


def map_inverse(w, tau, gg):
    """T = S^{-1}: (w, tau) -> (z, theta)."""
    theta = gg.theta_from_tau(tau)   # accurate in both tails (gg.ppf(norm.cdf(tau)) is inf for tau > 8.3)
    z = w * np.sqrt(theta)
    return z, theta


def log_posterior_gsbl_normalized(w, tau, y, forward, sigma, gg):
    """Prior-normalized GSBL posterior: phi^y(w, tau) ∝ f(T(w, tau); y) N(w; 0, I) N(tau; 0, I)."""
    z, theta = map_inverse(w, tau, gg)
    if np.any(~np.isfinite(theta)) or np.any(theta <= 0):
        return -np.inf
    return log_likelihood(z, y, forward, sigma) - 0.5 * np.dot(w, w) - 0.5 * np.dot(tau, tau)


def sample_prior_gsbl(n_inc, gg, rng):
    theta = gg.rvs(n_inc, rng)
    z = rng.standard_normal(n_inc) * np.sqrt(theta)
    return z, theta


def ias_theta_update(z, gg):
    """Closed-form theta-step of the iterative alternating scheme (IAS) for the MAP estimate of the GSBL
    posterior: for r = -1, theta_i = (vartheta + z_i^2 / 2) / (beta + 3/2)."""
    if gg.r != -1.0:
        raise NotImplementedError("closed-form IAS update implemented for r = -1 only")
    return (gg.vartheta + 0.5 * z**2) / (gg.beta + 1.5)


# ---------------------------------------------------------------- (3) baseline of Stietz et al. (2026): nodal values
def squared_exponential(n, sigma2, ell):
    """Sigma_ij = sigma2 exp(-(i - j)^2 / ell^2) on n nodes (index units, as in Stietz et al.)."""
    i = np.arange(n)
    return sigma2 * np.exp(-((i[:, None] - i[None, :]) ** 2) / ell**2)


def baseline_log_prior(b, cauchy_scale=BASELINE["cauchy_scale"], sigma2=BASELINE["sigma2"], ell=BASELINE["ell"],
                       Sigma_inv=None):
    """log of  prod_i Cauchy(b_i; 0, s) * N(b; 0, Sigma(sigma2, ell))  up to a constant (b: nodal values, not
    increments). Pass a precomputed Sigma_inv for speed."""
    b = np.asarray(b, dtype=float)
    if Sigma_inv is None:
        Sigma_inv = np.linalg.inv(squared_exponential(len(b), sigma2, ell) + 1e-12 * np.eye(len(b)))
    return -np.sum(np.log1p((b / cauchy_scale) ** 2)) - 0.5 * b @ Sigma_inv @ b


def baseline_proposal_chol(n, proposal_sigma2=BASELINE["proposal_sigma2"], ell=BASELINE["ell"]):
    """Cholesky factor of the random-walk proposal covariance Sigma(1e-6, 2) used by Stietz et al. for the
    measured data (b' = b + L xi, xi ~ N(0, I)); their alternative was an independent proposal with variance 1e-6."""
    return np.linalg.cholesky(squared_exponential(n, proposal_sigma2, ell) + 1e-15 * np.eye(n))

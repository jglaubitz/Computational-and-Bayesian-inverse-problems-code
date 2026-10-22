"""
Project A -- hyper-priors for the hierarchical prior on increments (NumPy/SciPy).
No variational code: deriving and implementing the theta-steps of variational Bayes is task M2.

    z_i | theta_i ~ N(0, theta_i),   theta_i ~ p(theta)

Hyper-priors provided (all with logpdf, cdf, sf, ppf, isf, rvs):
    InvGamma(beta, vartheta)   generalized gamma with r = -1; marginal of z: Student-t with 2 beta d.o.f.
    Gamma(beta, vartheta)      generalized gamma with r = 1 (shape beta, scale vartheta); beta = 1: Laplace marginal
                               with scale sqrt(vartheta / 2)
    GenGamma(r, beta, vartheta) generalized gamma GG(r, beta, vartheta) (scipy.stats.gengamma)
    HalfCauchySD(s)            sqrt(theta) ~ C+(0, s), i.e., p(theta) = s / (pi sqrt(theta) (s^2 + theta));
                               marginal of z: horseshoe (local scales only, global scale fixed to s)

Conditionals for Gibbs samplers (sample_conditional(z2, rng) draws theta_i | z_i for all i at once, z2 = z**2):
    InvGamma      exact: InvGamma(beta + 1/2, vartheta + z^2 / 2)
    Gamma         exact: generalized inverse Gaussian GIG(p = beta - 1/2, a = 2 / vartheta, b = z^2)
                  (density ∝ theta^(p-1) exp(-(a theta + b / theta) / 2), drawn with scipy.stats.geninvgauss)
    GenGamma      inverse transform on a fine grid in log(theta) (approximate; check the grid for your settings)
    HalfCauchySD  the two-level auxiliary representation of Makalic & Schmidt (2016),
                      theta | nu ~ InvGamma(1/2, 1 / nu),   nu ~ InvGamma(1/2, 1 / s^2),
                  with exact conditionals  theta | z, nu ~ InvGamma(1, z^2 / 2 + 1 / nu),
                                           nu | theta    ~ InvGamma(1, 1 / theta + 1 / s^2):
                  sample_conditional_aux(z2, nu, rng) returns (theta, nu); sample_conditional uses the grid.

The hyper-priors of the project are in DEFAULTS (see the project description, Table 1).
"""
import numpy as np
from scipy.special import gammaln
from scipy.stats import gengamma, geninvgauss, invgamma, gamma as gamma_dist, norm


class HyperPrior:
    name = "hyper-prior"

    # ---- to be provided by the subclasses: logpdf, cdf, sf, ppf, isf, rvs
    def pdf(self, theta):
        return np.exp(self.logpdf(theta))

    def theta_from_tau(self, tau):
        """theta = P^{-1}(Phi(tau)) (hierarchical prior normalization), accurate in both tails."""
        tau = np.asarray(tau, dtype=float)
        return np.where(tau < 0, self.ppf(norm.cdf(tau)), self.isf(norm.sf(tau)))

    def tau_from_theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        c = self.cdf(theta)
        return np.where(c < 0.5, norm.ppf(np.minimum(c, 0.5)), norm.isf(self.sf(theta)))

    def grid_conditional(self, z2, rng, u_min=np.log(1e-14), u_max=np.log(1e4), n_grid=2001):
        """theta_i | z_i ∝ p(theta) N(z_i; 0, theta) by inverse transform on a grid in u = log(theta).
        Approximate (piecewise constant in u, cell width (u_max - u_min) / (n_grid - 1)); fine for Gibbs references
        if the grid covers the conditional. Vectorized over i; costs O(len(z2) * n_grid)."""
        u = np.linspace(u_min, u_max, n_grid)
        th = np.exp(u)
        lg = (self.logpdf(th) + 0.5 * u)[None, :] - np.asarray(z2)[:, None] / (2.0 * th[None, :])
        w = np.exp(lg - lg.max(axis=1, keepdims=True))
        cdf = np.cumsum(w, axis=1)
        cdf /= cdf[:, -1:]
        idx = np.minimum((cdf < rng.random(len(z2))[:, None]).sum(axis=1), n_grid - 1)
        return np.exp(u[idx] + (u[1] - u[0]) * (rng.random(len(z2)) - 0.5))


class InvGamma(HyperPrior):
    def __init__(self, beta, vartheta):
        self.beta, self.vartheta = float(beta), float(vartheta)
        self._d = invgamma(a=self.beta, scale=self.vartheta)
        self.name = f"InvGamma({beta:g}, {vartheta:g})"

    def logpdf(self, th):
        b, v = self.beta, self.vartheta
        return b * np.log(v) - gammaln(b) - (b + 1.0) * np.log(th) - v / th

    def cdf(self, th): return self._d.cdf(th)
    def sf(self, th): return self._d.sf(th)
    def ppf(self, p): return self._d.ppf(p)
    def isf(self, p): return self._d.isf(p)
    def rvs(self, size, rng): return self._d.rvs(size=size, random_state=rng)

    def theta_from_tau(self, tau):
        """Closed form for beta = 1: P(theta) = exp(-vartheta / theta), theta = -vartheta / log Phi(tau)."""
        if self.beta == 1.0:
            from scipy.special import log_ndtr
            return -self.vartheta / log_ndtr(np.asarray(tau, dtype=float))
        return super().theta_from_tau(tau)

    def sample_conditional(self, z2, rng):
        return (self.vartheta + 0.5 * np.asarray(z2)) / rng.gamma(self.beta + 0.5, 1.0, len(z2))


class Gamma(HyperPrior):
    def __init__(self, beta, vartheta):
        self.beta, self.vartheta = float(beta), float(vartheta)
        self._d = gamma_dist(a=self.beta, scale=self.vartheta)
        self.name = f"Gamma({beta:g}, {vartheta:g})"

    def logpdf(self, th):
        b, v = self.beta, self.vartheta
        return -gammaln(b) - b * np.log(v) + (b - 1.0) * np.log(th) - th / v

    def cdf(self, th): return self._d.cdf(th)
    def sf(self, th): return self._d.sf(th)
    def ppf(self, p): return self._d.ppf(p)
    def isf(self, p): return self._d.isf(p)
    def rvs(self, size, rng): return self._d.rvs(size=size, random_state=rng)

    def sample_conditional(self, z2, rng):
        """GIG(p, a, b) with p = beta - 1/2, a = 2 / vartheta, b = z^2: in scipy's parametrization
        geninvgauss(p, b_s, scale) has density ∝ x^(p-1) exp(-b_s (x / scale + scale / x) / 2) / scale,
        so b_s = sqrt(a b) and scale = sqrt(b / a). Slow (scipy's ratio-of-uniforms); for beta = 1 the
        course sampler samplers.gibbs_laplace is faster and equivalent."""
        p, a = self.beta - 0.5, 2.0 / self.vartheta
        b = np.maximum(np.asarray(z2), 1e-300)
        return geninvgauss.rvs(p, np.sqrt(a * b), scale=np.sqrt(b / a), random_state=rng)


class GenGamma(HyperPrior):
    """GG(r, beta, vartheta): density |r| / (Gamma(beta) vartheta) (theta/vartheta)^(r beta - 1) exp(-(theta/vartheta)^r)."""

    def __init__(self, r, beta, vartheta):
        self.r, self.beta, self.vartheta = float(r), float(beta), float(vartheta)
        self._d = gengamma(a=self.beta, c=self.r, scale=self.vartheta)
        self.name = f"GG({r:g}, {beta:g}, {vartheta:.4g})"

    def logpdf(self, th):
        r, b, v = self.r, self.beta, self.vartheta
        return np.log(abs(r)) - gammaln(b) - np.log(v) + (r * b - 1.0) * np.log(th / v) - (th / v) ** r

    def cdf(self, th): return self._d.cdf(th)
    def sf(self, th): return self._d.sf(th)
    def ppf(self, p): return self._d.ppf(p)
    def isf(self, p): return self._d.isf(p)
    def rvs(self, size, rng): return self._d.rvs(size=size, random_state=rng)

    def sample_conditional(self, z2, rng):
        return self.grid_conditional(z2, rng)


class HalfCauchySD(HyperPrior):
    """sqrt(theta) ~ C+(0, s): p(theta) = s / (pi sqrt(theta) (s^2 + theta)), P(theta) = (2/pi) arctan(sqrt(theta)/s)."""

    def __init__(self, s):
        self.s = float(s)
        self.name = f"half-Cauchy({s:.4g})"

    def logpdf(self, th):
        return np.log(self.s / np.pi) - 0.5 * np.log(th) - np.log(self.s**2 + th)

    def cdf(self, th): return 2.0 / np.pi * np.arctan(np.sqrt(th) / self.s)
    def sf(self, th): return 2.0 / np.pi * np.arctan(self.s / np.sqrt(th))
    def ppf(self, p): return (self.s * np.tan(0.5 * np.pi * np.asarray(p))) ** 2
    def isf(self, p): return (self.s / np.tan(0.5 * np.pi * np.asarray(p))) ** 2
    def rvs(self, size, rng): return (self.s * np.abs(rng.standard_cauchy(size))) ** 2

    def sample_conditional(self, z2, rng):
        return self.grid_conditional(z2, rng)

    def sample_conditional_aux(self, z2, nu, rng):
        """One sweep of the two-level representation: theta | z, nu, then nu | theta (both exact)."""
        k = len(z2)
        theta = (0.5 * np.asarray(z2) + 1.0 / nu) / rng.gamma(1.0, 1.0, k)
        nu = (1.0 / theta + 1.0 / self.s**2) / rng.gamma(1.0, 1.0, k)
        return theta, nu


VT = 1.0e-3     # the scale of the inverse gamma hyper-prior of the lecture notes
DEFAULTS = {
    "IG":  InvGamma(1.0, VT),                                       # r = -1, Student-t with 2 d.o.f. (the notes)
    "G":   Gamma(1.0, 2 * 0.1**2),                                  # r = 1, Laplace prior with gamma = 0.1 (the notes)
    "HC":  HalfCauchySD(np.sqrt(VT)),                               # horseshoe with scale sqrt(vartheta)
    "GGm": GenGamma(-0.5, 2.0165, 1.2583e-3 * VT / 1.2308e-4),      # r = -1/2, see the project description
}

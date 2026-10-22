"""
Project B -- Bayesian atom-conserving chemical reaction neural networks (AC-CRNN)
Forward model in NumPy/SciPy.  No training or inference code: that is your job.

A CRNN (Ji & Deng, 2021) with N_r reaction nodes and N_s species encodes, for each
reaction j, the mass-action rate
        r_j(c) = k_j * prod_i c_i^{a_{ij}},         a_{ij} >= 0  (reaction orders of the reactants)
and the stoichiometric vector nu_j in R^{N_s}, so that
        dc/dt = sum_j nu_j r_j(c) = NU r(c),       NU = [nu_1, ..., nu_{N_r}]  (N_s x N_r).

Atom conservation (Doeppel & Votsmeier, 2023): with the molecular matrix N (N_atoms x N_s),
a physically realizable stoichiometry satisfies N nu_j = 0. Writing nu_j = B* w_j with the
atom-conservation basis B* (N_s x N_key, N_key = N_s - rank N, reduced column echelon form,
identity in the rows of the "key species") enforces this as a hard constraint and reduces
the number of parameters.

Identifiability warning (free orders): with a_j and w_j as independent parameters, the map
        (w_j, logk_j) -> (s w_j, logk_j - log s),   s > 0,
leaves every trajectory unchanged (nu_j r_j is invariant), for each node j separately.  Ji & Deng
(2021, case I) avoid this by tying the orders of the reactants to their stoichiometric coefficients,
        a_ij = max(-nu_ij, 0)           (elementary reactions),
which the functions *_tied below implement; this is the default model of the project.

Parameters of a node j (what you put priors on):
        a_j    in R^{N_s}    reaction orders (free-order model only; nonnegative;
                             the original CRNN uses ReLU(w_in))
        w_j    in R^{N_key}  key-species stoichiometric coefficients
        logk_j in R          log rate constant (isothermal case; Arrhenius otherwise)
Free-order model: pack, unpack, simulate, simulate_all, true_params_trimerization
                  (n_r (N_s + N_key + 1) parameters).
Tied model:       pack_tied, unpack_tied, simulate_tied, simulate_all_tied,
                  true_params_trimerization_tied (n_r (N_key + 1) parameters: 20 for the
                  trimerization case, 24 with B = I, i.e., without atom conservation).
"""
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------- atom-conservation basis
def atom_conservation_basis(N, key_species=None):
    """Return B* (N_s x N_key) with nu = B* w satisfying N nu = 0 for all w.

    N            : molecular matrix, shape (N_atoms, N_s), entry = number of atoms of type k in species i
    key_species  : indices of the N_key = N_s - rank(N) key species (default: first N_key species
                   that give a full-rank dependent block).
    The rows of B* belonging to the key species form the identity; the remaining rows are
    determined by atom conservation.
    """
    N = np.atleast_2d(np.asarray(N, dtype=float))
    n_atoms, n_s = N.shape
    rank = np.linalg.matrix_rank(N)
    n_key = n_s - rank
    if key_species is None:
        # greedy: choose dependent species so that N[:, dep] has full column rank
        dep = []
        for i in range(n_s - 1, -1, -1):
            trial = dep + [i]
            if np.linalg.matrix_rank(N[:, trial]) == len(trial) and len(trial) <= rank:
                dep = trial
        dep = sorted(dep)
        key = [i for i in range(n_s) if i not in dep]
    else:
        key = sorted(key_species)
        dep = [i for i in range(n_s) if i not in key]
    assert len(key) == n_key, "number of key species must equal N_s - rank(N)"
    # N_key w + N_dep nu_dep = 0  =>  nu_dep = -pinv(N_dep) N_key w
    N_key, N_dep = N[:, key], N[:, dep]
    B = np.zeros((n_s, n_key))
    B[key, :] = np.eye(n_key)
    B[dep, :] = -np.linalg.pinv(N_dep) @ N_key
    return B, key, dep


# ---------------------------------------------------------------- CRNN forward model
def unpack(params, n_s, n_r, n_key):
    """params = [a (n_r*n_s), w (n_r*n_key), logk (n_r)] -> (a, w, logk) with shapes (n_r, n_s), (n_r, n_key), (n_r,)."""
    params = np.asarray(params, dtype=float)
    i = 0
    a = params[i:i + n_r * n_s].reshape(n_r, n_s); i += n_r * n_s
    w = params[i:i + n_r * n_key].reshape(n_r, n_key); i += n_r * n_key
    logk = params[i:i + n_r]
    return a, w, logk


def pack(a, w, logk):
    return np.concatenate([np.ravel(a), np.ravel(w), np.ravel(logk)])


def stoichiometry(w, B):
    """NU (n_s x n_r) from key weights w (n_r x n_key) and basis B (n_s x n_key)."""
    return (B @ w.T)


def rhs(t, c, a, NU, k, eps=1e-10):
    """dc/dt = NU r(c),  r_j = k_j prod_i c_i^{a_ij}  (orders clipped at 0, concentrations at eps)."""
    cpos = np.maximum(c, eps)
    log_r = np.log(k) + np.maximum(a, 0.0) @ np.log(cpos)      # (n_r,)
    return NU @ np.exp(log_r)


def simulate(params, c0, t_eval, B, n_s, n_r, rtol=1e-7, atol=1e-9):
    """Integrate the CRNN ODE from c0 (n_s,) and return concentrations at t_eval (len(t_eval), n_s)."""
    n_key = B.shape[1]
    a, w, logk = unpack(params, n_s, n_r, n_key)
    NU = stoichiometry(w, B)
    k = np.exp(logk)
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), c0, t_eval=t_eval, args=(a, NU, k),
                    method="LSODA", rtol=rtol, atol=atol)
    if not sol.success:
        return np.full((len(t_eval), n_s), np.nan)
    return sol.y.T


def simulate_all(params, c0_all, t_eval, B, n_s, n_r, **kw):
    """Vectorized over experiments: c0_all (n_exp, n_s) -> (n_exp, len(t_eval), n_s)."""
    return np.stack([simulate(params, c0, t_eval, B, n_s, n_r, **kw) for c0 in c0_all])


# ---------------------------------------------------------------- tied orders (elementary reactions; default model)
def orders_from_stoichiometry(NU, max_order=None):
    """Reaction orders a (n_r x n_s) of elementary reactions: a_ij = max(-nu_ij, 0), optionally capped."""
    a = np.maximum(-np.asarray(NU, dtype=float).T, 0.0)
    return a if max_order is None else np.minimum(a, max_order)


def unpack_tied(params, n_r, n_key):
    """params = [w (n_r*n_key), logk (n_r)] -> (w, logk) with shapes (n_r, n_key), (n_r,)."""
    params = np.asarray(params, dtype=float)
    return params[:n_r * n_key].reshape(n_r, n_key), params[n_r * n_key:n_r * n_key + n_r]


def pack_tied(w, logk):
    return np.concatenate([np.ravel(w), np.ravel(logk)])


def simulate_tied(params, c0, t_eval, B, n_s, n_r, max_order=None, rtol=1e-7, atol=1e-9):
    """As `simulate`, with the orders tied to the stoichiometry. Without atom conservation pass B = I (n_s x n_s)."""
    w, logk = unpack_tied(params, n_r, B.shape[1])
    NU = stoichiometry(w, B)
    a = orders_from_stoichiometry(NU, max_order)
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), c0, t_eval=t_eval, args=(a, NU, np.exp(logk)),
                    method="LSODA", rtol=rtol, atol=atol)
    if not sol.success:
        return np.full((len(t_eval), n_s), np.nan)
    return sol.y.T


def simulate_all_tied(params, c0_all, t_eval, B, n_s, n_r, **kw):
    return np.stack([simulate_tied(params, c0, t_eval, B, n_s, n_r, **kw) for c0 in c0_all])


# ---------------------------------------------------------------- the true mechanism as CRNN parameters
def true_params_trimerization(B):
    """Express the trimerization mechanism (Searson et al.; Ji & Deng 2021, case 1) in CRNN form.

    Species: [A, A2, A*, A**, A3];   reactions:
        2 A       -> A2      k1 = 0.30
        A         -> A*      k2 = 0.10
        A*        -> A**     k3 = 0.20
        A2 + A**  -> A3      k4 = 0.13
    """
    NU_true = np.array([[-2, -1,  0,  0],
                        [ 1,  0,  0, -1],
                        [ 0,  1, -1,  0],
                        [ 0,  0,  1, -1],
                        [ 0,  0,  0,  1]], dtype=float)           # (n_s, n_r)
    a_true = np.array([[2, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 0, 1, 0]], dtype=float)             # (n_r, n_s) reaction orders
    k_true = np.array([0.30, 0.10, 0.20, 0.13])
    # key weights: least-squares fit NU = B w^T (exact if the mechanism conserves atoms)
    w_true = np.linalg.lstsq(B, NU_true, rcond=None)[0].T          # (n_r, n_key)
    assert np.allclose(B @ w_true.T, NU_true), "true mechanism must lie in the atom-conserving subspace"
    return pack(a_true, w_true, np.log(k_true)), NU_true, a_true, k_true


def true_params_trimerization_tied(B):
    """The true mechanism in the tied parametrization [w, logk] (20 parameters with the AC layer, 24 with B = I)."""
    _, NU_true, a_true, k_true = true_params_trimerization(B)
    w_true = np.linalg.lstsq(B, NU_true, rcond=None)[0].T
    assert np.allclose(orders_from_stoichiometry(NU_true), a_true), "true mechanism must be elementary"
    return pack_tied(w_true, np.log(k_true))


MOLECULAR_MATRIX_TRIMERIZATION = np.array([[1, 2, 1, 1, 3]], dtype=float)   # atoms of A per species
SPECIES_TRIMERIZATION = ["A", "A2", "A*", "A**", "A3"]

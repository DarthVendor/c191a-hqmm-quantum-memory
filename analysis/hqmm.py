"""Hidden Markov models (HMMs) and hidden quantum Markov models (HQMMs)
written in one common linear form, so the same code computes sequence
probabilities, Hankel matrices, sampling and predictive log loss for both.

Every model here is an observable-operator model (OOM):
    P(x_1 ... x_n) = sigma @ tau[x_n] @ ... @ tau[x_1] @ w0
  * HMM:   w0 = belief vector pi,  tau[x] = T_x (column-stochastic in total),
           sigma = all-ones row.
  * HQMM:  w0 = vec(rho0),  tau[x] = sum_w conj(K) kron K  (superoperator of
           the instrument element E_x(rho) = sum_w K rho K^dagger),
           sigma = vec(I) (the trace).
Convention: vec is column-major, so vec(A rho B) = (B^T kron A) vec(rho).
"""
from itertools import product

import numpy as np


class LinearModel:
    def __init__(self, w0, taus, sigma, name=""):
        self.w0 = np.asarray(w0)
        self.taus = [np.asarray(t) for t in taus]
        self.sigma = np.asarray(sigma)
        self.name = name

    @property
    def n_symbols(self):
        return len(self.taus)

    def prob(self, word):
        w = self.w0
        for x in word:
            w = self.taus[x] @ w
        return float(np.real(self.sigma @ w))

    def words(self, max_len):
        out = [()]
        for n in range(1, max_len + 1):
            out.extend(product(range(self.n_symbols), repeat=n))
        return out

    def hankel(self, max_len):
        """H[u, v] = P(u v) for all pasts u and futures v up to max_len."""
        ws = self.words(max_len)
        return np.array([[self.prob(u + v) for v in ws] for u in ws])

    def sample(self, length, rng):
        w = self.w0
        out = np.empty(length, dtype=int)
        for t in range(length):
            nxt = [t_x @ w for t_x in self.taus]
            p = np.array([np.real(self.sigma @ v) for v in nxt])
            p = np.clip(p, 0, None)
            p /= p.sum()
            x = rng.choice(self.n_symbols, p=p)
            out[t] = x
            w = nxt[x] / p[x]
        return out

    def log_loss_bits(self, seq):
        """Average -log2 p(x_t | x_<t), filtering forward from w0."""
        w = self.w0
        total = 0.0
        for x in seq:
            v = self.taus[x] @ w
            p = float(np.real(self.sigma @ v))
            total -= np.log2(p)
            w = v / p
        return total / len(seq)


def hankel_rank(H, rel_tol=1e-9):
    s = np.linalg.svd(H, compute_uv=False)
    return int(np.sum(s > rel_tol * s[0])), s


def _fixed_point(M):
    vals, vecs = np.linalg.eig(M)
    k = int(np.argmin(np.abs(vals - 1.0)))
    return vecs[:, k]


# ---------- classical HMMs ----------

def hmm_model(T, pi=None, name="HMM"):
    """T: list over symbols x of n x n arrays, T[x][j, i] = P(x, next=j | now=i)."""
    n = T[0].shape[0]
    if pi is None:
        pi = np.real(_fixed_point(sum(T)))
        pi = pi / pi.sum()
    return LinearModel(pi, T, np.ones(n), name)


def random_hmm(n, n_symbols, rng):
    raw = rng.random((n_symbols, n, n))
    raw /= raw.sum(axis=(0, 1), keepdims=True)  # each column i sums to 1 over (x, j)
    return [raw[x] for x in range(n_symbols)]


# ---------- quantum instruments ----------

def superop(K):
    return np.kron(K.conj(), K)


def hqmm_model(kraus, rho0=None, name="HQMM"):
    """kraus: list over symbols x of lists of d x d Kraus operators K_{x,w}."""
    d = kraus[0][0].shape[0]
    taus = [sum(superop(K) for K in group) for group in kraus]
    if rho0 is None:
        rho0 = _fixed_point(sum(taus)).reshape(d, d, order="F")
        rho0 = (rho0 + rho0.conj().T) / 2
        rho0 = rho0 / np.trace(rho0)
    vec_I = np.eye(d).reshape(-1, order="F")
    return LinearModel(rho0.reshape(-1, order="F"), taus, vec_I, name)


def completeness_error(kraus):
    d = kraus[0][0].shape[0]
    S = sum(K.conj().T @ K for group in kraus for K in group)
    return float(np.max(np.abs(S - np.eye(d))))


def embed_hmm(T):
    """Classical HMM as an HQMM: K_{x,i,j} = sqrt(T_x[j,i]) |j><i|."""
    n = T[0].shape[0]
    groups = []
    for Tx in T:
        group = []
        for i in range(n):
            for j in range(n):
                if Tx[j, i] > 0:
                    K = np.zeros((n, n), dtype=complex)
                    K[j, i] = np.sqrt(Tx[j, i])
                    group.append(K)
        groups.append(group)
    return groups


def coherent_output_embed(T):
    """K_{x,i} = |v_{x,i}><i| with v = sum_j sqrt(T_x[j,i]) |j>.
    The updated state has coherences, but each Kraus reads only <i|rho|i>."""
    n = T[0].shape[0]
    groups = []
    for Tx in T:
        group = []
        for i in range(n):
            K = np.zeros((n, n), dtype=complex)
            K[:, i] = np.sqrt(Tx[:, i])
            group.append(K)
        groups.append(group)
    return groups


def dephase(kraus, basis=None):
    """Remove coherence in the given basis (columns of `basis`, default computational).
    Returns the classical HMM T_x[j,i] = sum_w |<b_j| K_{x,w} |b_i>|^2."""
    d = kraus[0][0].shape[0]
    V = np.eye(d) if basis is None else basis
    return [sum(np.abs(V.conj().T @ K @ V) ** 2 for K in group) for group in kraus]


def _pure_state_from_bloch(n):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    rho = (np.eye(2) + n[0] * sx + n[1] * sy + n[2] * sz) / 2
    vals, vecs = np.linalg.eigh(rho)
    return vecs[:, -1]


def luders_qubit_kraus(bloch_vectors):
    """Rank-one Lueders instrument K_k = sqrt(2/m) |psi_k><psi_k| for m Bloch
    vectors summing to zero; after outcome k the memory is |psi_k>."""
    m = len(bloch_vectors)
    states = [_pure_state_from_bloch(np.asarray(n, float)) for n in bloch_vectors]
    return [[np.sqrt(2 / m) * np.outer(s, s.conj())] for s in states]


TETRAHEDRON = [(0, 0, 1),
               (2 * np.sqrt(2) / 3, 0, -1 / 3),
               (-np.sqrt(2) / 3, np.sqrt(2 / 3), -1 / 3),
               (-np.sqrt(2) / 3, -np.sqrt(2 / 3), -1 / 3)]
TRINE = [(np.sin(2 * np.pi * k / 3), 0, np.cos(2 * np.pi * k / 3)) for k in range(3)]


def hesse_sic_kraus():
    """Qutrit SIC (Weyl-Heisenberg orbit of (0, 1, -1)/sqrt 2), Lueders instrument
    K_k = |psi_k><psi_k| / sqrt 3 with nine outcomes."""
    d = 3
    omega = np.exp(2j * np.pi / d)
    X = np.roll(np.eye(d), 1, axis=0)
    Z = np.diag([omega ** k for k in range(d)])
    fiducial = np.array([0, 1, -1]) / np.sqrt(2)
    states = [np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b) @ fiducial
              for a in range(d) for b in range(d)]
    return [[np.outer(s, s.conj()) / np.sqrt(d)] for s in states]


def random_hqmm_kraus(d, n_symbols, kraus_per_symbol, rng):
    """Random instrument: a Haar-like isometry V (n_symbols*w*d x d) cut into d x d blocks."""
    rows = n_symbols * kraus_per_symbol * d
    G = rng.normal(size=(rows, d)) + 1j * rng.normal(size=(rows, d))
    Q, _ = np.linalg.qr(G)
    blocks = [Q[b * d:(b + 1) * d, :] for b in range(n_symbols * kraus_per_symbol)]
    return [blocks[x * kraus_per_symbol:(x + 1) * kraus_per_symbol] for x in range(n_symbols)]

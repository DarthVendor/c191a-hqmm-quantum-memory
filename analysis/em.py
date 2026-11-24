"""Baum-Welch (EM) for transition-emitting HMMs, T_x[j, i] = P(x, next=j | now=i),
the same form an HQMM takes once its coherence is removed. Several random
restarts run together as a batch so the Python loop over time runs once."""
import numpy as np


def _random_T(R, n, n_symbols, rng):
    raw = rng.gamma(1.0, size=(R, n_symbols, n, n))
    return raw / raw.sum(axis=(1, 2), keepdims=True)


def fit_mealy_hmm(seq, n, n_symbols, rng, restarts=8, max_iter=150, tol=1e-7):
    """Returns (best_T as list over symbols, per-restart final train loss in bits/symbol)."""
    T = _random_T(restarts, n, n_symbols, rng)
    L = len(seq)
    prev_loss = np.full(restarts, np.inf)
    for _ in range(max_iter):
        alpha = np.empty((L + 1, restarts, n))
        c = np.empty((L, restarts))
        alpha[0] = 1.0 / n
        for t, x in enumerate(seq):
            a = np.einsum("rji,ri->rj", T[:, x], alpha[t])
            c[t] = a.sum(axis=1)
            alpha[t + 1] = a / c[t][:, None]
        counts = np.zeros_like(T)
        beta = np.ones((restarts, n))
        for t in range(L - 1, -1, -1):
            x = seq[t]
            counts[:, x] += (np.einsum("rj,ri->rji", beta, alpha[t])
                             * T[:, x] / c[t][:, None, None])
            beta = np.einsum("rj,rji->ri", beta, T[:, x]) / c[t][:, None]
        T = counts / counts.sum(axis=(1, 2), keepdims=True)
        loss = -np.log2(c).mean(axis=0)
        if np.all(np.abs(prev_loss - loss) < tol):
            break
        prev_loss = loss
    best = int(np.argmin(loss))
    return [T[best, x] for x in range(n_symbols)], loss

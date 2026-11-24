"""Preliminary experiments for the C191A interim report.

E1  Classical embedding: an HMM rewritten as Kraus operators reproduces the
    HMM's sequence probabilities exactly; a variant whose updated state carries
    coherence still gives the same distribution.
E2  Hankel rank (a lower bound on the number of classical hidden states) for
    qubit HQMMs, their dephased versions, and random HQMMs.
E3  Baum-Welch fits of n-state classical HMMs to data from the qubit models.
E4  Cost of removing coherence: dephased models in every measurement basis.
Run:  .venv/bin/python -I analysis/prelim_experiments.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hqmm as H  # noqa: E402
from em import fit_mealy_hmm  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
SEED = 191
results = {"seed": SEED}


def e1_embedding(rng):
    T = H.random_hmm(3, 2, rng)
    hmm = H.hmm_model(T)
    out = {}
    for label, kraus in [("diagonal_embedding", H.embed_hmm(T)),
                         ("coherent_output_embedding", H.coherent_output_embed(T))]:
        q = H.hqmm_model(kraus, rho0=np.diag(hmm.w0).astype(complex))
        words = hmm.words(6)
        diff = max(abs(q.prob(w) - hmm.prob(w)) for w in words)
        # coherence left in the memory after one step from the stationary state
        rho = sum(K @ np.diag(hmm.w0) @ K.conj().T for g in kraus for K in g)
        out[label] = {"completeness_error": H.completeness_error(kraus),
                      "kraus_ops_per_symbol": len(kraus[0]),
                      "words_checked": len(words),
                      "max_abs_prob_diff": diff,
                      "max_offdiag_after_step": float(np.max(np.abs(rho - np.diag(np.diag(rho)))))}
    return out


def rank_summary(model, max_len):
    r, s = H.hankel_rank(model.hankel(max_len))
    return {"rank": r, "singular_values": [float(v) for v in s[:6]]}


def e2_hankel(rng):
    out = {}
    for name, vecs in [("tetrahedral_sic", H.TETRAHEDRON), ("trine", H.TRINE)]:
        kraus = H.luders_qubit_kraus(vecs)
        q = H.hqmm_model(kraus)
        deph = H.hmm_model(H.dephase(kraus))
        out[name] = {"memory_dim": 2, "n_symbols": len(vecs),
                     "completeness_error": H.completeness_error(kraus),
                     "quantum": rank_summary(q, 3), "dephased": rank_summary(deph, 3)}
    kraus = H.hesse_sic_kraus()
    out["qutrit_sic"] = {"memory_dim": 3, "n_symbols": 9,
                         "completeness_error": H.completeness_error(kraus),
                         "quantum": rank_summary(H.hqmm_model(kraus), 2),
                         "dephased": rank_summary(H.hmm_model(H.dephase(kraus)), 2)}
    for d, nx, w in [(2, 3, 1), (3, 3, 2)]:
        ranks_q, ranks_c = [], []
        for _ in range(20):
            kraus = H.random_hqmm_kraus(d, nx, w, rng)
            ranks_q.append(rank_summary(H.hqmm_model(kraus), 3)["rank"])
            ranks_c.append(rank_summary(H.hmm_model(H.dephase(kraus)), 3)["rank"])
        out[f"random_d{d}_x{nx}_w{w}"] = {"quantum_ranks": sorted(set(ranks_q)),
                                          "dephased_ranks": sorted(set(ranks_c)), "trials": 20}
    return out


def bloch_basis(theta, phi):
    n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    up = H._pure_state_from_bloch(n)
    down = np.array([-up[1].conj(), up[0].conj()])
    return np.column_stack([up, down])


def e3_e4(rng, train_len=10000, test_len=10000, max_n=5):
    out = {}
    for name, vecs in [("tetrahedral_sic", H.TETRAHEDRON), ("trine", H.TRINE)]:
        kraus = H.luders_qubit_kraus(vecs)
        q = H.hqmm_model(kraus)
        train, test = q.sample(train_len, rng), q.sample(test_len, rng)
        rec = {"true_model_test_bits": q.log_loss_bits(test), "hmm_fits": {}}
        for n in range(1, max_n + 1):
            t0 = time.time()
            T, restart_losses = fit_mealy_hmm(train, n, len(vecs), rng)
            m = H.hmm_model(T)
            rec["hmm_fits"][n] = {"test_bits": m.log_loss_bits(test),
                                  "train_bits": m.log_loss_bits(train),
                                  "restart_train_bits": sorted(float(v) for v in restart_losses),
                                  "seconds": round(time.time() - t0, 1)}
            print(f"{name} n={n}: test {rec['hmm_fits'][n]['test_bits']:.4f}", flush=True)
        scan = []
        for theta in np.linspace(0, np.pi, 13):
            for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                m = H.hmm_model(H.dephase(kraus, bloch_basis(theta, phi)))
                scan.append(m.log_loss_bits(test))
        rec["dephased_test_bits"] = {"computational_basis": H.hmm_model(H.dephase(kraus)).log_loss_bits(test),
                                     "best_over_bases": min(scan), "worst_over_bases": max(scan),
                                     "bases_scanned": len(scan)}
        out[name] = rec
    return out


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    results["E1_embedding"] = e1_embedding(rng)
    results["E2_hankel"] = e2_hankel(rng)
    print(json.dumps({k: results[k] for k in ("E1_embedding", "E2_hankel")}, indent=1))
    results["E3_E4_fits"] = e3_e4(rng)
    (OUT / "prelim_results.json").write_text(json.dumps(results, indent=1))
    print("wrote", OUT / "prelim_results.json")

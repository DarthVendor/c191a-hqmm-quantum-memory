# Interim report (due 2026-11-24): analysis

Inputs: the graded expanded proposal (Gradescope export `447091635.pdf`, 1/1, with feedback) and `C191A_Fall26_Final_Project_Rubric.pdf`.

## 1. What the interim report must contain, and where the proposal stands

| Rubric requirement (interim) | In the proposal? | Gap to close today |
|---|---|---|
| 3–5 pages, an *aspirational* version of the final report | 2 pages, written as a plan | Rewrite in the voice of the final report: sections that already contain explanations and results, with clearly marked "to do" parts |
| Intended structure of the final report | Only as the 4 work phases | Add a section-by-section outline of the final report (see §4) |
| Context | 3 references, all on HQMM learning or QML caveats | Add the quantum stochastic-modelling line: Monras et al. 2010 (HQMMs), Gu et al. 2012 (quantum models need less memory), Ghafari et al. 2019 (qubit experiment), Elliott et al. 2020 (unbounded advantage). The proposal's "published low-dimensional example" and "classical-memory lower bound" come from this literature, but none of it is cited |
| Motivation and expected impact | One sentence ("improved prediction or quantum speedup is not assumed") | Say why memory for prediction matters, and what a reader will be able to judge after reading the report |
| Methods | Yes (4 phases) | Make them concrete: the update rules as equations, the dephasing basis, the lower-bound tool, the fitting procedure |
| Preliminary analysis or results | None | Results in §2 below |
| AI use in detail at the end (project **and** writing; may run onto page 6) | None | Draft in §5 |
| Grader feedback: approachable for a general QC audience, no assumed Markov-model background, walk through specific examples | Not yet addressed | Explain an HMM from scratch with a weather-style toy example before any quantum notation; carry one worked qubit example (§2.3) through the report |

Grading emphasis: strong understanding, technical accuracy, clarity, organisation, thoughtful analysis, independent thought and critical engagement. "Understanding and explaining a simpler topic well is preferable to discussing an ambitious topic vaguely."

### Scope risk

Phase 3 of the proposal includes *learning* HQMMs from data (the Stiefel-manifold method of Adhikary et al. 2020). That is the most expensive and least certain part. Recommendation: make exact constructions, the rank lower bound, dephasing and classical EM fits the core of the report, and treat HQMM learning as a stretch goal stated as such in the interim report.

## 2. Preliminary results

All numbers come from `analysis/results/prelim_results.json`, produced by `.venv/bin/python -I analysis/prelim_experiments.py` (Python 3.12, NumPy, seed 191) on 2026-11-24. Code: `analysis/hqmm.py` (models), `analysis/em.py` (Baum-Welch), `analysis/prelim_experiments.py` (experiments).

**E1. Classical HMM inside the quantum formalism.** A random 3-state, 2-symbol HMM, written as Kraus operators K_{x,i,j} = √T_x[j,i] |j⟩⟨i| (9 per symbol): completeness error 1e-16, and sequence probabilities equal to the HMM's to within 6e-17 on all 127 words of length ≤ 6. A second embedding with only 3 Kraus operators per symbol leaves coherence in the memory (largest off-diagonal entry 0.36 after one step) but gives the *same* probabilities (max difference 6e-17). See §3.1 for why.

**E2. Hankel rank, a certified lower bound on the number of classical hidden states.**

| Model | Memory dimension | Symbols | Hankel rank (quantum) | Hankel rank after dephasing |
|---|---|---|---|---|
| Tetrahedral (SIC) qubit model | 2 | 4 | **4** | 2 |
| Trine qubit model | 2 | 3 | **3** | 2 |
| Qutrit SIC model | 3 | 9 | **9** | 3 |
| 20 random qubit HQMMs (3 symbols) | 2 | 3 | 4 in 20/20 | 2 in 20/20 |
| 20 random qutrit HQMMs (3 symbols, 2 Kraus each) | 3 | 3 | 9 in 20/20 | 3 in 20/20 |

Hankel matrices use all words up to length 3 (length 2 for the qutrit SIC). The rank cut-off is unambiguous: for the tetrahedral model the singular values are 1.36, 0.120, 0.120, 0.120, then 7e-17. A generic d-dimensional HQMM reaches the ceiling d²; removing coherence drops it to d.

**E3. Fitting classical HMMs to data from the qubit models.** 10,000 training and 10,000 held-out symbols sampled from each quantum model; transition-emitting HMMs with n states fitted by Baum-Welch with 8 random restarts each. Held-out predictive log loss in bits per symbol (lower is better):

| n (classical states) | Tetrahedral | Trine |
|---|---|---|
| 1 | 2.000 | 1.585 |
| 2 | 1.921 | 1.403 |
| 3 | 1.859 | **1.243** |
| 4 | **1.808** | 1.244 |
| 5 | 1.809 | 1.243 |
| True quantum model (1 qubit) | 1.804 | 1.242 |
| Exact entropy rate (analytic) | 1.792 | 1.252 |

Classical models reach the quantum model's performance exactly at the size the rank bound predicts (4 and 3 states) and not before: with 3 states the tetrahedral data cost 0.054 extra bits per symbol, with 2 states 0.116. The 8 restarts agree closely (n = 4 tetrahedral: training loss 1.7935 to 1.7944), so the shortfall below the bound is representational, not an optimisation failure, which is the distinction the proposal's methodology asked for. Caveat: held-out estimates carry about ±0.01 bits of sampling noise (true model 1.804 against an exact rate of 1.792); differences between rows share one test set, which makes them more precise than that.

**E4. The cost of removing coherence.** Dephasing the qubit model in a basis gives a 2-state classical HMM. Over 156 bases spread across the Bloch sphere, the best dephased tetrahedral model scores 1.921 bits (computational basis 1.924, worst 1.962), about the same as the best fitted 2-state HMM (1.921). For the trine, the best basis gives 1.400 and the worst gives 1.585, which is no memory at all (a basis perpendicular to the trine plane erases everything the model remembers). So "the dephased version" is basis-dependent and the report must say which basis it uses; the fair comparison is the best basis.

## 3. Conceptual findings worth putting in the report

**3.1 Coherence in the memory is not enough; the dynamics must read it.** The proposal plans to embed an HMM with "quantum operations that preserve diagonal density matrices". That is sufficient but not necessary. Claim (short proof, worth including): if every instrument element ignores input coherence, E_x ∘ Δ = E_x with Δ the dephasing map, then

P(x_1…x_n) = tr(E_{x_n} ⋯ E_{x_1}(ρ_0)) = tr(ΔE_{x_n}Δ ⋯ ΔE_{x_1}Δ(ρ_0)),

because Δ can be inserted before each E_x and tr ∘ Δ = tr. So the process is the d-state HMM T_x = ΔE_xΔ. Experiment E1 shows a concrete case: the embedding K_{x,i} = |v_{x,i}⟩⟨i| leaves off-diagonal entries up to 0.36 in the memory, yet its sequence probabilities match the HMM to 6e-17 on all 127 words of length ≤ 6. This is a clean, original-feeling point for the "independent analysis" criterion.

**3.2 The lower-bound tool: Hankel rank.** Write P(u v) for a past u and a future v into a matrix H. Any HMM with n hidden states gives rank H ≤ n, so rank H is a certified lower bound on the classical memory. Any HQMM with a d-dimensional memory gives rank H ≤ d² (its state, a d×d density matrix, lives in a d²-dimensional real space). Two consequences for research question 3:
- the rank bound can certify at most a quadratic gap (d versus d²). Larger gaps, such as the unbounded one-qubit advantage of Elliott et al. 2020, need a different certificate; the natural classical quantity is the nonnegative rank of H, which can exceed the rank;
- rank ≤ n also holds for signed classical linear models (observable operator models, OOMs). A 4-dimensional real OOM reproduces the qubit process exactly on a classical computer. The quantum advantage is relative to *positive* classical models (HMMs), which matches Adhikary et al.'s framing of HQMMs as an OOM subclass without negative probabilities.

**3.3 A worked example a general reader can follow** (answers the grader's request). One qubit of memory, four symbols. The four readouts are the tetrahedral states |ψ_k⟩ (Bloch vectors at the corners of a tetrahedron), K_k = (1/√2)|ψ_k⟩⟨ψ_k|. After reading symbol k the memory is |ψ_k⟩, so the next symbol repeats k with probability 1/2 and is each other symbol with probability 1/6 (from |⟨ψ_j|ψ_k⟩|² = 1/3). Classically the smallest memory is "remember the last symbol": four states, two bits. Quantum mechanics fits those four memory states into one qubit because they need not be perfectly distinguishable, only predictive. The trine version (three symbols, repeat probability 2/3) gives three classical states versus one qubit, the same kind of separation Ghafari et al. 2019 demonstrated with photons (we have not checked that their process is this one). For a d-dimensional SIC the transition matrix is (1/(d+1))·I + (1/(d(d+1)))·J, full rank d², so the construction meets the d² ceiling in every dimension where a SIC exists. Checked numerically for d = 2 and for the qutrit (Hesse) SIC, d = 3: Hankel rank 9 from a 3-dimensional memory, dephased rank 3, transition probabilities 1/3 (repeat) and 1/12 (each other symbol) as the formula predicts.

**3.4 Fine print: what "more compact" does and does not mean** (Aaronson 2015 applied to this topic). For the tetrahedral example:

| Measure | Qubit HQMM | Smallest classical HMM | Comment |
|---|---|---|---|
| Memory dimension | 2 | 4 (certified by rank) | the real separation |
| Memory entropy (stationary) | 1 bit (ρ = I/2) | 2 bits (uniform over 4 states) | 1 vs log₂3 ≈ 1.585 bits for the trine |
| Numbers a classical computer stores per step | 3 (Bloch vector) | 3 (belief on 4 states) | **no** classical simulation saving |
| Free parameters of the model family | 28 (8×2 isometry, before gauge) | 24 state-emitting, 60 transition-emitting | depends on the family; not a robust compactness measure |
| Prediction quality | exact | exact | no predictive advantage either |

So the defensible claim is about the size of a physical memory that has to be carried between steps, not about speed, prediction accuracy or parameter count.

## 4. Recommended structure for the interim report (3–5 pages + AI page)

Write it as the final report with some sections still thin. Pages are approximate.

1. **Introduction and motivation** (½ p). To predict a process you must remember something about its past. How small can that memory be, and does quantum mechanics change the answer? Impact: a precise account of what a "quantum memory advantage" claims, kept apart from claims of quantum speedup.
2. **Background for a general QC reader** (1 p). An HMM from scratch with a two-state toy example: the belief vector and its update as a matrix-vector product. Then the same thing in quantum language: memory state ρ, instrument {E_x}, update ρ → E_x(ρ)/tr E_x(ρ). End with the shared formula P(x₁…xₙ) = σ·τ_{xₙ}⋯τ_{x₁}·w₀.
3. **Classical models inside the quantum formalism** (¾ p). The Kraus embedding, the claim of §3.1, E1 numbers.
4. **Worked example: one qubit, four symbols** (¾ p). §3.3, with the Hankel-rank argument explained in words first; a figure of Hankel singular values (quantum versus dephased).
5. **Preliminary experiments** (½–¾ p). EM fits table (§2), cost of dephasing, random-model ranks.
6. **Resource accounting and limitations** (½ p). §3.4.
7. **Plan to the final report and poster** (¼–½ p). Reconstruct a published example; circuit realisation (Stinespring isometry V = Σ_x |x⟩ ⊗ K_x, i.e. one memory qubit plus two ancilla qubits measured each step); d = 3 SIC check; nonnegative-rank discussion; HQMM learning as a stretch goal.
8. **References**, then **9. AI use** (may run onto page 6).

## 5. AI-use record (for the disclosure section)

Facts the group can draw on; the group should add its own uses and say what it checked itself.

- Tool: Claude (Anthropic), model Claude Opus 5.5, run in Claude Code on 2026-11-24.
- Read the rubric and the graded proposal and wrote the gap analysis in §1.
- Checked the bibliographic details and abstract-level claims of six references by web search (Srinivasan et al. 2018; Adhikary et al. 2020; Monras et al. 2010; Gu et al. 2012; Ghafari et al. 2019; Elliott et al. 2020). It did not read the full papers.
- Wrote the analysis code (`analysis/hqmm.py`, `analysis/em.py`, `analysis/prelim_experiments.py`) and ran it; all numbers in §2 come from that run.
- Proposed the tetrahedral/trine worked examples, the "ignores input coherence" argument (§3.1), the Hankel-rank framing and the resource table (§3.4).
- Not yet done by a human (to do before submission): re-derive §3.1 and the SIC transition formula in §3.3 by hand, and read the code.

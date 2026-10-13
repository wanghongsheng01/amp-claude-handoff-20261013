# Independent prior-art challenge: finite-state guided AMP H5 sampling

Date: 2026-10-12. Role: non-author novelty auditor. Scope: one proposed hypothesis only—finite-horizon autoregressive peptide generation, length/additive charge/composition constraints, finite-state surrogate completion probabilities, and true-AR-corrected SMC. This audit read public papers and author code; it did not read peptide datasets, model checkpoints, manuscripts, H0–H4, or other projects; it did not install dependencies or run training/sampling. No empirical performance was reproduced.

**Decision: REJECT the current generic algorithmic novelty claim. HOLD any stronger approximation, efficiency, or scientific-contribution claim until independently reviewable evidence exists.** These are claim-level decisions, not predictions of a venue decision. The proposal is a technically legitimate composition of known finite-state guidance and corrected SMC, but the composition alone does not establish a new inference principle. An AMP application and smaller alphabet do not change that assessment.

## Closest-work matrix

All entries below are primary sources. “Overlap” is the auditor’s mapping of the proposed mechanism to the cited mechanism, not a claim of a reproduction.

| Work and source anchor | What the source supplies | Overlap / limiting distinction |
|---|---|---|
| [LSSAMP, KDD 2023](https://arxiv.org/html/2212.09450v2), §3.3, Algorithm 1, Eq. 10; [author code](https://github.com/dqwang122/LSSAMP) | Multiscale VQ-VAE sequence/secondary-structure representation; autoregressive priors over codebook indices; latent sampling and heuristic structural control. | Relevant AMP generative baseline/context. It does not supply the proposed corrected finite-state sampler. Being different from LSSAMP is insufficient novelty because the inference literature is closer. Its biological results cannot be transferred to H5. |
| [GeLaTo, ICML 2023](https://starai.cs.ucla.edu/papers/ZhangICML23.pdf), §3 | A tractable surrogate, instantiated as an HMM, supplies constraint-conditioned guidance to an autoregressive LM. | The surrogate-model idea and tractable future-constraint reasoning predate H5. Restricting to a first-order observable Markov chain is a simplification, not an expansion of expressive power. |
| [Ctrl-G, NeurIPS 2024](https://arxiv.org/html/2406.13892v2), Eq. 1, §3.2 Eqs. 3–4, Algorithm 1, §3.3; [author code](https://github.com/joshuacnf/Ctrl-G) | HMM×DFA dynamic programming calculates a surrogate’s future acceptance probability. The next-token law multiplies the true LM probability by that quantity and renormalizes. Setting positive guide values to one recovers logical masking. | This is the proposed guide and proposal, including combinations of finite-state constraints and length. Exact surrogate DP does not imply exact LM conditioning. Importance correction is the proposed addition, already covered by the next two SMC rows. |
| [Zhao et al., ICML 2024](https://proceedings.mlr.press/v235/zhao24c.html); [readable full text](https://arxiv.org/html/2404.17546v1), §2.2 Eq. 6, §3.1 Eqs. 9–12, §3.2 Eqs. 14–16, Algorithm 1; [author code](https://github.com/Silent-Zebra/twisted-smc-lm) | Arbitrary intermediate twists, a twist-induced proposal, incremental likelihood-ratio weights, and a final exact potential. The optimal twist is expected terminal potential. | Replacing a learned twist by a cheap finite-state completion probability is an instance of the framework. Its sampling correction does not require the twist to be accurate. Accuracy controls efficiency, while support and final target control validity. |
| [Loula et al., 2025](https://arxiv.org/html/2504.13139v1), §2 Eqs. 2–4 and subsequent importance-sampling/SMC paragraphs; [author code](https://github.com/probcomp/genlm-control) | Complete sequences explicitly include EOS. Programmable proposal/potential functions encode domain knowledge; weight corrections compensate local normalization; particles can be resampled. | Direct overlap with “domain-specific cheap constraints + true LM correction.” EOS-aware implementation is necessary correctness work and already explicit here. The paper’s particular applications differ; its framework does not require language-only semantics. |
| [Anaya Gonzalez et al., 2025](https://arxiv.org/html/2506.05754v1), §2.1, §3, Algorithm 1, Appendix E | Valid constrained proposals, suffix regeneration from a selected prefix, and Metropolis–Hastings correction toward the LM conditional target. | Different particle/chain mechanics, but directly defeats a claim that valid outputs plus distributional correction is new. It is a strong comparison for distributional fidelity. Finite-time chain output is not automatically an exact target draw. |
| [Adaptive Multilevel Twisted SMC, 2026 preprint](https://arxiv.org/html/2608.21736v1), dated 22 August 2026, §3, Algorithm 1 | Progressively harder rare events, adaptive thresholds, and importance-weighted twist learning. | Later adjacent prior art that must not be omitted from a current novelty search. It does not establish the same finite-state DP guide. It does block broad claims that rare-event twist adaptation itself is new. This audit did not validate its empirical results or all proofs. |

The correction also sits in established particle-filter history: [Pitt–Shephard’s author publication record](https://shephard.scholars.harvard.edu/publications/filtering-simulation-auxiliary-particle-filter) identifies auxiliary particle filtering (1999), and [Whiteley–Lee](https://arxiv.org/pdf/1210.0220) studies twisted filters (2014). The decisive rejection here relies on the explicit modern equations above, not a loose appeal to those older names.

## Exact mapping of the single hypothesis

The following is an independent algebraic audit, not a new candidate or a claim of a new theorem.

Let a complete path be `z = (x_1,…,x_n,EOS)` with an unambiguous maximum number of generation decisions. It may be padded after EOS with deterministic absorbing transitions. Let `A` include the desired terminal event and all constraints. The target is

`π(z) = p_AR(z) 1_A(z) / Z_A`.

If the baseline is first conditioned on termination within a horizon, its normalization cancels when conditioning further on an event that already includes that termination. That does **not** license changing the final decision from a genuine EOS probability to forced EOS probability one. A cap-censored path and an EOS-terminated path are different outcomes and must be specified separately.

For a live prefix `u`, define `h_p(u) = P_AR(A | u)`. The exact conditional next-token law is `p_AR(a|u) h_p(ua) / h_p(u)`. This follows directly from conditional probability. A DFA can represent bounded counters and length; an extra surrogate Markov state permits exact dynamic programming for `h_r(u) = P_r(A|u)`. The latter is exact under **r**, not under the LSTM.

With a terminally correct guide `ψ(z)=1_A(z)`, take

`B(u) = Σ_a p_AR(a|u) ψ(ua)`,

`q(a|u) = p_AR(a|u) ψ(ua) / B(u)`.

Setting `ψ=h_r` for live prefixes is precisely the Ctrl-G-shaped proposal. Define intermediate unnormalized densities `γ_t(u)=p_AR(u)ψ(u)` and the final density `γ_T(z)=p_AR(z)1_A(z)`. Then ordinary SMC gives

`G_t(u,a) = γ_t(ua) / [γ_(t-1)(u) q(a|u)] = B(u)/ψ(u)`.

This is Zhao et al.’s Eq. 16 after renaming symbols. The nonconstant weight is not optional merely because the guide was computed exactly: `h_r(u)=Σ_a r(a|u)h_r(ua)`, whereas `B(u)` sums with **p_AR**. Unless the relevant recursions agree, their ratio is not one. Use the standard `γ_0=1` convention; if instead an initial mass `h_r(empty)` is used, include that known mass in the normalizing-constant estimator.

On a particle set with normalized ancestor weights `W_i`, selecting ancestors proportional to `W_i B(u_i)/ψ(u_i)` and then drawing from `q` is the corresponding auxiliary proposal organization. It moves the same lookahead factor between resampling and propagation; it does not create a new target or correction principle. Merely renaming the guide a “completion certificate” does not change these equations.

Proper correction can make an SMC normalizing-constant estimator unbiased under its support/resampling assumptions. A finite particle collection, a self-normalized expectation, and a final resampled particle are generally approximate. They are not independent exact conditional samples. Retaining duplicate particles preserves the empirical weighted measure; deduplicating and reporting the survivors as equally weighted changes it.

## What finite-state compression does and does not buy

For a finite horizon, length and fixed additive residue statistics can be tracked with counter states. Several counters may produce a large product state space; polynomial language relative to an explicitly expanded automaton does not guarantee cheap construction. Rational charge/composition thresholds need an exact declared representation. Rounding or dropping a counter silently changes `A`.

Two prefixes can have identical counters and final residue but different LSTM hidden/cell states and different continuation probabilities. Merging those prefixes is exact for an explicitly finite-order surrogate only, unless a sufficient-state equivalence is proved for the true model. A small last-k surrogate, count-state cache, or HMM is not automatically an exact representation of an arbitrary LSTM. The full bounded prefix tree is finite but generally exponential. Carrying each particle’s actual recurrent state preserves the AR likelihood; it does not make true-LSTM completion sums tractable.

A fully supported surrogate with the same EOS semantics can preserve every target-feasible continuation. An observed-count Markov model without smoothing may instead assign zero guide mass to an AR-positive valid path. Once excluded, importance weights cannot recover it. Underflow, top-k/top-p truncation, premature EOS masks, and inconsistent acceptance states can cause the same loss. A fallback that changes the proposal must record and use the actual proposal probability.

An analytic counterexample separates validity from target fidelity: choose first symbols A/B with equal probability, second-symbol probability of A equal to 0.9 after A and 0.1 after B, and constrain the second symbol to A. Both first symbols are feasible. Feasibility masking keeps their first-step probabilities at 0.5/0.5, while the true conditional probabilities are 0.9/0.1. This is an algebraic toy example, not a model run or a peptide-data observation.

## Approximation bounds: current status HOLD

An assumption `exp(-δ) ≤ p_AR(a|u)/r(a|s(u)) ≤ exp(δ)` on **every reachable continuation** gives completion-probability ratios within `exp(±mδ)` for at most `m` remaining decisions, including EOS. This follows by multiplying path ratios and summing nonnegative path masses. It is a routine consequence of a strong uniform assumption; it is not evidence that the assumption holds for this LSTM. Checking train279/dev41 prefixes alone would not certify the unseen state space. Crude worst-case softmax bounds can be mathematically valid yet useless at the intended horizon.

Bounds for the approximate proposal, its normalizing-constant estimator, and the finite-particle output law are distinct. A small average next-token loss is not a uniform bound on a rare-event completion probability. A distributional guarantee needs explicit assumptions, constants, failure probability, target, and compute accounting. No new non-vacuous bound specific to this hypothesis has been supplied or identified in this audit.

## Kill criteria before natural runs

1. **Novelty kill—already met:** if the full mechanism maps to `h_r` plus the displayed `q` and `G`, reject “new constrained sampling algorithm.” A new application, alphabet size, or counter implementation alone does not reverse this.
2. **Exactness kill:** reject any claim that finite-state DP is exact for the LSTM, or that ordinary finite-N SMC produces unbiased/i.i.d. target draws.
3. **Target/support kill:** stop if EOS/censoring semantics differ across methods, a true feasible path receives proposal mass zero, a rounded constraint differs from the declared one, or a fallback/failed particle is omitted from accounting.
4. **Bound kill:** do not promote a theorem whose required uniform error is checked only on observed prefixes, or whose numeric guarantee is vacuous at the declared horizon. Empirical calibration must be called empirical.
5. **Comparison kill:** a gain only over rejection or uncorrected masks cannot isolate the finite-state guide. At minimum a scientific comparison must expose guide-only decoding, the same guide with full-sequence importance correction, and the same guide in standard SMC; otherwise the gain may be entirely generic correction/resampling. Constrained MCMC is a material adjacent baseline, subject to feasibility.
6. **Budget kill:** count surrogate fitting, DP construction/cache memory, true-AR evaluations, wasted/failed trajectories, resampling, and wall time. Report fidelity together with ESS/ancestor collapse and unique valid output yield; a high hit rate alone does not establish target quality or compute efficiency.
7. **Evidence kill:** no biological efficacy, safety, or submission-readiness conclusion follows from satisfying length/charge/composition or matching a sequence-model distribution. Keep all failed and negative results.

A claim can leave HOLD only after a concrete additional result is stated and audited against these works—for example an independently checked complexity or error result whose assumptions are both applicable and informative, or a clearly scoped empirical systems result with decisive matched comparisons. This audit does not assert that such a result exists and does not authorize further runs.

## Source/code inspection boundary

Repository feasibility, pinned commits, selected Git blob hashes, licenses and dependency observations are reported in [OFFICIAL_CODE_AUDIT_v1.md](code_audit/OFFICIAL_CODE_AUDIT_v1.md) and [OFFICIAL_CODE_AUDIT_v1.json](code_audit/OFFICIAL_CODE_AUDIT_v1.json). All four repository identifications are resolved. LSSAMP is pinned at `f8f9c7bce63693383818a83cf08639981498ef58`; twisted-smc-lm at `28d921a7fedd1584a1bbe97aa498a63013585cb5`; CASA at `54a11a6e934c4a9a4248c8ceb092d9e352a3c748`; Ctrl-G at `379463249c368739f81c46c1e71b8673bfcedf7c`. These are current snapshots, not verified publication-version reproductions.

A public paper’s license does not license its repository or checkpoints. No reuse license was established for LSSAMP or twisted-smc-lm, so source transplantation is not cleared. CASA and Ctrl-G have MIT licenses in the inspected snapshots. Reading public code is not evidence that it runs in the permitted CPU/single-thread environment. Status is **STATIC_SOURCE_VERIFIED / RUNNABLE_UNVERIFIED**. No repository was installed or executed in this audit. A faithful paper-based reimplementation with attribution and parity checks remains an adaptation, not a reproduced official baseline by declaration.

Read failures preserved in this record: the PMLR raw PDF endpoint and its alternative PDF path returned tool errors; arXiv `2404.17546v2` returned 404. The explicit arXiv v1 full text supplied all cited twisted-SMC equations. Ctrl-G v3 HTML returned an error; the explicit v2 full text was read. These are retrieval limitations, not model/experiment failures.

The separate draft-scope/cost review is in [DRAFT_SCOPE_COST_REVIEW_v1.md](DRAFT_SCOPE_COST_REVIEW_v1.md). It reads the authorized analysis draft only; it does not read a manuscript or original sequence data.

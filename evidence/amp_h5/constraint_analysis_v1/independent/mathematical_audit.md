# Independent mathematical challenge: AMP H5 surrogate-DP twisted SMC

Audit date: 2026-10-12 UTC. Role: independent non-author challenge. No original data, model checkpoints, or local implementation files were read by this auditor. No training, sampling, model execution, synthetic experiment, or natural run was performed. Local implementation semantics below were supplied by the root agent and remain an explicit dependency. Public primary papers were read. This is an analytical report, not an empirical validation.

**Verdict: REJECT the proposed core-mechanism novelty claim.** Exact finite-state surrogate completion probabilities used in a neural-model-guided proposal are directly covered by Ctrl-G; correction by twisted SMC is directly covered by existing twisted SMC. The stated probability-error bound is mathematically valid under strong uniform assumptions, but those assumptions have not been established. It gives no certified benefit for the proposed first-order surrogate. A correctly implemented instance could still be a useful application baseline; that is an empirical question, not evidence of a new mechanism.

## 1. Target and stopping rule must be frozen before any execution

Fix terminal condition tau in {Free, Amide}. The root reports that the existing sampler makes at most 48 categorical draws, stops at an early EOS, and treats 48 residues as a capped sequence without an additional EOS factor. Define the actual categorical transition probabilities after all masks and renormalization as p(a|x,tau). Then the target base measure is

    p_H(x|tau) = product_{i=1}^n p(x_i|x_<i,tau) * p(EOS|x,tau),  n<48;
    p_H(x|tau) = product_{i=1}^48 p(x_i|x_<i,tau),              n=48.

The constraint-conditioned target is pi_C(x|tau)=p_H(x|tau) 1_C(x,tau)/Z_tau. Z_tau must be positive. Adding a forty-ninth EOS probability silently changes the target. Restricting constraints to lengths <=47 is permissible only if explicit; it is still conditioning the same p_H process.

The root also reports a PAD output. Whether PAD is masked, the exact categorical renormalization, any temperature, any EOS/length mask, and the timing of hidden-state updates must be frozen. Importance weights must use the distribution actually sampled. Raw neural softmax probabilities cannot substitute for the implemented distribution when masks renormalize it.

A clean derivation embeds the process in T=48 decision slots. After an early EOS, use an artificial absorbing state with transition probability one under both p_H and the surrogate. This absorbing symbol is a bookkeeping device, not the neural model's PAD output. Constraint values remain fixed after termination. A live prefix at slot t has at most 48-t remaining decisions; a capped length-48 path is already terminal.

If Free and Amide results are combined, also freeze the estimand. Averaging separately conditioned terminal distributions with a preset terminal prior differs from conditioning the joint distribution: the latter has terminal weights proportional to prior(tau)*Z_tau. Equal terminal allocation does not silently estimate the joint conditioned law.

## 2. Exact surrogate DP, support, and what exactness does not mean

For the stated first-order surrogate, a sufficient DP state is s=(tau, live/ended, t, lastAA, accumulated integer charge, accumulated composition counts), with a BOS state. Let F(s,a) update this state. For surrogate categorical probabilities r(a|s), define

    u_T(s) = 1_C(s),
    u_t(s) = sum_a r(a|s) u_{t+1}(F(s,a)).

This is an exact completion probability **under r**, with the same cap and termination rules. It is not an exact completion probability under the LSTM p_H. An r fitted later by add-half smoothing can have full support, but smoothing must cover BOS, all terminal-conditioned contexts, all legal residues, and EOS, including unseen contexts. Exposure/counting at the cap must respect censoring.

For any valid path with positive p_H mass, every proposal transition along that path must have positive probability. A sufficient condition is positive r probability on every legal transition that can occur under p_H, together with exact DP arithmetic and identical constraint/stopping semantics. Then a surrogate zero cannot delete a true feasible suffix. With a fully supported effective neural categorical, u_t>0 exactly on combinatorially completable states, and those zeros also match the true completion probability h_t.

Support coverage is necessary; average fit quality is not a replacement. Top-k truncation, unsmoothed contexts, thresholding small u_t to zero, underflow, or an incorrect EOS rule can delete positive target mass and invalidate the correction. Stable log-domain DP and log weights are required for very small completion probabilities. A floor can preserve support if applied only with correct bookkeeping, but changes the proposal; the actual floored proposal and twists must enter the weights. An exact zero for genuine infeasibility is safe.

For fixed constraint dimension, DP costs roughly O(T * number_of_contexts * number_of_constraint_states * vocabulary_size), before pruning. It is not generally polynomial in the number of additive constraints: the joint state grid contains a product of count ranges. Increasing Markov order k also introduces up to 20^k contexts. Claims of cheap exact inference must state the actual joint state size, memory, and setup time.

## 3. Feasibility is not completion probability

Let B_t(x)=1 if at least one legal suffix satisfies C, and h_t(x)=P_p(C|x). With positive transitions, B_t=1 iff h_t>0. It says nothing quantitative about the magnitude of h_t: two feasible prefixes can have completion probabilities arbitrarily close to zero and one.

For an additive integer statistic with per-residue range [v_min,v_max], prefix value V_t, terminal offset b_tau, and m additional residues, a necessary interval condition is

    [V_t+b_tau+m*v_min, V_t+b_tau+m*v_max] intersects [V_min,V_max].

Analogous bounds hold for counts. Test one common m in the allowed remaining-length range against all conditions. Taking a different existential m for each constraint is unsound. Passing these intervals is still only an overapproximation: residue contributions correlate, integer reachable sets may contain holes, and several constraints can conflict. For example, with symbols A contributing (charge,countA)=(1,1) and B contributing (-1,0), a one-symbol suffix with charge 1 and countA 0 passes separate coordinate ranges but is impossible.

An overapproximate interval mask is safe only for pruning prefixes proved impossible; it cannot promise valid final output without an exact terminal check. Exact order-free additive feasibility can be formulated using nonnegative integer residue counts or a joint bounded DP, independently of neural probabilities. Neither exact feasibility nor interval feasibility produces a useful lower probability bound without additional information about neural transitions.

## 4. Correct twisting, scaling, and normalizer estimation

Write x_t for the full live/absorbed prefix, and u_t(x_t) for the surrogate DP value. Define

    gamma_t(x_t) = p_H(x_t) u_t(x_t),
    A_t(x_{t-1}) = sum_a p(a|x_{t-1}) u_t(x_{t-1}a),
    q_t(a|x_{t-1}) = p(a|x_{t-1})u_t(x_{t-1}a)/A_t(x_{t-1}).

For gamma_0=u_0 and gamma_T=p_H*1_C, the incremental importance factor is

    G_t = p(a|x_{t-1})u_t(x_t)/(q_t(a|x_{t-1})u_{t-1}(x_{t-1}))
        = A_t(x_{t-1})/u_{t-1}(x_{t-1}).

Resampling after every nonfinal step gives

    Zhat = u_0 * product_{t=1}^T mean_i G_t^i.

This is unbiased under the usual correctly evaluated proposal, support, fixed intermediate target, and conditionally unbiased offspring-count assumptions. Multinomial or a correctly implemented unbiased alternative qualifies. Deterministic top-particle selection does not. If setting gamma_0=1 instead, use denominator 1 at the first step and omit the external u_0 factor. Mixing those conventions is a common multiplicative error.

When resampling is skipped, accumulated weights must be carried between resampling times; the product of unweighted means of incremental weights at every step is generally wrong. At no-resampling intervals the normalizer update uses the current normalized particle weights. A standard SMC recursion or the product of average accumulated block weights is required. The chosen recursion must be frozen and independently checked.

Pure full-path importance sampling uses Q(x)=product_t q_t and W(x)=1_C(x)p_H(x)/Q(x), with E_Q[W]=Z. This identity is valid without resampling. Merely accumulating p/q while resampling according to unaccounted intermediate twists is not the same estimator: all twist and ancestor-selection effects must be handled by the SMC construction.

If u_t is replaced by c_t*u_t with positive constants depending only on decision time, q_t is unchanged, while G_t is multiplied by c_t/c_{t-1}. The initial factor cancels c_0, and the final normalizer estimates c_T*Z unless divided by c_T. Set c_T=1 for the original target. State-dependent scaling generally changes proposals or intermediate targets; it is not automatically harmless. Per-particle log-sum-exp centering is numerically safe only if the removed scale is restored when computing weights. Apply time scaling consistently to absorbed particles too.

The unbiased result concerns Zhat, and under the same conditions the unnormalized estimator Zhat*sum_i Wbar_i f(x_i). It does not make the self-normalized ratio sum_i Wbar_i f unbiased. A particle selected using Wbar is conditionally drawn from the random empirical measure; its unconditional finite-N law is generally not pi_C. N=1 already exposes the distinction: the selected output follows the proposal even though its importance-weight mean estimates Z. Resampled outputs also share ancestry and are not iid.

The log of Zhat is not unbiased. Jensen gives E[log Zhat]<=log Z where defined; zero estimates yield negative infinity. Dropping extinction, numerical-failure, or timeout runs creates selection bias. Report all such outcomes under frozen failure rules. With exact feasibility, fully positive transitions, and exact arithmetic, this particular proposal should not become extinct; a zero population is then evidence of a support, arithmetic, or implementation problem. Other baseline SMC schemes can legitimately become extinct.

## 5. Valid conditional error bound and its limited scope

Let h_t(x)=P_p(C|x), so h_T=1_C and

    k_t*(a|x)=p(a|x)h_t(xa)/h_{t-1}(x)

is the exact conditioned transition. Suppose for every reachable feasible prefix at each time t,

    exp(-delta_t)h_t(x) <= u_t(x) <= exp(delta_t)h_t(x),

with matching zeros. These are uniform assumptions over all relevant prefixes, not averages over a data split. Then A_t/h_{t-1} lies in [exp(-delta_t),exp(delta_t)], and

    exp(-2*delta_t) <= q_t/k_t* <= exp(2*delta_t).

For the **independent complete-path proposal Q**, define D=2*sum_{t=1}^T delta_t. Multiplying path ratios gives exp(-D)<=Q/pi_C<=exp(D). Consequently

    TV(Q,pi_C) <= tanh(D/2),
    KL(Q||pi_C) <= D,  KL(pi_C||Q) <= D.

The TV inequality follows by maximizing TV among two normalized distributions with likelihood ratio in [exp(-D),exp(D)]; it is a valid, possibly loose bound. Since u_T=h_T, delta_T=0. Delta_0 is not part of the proposal path-ratio product. Under cap censoring, a live prefix after t decisions has at most 48-t suffix decisions, so the root's horizon indexing is correct when the absorbing convention is used.

**This is not a bound on the finite-N resampled SMC output law.** That law depends on N, resampling, and genealogy. A separate particle approximation theorem with its own assumptions and constants would be required. Nor does the bound show that adding particles monotonically improves each realized diagnostic.

A sufficient but strong route to the assumed delta_t is a uniform per-transition certificate

    abs(log(p(a|x)/r(a|state(x)))) <= epsilon

for every relevant prefix and supported suffix transition, including EOS. Each suffix path then has p/r between exp(-(48-t)epsilon) and exp((48-t)epsilon), and summation gives delta_t<=(48-t)epsilon. Therefore the displayed crude D is at most 2256*epsilon for T=48. Even epsilon=0.001 gives TV upper bound tanh(1.128), about 0.81; epsilon=0.01 makes it essentially one. These are illustrative arithmetic consequences, not measured model errors. To make this particular TV certificate <=0.1 requires epsilon approximately <=8.9e-5. It is a sufficient bound, not a necessary condition for good performance.

Uniform transition TV bounds eta_t instead give an additive completion error at most 1-product_t(1-eta_t), hence at most sum_t eta_t. Converting that to relative error divides by a lower bound on completion probability. It therefore commonly becomes useless precisely for rare constraints. Train/dev likelihood, average KL, or empirical calibration on sampled prefixes establishes neither kind of uniform certificate.

## 6. Why symbolic DP states cannot merge neural histories

The pair AKK and KAK has identical length, residue composition, additive charge, and last residue. A neural recurrent state can nevertheless remember their order. Under a final-step composition/charge requirement, their true completion probabilities can be epsilon and 1-epsilon. The shared surrogate DP state assigns the same u to both.

If one shared u is to lie within exp(+-delta) of both completion probabilities, necessarily

    delta >= 0.5*log((1-epsilon)/epsilon).

The lower bound diverges as epsilon approaches zero. Thus low-order symbolic compression alone cannot ensure a nontrivial uniform relative error guarantee. This is an analytical counterexample class; no claim is made that the particular trained H5 model realizes it.

Exact neural merging requires equality of future predictive laws or an exact sufficient neural state; approximate merging requires a proved error bound. Equal rounded hidden vectors are not equality. Even equal current next-token distributions need not imply equal future distributions after subsequent recurrent updates. The LSTM cell and hidden states, model configuration, terminal condition, and all necessary constraint bookkeeping matter. Exact hidden-state equality is rare and does not create the proposed small finite-state DP in general.

It is logically possible to certify bounds without enumerating every prefix, but that requires additional model structure or verified neural bounds. For example, global output-logit intervals yield positive softmax lower/upper bounds; their products often give extremely loose horizon bounds. A separately proved contraction of recurrent dynamics plus a certified output Lipschitz bound can control loss from discarding a long prefix, but arbitrary trained LSTMs need not be contractive. A surrogate fitted from frequencies has no automatic relation to such a reference model. No such certificate has been supplied here. Merely writing a conditional theorem does not instantiate it, and this standard likelihood-ratio argument is not itself a new mechanism.

## 7. Direct prior-art mapping

Ctrl-G computes exact constraint-completion probabilities with an HMM and DFA, then uses neural next-token probability times that surrogate completion value. Its proposal is the candidate proposal. A first-order residue model is a special finite-state surrogate. See §3.2, Eqs. (3–4), Algorithm 1 and the computational theorem in the [published NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/d15c16cf5619a2b1606da5fc88e3f1a9-Paper-Conference.pdf).

Zhao et al. define intermediate targets p(prefix)*psi_t, the same twist-induced proposal, and the same importance correction. Setting psi_t=u_t embeds the candidate directly in that framework. Their optimal twists are true-model future potential expectations. See Definition 3.1, §3.2 and Algorithm 1 in the [authors' full text](https://arxiv.org/html/2404.17546v1); the [PMLR record](https://proceedings.mlr.press/v235/zhao24c.html) verifies the ICML 2024 publication. The distinction between unbiased normalizer estimation and approximate selected samples is explicit in §2.

The 2025 MCMC work constructs valid proposals and corrects them with neural likelihoods using Metropolis–Hastings. It is less mechanically close than the preceding two papers, but blocks a broader claim that valid proposal construction plus true-model correction is new. Its convergence claims require the stated assumptions; they are not universal finite-budget accuracy guarantees. See §3 and Appendix E of the [primary paper](https://arxiv.org/html/2506.05754v1).

This is a targeted novelty challenge, not an exhaustive literature review. It is sufficient to reject the exact mechanism claim as currently stated; it does not reject all possible AMP application value. No biological efficacy, synthesis suitability, or submission-readiness inference follows from constrained sequence statistics.

## 8. Minimum fair diagnostic design, if separately authorized later

No diagnostic was executed. Because the novelty gate fails, natural runs are not needed to establish the current verdict.

First use a fully enumerable, independently specified tiny AR process with EOS/cap handling. This is an implementation oracle, not AMP evidence. Include: exact u=h (constant importance weights), mismatched u, two histories sharing symbolic state but differing future law, impossible C, extremely small but positive support, early EOS, cap termination, both terminal conditions, harmless time scaling, N=1, and resampling always/never/at fixed checkpoints. Enumerating the algorithm's randomness for tiny cases can check normalizer identities independently of Monte Carlo noise. Any target mismatch, missing support, or incorrect normalizer expectation is a hard failure before natural runs.

If natural diagnostics are later justified, hold the trained p_H fixed and use only authorized train279/dev41. Freeze all hashes, splits, terminal semantics, constraint definitions, seeds, smoothing, numerical thresholds, budgets, resampling rules, evaluation functions, and failure handling before any natural sampling. Fit/tune no component on the forbidden test split. Any development choice after observing natural outcomes must be marked as adaptation, never silently replaced in the frozen result set.

The minimum comparison set is:

1. Exact-feasibility masked decoding, both unweighted and with standard SMC correction using psi=B.
2. The same surrogate-guided proposal without correction (the direct Ctrl-G-style comparator).
3. The candidate with full-path IS, and with its fixed SMC correction, to isolate resampling from guidance.
4. Learned-twist/CTL SMC if feasible within the authorized resource ceiling; include its training cost or label its omission as a limitation.
5. Base-model rejection under a fixed attempt/time budget, retaining zero-acceptance cases.
6. A correctly specified constrained MCMC comparator, with initialization, burn-in, repeated states, proposal scoring, and autocorrelation costs included.

For prefix-resampling MCMC, the proposal probability usually sums over every truncation index that can produce the same destination: q(y|x)=sum_i w_i(x)1[x_1:i=y_1:i]q_suffix(y_after_i|x_1:i). The MH ratio must use the complete forward and reverse mixture, or an explicitly proved auxiliary-variable construction. The probability of only the sampled component is not automatically the marginal proposal probability.

Report both equal total single-thread CPU wall-time and equal actual neural computation budgets. Count hidden-state/prefix computations, suffix rescoring, all failed proposals, surrogate/learned-twist fitting, DP construction, memory, and cache reuse. Report setup-plus-inference and a stated amortization curve, not only favorable amortized inference. Run every method serially after inspecting active H5 jobs. Freeze a repeated-run seed count; one run is not evidence of expected improvement.

On the enumerable oracle, report exact distribution error and exact Z with uncertainty for finite-run estimates, plus relative variance of Zhat, constraint validity, and genealogy/ESS. On natural data, true pi_C is generally not enumerable. A separate sufficiently large rejection reference can provide exact accepted target draws, but may itself fail the fixed resource budget. If it does, label target fidelity unresolved. Predetermined marginal event discrepancies are useful diagnostics but do not certify full-distribution TV or KL. Importance ESS, duplicate fraction, and sequence diversity are also diagnostics, not proof of target accuracy. Pooled observed-sequence support is not a reliable exact-KL oracle for a huge sequence space. Do not treat one long correlated MCMC chain as iid ground truth.

## 9. Hard failure and interpretation rules

- **Novelty failure now:** proposal plus correction maps directly to prior methods. No natural run can retroactively establish a new mathematical mechanism for this unchanged candidate.
- **Target failure:** EOS/cap/PAD/terminal mixing differs between generation, likelihood evaluation, r, DP, and reference methods.
- **Support failure:** any positive target path is removed by surrogate zeros, truncation, numerical thresholding, or a false infeasibility certificate.
- **Estimator failure:** wrong initial twist factor, terminal potential, skipped-resampling recursion, proposal probability, MCMC reverse mixture, or deterministic biased resampling.
- **Claim failure:** exact finite-N or iid sampling is claimed; an independent-proposal bound is presented as an SMC output bound; empirical fit is called a uniform certificate; a near-one bound is presented as informative.
- **Resource/fairness failure:** setup/scoring costs omitted, baselines given inferior stopping rules or feasibility information, unsuccessful runs dropped, or post-result threshold changes hidden.
- **Scope failure:** any test37 read, H0–H4/manuscript edit, parallel model execution, upload/push, paid call, production/credential access, or unsupported biological claim.

Retain every failure and negative result. BACKUP_BLOCKED is expected under the no-upload restrictions, not a reason to bypass them. The current defensible status is mathematically recognizable baseline composition, uncertified surrogate error, and no executed experimental evidence.

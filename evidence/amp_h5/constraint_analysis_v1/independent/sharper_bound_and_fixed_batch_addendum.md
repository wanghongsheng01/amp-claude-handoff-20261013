# Audit addendum: sharper conditional bound and fixed-batch accounting

Date: 2026-10-12 UTC. This addendum is authoritative where it sharpens the earlier `mathematical_audit.md`. It contains analytical checks only. No experiment, model/data read, or additional search was performed.

## Sharper bound for the exact surrogate DP

The earlier generic bound D=2*sum_t(delta_t), with delta_t<=(48-t)*epsilon, is valid but unnecessarily loose when u is an **exact** completion DP under r. It must not be used alone to infer that every possible uniform transition certificate is vacuous.

Assume, uniformly over every relevant reachable transition, including EOS and the shared cap convention,

    exp(-epsilon) <= p(a|x)/r(a|state(x)) <= exp(epsilon).

Both processes use T=48 decision slots and deterministic absorption after termination. Let u_t be the exact r completion DP and Z_r=u_0. Since

    u_{t-1}(x) = sum_a r(a|state(x)) u_t(xa),
    G_t(x) = A_t(x)/u_{t-1}(x)
           = E_{r(a|x,C)}[p(a|x)/r(a|state(x))],

we have exp(-epsilon)<=G_t<=exp(epsilon). The complete path likelihood ratio p_H/r_H is between exp(-T*epsilon) and exp(T*epsilon), and summing on C gives the same bounds for Z_p/Z_r. Meanwhile, telescoping the proposal factors gives

    Q(x) = p_H(x)1_C(x)/(Z_r * product_t G_t(x_<t)),
    Q(x)/pi_C(x) = (Z_p/Z_r)/(product_t G_t(x_<t)).

Therefore a sharper valid bound is

    exp(-2*T*epsilon) <= Q/pi_C <= exp(2*T*epsilon),
    TV(Q,pi_C) <= tanh(T*epsilon),
    KL(Q||pi_C), KL(pi_C||Q) <= 2*T*epsilon.

At T=48, the log-ratio bound is 96*epsilon rather than 2256*epsilon, and the stated TV bound is tanh(48*epsilon). These remain bounds for the independent complete-path proposal Q, not for finite-N resampled SMC outputs. They are elementary consequences of exact surrogate recursion and likelihood ratios; they do not introduce a new sampling mechanism. The uniform transition assumption is still unestablished for the H5 model/surrogate. No empirical average fit or prefix sample certifies it. Absence of a certificate is the current conclusion; whether a particular valid certificate would be informative cannot be settled from the loose bound alone.

## Fixed batches, censoring, and total ceilings

The root accepted these counterfactual design corrections; they do not authorize execution or reverse scientific admission NO.

1. Fix the number of complete SMC/IS systems before running. With N=32 and T=48, each system has at most 1,536 decision slots. Deterministic checkpoints after 2 and 10 whole systems have worst-case 3,072 and 15,360 slots. A 16,384-score cap is a guard; unused remainder must not select extra systems based on realized sequence lengths, weights, or timing. Active neural-score accounting may be lower after absorption or caching and must be recorded separately.
2. A completed all-zero SMC system contributes a legitimate zero normalizer estimate. A resource-capped or numerically incomplete system is unresolved, not a fabricated zero. Dropping incomplete systems and averaging only successful/completed ones can condition on outcome-dependent cost and bias summaries. Preserve every scheduled unit and hold the corresponding inferential comparison when it cannot complete.
3. Rejection and MCMC also need predetermined complete attempt/transition counts and a fully specified output/aggregation rule. Random token/time stopping can select completed outputs by length or computational cost. Partial data may be retained descriptively, but cannot silently replace the planned complete-batch estimand.
4. The evaluator's reference batch is fixed at 65,536 raw-p attempts per cell. If the whole batch cannot complete inside its 60-CPU-second guard, the reference is unresolved; partial accepts remain descriptive. Conditional on a fixed completed number of independent attempts, requiring at least 256 accepted samples is compatible with the accepted marks having the conditioned target law. The cost-censored partial-batch case is different.
5. The revised design has 8 arms, 4 cells, and 3 seeds: 96 inference units. At 16,384 active scores per unit, inference has a ceiling of 1,572,864 scores. Reference work adds at most 4*65,536*48=12,582,912 scores. Preparation adds 262,144 AR scores for the candidate/proxy and 262,144 for CTL. The declared full theoretical score ceiling is therefore **14,680,064**, not the inference-only number.
6. At 10 CPU seconds per inference unit, inference allows 960 seconds. Two 120-second preparation allowances plus four 60-second reference allowances bring this to 1,440 seconds. The overall 1,800-second ceiling leaves 360 seconds for shared feasibility construction, evaluation, and other charged overhead. These are administrative ceilings, not measured runtimes. A fixed schedule and complete ledger must preserve any global-cap censoring of later arms.
7. If a mature, validated CTL comparator cannot fit its cap, record HOLD. Do not substitute an untrained, failed, or intentionally weak baseline and call the comparison complete. The same rule applies to unavailable reference evidence or other required parity checks.

With 256 independent reference accepts, the worst-case single binary-feature standard error is approximately 0.031; an unadjusted approximate 95% halfwidth is 0.061. Eight features across four cells and only three run seeds provide limited power. Failure to resolve a worsening is not evidence of equivalence. These feature projections also cannot certify full-distribution TV/KL or detect all within-projection mode loss.

Freeze the SMC weighted-estimate aggregation, any final selection/resampling rule, and what counts as an output for unique-throughput calculations. Counting all weighted proposal particles as equivalent independent target draws would be misleading. Between-system ratio averaging and pooling unnormalized estimators are different finite-sample estimators and must not be selected after observing results.

**Final audit disposition:** the sharper mathematical bound is accepted conditionally; its uniform premise is unproved. Fixed-batch and full-ledger changes repair the design specification but do not demonstrate runtime feasibility. Core mechanism novelty remains rejected, natural experiments remain unadmitted, and no biological or submission claim is supported.

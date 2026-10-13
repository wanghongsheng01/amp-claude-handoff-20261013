# Independent preflight criteria

Scope: a bounded developmental test of terminal-conditioned partial pooling, not a claim of a new PBCT algorithm, antimicrobial efficacy, or submission readiness.

Before natural execution:

1. Input whitelist is train279 and one frozen dev41 check; test37 must not be opened. Any operational sequence-similarity grouping must use train only. Group identity and terminal labels cannot enter as unknown generation-time conditions.
2. All hyperparameters, context structure choices, seeds, metrics, resource limits, stop/failure rules, and input/code/config hashes must be frozen before natural model evaluation. Every tuning fold fits structure, counts, and hyperparameters only on its own training portion. Similarity components are kept together; neither terminal nor duplicated sequence may bypass grouping.
3. Pooled, separate, partial-pooling, and sensible smoothed n-gram controls must have explicit, comparable selection opportunities. The fixed strong AR receives terminal at BOS and is identified as a historical fixed comparator with its actual earlier tuning exposure disclosed; it must not be misrepresented as missing terminal information or newly train-only selected if not true.
4. Frozen conditional distributions score every residue from BOS and score EOS for sequences shorter than 48, but not a nonexistent extra EOS for a sequence capped at 48. The no-extra-EOS boundary rule must be the same for all arms.
5. Primary full-sequence conditional NLL is reported for each terminal, then the two terminal means are equally weighted. Token NLL and length effects can be secondary, not silently substituted. Missing terminal support in train folds or evaluation is a concrete admission problem.
6. Raw free generation uses the same predetermined attempts, terminal allocation, seed policy, and max48 protocol, without filtering or adaptive replenishment. Keep failures/empty samples, length, EOS/cap rate, duplication, nearest training sequence, and the established distribution metrics.
7. Independent synthetic tests cover normalized strictly positive predictive distributions, count totals, boundary handling, unseen contexts/terminals, predictable pooling limits, and grouped-fold count isolation. Imported modules must have no import-time natural-data/model execution.
8. One serial CPU thread and 20-minute natural computation maximum. Inspect active H5 jobs before launch. Budget exhaustion or unfair grouping yields a recorded stop, not a relaxed protocol.
9. Follow the stated negative-result stop rule: no repeated searches after partial pooling fails pooled/separate/strong AR, grouping removes benefit, or copying/EOS deterioration appears.

Post-run review recomputes summaries and invariants from saved outputs without new sampling/training. A PASS concerns bounded execution validity, not scientific confirmation.

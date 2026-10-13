# Addendum-specific independent evidence review

Scope: read-only review of the completed `eb_context_v1` protocol, runner, event ledger, selection records, saved scores, generation counts, and completed independent recomputation. No new fitting, forward scoring, sampling, or original dataset reads were performed for this addendum review. Earlier v1 materials remain unchanged. Draft/PDF claims review is pending until the author provides the addendum.

## Selection and chronology

The runner forms all three folds exclusively from train279, fitting context structures/counts within each fold-training partition. Fresh AR trajectories receive terminal information from BOS onward, use three fixed learning rates, and are scored at 250/500/1000 updates. Selection at runner line 216 minimizes equal-terminal, out-of-fold, full-sequence NLL over nine choices. The selected fresh AR is learning rate 0.0003 and 500 updates; the final fresh model is trained on all train279 at that selected configuration.

The event ledger records final model freeze at 125.013309 s, completion of all 384 raw generation requests at 125.107765 s, and the sole development-file load/evaluation phase at 125.109359 s. The code's development load occurs after the freeze and generation blocks. The `MODELS_FROZEN_BEFORE_DEV.json` receipt separately binds all six model files and the selected configurations while recording `development_opened=false`.

This supports **train-only selection for this fresh AR experiment**, not a fresh validation claim. The task history already reused dev41. Prior knowledge of dev and broader sequential research choices remain limitations even though this run did not use dev to select its nine configurations. The fixed historical AR is a secondary reference with prior development exposure, not a fresh train-only selected comparator.

## Full-sequence scoring and denominators

All residues are predicted from BOS and past residues; sequences below length 48 contribute exactly one first EOS. A sequence capped at 48 contributes 48 residue probabilities and no extra EOS at position 49. The score is a frozen predictive conditional likelihood, not PBCT's integrated marginal score or a score that omits the first D residues.

| Partition | Terminal | Sequences | Residue targets | EOS targets | Total scored targets |
|---|---|---:|---:|---:|---:|
| Grouped train CV | Free | 126 | 2554 | 126 | 2680 |
| Grouped train CV | Amide | 153 | 2314 | 153 | 2467 |
| Development | Free | 18 | 330 | 18 | 348 |
| Development | Amide | 23 | 371 | 23 | 394 |

All natural reference sequences here are below length 48. The cap48 behavior was validated by prior synthetic boundary checks and exercised in raw generated outputs; do not imply that natural dev targets include cap48 examples.

Primary equal-terminal NLL is half the sum of the two terminal mean **sequence** NLLs. It is neither a per-token mean nor the pooled 41-row mean. Secondary terminal token NLL divides total NLL by the corresponding total scored-target denominator above.

## Outcome and retained failures

The predeclared result is **NOT_SUPPORTED**. Partial pooling has dev equal-terminal sequence NLL 49.164309, versus pooled 48.587103, separate 48.917719, fresh AR 48.043021, n-gram 49.473964, and historical AR 57.966916. In grouped CV it fails pooled and fresh AR comparisons. Partial pooling does improve the aggregate grouped-CV score slightly over separate emissions, but its Amide CV NLL worsens; the addendum must not claim failure of every aggregate CV comparison.

There are 32 predetermined requests per terminal per method, six methods, 384 total. No filtering or replenishment occurred. For partial pooling, Amide outputs shorter than five residues are 11/32 (pooled 7/32; fresh AR 1/32); Free cap48 outputs are 2/32 (fresh AR 1/32; historical AR 0/32). These are retained degeneration failures. Empty and short outputs remain in request denominators, while the inherited MMD/JS/nearest summaries explicitly use valid canonical 1–48 sequences as their defined subset. The report must not describe those subset metrics as including empty sequences.

All methods have zero exact train-sequence copies in this small generation sample. This does not establish the absence of memorization. Existing MMD, JS, length, duplication, and nearest-sequence metrics should remain visible instead of selecting only favorable terminal/metric combinations.

## Independent validation and boundaries

The completed earlier audit (`eb_context_v1/independent/RESULT_AUDIT.json`) has status PASS_RECOMPUTED and reproduced 135 saved CV checkpoints, six dev models, selected configurations, all raw generation guards, and the negative stop decision. The legacy metric replay was a consistency check using shared frozen formulas, not an independently implemented metric library. Two auditor-only repairs—AR vocabulary mapping and JSON histogram-key coercion—were preserved. No natural experiment rerun was needed.

Natural runtime was 140.035380 s; all audit attempts total 37.034515 s; combined 177.069893 s, within the 1200 s natural-plus-audit cap. Grouping produced 227 operational edit-similarity components and three folds each containing 42 Free and 51 Amide sequences. These are not verified biological homology groups.

No novelty, biological efficacy, significance, general superiority, or submission-readiness claim follows. Shared-tree separate emissions are not independently learned per-terminal trees. Equal nine-choice tuning opportunities do not equalize architecture capacity or compute. Science NO_GO and the existing Bioinformatics substantive-AI-drafting policy block remain in effect. Private delivery is not journal submission or evidence that the policy conditions have been met.

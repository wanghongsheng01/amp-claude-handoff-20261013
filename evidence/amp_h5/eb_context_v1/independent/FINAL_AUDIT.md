# Independent final audit

Status: **PASS_RECOMPUTED** for engineering validity; scientific outcome **NOT_SUPPORTED**, stop with no retuning.

All 135 saved grouped-CV model checkpoints and all six development models were independently forward-scored without fitting or sampling. The auditor separately implemented EB probabilities from saved count/tree payloads and manually constructed AR input/target tensors using the frozen AR vocabulary. Complete residue plus first-EOS likelihoods (no extra EOS at cap48), terminal-equal aggregation, all 45 grid summaries and selected minima were reproduced. Fold training/holdout identifiers are disjoint; root counts and fit-row digests match their fold inputs.

All 384 raw generation slots were retained. Sample identity, seeds, boundary flags, exact-copy/short/empty/cap counts and every stopping reason agree. The frozen legacy distribution and nearest-neighbor metric code was rerun and matched saved outputs; this is a shared-formula consistency check rather than an independent implementation of those formulas.

Two auditor-only defects were repaired and retained: the first manual AR reconstruction wrongly used the EB alphabet; the second comparison overlooked JSON coercion of numeric histogram keys to strings. Neither required any natural model or result changes. Their scripts and failed receipts remain alongside the final audit.

Compute: natural runner 140.0354 s; final audit 17.7257 s plus prior auditor-only attempts 19.3088 s; total 177.0699 s. Both 200 s aggregate audit and 1,200 s natural-plus-audit budgets pass. CPU execution was serial and one thread.

Grouping: 227 train-only operational edit-similarity components, folds [{'AMD': 51, 'FREE': 42}, {'AMD': 51, 'FREE': 42}, {'AMD': 51, 'FREE': 42}]; no inherited component identifiers used to select folds. This does not establish biological homology.

| Model | Free full-sequence NLL | Amide full-sequence NLL | Equal-terminal NLL |
|---|---:|---:|---:|
| pooled | 53.05908 | 44.11513 | 48.58710 |
| separate | 52.93870 | 44.89674 | 48.91772 |
| partial | 53.00201 | 45.32661 | 49.16431 |
| ngram | 53.46059 | 45.48734 | 49.47396 |
| ar_cv | 51.75202 | 44.33403 | 48.04302 |
| ar_historical | 62.89841 | 53.03542 | 57.96692 |

Partial pooling did not beat pooled, separate, or the fresh strong AR on development balanced NLL. Grouped CV also failed pooled/fresh AR comparisons. Short-output/cap guards failed. No exact train copies occurred in these 384 requests, which is not proof against memorization generally.

Remaining limitations: one training seed; previously reused dev41; operational grouping rather than proven genealogy; shared-tree separate-emission ablation; unequal architecture capacity/compute despite equal nine-choice grids; fixed historical AR has prior development exposure; single-budget results cannot establish a learning-curve or biological activity benefit. PBCT-inspired grouping is prior art. No test37 access, original train/dev rereads by the auditor, uploads, or submission. BACKUP_BLOCKED preserved.

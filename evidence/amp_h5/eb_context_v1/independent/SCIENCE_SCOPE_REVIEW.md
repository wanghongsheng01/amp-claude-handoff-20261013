# Independent scientific scope review

The admitted question, if engineering checks pass, is whether terminal-conditioned shrinkage improves held-out full-sequence prediction and raw generation under this small development split. It is not an algorithm novelty claim.

Public sources directly checked on 2026-10-12:

- PBCT: https://link.springer.com/article/10.1007/s11222-026-10835-7 . The paper already groups sequential contexts and uses Bayesian predictive distributions to borrow strength, including protein-sequence applications.
- AGCT: https://arxiv.org/abs/1107.0312 . Group/shared context trees are established prior art.

The local model is independently implemented and should be labeled PBCT-inspired deterministic context grouping with empirical terminal shrinkage. It is not the author's PBCT software, exact posterior inference over trees, or a reproduction of its marginal likelihood evaluation. Shared pooled structures plus separate terminal emissions are an intentional emission ablation; they are not fully independently learned per-terminal structures.

The pooled prior includes both terminal groups, including the same terminal whose local counts appear in the shrinkage estimate. That defines a legitimate normalized plug-in predictive rule; it is not automatically a full hierarchical posterior or a calibrated uncertainty estimate. Lambda is chosen by train-only grouped cross-validation.

Equal nine-configuration opportunities do not equalize architecture capacity or total fitting compute. The experiment must publish compute and grid differences. A fresh train-only selected AR provides the primary fair comparison; any preexisting checkpoint is an explicitly historical secondary comparator.

Train-only connected components under a fixed global edit-similarity threshold are operational sequence-similarity groups. They neither establish biological homology nor exclude all remote relationships. Group provenance must exclude information from unopened partitions.

Even a favorable single-seed dev result is exploratory: dev41 has a historical reuse context and is not a fresh confirmation set. Count-based predictive gains, sequence-statistic gains, or terminal-dependent behavior cannot show antimicrobial potency, toxicity, biological efficacy, or submission readiness.

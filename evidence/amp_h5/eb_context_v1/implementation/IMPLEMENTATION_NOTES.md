# Independent grouped-context empirical Bayes implementation

Scope: model implementation and synthetic-only checks. No natural data, pretrained
model, source-root material, or heldout-data path is accessed by these modules.
Imports use Python's standard library only. No dependency install or downloaded
author code execution occurred.

## Prior-method inspection and differences

Official paper: https://link.springer.com/article/10.1007/s11222-026-10835-7
The paper's context partitions and recursive model-based agglomeration motivate
the structural baseline. Its likelihood is integrated categorical likelihood,
not a multinomial histogram probability. Its experimental counting omits initial
D symbols; that boundary convention is deliberately not used here. Its CRP prior
is not implemented. This prototype uses a fixed per-extra-child penalty and a
rare-symbol bucket/minimum-child-count rule. It is not a PBCT reproduction or a
claim of a new tree algorithm.

Author repository: https://github.com/daniyarghani/pbct
Read-only inspection covered README, `lib/pbct/tree_objects.py` and
`notebooks/test_pbct.ipynb` via public raw GitHub. The notebook constructs
`pbct.PBCT(V=...,max_depth=...,alpha=...)` and fits sequences; the tree exposes
`find_leaf_index` and `log_loss(x_train,x_test,eta,...)`. Official code uses NumPy
and additional notebook dependencies. No source from those files was executed
or installed. The independent code has different APIs and boundary/scoring rules.

## Fixed mechanism and API

`model.fit(rows, config)` receives dictionaries with `sequence` and `terminal`,
where terminals are exactly `Free` or `Amide`. The only accepted amino acids,
and output probability order, are `ACDEFGHIKLMNPQRSTVWY` followed by `<EOS>`.

Default structural constants: max depth 2 (allowed 1,2,3), tree Dirichlet alpha
0.5, penalty 1.0 nat per extra child, minimum child count 8. At each internal
node, singleton context symbols with fewer than 8 observations, including unseen
symbols and potentially BOS, enter a common rare bucket. The full deterministic
agglomerative path merges the pair giving the greatest integrated likelihood
gain. Among path partitions with every child count at least 8, choose the highest
integrated score minus the penalty. A single parent leaf always competes.
Ties use lexical cluster order; a split must strictly beat the parent.

The complete context alphabet is 20 residues plus BOS; EOS is never context.
Padded BOS permits prediction of the first residue. Every sequence has its own
boundary. Scoring length <48 includes the observed EOS, and length48 has no
additional EOS target, matching hard truncation. Immediate EOS produces an
empty sequence; sampling does not discard or retry it.

All three tree emission arms use the same pooled structure. **Separate** means
separate terminal emissions, not independently learned per-terminal trees.
For a leaf e, pooled probabilities use counts summed over both terminals and
add-alpha smoothing. Partial probabilities are
`(n[e,c,a] + lambda*p_global[e,a] + alpha)/(n[e,c] + lambda + 21*alpha)`.
Separate is exactly lambda=0; pooled is the lambda-to-infinity limit.

`with_emission` and `reconfigure` clone the tree/count snapshot and reconfigure
only emissions. They avoid structural refitting and isolate caller mutations.
`probs(prefix,terminal)` returns a tuple; `logscore(seq,terminal)` returns frozen
log probability (natural logarithms); `sample(seed,terminal,requests)` returns
unscreened sequence strings. `to_dict/from_dict` serialize the full model;
`digest` hashes the serialized snapshot. Evaluation never updates counts.

`fit_ngram` provides the same prediction/scoring/sampling/serialization interface.
It stores all suffix orders through max_depth and backs off to the longest
seen context. Separate mode uses terminal-specific context support, pooled mode
uses pooled support. The runner must specify the intended baseline mode.

## Tests and limits

Run `python3 -m unittest -v test_model` in this directory. Nine synthetic tests
cover normalization, unknown contexts, direct partial formula and its limiting
cases, first residue, EOS, truncation48, count conservation, shared structures,
row-order determinism, frozen evaluation, serialization, backoff, and invalid
inputs. The synthetic heldout check establishes no evaluation-time updates;
actual homology-fold integrity remains the runner's independent responsibility.

`profile_synthetic.py` constructs synthetic sequences without filesystem data
reads and profiles fit/scoring only. Its timing is a planning estimate, not a
guarantee for a natural dataset. No new biological efficacy, novelty or submission
readiness claim follows from this implementation.

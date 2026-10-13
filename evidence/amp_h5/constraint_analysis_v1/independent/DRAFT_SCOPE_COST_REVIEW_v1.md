# Independent review of the single-candidate analysis draft

Date: 2026-10-12. Reviewed artifact: `../DRAFT_FOR_AUDIT.txt`, SHA256 `4a1974fa25ecb7a850ce5e546ef49fab146d909e71681c249bb383abc3520928`. This is an analysis/protocol draft authorized for review, not a manuscript. Review did not inspect the original data or model; assertions about the existing architecture, parameter count and historical timing receipt were not independently verified here.

**Disposition: the draft’s current scientific NO / novelty REJECT conclusion is supported. No natural run is admitted by this review.** EOS/censor separation, nominal-statistic naming, genuine recurrent states, proposal support, weight correction, finite-N limitations and the conditional nature of the approximation bound match the independent prior-art audit.

The original seven-arm arithmetic is correct: 4 cells × 3 seeds × 7 arms × 16,384 scores = 1,376,256 scores and 840 CPU seconds for inference. These are proposed ceilings, not measured feasible runtimes. Nothing in this review validates 10-second runs, reference acceptance, or sufficient statistical resolution from three seeds.

Required clarifications before representing this as a complete counterfactual diagnostic:

1. Add same-guide full-path importance sampling, or explicitly state that weighting versus resampling attribution remains unresolved. Guide-only versus corrected SMC changes both. With eight arms, the inference caps become 1,572,864 active scores and 960 CPU seconds (96 cell-seed-arm units). This comparator is not another candidate.
2. A 120-second/1-GiB preparation limit cannot by itself establish a strong CTL comparator. If an adequately implemented baseline cannot fit, mark it unavailable/ineligible under the resource envelope and keep broad superiority HOLD. Do not record resource-ineligible CTL as a scientific failure of CTL. Apply the same completeness standard to MH reverse probabilities and all SMC baselines.
3. Keep end-to-end and amortized workloads separate, and explicitly book remaining automaton/table preparation against the total cap. The proposed single-thread cost frontier is more defensible than declaring equivalent costs across neural scores, DP operations and native grammar work.
4. Mark author-code evidence as static inspection. No source transplant of LSSAMP/twisted-smc-lm is cleared because reuse licensing was not established. Independent paper-based implementation requires attribution and parity evidence; it is not official executable reproduction by name.
5. Cite a verified venue record or use the year/arXiv identifier for the Loula paper. This auditor verified the full arXiv source, not the draft’s venue label. Preserve the distinction between a published equation and a reproduced experiment throughout.

The proposed ≥2× throughput screen cannot revive novelty. Fixed-feature agreement is a partial diagnostic, not full-sequence distance or biological evidence. All-zero/support failures, poor reference acceptance, exhausted caps and negative comparisons must remain visible. Since the main decision is rejection, none of these clarifications justifies running the illustrative design now.

One local evidence-hash command failed because `shasum`/Perl rejected the configured locale; Python hashlib produced the SHA256 above without changing the source artifact. No model execution was involved.

# Witten CNN architecture-equivalent CPU baseline

Synthetic-only validation passed. This is a PyTorch port of the **cached QMAP adaptation** of Witten CNN, not a reproduction of original-paper performance or a direct TensorFlow cross-framework test. No natural peptide records, train/dev/test split, pretrained weights or online resource was opened by these scripts.

## Source mapping

`AUTHOR_BINDING.json` pins the previously retrieved source snapshot and each relevant file's content SHA256. The source repository branch was main; hashes bind content, not an unverified original-paper commit.

| Operation | Source / local behavior |
|---|---|
| Input | `(batch,100,21)`, canonical uppercase amino-acid one-hot; zero vectors after sequence; channel21 unused |
| Token order | Author enumerates a Python set, so order varies; this implementation freezes `ACDEFGHIKLMNPQRSTVWY`. This is a channel permutation with a new random initialization, not pretrained-weight compatibility. |
| Unsupported tokens | Unknown/noncanonical/lowercase raises KeyError, as source encoder; no implicit uppercase/chemistry normalization |
| Length | Author encoder truncates100; QMAP trainer filters sequence length **<100**. Future dataset admission must enforce its chosen frozen policy separately. |
| Convolution stack | ZeroPad5 on both ends, Conv64/kernel5/stride1/valid+ReLU, MaxPool2/stride2, Conv64/kernel5/stride1/valid+ReLU, MaxPool2/stride2 |
| Length trace | 100 → 110 → 106 → 53 → 49 → 24; then channels-last flatten1536 |
| Head | Dropout0.5 → Dense100+ReLU → Dense20+ReLU → Dense1 linear |
| Parameters | 6784 + 20544 + 153700 + 2020 + 21 = **183069** |
| Initialization | Glorot uniform kernels, zero biases; equivalent distribution, different RNG from TensorFlow |
| Loss | Mean squared error, scalar target `(batch,1)`; source natural label comment is log10(MIC in µM). No MIC labels are used here. |
| Optimizer | Dense Adam lr0.001, betas0.9/0.999, epsilon-hat1e-7, no decay/AMSGrad/clipping. Defaults inferred from Keras2.x `optimizer='adam'`, not literal author assignments. Custom epsilon placement uses sqrt(v)+eps before bias-correction scale. |
| Fit defaults | Keras batch32/shuffleTrue; QMAP CLI5epochs by default, random split20; CLI negatives0. Synthetic fixed-batch test does not copy dataset/split/epoch schedule. |

Dense Adam here counts steps per parameter; matching Keras global-iteration semantics requires **every parameter receive a gradient each step**. The smoke loop checks that condition. Future conditional/frozen/partially disconnected models need a separate optimizer review. Keras/TensorFlow was not installed or executed; floating-point reduction, dropout streams, convolution implementation and seed trajectories are not expected to match bitwise.

The source CNN has no explicit terminal/strain/assay input. It cannot distinguish chemically different entities sharing a canonical sequence. Chemistry-rich data admission or additional inputs are subsequent controlled decisions, not validated by this baseline port.

## Frozen execution and evidence

`FREEZE.json` was written before each attempt. `PREFLIGHT.json` records the process inspection and single-model slot assignment. CPU torch intra/inter-op threads and BLAS environment all1; no MPS. Seed1729. Hard cap120s and1.5GB RSS, maximum105 full-model training steps plus bounded numerical tests. All sequence inputs are generated in memory.

`RESULT.json` contains 13 checks: encoding/padding/truncation/unknown rejection, parameter count, independent channels-last NumPy forward, central finite differences for the largest nonzero coordinate in each of10 weight/bias tensors, four-step NumPy Adam comparison including small gradients, finite loss/gradients and synthetic learnability, evaluation determinism, dropout activity and resource limits. The gradient test samples coordinates; it is not an exhaustive gradient proof.

Attempt1 failed while serializing NumPy integer coordinates to JSON, before optimizer/synthetic training. `attempt1/` preserves the source, freeze, partial result and stderr. `REPAIR.json` documents the logging-only cast and unchanged scientific configuration. Attempt2 passed. There was no numeric threshold loosening.

Measured on macOS15 arm64/Python3.12.13/torch2.2.2/NumPy1.26.4:

- NumPy forward maximum absolute error1.33e-15; selected gradient maximum scaled error7.08e-10.
- Adam four-step maximum absolute error0; synthetic evaluation MSE0.528477→0.000451876 after80steps. This is a software check, not biological prediction evidence.
- Batch32:20 timed training steps0.442096s =0.022105s/step;20 inference steps0.334957s. Peak process RSS290177024bytes (~277MiB); full run5.31s.

The timing sample is short and excludes natural-data parsing, grouped validation, search, repeat runs and external load variation. A provisional compute estimate is `ceil(N/32) * epochs * 0.022105` seconds **per model fit**, followed by a safety multiplier and a frozen step/time cap. It is not a measured natural-data training budget. No checkpoints or natural results were produced.

## Re-run

Use the existing environment; no installation is needed. Preserve a prior RESULT/stdout/stderr before a rerun, recheck that no other H5 model job is active, and obtain the shared model slot. Then:

```sh
/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/env/bin/python /Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5/witten_cpu_v1/validate_synthetic.py
```

Natural-data training remains gated on independent chemical identity/measurement review, family/source grouping and contamination isolation. This package does not establish any of those gates.

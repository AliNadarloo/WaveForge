<!-- Made by Ali Nadarloo. -->
# Waveforge 1.1 verification

Public results are summarized in verification/public_summary.json. The clean
repository excludes local-machine logs and optional original-paper audit snapshots.
The test suite skips optional snapshot comparisons when those fixtures are absent.
The scalar design checks here are separate from full-wave validation in the paper.

## Numerical and interface regression

The current complete suite passes 53 tests. It checks the unchanged paper
apertures, exact same-device dense matrices, complex fields, losses, phase
gradients, Adam steps, archived paper solutions, anisotropic dense/FFT agreement,
both padded ASM variants, batched gradients, all nine loss examples, all five
optimizers, independent polarizations, cancellation, CPU fallback and CST axes.

New checks cover disjoint and deterministic digit splits, nonoverlapping numbered
circles, four trainable layers, mini-batch learning, output-distance changes,
relevant-only form rows, the show-all override, theme/color preferences, explicit
per-plane distance controls and selected color maps in saved plots. Real Qt
widgets, worker threads, dual editing, guide controls, live phase views and export
are exercised. The public summary records the release checks; GitHub Actions repeats CPU tests.

## Paper reproduction

The original 3,000-epoch two-channel design produces 14/14 correct truth-table
cases per polarization. Final losses are 0.012996069 and 0.030661291. Its mean
contrasts reproduce the archived 24.10 and 20.40 dB results (measured 24.095116 and
20.404121 dB). Maximum circular phase differences from the six-decimal archived
phase tables are 0.000929 and 0.000370 rad. Numerical precision and hardware can
change a long optimization trajectory; bit-identical phases on every machine
are not claimed. The physical model, settings and phase convention are retained.

## Eight full GUI use cases

The validation script launches the actual Windows EXE, trains, exports, saves
screenshots and checks task-specific outcomes for eight separate frequencies:

| Design | Frequency | Required outcome |
|---|---:|---|
| Gaussian focus | 0.3 THz | Intensity correlation above 0.85 |
| Double-slit steering | 0.6 THz | Centroid within 0.1 lambda of target 3 lambda |
| ONN text hologram | 0.8 THz | Intensity correlation above 0.85 |
| Charge-2 OAM | 1.2 THz | Angular purity above 0.95, loss below 0.05 |
| Image aperture / Bessel-Gauss focus | 473.6 THz | Intensity correlation above 0.85 |
| Two observation planes | 0.9 THz | Combined loss below 0.2 |
| Independent dual channels | 0.1 THz | P1 image fit and P2 OAM purity above 0.95 |
| Full paper, both channels | 1.0 THz | 14/14 correct cases for each channel |

All channels must decrease loss, use CUDA, export finite phase maps and CST
coordinate tables, and keep the measured UI heartbeat's 95th percentile below
250 ms. Final metrics and responsiveness measurements are in verification/public_summary.json.
Screenshots of the application are in docs/images/. Correlation is mean-subtracted normalized image
correlation, not efficiency. Designs use separately scaled cell pitches; they do
not claim one physical cell operates over all frequencies.

## Four-layer handwritten digit routing

A complete source run at the default 100 epochs and batch size 64 reached:

- Training: 94.37% on 1,438 samples.
- Held-out: 93.04% on 359 separate samples.
- Loss: 4.955631 initially, 0.841653 finally.
- Mean correct-circle fraction: 0.4433 of sampled output power.
- GPU: NVIDIA GeForce RTX 3050, PyTorch 2.5.1+cu121.
- Training/evaluation time without GUI/export: 22.69 seconds.

The final digit GUI check reruns the design through the delivered EXE and requires
at least 90% held-out accuracy, all four phase/CST exports, a complete 359-sample
confusion matrix and predictions for both disjoint splits. Its report is
verification/public_summary.json. Representative fields show one held-out input per
digit; prediction tables cover every evaluated sample. The UCI/scikit-learn data
and attribution are in examples/digits/. This is an educational split of that
dataset, not MNIST, the original UCI benchmark split or full-wave validation.

## Lightweight Windows launcher

The new EXE contains readable app code, examples and offline guides. It contains
no Python, Qt, PyTorch or CUDA runtime. A compatible installed Python environment
is required; INSTALL.md provides CPU/GPU commands and verification, and the native
setup screen can select Python and copy commands. It never installs packages or
downloads CUDA automatically. Users can edit app_source/onnlab and restart.
The launcher preserves edited files using distribution hashes.

Actual CUDA, forced CPU, independent dual CUDA and ASM CUDA runs are checked
separately. Unavailable-CUDA fallback is unit tested; a second machine without an
NVIDIA driver is not available. The guide, all four themes, selected color maps
and setup screen are rendered for visual checks. Executable/source hashes and
actual device checks are summarized in verification/public_summary.json.

## Performance and delivery

The unchanged propagation benchmark for 14 inputs measured forward/backward
speedups of 6.75x dense, 9.66x FFT RS and 8.81x ASM versus individual evaluation on
the RTX 3050. These are propagation-path timings, not total-training speedups.
Paper presets preserve their original accumulation order. Live views reuse artists,
limit preview resolution and displayed trace points, while full fields and every
recorded epoch retain their resolution. Digit training uses bounded mini-batches.

Packaging checks the executable and source hashes and requires all final GUI
experiments to pass. Both ZIPs receive full CRC checks and SHA-256 manifests.
Only the new small release, readable source, examples, current verification data
and relevant training outputs are retained; obsolete bundled runtimes/builds are
removed within application/. Original project files remain unchanged.

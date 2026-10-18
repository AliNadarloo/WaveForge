<!-- Made by Ali Nadarloo. -->
# WaveForge validation

Validation results are summarized in `verification/public_summary.json`.
Optional paper-reference comparisons run when the required fixtures are available.
These scalar-model checks are separate from the paper’s full-wave electromagnetic validation.

## Numerical and interface tests

The release passes 53 automated tests covering:

- Paper aperture definitions, dense propagation matrices, complex fields, losses, phase gradients and Adam updates.
- Comparisons with saved paper solutions.
- Agreement between dense and FFT Rayleigh–Sommerfeld propagation with different x and y cell pitches.
- Both padded angular spectrum methods and batched gradients.
- All nine loss examples and five optimizers.
- Independent polarization channels, cancellation, CPU fallback and CST coordinates.
- Deterministic, disjoint digit training and evaluation splits.
- Four-layer digit training, detector placement, mini-batches and output-plane distances.
- Context-sensitive controls, themes, color maps and saved-plot appearance.
- Qt widgets, training threads, polarization editing, guides, live plots and export.

GitHub Actions runs the CPU test suite. Detailed release measurements are available in `verification/public_summary.json`.

## Paper reproduction

The original 3,000-epoch, two-channel scalar design produces **14/14 correct truth-table cases per polarization**.

| Measurement | Polarization 1 | Polarization 2 |
|---|---:|---:|
| Final loss | 0.012996069 | 0.030661291 |
| Mean contrast | 24.095116 dB | 20.404121 dB |
| Maximum circular phase difference from the saved six-decimal phase tables | 0.000929 rad | 0.000370 rad |

The measured contrasts agree with the saved results of 24.10 and 20.40 dB.

Numerical precision and hardware can affect a long optimization trajectory, so identical phase values on every machine are not guaranteed. The paper preset retains the physical model, settings and phase convention.

## Eight application examples

Eight examples were tested through the Windows executable, including training, live visualization and export.

| Design | Frequency | Acceptance criterion |
|---|---:|---|
| Gaussian focus | 0.3 THz | Intensity correlation above 0.85 |
| Double-slit steering | 0.6 THz | Centroid within 0.1 wavelength of the target at 3 wavelengths |
| ONN text hologram | 0.8 THz | Intensity correlation above 0.85 |
| Charge-2 OAM | 1.2 THz | Angular purity above 0.95 and loss below 0.05 |
| Image aperture with Bessel–Gauss focusing | 473.6 THz | Intensity correlation above 0.85 |
| Two observation planes | 0.9 THz | Combined loss below 0.2 |
| Independent polarization channels | 0.1 THz | Polarization 1 image fit and Polarization 2 OAM purity above 0.95 |
| Paper logic gates, both channels | 1.0 THz | 14/14 correct cases per polarization |

All eight trials passed their acceptance criteria. Each trial used CUDA, decreased the loss and exported finite phase maps and CST coordinate tables. The measured interval between UI heartbeat events remained below 250 ms at the 95th percentile.

Final metrics and responsiveness measurements are in `verification/public_summary.json`. Application screenshots are in `docs/images/`.

Intensity correlation means mean-subtracted normalized image correlation; it does not measure optical efficiency. Each example uses its own cell dimensions. These examples do not imply that one physical unit cell operates across all listed frequencies.

## Four-layer handwritten digit routing

The digit example trains four phase-only surfaces to direct light toward the output circle assigned to each digit.

A source run with 100 epochs and a batch size of 64 produced:

| Measurement | Result |
|---|---:|
| Training accuracy | 94.37% on 1,438 samples |
| Held-out accuracy | 93.04% on 359 separate samples |
| Initial loss | 4.955631 |
| Final loss | 0.841653 |
| Mean fraction of sampled output power in the correct circle | 0.4433 |
| Training and evaluation time, excluding GUI and export | 22.69 seconds |

Hardware and software: NVIDIA GeForce RTX 3050, PyTorch 2.5.1+cu121.

The executable test requires at least 90% held-out accuracy, phase and CST exports for all four surfaces, a confusion matrix covering all 359 held-out samples and prediction tables for both splits.

Representative field plots show one held-out input per digit. Prediction tables cover every evaluated sample.

The dataset and its attribution are documented in `examples/digits/`. This example uses an educational split of the UCI/scikit-learn digits dataset. It does not use MNIST or the original UCI benchmark split, and it does not constitute full-wave electromagnetic validation.

## Windows launcher and device support

The lightweight executable contains the application source, examples and offline guides. Python, Qt, PyTorch and CUDA runtimes are external dependencies.

A compatible Python environment is required. `INSTALL.md` provides CPU and GPU installation commands and device checks. The setup screen lets users select a Python environment and copy the required commands. It does not install packages or download CUDA automatically.

Users can edit `app_source/onnlab` and restart the application. Distribution hashes allow the launcher to preserve edited files.

Executable checks cover:

- CUDA training.
- Forced CPU training.
- Independent dual-channel CUDA training.
- Angular spectrum propagation on CUDA.
- Guides, themes, selected color maps and the setup screen.

Fallback when CUDA is unavailable is unit tested. Testing on a separate machine without an NVIDIA driver was not performed.

Device checks and executable/source hashes are summarized in `verification/public_summary.json`.

## Performance

For a propagation benchmark with 14 inputs on the RTX 3050, batched forward/backward evaluation achieved the following speedups over evaluating inputs individually:

| Propagation method | Measured speedup |
|---|---:|
| Dense Rayleigh–Sommerfeld | 6.75× |
| FFT Rayleigh–Sommerfeld | 9.66× |
| Angular spectrum | 8.81× |

These measurements describe propagation performance, rather than total training time.

Paper presets retain their original accumulation order. Live plots reuse existing plot objects and limit preview resolution and displayed trace points to keep the interface responsive. Full field arrays and the complete recorded loss history remain available. Digit training uses bounded mini-batches.

## Export checks

Export checks cover phase maps, finite numerical values, selected plot color maps and CST coordinate tables.

CST exports use metasurfaces in the xy plane with illumination traveling from −z toward +z. The application explains this convention when exporting.

Field plots use display interpolation according to the selected appearance settings. Metasurface phase-cell plots remain unsmoothed.

Release ZIPs are checked for archive integrity. SHA-256 checksums are supplied with the downloads.

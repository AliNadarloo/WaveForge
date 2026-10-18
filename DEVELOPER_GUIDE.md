<!-- Made by Ali Nadarloo. -->
# Make Waveforge your own

Made by Ali Nadarloo.

The source is ordinary Python in small modules. There is no generated UI file,
remote service, private API, encrypted model or dependency on the parent paper
folder. Start by running `python main.py`; edit a file and restart to see changes.

## Set up an editable environment

Use Python 3.10 on Windows to reproduce the tested build. An isolated environment
is recommended so installing your own versions does not change other projects.
Start with [INSTALL.md](INSTALL.md), including its CPU and GPU installation commands,
environment selection, driver requirements and verification check.
`requirements-lock.txt` records the original tested environment; `requirements.txt` gives
the compatible version ranges. Install a CUDA-enabled PyTorch wheel to train on
an NVIDIA GPU. The tested torch 2.5.1+cu121 wheel is available from the official
PyTorch CUDA 12.1 wheel index. Other machines can use CPU PyTorch to run the source.

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python main.py
```

The lightweight executable uses your installed Python environment. It contains no
Python, Qt, PyTorch or CUDA runtime. The native setup screen finds/selects Python
and displays package instructions. The app's editable files are extracted under
`app_source/onnlab/`; edit and restart to see changes. User edits are preserved
when an updated EXE extracts its files. Remove a file to restore its embedded
version. Additional dependencies must be installed in the selected environment.

## Where to change things

| File | Responsibility |
|---|---|
| `main.py` | Command-line arguments and launch mode |
| `onnlab/bootstrap.py` | Responsive launch window and background imports |
| `onnlab/config.py` | Named settings, units, validation and channel separation |
| `onnlab/inputs.py` | Apertures, images/text, beam profiles, target fields |
| `onnlab/physics.py` | Grids, transfer functions, cell phases, cascades |
| `onnlab/objectives.py` | Prepared tensors/functions and example Python losses |
| `onnlab/training.py` | Optimizers, gradients, progress, cancellation, CPU fallback |
| `onnlab/recommend.py` | Explainable propagation-method recommendation |
| `onnlab/conventions.py` | CST axes, handedness, physical cell-center coordinates |
| `onnlab/export.py` | Human-readable phase tables, fields, metrics and figures |
| `onnlab/presets.py` | Complete ready-to-run experiments |
| `onnlab/ui.py` | Widgets, styling, live plots, worker signals |
| `onnlab/guide.py` | Offline quick start, manual topics and experiment walkthroughs |
| `onnlab/appearance.py` | Persistent themes and independent field color maps |
| `onnlab/digits.py` | Bundled handwritten data, split and circular detector geometry |
| `onnlab/digit_training.py` | Mini-batches, held-out evaluation and representative fields |
| `onnlab/digit_export.py` | Predictions, confusion matrix and accuracy reports |
| `onnlab/paths.py` | Shared app home beside the EXE or source launcher |
| `launcher/WaveforgeLauncher.cs` | Small native environment detector, setup screen and Python launcher |

Shared geometry / optimizer fields are listed in `SHARED_FIELDS`. The remaining
`CHANNEL_FIELDS` belong to each scalar polarization. Add a setting to the dataclass
and put it in the correct group. Lengths in the physics engine are metres; the UI
uses micrometres, THz and wavelengths explicitly.

## Write a new loss first

The easiest extension needs no engine changes. Copy a file from
`examples/loss_functions/` into the Python editor, then edit `loss(ctx)`.
The Parameters tab describes every available tensor and function. Use PyTorch
operations to keep gradients. Optional `metrics(ctx)` and `plots(ctx)` describe
your diagnostics. Preview checks the loss and phase gradients before training.

Save an experiment JSON to keep all settings and the loss together. The eight
validated frequency / task configurations are in `examples/validated_uses/`;
their referenced picture is relative, so they remain portable after extraction.
You can load them through **Load experiment** in the application.

## Extend an aperture, beam or propagation method

Add the implementation to `inputs.py` or `physics.py`, expose its name in the
relevant UI selector, and update validation / recommendations. `Propagator.forward`
accepts a complex `[H,W]` or `[batch,H,W]` tensor. Methods must use the common
`exp(+iωt)` phasor convention. New propagation tests should check an analytic field,
phase sign, finite gradients, coordinates and batch equivalence.

An ASM fine grid is distinct from the trainable cell grid. `surface_phases` always
has one phase per physical cell. Preserve field samples between surfaces and use
`field_dx` / `field_dy` for sampled-field powers and observation coordinates.

## Performance choices

Transfer functions, inputs, targets, detector masks and extra-plane propagators
are reused. General multi-input cases use batched GPU propagation; dense paper
presets retain the original per-sample order for numerical reproduction. Static
labels stay as Python values when used in Python branches, avoiding repeated GPU
synchronization. Finite-gradient checking shares the clipping norm calculation.

Heavy imports run behind a lightweight animated launch window. Training uses a
worker thread. Live images are limited to 256 samples per axis before GPU-to-CPU
transfer; full-resolution fields and exports are unchanged. The plot reuses image
and line objects, keeps every recorded epoch, and draws at most 2,500 trace points.

`python benchmark.py` measures the propagation forward/backward path with warmup
and explicit CUDA synchronization. The recorded 14-input GPU speedups are about
6.75× (dense), 9.66× (FFT RS) and 8.81× (ASM), compared with individual propagation.
These measurements do not claim the same speedup for an entire training run.

## Test and build

```text
python -m pytest tests -q
python build.py --output release/Waveforge
python verify_executable.py
python validate_usecases.py
```

`build.py` creates `release_next/Waveforge/` by default; pass `--output` to choose
another folder. It uses Windows' .NET Framework C# compiler to compile the small
launcher and embeds a compressed app payload. No PyInstaller or third-party
compiler package is needed. Python, Qt, PyTorch and CUDA remain external. The
launcher extracts files with path containment checks, detects Python with the
required packages and preserves user-edited files using recorded content hashes.
WAVEFORGE_HOME points to the launcher directory; WAVEFORGE_PYTHON can select an
existing environment. `Waveforge.exe --setup` opens its environment/setup guide.
If upgrading PyTorch, repeat numerical and executable checks. The optional original-paper
reference snapshots are for review and regression tests only.

The validation scripts write only under this application directory. The final
source ZIP contains all application modules, tests, build scripts, examples,
reference snapshots, environment details and measured verification results.

## Extend the adaptive interface and objective workshop

`ui.py` records each form row and hides unrelated rows in `refresh_context`.
The hidden values are retained for round-tripping and custom code. Mode changes
select a matching built-in loss, while user-edited Python is preserved. Add new
dependencies between controls there, and test both sides of each transition.
`SpacingsEditor` keeps the N+1 distances explicit and labels the final output plane.
Presentation preferences are separate from optical configuration and are stored
beside the launcher; changing colors must not change the training result.

The in-app Loss lab reads `LOSS_GUIDE.md`. Update that file, `PARAMETERS` in
objectives.py and the copyable recipe files when adding context tensors. Full
INSTALL.md, README.md and this developer guide can also be read in the app.

## Appearance and figure export

`appearance.py` defines the shared palettes and saved-figure styling. `ui.py`
uses bilinear display interpolation for sampled fields and nearest-neighbor
rendering for individual metasurface phases. `export.py` uses the same rules;
wrapped output-field phases interpolate colors rather than averaging angles
across the −pi/+pi seam. Numerical fields and phase tables are never resampled.
The export worker captures current preferences when export begins. Results can
be exported again with a different appearance, reusing the completed model.
`digit_export.py` retains discrete confusion-matrix cells and styles its plots
with the same palette. `tests/test_plot_appearance.py` checks all four themes.

Build the Windows launcher with `python build.py --output release/Waveforge`.
Then run `python package.py` to create the EXE, Windows ZIP, source ZIP and
`SHA256SUMS.txt` in `delivery/`. Python and the numerical runtime stay external.
Packaging checks the distributed files against the build manifest and checks each ZIP.

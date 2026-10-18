<!-- Made by Ali Nadarloo. -->
# Install and run Waveforge

## The lightweight Windows executable

Waveforge.exe is a small launcher containing the app's readable Python code,
guides and examples. It **does not bundle Python, PyTorch or CUDA libraries**.
On first launch it places the editable files beside itself. Keep it in a writable
folder, such as a folder on your Desktop. Results are saved in results/ beside it.

It looks for a compatible Python installation with NumPy, PyTorch, Matplotlib,
Pillow and PySide6. It first checks a saved selection, WAVEFORGE_PYTHON, an adjacent
.venv, the Python launcher, PATH, and normal per-user Python install locations.
If none is ready, it opens a setup screen. Use Browse to select python.exe in your
environment, then Check and start. Run Waveforge.exe --setup to reopen that screen.

## If the required packages are already installed

Double-click Waveforge.exe. No new CUDA download is performed by the launcher.
Its setup screen lets you select the existing Python environment to use.
An environment containing CPU-only PyTorch can run CPU training; a CUDA-capable
PyTorch environment can use an NVIDIA GPU when its driver is compatible.

## Fresh setup: Python and a local environment

Install a supported Python release from https://www.python.org/downloads/.
The pinned PyTorch commands below are intended for Python 3.10–3.12; choose
Python 3.12 for a new environment using those commands. Newer Python releases
may require newer PyTorch versions from the official selector. This application
was tested with Python 3.10.7 and PyTorch 2.5.1+cu121 on Windows.

Open Terminal or PowerShell in the folder containing Waveforge.exe. Run it once
if necessary to extract the guides and requirements file, then enter:

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
```

If using an existing Python 3.10 or 3.11 instead, replace -3.12 appropriately.
You do not need to activate the environment: these commands explicitly select it.

## Choose CPU or NVIDIA GPU support

For CPU-only use:

```text
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

For the CUDA 12.1 PyTorch build tested with this application:

```text
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Choose one. The GPU package is a large, separate installation into your environment;
it is not hidden inside the executable. Keep an existing working CUDA-enabled
PyTorch installation instead of reinstalling it if you already have one.

An installed CUDA Toolkit alone does not provide PyTorch, Qt or the other required
Python packages. Standard CUDA-enabled PyTorch wheels supply their corresponding
runtime libraries; they require a compatible NVIDIA driver and normally do not
need a separately installed CUDA Toolkit. Installing a Toolkit does not convert
CPU-only PyTorch into CUDA-enabled PyTorch. Building PyTorch against your own
Toolkit is an advanced alternative outside this app's setup process.

Official wheel choices and commands:
https://pytorch.org/get-started/previous-versions/#v251
https://pytorch.org/get-started/locally/

## Check the environment and GPU

```text
.venv\Scripts\python.exe -c "import torch; from PySide6 import QtWidgets; import numpy, matplotlib, PIL; print('PyTorch',torch.__version__); print('CUDA build',torch.version.cuda); print('GPU available',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU training ready')"
```

GPU available True confirms PyTorch can access CUDA. Launch the app, choose
Auto (CUDA / CPU), run Preview, and confirm its reported device is CUDA.
The run summary records the actual compute device. CPU explicitly selects CPU;
Auto falls back to CPU when CUDA is unavailable. No GPU is required to use the app.

If CUDA reports False: check that PyTorch is a CUDA build, your NVIDIA driver is
installed and compatible, and you selected the same Python environment used for
the check. Use the Windows Start menu NVIDIA tools or nvidia-smi to inspect the
driver. If updating packages, repeat Preview and a small training run.

## First experiment

1. Select Gaussian focus.
2. Set the number of cells, pitches and frequency in Optical system.
3. Set each plane distance; the last row is the output distance from the last surface.
4. Set the aperture, illumination and target. Only relevant controls appear.
5. Insert a matching Python objective recipe; Preview checks fields and gradients.
6. Launch training. Results export automatically when finished.

Guide & examples contains an offline quick start, manual, walkthroughs and Loss lab.
Read LOSS_GUIDE.md for the full objective workshop. Appearance controls themes
and independent input/intensity/phase color maps. Choices are saved beside the app.

## Source-code users

Extract the source ZIP. Install the same runtime packages and PyTorch, then run:

```text
python main.py
```

For tests also install pytest and pandas (pandas is needed only by the unchanged
paper-reference helper). See DEVELOPER_GUIDE.md for file layout, build and checks.
The launcher source is launcher/WaveforgeLauncher.cs. App modules are ordinary
Python; the light executable extracts them under app_source/onnlab for editing.

## Common setup problems

- Wrong Python: use Waveforge.exe --setup and Browse to your environment's python.exe.
- Packages appear missing: install them with that exact python.exe -m pip.
- Cannot save: move the app to a writable folder before launching.
- Startup fails: check waveforge_error.log beside the app. The setup screen can
  open this guide and copy installation commands without installing anything itself.
- To reset appearance, remove ui_preferences.json beside the app while it is closed.
- An edited app_source file is preserved by the launcher. To restore a specific
  file to the embedded version, remove that file and launch again.

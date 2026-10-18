<!-- Made by Ali Nadarloo. -->
# Improve WaveForge

Made by Ali Nadarloo. Community contributions are welcome.

Useful contributions include clearer workflows, new optical objectives, input
formats, propagation accuracy/performance, plots, accessibility and documentation.
Small, focused pull requests are easiest to review. For a substantial feature,
open an issue describing the optical use case and proposed behavior first.

## Run and test

Follow INSTALL.md to create an environment with CPU or CUDA-enabled PyTorch and
the runtime packages. Then:

```text
python -m pip install pytest pandas
python main.py
python -m pytest tests -q
```

CPU is enough to contribute. CUDA-specific comparisons run only when CUDA is
available. Paper-reference comparisons use optional files in `reference/`.
Those comparisons are skipped when the required files are unavailable.

## Make a change

1. Fork the repository and create a branch describing the improvement.
2. Keep optical calculations separate from UI and export code.
3. Add a meaningful check for changed physics, gradients, axes or batch behavior.
4. Exercise changed UI controls and update the related guide/recipe.
5. Open a pull request explaining the user problem, resulting behavior and checks.

Lengths inside physics/objectives are metres; UI labels use explicit units.
Keep the common exp(+i omega t) convention and outgoing exp(-ikR) propagation.
Do not rename scalar channels to a physical polarization basis automatically.
Presentation settings must not change the optical calculation. New losses should
return a finite real scalar tensor and retain differentiability.

The code map is in DEVELOPER_GUIDE.md. LOSS_GUIDE.md is the objective workshop.
Keep existing copyright notices and third-party attribution. New contributions
are provided under the project's MIT license; contributors may add appropriate
notices for their own work.

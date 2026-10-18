<!-- Made by Ali Nadarloo. -->
# Third-party material

Made by Ali Nadarloo applies to the Waveforge application and integration work.
The MIT license in LICENSE covers that application code. It does not replace
licenses or authorship of separately supplied data, reference snapshots or
installed dependencies.

- The bundled handwritten digit data is credited to E. Alpaydin and C. Kaynak,
  UCI Machine Learning Repository, under CC BY 4.0. See examples/digits/DATASET.md
  for the source, DOI, preprocessing and license link.
- Python, PyTorch, NumPy, Matplotlib, Pillow and PySide6 retain their own licenses.
  They are external installations and are not embedded in the lightweight EXE.
- reference/ contains local audit copies supplied with the original project.
  Existing author notices remain. An integration credit comment identifies the
  Waveforge packaging; it does not claim authorship of the reference source.
  These snapshots are excluded from the clean GitHub bundle. The application
  does not need them at runtime; local paper-regression tests use them when present.

Mathematical conventions, data provenance and measured comparisons are documented
in README.md, VERIFICATION.md and the optional reference provenance files.

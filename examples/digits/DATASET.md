<!-- Made by Ali Nadarloo. -->
# Handwritten digit example data

Optical Recognition of Handwritten Digits, E. Alpaydin and C. Kaynak (1998),
UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C50P49.
Dataset license: Creative Commons Attribution 4.0 International,
https://creativecommons.org/licenses/by/4.0/.

The bundled NPZ contains the 1,797 8×8 images and labels distributed by
scikit-learn's `load_digits`, a copy of the UCI dataset's original test partition.
Values were stored as uint8 (images) and int64 (labels), without altering pixels.
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

Waveforge makes its own deterministic stratified 80/20 training/held-out split
within these samples. This is an educational optical classification example,
not a score on the original UCI benchmark split and not MNIST. Runtime loading
needs only NumPy. Fields resize and center each image, flip raster rows for
Cartesian axes, multiply the chosen beam and normalize input energy.

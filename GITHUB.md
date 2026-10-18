<!-- Made by Ali Nadarloo. -->
# Repository and release maintenance

Repository: https://github.com/AliNadarloo/waveforge-d2nn

Made by Ali Nadarloo, School of Electrical Engineering, Iran University of
Science and Technology, Tehran, Iran. The paper and master's thesis context,
DOI and recommended citation are in README.md and CITATION.cff.

## Contributing

Use CONTRIBUTING.md for the contribution workflow and DEVELOPER_GUIDE.md for
installation, the module map, optical conventions, tests and builds. CPU-only
contributions are welcome. Issues accept bugs, example requests and proposals.
Keep changes focused and explain measured optical or interface behavior.

## Release checklist

1. Run the CPU tests and any applicable CUDA comparisons. Verify fields,
   gradients, explicit plane distances, CST axes and phase sign for physics changes.
2. Exercise the actual Windows launcher, an example training run, Appearance,
   custom objectives and the guide. Saved figures must match the chosen palette;
   field interpolation must not smooth physical metasurface phase cells.
3. Update version, changelog, installation instructions and public verification.
4. Build the small launcher and test with external CPU/GPU Python environments.
5. Publish only the clean application source and intended release assets. Exclude
   local results, credentials, environment selections and machine-specific logs.
6. Attach the Windows ZIP, executable, clean source ZIP and their SHA-256 checksums
   to a versioned GitHub release. The repository's automatic source archives also
   provide the exact tagged application source.

## Project metadata

Suggested description: Optical neural network (ONN) and D2NN design lab: train
metasurfaces, optical logic gates and OAM with RS/ASM propagation, PyTorch and CUDA.

Relevant topics: optical-neural-network, onn, d2nn, diffractive-neural-network,
metasurface, optical-logic-gates, rayleigh-sommerfeld, angular-spectrum-method,
computational-optics, photonics, optical-computing, pytorch, cuda, terahertz.
Use accurate, descriptive terminology. Search indexing and project popularity
are controlled by GitHub, search engines and the community.

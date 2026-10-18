# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""Offline help and experiment walkthroughs, kept separate from interface code."""
from html import escape


def document(body):
    """Readable help that follows the desktop's dark palette."""
    return ('<html><head><style>body{color:#dce6f3;font-family:Segoe UI;}'
            'h2,h3{color:#78f6df;}p,li{line-height:145%;}code,pre{color:#a6e9e0;}'
            'a{color:#78f6df;}th,td{padding:6px;text-align:left;}'
            '</style></head><body>'+body+'</body></html>')


QUICK_START = document('''
<h2>Your first trained surface</h2>
<p>Start with <b>Gaussian focus</b> in the top menu. It supplies a complete optical
system, incident beam, target and Python objective.</p>
<ol>
<li><b>Optical system:</b> choose the number of cells, cell pitches in µm, frequency
in THz and number of surfaces. For N surfaces set N+1 individually labeled distances in wavelengths:
input → surface 1 → … → surface N → output.</li>
<li><b>Input &amp; target:</b> choose the aperture and incident beam, then the desired
output. Aperture transmission multiplies the complex illumination.</li>
<li><b>Python objective:</b> insert a suitable loss example. Target settings prepare
tensors for your code; changing the target does not rewrite the loss.</li>
<li><b>Preview / validate loss:</b> check the initial fields, loss and gradients.
Preview does not optimize or export a trained design.</li>
<li><b>Launch training:</b> follow normalized loss, wavefronts, metrics, elapsed time
and ETA. Choose another input, surface, output view or polarization while it runs.</li>
<li><b>Results archive:</b> the app automatically exports the final phases, fields,
figures and configuration. Double-click a plot or open the results folder.</li>
</ol>
<h3>A useful first change</h3>
<p>Try 150 epochs, then move the focus with Target x/y and train again. Keep the
intensity objective. Save experiment records your settings and Python code.</p>
<h3>Two polarizations</h3>
<p>Set Scalar polarizations to 2. Use <b>Editing</b> to configure each channel's
input, target and loss. Geometry and training settings are shared. Each channel
has independent phase maps; its physical basis comes from your unit cell.</p>
<h3>Computing and saving</h3>
<p>Auto uses CUDA when available and falls back to CPU. The compute status and run
summary show the device actually used. <b>Stop &amp; export</b> saves the current
solution and marks the run interrupted. Results live beside the launcher in
<code>results/</code>, with a unique folder per run.</p>
<h3>Please cite the paper</h3>
<p>If you use WaveForge, its code, examples or research method in your work,
please cite Nadarloo, Hajiahmadi &amp; Soleimani, <i>Scientific Reports</i> (2026),
DOI: 10.1038/s41598-026-69471-w. The full citation is under Guide → Paper &amp; citation.</p>
''')


GUIDE_TOPICS = {
"Installation & GPU setup": "@file:INSTALL.md",
"Developer setup & modifying the app": "@file:DEVELOPER_GUIDE.md",
"Complete user manual": "@file:README.md",
"Paper & citation": '''
<h2>Please cite the paper when using WaveForge</h2>
<p>If you use this application, its code, examples or research method in your
research, teaching or publications, please cite:</p>
<p>Ali Nadarloo, Mohammad Javad Hajiahmadi &amp; Mohammad Soleimani.
<b>A single passive terahertz metasurface implementing eight Boolean logic gates
through polarization multiplexing.</b> <i>Scientific Reports</i> (2026).
DOI: 10.1038/s41598-026-69471-w.</p>
<p>Made by Ali Nadarloo, School of Electrical Engineering, Iran University of
Science and Technology, Tehran, Iran. Developed from the computational work for
the paper and Ali Nadarloo's master's thesis research.</p>
<p>CITATION.cff and CITATION.bib beside the application provide reusable citation
metadata. The application source is distributed under the MIT license.</p>''',
"Smart controls, themes & color maps": '''
<h2>A workspace that follows the task</h2>
<p>Aperture, illumination, target, solver and optimizer determine which controls
appear. A plane-wave logic experiment hides OAM charge, beam waist, picture paths,
text settings and digit controls. Selecting an OAM beam reveals its charge;
selecting an image target reveals its file selector. Hidden values are preserved.</p>
<p>Target choices follow the aperture. Known built-in objective recipes switch to
the matching target recipe; custom Python is preserved. Review the code after a
task change. Show all controls exposes additional context parameters for custom
Python work.</p>
<p>Optical system lists every plane distance separately, including the final
surface-to-output distance. Changing surface count preserves input/output
distances and fills new inter-surface spacings with 4λ; inspect them before running.</p>
<h3>Appearance</h3><p>Choose Deep space, Nebula violet, Solar amber or Daylight.
Input amplitude, output intensity/amplitude and phase have independent color-map
selectors, including hot, jet and winter. Changes apply to live plots immediately.
The phase scale stays −π…+π. Saved figures use the current theme and maps when
export begins. Field plots interpolate for display; metasurface phase cells remain
unsmoothed. Results archive → Export again with current appearance saves the
completed design in a new folder without retraining. Preferences are saved beside
the app; they do not change any optical parameter.</p>''',
"Four-layer handwritten digit routing": '''
<h2>Recognize 0–9 and send light to the right circle</h2>
<p>Load Advanced • four-layer digit router. It contains four trainable phase
surfaces and a bundled handwritten-digit dataset. Labels are actual digit IDs.
Ten nonoverlapping output circles are labeled 0–9, counterclockwise from the top.</p>
<ol><li>Keep the default 32×32 cells and FFT RS for a first run.</li>
<li>Inspect the five plane distances, especially Surface 4 → output (default 8λ).</li>
<li>Set detector ring radius and circle radius. Both are in wavelengths.</li>
<li>Use batch size 64 and 0 training samples per digit to use the full training set.
Reduce samples per digit or epochs for a quick check.</li>
<li>Keep the Digit recognition / ten detectors loss. Loss lab explains it.</li>
<li>Train; monitor training and held-out accuracy. Select each held-out digit in
the live sample selector and inspect the numbered circles.</li>
<li>Inspect digit_confusion.png, digit_predictions.csv, digit_accuracy.png and the
phase exports. Each prediction is based on integrated circle power.</li></ol>
<p>Held-out inputs never enter gradient updates. Accuracy is evaluated every five
epochs and at the beginning/end. Final representative fields contain one held-out
sample per digit; prediction tables contain every evaluated sample. This example
uses the bundled UCI/scikit-learn images, not MNIST or generated font glyphs.</p>''',
"Geometry, units & polarization": '''
<h2>Geometry and units</h2>
<p>Cells along x/y count the rectangular unit cells. Pitch is the center-to-center
distance in µm, so the surface width is Nx × pitch x and height is Ny × pitch y.
Frequency is in THz; the app computes λ = c/f. Spacings, beam waist, aperture
radius and target offsets are entered in λ.</p>
<p>One surface needs two spacings; two surfaces need three. Each spacing must be
positive. Output window scale enlarges the observation area for Dense RS and ASM;
FFT RS uses scale 1. Arrays are ordered [y,x], with pixel-centered coordinates.</p>
<h3>Independent scalar channels</h3>
<p>Polarization 1 and 2 share cells, propagation distances, frequency and optimizer.
Inputs, beams, targets, detector settings, Python objectives and phases can differ.
Switch Editing before changing channel-specific settings. Training processes each
channel in sequence. The scalar model omits coupling and polarization conversion;
your cell geometry establishes the actual polarization basis.</p>''',
"Inputs, pictures, text & beams": '''
<h2>Construct the incident field</h2>
<p><b>Aperture × beam</b> forms the input complex field. Open, circle, double slit,
logic gates, text/number and image/array apertures are available. Logic gates create
a batch of input cases; use the live sample selector to inspect each.</p>
<p>Raster pictures are converted to grayscale, alpha-multiplied and resized. White
transmits and black blocks. The first frame is used for GIF. Raster rows are
flipped to display upright on Cartesian axes. Text keeps its aspect ratio.</p>
<p>NPY, CSV and TXT arrays must match [Ny,Nx]. Their row order is preserved.
Complex input arrays specify amplitude and phase; real inputs must be nonnegative.
Target arrays are nonnegative intensity. NPY loading disables pickle.</p>
<h3>Illumination</h3>
<p>Plane wave has uniform amplitude. Gaussian waist is the field amplitude 1/e
radius. OAM uses an LG p=0 radial profile with integer charge. Bessel–Gauss uses
J0(4r/waist) with a Gaussian envelope. All support x/y beam tilts in degrees.</p>
<h3>Output targets</h3>
<p>Focus, image and text supply a normalized intensity target. OAM supplies a
unit-norm complex field. Logic detectors supply left/right masks and truth labels.
Choose the matching Python loss when changing tasks.</p>''',
"Propagation & numerical accuracy": '''
<h2>Choose propagation</h2>
<p><b>Dense RS</b> uses the original paper's sampled Rayleigh–Sommerfeld matrix.
Use the unedited paper presets for reproduction. Memory grows rapidly with cell count.</p>
<p><b>FFT RS</b> computes linear convolution of the same sampled kernel on equal
grids. It is efficient for large arrays; output scale must be 1.</p>
<p><b>ASM • evanescent decay</b> propagates a padded angular spectrum and retains
decaying evanescent components. <b>ASM • propagating only</b> removes them.
Both expand constant-phase cells into fine field samples. Default 4 samples per
cell means four samples along each cell axis, with a 3× padded FFT window.</p>
<p>Apply recommended propagation shows a recommendation based on your geometry
and task. It changes the solver only when you click the button.</p>
<h3>Check a design before fabrication</h3>
<p>Compare increased samples per cell, padding and observation window. Inspect
whether the useful field or target is clipped. Short distances and high-angle
features need particular care. A scalar phase-only design does not include
unit-cell dispersion, loss or full-wave interactions.</p>
<p>All methods use exp(+iωt), with outgoing propagation exp(−ikR). An ASM field
grid is finer than its trainable cell phase grid.</p>''',
"Training, live plots & speed": '''
<h2>Training controls</h2>
<p><b>Adam</b> is the default. AdamW adds decoupled weight decay; SGD with momentum,
RMSprop and Adagrad are alternatives. Learning rate controls update size; epochs
control update count. Momentum applies to SGD/RMSprop. Keep paper settings when
comparing to the original results.</p>
<p>Zero initialization reproduces the paper. Small random phases help break
symmetry for OAM conversion. Seed controls reproducibility; hardware and library
versions can still change a long floating-point optimization trajectory.</p>
<h3>Read the live view</h3>
<p>The trace defaults to loss(epoch)/max(|initial loss|, 10⁻¹⁶). Switch to raw loss
when needed. A negative objective may become more negative as it improves; a ratio
is not an accuracy score. All epoch values are saved. The graph limits visible
points and refreshes previews at a modest rate to keep the interface responsive.</p>
<p>Choose a sample, surface, polarization and output intensity/phase/amplitude.
Live fields are evaluated after updates; the recorded loss is measured before
each update. Custom scalar metrics also appear during training.</p>
<h3>Compute</h3>
<p>Auto and CUDA preferred try the GPU and fall back to CPU if CUDA is unavailable.
CPU explicitly selects CPU. Actual compute appears in the run summary. Reduce grid
size, padding, sampling or layer count if memory is insufficient. Use FFT RS for
large compatible sampled-kernel problems. Exports retain full resolution.</p>''',
"Write your own Python objective": '''
<h2>Start with a recipe</h2>
<p>In Examples, inspect a loss recipe and insert it into Python objective for the
currently edited polarization. The Parameters tab lists every prepared tensor
and function with units and shapes.</p>
<p><code>def loss(ctx):</code> must return one finite real scalar PyTorch tensor
connected to the phases. The field is complex [batch,height,width]; intensity is
|field|². Use torch operations on the existing device. Keep .item(), .detach(),
NumPy and discrete argmax out of the differentiable loss.</p>
<p>For intensity fitting use intensity/total_power[:,None,None] against
target_intensity. For complex shaping compare to target_field through overlap.
Circular phase errors avoid a jump at ±π. Detector fractions are normalized to
sampled output power; they differ from input-to-output efficiency.</p>
<h3>Extra planes and reports</h3>
<p><code>ctx['at_z'](z_m)</code> returns a differentiable field at a distance in
metres from just after the last surface. Reuse a fixed finite set of distances.
Coordinates and wavelengths in Python are in metres; phases are radians.</p>
<p>Optional <code>metrics(ctx)</code> returns named scalars. Optional
<code>plots(ctx)</code> returns named real 1D/2D arrays for saved figures. Metrics
update live; plots are evaluated on final fields. Python executes locally with
your usual privileges. Preview checks the returned loss and gradients.</p>
<h3>ASM tensor sizes</h3>
<p>dx/dy and pixel_area describe the fine field grid; cell_dx/cell_dy describe
physical unit cells. surface_phases stays at [Ny cells,Nx cells]. Input/output
fields and layer fields use the fine grid. Avoid assuming these shapes match.</p>''',
"Results, CST & continuing a run": '''
<h2>What gets saved</h2>
<p>Each run includes config.json, loss.py, summary.json, fields.npz, phases.npz,
training_loss.csv and figures. Output intensity/phase and user-defined plots are
saved for the relevant task. Logic tasks include a truth table. Dual runs have
separate Polarization_1 and Polarization_2 directories.</p>
<p>Use Save experiment to keep an editable configuration. To continue a design,
load its config, Import phases from phases.npz, adjust epochs or learning rate,
then train. This starts a new optimizer; it does not restore Adam's moment state.</p>
<h3>CST: −z toward +z</h3>
<p>The default illuminates surfaces in the xy plane from −z toward +z.
CST_phase_layer_N.txt contains x_um, y_um, z_um, phase_rad, phase_deg, amplitude.
Columns advance negative to positive x; rows advance negative to positive y.
There is no mirror, transpose or conjugation. Surface z includes input spacing.</p>
<p>Transmission is t=exp(+iφ) with the app's exp(+iωt) convention. Match your CST
cell library's transmission reference plane. The export pop-up and saved
CST_EXPORT_README.txt explain the convention. Alternative axes apply a recorded
right-handed rotation.</p>
<p>Paper-compatible best_meta_layer_N.txt uses indexed i,j,phase,amplitude columns
and phases [0,2π). Despite the historical filename, exports are the final iterate,
not automatic best-epoch selection. Stop &amp; export records an interrupted run.</p>''',
"Troubleshooting & paper reproduction": '''
<h2>Common fixes</h2>
<p><b>Loss is flat:</b> preview the loss; check the target and objective match,
try small random phase initialization for OAM, and inspect whether light reaches
the target window. Reduce learning rate if optimization oscillates.</p>
<p><b>No gradient / invalid loss:</b> return a real scalar torch tensor without
detaching. Avoid NumPy conversions and discrete decisions in the loss. Keep
denominators positive and check that ROI masks contain pixels.</p>
<p><b>Unexpected image:</b> confirm array [y,x] ordering, the difference between
amplitude and intensity, and raster versus numeric row conventions.</p>
<p><b>Memory limit:</b> reduce cell count or choose FFT RS/ASM, then check the
recommendation. ASM sampling and padding both increase memory.</p>
<p><b>Cannot save:</b> extract the whole portable folder to a writable location;
keep _internal and app_source beside WaveForge.exe.</p>
<h3>Reproduce the paper</h3>
<p>Load Paper • two polarizations and retain 32×32 cells, 150 µm pitches, 1 THz,
one surface, [1.2,10.8] λ spacings, Dense RS, zero initial phases and 3,000 Adam
epochs at 0.008. The preset includes the original apertures, class weights,
detectors and loss. Channel 1 trains OR/AND/XOR/BUFFER; channel 2 trains
NOR/NAND/XNOR/NOT. Hardware-dependent floating-point changes can slightly alter
final phases. VERIFICATION.md gives the measured agreement.</p>'''
}


# Each walkthrough loads a complete configuration through the existing preset system.
WALKTHROUGHS = {
"Advanced • four-layer digit router": "Advanced · Four shared phase surfaces classify handwritten digits 0–9 into ten output circles. Start with 100 epochs and batch size 64. Inspect held-out accuracy and confusion, not just training accuracy. Adjust the last plane distance and detector ring/circle radii before training. Loss lab explains the batch loss and detector tensors; the Guide contains a full walkthrough.",
"Gaussian focus": "Beginner · Gaussian illumination → concentrated spot. Preview, train 150 epochs, then inspect output intensity and intensity_error. Move Target x/y to place the spot elsewhere; keep the intensity loss.",
"Paper • Polarization 1": "Paper reproduction · OR, AND, XOR and BUFFER. Keep the preset unchanged and run 3,000 epochs. Select each input in the live view; inspect truth_table.csv and detector fractions after training.",
"Paper • Polarization 2": "Paper reproduction · NOR, NAND, XNOR and NOT, with the original complementary apertures and loss. Run 3,000 epochs; inspect all 14 truth-table cases.",
"Paper • two polarizations": "Dual-channel paper reproduction · Both complete gate groups. Use Editing to inspect each channel's aperture and loss. Launch training once; results contain separate phases and truth tables for each polarization.",
"OAM conversion": "Intermediate · Two surfaces convert Gaussian illumination into an OAM mode. Choose target charge and waist. Keep the complex-overlap loss and small random initialization. Inspect phase winding and target_angular_purity; angular purity and radial-mode overlap measure different properties.",
"Text hologram": "Beginner · Form ONN, a word or a number in output intensity. Change Target text, preview, then train. Keep the intensity-fitting objective. Inspect the target and error map, and compare image shape rather than only brightness.",
"Image hologram": "Beginner · A bundled target image is ready to run. Replace Target image / array with your own picture to shape its intensity at the output. The image is resized; numeric targets must match the output grid. Preview before training.",
"Double-slit steering": "Intermediate · Move the intensity centroid to x=3λ, y=0. The centroid recipe specifies 3 directly in its Python code; edit that constant for another displacement. Inspect centroid_x_wavelengths as well as the output pattern: centroid alone does not constrain beam shape.",
"Two-plane focus": "Advanced · Concentrate power in a central ROI at 5λ and 12λ after the final surface. Edit the distances and ROI in the loss if needed. Inspect both near_intensity and far_intensity exports; these planes are measured from the last surface.",
"ASM • subcell OAM": "Advanced · OAM conversion with four samples per cell and 3× padded angular-spectrum propagation. Choose charge, train, inspect output phase and OAM spectrum. Compare higher sampling/padding for convergence; phase files still have one value per physical cell."
}


def walkthrough(name):
    return document('<h2>'+escape(name)+'</h2><p>'+escape(WALKTHROUGHS[name])+
                    '</p><p><b>Load complete example</b> replaces the current settings and loss. '
                    'Save your experiment first if you want to keep edits.</p>')

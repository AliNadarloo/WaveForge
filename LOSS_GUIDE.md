<!-- Made by Ali Nadarloo. -->
# Loss lab: designing your own optical objective

## 1. What the app optimizes

The trainable values are the transmission phases of the metasurface cells.
The app creates the input field, propagates through every surface, then calls
your Python `loss(ctx)`. PyTorch differentiates that scalar back to the phases;
the optimizer updates the phases to make the scalar smaller.

You define what “better” means. A target selector prepares helpful tensors; it
does not impose a loss on custom code. For example, selecting an OAM target
prepares a helical complex target field. You still need a phase-sensitive loss
if you want the output to have that phase.

1. Load a complete experiment close to your task.
2. Set the input, output target and every plane distance.
3. Open Python objective and insert an appropriate recipe.
4. Read its comments; edit one part at a time.
5. Click Preview / validate loss. It runs a forward pass and checks gradients.
6. Train for a short run and inspect the actual fields and metrics.
7. Increase epochs only after confirming that the objective measures your intent.

The current editing selector controls which polarization's code you are editing.
Save experiment stores the code and all optical settings together. Loading a
complete preset replaces both; inserting a recipe replaces only the active
polarization's code. Custom code is preserved when changing task selectors.

## 2. The minimum contract

```python
def loss(ctx):
    field = ctx["field"]
    intensity = field.abs() ** 2
    # One real scalar Tensor; lower must mean better.
    return intensity.mean()
```

This is a valid differentiable example, but minimizing total intensity may simply
send light outside the sampled output window. A valid program can describe the
wrong optical goal. Always inspect its physical meaning.

The result must be finite, real and contain one value. Use normal PyTorch tensor
operations so the result remains connected to the surface phases. `torch` and
`math` are already available. Code executes locally with normal Python privileges.

## 3. Shapes, coordinates and units

- `field`: complex `[B,H,W]`, one output field per input sample.
- `intensity`, `amplitude`, `phase`: same shape; intensity = |field|².
- `phase` and `surface_phases`: radians; `surface_phases` is a list of `[Ny,Nx]`.
- `target_intensity`: real `[H,W]`, sum = 1. This is intensity, not amplitude.
- `target_field`: complex `[H,W]`, sum |target_field|² = 1.
- `total_power`: `[B]`, sum of output intensity over the sampled window.
- `input_power`: `[B]`, sum of incident intensity on the corresponding field grid.
- `X,Y`: real `[H,W]`, output pixel-center coordinates in **metres**.
- `wavelength`, `dx`, `dy`: metres. `pixel_area = dx*dy`, in m².
- `cell_dx,cell_dy`: physical cell pitches in metres.

The UI uses THz, µm and wavelengths. Python lengths use metres. Thus a radius of
2λ is `2 * ctx["wavelength"]`, not `2`.

`B` is the batch dimension. For a batch scalar to broadcast over pixels, use
`[:,None,None]`. Reduce spatial axes with `.sum((-2,-1))` and reduce independent
samples with `.mean()`. A `[H,W]` target broadcasts over all B inputs.

ASM uses multiple field samples per cell, so its H/W can exceed physical cell
counts. Field and layer tensors use that fine grid; surface phases remain one
parameter per cell. Use tensor shapes and the provided coordinates instead of
hard-coded dimensions. The Parameters tab contains the complete context reference.

## 4. Beginner: match an intensity picture or focus

```python
def loss(ctx):
    predicted = ctx["intensity"] / ctx["total_power"][:,None,None]
    target = ctx["target_intensity"]
    error = ((predicted - target) ** 2).mean()
    # Scale so values stay convenient when changing grid dimensions.
    return error * predicted[0].numel() ** 2
```

Use Focus, Image intensity or Text intensity as the target. White picture pixels
mean high desired intensity. The loss compares shapes, independently of overall
brightness. A small shape error does not prove high transmitted efficiency.
Report or penalize efficiency separately when you need it.

```python
def metrics(ctx):
    return {"sampled_power_ratio":
            (ctx["total_power"] / ctx["input_power"]).mean()}

def plots(ctx):
    predicted = ctx["intensity"][0] / ctx["total_power"][0]
    return {"intensity_error": predicted - ctx["target_intensity"]}
```

All common scalar metrics update during training. Custom plots are generated
after training; they must be real 1D or 2D arrays. Export complex fields using
`fields.npz`; convert to intensity or angle for a custom figure.

## 5. Detector efficiency and leakage

```python
def loss(ctx):
    radius = 2.0 * ctx["wavelength"]
    mask = ctx["roi"](0.0, 0.0, radius)
    collected = (ctx["intensity"] * mask).sum((-2,-1))
    efficiency = collected / ctx["input_power"]
    leakage = (ctx["total_power"] - collected) / ctx["input_power"]
    return (-efficiency + 0.2 * leakage).mean()
```

The minus sign turns maximization into minimization. This loss may be negative;
more negative can be better. A normalized live loss is loss/|initial loss|, not
an accuracy score. Efficiency uses the input as denominator; detector fractions
in the original logic recipe use sampled output power as denominator.

## 6. Phase, OAM and complex overlap

Matching a doughnut intensity alone does not determine OAM charge. Select OAM mode
and use a complex-field objective:

```python
def loss(ctx):
    overlap = (ctx["target_field"].conj() * ctx["field"]).sum((-2,-1))
    normalized_overlap = overlap.abs()**2 / ctx["total_power"]
    return (1 - normalized_overlap).mean()
```

This compares radial profile and phase up to one global phase. The target is
already unit-norm. Use small random initial phases for the OAM examples: some
symmetric zero-phase starting points have almost zero overlap and poor gradients.

For phase-only matching, avoid subtracting wrapped angles directly:

```python
def loss(ctx):
    delta = ctx["phase"] - torch.angle(ctx["target_field"])
    return ((1 - torch.cos(delta)) * ctx["target_intensity"]).sum((-2,-1)).mean()
```

The cosine handles the ±π wrap. Weighting by target intensity reduces pressure
on dark pixels where phase is not useful. The OAM spectrum helper reports angular
power around the configured target center. Angular purity does not test radial
profile; it differs from full complex overlap.

## 7. Logic gates

Load the paper preset for the gate group you need. `detector_fractions` has shape
`[B,3]`: left detector, right detector, outside both. `labels` contains expected
truth values. `label_values` is the same constant data as a Python list.

The supplied Paper logic gates recipe reproduces the original per-case weights,
desired fraction and leakage penalty. Keep it and the preset geometry unchanged
when comparing to the paper. Do not put complementary gates with identical input
apertures in the same scalar channel: use Polarization 1 and 2.

For a custom objective, each row's correct detector depends on its truth label.
Avoid optimizing only the first input case. Evaluate all truth-table cases, then
inspect both accuracy and contrast; accuracy alone can hide weak separation.

## 8. Advanced: recognize digits with four layers

Load Advanced • four-layer digit router. Inputs are handwritten 0–9 amplitudes,
not generated text. The dataset is bundled and attributed in examples/digits.
Its deterministic split contains 1,438 training and 359 held-out samples.

The four surfaces share a single phase design across all digits. The app shuffles
training samples and updates in mini-batches; it never updates on held-out images.
An epoch is one pass over the selected training set. Held-out accuracy is evaluated
every five epochs and at the first/final epoch. You can reduce training samples
per digit for a quick experiment; 0 uses all available training samples.

```python
def loss(ctx):
    # [batch,10], integrated power in each numbered output circle / total output.
    fractions = ctx["digit_fractions"]
    correct = fractions.gather(1, ctx["labels"][:,None]).squeeze(1)
    return -torch.log(correct.clamp(min=1e-10)).mean()
```

The loss rewards power in the expected circle, including the penalty for sending
power elsewhere. Recognition chooses the circle with highest integrated power.
`digit_detector_masks` is `[10,H,W]`; `digit_detector_centers` is `[10,2]` in metres.
Digit `labels` are integers 0–9, unlike the binary labels in logic-gate examples.
These extra keys exist only for the Digit detectors target.

The final control under Optical system → Plane distances is the output-plane
distance from the last surface. Change it **before** training to train for a
different output plane. Detector ring radius and circle radius are separate
parameters. Every circle must fit and must not overlap another circle.

Live samples show one held-out input per digit, with numbered circles on output.
Final reports include all predictions, the held-out confusion matrix, accuracy
curves and physical detector coordinates. Only representative fields are exported
to keep file size and plotting time reasonable. Held-out accuracy is evidence of
generalization within this dataset; it is not a guarantee for every handwriting
style or a replacement for a full-wave device model.

## 9. Combine objectives and regularization

```python
def loss(ctx):
    prediction = ctx["intensity"] / ctx["total_power"][:,None,None]
    fit = ((prediction - ctx["target_intensity"])**2).mean()
    fit = fit * prediction[0].numel()**2
    smooth = sum(
        (1 - torch.cos(p[:,1:] - p[:,:-1])).mean()
        + (1 - torch.cos(p[1:,:] - p[:-1,:])).mean()
        for p in ctx["surface_phases"])
    return fit + 0.01 * smooth
```

Inspect each term's size in metrics before choosing weights. A coefficient of
0.01 has no universal meaning when other terms change scale. Circular smoothness
respects phase wrapping. Ordinary squared differences of wrapped phases can
penalize physically equivalent values close to −π and +π.

## 10. Train at extra observation planes

```python
def loss(ctx):
    wl = ctx["wavelength"]
    near = ctx["at_z"](5 * wl)
    far = ctx["at_z"](12 * wl)
    mask = ctx["roi"](0., 0., 2 * wl)
    def concentration(field):
        intensity = field.abs()**2
        return (intensity * mask).sum((-2,-1)) / (intensity.sum((-2,-1)) + 1e-16)
    return (2 - concentration(near) - concentration(far)).mean()
```

`at_z` starts immediately after the **last** surface, not at the input or current
output plane. Its distance is in metres. The extra-plane operators are cached;
use a fixed finite set of distances. Return near/far intensity maps in `plots(ctx)`
to confirm that both terms improve.

## 11. Common errors and a practical debugging sequence

**No gradient:** do not use `.item()`, `.detach()`, `.cpu().numpy()` or a Python
float as the returned loss. Do not use `argmax` as the quantity to optimize.
Those are useful for reports; replace optimization decisions with smooth losses.

**Non-scalar result:** call `.mean()` or a deliberate weighted sum after reducing
spatial dimensions. Do not accidentally average the batch into one field first:
independent inputs represent separate illuminations, not a coherent superposition.

**Complex loss:** use `abs()**2`, `.real` where physically justified, or a real
phase error. Return one real scalar, not the complex overlap itself.

**Wrong device:** create constants with `device=ctx['field'].device` or use
`torch.ones_like(...)`. Avoid hard-coded `.cuda()` because CPU fallback must work.

**NaNs:** check zero denominators, log of zero, empty masks and very large weights.
Use `.clamp(min=...)` or a small positive denominator where physically appropriate.

**Flat loss:** check the input is illuminated, the target is inside the window,
the loss matches the intended task and initialization is suitable. Preview only
checks that gradients exist and are finite; it cannot decide whether the goal is
physically achievable or whether gradients are large enough to be useful.

**Good loss, wrong pattern:** export the exact term your loss measures. A centroid
loss can produce several lobes with the right centroid. An intensity loss cannot
determine the phase. A sampled-window power fraction can ignore light outside it.

Start with a known recipe, run a short optimization, inspect both the field and
the metric, then add one term at a time. Save each useful configuration so it is
easy to return to a working experiment.

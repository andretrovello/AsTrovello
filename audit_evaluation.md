# Assessment of the Claude Code audit (`CLAUDE.md`)

An evaluation of the repository audit, written from the perspective of the
design discussions that produced the pipeline. The auditor had the repository
and `astrovello_pipeline_documentation.md`; it did not have the months of
debugging history in which each decision was made and each number measured.
That difference matters in a handful of places, and is noted where relevant.

**Summary judgement: the audit is of high quality and should be acted on.**
I verified independently every finding I could check from first principles, and
those all held. Two findings are more serious than their placement suggests,
one is overstated, and a few need context that the auditor could not have had.

---

## 1. Findings I verified independently and agree with

### A2 — `NATPXAR` written but never read (0.61% colour systematic)

**Agree, and this is more important than its "A2" position implies.**

Reproduced the arithmetic exactly:

| | arcsec/px | area (arcsec²) |
|---|---|---|
| WCS (true drizzled) | 0.039620 | 0.00156974 |
| `config.py` constant | 0.039500 | 0.00156025 |

Dividing by the smaller area inflates the HST fluxes by **0.61%**.

Why this matters more than a sub-percent number suggests: it is a **pure
colour term**. It affects the five HST bands and none of the IR bands, which
is precisely the class of error this whole pipeline redesign was built to
eliminate. Throughout our work the guiding rule was *"an error that affects
some bands and not others is a colour error, and a colour error becomes an age
error."* A uniform 0.61% would be harmless; a 0.61% offset between the UV/optical
and the IR is not.

It is also an unusually pointed finding because `NATPXAR` exists **only** to
prevent this. The header key was introduced specifically so the conversion
would stop trusting the config constant. Writing it and then not reading it
leaves the pattern half-implemented — the documentation describes the intended
design, the code implements the old one.

The auditor's proposed fix is right, including the detail that
`bin_for_convolution` returns early when `factor <= 1`, so the fallback is
load-bearing:

```python
fits_header.get('NATPXAR', self.config["pixel_scale_arcsec"]**2)
```

### B2 — `inspect_kernel` measures after normalisation, and is therefore blind

**Agree, and this is the finding with the most direct consequence for the
dissertation.**

Reproduced the blind spot on a synthetic kernel matching the real one
(all-negative, sum −0.4958):

```
raw kernel      : sum = -0.4958   negative power = 100.00%
after / sum     : sum = +1.0000   negative power =   0.00%
healthy kernel  : sum = +1.0000   negative power =   0.00%
```

A pathological kernel and a healthy one give **the same reading**. Dividing by
a negative sum flips every sign, so 100% negative becomes 100% positive and the
metric reads zero.

This directly undermines a number currently in the documentation. Section 4
cites *"Measured: 0.00% for the five HST → IRAC kernels"* as evidence of kernel
health, and compares it favourably to Aniano's canonical `W₋ = 0.07`. That
comparison is not valid as measured, because the quantity is computed after a
transformation that guarantees the answer.

**Context the auditor did not have, which makes this less alarming than it
sounds:** during the debugging we *did* measure the raw kernels, with
`diagnostico_etapa1b.py`, reading the files on disk before any normalisation.
That is where the "100% negative power, sum −0.50" signature came from, and it
was one of two independent indicators that led to the grid diagnosis. So the
v1.x finding was real and correctly obtained. What broke is the *in-pipeline*
check, which was written later and inadvertently placed after the
normalisation.

The auditor's conclusion is nonetheless the right one and I endorse it without
reservation: **do not cite 0.00% in the dissertation until the metric is
measured pre-normalisation.** The fix is one line — call `inspect_kernel` on
`kernel_data` rather than `kernel_norm`.

A secondary point in the same finding also deserves attention: the sign-flip
comment in the code (*"PyPHER sometimes returns the kernel with the global sign
inverted, and that is cosmetic"*) is too relaxed. That comment is mine. It was
written when the sign inversion appeared alongside the grid bug and normalising
happened to recover a usable kernel. An all-negative kernel summing to −0.5 is
a symptom, and treating it as cosmetic is the kind of assumption this project
has repeatedly been punished for.

### A3 — off-by-one in `crop_to_mask_bbox`

**Agree.** Reproduced: `coords.max(axis=0)` returns an inclusive index used as
an exclusive slice bound.

```
valid block 6×6 (36 px)
mask[ymin:ymax, xmin:xmax]     -> (5,5), 25 px   (loses 11)
mask[ymin:ymax+1, xmin:xmax+1] -> (6,6), 36 px   (correct)
```

Called twice per cube, so two rows and two columns of real signal are lost at
the top/right edge. Small in absolute terms, trivially fixed, and the kind of
thing that is much cheaper to correct now than to explain later.

### A1 — datacube WCS has `CDELT = 1 deg/px`

**Agree, and the severity rating is correct.**

This is a defect we knew about in a weaker form. During the cube-building work
the astropy warning `cdelt will be ignored since cd is present` appeared, and I
recorded it as a pending item (`use w_2d.sub([1,2,0])` rather than the manual
element-by-element copy) but classified it as *latent* — "nothing uses the
cube's WCS today". The auditor establishes that it is not latent at all: the
written header claims 1 degree per pixel, a factor 4800 error, and any sky
coordinate or angular scale derived from those cubes is wrong.

My earlier classification was the error. It rested on "nothing downstream reads
this yet", which is an assumption about future use, not a property of the data.
The auditor is right to promote it to critical and to put it first in the fix
order. The bare `except Exception: continue` that hides the failure should go
with it.

### B3 — the grid guard is skipped exactly when it matters

**Agree.** The guard in `create_convolvedFITS` exists specifically as the
safety net against the v1.x grid bug returning. If `PIXSCALE` is missing and
the WCS fallback also fails, `k_px = None` and the check is silently skipped —
and the stale v1.x kernels sitting in `Output/PSF_Kernels/` are exactly the
files with no `PIXSCALE`. A guard that cannot fire on its own motivating case
is not a guard. Making a missing `PIXSCALE` a hard error is correct.

---

## 2. Findings I agree with and consider under-weighted

### B1 — `binary_dilation` with a full-kernel square structuring element

The auditor labels this "CRITICAL for usability" and measures ~1400× slowdown
(29 s versus 0.02 s per band), plus a hard `MemoryError` at 201×201.

**I would go further: this is very likely the explanation for a problem we
spent real time on and never fully resolved.** During the convolution work the
stage was repeatedly slow and at one point was killed by the OS. We diagnosed
that as the FFT cost on 12000×12000 images and fixed it with
`bin_for_convolution`, which was a genuine and necessary fix — memory per array
went from ~2.5 GB to ~0.04 GB. But the stage remained slower than the FFT size
alone justified, and I attributed the residual to general FFT cost. If the
dilation is 1400× off, that residual was probably this, not the FFT.

There is also a specific observation from our logs that now makes sense. After
fixing the JWST invalid-pixel convention, you noted the convolution had become
*faster*. I explained it as the FFT no longer interpolating over a huge invalid
region, which is true but partial: correcting the mask also shrank
`border_mask`, which is the input to this dilation. The dominant term in that
speed-up was probably the dilation, not the FFT.

The proposed fix is exact — a flat `k×k` square dilation is identically `k//2`
iterations of a 3×3 square — and the auditor verified equivalence with
`np.array_equal`. This should be second in the fix order, as proposed, and I
would argue it is the single highest value-per-line change in the list.

### B6 — `min_blur` is global rather than per-survey

**Agree, and the reasoning is sound.** Taking the minimum required blur over
*all* bands and applying it to *every* survey means the HST grid is constrained
by the IRAC1↔IRAC2 pair — a pair on a different survey's grid that is then
skipped as unresolvable. Computing it per survey would give HST bin 14 instead
of bin 5, cutting the convolution to ~13% of the pixels while still leaving
3.1 px/FWHM on the master.

This is a design flaw I introduced. When I wrote `derive_bin_factors` the
argument was that the binning factor is a property of the *target*, not of the
source survey — which is correct and was the fix for a real bug. But I then
implemented "property of the target" as a single global minimum, when the
correct reading is "property of the target *as seen from each source grid*".
The auditor caught the gap between the principle and its implementation.

One caveat before applying it: bin 14 puts the HST convolution grid at
0.5547″/px, and the resulting 3.1 px/FWHM on the master is above Nyquist but
tighter than the current 8.7. It is safe by the criterion we adopted, but worth
running `--preflight_only` to see the full table before committing, since the
margin for the HST pairs themselves also changes.

---

## 3. Findings that need context the auditor did not have

### The r50 "~1.06" tolerance (raised under B2)

The auditor notes the documentation reports the ratio as "~1.06" and treats it
as passing *"without stating a tolerance"*, and suggests reporting Aniano's `D`
statistic instead.

**The criticism of the presentation is fair; the implied doubt about the result
is not.** The context: the ratio was 0.08 before the fix and ~1.06 after. The
diagnostic value is in the factor-of-13 change, not in the residual 6%. A test
that moves from 0.08 to 1.06 has demonstrated that the kernel went from doing
almost nothing to doing approximately the right thing.

That said, the residual 6% does deserve a stated interpretation rather than
being waved through. My reading is that it is discretisation: on the 0.198″/px
convolution grid the HST PSF is ~0.4 px across — effectively a delta function —
so the r50 of the convolved result is quantised at the level of a fraction of a
pixel. The honest formulation for the dissertation is to state that, and to
give the tolerance explicitly rather than implying one.

Reporting `D` as well would be strictly better and I endorse the suggestion.

### "Binning by mean has no aliasing and no loss of information" (Sect. D)

The auditor calls this overstated and recommends softening it before a referee
sees it, on the grounds that a box filter's sinc transfer function does not cut
off cleanly at Nyquist.

**Agree, and the correction is mine to own.** I wrote that sentence. It is
right in the sense that matters operationally — area-exact averaging is
categorically better than point sampling, and there is no *decimation* aliasing
of the kind that motivated the change — but "no aliasing" as an absolute claim
is wrong. A box filter does alias weakly. The formulation should be something
like: *"area-exact averaging, which is the correct operation for a pixel
integral and does not introduce the decimation aliasing that point sampling
would; the box filter's transfer function is not an ideal low-pass, so weak
aliasing remains, negligible at the sampling factors used here."*

### "The doc claims guards that do not exist" (Sect. D table)

Three rows of that table are accurate and should be fixed: the "raises" claims
in Sect. 10 (both the matched/unmatched overlap and the missing-kernel case)
and the preflight master-grid check that only prints.

Worth recording *how* that drift happened, because it is instructive: those
guards were specified in conversation and written into the documentation as the
design, and the corresponding code was drafted but in at least one case the
final version that landed printed rather than raised. The lesson the auditor
draws in Sect. 7 — *"when a behaviour changes, update that document in the same
commit"* — is the right one, and applies in both directions.

---

## 4. Where I would qualify or disagree

### C3 — `reproject_interp` uses bilinear by default

The auditor is factually right that `reproject_interp` defaults to bilinear and
that the documentation's Nyquist argument implicitly assumes something
higher-order. Stating the interpolation order explicitly is a fair request.

**But I would resist the implication that this is a defect to fix.** At
2.29 px/FWHM the field is sampled above Nyquist, and the additional smoothing
from bilinear is small compared to the 1.72″ PSF the image already carries. We
tested the related question — whether `reproject_adaptive` would recover SNR —
and found it would not, because the kernel's correlation area (~4200 native px)
is ~12× larger than an output pixel (~350 native px), so the pixels being
averaged share the same noise realisation.

The right action is documentation, not code: state the interpolation order and
note that the choice was tested. Changing to `order='biquadratic'` is defensible
but should be measured before being adopted, not assumed better.

### E4 — "the current code cannot run against the current data layout"

Accurate as a statement about the repository. Worth noting for the reader that
this is a branch-state artefact rather than a pipeline defect: you have been
running these stages successfully throughout, with directory names that match
your local configuration. It belongs in the packaging group (fix order item 7),
where the auditor put it.

### The framing of Section 6 ("what is working well")

No disagreement — I note only that the list is accurate and worth keeping. In
particular the auditor independently reproduced the r80 master selection from
the PRF files and confirmed the grid fix against Boucaud's Algorithm 1. An
audit that verifies the load-bearing decisions rather than only listing defects
is considerably more useful, and this one did that.

---

## 5. Recommended action

I endorse the auditor's fix order with one adjustment: **promote B2 to the
group with A2**, because it affects a number currently cited as evidence in
dissertation material, and the fix is one line.

| Priority | Item | Why |
|---|---|---|
| 1 | **A1** — cube WCS | Corrupts every cube produced; factor 4800 in angular scale |
| 2 | **B1** — dilation | One line, ~1400×, and probably the unexplained slowness |
| 3 | **A2** — read `NATPXAR` | One line; removes a 0.61% *colour* systematic before SED fitting |
| 4 | **B2** — inspect before normalising | One line; a validation number in the dissertation depends on it |
| 5 | A3, B3, B4, B5 | Correctness, cheap |
| 6 | B6 — per-survey `min_blur` | Real compute win; re-run preflight before committing |
| 7 | C1, C2 — SIP keys | Latent today, silently wrong astrometry when it activates |
| 8 | Documentation reconciliation | The "raises" claims, the "no aliasing" claim, the r50 tolerance |
| 9 | Packaging (E1–E5) | When handing the code to someone else |

Two items the auditor correctly identifies as decisions rather than fixes —
the 15.5″-per-side border loss from the dilation, and whether `--apply_mask`
should default on — are yours to make, and both are worth raising with Paula
rather than settling in code.

---

## 6. Overall

The audit is careful, specific, and reproducible. Every claim I could check
independently held, including the two that cost me something to concede: the
cube WCS being worse than I had classified it, and the `inspect_kernel` blind
spot undermining a number I had written into the documentation as evidence.

Its principal blind spot is the inverse of its strength: having only the
repository and the documentation, it cannot distinguish a decision that was
tested and deliberately taken from one that was never considered. C3 and the
r50 tolerance are the two places where that shows. Neither changes its
conclusions.

The most valuable thing in it is not any single defect but the pattern the
defects share. `NATPXAR` written but not read, guards documented but printing
instead of raising, a metric measured after the transformation that hides what
it looks for — all three are cases where the *intent* was recorded correctly
and the *implementation* drifted. That is precisely the failure mode this
pipeline was rebuilt to eliminate, appearing one level up: no longer in the
data, but in the relationship between the code and its own documentation.

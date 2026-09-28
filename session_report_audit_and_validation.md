# AsTrovello — audit corrections and validation, session report

**Galaxy:** NGC 1087 · **Configuration:** PHANGS-HST + PHANGS-JWST, master `f2100w`
**Branch:** `v2-dev/audit-*` · **Date:** September 2026

This report covers one working session: four defects found by the repository
audit were fixed and verified, three validation checks were written and run,
two of them were integrated into the pipeline as abort gates, the vocabulary
was corrected, and the cubes were regenerated.

Every correction was verified against a prediction made **before** the
measurement. That is the organising principle of what follows: each section
states what was expected, what was measured, and whether they agree.

---

## Contents

1. [Summary of results](#1-summary-of-results)
2. [Audit corrections](#2-audit-corrections)
3. [Check 1 — WCS consistency](#3-check-1--wcs-consistency)
4. [Check 2 — PSF matching](#4-check-2--psf-matching)
5. [Check 3 — flux conservation](#5-check-3--flux-conservation)
6. [The 1.06 investigation](#6-the-106-investigation)
7. [In-pipeline gates](#7-in-pipeline-gates)
8. [Vocabulary](#8-vocabulary)
9. [Regenerated cubes](#9-regenerated-cubes)
10. [Files created or modified](#10-files-created-or-modified)
11. [Open items](#11-open-items)

---

## 1. Summary of results

| item | criterion | measured | verdict |
|---|---|---|---|
| Check 1 — WCS consistency (HST↔JWST) | < 0.5 px | **0.079 px** (0.0087″) | PASS, 6× margin |
| Check 2 — PSF matching, 12 pairs | 0.95–1.05 | **1.0000–1.0001** | PASS |
| Check 3 — flux conservation, 13 bands | < 1% | **0.15% median**, 0.84% worst | PASS |
| A1 — cube WCS | scales must agree | 0.110905 vs 0.110905 | fixed |
| A2 — HST pixel area | −0.61% on HST, 0% on IR | −0.604% / +0.000% | fixed |
| A3 — crop off-by-one | +1 px per axis | 1276×1134 → 1277×1135 | fixed |
| B1 — dilation cost | identical result, much faster | bit-identical, minutes → seconds | fixed |

---

## 2. Audit corrections

### A1 — the cube WCS declared 1 degree per pixel

**Defect.** `create_data_cube` built the 3D WCS by copying `wcs.crpix`,
`wcs.crval`, `wcs.cdelt` and `wcs.ctype` field by field from the 2D master. A
FITS header may express the same geometry either as `CDELT + PC` or as a `CD`
matrix. When the source uses `CD` — which the S4G master does — astropy leaves
`wcs.cdelt` at its default `[1.0, 1.0]`, and it was that default which got
copied into the cube.

**Why it went unnoticed.** `PIXAREA` is computed separately, with
`proj_plane_pixel_area`, which reads the full transformation and is therefore
correct regardless of convention. So the cube carried a correct `PIXAREA`
alongside a `CDELT` that was wrong by a factor 4800, and everything that used
`PIXAREA` (the surface-density maps) was unaffected.

**Reproduction, on a synthetic CD-matrix header:**

```
source (CD matrix)     : true area 0.5625 arcsec2,  wcs.cdelt = [1. 1.]
copied field by field  : CDELT = 1.0  ->  3600.0 arcsec/px
expected               :                    0.75 arcsec/px
error factor           : 4800
```

**Fix.** Let astropy write the spatial WCS with `w_2d.to_header()`, then add
the third axis, instead of copying fields. `to_header()` emits whichever
convention is correct for that WCS.

**Verification.** Tested across four configurations (CD and CDELT+PC, with and
without a 30° rotation): the old method fails on both CD cases and passes on
both CDELT+PC cases; the new method passes all four. Sky coordinates from the
3D WCS reproduce the 2D ones to better than 10⁻⁹ mas.

### A2 — the native pixel area came from a stale constant

**Defect.** `PHANGS_Driver.convert2Jansky` divided by
`config["pixel_scale_arcsec"] ** 2`. The header key `NATPXAR`, written by
`bin_for_convolution` precisely to carry the true native area, was never read.

```
WCS (true, drizzled)   : 0.039620 arcsec/px  ->  0.00156974 arcsec2
config constant        : 0.039500 arcsec/px  ->  0.00156025 arcsec2
```

Dividing by the smaller area inflates the HST fluxes by **0.61%**.

**Why it matters more than 0.61% suggests.** It is a pure **colour** term: it
affects the five HST bands and none of the IR bands. A uniform 0.61% would be
harmless; an offset between the UV/optical and the IR is not, because it
propagates into every colour and therefore into age and mass.

**Fix.** Read `NATPXAR` from the header, with the config constant as fallback
(the fallback is load-bearing: `bin_for_convolution` returns early when
`factor <= 1` and writes no key).

**Verification**, comparing the same band before and after:

```
HST  f814w  : 0.993960   (predicted 0.99394)
JWST f2100w : 1.000005   (predicted 1.000000)
```

Both sides of the hypothesis confirmed: the HST drops by exactly the area
ratio, and the surface-brightness bands are untouched. The residual in the
fifth decimal is the 1-pixel misalignment between the two cubes, not a
calibration difference.

### A3 — one row and one column lost per crop

**Defect.** `crop_to_mask_bbox` used `coords.max(axis=0)`, which is an
inclusive index, as an exclusive slice bound.

```
valid block 6x6 (36 px)
mask[ymin:ymax,     xmin:xmax]     -> (5,5), 25 px   loses 11
mask[ymin:ymax+1,   xmin:xmax+1]   -> (6,6), 36 px   correct
```

Called twice per cube, so two rows and two columns of real signal at the
top/right edge.

**Verification**, same configuration before and after:

```
old pipeline : Footprint crop: (2017, 1640) -> (1276, 1134)   947,925 valid px
new pipeline : Footprint crop: (2017, 1640) -> (1277, 1135)   947,959 valid px
```

+1 per axis, +34 pixels of real signal recovered. Only one crop shows the
effect because `--apply_mask` was off, leaving the second crop inactive.

### B1 — the border dilation was ~1400× slower than necessary

**Defect.** `binary_dilation` with a flat `k × k` square structuring element.
scipy builds that footprint explicitly.

**Fix.** A flat `k × k` square is mathematically identical to `k//2` iterations
of a 3×3 one.

**Verification**, on a 600×600 mask:

| k | identical? | old | new | speedup |
|---:|:---:|---:|---:|---:|
| 15 | yes | 0.046 s | 0.004 s | 12× |
| 41 | yes | 0.397 s | 0.005 s | 82× |
| 101 | yes | 5.684 s | 0.007 s | 771× |
| 157 | — | **MemoryError** | 0.007 s | — |

Identity checked with `np.array_equal`: bit for bit, not an approximation.

The kernels in this configuration are 157 px (HST) and **281 px** (JWST), so
the old form could not run at all on the JWST bands. Observed in practice:
convolution went from minutes per filter to seconds.

This also explains an unresolved observation from earlier work — after fixing
the JWST invalid-pixel convention the convolution became noticeably faster.
That was attributed to the FFT no longer interpolating over a large invalid
region, which is true but partial: correcting the mask also shrank
`border_mask`, the input to this dilation, and that was probably the dominant
term.

---

## 3. Check 1 — WCS consistency

**Script:** `check1_astrometry.py`
**Criterion, fixed in advance:** median offset < 0.5 pixel of the final grid.

### Question

`reproject` reads the WCS of source and destination and interpolates between
them. It never verifies that those WCS agree. Mixing two surveys therefore
rests on an assumption that had not been tested.

### Method

Five point sources were selected by eye in DS9, with the two frames locked by
WCS, and their approximate coordinates read off the **HST** frame.

Starting from a single coordinate per source is the crux. Marking each source
independently in both frames would fold the error of the hand — about one
screen pixel — into a measurement of half an image pixel. Instead each sky
coordinate is converted to pixel in **both** images, the centroid is refined
independently in each with a 2D Gaussian fit (`photutils.centroid_2dg`,
`box_size` 15 px on HST and 7 px on JWST, chosen so the window covers a
comparable patch of sky), converted back to sky, and the two refined positions
are compared.

Sources used (ICRS degrees, read on HST/f814w):

```
1  41.6036123  -0.5152057
2  41.6007478  -0.5130389
3  41.6135966  -0.5130459
4  41.6014181  -0.4820472
5  41.6173553  -0.4850417
```

They span ~60″ in RA and ~120″ in Dec, so a rotation or scale error between
the two solutions would show up, not only a translation.

### Sanity test

The same procedure with HST against **itself** returns exactly zero on all five
sources, with identical centroid shifts in both columns. Without that, a small
offset in the real measurement could not be distinguished from a bug in the
coordinate chain.

### Result

```
src   sep (arcsec)   sep (px)   dRA (arcsec)   dDec (arcsec)
 1        0.0072      0.065        +0.0031        +0.0065
 2        0.0091      0.082        -0.0021        +0.0089
 3        0.0018      0.016        -0.0013        -0.0012
 4        0.0110      0.099        +0.0080        -0.0076
 5        0.0087      0.079        -0.0038        +0.0079

median separation : 0.0087 arcsec = 0.079 px of the final grid
scatter (std)     : 0.0031 arcsec
mean offset       : dRA +0.0008, dDec +0.0029 arcsec
systematic/median : 0.34
```

**PASS**, with 6× margin.

### Interpretation

The mean offset vector is three times smaller than the scatter between
sources, and the signs of dRA alternate (+, −, −, +, −). There is therefore **no
systematic displacement** between the two WCS solutions: what was measured is
the precision of the centroiding method, not a disagreement between the
surveys.

### Scope

This result covers **PHANGS-HST against PHANGS-JWST only**, on NGC 1087. S4G,
J-PAS and any future survey remain unverified; for those the agreement is
assumed rather than measured, and the risk of combining them has not been
quantified.

---

## 4. Check 2 — PSF matching

**Scripts:** `check2_psf_matching.py` (standalone), `validation.py` (in-pipeline)
**Criterion, fixed in advance:** 0.95 ≤ ratio ≤ 1.05, every band.

### Method

For each kernel, convolve the source PSF with it and compare the
enclosed-energy radius of the result with the master's:

```
ratio = r50(source * kernel) / r50(target)
```

A correct kernel reproduces the target at all radii (Aniano et al. 2011,
Sect. 5: *"For a perfect kernel, both quantities should coincide at all
radii"*). The ratio reduces that to one number, which must be 1.0.

The script discovers the pairs on its own: in each `PSF_CLEAN/` directory the
`master_*` file is the target and the rest are sources. A pair whose PSFs sit
on grids differing by more than 2% is **skipped rather than measured**, because
a kernel between different grids is meaningless — that is the defect that made
the v1.x pipeline blur by a factor ~12 too little.

Three secondary metrics accompany the ratio:

- **r80 ratio** — the same test weighted towards the wings
- **negative power** — Aniano's W₋ in normalised form, `W₋/(1+2W₋)`; the
  recommended limit W₋ ≳ 1.2 corresponds to 35.3% here
- **D** — Aniano Eq. 20, `∫|Ψ_B − K ⊛ Ψ_A|`, more sensitive to shape than any
  single radius

### Result

**PHANGS-HST**, grid 0.1981″/px, master `f2100w`:

| band | px/FWHM | r50 ratio | r80 ratio | neg pwr | D |
|---|---:|---:|---:|---:|---:|
| f275w | 0.39 | 1.0001 | 1.0000 | 0.00% | 1.63e-04 |
| f336w | 0.41 | 1.0001 | 1.0000 | 0.00% | 1.66e-04 |
| f438w | 0.42 | 1.0001 | 1.0000 | 0.00% | 1.69e-04 |
| f555w | 0.42 | 1.0001 | 1.0000 | 0.00% | 1.73e-04 |
| f814w | 0.40 | 1.0000 | 1.0000 | 0.00% | 1.93e-04 |

**PHANGS-JWST**, grid 0.1109″/px, master `f2100w`:

| band | px/FWHM | r50 ratio | r80 ratio | neg pwr | D |
|---|---:|---:|---:|---:|---:|
| f200w | 0.56 | 1.0000 | 1.0000 | 2.54% | 5.38e-04 |
| f300m | 0.86 | 1.0000 | 1.0000 | 3.17% | 5.80e-04 |
| f335m | 0.96 | 1.0000 | 1.0000 | 3.65% | 6.37e-04 |
| f360m | 1.04 | 1.0000 | 1.0000 | 3.96% | 6.77e-04 |
| f770w | 2.13 | 1.0000 | 1.0000 | 9.71% | 1.25e-03 |
| f1000w | 2.81 | 1.0000 | 1.0000 | 13.86% | 1.27e-03 |
| f1130w | 3.22 | 1.0000 | 1.0000 | 16.67% | 1.25e-03 |

**PASS**, all 12 pairs.

### A physical pattern worth noting

The negative power of the JWST kernels grows monotonically as the source PSF
approaches the master in width:

```
f200w  (0.062")   2.54%     <- narrowest, kernel well conditioned
f1130w (0.357")  16.67%     <- closest to the master (0.643")
```

This is the expected behaviour: the closer the two PSFs, the less blur the
kernel must apply, the nearer it is to a delta function — the regime where
Aniano et al. warn that performance degrades. All remain below the 35.3% limit,
but `f1130w` is the one to watch.

The HST kernels at 0.00% are the opposite extreme: a source PSF eight times
narrower than the target gives a very well-conditioned kernel.

### A note on the px/FWHM column

The sampling figures come from a table of nominal FWHM, not from measuring the
cleaned PSF. Once resampled onto a coarse grid, a PSF narrower than the pixel
is concentrated in a single pixel and **no measurement of the sampled image can
recover its true width** — the information is no longer there. An earlier
version measured it and reported 0.00 for every HST band.

---

## 5. Check 3 — flux conservation

**Scripts:** `check3_flux_conservation.py` (standalone), `validation.py` (in-pipeline)
**Criterion, fixed in advance:** flux conserved to within 1%.

### Method, and one correction along the way

The measurement is made in **apertures fixed in sky coordinates** — the same
patch of sky on each grid — with the sum in each aperture **weighted by the
pixel area** of its grid.

The weighting is the crux, and the first version of this check omitted it. The
result was a spectacular false failure:

```
HST bands        : ratio 0.0697        (0.1981/0.75)^2 = 0.0698
MIRI bands       : ratio 0.0218        (0.1109/0.75)^2 = 0.0219
NIRCam LW        : ratio 0.0071        (0.0630/0.75)^2 = 0.0071
NIRCam SW        : ratio 0.0017        (0.0310/0.75)^2 = 0.0017
irac1            : ratio 0.9916        already on the master grid
```

Every ratio is exactly the ratio of pixel areas. `reproject_interp` preserves
the **value** of each pixel, not the sum — surface-brightness behaviour, which
is what the pipeline relies on and what `convert2Jansky` later accounts for.
Summing raw values across grids measures the change of pixel size: from the
0.1981″ HST grid to a 0.75″ master grid the raw sum falls by 14.3 with no
photon lost anywhere.

The `irac1` line was the control that confirmed it: already on the master grid,
no resampling, ratio 0.99 — while every resampled band showed its own area
ratio.

A second refinement followed. Apertures landing on blank sky return noise over
noise; their scatter reached 23,000%. The check now requires each aperture to
contain at least 5% of the flux of the brightest one, and scatter fell to
~0.3%.

### Validation of the method

Tested on synthetic reprojections of known answer, across grid changes up to
24× in scale (600× in area):

```
HST -> irac2         0.1981 -> 0.7500   ratio 0.9991   scatter 0.29%   PASS
MIRI -> irac2        0.1109 -> 0.7500   ratio 1.0005   scatter 0.41%   PASS
NIRCam SW -> irac2   0.0310 -> 0.7500   ratio 1.0000   scatter 0.38%   PASS
```

Deliberately corrupting one case by 5% makes the check fail, as it should.

### Result

| band | n ap | grid in | grid out | ratio | scatter | deviation |
|---|---:|---:|---:|---:|---:|---:|
| f275w | 8 | 0.1981 | 0.7500 | 0.9974 | 0.21% | +0.26% |
| f336w | 8 | 0.1981 | 0.7500 | 0.9976 | 0.20% | +0.24% |
| f438w | 8 | 0.1981 | 0.7500 | 0.9987 | 0.15% | +0.13% |
| f555w | 8 | 0.1981 | 0.7500 | 0.9991 | 0.11% | +0.09% |
| f814w | 8 | 0.1981 | 0.7500 | 0.9992 | 0.11% | +0.08% |
| f200w | 6 | 0.0307 | 0.7500 | 0.9922 | 7.86% | +0.78% |
| f300m | 8 | 0.0630 | 0.7500 | 1.0001 | 0.26% | +0.01% |
| f335m | 8 | 0.0630 | 0.7500 | 0.9994 | 0.19% | +0.06% |
| f360m | 8 | 0.0630 | 0.7500 | 0.9989 | 0.19% | +0.11% |
| f770w | 8 | 0.1109 | 0.7500 | 0.9979 | 0.09% | +0.21% |
| f1000w | 8 | 0.1109 | 0.7500 | 0.9983 | 0.09% | +0.17% |
| f1130w | 8 | 0.1109 | 0.7500 | 0.9980 | 0.09% | +0.20% |
| f2100w | 8 | 0.1109 | 0.7500 | 0.9989 | 0.21% | +0.11% |
| irac1 | 5 | 0.7500 | 0.7500 | 0.9916 | 1.43% | +0.84% |

```
worst deviation  : 0.840%
median deviation : 0.152%
```

**PASS**, all 14 bands.

Two entries deserve comment. `f200w` has the largest scatter (7.86%) and used
only 6 apertures — expected, since it undergoes the most extreme grid change
(24× in scale, 600× in area). And `irac1` shows the **worst** deviation despite
being the only band **not** reprojected (0.75″ → 0.75″); the likely cause is the
SIP correction applied in `reproject_to_reference`, which re-attaches the
`-SIP` suffix. Not blocking, since S4G is deferred, but unexplained.

---

## 6. The 1.06 investigation

Earlier work measured an r50 ratio of **1.06** for the HST→IRAC pairs, outside
the 0.95–1.05 criterion. Before implementing check 2 as an abort gate the cause
had to be understood, otherwise the gate would simply be a wall.

**Script:** `test_pypher_regularisation.py`

### Design

The test rests on a **synthetic control**: two Gaussians of known width. For
Gaussians the exact kernel is analytic (σ = √(σ_B² − σ_A²)), so a correct method
must return 1.000. If it does not, the problem is in the method rather than in
the real PSFs. That is what makes everything else interpretable.

### Hypothesis 1: Wiener regularisation

PyPHER regularises with a parameter `r` (default 1e-4, never chosen by the
pipeline — it is PyPHER's own). Penalising high frequencies smooths, and
smoothing widens.

Sweeping four orders of magnitude, on both the synthetic pair and a real one:

```
        r    r50 ratio (synthetic)   r50 ratio (f1130w->f2100w)
    1e-06           1.0000                    1.0000
    1e-05           1.0000                    1.0000
    1e-04           1.0000                    1.0000
    1e-03           1.0000                    1.0000
    1e-02           1.0000                    1.0000
```

**Discarded.** The ratio does not move. Regularisation does affect the negative
power and D, but not the width of the result.

### Hypothesis 2: the measurement itself

**Discarded.** The synthetic control returns 1.0000 with D ~10⁻⁶. PyPHER, the
r50 measurement and the convolution are all correct.

### Hypothesis 3: undersampling of the source PSF

On the 0.198″ convolution grid the HST PSF spans 0.40 px/FWHM — badly
undersampled. Tested with the **analytic** kernel, varying only the grid:

```
grid       px/FWHM    r50 ratio         D
0.0396″      1.99       1.0000     7.2e-08
0.0990″      0.80       1.0000     1.2e-03
0.1980″      0.40       1.0000     1.6e-03
0.3960″      0.20       1.0000     1.6e-03
```

**Discarded.** Undersampling alone does not bias the ratio, even at
0.20 px/FWHM. It does degrade the shape — D rises five orders of magnitude —
but not in a way r50 captures. Running PyPHER on the undersampled synthetic
pair also returned 1.0000.

### Conclusion

None of the three hypotheses accounts for the 1.06, and **the current
configuration does not reproduce it**: the worst HST pair, `f275w → f2100w` at
0.39 px/FWHM, measures 1.0001.

The most likely explanation is that 1.06 belonged to the HST+S4G configuration
(master `irac2`) and was eliminated by one of the intervening corrections —
the grid fix, `_recenter_odd`, or `copy_as_convolved(is_master=True)`. Since
S4G is deferred, it has been left as a historical note rather than pursued.

One quantitative by-product: the effect of `r` on a real PSF pair does exist
but only far from the default.

```
f275w -> f2100w:   r=1e-06  0.9999
                   r=1e-04  1.0001   <- default
                   r=1e-02  1.0167
```

The hypothesis was conceptually right and quantitatively irrelevant: at the
default the effect is 0.01%. The default sits in a safe region, and now we know
by how much.

---

## 7. In-pipeline gates

**Module:** `validation.py` (new)

Checks 2 and 3 run inside the pipeline and **abort it on failure**, so a bad
kernel or a leaking resampling is caught at the stage that produced it rather
than surfacing later as a wrong number in a map.

| gate | runs | rationale |
|---|---|---|
| `check_psf_matching` | after kernels are built, **before any image is convolved** | convolving 13 bands with a bad kernel costs an hour and yields bands at the wrong resolution — invisible in the images, corrupting every colour |
| `check_flux_conservation` | after reprojection, **before unit conversion** | a leak that reaches the Jy files has already propagated, and the conversion rescales the numbers, making it harder to attribute |

The criteria are fixed in the module rather than passed in as arguments: a
threshold that can be relaxed at the call site is not a gate. The only way past
is `--skip_checks`, which is printed in the log.

Both raise `ValidationError`, which aborts with the failing bands, their values,
and why it matters.

### Verified behaviour

```
correct kernel    (ratio 1.0000)  -> PASS
kernel 30% wide   (ratio 1.5811)  -> ValidationError, pipeline aborts

correct reprojection (1.0010)     -> PASS
flux leaking 5%      (0.9509)     -> ValidationError, pipeline aborts
```

### One bug found during integration

The first version attributed kernels to surveys by substring match. Since the
master appears in **every** kernel filename (`kernel_f275w_to_f2100w.fits`),
every kernel matched the master's survey and the HST bands were attempted
against the JWST PSF directory. Harmless in effect — they were skipped with a
warning — but it cluttered the log and could mask a real problem later. Fixed
by parsing the source filter out of the name.

---

## 8. Vocabulary

Three operations were being conflated. They are now kept apart in code,
documentation and module names:

| term | operation | status |
|---|---|---|
| **reprojection** (regridding, resampling) | bring images onto a common pixel grid, trusting their WCS | what the pipeline does |
| **astrometric registration** (astrometric alignment) | measure offsets between common sources and correct the WCS | **not implemented** |
| **PSF matching** (homogenisation) | bring images to a common angular resolution | what PyPHER and the convolution stage do |

Changes applied:

- `alignment_2_0.py` → **`reprojection_2_0.py`**, with a module header stating
  what it does and does not do
- `--mode alignment_only` → **`--mode reprojection_only`**
- `target_pixel_scale_arcsec` → **`convolution_grid_arcsec`** (10 occurrences).
  The old name invited exactly the confusion it caused: the parameter is the
  grid where kernel and image meet, and has **no relation** to the target PSF —
  it is `output_size` that derives from the target.
- the project description no longer says *"a single spatially-registered
  datacube"*. It now says the cube is reprojected onto a common grid, trusting
  the input WCS, and states explicitly which pair has been verified and which
  have not.

The scope caveat appears in three places — the document opening, the module
header, and the start of Section 6 — so that a reader who opens only one of
them still meets it.

---

## 9. Regenerated cubes

```bash
python astrovello_cli_2.0.py --mode full --galaxy ngc1087 --create_kernel
# surveys: phangs-hst, phangs-jwst
```

Header of the resulting cube:

```
CDELT1  : 3.0807051217028e-05      (was 1.0)
CDELT2  : 3.0807051217028e-05
PC1_1   : -0.99999999991961
PIXAREA : 0.0123

scale from WCS     : 0.11090538 arcsec/px
scale from PIXAREA : 0.11090537 arcsec/px     <- agree to 8 decimals

bands : 13
shape : 13 x 1277 x 1135                      (was 1276 x 1134)
```

All four corrections visible in one header: the two scales agree (A1), the
dimensions are +1 per axis (A3), and the cube was produced through the fixed
conversion (A2) and the fast dilation (B1).

---

## 10. Files created or modified

### New

| file | role |
|---|---|
| `validation.py` | in-pipeline gates; `check_psf_matching`, `check_flux_conservation`, `ValidationError` |
| `check1_astrometry.py` | standalone WCS consistency, with the HST-against-itself sanity test |
| `check2_psf_matching.py` | standalone PSF matching over all pairs, with auto-discovery |
| `check3_flux_conservation.py` | standalone flux conservation in sky apertures |
| `test_pypher_regularisation.py` | the 1.06 investigation: synthetic control plus regularisation sweep |

### Modified

| file | change |
|---|---|
| `cube_2_0.py` | A1 — 3D WCS via `to_header()` |
| `drivers.py` | A2 — read `NATPXAR`; B1 — dilation by iterations (3 places) |
| `mask_2_0.py` | A3 — inclusive/exclusive bound in `crop_to_mask_bbox` |
| `convolution_2_0.py` | `convolution_grid_arcsec` rename |
| `alignment_2_0.py` → `reprojection_2_0.py` | renamed, new module header |
| `astrovello_cli_2.0.py` | gates wired in, `--skip_checks`, renames, `kernel_source_filter` |
| `astrovello_pipeline_documentation.md` | vocabulary, scope caveat, renames |

All check scripts write to `Output/checks/<check name>/` inside the repository,
so their products stay with the data they describe.

---

## 11. Open items

**`irac1` at 0.84% in check 3.** The worst deviation, and the only band **not**
reprojected. Likely the SIP correction in `reproject_to_reference`. Not
blocking while S4G is deferred, but unexplained.

**Astrometric verification of the other surveys.** Only HST↔JWST has been
measured. Running `check1_astrometry.py` on any new combination is the
precondition for mixing it.

**The `--apply_mask` default.** The CLI defaults to off while
`create_data_cube` defaults to on. The cubes above were produced without it, so
the second crop never acted. Worth deciding deliberately, since it changes the
cube dimensions.

**Per-survey `min_blur`** (audit B6). `derive_bin_factors` takes the minimum
required blur over all bands globally, so the HST grid is constrained by a pair
on a different survey's grid. Computing it per survey would allow bin 14
instead of 5 for HST, cutting the convolution to ~13% of the pixels while
keeping 3.1 px/FWHM on the master. Worth running `--preflight_only` before
adopting.

**Sagui segmentation.** Paula's observation that the default segmentation is
not scale-invariant — it translates the SED without renormalising, so it groups
by stellar population mixed with a surface-brightness proxy, which is
morphology. This would explain two results that had no explanation: the nearly
binary age map of NGC 1566 and the resemblance between the segmentation map and
the galaxy's morphology. ProSpect is paused until this is understood.

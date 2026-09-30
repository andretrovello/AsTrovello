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
11. [Cross-review](#11-cross-review)
12. [Open items](#12-open-items)

---

## 1. Summary of results

| item | criterion | measured | verdict |
|---|---|---|---|
| Check 1 — WCS consistency (HST↔JWST) | < 0.5 px | **0.079 px** (0.0087″) | PASS, 6× margin |
| Check 2 — PSF matching, 12 pairs | 0.95–1.05 | **1.0000–1.0001** | PASS |
| Check 3 — flux conservation, 12 bands | < 1% | **0.01% worst**, 0.10% max scatter | PASS |
| A1 — cube WCS | scales must agree | 0.110905 vs 0.110905 | fixed |
| A2 — HST pixel area | −0.61% on HST, 0% on IR | −0.604% / +0.000% | fixed |
| A3 — crop off-by-one | +1 px per axis | 1276×1134 → 1277×1135 | fixed |
| B1 — dilation cost | identical result, much faster | bit-identical, minutes → seconds | fixed |
| Per-band convolution grid | kernel on the grid where it is applied | 4 grids, all kernels ~19.8″ extent | fixed later, see §9 |

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

The kernels in this configuration are 157 px (HST) and **281 px** (JWST). At
those sizes the old form can fail with `MemoryError` — it did here at k=157 on
a 600² mask, though the audit ran the same k on a 1600² image in 29 s, so the
failure threshold is machine-dependent. Observed in practice: convolution went
from minutes per filter to seconds.

**A mechanism this does not explain.** An earlier draft suggested that the
speed-up seen after correcting the JWST invalid-pixel convention was dominated
by this dilation, on the grounds that a corrected mask is smaller. The audit
had already measured the opposite: scipy short-circuits pixels that are
already `True`, so a **smaller** mask makes the full-square dilation **slower**
(0.4% True → 2.32 s; 43.6% True → 1.33 s). That earlier speed-up was the FFT,
as originally diagnosed. The correction here is a separate and additive gain.

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

### What this check does and does not establish

The gate convolves the **same** `PSF_CLEAN` files that PyPHER used to build the
kernel. A ratio near 1.0000 is therefore close to guaranteed by construction,
and the numbers above should not be read as independent evidence that the
science images reach the master resolution.

What the check does catch, and what makes it worth running as a gate:

- a stale kernel, built for a different master
- a kernel and its PSFs on mismatched grids (the v1.x defect, which raises)
- a sign-flipped kernel (raises, since the B2 correction)
- a kernel deliberately widened: a 30% wider kernel measures 1.5811 and aborts

Independent evidence would require measuring the PSF on the convolved science
images themselves — for instance the FWHM of field stars before and after
convolution. That has not been done.

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
result was a spectacular false failure. The run below was made on an
HST+JWST+S4G configuration (master `irac2`, 0.75 arcsec/px), which is why the
band list differs from the final table above:

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

Configuration: PHANGS-HST + PHANGS-JWST, master `f2100w`, reference grid
0.1109 arcsec/px.

| band | n ap | grid in | grid out | ratio | scatter | deviation |
|---|---:|---:|---:|---:|---:|---:|
| f275w | 8 | 0.1981 | 0.1109 | 0.9999 | 0.04% | +0.01% |
| f336w | 8 | 0.1981 | 0.1109 | 0.9999 | 0.03% | +0.01% |
| f438w | 8 | 0.1981 | 0.1109 | 0.9999 | 0.02% | +0.01% |
| f555w | 8 | 0.1981 | 0.1109 | 0.9999 | 0.01% | +0.01% |
| f814w | 8 | 0.1981 | 0.1109 | 0.9999 | 0.05% | +0.01% |
| f200w | 7 | 0.0307 | 0.1109 | 0.9999 | 0.10% | +0.01% |
| f300m | 6 | 0.0630 | 0.1109 | 0.9999 | 0.07% | +0.01% |
| f335m | 8 | 0.0630 | 0.1109 | 1.0000 | 0.01% | +0.00% |
| f360m | 8 | 0.0630 | 0.1109 | 0.9999 | 0.03% | +0.01% |
| f770w | 8 | 0.1109 | 0.1109 | 0.9999 | 0.00% | +0.01% |
| f1000w | 8 | 0.1109 | 0.1109 | 0.9999 | 0.01% | +0.01% |
| f1130w | 8 | 0.1109 | 0.1109 | 0.9999 | 0.01% | +0.01% |

```
worst deviation : 0.01%
max scatter     : 0.10%
```

**PASS**, all 12 bands, with 100x margin on the criterion.

Twelve bands rather than thirteen: `f2100w` is the master, so it is not
reprojected and has no before/after pair.

Three bands (`f770w`, `f1000w`, `f1130w`) show `grid in = grid out = 0.1109` —
they are already on the reference grid, and the resampling is a no-op. They
return 0.9999 with 0.00-0.01% scatter, which is the control this table
contains: where nothing is resampled, nothing changes.

### A correction to an earlier version of this report

An earlier draft carried a Check 3 table with `grid out = 0.7500` for every
band, an `irac1` row, and 14 bands against the 13 stated in the summary. That
table came from an **HST+JWST+S4G** run and did not belong to the
configuration this report describes; it was pasted without being checked
against the document's own heading, and the `irac1` row was visible evidence
that it did not fit.

The measurement above replaces it. The difference is not small: the stale
table reported a worst deviation of 0.84%, this one 0.01%.

The `irac1` anomaly discussed there (0.84% on a band that was **not**
reprojected) belongs to the S4G configuration, not to this one. Investigating
it uncovered a genuine defect, described in the next subsection.

### What the `irac1` investigation found, and what it got wrong

The starting hypothesis was that the deviation was an artefact of the check:
SIP handled differently either side of the reprojection, so the same sky
aperture landing on different pixels.

Reading the code appeared to confirm something worse.
`reproject_to_reference` replaces the linear WCS of the output with the
reference's — removing `CRPIX`, `CRVAL`, `CDELT`, `CD`, `PC` and `CTYPE` — but
the list does not include the SIP coefficients (`A_p_q`, `B_p_q`, `A_ORDER`,
`AP_*`, `BP_*`). Those describe the distortion of the SOURCE detector in
SOURCE pixel coordinates, so the output header seemed to carry the reference
linear WCS combined with a distortion that no longer referred to anything. A
synthetic test with second-order coefficients showed displacements of 0.04″ to
0.38″, and the coefficients were removed.

**That was wrong, and the measurement says so.** Running
`test_sip_hypothesis.py` on the real S4G files:

```
astrometric shift from removing the SIP keys, over 25 field positions:
    median 6.6119"   min 0.0416"   max 49.7211"
    in pixels of the 0.75" grid:  median 8.8,  max 66.3

flux-conservation deviation, irac1:
    with the SIP keys     0.840%   (scatter  1.43%)
    without them         11.273%   (scatter 15.42%)
```

Removing the coefficients made the WCS **worse by more than an order of
magnitude**. A displacement of 50 arcsec is not residual field distortion: in
these files the SIP terms carry part of the S4G astrometric solution itself.
That is consistent with what the function already does elsewhere — it
deliberately re-attaches the `-SIP` suffix that the S4G headers omit,
precisely so those coefficients are applied.

The change was reverted, and the reasoning recorded in the code so the same
argument is not made again.

Two things are worth taking from this. The synthetic test was built with
plausible-looking coefficients and produced a displacement two orders of
magnitude smaller than the real ones; it validated the mechanism but said
nothing about the magnitude, and treating it as evidence about the data was
the error. And the hypothesis remains **unresolved**: the 0.84% is not
explained by SIP handling, and its cause is still unknown.

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

### A fourth mechanism, identified by the audit

The three hypotheses above were tested and discarded, but they were not the
only candidates. The audit points to a sharper one that this investigation
missed: `diagnostico_etapa1c.py` (lines 207-209) measures the **numerator** on
the convolution grid and the **denominator** on the raw PRF grid. A ratio
whose two terms come from different grids carries a systematic that has
nothing to do with the kernel.

That would explain both facts at once — why 1.06 appeared then, and why it is
absent now: the current checks measure both terms on the same grid, which
alone would remove it.

**This remains untested.** The cheap test is to re-run the etapa1c comparison
with the target r50 measured on the convolution grid. If the ratio drops to
~1.00, the 1.06 was a measurement systematic and never a kernel defect.

Until that test is run, the attribution stands as: 1.06 belonged to the
HST+S4G configuration, its cause is most likely the mixed-grid ratio above,
and the current configuration measures 1.0001. Since S4G is deferred, it has
been left as a note rather than pursued.

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

### This cube has since been superseded

**`ngc1087_datacube_sci_1277x1135` must not be used.** A later round found
that the four PHANGS-JWST NIRCam bands in it were convolved with kernels built
on the MIRI grid — 3.6x too coarse for f200w, 1.8x for the long-wave bands.
That is a colour error between NIRCam and every other band in the cube.

The defect had been invisible because the grid guard in `create_convolvedFITS`
had never once fired: PyPHER does not carry non-standard keys from its inputs,
so `PIXSCALE` never reached a kernel, and the guard fell through its
`k_px is None` branch on every band of every run. Making that branch fatal
(the B3 fix above) is what exposed it.

The replacement is **`ngc1087_datacube_sci_1109x1202`**, produced after the
convolution grid was made a property of the band rather than of the survey.
The change, its verification across five survey configurations, and the full
accounting of the dimension change are in `per_band_grid_report.md`.

---

## 10. Files created or modified

### New

| file | role |
|---|---|
| `validation.py` | in-pipeline gates; `check_psf_matching`, `check_flux_conservation`, `ValidationError` |
| `check1_astrometry.py` | standalone WCS consistency, with the HST-against-itself sanity test. **Not yet committed** — the repository carries only a stub (`src/astrovello/diagnostico_astrometria.py`), so the Check 1 result cannot currently be reproduced from it. Commit before citing. |
| `check2_psf_matching.py` | standalone PSF matching over all pairs, with auto-discovery |
| `check3_flux_conservation.py` | standalone flux conservation in sky apertures |
| `test_pypher_regularisation.py` | the 1.06 investigation: synthetic control plus regularisation sweep. **In the repository as `pypher_regularisation_test.py`** |
| `test_sip_hypothesis.py` | strips the SIP keys from a copy of a reprojected file and re-measures flux conservation, to test whether they explain the `irac1` deviation |

### Modified

| file | change |
|---|---|
| `cube_2_0.py` | A1 — 3D WCS via `to_header()` |
| `drivers.py` | A2 — read `NATPXAR`; B1 — dilation by iterations (3 places) |
| `mask_2_0.py` | A3 — inclusive/exclusive bound in `crop_to_mask_bbox` |
| `convolution_2_0.py` | `convolution_grid_arcsec` rename |
| `alignment_2_0.py` → `reprojection_2_0.py` | renamed, new module header. A revision that removed the source SIP coefficients was reverted after measurement showed it degrades the WCS (Sect. 5); the reasoning is recorded in the code |
| `astrovello_cli_2.0.py` | gates wired in, `--skip_checks`, renames, `kernel_source_filter` |
| `astrovello_pipeline_documentation.md` | vocabulary, scope caveat, renames |

All check scripts write to `Output/checks/<check name>/` inside the repository,
so their products stay with the data they describe.

---

## 11. Cross-review

This report was reviewed against the repository by the audit tool that
produced the original defect list (`fix_verification_2026-09-28.md`). The
review confirmed A1, A2, A3 and B1 as fixed, and found four problems in this
report or in the fixes. All four have been addressed above:

| finding | where | resolution |
|---|---|---|
| A2 only fixed on the binned path — `NATPXAR` absent when `factor <= 1`, so the fallback reinstates the stale constant | §2 A2 | key now written on every path; the docstring's "no aliasing" claim also corrected |
| B2 not fixed, and repeated in the new gate — negative power still measured after normalisation, in three places | §7 | measured on the raw kernel in all three; a negative sum is now a hard failure |
| the Check 3 table belonged to an HST+JWST+S4G run, not to the configuration this report describes | §5 | re-measured on the f2100w configuration; the correction is documented in place |
| the B1 speed-up mechanism had already been refuted by measurement, and the 1.06 investigation missed a fourth candidate | §2 B1, §6 | both paragraphs corrected; the mixed-grid hypothesis added |

Two further points from the review are reflected in the text: the limit of
what Check 2 establishes (§4) and the file-name mismatches (§10).

The review also verified something this report could not: it read the **raw**
PyPHER kernels and confirmed that the 0.00% negative power reported for the
HST bands is a genuine property of those kernels, not an artefact of measuring
after normalisation. The number was right; the method that produced it was
not, and has been corrected.

---

## 12. Open items

**`irac1` at 0.84% — cause unknown.** The SIP hypothesis was tested on the
real files and **refuted**: removing the coefficients raises the deviation to
11.3% (Sect. 5). Whatever produces the 0.84% on a band that is not resampled
is still unidentified. Not blocking while S4G is deferred, but it should be
settled before S4G returns, since it is the one band whose flux is not
conserved to the level every other band reaches.

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

**Audit items still open**, per the cross-review: B3 (a missing `PIXSCALE`
skips the grid guard, and the stale v1.x kernels are exactly the files without
that key), B5 (border filled with `0.0` rather than `NaN`), B6 (per-survey
`min_blur`), B7, C1, C2, the documentation items under D, and the packaging
items under E.

**`check1_astrometry.py` is not in the repository.** The Check 1 result cannot
be reproduced from it until the script is committed.

**Sagui segmentation.** Paula's observation that the default segmentation is
not scale-invariant — it translates the SED without renormalising, so it groups
by stellar population mixed with a surface-brightness proxy, which is
morphology. This would explain two results that had no explanation: the nearly
binary age map of NGC 1566 and the resemblance between the segmentation map and
the galaxy's morphology. ProSpect is paused until this is understood.

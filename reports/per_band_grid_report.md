# AsTrovello — per-band convolution grid

**Follow-up to `fix_verification_2026-09-29.md`**
Galaxy: NGC 1087 · Configuration: PHANGS-HST + PHANGS-JWST, master `f2100w`

Your B3 fix — making a missing `PIXSCALE` a hard error instead of a silent
skip — turned out to expose two things rather than one. This document reports
both, and the restructuring that followed.

---

## 1. The grid guard had never once fired

Making the `k_px is None` branch fatal produced an immediate abort:

```
ValueError: Kernel kernel_f336w_to_f2100w.fits declares no pixel scale:
it has no PIXSCALE keyword and no usable WCS.
```

Clearing `Output/PSF_Kernels` and regenerating did not help. The cause:

```
PSF    PSFSTD_WFC3UV_F275W.fits      PIXSCALE = 0.1981
PSF    PSFSTD_WFC3UV_F336W.fits      PIXSCALE = 0.1981
KERNEL kernel_f1000w_to_f2100w.fits  PIXSCALE = None
KERNEL kernel_f1130w_to_f2100w.fits  PIXSCALE = None
```

`clean_psf` writes `PIXSCALE` into the cleaned PSFs, but the kernel is written
by **PyPHER**, which does not carry over non-standard keys from its inputs. The
key never reached a single kernel.

So the guard in `create_convolvedFITS` — the protection against the v1.x grid
mismatch, the defect that cost weeks to find — had **never had a scale to
compare**. It fell through the `k_px is None` branch on every band of every
run since it was written. It was reviewed, documented, and inoperative.

**Fix:** `build_kernels` now stamps `PIXSCALE` onto each kernel after PyPHER
returns, with the grid that kernel was built for. Scoped to the current
survey's bands, since the loop runs once per survey and each has its own grid.

---

## 2. What the guard found once it could speak

With `PIXSCALE` present, the next run aborted again — this time on real data:

```
ValueError: Kernel for 'f300m' is on 0.1109 arcsec/px but the convolution
image is on 0.0630 arcsec/px. The blur would be wrong by a factor of ~0.6.
```

The PHANGS-JWST bands are **not on a common grid**:

| instrument | bands | native scale |
|---|---|---|
| NIRCam short-wave | f200w | 0.0307″ |
| NIRCam long-wave | f300m, f335m, f360m | 0.0630″ |
| MIRI | f770w, f1000w, f1130w, f2100w | 0.1109″ |

A factor 3.6 between the extremes. But `derive_bin_factors` did:

```python
native = science_pixel_scale(imgs[0], ...)
```

`imgs[0]` is whichever file sorts first — arbitrary. One grid was chosen for
all eight bands. It happened to be a MIRI band, so the kernels for the four
NIRCam bands were built on 0.1109″ and applied to images at 0.0307″ and
0.0630″.

**This is the v1.x defect in another form**, at a smaller factor: a kernel
generated on one grid and applied on another. It was present in every cube
produced before today, silently, because the guard could not fire.

---

## 3. The restructuring: one grid per band

The convolution grid is a property of the **band**, not of the survey. Paula's
instruction in the September email said as much — *"no pypher, para cada banda,
leve a PSF alvo para a grade nativa daquela imagem"* — and the per-survey
implementation did not honour it.

`build_kernels` was rewritten:

- bands of each survey are grouped by their own native scale × `bin_factor`
- the master PSF is cleaned **once per distinct grid**, not once per survey
- source PSFs are cleaned onto the grid of their own science image
- PyPHER runs once per grid group
- `PIXSCALE` is stamped with that group's grid

Cleaned PSFs are now prefixed with their grid (`0.0630_PSF_NIRCam_F300M.fits`,
`master_0.0630_PSF_MIRI_F2100W.fits`), because the same band can otherwise
appear on two grids and overwrite itself.

`_prepare_image` was **not** changed: it bins by `bin_factor`, and the kernel
grid is now `native_of_that_band × bin_factor`, so the two coincide by
construction.

The Check 2 gate groups kernels by their `PIXSCALE` and compares each against
the master of its own grid. It raises if a kernel carries no `PIXSCALE`, since
then its grid cannot be determined.

---

## 4. Verification

### Kernels are on the grids they are applied to

```
kernel_f275w_to_f2100w.fits    PIXSCALE=0.1981   99px
kernel_f336w_to_f2100w.fits    PIXSCALE=0.1981   99px
kernel_f438w_to_f2100w.fits    PIXSCALE=0.1981   99px
kernel_f555w_to_f2100w.fits    PIXSCALE=0.1981   99px
kernel_f814w_to_f2100w.fits    PIXSCALE=0.1981   99px
kernel_f200w_to_f2100w.fits    PIXSCALE=0.0307  647px
kernel_f300m_to_f2100w.fits    PIXSCALE=0.0630  315px
kernel_f335m_to_f2100w.fits    PIXSCALE=0.0630  315px
kernel_f360m_to_f2100w.fits    PIXSCALE=0.0630  315px
kernel_f770w_to_f2100w.fits    PIXSCALE=0.1109  179px
kernel_f1000w_to_f2100w.fits   PIXSCALE=0.1109  179px
kernel_f1130w_to_f2100w.fits   PIXSCALE=0.1109  179px
```

The sizes are the confirmation:

```
 99 px x 0.1981" = 19.6"
179 px x 0.1109" = 19.9"
315 px x 0.0630" = 19.8"
647 px x 0.0307" = 19.9"
```

Every kernel covers the **same angular extent**, sampled on the grid of its
own band. That is what "the kernel lives on the grid where it is applied"
means, and it is what was missing.

### Check 2, now four groups

```
PHANGS-HST @ 0.1981    f275w f336w f438w f555w f814w    all 1.0000-1.0001  PASS
PHANGS-JWST @ 0.0307   f200w                            1.0000             PASS
PHANGS-JWST @ 0.0630   f300m f335m f360m                0.9999-1.0000      PASS
PHANGS-JWST @ 0.1109   f770w f1000w f1130w              1.0000             PASS
```

The `f200w` negative power fell from 2.54% to 2.02% — its kernel improved, as
expected now that it is built on the right grid.

Check 3 passes. The convolution was fast despite the 647 px kernel, which is
the B1 dilation fix earning its keep: with the old full-square footprint a
kernel that size would very likely have raised `MemoryError`.

---

## 5. Cube dimensions changed — accounted for

```
before: 1277 x 1135
after : 1202 x 1109      (-5.9% in y, -2.3% in x)
```

I could not explain the asymmetry; the cross-review measured it band by band
and closed it. **The NIRCam long-wave bands set the new footprint, not
f200w.** From the intersection of the reprojected files:

| intersection without | x range | y range |
|---|---|---|
| (all bands) | 367–1475 | 635–1836 |
| f200w | 367–1475 | 635–1836 — unchanged |
| NIRCam LW | 367–1487 | 621–1840 |
| all NIRCam | 367–1497 | 547–1872 |
| MIRI | 294–1475 | 635–1836 |

The LW kernel went from 179 px to 315 px at 0.0630″, so the LW border grew by
(157 − 89) × 0.0630″ = 4.28″ = 38.6 master px. Edge by edge, against the old
cube's position recovered from its CRPIX:

| edge | old | new | change | binding band |
|---|---|---|---|---|
| y bottom | 597 | 635 | +38 | NIRCam LW before and after — the full 38.6 |
| y top | 1873 | 1836 | −37 | MIRI/others before, LW now |
| x right | 1498 | 1475 | −23 | MIRI before, now LW, which crosses it by 23 |
| x left | 364 | 367 | +3 | MIRI f1000w both times; consistent with the B5 NaN ring |

y: 38 + 37 = 75. x: 3 + 23 = 26. Both match the observed change exactly.

The asymmetry is geometry, not a defect: on both y edges the LW footprint is
the limit, while in x the MIRI edge was already binding and LW overtakes it on
one side only.

**This is correct behaviour.** Every band now loses the same ~9.9″ per side.
NIRCam lost less before only because its kernels were physically too small.

### Per-band `bin_factor` is not the answer, and the reason corrects my reasoning

I proposed giving f200w bin 2 to halve its kernel and its border. That reasons
in pixels; the relevant quantity is angular. `driver.convolve` dilates by
`kernel_size // 2` iterations **on the convolution grid**, so the masked
border is half the kernel's angular extent regardless of the bin. That extent
comes from the master PSF file — the WebbPSF F2100W OVERSAMP array is
724 px × 0.0277″ = 20.05″, hence ~9.9″ per side for every band. At bin 2,
f200w would have a 323 px kernel on 0.0614″ and the same 9.9″ border. And
f200w is not the band that limits the footprint anyway.

Per-band binning would still buy compute (f200w could take bin 6, ~36× fewer
pixels), but it does not touch the footprint. The only levers on the border
are the kernel's angular extent and the dilation radius itself.

---

## 6. An attempted before/after comparison, and why I am not reporting a number

I tried to quantify the error that was in the old cubes by dividing the new
cube by the old one band by band. Two attempts:

**By pixel index** — meaningless. The two cubes have different footprints, so
`[:ny,:nx]` compares different patches of sky. The giveaway was `f2100w`
showing 19% difference: it is the master, copied without convolution or
reprojection, so it cannot change at all.

**By sky aperture** — better. `f2100w` returned 0.9963, which validates the
method. But:

```
band      ratio    scatter
f2100w   0.9963      8.8%     <- control, should be 1.0000
HST      0.963-0.988  9-29%
MIRI     0.990-0.991 18-26%
NIRCam   0.946-0.963  7-11%
```

The NIRCam group is systematically the most displaced, consistent with being
the only one whose kernels were wrong. But with 10-30% scatter a 4% median is
not distinguishable from zero. The apertures were placed on `f814w` peaks,
which in a star-forming disc are clusters rather than isolated sources, so
each aperture picks up neighbours differently on the two footprints.

**I am not reporting a magnitude.** What the comparison supports is the
direction — NIRCam moved most — and nothing quantitative. A cleaner
measurement would use the five isolated field stars from Check 1, whose
coordinates are already recorded.

I stopped there because the measurement quantifies an error in cubes that will
be discarded. The new cubes are correct by construction: the kernels are
demonstrably on the right grids, and all three checks pass.

---

## 7. Cross-review outcome

Reviewed against the repository (`fix_verification_2026-09-30.md`). The four
questions in the original version of this section were all answered.

**The per-band grid is a blocker fix, not hardening.** The review reclassified
its own earlier assessment: `CLAUDE.md` §0 had stated that no blocking code
fix remained for the HST+JWST cube. That was wrong until this change. Every
HST+JWST cube produced before 2026-09-30 convolved the four NIRCam bands with
kernels built on the MIRI grid — 3.6× too coarse for f200w, 1.8× for LW —
which is a colour error between NIRCam and everything else. **Those cubes must
not be used.**

**The rewrite does not break other configurations.** The kernel stage was
re-run in an isolated copy of `Input/` for five configurations, comparing each
kernel's `PIXSCALE` against its own image's native scale × bin:

| configuration | master | bins | kernels / grids | grid check | Check 2 |
|---|---|---|---|---|---|
| HST only | f814w | 1 | 0 — four bands unmatched (0.067–0.079″ ≈ 2 px) | — | nothing to check |
| S4G only | irac2 | 1 | 0 — irac1 unmatched (0.658″) | — | nothing to check |
| JWST only | f2100w | 1 | 7 on 3 grids | 7/7 | 7/7 PASS |
| HST+JWST | f2100w | 5 / 1 | 12 on 4 grids | 12/12 | 12/12 PASS |
| HST+JWST+S4G | irac2 | 12 / 4 / 1 | 13 on 5 grids | 13/13 | 13/13 PASS |

The HST+JWST run reproduced the kernels of §4 exactly, including the f200w
negative power of 2.02%. The unmatched classifications in the single-survey
runs are the existing resolvability rule, unchanged.

**What the change leaves behind**, all classified as hardening or cosmetic:
`derive_bin_factors` still takes `imgs[0]` for the bin factor (safe today only
because of filename sort order); `rediscover_unmatched` and the preflight
table still assume one grid per survey; the Check 2 gate prints rather than
raises when a master is missing; a dead pre-per-grid fallback remains in
`validation.py`; the standalone scripts still assume the old PSF naming.

**Housekeeping.** The pre-fix cube
`ngc1087_datacube_sci_1135x1277_Jy_per_pixel.fits` is still beside the new one
and carries the NIRCam grid error. It should be moved or deleted before
anything is frozen.

---

## 8. Status

| item | state |
|---|---|
| `PIXSCALE` stamped on kernels | fixed; the grid guard functions for the first time |
| NIRCam kernels on the wrong grid | fixed by the per-band grid — **this was a blocker** |
| other survey configurations | verified by the review, five configurations |
| Check 2, 12 pairs across 4 grids | PASS |
| Check 3 | PASS |
| cube regenerated | 13 × 1202 × 1109 |
| cube dimension change | fully accounted for (§5) |
| per-band `bin_factor` | not needed for the footprint; compute only |
| magnitude of the old NIRCam error | direction established, magnitude too noisy to report |
| pre-fix cube on disk | **delete before freezing** |

Still open: B7, C1, C2, the documentation items under D, packaging under E,
`check1_astrometry.py` uncommitted, C4 (S4G SIP) if IRAC is ever used, and the
hardening items listed in §7.

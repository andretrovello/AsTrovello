# AsTrovello — verification of the audit fixes

**Date:** 2026-09-28 · **Branch:** `v2-dev/reproject_changes-audit`
**Scope:** the fixes reported in `session_report_audit_and_validation.md`
(A1, A2, A3, B1), the new `validation.py` gates and CLI wiring, and the
consistency of the session report itself.

---

## 0. What could and could not be verified

The pipeline run described in the session report was made on a different
machine (the PyPHER logs record `/Users/andretrovello/Research/AsTrovello/...`).
This repository contains:

- the final cube `Output/datacubes/ngc1087/ngc1087_datacube_sci_1135x1277_Jy_per_pixel.fits`
- the kernels written by `check2_psf_matching.py` under `Output/checks/check2_psf_matching/`

It does **not** contain the `kernel_*_to_f2100w.fits` kernels, the JWST inputs
(`Input/` has only `PHANGS/` and `S4G/`) or any reprojected files for NGC 1087.
`Output/PSF_Kernels/` still holds the six v1.x-era `*_to_irac1` kernels.

The verification below is therefore **code review plus measurement on the
artefacts that are present**, not a re-run of the pipeline.

---

## 1. Verdict on the four fixes

| defect | code | evidence on real data | verdict |
|---|---|---|---|
| **A1** cube WCS | `cube_2_0.py:227-236` — spatial WCS written via `w_2d.to_header()`, third axis added after; bare `except` removed | cube header: `CDELT1 = 3.0807e-5 deg`, `PC` matrix present, no `CD`/SIP keys. Scale from WCS 0.11090538″/px, `PIXAREA` 0.0123 arcsec² → agree | **Fixed** |
| **A2** `NATPXAR` read | `drivers.py:167-168` — `fits_header.get('NATPXAR', config**2)` | report measures 0.993960 for f814w against predicted 0.99395 (area ratio); header chain checked: `NATPXAR` written by `bin_for_convolution`, survives `reproject_to_reference` (only WCS keys are popped) | **Fixed while HST is binned** — see §2.1 |
| **A3** crop bound | `mask_2_0.py:24-25` — `y_max + 1 + padding`, `x_max + 1 + padding` | cube 1277×1135; all 13 planes carry exactly 947,959 finite pixels; bounding box is tight on all four edges | **Fixed** |
| **B1** dilation | `drivers.py:134-136, 235-237, 311-313` — `structure=np.ones((3,3)), iterations=max(k//2, 1)` in all three drivers | not re-measured here; the identity (k×k flat square ≡ k//2 iterations of 3×3) is exact | **Fixed** |

Note on A1: the other cubes in `Output/datacubes/` (ngc1433 sci/err, ngc2903
sci/err ×3) still read `CDELT1 = 1.0` — they predate the fix and must be
regenerated before use.

Note on B1: `max(k//2, 1)` differs from the old code only at `k = 1`
(old: no dilation; new: one 3×3 pass). No real kernel is 1 px, and the guard
is necessary — `iterations=0` in scipy means *repeat until convergence*.

---

## 2. Residual gaps in the fixes

### 2.1 A2 is only fixed on the binned path

`bin_for_convolution` returns before writing any header key when
`factor <= 1` (`convolution_2_0.py:622-623`). `NATPXAR` is then absent, the
fallback fires, and the stale 0.0395″ constant is used — reintroducing the
0.61% HST-only colour term. This applies equally to convolved and to
unmatched (`copy_as_convolved`) bands, since both go through `_prepare_image`;
the only trigger is a bin factor of 1 or `None`. Latent in the current
configuration (HST bin 5), live the moment a configuration derives bin 1.

The audit recommended the fallback; that recommendation was incomplete. A
fallback to a known-wrong constant hides the defect instead of preventing it.

**Fix:** write `NATPXAR` (and `BINFACT = 1`) before the early return, so the
key is present on every path. Optionally make its absence a hard error in
`PHANGS_Driver.convert2Jansky`.

### 2.2 B2 is not fixed, and the new gate repeats it

- `convolution_2_0.py:934` still calls `inspect_kernel(kernel_norm, filt)` —
  after dividing by the sum.
- `validation.check_psf_matching` also normalises (`kernel = kernel / kernel.sum()`)
  before `_negative_power_pct`. So does `check2_psf_matching.measure`.

Neither gate can detect a sign-flipped kernel (v1.x signature: sum −0.50,
100% negative → reads as 0.00% after normalisation).

**The Check 2 numbers in the report are nevertheless genuine.** Measured on
the raw PyPHER output saved by `check2_psf_matching.py`:

```
HST  (5 kernels, 99x99)   raw sum +1.0000   raw neg power 0.00%
JWST (7 kernels, 179x179) raw sum +1.0000   raw neg power 2.54% .. 16.67%
raw and normalised negative power identical for all 12
```

So the 0.00% for HST is a real property of these kernels, not an artefact —
but that was established by reading the files, not by the gate.

**Fix:** measure negative power on the raw kernel, and raise if `sum < 0`.

---

## 3. Issues in `session_report_audit_and_validation.md`

### 3.1 The Check 3 table belongs to a different configuration

The report is headed *PHANGS-HST + PHANGS-JWST, master f2100w*, and the cube
(Sect. 9) is on a 0.1109″ grid. The Check 3 table (Sect. 5) shows:

- `grid out = 0.7500` for every band — an S4G master grid, not f2100w's
- an `irac1` row — S4G is not in this configuration
- `f200w` `grid in = 0.0307` — but Check 2 places JWST on a 0.1109″ convolution grid
- 14 bands in the table vs. "13 bands" in the summary (Sect. 1)

The table was almost certainly produced by an earlier HST+JWST+S4G run.
Either re-run Check 3 on the f2100w configuration, or label the table with the
run it belongs to. The `irac1` open item (Sect. 11) inherits the same problem.

### 3.2 The B1 section re-derives a mechanism already refuted

Sect. 2 (B1) says the earlier JWST-mask speed-up was "probably" dominated by
the dilation, because a corrected mask shrank `border_mask`. CLAUDE.md §4 B1
records the measurement that refutes this: scipy short-circuits pixels already
`True`, so a **smaller** mask makes the full-square dilation **slower**
(0.4% True → 2.32 s; 43.6% True → 1.33 s). That speed-up was the FFT.

Also: "the old form could not run at all" on the 157/281 px kernels is
machine-dependent — the audit ran 157 px on a 1600² image in 29 s. Prefer
"can fail with MemoryError".

### 3.3 The 1.06 investigation misses the known mechanism

Sect. 6 tests regularisation, the measurement, and source undersampling,
discards all three, and attributes 1.06 to the S4G configuration. CLAUDE.md
§4 D already identifies a sharper cause: `diagnostico_etapa1c.py:207-209`
measures the numerator on the **convolution grid** and the denominator on the
**raw PRF grid**. The new checks measure both on the same grid, which alone
would make 1.06 disappear.

**Cheap test:** re-run etapa1c's comparison with r50 of the target measured on
the convolution grid. If the ratio drops to ~1.00, the 1.06 was a
measurement systematic, never a kernel defect.

### 3.4 File names in Sect. 10 do not match the repository

- `check1_astrometry.py` does not exist. The only astrometry file,
  `src/astrovello/diagnostico_astrometria.py`, is a two-line stub (imports
  only). Check 1 cannot be reproduced from this repository — commit the
  script before citing its result.
- `test_pypher_regularisation.py` is in the repo as
  `pypher_regularisation_test.py`.

### 3.5 Check 2 is close to circular — state its limit

The gate convolves the **same** `PSF_CLEAN` files PyPHER used to build the
kernel, so r50 ≈ 1.0000 is nearly guaranteed by construction. It is still a
valuable gate — it catches stale kernels, PSF/kernel grid mismatch, and a
deliberately widened kernel (the 1.58 test) — but it is not independent
evidence that the science images reach the master resolution. The dissertation
text should say what it tests.

### 3.6 A hypothesis for the `irac1` 0.84%

The pre-reprojection S4G file carries SIP coefficients but `CTYPE` without the
`-SIP` suffix, so `WCS(header)` in `check_flux_conservation` ignores the
distortion. The reprojected file carries `-SIP` (re-attached by
`reproject_to_reference`). The same sky aperture then maps to slightly
different pixels on the two sides — an artefact of the check, not a flux loss.
**Untested.** Test by reading the "before" file with the suffix re-attached.

---

## 4. Review of the new validation code

`validation.py` and its wiring in `astrovello_cli_2.0.py` are sound:

- the gates sit at the right stages (Check 2 before any convolution, Check 3
  after reprojection and before unit conversion)
- thresholds are module constants, overridable only by `--skip_checks`, which
  is logged
- missing master `PIXSCALE` and source/master grid mismatch raise
- `kernel_source_filter` correctly parses the source filter instead of
  substring-matching (the master name is in every kernel filename)
- flux is area-weighted with `|det(pixel_scale_matrix)|`, correct under CD,
  PC and rotation

Interactions with still-open defects:

- **B4 now reaches the gate.** `cli:499` globs `kernel_*_to_*.fits`, so a
  leftover kernel for a previous master is checked against the current master
  and would abort the run spuriously (or, without the gate, be applied).
- **B2** as in §2.2.

---

## 5. Audit items still open

| item | status |
|---|---|
| B2 inspect before normalising | **open** (§2.2), also in `validation.py` and `check2_psf_matching.py` |
| B3 missing `PIXSCALE` skips grid guard | **open** — `convolution_2_0.py:941-957` unchanged |
| B4 kernel glob not master-scoped | **open** — `cli:499`; now also feeds Check 2 |
| B5 border `0.0` vs `NaN` | **open** — `drivers.py:137, 238` still `0.0` |
| B6 per-survey `min_blur` | open (listed in the report) |
| B7, C1, C2 | open |
| D documentation | open — e.g. `bin_for_convolution` docstring still claims "no aliasing and no loss of information" |
| E packaging | open |

---

## 6. Recommended next steps

1. Write `NATPXAR` on every path (§2.1).
2. B2 in all three places; add `sum < 0` as a hard failure (§2.2).
3. B3, B4, B5 — cheap, and B4 now affects the gate.
4. Re-run Check 3 on the f2100w configuration; correct Sects. 1, 5, 11 of the
   session report (§3.1).
5. Correct the B1 mechanism paragraph (§3.2) and add the grid-systematic
   hypothesis to the 1.06 section (§3.3).
6. Test the `irac1` SIP hypothesis before attributing it to the data (§3.6).

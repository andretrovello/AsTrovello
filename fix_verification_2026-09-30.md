# Fix verification — 2026-09-30

Independent check of the per-band convolution grid change reported in
`per_band_grid_report.md` (the `build_kernels` rewrite, `PIXSCALE` stamped
on kernels, Check 2 grouped by grid), and answers to the four questions in
§7 of that report.

Method: code review of the uncommitted diff on `v2-dev/reproject_changes-audit`
(`astrovello_cli_2.0.py`, `validation.py`). Measurements were made on
`Output/reprojected_files/ngc1087/` and on both cubes in
`Output/datacubes/ngc1087/`. The kernel pipeline was re-run in an isolated
scratch copy of `Input/` for five survey configurations. No pipeline code was
changed, and the real `Output/PSF_Kernels` and `Input/*/PSF_CLEAN` were not
touched.

---

## 0. Relevance — read this first

**The per-band grid was a blocker fix for the HST+JWST (master `f2100w`)
cube, and it holds.** Every HST+JWST cube made before 2026-09-30 convolved the
four NIRCam bands with kernels built on the MIRI grid, 3.6× too coarse for
f200w and 1.8× for LW. That is a colour error between NIRCam and everything
else. Those cubes must not be used. Sect. 0 of CLAUDE.md said "no blocking code
fix remains" for this cube. That was wrong until this change.

Nothing found in this round changes a number in the regenerated cube
(`ngc1087_datacube_sci_1109x1202_Jy_per_pixel.fits`).

| Finding | Changes the current cube? | Class |
|---|---|---|
| Per-band grid + `PIXSCALE` stamp | Yes — it is the fix | **Blocker, fixed and verified** |
| Cube shrinkage 1277×1135 → 1202×1109 | No — explained, correct behaviour (§3) | — |
| `derive_bin_factors` still uses `imgs[0]` (§4.1) | No — safe by filename sort order today | Hardening |
| `rediscover_unmatched`, preflight table still one grid per survey (§4.2) | No | Hardening |
| Check 2 "skipped" on missing master prints, does not raise (§4.3) | No | Hardening |
| Dead pre-per-grid fallback in `validation.py` (§4.3) | No | Hardening |
| PSF-name vs science-name comparison in `build_kernels` (§4.4) | No | Hardening |
| Standalone `check2_psf_matching.py`, `pypher_regularisation_test.py` (§4.5) | No | Cosmetic / dissertation |
| Methodology doc and docstrings (§5) | No | Documentation, dissertation |

**Housekeeping before freezing:** the pre-fix cube
`ngc1087_datacube_sci_1135x1277_Jy_per_pixel.fits` (27 Sep) is still in
`Output/datacubes/ngc1087/` next to the new one. Move or delete it.

---

## 1. Did the rewrite break other survey configurations?

No. The kernel stage was re-run for each configuration below:
`select_master`, then `derive_bin_factors`, then `build_kernels`, then the
Check 2 gate logic copied from the CLI. For each kernel, its `PIXSCALE` was
compared with its own image's native scale × bin. That is the comparison
`create_convolvedFITS` makes.

| Configuration | Master | Bins | Kernels / grids | Grid check | Check 2 |
|---|---|---|---|---|---|
| HST only | f814w | HST 1 | 0 — f275w/336w/438w/555w unmatched (blur 0.067–0.079″ ≈ 2 px on 0.0396″) | — | nothing to check |
| S4G only | irac2 | S4G 1 | 0 — irac1 unmatched (0.658″) | — | nothing to check |
| JWST only | f2100w | JWST 1 | 7 on 3 grids | 7/7 OK | 7/7 PASS |
| HST+JWST | f2100w | 5 / 1 | 12 on 4 grids | 12/12 OK | 12/12 PASS |
| HST+JWST+S4G | irac2 | 12 / 4 / 1 | 13 on 5 grids | 13/13 OK | 13/13 PASS |

The HST+JWST run reproduces the report's kernels exactly. Sizes were 99 / 647 / 315 / 179 px at
0.1981 / 0.0307 / 0.0630 / 0.1109″, and the f200w negative power was 2.02%.
The unmatched classifications in the HST-only and S4G-only runs are the
existing resolvability rule, unchanged by the rewrite.

Check 2 values on the three-survey (irac2) run, for the record: HST
1.0000, f200w 0.9956, LW 0.9999–1.0000, MIRI 0.9875–0.9880. The MIRI values
come from quantisation on the coarse 0.4436″ grid and are inside the gate.

---

## 2. Is a per-band `bin_factor` worth adding?

**Not for the cube footprint.** The masked border is fixed in *arcsec*, not
pixels. `driver.convolve` dilates by `kernel_size // 2` iterations on the
convolution grid, so the border is half the kernel's angular extent whatever
the bin. That extent is set by the master PSF file: the WebbPSF F2100W
`OVERSAMP` array is 724 px × 0.0277″ = 20.05″, hence ~9.9″ per side for every
band. At bin 2, f200w would get a ~323 px kernel on 0.0614″, and the border
would still be 9.9″. f200w is also not the band that limits the footprint
(§3).

**For compute only.** Deriving the factor per band (same constraints as
`choose_bin_factor`) would give f200w bin 6 (0.184″) and NIRCam LW bin 3
(0.189″), about 36× fewer pixels for f200w. That is not needed under the
stopping rule. If it is ever added, the master must keep its own factor,
because that factor defines the cube's grid.

The only levers on the border are the kernel's angular extent
(`max_extent_arcsec`, evaluated and rejected, see CLAUDE.md §7) and the
dilation radius itself. Both belong to the border decision already listed for
Paula.

---

## 3. The cube dimension change — fully accounted for

Axes: NAXIS2 (y) 1277 → 1202 (−5.9%), NAXIS1 (x) 1135 → 1109 (−2.3%).

**The NIRCam long-wave bands (f300m, f335m, f360m) set the new footprint.
f200w does not.** This was measured on the `*_on_phangs-jwst_f2100w_projection_Jy_per_pixel.fits` files.
The valid mask is `isfinite & != 0`, and the bounding box of the
intersection is 367–1475 in x and 635–1836 in y. Leaving groups out:

| Intersection without | x range | y range |
|---|---|---|
| (all bands) | 367–1475 | 635–1836 |
| f200w | 367–1475 | 635–1836 — unchanged |
| NIRCam LW | 367–1487 | 621–1840 |
| all NIRCam | 367–1497 | 547–1872 |
| MIRI | 294–1475 | 635–1836 |

The LW kernel went from 179 to 315 px at 0.0630″, so the LW border grew by
(157 − 89) × 0.0630″ = 4.28″ = **38.6 master px**. For f200w the growth was
7.18″, but f200w is not binding.

The old cube's position is recovered from its CRPIX against the master's
(both cubes share CRVAL):

| Edge | Old | New | Change | Binding band |
|---|---|---|---|---|
| y bottom | 597 | 635 | +38 | NIRCam LW before and after → the full 38.6 |
| y top | 1873 | 1836 | −37 | MIRI/others before (1872 without NIRCam), LW now |
| x right | 1498 | 1475 | −23 | MIRI before (1497), now LW, which crossed it by 23 |
| x left | 364 | 367 | +3 | MIRI f1000w both times; 3 px consistent with the B5 NaN ring (old cube predates B5) |

y: 38 + 37 = 75 px. x: 3 + 23 = 26 px. Both match the observed changes exactly.
The asymmetry is geometry: on both y edges the LW footprint limits the
cube, while in x the MIRI edge was already binding and LW overtakes it by only
23 px on one side.

This is the correct behaviour. Every band now loses the same ~9.9″ per side.
NIRCam lost less before only because its kernels were physically too small.

---

## 4. What the per-band grid leaves behind

### 4.1 `derive_bin_factors` still takes `imgs[0]`  [hardening]

The kernel grid is per band, but the *bin factor* is still derived from
`science_pixel_scale(imgs[0])`, the file that sorts first. For PHANGS-JWST
that is always a MIRI file (`miri` < `nircam`), the coarsest instrument. A
factor that is safe for the coarsest band is safe for the finer ones, so
today's runs are correct. The failure mode appears if the first file is a
fine band: a factor derived for 0.0307″ applied to MIRI would undersample the
master. With an f2100w master this becomes ~0.9 px/FWHM at bin 7, and nothing
downstream would catch it. The preflight uses the same `imgs[0]`, and the
kernel/image grid guard would still agree.

Fix (one line): use the largest native scale among the survey's bands.

### 4.2 Two more single-grid-per-survey assumptions  [hardening]

- `rediscover_unmatched` computes `grid` from `survey_imgs[0]`. It only feeds
  `PSFRESID` for bands without a kernel when run without `--create_kernel`.
  With per-band grids, the resolvability decision it recomputes can disagree
  with the one `build_kernels` made.
- `preflight._check_grids` prints every JWST pair on 0.1109″. It no longer
  describes the run. All margins are ≥ 8 px today, so no verdict changes.
  `_estimate_cube_size` similarly reads `imgs[0]` of the master's survey
  rather than the master image (cosmetic).

### 4.3 Check 2 gate  [hardening]

- A grid whose `master_<grid>_*.fits` is missing prints "CHECK 2 skipped"
  and continues. Under Invariant 8 an unverifiable gate should raise.
- `validation.check_psf_matching` falls back to unprefixed names "for PSFs
  written before the per-grid naming". That path cannot be reached, because
  `PSF_CLEAN` is always regenerated together with the kernels, and kernels
  without `PIXSCALE` already abort. It exists only to paper over a mismatch.
  Remove it.

### 4.4 Filter-name comparison in `build_kernels`  [hardening]

`filters_here` holds names from `get_sci_filter_name`, and the PSF loop
filters on `get_psf_filter_name`. If the two conventions ever differ for a
band, its PSF is skipped silently. No kernel is made, the band is neither
matched nor unmatched, and the only signal is the warning in
`print_matching_summary`. Names agree for all current surveys.

### 4.5 Scripts that assume the old naming  [cosmetic / dissertation]

- `check2_psf_matching.py` (`discover_pairs`) requires exactly one
  `master_*.fits` per `PSF_CLEAN`. JWST now has three, so the standalone
  Check 2, and its Aniano D values, skips JWST entirely. Update it before
  citing its output.
- `pypher_regularisation_test.py:58` hardcodes
  `master_PSF_MIRI_in_flight_opd_filter_F2100W.fits`, now
  `master_0.1109_PSF_...`. It is forensic history, so keep it runnable.

---

## 5. Documentation to reconcile  [dissertation]

- `astrovello_pipeline_documentation.md` describes a per-survey kernel loop
  (~l. 409–413). It also presents the grid guard as what "prevents the
  original grid bug from ever returning silently". Record in the chapter that
  the guard never fired until 2026-09-30, because PyPHER drops `PIXSCALE`,
  and that NIRCam kernels were on the MIRI grid in every earlier HST+JWST
  cube.
- `pypher_kernel_creation` docstring: "convolution grid of the source
  survey" should say "of the band".
- CLAUDE.md §8 asks for the methodology document to change in the same
  commit as the behaviour. It has not changed yet.

---

## 6. On the old-cube comparison

The report's decision not to quote a magnitude is right. The sky-aperture
scatter (7–30%) swamps a ~4% median, and the cubes in question are
discarded. The direction (NIRCam moved most) is all the data support.

---

## 7. Reproduction

Footprint attribution (§3), from the repo root:

```python
from astropy.io import fits; import numpy as np, glob, os
fs = sorted(glob.glob('Output/reprojected_files/ngc1087/*_on_phangs-jwst_f2100w_projection_Jy_per_pixel.fits')) \
     + ['Output/reprojected_files/ngc1087/ngc1087_phangs-jwst_f2100w_master_Jy_per_pixel.fits']
M = {os.path.basename(f).split('_')[2]: (lambda d: np.isfinite(d) & (d != 0))(fits.getdata(f)) for f in fs}
def bbox(v): ys, xs = np.where(v); return xs.min(), xs.max(), ys.min(), ys.max()
for drop in ([], ['f200w'], ['f300m','f335m','f360m'], ['f770w','f1000w','f1130w']):
    print(drop, bbox(np.logical_and.reduce([v for k, v in M.items() if k not in drop])))
# old cube origin in master px = master CRPIX - cube CRPIX
```

Configuration tests (§1): scratch tree `root/Input/<survey>/{PSF,galaxies/ngc1087}`
symlinked to the real inputs. The CLI module was loaded with `importlib`, and
`calculateFWHM`, `select_master`, `derive_bin_factors` and `build_kernels` were
called with `input_dir` and `kernel_dir` inside the scratch tree, followed by
the Check 2 grouping loop from `main()`. Because `build_kernels` deletes
`PSF_CLEAN` and the kernel directory, never point it at the real `Input/` for
a test.

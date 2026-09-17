# AsTrovello 2.0 — working notes for Claude

Multi-survey datacube construction for spatially-resolved SED fitting.
Ingests PHANGS-HST, PHANGS-JWST and S4G/Spitzer imaging of one galaxy and
produces a single spatially-registered datacube in Jy/pixel, at a common
angular resolution, on a common pixel grid. That cube feeds the segmentation
and SED-fitting stages (Capivara).

Author: André Almeida Trovello (IAG-USP, MSc). The companion methodological
document is `astrovello_pipeline_documentation.md` — it is the basis for the
dissertation's data-reduction chapter. **Read that document before changing
anything in the convolution or alignment stages**; it records *why* each
choice was made and cites the literature (Aniano et al. 2011; Boucaud et al.
2016) that prescribes it.

`audit_evaluation.md` is the author's point-by-point response to the audit in
Section 4 below. It supplies the debugging history the audit could not see and
records which findings the author accepts, qualifies or disputes. Where the two
disagree, the resolution has been folded into Section 4 and the disagreement
noted.

---

## 1. How to run it

The pipeline is **not** currently installable. It runs only as a script, from
its own directory:

```bash
conda activate capivara
cd src/astrovello
python astrovello_cli_2.0.py --galaxy ngc1087 --create_kernel
```

Two hard requirements that are easy to trip over:

- **CWD must be `src/astrovello`.** `astrovello_cli_2.0.py:339` computes
  `BASE_DIR = Path.cwd().parents[1]` to find `Input/` and `Output/`, and every
  module uses flat imports (`from config import ...`, `from drivers import ...`)
  rather than package-relative ones.
- **`Input/` subdirectory names must match `SURVEY_CONFIG` keys exactly**
  (`PHANGS-HST`, `PHANGS-JWST`, `S4G`). Survey identity is resolved by
  substring-matching the config key against the full file path
  (`drivers.py:47-51`), so the directory name *is* the survey label. See
  defect E4.

Useful flags: `--preflight_only` (validate config in seconds and exit),
`--create_kernel` (regenerate PSFs + kernels; also forces reconvolution),
`--bin_factor N` (override the derived convolution grid), `--allow_mixed`
(tolerate leftover files from another configuration).

### Layout

```
Input/<SURVEY>/galaxies/<galaxy>/   science mosaics
Input/<SURVEY>/PSF/                 raw PSFs/PRFs
Input/<SURVEY>/PSF_CLEAN/           resampled PSFs (written by build_kernels)
Output/PSF_Kernels/                 kernel_<filt>_to_<master>.fits
Output/convolved_fits/<galaxy>/     convolved + master + unmatched bands
Output/reprojected_files/<galaxy>/  reprojected, then *_Jy_per_pixel.fits
Output/datacubes/<galaxy>/          final cube
```

---

## 2. The four stages

1. **PSF matching + convolution** — degrade every band to the resolution of
   the worst-resolved band (the *master*).
2. **Alignment** — reproject every band onto the master's pixel grid.
3. **Unit conversion** — convert every band to Jy/pixel.
4. **Datacube** — stack, sky-subtract, mask, write.

Stage 3 lives *inside* the alignment block in the CLI, so `--mode
alignment_only` also converts units.

### Module map

| File | Role |
|---|---|
| `astrovello_cli_2.0.py` | orchestrator; master selection, bin factors, stage dispatch |
| `config.py` | per-survey constants only — things that cannot be measured |
| `drivers.py` | per-survey behaviour: HDU, invalid mask, `convolve`, `convert2Jansky` |
| `convolution_2_0.py` | PSF widths, `clean_psf`, kernel generation, binning, convolution |
| `alignment_2_0.py` | file discovery by header, `reproject_to_reference` |
| `units_2_0.py` | thin dispatcher to the driver's `convert2Jansky` |
| `cube_2_0.py`, `mask_2_0.py` | cube assembly, footprint/signal masks, sky subtraction |
| `preflight.py` | validate the whole configuration before any compute |
| `utils_2_0.py` | WCS pixel scale, wavelength ordering |
| `diagnostico_etapa1*.py` | read-only forensic scripts from the v1.x debugging |

---

## 3. Invariants — do not break these

These are the load-bearing decisions. Most of the non-obvious code exists to
enforce one of them.

1. **A kernel is valid only on the grid of the PSFs used to build it.**
   PyPHER returns the kernel on the input PSFs' grid (Boucaud Algorithm 1:
   `N = N_b ; p = p_b`). Since the kernel is applied to a *science image*, the
   PSFs must be resampled to the **convolution grid**, not to any detector's
   native grid. This was v1.x's central bug: kernels built on the IRAC grid
   and applied on the HST grid gave ~12x too little blur. Enforced by
   `clean_psf(target_pixel_scale_arcsec=...)` and the guard in
   `pypher_kernel_creation`.

2. **Convolution before reprojection.** The master (FWHM 1.72") on the final
   0.75"/px grid is at 2.29 px/FWHM — above Nyquist, so reprojecting an
   already-blurred image loses nothing. An unconvolved HST image (FWHM 0.08")
   on that grid would be at 0.11 px/FWHM, undersampled ~17x.

3. **The master is chosen by r80, never by Gaussian FWHM.** PSF matching is
   dominated by the wings. The Gaussian fit responds to the core and orders
   the two IRAC channels *backwards* (they differ by ~4%). Picking the wrong
   master turns smoothing into deconvolution. Verified against the PRF files:
   r80(IRAC1) = 2.0864", r80(IRAC2) = 2.1878" — IRAC2 is the master.

4. **Binning before convolution uses the MEAN, not the sum.** Each binned
   pixel still means "e/s per *native* pixel", which is what step 1 of
   `convert2Jansky` expects. A sum would change the native area and break the
   unit conversion by `bin^2`.

5. **Pixel scales for science images come from the WCS, never from
   `config.py`.** The config constant is the *PSF reference* scale, which for
   S4G (1.221") is not the mosaic scale (0.75"). Conflating them was the
   original unit bug. (See defect A2 — `convert2Jansky` still violates this.)

6. **The binning factor is a property of the TARGET, not the source survey.**
   It depends on which filter ends up master, so it must be derived per run
   (`choose_bin_factor`). The `config.py` value is a fallback only.

7. **Information travels with the data.** Provenance goes in the header
   (`PSFTARGT`, `PSFMATCH`, `PSFRESID`, `NATPXAR`, `BINFACT`) and discovery
   selects on the header, not the filename. (See defect A2 — `NATPXAR` is
   written but never read.)

---

## 4. Known defects

Findings from a full read-only audit (2026-09-17), reconciled against the
author's response in `audit_evaluation.md`. Every quantitative claim below was
verified against the real data in `Input/` or the real outputs in `Output/`.
Nothing here has been fixed yet.

Status key: **[agreed]** author concurs; **[agreed, reweighted]** author
concurs and considers it more serious than first placed; **[qualified]**
scope narrowed after author's evidence; **[corrected]** an explanation in the
audit or the response was wrong and has been revised.

### A. Corrupts science output

#### A1. Datacube WCS is wrong — CDELT = 1 deg/px  [CRITICAL]

`cube_2_0.py:217-224` builds the 3D WCS by copying `crpix/crval/cdelt/ctype/
cunit` element-by-element. It never copies the **CD or PC matrix**.

The S4G master uses a CD matrix with no CDELT:

```
CD1_1 = -0.00020833 (= 0.75"/px)    wcs.has_cd() = True
wcs.cdelt = [1.0, 1.0]              <- astropy default when CD is in use
```

So `cdelt` copies as `[1.0, 1.0]`. Confirmed in a real output,
`Output/datacubes/ngc2903/ngc2903_datacube_sci_6615x7560_Jy_per_pixel.fits`:

```
CDELT1 = 1.0 / [deg] Coordinate increment at reference point
CDELT2 = 1.0 / [deg]
```

Every existing cube claims 1 degree per pixel instead of 0.75 arcsec — a
factor of 4800. Any downstream sky coordinates, arcsec apertures or physical
scales computed from these cubes are wrong. `PIXAREA` is computed separately
via `proj_plane_pixel_area` and *is* correct, which is why this went unnoticed.

`cube_2_0.py:221` also swallows the failure with a bare `except Exception:
continue`.

**[agreed]** The author had recorded this as a pending item (`use
w_2d.sub([1,2,0])`) but classified it *latent* on the grounds that "nothing
uses the cube's WCS today" — an assumption about future use, not a property of
the data. It is not latent: the header is already wrong on disk.

Fix — the author's intended idiom is correct, verified:

```
w_2d.sub([1,2,0])  ->  PC1_1 = -0.00020833, CDELT1 = 1.0
                       proj_plane_pixel_area recovers 0.7500 arcsec/px  [correct]
```

`sub` converts CD to the equivalent PC+CDELT form and preserves the scale.
**Gotcha:** it initialises the new third axis to `CRPIX3 = 0.0`, not 1.0, so
the existing `w_3d.wcs.crpix[2], crval[2], cdelt[2], ctype[2] = 1, 0, 1,
'FILTER'` line must be kept after the `sub` call. Remove the bare except.
Re-make any cube already produced.

#### A2. `NATPXAR` is written but never read — HST fluxes ~0.6% too high

`astrovello_pipeline_documentation.md` states twice (Sects. 5 and 7) that
`NATPXAR` is what stops the unit conversion reading the binned WCS area by
mistake. Grep shows `NATPXAR` and `BINFACT` appear **only** at
`convolution_2_0.py:661,663`, where they are written. Nothing reads them.

`drivers.py:153` uses the config constant instead:

```python
native_pixel_area_arcsec2 = self.config["pixel_scale_arcsec"] ** 2   # 0.0395
```

Measured against `Input/PHANGS/galaxies/ngc1087/..._f814w_..._exp-drc-sci.fits`:

| | arcsec/px | area (arcsec^2) |
|---|---|---|
| WCS (true drizzled) | 0.039620 | 0.00156974 |
| `config.py` constant | 0.039500 | 0.00156025 |

Dividing by the smaller (config) area inflates the result: **all five HST
bands come out 0.61% too bright.** It is a pure colour term between HST and
the IR bands — exactly the systematic that propagates into SED-fitted ages
and dust.

Fix: `fits_header.get('NATPXAR', self.config["pixel_scale_arcsec"]**2)`.
Note `bin_for_convolution` returns early when `factor <= 1`
(`convolution_2_0.py:623`), so `NATPXAR` is absent whenever bin == 1 — the
fallback matters.

#### A3. Off-by-one in `crop_to_mask_bbox`

`mask_2_0.py:19-24`: `coords.max(axis=0)` is an **inclusive** index used as an
**exclusive** slice bound.

```
valid block rows 2..7, cols 3..8 (6x6)
code slices mask[2:7, 3:8] -> (5,5)   # 25 of 36 valid pixels kept
```

Needs `y_max + 1` and `min(ny, y_max + 1 + padding)`. Called **twice** per
cube (`cube_2_0.py:153` and `:208`), so two rows and two columns of real
signal are lost off the top/right edge.

### B. Convolution stage

#### B1. `binary_dilation` with a full-kernel square structuring element  [CRITICAL for usability]

`drivers.py:130-131` (identical at JWST `:216-217`, S4G `:289-290`):

```python
structure = np.ones((kernel_size, kernel_size))
expanded_border = binary_dilation(border_mask, structure=structure)
```

`kernel_size` is the **full kernel array width**. With `max_extent_arcsec=0`
(the correct default per Aniano's 99.9%-energy criterion), the IRAC PRF on the
0.198"/px HST grid gives a **157x157** kernel. scipy does not decompose a flat
square footprint. Measured on the real configuration (8000^2 binned 5x ->
1600^2):

```
full-square dilation    : 29.0 s  per band
iterated 3x3 equivalent :  0.02 s  (identical result, np.array_equal verified)
structure 201x201 on 1500x1500 : MemoryError
```

~1400x slower, and it hard-fails on larger kernels. **This is the most likely
cause of the convolution-stage difficulty.**

Fix: a `k x k` flat square dilation is exactly `k//2` iterations of a 3x3
square. Drop-in replacement:

```python
binary_dilation(border_mask, structure=np.ones((3,3)), iterations=kernel_size//2)
```

Separately, note this correctly discards `78 px = 15.5"` of border per side.
That is the right behaviour — just confirm it is acceptable for the fields.

**[agreed, reweighted]** The author reports that the convolution stage was
repeatedly slow and was at one point killed by the OS. That was diagnosed as
FFT cost and fixed with `bin_for_convolution` (~2.5 GB -> ~0.04 GB per array),
a genuine and necessary fix — but the stage stayed slower than the FFT size
alone justified, and the residual was attributed to general FFT cost. This
dilation is the likelier explanation for that residual.

**[corrected] One attribution in `audit_evaluation.md` is backwards.** The
response suggests that the speed-up observed after fixing the JWST
invalid-pixel convention came from this dilation, because a corrected mask
shrinks `border_mask`. Measured, the dependence runs the other way — scipy
short-circuits pixels already `True`, so a *smaller* mask is *slower*:

```
True fraction   0.4%  ->  2.32 s
True fraction   1.8%  ->  2.29 s
True fraction   9.5%  ->  2.04 s
True fraction  27.6%  ->  1.66 s
True fraction  43.6%  ->  1.33 s     (900x900, 81x81 structure, best of 3)
```

So that particular speed-up was the FFT after all — the original explanation
was right. B1 stands on its own measurement (28 s vs 0.02 s); it just does not
own that log observation. Worth recording so the wrong mechanism is not
re-derived later.

**Interaction with B6 — order matters.** A coarser convolution grid shrinks the
kernel array too, so B6 largely hides B1's symptom:

```
bin  5: grid 0.1981"/px  image 1600x1600  kernel 157px -> full-square 28.27 s
bin 14: grid 0.5547"/px  image  571x571   kernel  57px -> full-square  0.46 s
```

Applying B6 first would make the slowness disappear without removing the
failure mode, leaving the `MemoryError` cliff latent for any future
configuration with a finer grid. **Do B1 before or together with B6, not
after.**

#### B2. `inspect_kernel` cannot see the failure it exists to catch

`convolution_2_0.py:934-938` normalises **before** inspecting:

```python
kernel_norm = (kernel_data / ksum)
inspect_kernel(kernel_norm, filt)
```

Two of its four metrics are vacuous:

- **sum** is 1.0 by construction — it can never detect a normalisation problem.
- **negative power** is measured after the sign flip. Measured on the kernels
  currently in `Output/PSF_Kernels/`:

```
raw kernel : sum = -0.4958   negative power = 100.00%
after /sum : sum = +1.0000   negative power =   0.00%   <- what inspect_kernel prints
```

The documentation's headline validation number — *"Measured: 0.00% for the
five HST -> IRAC kernels"* (Sect. 4) — is exactly the value this blind spot
produces for a kernel PyPHER returned **100% negative**. `diagnostico_etapa1b.py`
found that all-negative signature by inspecting the raw file on disk; the
in-pipeline check can no longer see it.

This does not mean the current v2 kernels are bad — `diagnostico_etapa1c.py`'s
r50 test (0.08 -> ~1.06) is the right test and it passes. The point is that
`inspect_kernel` can no longer distinguish a healthy kernel from a
pathological one, **so 0.00% must not be cited as evidence in the
dissertation** until the metric is measured pre-normalisation.

**[agreed]** Context from the author: the raw kernels *were* measured
pre-normalisation during the v1.x debugging, by `diagnostico_etapa1b.py`
reading the files on disk. That is where the "100% negative, sum -0.50"
signature came from, and it was one of two independent indicators that led to
the grid diagnosis. So the v1.x finding was legitimately obtained; what broke
is the *in-pipeline* check, written later and placed after the normalisation.

That sharpens rather than softens the finding: the working diagnostic lives in
a standalone script that the pipeline never calls, so **if the grid bug
returns, nothing in the pipeline will notice.** The fix restores a guard, not
just a dissertation number.

Fix: call `inspect_kernel(kernel_data, filt)` before normalising.

Related: the doc reports the r50 ratio as "~1.06" and treats it as passing
without stating a tolerance. A 6% residual in the applied blur should be
either justified or reported as Aniano's `D` statistic instead.

#### B3. The grid guard is skipped exactly when it matters most

`convolution_2_0.py:941-957`: if `PIXSCALE` is missing it falls back to the
WCS; if that also fails, `k_px = None` and **the check is silently skipped**.

`Output/PSF_Kernels/` currently holds six v1.x kernels with no `PIXSCALE`,
25x25, summing to -0.50. Running without `--create_kernel` would apply them
with the "safety net against a returning grid bug" disabled — the stale-kernel
case is precisely the one it cannot catch.

Fix: make a missing `PIXSCALE` a hard error.

#### B4. Kernel glob is not master-scoped

`astrovello_cli_2.0.py:469` globs `kernel_*_to_*.fits`, while
`rediscover_unmatched` (`:244`) correctly globs `kernel_*_to_{master}.fits`.
Without `--create_kernel`, a band with no kernel for the current master but a
leftover kernel for a *previous* master is classified unmatched by one path
and convolved with the wrong kernel by the other. If both masters share a
grid, B3 will not catch it either.

#### B5. Inconsistent invalid-pixel convention between drivers

HST/JWST set the dilated border to **`0.0`** (`drivers.py:132,218`); S4G sets
it to **`np.nan`** (`:291`). `intersection_footprint_mask` defines valid as
`~isnan`, so HST/JWST borders would count as valid. This is rescued only
because `units_2_0.py:22` blanket-converts `data == 0` to NaN after
reprojection — fragile, because `reproject_interp` interpolating across the
0/valid boundary produces small non-zero values that survive, and genuinely
zero science pixels get killed.

Fix: set NaN in all three drivers.

#### B6. `min_blur` is global rather than per-survey — 8x more pixels than needed

`astrovello_cli_2.0.py:115-116` takes the minimum required blur over **all**
bands and applies it to **every** survey:

```
min blur over ALL bands (used) : 0.6583"  <- set by irac1, a pair that is then SKIPPED
min blur over HST bands only   : 2.1850"

code (global min_blur) -> HST bin  5, grid 0.1981"/px, image 1600x1600
per-survey min_blur    -> HST bin 14, grid 0.5547"/px, image  571x571 (0.13x pixels)
```

The HST grid is constrained by an IRAC1<->IRAC2 pair that lives on a different
survey's grid *and* is then skipped as unresolvable. Computing `min_blur` from
the bands a given survey will actually match would cut the HST convolution to
~13% of the pixels while still giving 3.1 px/FWHM on the master and 3.9 px for
the HST blur. Together with B1 this is most of the compute problem.

**[agreed]** The author's diagnosis of the root cause is worth keeping
verbatim, because it names the gap precisely: the principle "the binning factor
is a property of the *target*" is correct and was the fix for a real bug, but
it was implemented as a single global minimum when the correct reading is
*"property of the target **as seen from each source grid**"*.

**Caveat before applying.** Bin 14 lands the master at 3.1 px/FWHM against a
`choose_bin_factor` constraint of `px_per_fwhm = 3.0` — i.e. essentially no
headroom left in the derivation, versus 8.7 today. Run `--preflight_only` and
read the full per-pair table before committing, since the margins for the HST
pairs move too. See also the ordering constraint in B1.

#### B7. Smaller convolution notes

- `convolution_2_0.py:122` — `np.interp(fraction, cum, ...)` requires
  increasing `xp`. After background subtraction the cumulative curve is
  **non-monotonic in ~19% of steps** (2493/12851 for IRAC1). At
  `fraction=0.8` it is still rising steeply so the answer is stable (it
  reproduces the published numbers), but at 0.9/0.95 it would fail silently.
  Consider `np.maximum.accumulate`.
- `get_half_light_radius` normalises by the flux inside `r_max =
  min(shape)/2`, so r80 means "80% within the array half-width". Fine for
  ranking the two IRAC channels (same array size); less apples-to-apples when
  comparing across surveys with very different PSF array extents.
- `clean_psf:311` — `block_reduce` aligns blocks to index 0, not to the PSF
  peak, so the core is smeared slightly asymmetrically before `_recenter_odd`
  corrects the offset. Sub-target-pixel, but real.
- `convolution_2_0.py:933-935` — the comment "PyPHER sometimes returns the
  kernel with the global sign inverted, and that is cosmetic" deserves more
  scepticism. An all-negative kernel summing to -0.5 signals something wrong
  upstream, even if normalisation happens to recover a usable kernel.

### C. Reprojection stage

#### C1. Stale SIP coefficients survive into the reprojected header

`alignment_2_0.py:182-190` pops the linear WCS keys but **not** `A_ORDER`,
`A_i_j`, `B_*`, `AP_*`, `BP_*`, nor `CROTA2`, `LONPOLE`, `LATPOLE`. So
`img_base_new_header` keeps the *source* image's SIP while the linear WCS is
replaced by the reference grid.

Today the S4G reference's own SIP keys overwrite them via
`w_ref.to_header(relax=True)`, so it gets away with it. The moment the source
has SIP and the reference does not (or they differ in order), the output
carries one image's distortion polynomial on another image's grid — silently
wrong astrometry.

Fix: pop the SIP keys explicitly.

#### C2. Re-attaching `-SIP` is a silent no-op when binning stripped the coefficients

`bin_for_convolution:650-656` correctly deletes SIP coefficients (they are in
pixel units). But `apply_sip_correction` is a static config flag, so
`alignment_2_0.py:152-153` re-attaches `RA---TAN-SIP` regardless. Verified:

```
WCS built with -SIP CTYPE but no coefficients -> w.sip is None
```

No exception. The requested correction silently does nothing, and the written
header advertises a distortion it does not carry. Currently latent because
S4G derives bin == 1 so the coefficients survive; it activates the moment
`choose_bin_factor` gives S4G a factor > 1.

#### C3. `reproject_interp` uses bilinear by default

The justification in Sect. 6 of the doc is sound and correctly identifies the
correlated-noise argument. But `reproject_interp` defaults to **bilinear**,
not the higher-order interpolation the Nyquist argument implicitly assumes.

**[qualified — this is a documentation item, not a code fix.]** The author
pushed back and is right. At 2.29 px/FWHM the field is above Nyquist and the
extra smoothing from bilinear is small against the 1.72" PSF the image already
carries. The related question — whether `reproject_adaptive` recovers SNR — was
tested and answered no, because the kernel's correlation area (~4200 native px)
is ~12x an output pixel (~350 native px), so the averaged pixels share a noise
realisation.

Action: **state the interpolation order in the documentation and note that the
alternative was tested.** Do not switch to `order='biquadratic'` on the
assumption it is better; measure first if it is ever worth revisiting.

### D. Documentation vs. code

`astrovello_pipeline_documentation.md` is accurate on the physics and on the
survey facts — the S4G SIP-without-suffix claim, the 0.75" mosaic scale,
MJy/sr, and the r80 values were all verified against the real files. These are
the places where it describes something the code does not do:

| Doc claim | Reality |
|---|---|
| Sects. 5, 7: `NATPXAR` "closes the unit bug"; conversion reads the native area from the header | Written, never read. Config constant used instead (A2) |
| Sect. 10: "a band cannot be both matched and unmatched (**raises**)" | No such check exists. `print_matching_summary:288` prints a warning |
| Sect. 10: "missing kernels for the current master **raise** with a clear message" | `rediscover_unmatched` silently reclassifies them as unmatched |
| Sect. 9: preflight checks "master-grid consistency" | `_check_master_grid` takes `errors` and never appends to it — it only prints a reminder, and only when bin > 1 |
| Sect. 3: `get_fwhm` carries a warning not to use it for ranking | True — but `derive_bin_factors:128` then passes that same Gaussian FWHM as the master width for the grid decision |
| Sect. 5: binning by mean has "no aliasing and no loss of information" | Overstated. A box filter + decimation is far better than interpolation, but its sinc transfer function does not cut off cleanly at Nyquist — it aliases, weakly. **Soften before a referee sees it** |
| Sect. 8: "crops again by the mask **with padding**" | `padding=50` is defeated: the mask is applied at `cube_2_0.py:205` *before* the crop, and `crop_to_mask_bbox:28` re-applies it, so the padded ring is guaranteed all-NaN |
| Sect. 1: "four stages" | Unit conversion runs inside the alignment block (`cli:592`) |

**[corrected] The r50 tolerance criticism was imprecise.** The audit said the
"~1.06" ratio is treated as passing *"without stating a tolerance"*. The **code**
does state one — `diagnostico_etapa1c.py:222-230` grades `0.85 <= razao <= 1.15`
as OK, with coarser bands for marginal/failure, and line 215 additionally
refuses to rule at all when `px_src > 0.4 * r50_tgt` ("indeterminado, grade
grosseira"). It is the *documentation* that omits both. Credit where due: the
script is better than the doc implies.

The author attributes the residual 6% to discretisation, which is the right
kind of explanation, and there is a sharper mechanism visible in the code:
`diagnostico_etapa1c.py:207-209` measures the numerator on the **convolution
grid** (`px_src`) and the denominator on the **raw PRF grid** (`px_tgt_raw`).
Numerator and denominator are therefore quantised differently, so a few percent
offset is expected by construction, independent of kernel quality.

Actions: quote the code's +/-15% band and the quantisation guard in the
documentation; measure both r50 values on the same grid to remove a known
systematic; and report Aniano's `D` alongside, which the author endorses.

Also: `--apply_mask` is `store_true` (default **False**) while
`create_data_cube` defaults `apply_mask=True`. By default no signal mask is
built and `mask_final = np.isfinite(cropped_images[0])` — band 0 alone defines
the cube footprint. The doc describes the masked path as the normal one.

### E. Repository state

- **E1. `src/astrovello/__init__.py` is dead.** `import astrovello` raises
  `ModuleNotFoundError: No module named 'astrovello.alignment'`. It imports
  v1.x module names (`alignment`, `convolution`, `units`, `mask`, `cube`,
  `utils`) and v1.x symbols (`S4G2PHANGS_reproject`, `final_clean_psf`,
  `soma_img`, `create_cutout`, ...) that no longer exist.
- **E2. `pyproject.toml` entry points are broken**:
  `astrovello.astrovello_cli:main` (that file was deleted on this branch) and
  `astrovello.batch_runner:main` (never existed). `pip install astrovello` as
  the README instructs yields two non-functional console scripts.
- **E3. Flat imports + `CWD.parents[1]`** mean the pipeline runs only from
  `src/astrovello`. The dot in `astrovello_cli_2.0.py` also makes it
  un-importable as a module. Renaming to `astrovello_cli_v2.py` plus
  package-relative imports would make it installable.
- **E4. `Input/` directory names do not match `SURVEY_CONFIG` keys.** The repo
  has `Input/PHANGS/`, config expects `PHANGS-HST`. `cli:407` does
  `DRIVERS[survey]` on directory names -> `KeyError: 'PHANGS'`. **The current
  code cannot run against the current data layout without renaming those
  directories.** **[qualified]** This is a branch-state artefact, not a
  pipeline defect — the author has been running these stages successfully with
  directory names matching the local configuration. It belongs in the packaging
  group, not with the correctness defects.
- **E5. `get_survey` is fragile** (`drivers.py:47-51`): it substring-matches
  the config key against the full path, and the keys are uppercase while the
  filenames are lowercase (`hlsp_phangs-hst_...`). Survey identity therefore
  depends entirely on the directory name, and returns `None` silently
  otherwise — which becomes `drivers[None] -> KeyError` at `cli:142`. Better:
  read `TELESCOP`/`INSTRUME` from the header (both keys are already in
  `config.py` and unused).
- **E6. `tests/` contains no tests** — two notebooks and `classes.py`, a stale
  copy of `drivers.py` still using the old `{mode}_suffix` key. Several of the
  defects above (A1 especially) are one-assertion-detectable; a handful of
  pytest cases on synthetic FITS would pay for itself.
- **E7. README describes v1.x**: "Reprojects S4G images onto the HST pixel
  grid" — the opposite of the current design. No JWST.
- **E8.** `--error` and `--valid_pixels_cut` are declared but unimplemented.
  `config.py:30` has a typo: `Notesf`.

---

## 5. Suggested fix order

Reconciled with `audit_evaluation.md`. The author's one adjustment — promoting
B2 to sit with A2, because a number cited as evidence in dissertation material
depends on it — is adopted.

| # | Item | Why |
|---|---|---|
| 1 | **A1** — cube WCS | Corrupts every cube produced; factor 4800 in angular scale. Fix idiom verified (`w_2d.sub([1,2,0])`, keep the axis-3 defaults) |
| 2 | **B1** — dilation | One line, ~1400x, and the likeliest explanation for the unexplained residual slowness. **Must precede or accompany B6** |
| 3 | **A2** — read `NATPXAR` | One line; removes a 0.61% *colour* systematic before SED fitting |
| 4 | **B2** — inspect before normalising | One line; restores a guard the pipeline currently lacks, and a dissertation number depends on it |
| 5 | A3, B3, B4, B5 | Correctness, cheap |
| 6 | **B6** — per-survey `min_blur` | Real compute win; run `--preflight_only` first (master drops to 3.1 px/FWHM) |
| 7 | C1, C2 — SIP keys | Latent today, silently wrong astrometry when it activates |
| 8 | Documentation reconciliation | The "raises" claims, the "no aliasing" claim, the r50 tolerance and grid systematic, the `reproject_interp` order (C3) |
| 9 | Packaging (E1-E5) | When handing the code to someone else |

Two items are **decisions, not fixes**, and are the author's to make: whether
the 15.5"-per-side border loss from the dilation is acceptable for these
fields, and whether `--apply_mask` should default on. The author intends to
raise both with Paula rather than settle them in code.

---

## 6. What is working well

Worth keeping in view when refactoring — these are the parts that are right:

- The v1.x grid diagnosis and its fix. Building kernels on the grid where they
  are applied, enforced by `clean_psf(target_pixel_scale_arcsec=...)` plus the
  guard in `pypher_kernel_creation`, matches Boucaud Algorithm 1.
- Master selection by r80 rather than Gaussian FWHM, with the correct
  justification. Reproduced independently from the PRF files.
- Declining to match IRAC1<->IRAC2 rather than forcing a sub-pixel kernel, and
  recording `PSFRESID` so the decision travels with the data.
- `_recenter_odd` — catching that the naive parity crop *shifts* rather than
  recentres, and connecting that to a per-region colour error.
- Binning by mean with the per-native-pixel reasoning.
- The literature mapping in doc Sects. 4 and 6. The verbatim Aniano/Boucaud
  quotations tied to specific functions transfer to the dissertation chapter
  nearly as-is.
- Writing provenance to the header and selecting on it rather than on
  filenames. `NATPXAR` only needs to actually be read for the pattern to be
  complete.

---

## 7. Design rationale recorded by the author

Context from `audit_evaluation.md` that is not derivable from the code, and
that should survive future refactors:

- **"An error that affects some bands and not others is a colour error, and a
  colour error becomes an age error."** This is the rule the whole redesign was
  built around, and it is why a sub-percent defect like A2 is not a rounding
  detail: a uniform 0.61% would be harmless, a 0.61% offset between the
  UV/optical and the IR is not. Apply this test when judging the severity of any
  new photometric defect.
- **`bin_for_convolution` was a real and necessary fix**, not an optimisation:
  it took per-array memory from ~2.5 GB to ~0.04 GB after the convolution stage
  was killed by the OS. B1 explains the *residual* slowness that survived it; it
  does not supersede it.
- **The IRAC1<->IRAC2 skip and the r80 master criterion came from measurement,
  not from theory** — two independent indicators (the all-negative kernels found
  by `diagnostico_etapa1b.py`, and the r50 reproduction test in
  `diagnostico_etapa1c.py`) drove the v1.x grid diagnosis. Those scripts are
  forensic history, not dead code; keep them runnable.
- **`reproject_adaptive` was evaluated and rejected on evidence**, not skipped.
  Same for the wing-truncation default (`max_extent_arcsec = 0`). Do not
  "improve" either without re-measuring.

### The pattern worth remembering

The author's own summary of what the defects share is the most useful line in
either document, and is worth restating: `NATPXAR` written but never read,
guards documented as raising but only printing, a metric measured after the
transformation that hides what it looks for — in all three the *intent* was
recorded correctly and the *implementation* drifted. That is the same failure
mode the pipeline was rebuilt to eliminate, moved up one level: no longer in
the data, but in the relationship between the code and its own documentation.

Practical consequence: **treat a claim in `astrovello_pipeline_documentation.md`
as a specification to be checked against the code, not as a description of it.**

---

## 8. Conventions for edits

- Do not change the convolution or alignment algorithms without re-reading
  `astrovello_pipeline_documentation.md` first — the non-obvious code is
  load-bearing and the reasons are recorded there.
- When a behaviour changes, update that document in the same commit. It is
  dissertation material, not just a README.
- Comments and docstrings are mixed English/Portuguese. New code in English.
- Bibliography caveat carried by the doc itself: only Aniano et al. (2011) and
  Boucaud et al. (2016) were verified against the PDFs. Every other reference
  in Sect. 12 needs its year/volume/page checked on ADS before citing.

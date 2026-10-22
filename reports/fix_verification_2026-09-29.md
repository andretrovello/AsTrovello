# Fix verification — 2026-09-29

Independent check of the second fix session (items 2.1, 2.2, B3, B4, B5, B6
from `fix_verification_2026-09-28.md`), plus a re-examination of the irac1
0.84% Check 3 deviation after the author reported the SIP hypothesis as
refuted.

Method: code review of the uncommitted diff on `v2-dev/reproject_changes-audit`
and measurements on the files in this checkout (`Input/S4G/galaxies/ngc1087/`,
`Output/convolved_fits/ngc1087/`, `Output/reprojected_files/ngc1087/`). No code
was changed.

---

## 0. Relevance — read this first

**None of the findings below changes a number in the HST+JWST (master
`f2100w`) cube. Only one (C4) matters, and only for cubes that include S4G.**
Everything else is hardening: making the guards tighter so that fixed bugs
cannot come back. It is not correcting output.

| Finding | Changes a cube on disk today? | Class |
|---|---|---|
| C4 — S4G SIP applied | **Only S4G cubes (irac2 master).** f2100w cube: no | **Immediate if IRAC bands are used**; ~5-line fix |
| NATPXAR fallback (§2.1) | No — HST is binned in current runs, key present on disk | Hardening |
| No `abs(sum − 1)` gate (§2.2) | No — all current kernels sum to +1.0000 | Hardening |
| B3 message order, B6 margin, B5 comment (§2.3–2.6) | No | Cosmetic |

Scale of C4 for NGC 1087: ≤ 0.17" inside ~1.5′ (most of the disc, ~10% of the
1.72" PSF FWHM), ~0.8" in the outer disc. The irac1 flux error (0.84%) is
already under the 1% gate. Worth the fix if IRAC is used. It does not
invalidate existing results. The Gaia check (§4.5) is needed only to *claim*
the SIP result in the dissertation, not to apply the fix.

**Trajectory.** The first audit found errors in the data on disk (cube WCS
off by 4800×, 0.61% HST colour error, lost edge pixels). The second round
found holes in the guards. This round found no errors in the fixes
themselves. That is convergence, not circling.

**Stopping rule (proposed):**

1. Decide which cube(s) the dissertation uses. HST+JWST only: no code fix
   remaining. With S4G: apply C4 first.
2. One clean run: empty `Output/convolved_fits`, `reprojected_files` and
   `datacubes` for the galaxy, run with `--create_kernel`. This removes the
   stale-file concern (§2.1) without code changes.
3. If both gates pass, freeze and tag that commit as the dissertation version.
4. Everything else goes to a backlog. From here, an item is a blocker only if
   it changes a number in the cube that is actually used.

---

## 1. Summary

| Item | Verdict | Remaining |
|---|---|---|
| 2.1 NATPXAR on every path | **Holds** | Fallback in `drivers.py` still present; stale files reused (§2.1) |
| 2.2 raw-kernel inspection | **Holds** | No gate on `|sum − 1|`; a +0.50 kernel passes (§2.2) |
| B3 missing PIXSCALE raises | **Holds** | Cosmetic only (§2.3) |
| B4 master-scoped glob | **Holds** | — |
| B5 NaN border | **Holds** | Label the 0.7% as synthetic (§2.5, §3) |
| B6 per-survey `min_blur` | **Holds** | Thin margin, still Gaussian-FWHM driven (§2.6) |
| irac1 0.84% | **Explained: SIP, applied when it should not be** | Absolute Gaia test outstanding (§4) |

The central result reverses the session report's conclusion: the S4G SIP
coefficients appear to be **spurious**, and applying them is what turns an
exact integer-pixel copy of irac1 into a genuine resampling. With SIP off on
both sides, Check 3 for irac1 gives **1.0000, 0.00% scatter**.

---

## 2. The six fixes

### 2.1 NATPXAR written on every path — holds, one gap

`bin_for_convolution` now writes `NATPXAR` and `BINFACT = 1` on the early
return. All three routes (convolved, unmatched, master) go through
`_prepare_image`, so all are covered. The key survives reprojection (checked:
`ngc1087_phangs-hst_f814w_on_phangs-jwst_f2100w_projection.fits` carries
`NATPXAR = 0.0015697`).

**Gap — the fallback is still there.** `drivers.py:178-179` still reads
`fits_header.get('NATPXAR', self.config["pixel_scale_arcsec"] ** 2)`, and the
comment at `:176-177` ("bin_for_convolution returns early when factor <= 1 and
writes no NATPXAR") is now false.

This is not hypothetical. `create_convolvedFITS` (`convolution_2_0.py:915`)
and `copy_as_convolved` (`:862`) skip any existing file whose `PSFTARGT`
matches the current master. A run without `--create_kernel` /
`--force_convolution` therefore reuses files written before the fix. This
checkout contains one: `ngc1087_phangs-jwst_f2100w_master.fits` has
`NATPXAR = None`, `BINFACT = None`. For an HST band on that path the 0.61%
colour error returns silently.

Fix: make a missing `NATPXAR` a hard error in the PHANGS-HST `convert2Jansky`,
with the same reasoning as B3. Optionally also treat a missing `NATPXAR` as
"stale" in the two skip checks.

### 2.2 Negative power on the raw kernel — holds, but the real guard is the sign

Consistent in all three places (`convolution_2_0.create_convolvedFITS` +
`inspect_kernel`, `validation.check_psf_matching`,
`check2_psf_matching.measure`). No new divergence between copies.

Observation: negative power is invariant under multiplication by a positive
scalar, and every negative-sum kernel now raises *before* the metric is
computed. So "measured on the raw kernel" and "measured on the normalised
kernel" are identical for every kernel that reaches the metric. The protective
change is the `ksum < 0` check, which is correct.

**Gap — magnitude of the sum.** The v1.x signature was `|sum| = 0.50`. A
kernel summing to **+0.50** passes all three gates. The current PyPHER
kernels measure raw sum +1.0000, so a check such as
`abs(ksum - 1) > 0.01 → fail` costs nothing and closes the remaining half of
the signature. Put it in all three places at once.

Minor: `check2_psf_matching.measure` divides by `ksum` without a zero guard.

### 2.3 B3 — missing PIXSCALE raises — holds

The silent-skip path is gone. Cosmetic points:

- In `create_convolvedFITS` the sign check runs before the PIXSCALE check, so
  a stale v1.x kernel (no PIXSCALE, sum −0.50) is reported as "negative sum"
  rather than "leftover, regenerate with --create_kernel". Swapping the order
  gives the more actionable message.
- The WCS fallback for the kernel scale, on a header with no WCS keys, yields
  astropy's default 1 deg/px → 3600"/px. The mismatch check then raises with
  "factor ~0.0". Not silent, but confusing. Kernels written by this pipeline
  always carry PIXSCALE; dropping the fallback is simpler.

### 2.4 B4 — master-scoped glob — holds

Scoped at both sites (`build_kernels` count, `astrovello_cli_2.0.py:306`; main
run, `:568`), consistent with `rediscover_unmatched`. `convolved_dict` and the
Check 2 gate now receive only current-master kernels. Stray kernels are listed.

### 2.5 B5 — NaN border — holds

HST (`drivers.py:148`) and JWST (`:260`) changed; S4G already used NaN.
Downstream consumers are NaN-safe: the negative-pixel count uses
`nan_to_num`, Check 3 uses `nan_to_num` plus a coverage map, masks use
`isfinite`. The "0.7% of the field" in the code comment comes from a
synthetic 10 px border on a 600 px image; say so in the comment, since the
fraction depends on geometry.

### 2.6 B6 — per-survey `min_blur` — holds

`own_bands` restricts the constraint to each survey's own bands; an empty set
gives `inf`, leaving the master-sampling constraint alone, which is correct.
The skipped IRAC1 pair now constrains only S4G, whose native 0.75" grid gives
bin 1 regardless.

Caveat: the reported master sampling (3.21 px/FWHM) sits just above the
`px_per_fwhm = 3.0` constraint, and that constraint still uses the master's
**Gaussian** FWHM (`fwhm_dict`, Sect. D of CLAUDE.md). Adequate today; fragile
under a change of master.

### 2.7 Not requested, reviewed anyway

`configure_warnings` / `--show_warnings` (CLI): correct. The astropy SIP
message is logged at INFO and is not suppressed, as the docstring says.

---

## 3. B5 — is 0.7% of the field an acceptable price?

Yes, and it is not really a cost. The pixels NaN removes are exactly those
where `reproject_interp` (bilinear) interpolated across the old 0.0 / valid
boundary. Under the 0.0 convention those pixels carried values between 0 and
the true flux — biased low, but finite and indistinguishable from data (the
"small non-zero values that survive" in B5). NaN does not discard good data;
it removes corrupted pixels that previously looked valid. The extra ring is
also small against the ~15.5" per side the kernel-width dilation already
removes.

No reason found to keep 0.0. `units_2_0.py:22` (`data == 0 → NaN`) is now
redundant for the border; leave it until it is confirmed that nothing else
writes exact zeros, and update its comment.

---

## 4. The irac1 0.84% — SIP after all, in the opposite sense

### 4.1 Two premises that do not hold

**irac1 is resampled.** The two S4G mosaics are not on the same pixel grid:

| | NAXIS1×2 | CRPIX1, 2 | CRVAL | CD | A_2_0 | B_0_2 |
|---|---|---|---|---|---|---|
| `NGC1087.phot.1.fits` | 737×1052 | 270, 271 | 41.6049, −0.49868 | ±0.75"/px | −2.39e-5 | 2.31e-5 |
| `NGC1087.phot.2.fits` | 733×1050 | 468, 778 | same | same | +1.91e-5 | 3.44e-5 |

Same CRVAL and CD, integer CRPIX offset, different SIP. Without SIP the
irac2 → irac1 pixel mapping is **exactly (−198.000, −507.000) px over the
whole field**: S4G mosaicked both channels onto one common tangent-plane
grid. With SIP the mapping varies across the field and irac1 is genuinely
bilinearly interpolated — an *unconvolved* band (irac1 is unmatched).

**astropy applies SIP without the `-SIP` suffix.** Its own message reads
"astropy.wcs is using the SIP distortion coefficients". CLAUDE.md §4a stated
the opposite ("astropy ignores them"); the hypothesis built on that premise
was wrong in its premise, not only in its conclusion.

### 4.2 Why the session's test read as a refutation

`test_sip_hypothesis.py` stripped SIP from the **after** file only. That
file's pixels had been placed by `reproject_interp` using SIP, and astropy
still applies SIP when reading the **before** file. The test therefore
measured header/pixel self-consistency — whether the header still describes
how the pixels were placed — not which astrometry is correct. It could only
get worse (11.3%). The 6.6" median / 49.7" max "shift" is the size of the SIP
term across the full mosaic, not evidence that the term is right.

### 4.3 Measurements

**Check 3, same function (`validation.check_flux_conservation`), consistent
WCS on both sides:**

| Configuration | ratio | scatter |
|---|---|---|
| pipeline (SIP applied on both sides) | 0.9916 | 1.43% |
| SIP removed from before and after, reprojected afresh | **1.0000** | **0.00%** |

The pipeline figure reproduces the reported 0.84% exactly.

**irac1 vs irac2 star positions** (peaks > 15σ, centroided, matched in sky):

| | n (< 1.5") | median separation |
|---|---|---|
| with SIP | 131 | 0.321" |
| without SIP | 137 | **0.126"** |

By distance from the galaxy centre (matches < 5"):

| radius | with SIP | without SIP |
|---|---|---|
| 0–1.5′ (n = 98) | 0.193" | 0.090" |
| 1.5–3′ (n ≈ 44) | 0.544" | 0.153" |
| 3–5′ (n ≈ 4) | 1.458" | 0.121" |

With SIP the two channels diverge as the radius grows — the signature of a
spurious distortion. A real distortion solution common to the sky would make
them agree better. Without SIP they agree at ~0.1" everywhere, consistent
with centroid noise (~0.15 px).

Interpretation: the coefficients were most likely inherited from the
single-frame BCD headers, and the missing `-SIP` suffix in the S4G mosaics is
deliberate. This is an inference from the header structure plus the
measurements above, not something found in S4G documentation.

**SIP displacement magnitude** (pixel→sky, SIP vs no SIP, median by radius
from CRPIX):

| | 0–0.5′ | 0.5–1′ | 1–1.5′ | 1.5–3′ | 3–5′ |
|---|---|---|---|---|---|
| irac1 | 0.01" | 0.05" | 0.15" | 0.67" | 3.51" |
| irac2 | 0.01" | 0.07" | 0.17" | 0.79" | 5.45" |

**Absolute test (inconclusive).** S4G stars against
`ngc1087_phangs-jwst_f200w_to_s4g_irac2_convolved.fits` (JWST, Gaia-tied):
only 7–8 matches, all within ~1′ where SIP is ≤ 0.15"; SIP vs no-SIP
indistinguishable (median 0.39–0.45" either way). The same test hints at a
mean S4G − JWST offset of ~−0.36" in RA. With n = 8 and extended sources this
is a hint only, but S4G registration has never been verified (CLAUDE.md
scope caveat), so it is worth following up.

### 4.4 Consequences

- **Check 3:** the 0.84% is an artefact of applying SIP, not a flux loss in a
  band that "is not resampled". With SIP off, irac1 is an exact copy.
- **irac2-master cubes:** `reproject_to_reference` re-attaches `-SIP` to the
  *reference* header, so every HST/JWST band is placed on the irac2 grid
  through irac2's SIP. Within the disc that is 0.07–0.17" at 0.5–1.5′ and
  ~0.8" at 1.5–3′: a radial misregistration growing outwards, i.e. a
  position-dependent colour error. The HST+JWST (master f2100w) cube does not
  involve S4G and is unaffected.
- **The fix is not only dropping the suffix re-attachment.** Because astropy
  applies SIP by default, the coefficients must be stripped at ingestion (S4G
  driver, `_prepare_image`), otherwise Check 3, photutils and any WCS built
  from the headers keep applying them. The note at
  `reprojection_2_0.py:199-214` records the refuted reasoning and should be
  revised.

### 4.5 Outstanding

The decisive test is absolute: Gaia DR3 stars at 3–5′ from the centre, where
SIP moves positions by 3.5–5.5". It should be run before this conclusion goes
into the dissertation.

---

## 5. Reproduction

The measurements were made with three short scripts written during this
verification (not committed): `sipstars.py` (irac1↔irac2 star matching, with
and without SIP, by radius), `sipflux.py` (Check 3 for irac1 with SIP removed
on both sides) and `sipabs.py` (S4G vs JWST f200w). Method: peaks above 15σ
(MAD), `maximum_filter` window 7 px, 5×5 centre-of-mass centroid,
`match_to_catalog_sky`; "no SIP" means `WCS(header).sip = None`.

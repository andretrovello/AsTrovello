export const meta = {
  name: 'verify-astrovello-audit-fixes',
  description: 'Verify the AsTrovello audit fixes (A1,A2,A3,B1), sweep remaining defects, review new validation code, cross-check the session report',
  phases: [
    { title: 'Investigate', detail: 'one agent per defect group / new module' },
    { title: 'Verify', detail: 'adversarial refutation of each finding' },
    { title: 'Critic', detail: 'what was missed' },
    { title: 'Synthesize', detail: 'final verdict table' },
  ],
}

const ROOT = '/mnt/c/Users/dedet/Desktop/AsTrovello'
const SRC = ROOT + '/src/astrovello'

const COMMON = `
CONTEXT
Repository: ${ROOT}  (source in ${SRC}).
The project CLAUDE.md has already been injected into your context: Section 4 of it is the
full read-only audit with defect IDs (A1, A2, A3, B1..B7, C1..C3, D, E1..E8). Use those IDs.
The file ${ROOT}/session_report_audit_and_validation.md is the AUTHOR'S REPORT claiming which
defects were fixed and how they were verified. Your job is to check the CODE, not to trust the report.

ENVIRONMENT
- python3 with astropy 7.2.0, numpy, scipy is available. Run it directly ("python3 -c ...").
  Do NOT try to "conda activate capivara" - just use python3.
- The pipeline modules use flat imports and require CWD=${SRC} to import.
  Some modules import 'pypher' or other packages that may be missing; if an import fails,
  test the logic by copying the relevant function body into a standalone script instead.
- Write any scratch scripts under /tmp/claude-1000/-mnt-c-Users-dedet-Desktop-AsTrovello/91a568b7-8dec-4773-8d78-fe8db735d151/scratchpad
- You are READ-ONLY on the repository: never edit, never write inside ${ROOT}.
- Real data lives in ${ROOT}/Input and ${ROOT}/Output. Note Output/ currently contains only
  OLD v1.x artifacts (kernels named *_to_irac1, cubes for ngc1433/ngc2903 from July);
  the ngc1087 products described in the session report are NOT on disk here. Do not assume
  they exist; if you need them, say so rather than inventing numbers.

STANDARD OF EVIDENCE
- Prefer running code over reasoning about it. Quote exact file:line and exact measured output.
- A "fix" counts as CORRECT only if (a) the logic is right, (b) it is applied at EVERY call site,
  and (c) nothing downstream undoes it. Check all three.
- Distinguish: fix correct and complete / fix correct but incomplete / fix wrong / not attempted.
- Do not report style nits. Report defects that change numbers, crash, or silently disable a guard.
- If you find NOTHING wrong in your area, say so explicitly and clearly - a clean verdict is a
  valuable result. Do not manufacture findings.
`

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    area: { type: 'string', description: 'the area you investigated' },
    verdicts: {
      type: 'array',
      description: 'per-defect-ID verdict for the defects in your scope',
      items: {
        type: 'object',
        properties: {
          defect_id: { type: 'string' },
          status: { type: 'string', enum: ['fixed_correct', 'fixed_incomplete', 'fix_wrong', 'not_fixed', 'not_applicable', 'unverifiable'] },
          evidence: { type: 'string', description: 'file:line plus exact measured output supporting the status' },
        },
        required: ['defect_id', 'status', 'evidence'],
      },
    },
    findings: {
      type: 'array',
      description: 'NEW problems found (bugs, regressions, incomplete fixes, false claims). Empty if none.',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          file: { type: 'string' },
          line: { type: 'number' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          description: { type: 'string' },
          failure_scenario: { type: 'string', description: 'concrete inputs/state -> wrong output' },
          evidence: { type: 'string', description: 'exact commands run and exact output' },
          suggested_fix: { type: 'string' },
        },
        required: ['title', 'file', 'severity', 'description', 'failure_scenario', 'evidence'],
      },
    },
    notes: { type: 'string', description: 'anything important that is not a finding' },
  },
  required: ['area', 'verdicts', 'findings'],
}

const INVESTIGATORS = [
  {
    key: 'A1-cube-wcs',
    prompt: `${COMMON}

YOUR AREA: defect A1 - the datacube 3D WCS.

The report claims cube_2_0.py now builds the 3D WCS with w_2d.to_header() instead of copying
wcs.cdelt field by field, and that this fixes the CDELT=1.0 (factor 4800) bug.

Do all of this:
1. Read ${SRC}/cube_2_0.py in full. Find create_data_cube and the header construction (~line 195-260).
2. Empirically test the NEW idiom on synthetic headers covering: (a) CD matrix, no CDELT;
   (b) CDELT+PC; (c) CD matrix with rotation; (d) SIP present. For each, build a 2D WCS, run the
   pipeline's exact new code path, then read the result back with astropy and compare
   proj_plane_pixel_area and a few sky coordinates against the source. Report exact numbers.
3. Check the THIRD axis handling specifically: the code sets WCSAXES=3, CTYPE3, CRPIX3, CRVAL3, CDELT3
   AFTER to_header(). Verify astropy can actually read that header back as a 3-axis WCS without the
   spatial axes being corrupted. Watch for: missing PC3_3 / CD3_3, a CD matrix (CD1_1 etc.) coexisting
   with a scalar CDELT3 (astropy/FITS forbids mixing CD and CDELT - does to_header() emit CD or PC+CDELT
   for a CD-matrix source?), missing CUNIT3, and whether NAXIS/NAXIS3 are consistent with the data.
   If a CD matrix is emitted and CDELT3 is added, determine empirically whether the spatial scale is
   then misread. This is the single most important thing to test.
4. Verify the header actually fed in (final_header from crop_to_mask_bbox) has correct CRPIX after the
   crop, and that NAXIS1/NAXIS2 written by the crop do not conflict with the cube's real shape.
5. Check the bare "except Exception: continue" the audit flagged - is it gone from the WCS block?
   Look at what the remaining bare except at cube_2_0.py:46 does and whether it can hide a real failure.
6. Check the ERROR cube path (if create_data_cube is called twice, sci and err) gets the same header.
7. Inspect an actual old cube on disk (${ROOT}/Output/datacubes/ngc2903/*sci*.fits) with astropy to
   confirm the OLD bug is real (CDELT1=1.0), using only the header (do not load the 1.4GB data array -
   use fits.getheader).

Report a verdict for A1 and any new findings.`,
  },
  {
    key: 'A2-natpxar',
    prompt: `${COMMON}

YOUR AREA: defect A2 - NATPXAR (native pixel area) in the unit conversion.

The report claims drivers.py now reads NATPXAR from the header with the config constant as fallback,
removing a 0.61% colour error on the five HST bands.

This is a DATA-FLOW question, not a one-line question. The fix only works if NATPXAR is actually
present in the header that reaches convert2Jansky. Trace it end to end:
1. Where is NATPXAR written? (convolution_2_0.py ~line 661, inside bin_for_convolution). Confirm the
   VALUE written is the true NATIVE (pre-binning) pixel area measured from the WCS, not the binned one,
   and not the config constant. Read the surrounding function carefully.
2. bin_for_convolution returns early when factor <= 1 - confirm, and note which surveys/configurations
   get factor 1 (so no NATPXAR is written and the fallback constant is used - meaning the 0.61% bug
   SURVIVES for those). For the configuration in the report (PHANGS-HST + PHANGS-JWST, master f2100w),
   which surveys get bin > 1?
3. Follow the header through: convolution output -> reprojection (${SRC}/reprojection_2_0.py,
   reproject_to_reference) -> the *_Jy_per_pixel writing (units_2_0.py + drivers). Does NATPXAR SURVIVE
   the reprojection header rebuild? reproject_to_reference pops WCS keys and merges w_ref.to_header();
   determine exactly whether NATPXAR is preserved, dropped, or overwritten. Read the code and, if you
   can, demonstrate it with a synthetic header run through the same code path.
   If NATPXAR is dropped before convert2Jansky, the fix is INEFFECTIVE and the report's claim is wrong.
4. Check the MASTER band path: a master band is copied, not convolved/binned - does it get NATPXAR?
   Does its conversion use the right area?
5. Check the other two drivers (PHANGS_JWST_Driver, S4G_Driver) convert2Jansky: they use MJy/sr
   (surface brightness) so the native area should be irrelevant - confirm they are genuinely unaffected,
   and check they do not have an analogous stale-constant problem.
6. Check whether the config constant fallback value and the true WCS value are what the audit says
   (0.0395 vs 0.039620) by reading config.py and the actual HST mosaic header in ${ROOT}/Input.
   Use fits.getheader, do not load data.

Report a verdict for A2 and any new findings. The key question to answer unambiguously:
does convert2Jansky actually SEE NATPXAR at runtime, for which bands?`,
  },
  {
    key: 'A3-B1-crop-dilation',
    prompt: `${COMMON}

YOUR AREA: defects A3 (crop off-by-one) and B1 (binary_dilation cost).

A3 - ${SRC}/mask_2_0.py crop_to_mask_bbox:
1. Verify the +1 is correct for both axes and interacts correctly with padding. Note the asymmetry in
   the current code: y_min uses max(0, y_min - padding) but y_max uses min(ny, y_max + 1 + padding).
   Test empirically with a synthetic mask (including padding>0, mask touching the array edge, mask of
   a single pixel) that the returned crop contains exactly the valid block plus the padding.
2. Check the header update: CRPIX1/CRPIX2 shifted by x_min/y_min, NAXIS1/NAXIS2 set. Are NAXIS values
   correct given the +1? Is CRPIX shifted by the right sign/amount (FITS is 1-indexed, numpy 0-indexed -
   verify the convention is right by round-tripping a sky coordinate through the cropped WCS; an
   off-by-one here would be a half-arcsec astrometric error and would NOT have been caught by the
   author's check1 astrometry test, which compared input images, not the cube).
3. Both call sites in cube_2_0.py - confirm both benefit. Also evaluate the audit's D-section claim that
   padding=50 on the SECOND crop is defeated because the mask is applied before the crop
   (cube_2_0.py ~line 205-212) and crop_to_mask_bbox re-applies it at the end, so the padded ring is
   guaranteed all-NaN. Is that still true in the current code? Demonstrate.

B1 - ${SRC}/drivers.py, three places (~line 134, 235, 311):
4. Verify the mathematical identity empirically: is binary_dilation(mask, structure=ones((3,3)),
   iterations=k//2) EXACTLY equal to binary_dilation(mask, structure=ones((k,k)))? Test for k odd AND
   k even, several mask shapes, and masks touching the array border (scipy's border_value default
   matters). np.array_equal, report exact results.
5. Examine the max(kernel_size // 2, 1) guard: for kernel_size = 1 or 2 the original code dilates by
   a 1x1 or 2x2 footprint, while the new code does at least 1 iteration of 3x3. Is that a behaviour
   change? Does kernel_size ever reach such small values in practice? Check how kernel_size is derived
   at each of the three sites.
6. Confirm all three drivers were changed and that none still builds a k x k structure anywhere in the
   repository (grep the whole src tree).
7. Time it on a realistic array (e.g. 1600x1600 mask, k=157) to sanity-check the claimed speedup
   magnitude - but keep the run short, do not spend more than ~60s of CPU.

Report verdicts for A3 and B1 and any new findings.`,
  },
  {
    key: 'B2-B3-B4-kernels',
    prompt: `${COMMON}

YOUR AREA: defects B2, B3, B4, B7 - the kernel-generation guards in ${SRC}/convolution_2_0.py.
These were NOT in the report's list of fixed items, so the expected answer is "still open" - but verify,
because the author may have fixed them silently, or the new validation.py may now cover them.

1. B2 - inspect_kernel is called AFTER normalisation (the audit says line ~934-938), so its "sum" and
   "negative power" metrics are vacuous. Read the current code. Is inspect_kernel now called on the RAW
   kernel? If not, confirm the blind spot still exists and demonstrate it: take a real kernel from
   ${ROOT}/Output/PSF_Kernels/, compute sum and negative-power fraction before and after dividing by
   the sum, and show the numbers. Also check whether validation.py's check_psf_matching now measures
   negative power on the raw or normalised kernel - if it does it on the raw kernel, B2's guard may be
   restored by a different route; say so explicitly.
2. B3 - the PIXSCALE grid guard is skipped when PIXSCALE is missing and the WCS fallback also fails
   (k_px = None -> check silently skipped). Read the current code (~line 941-957). Is a missing PIXSCALE
   now a hard error? Check the six v1.x kernels actually in ${ROOT}/Output/PSF_Kernels/ - do they have
   PIXSCALE? What happens if the pipeline is run without --create_kernel and picks them up?
3. B4 - the kernel glob in astrovello_cli_2.0.py (audit said line ~469) is not master-scoped
   (kernel_*_to_*.fits) while rediscover_unmatched globs kernel_*_to_MASTER.fits. Read the current
   CLI. Is the glob now master-scoped? If not, construct the concrete failure: leftover kernels for a
   previous master + a new master -> which path misclassifies. Note the real Output/PSF_Kernels holds
   *_to_irac1 kernels, so this is live, not hypothetical.
4. B7 - the smaller items: (a) np.interp on a non-monotonic cumulative curve in the enclosed-energy
   radius (~line 122) - is np.maximum.accumulate now applied? (b) get_half_light_radius normalising by
   flux within r_max = min(shape)/2; (c) clean_psf block_reduce alignment; (d) the "sign inversion is
   cosmetic" comment. Report which are still as the audit described.
5. While you are in this file: review the kernel-generation path (pypher_kernel_creation, clean_psf,
   the convolution_grid_arcsec rename mentioned in the report) for any NEW bug introduced by the rename
   - e.g. a call site still passing target_pixel_scale_arcsec, or a renamed parameter with a changed
   meaning. Grep for both names across the whole src tree and tests/.

Report verdicts and findings.`,
  },
  {
    key: 'B5-B6-preflight',
    prompt: `${COMMON}

YOUR AREA: defects B5 (invalid-pixel convention) and B6 (global min_blur), plus preflight.py.

1. B5 - HST/JWST drivers set the dilated border to 0.0 while S4G sets np.nan; intersection_footprint_mask
   defines valid as ~isnan, so the HST/JWST borders count as valid, rescued only by units_2_0.py:22
   blanket-converting data==0 to NaN after reprojection. Read ${SRC}/drivers.py (all three convolve
   methods) and ${SRC}/units_2_0.py. Is the convention now uniform? If not:
   - demonstrate the concrete failure: reproject_interp interpolating across a 0/valid boundary produces
     small non-zero values that survive the ==0 test (show this with a small synthetic reprojection), and
     genuinely-zero science pixels get killed.
   - quantify how many pixels in a real convolved file are exactly 0.0 vs near-0, if a real convolved
     file is available in ${ROOT}/Output/convolved_fits/ (use memmap / read one slice, these are large).
2. B6 - per-survey min_blur. Read astrovello_cli_2.0.py derive_bin_factors / choose_bin_factor
   (audit said ~line 115-128). Confirm min_blur is still a global minimum over ALL bands. Also verify
   the audit's separate observation that derive_bin_factors passes the GAUSSIAN FWHM as the master width
   for the grid decision even though get_fwhm carries a warning not to use it for ranking (doc section D).
   Do not propose applying B6 - just state precisely whether it is still open and whether the code
   contains anything new about it.
3. preflight.py - read it. Check the audit's claim that _check_master_grid takes an "errors" list and
   never appends to it (so "master-grid consistency" is documented as a check but only prints). Is it
   still so? Also check whether preflight was updated for the new validation gates / renames, and
   whether --preflight_only still works given the renamed --mode values.
4. Check the doc-vs-code claims in CLAUDE.md section D that concern the CLI/preflight:
   - "a band cannot be both matched and unmatched (raises)" - does any check raise?
   - "missing kernels for the current master raise with a clear message" - does rediscover_unmatched
     raise or silently reclassify?
   - "--apply_mask is store_true (default False) while create_data_cube defaults apply_mask=True"
   State for each whether it is still as the audit described in the CURRENT code.

Report verdicts and findings.`,
  },
  {
    key: 'C1-C2-reprojection',
    prompt: `${COMMON}

YOUR AREA: defects C1, C2 - SIP handling in the reprojection stage - plus the unexplained irac1 anomaly.

The module was RENAMED from alignment_2_0.py to reprojection_2_0.py. Read ${SRC}/reprojection_2_0.py
in full.

1. C1 - reproject_to_reference pops the linear WCS keys but not A_ORDER/A_i_j/B_*/AP_*/BP_*/CROTA2/
   LONPOLE/LATPOLE, so the source image's SIP polynomial can survive onto the reference grid. Is it
   fixed? If not, demonstrate concretely: build a synthetic source header WITH SIP and a reference
   header WITHOUT SIP, run the header-building code path, and show that the output header carries the
   source's SIP coefficients over the reference's linear WCS, and compute how large an astrometric
   error that implies (evaluate the SIP polynomial at the image corner using real coefficients from a
   PHANGS-HST file in ${ROOT}/Input - use fits.getheader).
2. C2 - apply_sip_correction is a static config flag, so the -SIP CTYPE suffix is re-attached even
   when bin_for_convolution deleted the coefficients, giving a WCS whose .sip is None with no error.
   Is it fixed? Verify the "silent no-op" empirically with astropy (build a header with CTYPE
   RA---TAN-SIP and no A_/B_ coefficients, confirm w.sip is None and no exception).
   Determine for which surveys/bin factors this activates in the CURRENT configuration.
3. THE OPEN ITEM WORTH REAL EFFORT: the session report section 5 records that irac1 - the ONLY band not
   reprojected (0.75" -> 0.75") - had the WORST flux deviation (0.84%), and guesses "likely the SIP
   correction in reproject_to_reference". Investigate this properly. Read the code path a
   same-grid band takes. Candidate mechanisms to test: (a) the -SIP suffix being attached to a
   reference WCS whose SIP coefficients came from somewhere else; (b) reproject_interp being run at all
   for an identity transform and bilinear interpolation still smoothing if the grids are offset by a
   sub-pixel amount; (c) a half-pixel CRPIX convention error. If you can, demonstrate with a synthetic
   same-grid reprojection whether reproject_interp alone loses ~0.8% of flux in apertures, or whether
   the SIP is needed to produce it. State which mechanism the evidence supports.
4. Check the module header / docstring the report says was added, and whether any caller still imports
   alignment_2_0 or calls reproject functions by their old names (grep the whole repo including tests/
   and notebooks).

Report verdicts and findings.`,
  },
  {
    key: 'validation-gates',
    prompt: `${COMMON}

YOUR AREA: the NEW module ${SRC}/validation.py and how it is wired into astrovello_cli_2.0.py.

This is new code written to be an ABORT GATE. A gate that passes vacuously is worse than no gate,
so review it adversarially.

1. Read validation.py in full. For check_psf_matching and check_flux_conservation determine:
   - Can the gate pass VACUOUSLY? i.e. zero pairs discovered, all pairs skipped, an exception swallowed,
     an empty list -> "PASS". Trace every early return and every continue. The report itself mentions
     pairs being "skipped rather than measured" when grids differ by >2% - what happens if ALL pairs are
     skipped? What if a PSF file is missing? What if a glob matches nothing?
   - Are the thresholds actually enforced (report claims 0.95-1.05 for r50, <1% for flux)? Read the exact
     comparison operators. Is the median or the worst case compared?
   - Does it raise ValidationError on failure, and does the CLI let that propagate (not caught by a
     try/except somewhere up the stack)?
2. Check the gate WIRING in astrovello_cli_2.0.py (around lines 500-530 and 650-660):
   - check_psf_matching runs "after kernels are built, before any image is convolved" - verify that is
     true in the control flow, including the branch where kernels are NOT regenerated
     (no --create_kernel): does the gate still run on the reused kernels? The condition is
     "if kernel_files and not args.skip_checks" - determine what kernel_files contains in each branch.
   - check_flux_conservation runs "after reprojection, before unit conversion" - verify, given that the
     CLI does unit conversion INSIDE the reprojection block. Which happens first in the actual code?
   - the report mentions a bug found during integration: kernels attributed to surveys by substring
     match, fixed by parsing the source filter out of the filename. Find that code
     (kernel_source_filter) and check the parser is robust: filter names containing "_to_", masters whose
     name is a substring of another filter, filenames not matching the pattern.
3. Check the survey-to-filters mapping built around cli line 451 that the gate uses for attribution - can
   a band be attributed to the wrong survey, or dropped from the check entirely (silently reducing
   coverage)?
4. Cross-check the report's claim "correct kernel -> PASS, kernel 30% wide (ratio 1.5811) -> abort".
   A 30% wider Gaussian giving r50 ratio 1.5811 rather than ~1.3 is suspicious - work out what the test
   actually did if you can find it, and whether the discrepancy indicates the ratio metric is not what
   the report says it is. Note 1.5811 = sqrt(2.5) exactly.
   Decide whether this is a real inconsistency or an innocent one, and say which.
5. Anything in validation.py that would abort a CORRECT pipeline (false positive) is as serious as a
   missed failure - look for those too.

Report verdicts (use defect_id "validation.py" etc.) and findings.`,
  },
  {
    key: 'check-scripts',
    prompt: `${COMMON}

YOUR AREA: the standalone check scripts - ${SRC}/check2_psf_matching.py,
${SRC}/check3_flux_conservation.py, ${SRC}/pypher_regularisation_test.py, and
${SRC}/diagnostico_astrometria.py.

The session report presents numbers from these scripts as EVIDENCE that the pipeline is correct
(check 2: all 12 pairs at r50 ratio 1.0000-1.0001; check 3: 0.15% median flux deviation;
check 1: 0.079 px astrometric agreement). Your job is to decide whether those numbers mean what the
report says they mean.

1. check2_psf_matching.py: read it fully. The reported r50 ratios are 1.0000 or 1.0001 for ALL TWELVE
   pairs - suspiciously perfect. Determine whether the test is capable of failing:
   - Is the comparison r50(source PSF convolved with kernel) vs r50(target PSF), both measured ON THE
     SAME GRID with the SAME estimator? If the kernel was BUILT by pypher from exactly these two PSFs,
     is the test partly circular - i.e. does it verify the kernel reproduces the target by construction,
     while the thing that actually broke in v1.x (kernel built on one grid, APPLIED on another) is not
     exercised because the script uses the PSFs on the kernel's own grid?
   - Concretely: does this check, as written, detect the v1.x grid bug? Try to construct the v1.x
     failure mode and see whether the script's logic flags it. This is the decisive question -
     the report's headline claim is that these gates would catch a regression.
   - Check the enclosed-energy radius estimator for the same non-monotonicity issue as B7, and the
     2%-grid-difference skip rule (does skipping quietly reduce the number of pairs tested?).
   - Check the negative-power formula and the claimed 35.3% limit against Aniano et al. 2011
     as quoted in the project documentation ${ROOT}/astrovello_pipeline_documentation.md.
     Is the normalisation right, and is it computed on the raw or normalised kernel?
2. check3_flux_conservation.py: read it fully. The key correction described in the report is weighting
   each aperture sum by the pixel area of its grid. Verify the weighting is implemented correctly
   (and that it is not simply forcing the ratio to 1 by construction). Check the 5%-brightness
   criterion for selecting apertures - does that bias toward apertures where flux is conserved?
   Would a genuine 1% loss in the faint outskirts be detected?
3. pypher_regularisation_test.py: the report's section 6 concludes the 1.06 could not be reproduced.
   Check the synthetic Gaussian control is correct (sigma_kernel = sqrt(sigma_B^2 - sigma_A^2)) and that
   the sweep really varies what it claims. Note the report calls this file test_pypher_regularisation.py
   while on disk it is pypher_regularisation_test.py - note any such report/disk mismatch.
4. diagnostico_astrometria.py: the report describes check1_astrometry.py with detailed results
   (0.079 px median over five sources). On disk, check1_astrometry.py does NOT exist and
   diagnostico_astrometria.py is a 46-byte stub containing only two import lines. There is also
   diagnostico_astrometria_nb.ipynb (~500KB). Determine where the check-1 code actually lives (look in
   the notebook - extract its code cells and read them), whether it implements what the report describes
   (5 sources, photutils centroid_2dg, HST-against-itself sanity test), and whether the method is sound.
   Be precise and fair: the work may be entirely real and just living in a notebook.

Report verdicts and findings. Be especially careful to distinguish "the check is weak" from
"the check is wrong".`,
  },
  {
    key: 'cli-repo-state',
    prompt: `${COMMON}

YOUR AREA: ${SRC}/astrovello_cli_2.0.py as a whole, plus repository state (defects E1-E8) and the
consistency of the rename work.

1. Read astrovello_cli_2.0.py in full. Look for bugs introduced by this session's changes:
   - the --mode rename (alignment_only -> reprojection_only): any leftover string comparison against
     "alignment_only" anywhere in the repo, in docs, in notebooks, in preflight?
   - the target_pixel_scale_arcsec -> convolution_grid_arcsec rename (report says 10 occurrences):
     grep the ENTIRE repository (src, tests, notebooks, docs) for both names and report any call site
     left on the old name or any place where the two names now coexist with different meanings.
   - the new --skip_checks and the validation imports at the top (line ~60): does the module import
     cleanly? Try running, from ${SRC}, a python3 snippet that loads the file with
     importlib.util.spec_from_file_location and exec_module, and report exactly what happens
     (an ImportError for a missing third-party package is fine and expected; a NameError / SyntaxError /
     ImportError for a LOCAL module is a real finding).
     Do the same for each local module: config, drivers, convolution_2_0, reprojection_2_0, units_2_0,
     cube_2_0, mask_2_0, utils_2_0, preflight, validation.
   - master selection, bin-factor derivation, stage dispatch: any place where a variable is used before
     assignment on some branch, or a stage silently no-ops.
2. E1 - src/astrovello/__init__.py: is it still importing v1.x module names? Test it.
3. E2 - pyproject.toml entry points: still broken? Note there are TWO files, pyproject.toml and
   .pyproject.toml - check both.
4. E4/E5 - the Input/ directory names vs SURVEY_CONFIG keys, and get_survey's substring matching.
   List ${ROOT}/Input/ and compare against the keys in config.py. State whether the pipeline as
   committed can run against the data layout as committed. Check whether get_survey was changed to read
   TELESCOP/INSTRUME from the header as the audit suggested.
5. E6 - tests/: still no real tests? The audit suggested a handful of pytest cases would be worth it.
   Check whether any were added (tests/ and anywhere else). If tests exist, run them.
6. E8 - --error and --valid_pixels_cut declared but unimplemented; config.py "Notesf" typo.
   Still present?
7. Report on git state: run "git log --stat -5" and confirm which fixes are COMMITTED versus only
   present in the working tree (note: the working-tree diff appears to be line-endings only - verify
   that claim with git diff --stat and git diff --ignore-all-space).

Report verdicts and findings.`,
  },
  {
    key: 'report-vs-reality',
    prompt: `${COMMON}

YOUR AREA: the session report itself - ${ROOT}/session_report_audit_and_validation.md.
You are checking the REPORT for accuracy against the repository, because it is dissertation-adjacent
material and the project's own stated lesson is "treat a claim in the documentation as a specification
to be checked against the code, not a description of it".

Be scrupulously fair. The author did real work; your job is to find claims that a reader could not
reproduce, not to be contrarian.

1. Section 10 "Files created or modified" lists five NEW files: validation.py, check1_astrometry.py,
   check2_psf_matching.py, check3_flux_conservation.py, test_pypher_regularisation.py. Check each
   against disk. Report exactly which exist, which have different names, and which are missing or
   stubs. (Known: check1_astrometry.py and test_pypher_regularisation.py do not exist under those
   names.) Also check the claim "All check scripts write to Output/checks/<check name>/" - does that
   directory exist, and does the code actually write there?
2. Section 9 "Regenerated cubes" quotes a header from an ngc1087 cube (13 x 1277 x 1135, CDELT1
   3.0807051217028e-05, PIXAREA 0.0123). Check ${ROOT}/Output/datacubes/ for ngc1087. If absent, say so
   plainly and note that the quoted header cannot be verified from this checkout - do NOT claim the
   author fabricated it; state only what is and is not verifiable here.
   Independently CHECK THE NUMBERS FOR SELF-CONSISTENCY: CDELT1 = 3.0807051217028e-05 deg - convert to
   arcsec and compare with the quoted "scale from WCS 0.11090538 arcsec/px" and with
   PIXAREA 0.0123 arcsec^2. Do they agree? Report the arithmetic.
   Also check the quoted PC1_1 = -0.99999999991961 - is a PC matrix with CDELT consistent with the A1
   fix as implemented (to_header on a CD-matrix source)? Does it imply a rotation, and is that expected?
3. Section 1 summary table, section 2 verification numbers: check each internally-consistent claim by
   arithmetic where possible. E.g. A2 "HST f814w 0.993960 (predicted 0.99394)" - is 0.99394 really
   (0.0395/0.039620)^2? Compute it. A3 "1276x1134 -> 1277x1135, +34 valid pixels" - is +34 consistent
   with adding one row and one column to a 1276x1134 array? Is that plausible or does it indicate
   something else?
   B1 table: k=157 MemoryError at 600x600 - is a 157x157 structure really enough to MemoryError?
   Check what scipy actually does and whether the claim holds (test it if cheap).
4. Section 4: "The HST kernels at 0.00% negative power" - the audit (CLAUDE.md B2) says 0.00% is exactly
   the artefact produced by measuring after normalisation. Check whether check2/validation measure it
   before or after, and therefore whether this table reproduces the B2 blind spot in new clothes. This
   is the most important item in your scope.
5. Section 7 claims the gates "abort on failure" with only --skip_checks as an escape. Cross-check
   against the CLI control flow (another agent is reviewing validation.py in depth; you only need to
   confirm or contradict the report's high-level claim).
6. Section 11 "Open items" vs CLAUDE.md section 4: which audit defects does the report NOT mention at
   all? Produce the complete list of audit defect IDs (A1,A2,A3,B1..B7,C1,C2,C3,D-items,E1..E8) with
   "claimed fixed" / "listed as open" / "not mentioned". This list is a deliverable.
7. Does the report's own stated organising principle ("every correction verified against a prediction
   made before the measurement") hold up - are there claims stated as verified where no verification
   artefact exists in the repo?

Report verdicts and findings.`,
  },
]

phase('Investigate')
log(`Investigating ${INVESTIGATORS.length} areas in parallel`)

const results = await pipeline(
  INVESTIGATORS,
  inv => agent(inv.prompt, { label: `investigate:${inv.key}`, phase: 'Investigate', schema: FINDINGS_SCHEMA, effort: 'high' }),
  (res, inv) => {
    if (!res || !res.findings || res.findings.length === 0) return { inv, res, verified: [] }
    const LENSES = [
      'CORRECTNESS: is the described mechanism real? Re-derive it from the code yourself. Run the code path. If the claim rests on a measurement, repeat the measurement independently.',
      'SCOPE: even if the mechanism is real, does it actually fire in this pipeline as configured? Is it dead code, an unreachable branch, already guarded upstream, or does something downstream neutralise it? Also: is it a duplicate of a defect already recorded in CLAUDE.md section 4 as known and unfixed, in which case it is not a NEW finding?',
    ]
    return parallel(res.findings.map(f => () =>
      parallel(LENSES.map(lens => () =>
        agent(`${COMMON}

You are an adversarial verifier. Your default is REFUTED. Only mark refuted=false if you
independently reproduced the problem.

CLAIMED FINDING (from area "${inv.key}"):
  title: ${f.title}
  file: ${f.file}${f.line ? ':' + f.line : ''}
  severity: ${f.severity}
  description: ${f.description}
  failure scenario: ${f.failure_scenario}
  evidence offered: ${f.evidence}

VERIFY THROUGH THIS LENS -> ${lens}

Go to the file, read it, and run whatever code settles it. Report what you actually observed,
and your corrected severity if the finding survives but is over- or under-stated.`,
          { label: `verify:${inv.key}:${f.title.slice(0, 28)}`, phase: 'Verify', effort: 'high', schema: {
            type: 'object',
            properties: {
              refuted: { type: 'boolean' },
              reasoning: { type: 'string' },
              observed: { type: 'string', description: 'what you actually ran and saw' },
              corrected_severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'none'] },
              corrected_claim: { type: 'string', description: 'if the finding is real but mis-stated, the accurate version' },
            },
            required: ['refuted', 'reasoning', 'observed', 'corrected_severity'],
          } })
      )).then(votes => {
        const v = votes.filter(Boolean)
        const upheld = v.filter(x => !x.refuted).length
        return { finding: f, area: inv.key, votes: v, survives: upheld >= 1 && v.length > 0, unanimous: upheld === v.length }
      })
    )).then(verified => ({ inv, res, verified: verified.filter(Boolean) }))
  }
)

const ok = results.filter(Boolean)
const allVerdicts = ok.flatMap(r => (r.res && r.res.verdicts ? r.res.verdicts : []).map(v => Object.assign({}, v, { area: r.inv.key })))
const survivors = ok.flatMap(r => r.verified.filter(v => v.survives))
const killed = ok.flatMap(r => r.verified.filter(v => !v.survives))

log(`${allVerdicts.length} defect verdicts; ${survivors.length} findings survived verification, ${killed.length} refuted`)

phase('Critic')
const critic = await agent(`${COMMON}

A multi-agent review just finished. Here is everything it concluded.

DEFECT VERDICTS:
${JSON.stringify(allVerdicts, null, 1)}

SURVIVING FINDINGS:
${JSON.stringify(survivors.map(s => ({ area: s.area, title: s.finding.title, file: s.finding.file, severity: s.finding.severity, desc: s.finding.description, unanimous: s.unanimous, votes: s.votes.map(v => ({ refuted: v.refuted, sev: v.corrected_severity, corrected: v.corrected_claim })) })), null, 1)}

REFUTED FINDINGS (for your awareness - do not resurrect without new evidence):
${JSON.stringify(killed.map(s => ({ title: s.finding.title, why: s.votes.map(v => v.reasoning.slice(0, 300)) })), null, 1)}

Your job: find what was MISSED. Specifically:
1. Is any audit defect ID from CLAUDE.md section 4 (A1,A2,A3,B1,B2,B3,B4,B5,B6,B7,C1,C2,C3,
   the D documentation items, E1-E8) absent from the verdict list? If so, check it yourself now
   and report its status.
2. Is there any INTERACTION between fixes that no single agent would have seen? Specifically consider:
   the A1 cube-WCS fix interacting with the A3 crop header (CRPIX/NAXIS), the A2 NATPXAR fix
   interacting with the B6 bin factors (bin=1 -> no NATPXAR -> fallback constant -> bug survives),
   and the new gates interacting with --create_kernel / stale kernels (B3/B4).
3. Did anything the report claims as VERIFIED rest on an artefact that does not exist in the repo?
4. Is there a defect in the NEW code (validation.py, check scripts, the renames) that the
   area-by-area split would have let fall between the cracks - e.g. a module that no agent read?
   List every .py file under ${SRC} and confirm each was covered by some area; read any that were not.

Run code to settle anything you assert. Return only things you verified yourself.`,
  { label: 'completeness-critic', phase: 'Critic', effort: 'high', schema: FINDINGS_SCHEMA })

phase('Synthesize')
const synthesis = await agent(`${COMMON}

You are writing the final verdict for the user (Andre, the pipeline's author, an MSc student at
IAG-USP). He asked: "check the session report and the new version of my code to see if it corrected
the bugs."

Here is all verified evidence from the review.

DEFECT VERDICTS BY AREA:
${JSON.stringify(allVerdicts, null, 1)}

CONFIRMED FINDINGS (survived adversarial verification):
${JSON.stringify(survivors.map(s => ({ area: s.area, finding: s.finding, unanimous: s.unanimous, verifier_notes: s.votes.map(v => ({ refuted: v.refuted, severity: v.corrected_severity, corrected_claim: v.corrected_claim, observed: v.observed.slice(0, 600) })) })), null, 1)}

COMPLETENESS CRITIC:
${JSON.stringify(critic, null, 1)}

Produce a synthesis with these parts, in this order:

PART 1 - The four claimed fixes (A1, A2, A3, B1): one verdict each, with the single most decisive
piece of evidence. Say plainly whether each is correct AND complete, or where it falls short.

PART 2 - Report accuracy: which claims in session_report_audit_and_validation.md are supported by
the repository, which cannot be verified from this checkout, and which are wrong. Be precise and
fair; distinguish "not verifiable here" from "incorrect".

PART 3 - New problems introduced or newly discovered, ranked by severity, each with file:line, the
concrete failure, and a one-or-two-line fix. Only confirmed ones.

PART 4 - Audit defects still open: a compact table of every ID from CLAUDE.md section 4 with status.

PART 5 - Recommended next actions in priority order, with a one-line justification each. Apply the
author's own severity rule: "an error that affects some bands and not others is a colour error, and
a colour error becomes an age error."

Be concise and concrete - file:line and numbers, not prose. Do not repeat evidence already obvious.
Do not soften real problems, and do not inflate minor ones. If the fixes are simply correct,
say so without hedging.`,
  { label: 'synthesis', phase: 'Synthesize', effort: 'high' })

return {
  synthesis,
  verdicts: allVerdicts,
  confirmed_findings: survivors.map(s => Object.assign({ area: s.area }, s.finding, { unanimous: s.unanimous, corrections: s.votes.map(v => v.corrected_claim).filter(Boolean) })),
  refuted_count: killed.length,
  critic_findings: critic && critic.findings ? critic.findings : [],
}

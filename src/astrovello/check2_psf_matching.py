"""
Check 2 - PSF matching validation, all pairs.

For every source PSF, convolve it with the kernel that maps it onto the master
and compare the result with the master itself. A correct kernel reproduces the
target at all radii (Aniano et al. 2011, Sect. 5); the enclosed-energy radius
ratio reduces that to one number, which must be 1.0.

    ratio = r50(source * kernel) / r50(target)

Criterion, agreed in advance: 0.95 <= ratio <= 1.05, for every band.

The script discovers the pairs on its own: inside each PSF_CLEAN directory the
file whose name starts with "master_" is the target and everything else is a
source. Both must already be on the same grid - that is what build_kernels
produces, and a pair on mismatched grids is skipped rather than measured,
because a kernel between different grids is meaningless.

Three secondary metrics accompany the ratio, each catching a failure mode the
ratio alone would miss:

    r80 ratio    - the same test weighted towards the wings
    neg power    - Aniano's W-, normalised: a smoothing kernel is ~0%, a
                   sharpening (deconvolving) one is tens of percent
    D            - Aniano Eq. 20, the integral of the absolute difference;
                   more sensitive to shape than any single radius

Usage:
    python check2_psf_matching.py
    python check2_psf_matching.py --regularisation 1e-5
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve

# ------------------------------------------------------------------ CONFIG --
INPUT_ROOT = Path("~/Research/AsTrovello/Input").expanduser()

# Surveys to check. Each must have a PSF_CLEAN/ directory holding PSFs already
# resampled onto that survey's convolution grid.
SURVEYS = ["PHANGS-HST", "PHANGS-JWST"]

CRITERION_LO, CRITERION_HI = 0.95, 1.05

# Aniano et al. (2011), Sect. 7: "We do not recommend using any kernel with
# W- >~ 1.2". Converted to the normalised form reported here.
NEG_POWER_WARN_PCT = 100.0 * 1.2 / (1.0 + 2.4)

GRID_TOLERANCE = 0.02       # two PSFs must agree in scale to within 2%

# Everything this check produces goes under the repository, not the directory
# the script happens to be run from, so the outputs stay with the data they
# describe.
REPO_ROOT = Path("~/Research/AsTrovello").expanduser()
WORK_DIR = REPO_ROOT / "Output" / "checks" / "check2_psf_matching"

# Nominal FWHM in arcsec, as measured on the NATIVE (supersampled) PSF files by
# the pipeline's resolution stage. Reported here only for context, to show how
# well each source PSF is sampled on the convolution grid.
#
# It has to come from a table rather than from measuring the cleaned PSF: once
# resampled onto a coarse grid, a PSF narrower than the pixel is concentrated in
# a single pixel, and no measurement of the sampled image can recover its true
# width - the information is no longer there. That is exactly the HST case here
# (0.079" on a 0.198" grid), and it is why the previous attempt reported 0.00.
NOMINAL_FWHM_ARCSEC = {
    'f275w': 0.0766, 'f336w': 0.0810, 'f438w': 0.0838,
    'f555w': 0.0825, 'f814w': 0.0790,
    'f200w': 0.0623, 'f300m': 0.0951, 'f335m': 0.1070, 'f360m': 0.1154,
    'f770w': 0.2361, 'f1000w': 0.3119, 'f1130w': 0.3573, 'f2100w': 0.6435,
    'irac1': 1.5620, 'irac2': 1.5253,
}


# --------------------------------------------------------------- utilities --
def enclosed_radius(data, scale_arcsec, fraction=0.5):
    """Radius enclosing `fraction` of the flux, in arcsec.

    Sorts pixels by distance from the peak and interpolates where the
    cumulative flux crosses the requested fraction. Independent of profile
    shape, unlike a Gaussian fit - which matters for PSFs with broad wings.
    """
    peak = np.array(np.unravel_index(np.nanargmax(data), data.shape))
    y, x = np.indices(data.shape)
    r = np.hypot(y - peak[0], x - peak[1]).ravel()
    v = np.nan_to_num(data.ravel())
    order = np.argsort(r)
    cum = np.cumsum(v[order])
    if cum[-1] <= 0:
        return np.nan
    return float(np.interp(fraction, cum / cum[-1], r[order])) * scale_arcsec


def negative_power_pct(kernel):
    """Negative power as a percentage of the total absolute power.

    Relates to Aniano's W- by  pct = W- / (1 + 2 W-) * 100, because
    integral(|K|) = 1 + 2 W- for a unit-sum kernel.
    """
    total = np.sum(np.abs(kernel))
    if total <= 0:
        return np.nan
    return 100.0 * abs(np.sum(kernel[kernel < 0])) / total


def aniano_D(target, convolved):
    """D = integral |psi_B - K * psi_A| (Aniano Eq. 20). Perfect kernel: D = 0."""
    a = target / np.nansum(target)
    b = convolved / np.nansum(convolved)
    return float(np.nansum(np.abs(a - b)))


def filter_from_name(path):
    """Pull the filter token out of a PSF filename.

    Works for PSFSTD_WFC3UV_F814W.fits, PSF_NIRCam_..._F200W.fits and the
    master_ variants, by taking the last underscore-separated token that looks
    like a filter (F + digits + letter).
    """
    stem = path.stem
    for token in reversed(stem.split('_')):
        t = token.upper()
        if t.startswith('F') and any(c.isdigit() for c in t):
            return t.lower()
    return stem.lower()


def discover_pairs(survey_dir):
    """Return (master_path, [source_paths]) for one PSF_CLEAN directory."""
    psf_dir = survey_dir / "PSF_CLEAN"
    if not psf_dir.is_dir():
        return None, []
    masters = sorted(psf_dir.glob("master_*.fits"))
    sources = sorted(p for p in psf_dir.glob("*.fits")
                     if not p.name.startswith("master_"))
    if len(masters) != 1:
        print(f"  [!] expected exactly one master_*.fits in {psf_dir}, "
              f"found {len(masters)}")
        return None, []
    return masters[0], sources


def run_pypher(src, tgt, out, r_value):
    """Call PyPHER; return the kernel array, or None on failure."""
    out = Path(out)
    out.unlink(missing_ok=True)
    out.with_suffix('.log').unlink(missing_ok=True)
    cmd = ["pypher", str(src), str(tgt), str(out), "-r", f"{r_value:.3e}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("  [!] pypher is not on PATH. Activate the environment that has "
              "it, or install with: pip install pypher")
        return None
    if res.returncode != 0 or not out.exists():
        msg = res.stderr.strip().splitlines()[-1] if res.stderr else "no output"
        print(f"  [!] pypher failed: {msg}")
        return None
    return fits.getdata(out).astype(np.float64)


def measure(src_path, tgt_path, kernel, scale):
    """All four metrics for one pair."""
    psf_src = fits.getdata(src_path).astype(np.float64)
    psf_tgt = fits.getdata(tgt_path).astype(np.float64)

    # Negative power on the RAW kernel: after dividing by the sum, a
    # sign-flipped kernel (sum < 0, 100% negative) reads as 100% positive.
    ksum = kernel.sum()
    neg_raw = negative_power_pct(kernel)
    sign_flipped = ksum < 0

    kernel = kernel / ksum                  # unit sum, as the pipeline does
    conv = fftconvolve(psf_src, kernel, mode='same')
    r50_t = enclosed_radius(psf_tgt, scale, 0.5)
    r80_t = enclosed_radius(psf_tgt, scale, 0.8)
    return {
        'ratio_r50': enclosed_radius(conv, scale, 0.5) / r50_t if r50_t else np.nan,
        'ratio_r80': enclosed_radius(conv, scale, 0.8) / r80_t if r80_t else np.nan,
        'neg_pct': neg_raw,
        'sign_flipped': sign_flipped,
        'ksum': float(ksum),
        'D': aniano_D(psf_tgt, conv),
        'sampling': None,        # filled by the caller
    }


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--regularisation', type=float, default=1e-4,
                    help="PyPHER Wiener parameter r (default: 1e-4, PyPHER's own)")
    ap.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    args = ap.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for survey in SURVEYS:
        survey_dir = args.input_root / survey
        master, sources = discover_pairs(survey_dir)
        if master is None:
            print(f"\n[!] no usable PSF_CLEAN in {survey_dir}; skipping.")
            continue

        hdr_m = fits.getheader(master)
        scale_m = hdr_m.get('PIXSCALE')
        filt_m = filter_from_name(master)

        print(f"\n{'=' * 78}")
        print(f"{survey}   grid {scale_m:.4f} arcsec/px   master {filt_m}")
        print(f"{'=' * 78}")
        print(f"{'band':>8} {'px/FWHM':>9} {'r50 ratio':>11} {'r80 ratio':>11} "
              f"{'neg pwr':>9} {'D':>10}  verdict")
        print("-" * 78)

        work = WORK_DIR / survey
        work.mkdir(parents=True, exist_ok=True)

        for src in sources:
            filt = filter_from_name(src)
            scale_s = fits.getheader(src).get('PIXSCALE')

            # A kernel between PSFs on different grids is meaningless - this is
            # the defect that made the v1.x pipeline blur by a factor ~12 too
            # little. Skip rather than measure.
            if scale_s is None or scale_m is None:
                print(f"{filt:>8}  {'PIXSCALE missing':>50}")
                continue
            if abs(scale_s - scale_m) / scale_m > GRID_TOLERANCE:
                print(f"{filt:>8}  grid mismatch: {scale_s:.4f} vs "
                      f"{scale_m:.4f} arcsec/px - SKIPPED")
                continue

            kernel = run_pypher(src, master, work / f"kernel_{filt}.fits",
                                args.regularisation)
            if kernel is None:
                continue

            m = measure(src, master, kernel, scale_s)

            # Sampling of the SOURCE psf on this grid, for context. Taken from
            # the nominal table, not measured: see the note by the table.
            fwhm_nom = NOMINAL_FWHM_ARCSEC.get(filt)
            sampling = fwhm_nom / scale_s if fwhm_nom else np.nan

            ok = (CRITERION_LO <= m['ratio_r50'] <= CRITERION_HI
                  and not m['sign_flipped'])
            verdict = "PASS" if ok else "FAIL"
            if m['sign_flipped']:
                verdict = f"FAIL (sum {m['ksum']:+.3f} < 0)"
            if ok and m['neg_pct'] > NEG_POWER_WARN_PCT:
                verdict = "PASS (neg pwr high)"

            samp_str = f"{sampling:.2f}" if np.isfinite(sampling) else "?"
            print(f"{filt:>8} {samp_str:>9} {m['ratio_r50']:>11.4f} "
                  f"{m['ratio_r80']:>11.4f} {m['neg_pct']:>8.2f}% "
                  f"{m['D']:>10.2e}  {verdict}")

            rows.append({'survey': survey, 'band': filt, 'master': filt_m,
                         'ratio': m['ratio_r50'], 'ok': ok,
                         'neg': m['neg_pct'], 'D': m['D']})

    # ------------------------------------------------------------ summary ---
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"  kernels written to: {WORK_DIR}")
    if not rows:
        print("No pair could be measured. Check the PSF_CLEAN directories and "
              "that pypher is available.")
        return 2

    ratios = np.array([r['ratio'] for r in rows])
    failed = [r for r in rows if not r['ok']]

    print(f"  pairs measured    : {len(rows)}")
    print(f"  ratio range       : {ratios.min():.4f} to {ratios.max():.4f}")
    print(f"  median ratio      : {np.median(ratios):.4f}")
    print(f"  criterion         : {CRITERION_LO} <= ratio <= {CRITERION_HI}")
    print(f"  regularisation r  : {args.regularisation:.0e}")

    high_neg = [r for r in rows if r['neg'] > NEG_POWER_WARN_PCT]
    if high_neg:
        print(f"\n  [!] {len(high_neg)} kernel(s) with negative power above "
              f"{NEG_POWER_WARN_PCT:.0f}% (Aniano's W- ~ 1.2 limit):")
        for r in high_neg:
            print(f"        {r['survey']} {r['band']}: {r['neg']:.1f}%")

    print()
    if failed:
        print(f"VERDICT: FAIL - {len(failed)} of {len(rows)} pair(s) outside "
              f"the criterion:")
        for r in failed:
            print(f"    {r['survey']:>12} {r['band']:>7}: ratio {r['ratio']:.4f}")
        return 1

    print(f"VERDICT: PASS - all {len(rows)} pairs within "
          f"{CRITERION_LO}-{CRITERION_HI}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

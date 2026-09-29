"""
Does removing the orphaned SIP coefficients fix the irac1 flux deviation?

Check 3 measured irac1 at 0.9916 (0.84% off, 1.43% scatter) - the worst of
fourteen bands, and the only one that was NOT resampled: its grid is 0.75"
before and after. A band that is not resampled should conserve flux exactly.

The diagnosis: `reproject_to_reference` replaced the linear WCS with the
reference's but left the SOURCE's SIP coefficients in the header. astropy
applies SIP whenever the keys are present, so the output file carries the
reference linear WCS combined with the source distortion. The same sky
aperture then lands on different pixels either side of the reprojection.

This script tests that diagnosis WITHOUT re-running the pipeline. It takes the
existing reprojected file, writes a copy with the SIP keys stripped - which is
what the corrected code now produces - and re-measures both against the same
"before" file.

    if the deviation falls from ~0.84% to ~0.1%, the diagnosis is confirmed
    if it does not move, the cause is elsewhere and the fix is unrelated

Usage:
    python test_sip_hypothesis.py --galaxy ngc1087
"""

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

warnings.filterwarnings('ignore')

REPO_ROOT = Path("~/Research/AsTrovello").expanduser()
WORK_DIR = REPO_ROOT / "Output" / "checks" / "sip_hypothesis"

SIP_PREFIXES = ('A_', 'B_', 'AP_', 'BP_')


def sip_keys(header):
    """SIP-related keys present in a header."""
    return [k for k in header if k.startswith(SIP_PREFIXES)]


def strip_sip(src_path, dst_path):
    """Copy a file with every SIP key removed, as the corrected code writes it."""
    shutil.copy2(src_path, dst_path)
    with fits.open(dst_path, mode='update') as hdul:
        hdr = hdul[0].header
        removed = sip_keys(hdr)
        for k in removed:
            del hdr[k]
        for k in ('CTYPE1', 'CTYPE2'):
            if k in hdr:
                hdr[k] = str(hdr[k]).replace('-SIP', '')
        hdul.flush()
    return removed


def astrometric_shift(path_with, path_without):
    """How far the SIP keys move pixel-to-sky, across the field."""
    with fits.open(path_with) as h:
        w_with = WCS(h[0].header, naxis=2)
        ny, nx = h[0].data.shape
    with fits.open(path_without) as h:
        w_without = WCS(h[0].header, naxis=2)

    seps = []
    for fy in (0.2, 0.4, 0.5, 0.6, 0.8):
        for fx in (0.2, 0.4, 0.5, 0.6, 0.8):
            px, py = fx * nx, fy * ny
            c1 = w_with.pixel_to_world(px, py)
            c2 = w_without.pixel_to_world(px, py)
            seps.append(c1.separation(c2).arcsec)
    return np.array(seps)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxy', required=True)
    ap.add_argument('--band', default='irac1',
                    help='band to test (default: irac1, the anomalous one)')
    ap.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    args = ap.parse_args()

    gal = args.galaxy.lower()
    conv_dir = args.repo_root / 'Output' / 'convolved_fits' / gal
    proj_dir = args.repo_root / 'Output' / 'reprojected_files' / gal

    before = sorted(conv_dir.glob(f'*{args.band}*convolved.fits')) + \
             sorted(conv_dir.glob(f'*{args.band}*master.fits'))
    after = sorted(proj_dir.glob(f'*{args.band}*projection.fits'))

    if not before or not after:
        print(f"Could not find both files for {args.band}:")
        print(f"  convolved   : {[p.name for p in before] or 'none'}")
        print(f"  reprojected : {[p.name for p in after] or 'none'}")
        print(f"\nThese are produced by the HST+JWST+S4G configuration. If the")
        print(f"last run was PHANGS-only, they will not be present.")
        return 2

    path_before, path_after = before[0], after[0]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    path_after_fixed = WORK_DIR / f'{path_after.stem}_nosip.fits'

    print(f"\n{'=' * 74}")
    print(f"SIP hypothesis test - {args.band} ({gal})")
    print(f"{'=' * 74}")
    print(f"  before : {path_before.name}")
    print(f"  after  : {path_after.name}")

    # --- what is actually in the headers -----------------------------------
    hdr_b = fits.getheader(path_before)
    hdr_a = fits.getheader(path_after)
    print(f"\n  SIP keys in the convolved file   : {len(sip_keys(hdr_b))}")
    print(f"  SIP keys in the reprojected file : {len(sip_keys(hdr_a))}")
    print(f"  CTYPE1 before / after            : "
          f"{hdr_b.get('CTYPE1')} / {hdr_a.get('CTYPE1')}")

    if not sip_keys(hdr_a):
        print("\n  The reprojected file carries no SIP keys, so either it was")
        print("  written by the corrected code already, or this band never had")
        print("  them. Nothing to test here.")
        return 0

    removed = strip_sip(path_after, path_after_fixed)
    print(f"\n  wrote a copy without SIP: {path_after_fixed.name}")
    print(f"  ({len(removed)} keys removed)")

    # --- how much do those keys move the astrometry? -----------------------
    shifts = astrometric_shift(path_after, path_after_fixed)
    scale = np.sqrt(np.abs(np.linalg.det(
        WCS(hdr_a, naxis=2).pixel_scale_matrix))) * 3600
    print(f"\n  Astrometric shift caused by the orphaned SIP, over 25 points:")
    print(f"    median {np.median(shifts):.4f}\"  "
          f"min {shifts.min():.4f}\"  max {shifts.max():.4f}\"")
    print(f"    in pixels of this {scale:.4f}\" grid: "
          f"median {np.median(shifts)/scale:.3f}, max {shifts.max()/scale:.3f}")

    # --- re-measure flux conservation both ways ----------------------------
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from validation import check_flux_conservation
    except ImportError:
        print("\n  validation.py not importable from here; run this script from")
        print("  the directory that holds it to get the flux comparison.")
        return 0

    print(f"\n{'-' * 74}")
    print("  flux conservation, WITH the orphaned SIP (as the old code wrote it)")
    res_old = check_flux_conservation([(args.band, path_before, path_after)],
                                      raise_on_fail=False)
    print(f"\n  flux conservation, WITHOUT it (as the corrected code writes it)")
    res_new = check_flux_conservation([(args.band, path_before, path_after_fixed)],
                                      raise_on_fail=False)

    # --- verdict ------------------------------------------------------------
    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    if not res_old or not res_new:
        print("  One of the measurements produced nothing; cannot compare.")
        return 2

    dev_old = abs(res_old[0]['ratio'] - 1) * 100
    dev_new = abs(res_new[0]['ratio'] - 1) * 100
    print(f"  deviation with SIP    : {dev_old:.3f}%  "
          f"(scatter {res_old[0]['scatter']:.2f}%)")
    print(f"  deviation without SIP : {dev_new:.3f}%  "
          f"(scatter {res_new[0]['scatter']:.2f}%)")

    # A deviation that was already negligible cannot be "fixed": comparing two
    # numbers near zero would produce a verdict out of rounding noise.
    if dev_old < 0.1:
        print(f"\n  INCONCLUSIVE. The deviation was already below 0.1% before")
        print(f"  removing the SIP keys, so there is nothing for them to")
        print(f"  explain. This band does not reproduce the anomaly.")
    elif dev_new < dev_old / 2:
        print(f"\n  CONFIRMED. Removing the orphaned SIP coefficients accounts")
        print(f"  for most of the deviation. It was a WCS defect in the")
        print(f"  reprojected header, not a loss of flux.")
    elif dev_new < dev_old * 0.9:
        print(f"\n  PARTIAL. The SIP keys explain some of the deviation but not")
        print(f"  all of it. Something else contributes.")
    else:
        print(f"\n  NOT CONFIRMED. The deviation does not come from the SIP")
        print(f"  coefficients. Look elsewhere.")

    print(f"\n  Note: the astrometric shift above is a property of the header")
    print(f"  and is real regardless of this verdict. Whether it shows up as a")
    print(f"  flux deviation depends on the aperture size relative to the")
    print(f"  shift - a 3\" aperture is forgiving of a 0.05\" displacement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

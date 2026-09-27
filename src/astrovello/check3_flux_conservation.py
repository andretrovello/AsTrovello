"""
Check 3 - flux conservation through reprojection.

reproject_interp resamples an image onto another pixel grid by interpolation.
Interpolation preserves values but does not strictly conserve summed flux, and
a poorly behaved resampling leaks flux in a way that is invisible by eye. This
check measures the leak.

    criterion, agreed in advance: flux conserved to within 1%

WHAT IS COMPARED, AND WHY

The measurement is made in APERTURES FIXED IN SKY COORDINATES - the same patch
of sky on each grid - and the sum in each aperture is WEIGHTED BY THE PIXEL
AREA of its grid.

The weighting is the crux. reproject_interp interpolates, which preserves the
VALUE of each pixel rather than the sum: that is surface-brightness behaviour,
and it is what the pipeline relies on (reproject_to_reference records it in the
output header, and convert2Jansky later multiplies by the true pixel area).
Summing raw values across grids would therefore measure the change of pixel
size, not a loss of flux. Concretely, going from the 0.1981" HST convolution
grid to a 0.75" master grid, the raw sum falls by (0.75/0.1981)^2 = 14.3 with
no photon lost anywhere. Multiplying each sum by its own pixel area removes
that factor and leaves the quantity that must actually be conserved.

The calibration constant of the band is irrelevant: the check compares the SAME
band with itself across the resampling, so any constant cancels in the ratio.

Apertures are placed on real structure - bright compact sources - rather than
at random, because empty sky has no flux to conserve and would return a ratio
dominated by noise.

Usage:
    python check3_flux_conservation.py --galaxy ngc1087
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

try:
    from photutils.aperture import SkyCircularAperture, aperture_photometry
except ImportError:
    print("This check needs photutils:  conda install -c conda-forge photutils")
    sys.exit(2)

# ------------------------------------------------------------------ CONFIG --
REPO_ROOT = Path("~/Research/AsTrovello").expanduser()
OUTPUT_ROOT = REPO_ROOT / "Output"

# Where this check writes anything it produces (currently only the report is
# printed, but keeping the path here means a future plot lands with the data).
CHECK_DIR = OUTPUT_ROOT / "checks" / "check3_flux_conservation"

CRITERION_PCT = 1.0          # flux must be conserved to within this

# Aperture radius in arcsec. Must be comfortably larger than the PSF of the
# convolved images (all matched to the master, FWHM ~0.64" for f2100w) so the
# aperture captures essentially all the flux of a point source; otherwise a
# sub-pixel shift moves flux across the boundary and shows up as a false leak.
APERTURE_RADIUS_ARCSEC = 3.0

N_APERTURES = 8              # how many bright sources to test
MIN_SEPARATION_ARCSEC = 10.0  # keep apertures from overlapping


def find_bright_sources(data, wcs, n_sources, min_sep_arcsec, radius_arcsec):
    """Pick the n brightest well-separated peaks, as SkyCoord.

    Deliberately simple: this is not source detection, only a way of placing
    apertures where there is flux to conserve. Peaks are taken from a coarsely
    smoothed copy so that single hot pixels are not chosen.
    """
    from scipy.ndimage import uniform_filter, maximum_filter

    finite = np.isfinite(data)
    if finite.sum() == 0:
        return []

    smooth = uniform_filter(np.nan_to_num(data), size=5)
    smooth[~finite] = -np.inf

    # Local maxima. The window is sized to the APERTURE, not to the requested
    # separation: on a fine grid the separation would translate into a window
    # of hundreds of pixels, which lets a single peak through for the whole
    # image. The minimum separation is enforced later, when the apertures are
    # chosen, where it belongs.
    scale_tmp = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0
    win = max(int(radius_arcsec / scale_tmp) | 1, 3)    # odd, at least 3
    peaks = (smooth == maximum_filter(smooth, size=win)) & finite
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return []

    order = np.argsort(smooth[ys, xs])[::-1]
    ys, xs = ys[order], xs[order]

    # stay away from the border by at least the aperture radius
    margin = int(np.ceil(1.5 * radius_arcsec / scale_tmp))
    ny, nx = data.shape
    keep = (ys > margin) & (ys < ny - margin) & (xs > margin) & (xs < nx - margin)
    ys, xs = ys[keep], xs[keep]

    chosen = []
    coords = []
    for y, x in zip(ys, xs):
        c = wcs.pixel_to_world(x, y)
        if any(c.separation(o).arcsec < min_sep_arcsec for o in coords):
            continue
        coords.append(c)
        chosen.append((y, x))
        if len(coords) >= n_sources:
            break
    return coords


def aperture_flux(data, wcs, coords, radius_arcsec):
    """Area-weighted flux in each sky aperture, plus the fraction covered.

    The sum is multiplied by the pixel area of THIS grid, so the result is a
    quantity proportional to the physical flux and directly comparable between
    grids of different pixel size. Without that factor the comparison measures
    the change of pixel size (see the module docstring).
    """
    ap = SkyCircularAperture(SkyCoord(coords), r=radius_arcsec * u.arcsec)
    ap_pix = ap.to_pixel(wcs)

    # NaNs are off-footprint pixels; treat them as zero contribution but record
    # how much of the aperture they cover, since a partially blank aperture
    # cannot be compared fairly
    filled = np.nan_to_num(data)
    phot = aperture_photometry(filled, ap_pix)
    covered = aperture_photometry(np.isfinite(data).astype(float), ap_pix)

    pixel_area_arcsec2 = np.abs(np.linalg.det(wcs.pixel_scale_matrix)) * 3600 ** 2
    flux = np.asarray(phot['aperture_sum']) * pixel_area_arcsec2
    fraction = np.asarray(covered['aperture_sum']) / ap_pix.area
    return flux, fraction


def check_band(path_before, path_after, radius_arcsec, n_ap, min_sep):
    """Compare one band before and after reprojection."""
    with fits.open(path_before) as h:
        data_b, wcs_b = h[0].data.astype(np.float64), WCS(h[0].header, naxis=2)
    with fits.open(path_after) as h:
        data_a, wcs_a = h[0].data.astype(np.float64), WCS(h[0].header, naxis=2)

    scale_b = np.sqrt(np.abs(np.linalg.det(wcs_b.pixel_scale_matrix))) * 3600
    scale_a = np.sqrt(np.abs(np.linalg.det(wcs_a.pixel_scale_matrix))) * 3600

    coords = find_bright_sources(data_b, wcs_b, n_ap, min_sep, radius_arcsec)
    if not coords:
        return None, "no usable source found"

    flux_b, cov_b = aperture_flux(data_b, wcs_b, coords, radius_arcsec)
    flux_a, cov_a = aperture_flux(data_a, wcs_a, coords, radius_arcsec)

    # An aperture qualifies only if it is essentially complete on BOTH grids
    # AND actually contains signal. An aperture on blank sky returns noise over
    # noise: the ratio is meaningless and its scatter swamps the statistic.
    # The signal floor is set relative to the brightest aperture, so it adapts
    # to the calibration of each band instead of assuming an absolute level.
    complete = (cov_b > 0.99) & (cov_a > 0.99)
    if not complete.any():
        return None, "every aperture falls partly outside one of the grids"

    peak = np.nanmax(np.abs(flux_b[complete]))
    has_signal = np.abs(flux_b) > 0.05 * peak
    ok = complete & has_signal
    if ok.sum() == 0:
        return None, "no aperture contains enough signal to compare"

    ratio = flux_a[ok] / flux_b[ok]
    return {
        'n_ap': int(ok.sum()),
        'n_tried': len(coords),
        'n_no_signal': int((complete & ~has_signal).sum()),
        'scale_b': scale_b,
        'scale_a': scale_a,
        'ratio_median': float(np.median(ratio)),
        'ratio_min': float(ratio.min()),
        'ratio_max': float(ratio.max()),
        'scatter_pct': float(100 * np.std(ratio)),
    }, None


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--galaxy', required=True)
    ap.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    ap.add_argument('--radius', type=float, default=APERTURE_RADIUS_ARCSEC,
                    help='aperture radius in arcsec')
    ap.add_argument('--n-apertures', type=int, default=N_APERTURES)
    args = ap.parse_args()

    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    gal = args.galaxy.lower()
    dir_before = args.output_root / 'convolved_fits' / gal
    dir_after = args.output_root / 'reprojected_files' / gal

    if not dir_before.is_dir() or not dir_after.is_dir():
        print(f"Missing input directories:\n  {dir_before}\n  {dir_after}")
        return 2

    # pair up the convolved file with its reprojected counterpart by filter
    def filt_of(p):
        parts = p.stem.split('_')
        if p.stem.endswith('_master'):
            return parts[-2]
        for i, tok in enumerate(parts):
            if tok in ('to', 'on') and i >= 1:
                return parts[i - 1]
        return None

    before = {filt_of(p): p for p in sorted(dir_before.glob('*.fits'))}
    after = {filt_of(p): p for p in sorted(dir_after.glob('*projection.fits'))}
    common = sorted(set(before) & set(after) - {None})

    if not common:
        print("No band has both a convolved and a reprojected file.")
        print(f"  convolved   : {sorted(k for k in before if k)}")
        print(f"  reprojected : {sorted(k for k in after if k)}")
        return 2

    print(f"\n{'=' * 78}")
    print(f"CHECK 3 - flux conservation through reprojection   ({gal})")
    print(f"aperture radius {args.radius}\", criterion {CRITERION_PCT}%")
    print(f"{'=' * 78}")
    print(f"{'band':>8} {'n ap':>6} {'grid in':>9} {'grid out':>9} "
          f"{'median':>9} {'scatter':>9}  verdict")
    print("-" * 78)

    rows = []
    for filt in common:
        res, err = check_band(before[filt], after[filt], args.radius,
                              args.n_apertures, MIN_SEPARATION_ARCSEC)
        if res is None:
            print(f"{filt:>8}  {err}")
            continue
        dev_pct = abs(res['ratio_median'] - 1.0) * 100
        ok = dev_pct <= CRITERION_PCT
        print(f"{filt:>8} {res['n_ap']:>6} {res['scale_b']:>9.4f} "
              f"{res['scale_a']:>9.4f} {res['ratio_median']:>9.4f} "
              f"{res['scatter_pct']:>8.2f}%  "
              f"{'PASS' if ok else 'FAIL'}  ({dev_pct:+.2f}%)")
        rows.append((filt, res, ok))

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    if not rows:
        print("Nothing could be measured.")
        return 2

    failed = [r for r in rows if not r[2]]
    devs = [abs(r[1]['ratio_median'] - 1) * 100 for r in rows]
    print(f"  bands measured    : {len(rows)}")
    print(f"  worst deviation   : {max(devs):.3f}%")
    print(f"  median deviation  : {np.median(devs):.3f}%")
    print()
    if failed:
        print(f"VERDICT: FAIL - {len(failed)} band(s) leak more than "
              f"{CRITERION_PCT}%:")
        for filt, res, _ in failed:
            print(f"    {filt}: {(res['ratio_median'] - 1) * 100:+.2f}%")
        return 1
    print(f"VERDICT: PASS - all {len(rows)} bands conserve flux to within "
          f"{CRITERION_PCT}%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
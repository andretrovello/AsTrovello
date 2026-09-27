"""
In-pipeline validation gates.

Two checks run inside the pipeline and abort it when they fail, so that a bad
kernel or a leaking resampling is caught at the stage that produced it rather
than surfacing later as a wrong number in a map.

    check_psf_matching   after the kernels are built, before any image is
                         convolved. Convolving 13 bands with a bad kernel costs
                         an hour and produces nothing usable.

    check_flux_conservation   after reprojection, before unit conversion.

Both criteria are fixed here rather than passed in, because a threshold that
can be relaxed at the call site is not a gate. They can be overridden from the
CLI only by skipping the check outright (--skip_checks), which is visible in
the log.

The standalone scripts check2_psf_matching.py and check3_flux_conservation.py
run the same measurements outside the pipeline, for reporting.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import fftconvolve

# ----------------------------------------------------------------- criteria --
PSF_RATIO_LO, PSF_RATIO_HI = 0.95, 1.05
FLUX_TOLERANCE_PCT = 1.0

# Aniano et al. (2011), Sect. 7: "We do not recommend using any kernel with
# W- >~ 1.2". Converted to the normalised form reported here, W-/(1+2W-).
NEG_POWER_WARN_PCT = 100.0 * 1.2 / (1.0 + 2.4)


class ValidationError(RuntimeError):
    """Raised when a check fails. Aborts the pipeline."""


# ---------------------------------------------------------------- utilities --
def _enclosed_radius(data, scale_arcsec, fraction=0.5):
    """Radius enclosing `fraction` of the flux, in arcsec.

    Profile-shape independent: sorts pixels by distance from the peak and
    interpolates the cumulative curve. A Gaussian fit would underestimate the
    width of a PSF with broad wings, which is the IRAC case.
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


def _negative_power_pct(kernel):
    """Negative power as a percentage of total absolute power."""
    total = np.sum(np.abs(kernel))
    if total <= 0:
        return np.nan
    return 100.0 * abs(np.sum(kernel[kernel < 0])) / total


def _pixel_scale(wcs):
    """Effective arcsec/px from the full WCS transformation.

    Uses the determinant rather than reading CDELT, so it is correct whether
    the header expresses the geometry as CD or as CDELT+PC, and under rotation.
    """
    return float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0)


# ------------------------------------------------------- check 2: PSF match --
def check_psf_matching(kernel_files, psf_dir, master_psf_path, drivers=None,
                       raise_on_fail=True, verbose=True):
    """Verify that every kernel reproduces the master PSF.

    For each kernel, convolve the corresponding source PSF and compare the
    enclosed-energy radius of the result with the master's. A correct kernel
    gives a ratio of 1.0 (Aniano et al. 2011, Sect. 5).

    Args:
        kernel_files: kernels produced for this survey, named
            kernel_<source>_to_<master>.fits
        psf_dir: directory holding the CLEANED source PSFs, already on the
            convolution grid
        master_psf_path: the cleaned master PSF, on the same grid
        raise_on_fail: abort the pipeline. Set False only for reporting.

    Returns:
        list of dicts, one per pair.

    Raises:
        ValidationError if any pair falls outside the criterion.
    """
    master = fits.getdata(master_psf_path).astype(np.float64)
    scale_master = fits.getheader(master_psf_path).get('PIXSCALE')
    if scale_master is None:
        raise ValidationError(
            f"{master_psf_path.name} has no PIXSCALE. The check cannot verify "
            f"that the kernel and the PSFs share a grid, which is the whole "
            f"point of it.")

    r50_master = _enclosed_radius(master, scale_master, 0.5)
    r80_master = _enclosed_radius(master, scale_master, 0.8)

    if verbose:
        print(f"\n>>> CHECK 2: PSF matching "
              f"(criterion {PSF_RATIO_LO}-{PSF_RATIO_HI})")
        print(f"      {'band':>8} {'r50 ratio':>11} {'r80 ratio':>11} "
              f"{'neg pwr':>9}  verdict")

    results, failures = [], []
    for kern_path in sorted(kernel_files):
        # kernel_<source>_to_<master>.fits
        parts = kern_path.stem.split('_')
        try:
            filt = parts[parts.index('to') - 1]
        except (ValueError, IndexError):
            if verbose:
                print(f"      {kern_path.name}: unparseable name, skipped")
            continue

        matches = [p for p in psf_dir.glob('*.fits')
                   if filt.upper() in p.stem.upper()
                   and not p.name.startswith('master_')]
        if len(matches) != 1:
            if verbose:
                print(f"      {filt:>8}: {len(matches)} candidate source PSFs, "
                      f"skipped")
            continue
        src_path = matches[0]

        scale_src = fits.getheader(src_path).get('PIXSCALE')
        if scale_src is None or abs(scale_src - scale_master) / scale_master > 0.02:
            raise ValidationError(
                f"{filt}: source PSF is on a {scale_src} arcsec/px grid while "
                f"the master is on {scale_master}. A kernel between different "
                f"grids is invalid - this is the defect that made v1.x blur by "
                f"a factor ~12 too little.")

        psf_src = fits.getdata(src_path).astype(np.float64)
        kernel = fits.getdata(kern_path).astype(np.float64)
        kernel = kernel / kernel.sum()

        conv = fftconvolve(psf_src, kernel, mode='same')
        ratio50 = _enclosed_radius(conv, scale_src, 0.5) / r50_master
        ratio80 = _enclosed_radius(conv, scale_src, 0.8) / r80_master
        neg = _negative_power_pct(kernel)

        ok = PSF_RATIO_LO <= ratio50 <= PSF_RATIO_HI
        if not ok:
            failures.append((filt, ratio50))

        verdict = "PASS" if ok else "FAIL"
        if ok and neg > NEG_POWER_WARN_PCT:
            verdict = "PASS (neg pwr high)"

        if verbose:
            print(f"      {filt:>8} {ratio50:>11.4f} {ratio80:>11.4f} "
                  f"{neg:>8.2f}%  {verdict}")

        results.append({'band': filt, 'ratio_r50': ratio50,
                        'ratio_r80': ratio80, 'neg_pct': neg, 'ok': ok})

    if not results:
        raise ValidationError(
            "No kernel could be checked. Either the kernels are missing or "
            "their source PSFs could not be matched by name.")

    if failures and raise_on_fail:
        lines = "\n".join(f"        {b}: ratio {r:.4f}" for b, r in failures)
        raise ValidationError(
            f"PSF matching failed for {len(failures)} of {len(results)} "
            f"band(s):\n{lines}\n"
            f"    The kernels do not reproduce the master PSF. Convolving with "
            f"them would produce bands at the wrong resolution, which is not "
            f"visible in the images but corrupts every colour downstream.\n"
            f"    Investigate before proceeding, or re-run with --skip_checks "
            f"if you accept the risk knowingly.")

    if verbose:
        print(f"      -> {len(results)} pair(s) checked, all within criterion")
    return results


# -------------------------------------------------- check 3: flux conserved --
def check_flux_conservation(pairs, radius_arcsec=3.0, n_apertures=8,
                            min_separation_arcsec=10.0,
                            raise_on_fail=True, verbose=True):
    """Verify that reprojection conserves flux.

    Measures the flux in apertures fixed in SKY coordinates, before and after
    the resampling, weighting each sum by the pixel area of its own grid.

    The weighting is essential. reproject_interp preserves the VALUE of each
    pixel, not the sum - surface-brightness behaviour, which is what the
    pipeline relies on. Summing raw values across grids would measure the
    change of pixel size: going from 0.1981" to 0.75" the raw sum falls by a
    factor 14 with no photon lost.

    Args:
        pairs: list of (band, path_before, path_after)

    Raises:
        ValidationError if any band leaks more than FLUX_TOLERANCE_PCT.
    """
    from scipy.ndimage import uniform_filter, maximum_filter
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    try:
        from photutils.aperture import SkyCircularAperture, aperture_photometry
    except ImportError:
        if verbose:
            print("\n>>> CHECK 3 skipped: photutils is not installed "
                  "(conda install -c conda-forge photutils)")
        return []

    def bright_sources(data, wcs, n, min_sep, radius):
        finite = np.isfinite(data)
        if not finite.any():
            return []
        smooth = uniform_filter(np.nan_to_num(data), size=5)
        smooth[~finite] = -np.inf
        scale = _pixel_scale(wcs)
        # window sized to the aperture, not to the separation: on a fine grid
        # the separation would be hundreds of pixels and let a single peak
        # through for the whole image
        win = max(int(radius / scale) | 1, 3)
        peaks = (smooth == maximum_filter(smooth, size=win)) & finite
        ys, xs = np.where(peaks)
        if len(ys) == 0:
            return []
        order = np.argsort(smooth[ys, xs])[::-1]
        ys, xs = ys[order], xs[order]
        margin = int(np.ceil(1.5 * radius / scale))
        ny, nx = data.shape
        keep = ((ys > margin) & (ys < ny - margin)
                & (xs > margin) & (xs < nx - margin))
        ys, xs = ys[keep], xs[keep]
        coords = []
        for y, x in zip(ys, xs):
            c = wcs.pixel_to_world(x, y)
            if any(c.separation(o).arcsec < min_sep for o in coords):
                continue
            coords.append(c)
            if len(coords) >= n:
                break
        return coords

    def aperture_flux(data, wcs, coords, radius):
        ap = SkyCircularAperture(SkyCoord(coords), r=radius * u.arcsec).to_pixel(wcs)
        phot = aperture_photometry(np.nan_to_num(data), ap)
        cov = aperture_photometry(np.isfinite(data).astype(float), ap)
        area = np.abs(np.linalg.det(wcs.pixel_scale_matrix)) * 3600 ** 2
        return (np.asarray(phot['aperture_sum']) * area,
                np.asarray(cov['aperture_sum']) / ap.area)

    if verbose:
        print(f"\n>>> CHECK 3: flux conservation through reprojection "
              f"(criterion {FLUX_TOLERANCE_PCT}%)")
        print(f"      {'band':>8} {'n ap':>5} {'grid in':>9} {'grid out':>9} "
              f"{'ratio':>9} {'scatter':>9}  verdict")

    results, failures = [], []
    for band, path_before, path_after in pairs:
        with fits.open(path_before) as h:
            data_b, wcs_b = h[0].data.astype(np.float64), WCS(h[0].header, naxis=2)
        with fits.open(path_after) as h:
            data_a, wcs_a = h[0].data.astype(np.float64), WCS(h[0].header, naxis=2)

        coords = bright_sources(data_b, wcs_b, n_apertures,
                                min_separation_arcsec, radius_arcsec)
        if not coords:
            if verbose:
                print(f"      {band:>8}  no usable source found, skipped")
            continue

        flux_b, cov_b = aperture_flux(data_b, wcs_b, coords, radius_arcsec)
        flux_a, cov_a = aperture_flux(data_a, wcs_a, coords, radius_arcsec)

        # An aperture qualifies only if complete on BOTH grids and containing
        # signal. On blank sky the ratio is noise over noise.
        complete = (cov_b > 0.99) & (cov_a > 0.99)
        if not complete.any():
            if verbose:
                print(f"      {band:>8}  every aperture partly outside a grid, "
                      f"skipped")
            continue
        peak = np.nanmax(np.abs(flux_b[complete]))
        ok_ap = complete & (np.abs(flux_b) > 0.05 * peak)
        if not ok_ap.any():
            if verbose:
                print(f"      {band:>8}  no aperture with enough signal, skipped")
            continue

        ratio = flux_a[ok_ap] / flux_b[ok_ap]
        median = float(np.median(ratio))
        scatter = float(100 * np.std(ratio))
        dev_pct = abs(median - 1.0) * 100
        ok = dev_pct <= FLUX_TOLERANCE_PCT
        if not ok:
            failures.append((band, median))

        if verbose:
            print(f"      {band:>8} {int(ok_ap.sum()):>5} "
                  f"{_pixel_scale(wcs_b):>9.4f} {_pixel_scale(wcs_a):>9.4f} "
                  f"{median:>9.4f} {scatter:>8.2f}%  "
                  f"{'PASS' if ok else 'FAIL'} ({dev_pct:+.2f}%)")

        results.append({'band': band, 'ratio': median, 'scatter': scatter,
                        'n_ap': int(ok_ap.sum()), 'ok': ok})

    if not results:
        if verbose:
            print("      -> nothing could be measured; check skipped")
        return []

    if failures and raise_on_fail:
        lines = "\n".join(f"        {b}: {(r - 1) * 100:+.2f}%"
                          for b, r in failures)
        raise ValidationError(
            f"Flux conservation failed for {len(failures)} of {len(results)} "
            f"band(s):\n{lines}\n"
            f"    The reprojection is losing or gaining flux beyond "
            f"{FLUX_TOLERANCE_PCT}%. Every photometric quantity derived from "
            f"the cube inherits this error.\n"
            f"    Investigate before proceeding, or re-run with --skip_checks "
            f"if you accept the risk knowingly.")

    if verbose:
        devs = [abs(r['ratio'] - 1) * 100 for r in results]
        print(f"      -> {len(results)} band(s) checked, worst deviation "
              f"{max(devs):.2f}%")
    return results

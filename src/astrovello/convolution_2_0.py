"""
Convolution utilities for the AsTrovello pipeline.

PSF matching (homogenisation) across surveys, and convolution of the science
images.

Central principle
-----------------
PyPHER returns the kernel on the grid of the PSFs it was given. Since that
kernel is later applied to a science image, **the PSFs must live on the grid
where the convolution will happen** - not on each detector's native grid.

That is why `clean_psf` takes `target_pixel_scale_arcsec` explicitly and
`pypher_kernel_creation` verifies that every PSF sits on that grid before
building the commands. The check raises at kernel-generation time rather than
six stages later in the age map.

The convolution grid may be coarser than the image's native grid; see
`bin_for_convolution`. Since the final product is resampled onto the master
grid anyway, convolving on HST's native grid costs 25x the memory for no
physical gain.

Resolvability limit
-------------------
Matching two PSFs of similar width requires a kernel of width
`sqrt(target^2 - source^2)`. Below roughly 2 pixels of the working grid the
kernel degenerates into a near-delta and the Fourier division returns sinc
side lobes - ringing, not matching. `psf_matching_resolvable` detects this and
`pypher_kernel_creation` skips the pair, leaving the band unmatched with the
residual mismatch recorded in the header (`PSFRESID`).

This is what happened to the IRAC1/IRAC2 pair in NGC 2903: r80 of 2.0864" and
2.1878" require a 0.658" kernel, which is 0.88 px on the 0.75"/px grid. In
both directions PyPHER returned ~30% negative power and ~0.2 px of spurious
offset, to correct a sub-pixel effect. Not matching is the better choice.

Choosing the working grid
-------------------------
The binning factor is a property of the TARGET, not of the source survey: it
depends on which filter ends up being the master. `choose_bin_factor` derives
it and `report_convolution_grid` prints the consequences, so the choice is
informed rather than inherited from a config file.
"""

from pathlib import Path
import os
import shutil
import warnings

import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.nddata import block_reduce
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from scipy.ndimage import zoom, shift as ndshift


# ======================================================================
# PSF width measurements
# ======================================================================
def get_fwhm(data: np.ndarray) -> float:
    """FWHM in pixels, from a 2D Gaussian fit.

    Warning: for PSFs with broad wings (the IRAC PRF, for instance) this
    estimator responds to the CORE and underestimates the true width by ~25%.
    Fine for an order of magnitude, but NOT reliable for ranking two similar
    PSFs - use `get_half_light_radius` for that.
    """
    data_clean = np.nan_to_num(data, nan=0.0)
    y, x = np.mgrid[:data_clean.shape[0], :data_clean.shape[1]]

    max_val = np.max(data_clean)
    y_center, x_center = np.unravel_index(np.argmax(data_clean), data_clean.shape)

    g_init = models.Gaussian2D(amplitude=max_val, x_mean=x_center, y_mean=y_center,
                               x_stddev=2.0, y_stddev=2.0)
    fit_g = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g_fit = fit_g(g_init, x, y, data_clean)

    # geometric mean copes better with slightly elliptical PSFs
    sigma_eff = np.sqrt(abs(g_fit.x_stddev.value * g_fit.y_stddev.value))
    return float(2.3548 * sigma_eff)


def get_half_light_radius(data: np.ndarray, pixel_scale_arcsec: float = 1.0,
                          fraction: float = 0.5) -> float:
    """Radius enclosing `fraction` of the flux, in arcsec (or px if scale = 1).

    Independent of the profile shape, unlike a Gaussian fit. A robust
    background (median of the outer annulus) is subtracted before
    accumulating: without it the cumulative sum grows with area and the radius
    loses meaning.

    Use `fraction=0.8` or `0.9` to probe the wings, which dominate PSF
    matching.
    """
    d = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0)
    if d.ndim == 3:
        d = d.mean(axis=0)

    cy, cx = np.unravel_index(np.argmax(d), d.shape)
    y, x = np.indices(d.shape)
    r = np.sqrt((x - cx) ** 2.0 + (y - cy) ** 2.0)
    r_max = min(d.shape) / 2.0

    outer = (r > 0.85 * r_max) & (r <= r_max)
    if outer.sum() > 20:
        d = d - np.median(d[outer])

    inside = r <= r_max
    rf, vf = r[inside].ravel(), d[inside].ravel()
    order = np.argsort(rf)
    cum = np.cumsum(vf[order])
    if cum[-1] <= 0:
        return float('nan')
    cum = cum / cum[-1]
    return float(np.interp(fraction, cum, rf[order])) * pixel_scale_arcsec


def calculateFWHM(psf_file_list: list, drivers: dict) -> tuple[dict, list]:
    """FWHM in arcsec per filter.

    See the warning in `get_fwhm`: do not use this alone to pick the master
    when two PSFs are similar (the two IRAC channels differ by ~4%). Use
    `calculate_half_light_radii` to break the tie.
    """
    FWHM_dict, valid_files = {}, []
    warnings.simplefilter('ignore', category=AstropyWarning)

    for file in psf_file_list:
        if file.name.startswith('.'):
            continue

        str_file = str(file)
        survey = drivers["BASE"].get_survey(file_path=str_file)
        if not survey or survey not in drivers:
            continue

        driver = drivers[survey]
        filter_name = driver.get_psf_filter_name(filename=str_file)
        psf_pixscale = driver.get_psf_pixel_scale(filter_name=filter_name)

        try:
            with fits.open(file, ignore_missing_end=True,
                           ignore_missing_simple=True) as hdu:
                data = next((h.data for h in hdu if h.data is not None), None)

                if data is not None:
                    if data.ndim == 3:
                        data = np.mean(data, axis=0)

                    fwhm_pixels = get_fwhm(data)
                    FWHM_dict[filter_name] = np.float32(fwhm_pixels * psf_pixscale)
                    valid_files.append(file.name)
                    print(f"Successfully read: {filter_name} "
                          f"(FWHM: {FWHM_dict[filter_name]:.4f} arcsec)")

        except Exception as e:
            print(f"Processing error {file.name}: {e}")

    warnings.simplefilter('default', category=AstropyWarning)
    return FWHM_dict, valid_files


def calculate_half_light_radii(psf_file_list: list, drivers: dict,
                               fraction: float = 0.8) -> dict:
    """r80 (default) in arcsec per filter - the robust criterion for the master.

    PSF matching is dominated by the WINGS, not by the core, so r80 is a
    better criterion than the Gaussian FWHM for deciding which PSF is "the
    worst".
    """
    radii = {}
    warnings.simplefilter('ignore', category=AstropyWarning)

    for file in psf_file_list:
        if file.name.startswith('.'):
            continue
        survey = drivers["BASE"].get_survey(file_path=str(file))
        if not survey or survey not in drivers:
            continue

        driver = drivers[survey]
        filt = driver.get_psf_filter_name(filename=str(file))
        pixscale = driver.get_psf_pixel_scale(filter_name=filt)

        try:
            with fits.open(file, ignore_missing_end=True,
                           ignore_missing_simple=True) as hdu:
                data = next((h.data for h in hdu if h.data is not None), None)
            if data is not None:
                radii[filt] = get_half_light_radius(data, pixscale, fraction)
        except Exception as e:
            print(f"Processing error {file.name}: {e}")

    warnings.simplefilter('default', category=AstropyWarning)
    return radii


# ======================================================================
# Preparing the PSFs for PyPHER
# ======================================================================
def _recenter_odd(data: np.ndarray) -> np.ndarray:
    """Crop/pad to an odd size with the centroid on the central pixel.

    Replaces the naive parity crop (`data[:-1, :]`), which SHIFTS the content
    by half a pixel instead of recentring it. That half pixel becomes a real
    offset of the convolved band relative to the others - a per-region colour
    error, which is the hardest kind to notice afterwards.
    """
    total = data.sum()
    if total <= 0:
        return data

    y, x = np.indices(data.shape)
    cy, cx = (data * y).sum() / total, (data * x).sum() / total

    n = min(data.shape)
    if n % 2 == 0:
        n -= 1
    half = n // 2

    # sub-pixel shift so the centroid lands on an integer index
    iy, ix = int(np.floor(cy + 0.5)), int(np.floor(cx + 0.5))
    dy, dx = iy - cy, ix - cx
    if abs(dy) > 1e-3 or abs(dx) > 1e-3:
        data = ndshift(data, (dy, dx), order=3, mode="constant", cval=0.0)
        data[data < 0] = 0.0

    # crop symmetrically around (iy, ix), zero-padding where it runs out
    out = np.zeros((n, n), dtype=data.dtype)
    y0s, y1s = iy - half, iy + half + 1
    x0s, x1s = ix - half, ix + half + 1
    ys0, ys1 = max(0, y0s), min(data.shape[0], y1s)
    xs0, xs1 = max(0, x0s), min(data.shape[1], x1s)
    out[ys0 - y0s:ys1 - y0s, xs0 - x0s:xs1 - x0s] = data[ys0:ys1, xs0:xs1]
    return out


def _resize_centered(data: np.ndarray, size: int) -> np.ndarray:
    """Take a centred odd-sized array to another odd size, keeping the centre."""
    if size % 2 == 0:
        size -= 1
    n = data.shape[0]
    if n == size:
        return data

    out = np.zeros((size, size), dtype=data.dtype)
    h_in, h_out = n // 2, size // 2
    k = min(h_in, h_out)
    out[h_out - k:h_out + k + 1, h_out - k:h_out + k + 1] = \
        data[h_in - k:h_in + k + 1, h_in - k:h_in + k + 1]
    return out


def clean_psf(input_file: str, output_file: str,
              psf_pixel_scale_arcsec: float,
              target_pixel_scale_arcsec: float,
              max_extent_arcsec: float = 0.0,
              output_size: int | None = None):
    """Resample a PSF onto the grid where the kernel will be APPLIED.

    PyPHER requires both PSFs on the same grid and returns the kernel on that
    grid. Since the kernel is applied to the science image (possibly already
    binned, see `bin_for_convolution`), the common grid must be the
    convolution grid - not any detector's native grid.

    Args:
        input_file: original PSF/PRF, typically oversampled.
        output_file: output path.
        psf_pixel_scale_arcsec: scale of the INPUT PSF, i.e. the detector's
            native scale divided by the PRF oversampling factor.
        target_pixel_scale_arcsec: scale of the convolution grid.
        max_extent_arcsec: maximum total extent of the resampled PSF, to cap
            the kernel size. **Default 0 = no truncation**, which is what you
            want when convolving on a binned grid: the kernel is cheap and you
            keep the wings. Truncating at 16" discards ~5% of the IRAC PRF
            flux, and after renormalisation that leaves the kernel too
            concentrated, under-blurring the image.
        output_size: if given, resize the final array to this odd size,
            zero-padding or cropping symmetrically. Used to hand PyPHER two
            arrays of the same shape.
    """
    with fits.open(input_file, ignore_missing_end=True,
                   ignore_missing_simple=True) as hdu:
        data = next((h.data for h in hdu if h.data is not None), None)

    if data is None:
        print(f"==> Error: no valid data found in {input_file}")
        return

    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 3:
        data = np.nanmean(data, axis=0)
    data = np.nan_to_num(data, nan=0.0)

    # --- Resample onto the target grid -----------------------------------
    # ratio = how many input PSF pixels fit into one target pixel
    ratio = target_pixel_scale_arcsec / psf_pixel_scale_arcsec

    if ratio >= 2.0:
        # Large downscale: the bulk goes through AREA AVERAGING
        # (block_reduce), the exact integral over the target pixel, which does
        # not alias. Only the non-integer remainder is interpolated.
        factor = int(round(ratio))
        data = block_reduce(data, block_size=factor, func=np.sum)
        residual = ratio / factor
        if abs(residual - 1.0) > 0.005:
            data = zoom(data, 1.0 / residual, order=3, mode="constant", cval=0.0)
        print(f"\t\tblock_reduce {factor}x + zoom {1.0/residual:.4f}x  "
              f"({psf_pixel_scale_arcsec:.4f} -> {target_pixel_scale_arcsec:.4f})")
    elif abs(ratio - 1.0) > 0.01:
        # Upscale (or small downscale): cubic interpolation. Safe on upscale
        # because the input PRF is already oversampled.
        data = zoom(data, 1.0 / ratio, order=3, mode="constant", cval=0.0)
        print(f"\t\tzoom {1.0/ratio:.4f}x  "
              f"({psf_pixel_scale_arcsec:.4f} -> {target_pixel_scale_arcsec:.4f})")

    # cubic interpolation can produce spurious negatives
    data[data < 0] = 0.0

    # Note: `zoom` preserves VALUES rather than sums, so the total changes by
    # the area factor. The normalisation below fixes that, and the SHAPE comes
    # out correct either way.

    # --- Optional truncation ---------------------------------------------
    if max_extent_arcsec and max_extent_arcsec > 0:
        n_max = int(max_extent_arcsec / target_pixel_scale_arcsec)
        if data.shape[0] > n_max:
            cy, cx = np.unravel_index(np.argmax(data), data.shape)
            half = n_max // 2
            y0, y1 = max(0, cy - half), min(data.shape[0], cy + half + 1)
            x0, x1 = max(0, cx - half), min(data.shape[1], cx + half + 1)
            kept = data[y0:y1, x0:x1].sum() / data.sum()
            data = data[y0:y1, x0:x1]
            print(f"\t\ttruncated to {data.shape[0]}px "
                  f"({max_extent_arcsec:.1f} arcsec, {kept:.2%} of flux kept)")
            if kept < 0.99:
                print(f"\t\t[!] more than 1% of the PSF flux was discarded; "
                      f"consider max_extent_arcsec=0")

    # --- Centring + odd parity -------------------------------------------
    data = _recenter_odd(data)

    # --- Normalisation ----------------------------------------------------
    total = data.sum()
    if total <= 0:
        print(f"==> Error: PSF sums to {total} in {input_file}")
        return
    data = data / total

    # --- Common array size (optional) -------------------------------------
    if output_size is not None:
        data = _resize_centered(data, output_size)
        s = data.sum()
        if s > 0:
            data = data / s
        print(f"\t\tresized to {data.shape[0]}px (same grid and shape as target)")

    # --- Write -------------------------------------------------------------
    pixel_scale_deg = target_pixel_scale_arcsec / 3600.0
    new_hdu = fits.PrimaryHDU(data)
    new_hdu.header.update({
        'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
        'CRVAL1': 0.0, 'CRVAL2': 0.0,
        'CRPIX1': (data.shape[1] // 2) + 1,
        'CRPIX2': (data.shape[0] // 2) + 1,
        'CDELT1': -pixel_scale_deg, 'CDELT2': pixel_scale_deg,
        'PIXSCALE': target_pixel_scale_arcsec,
    })
    new_hdu.writeto(output_file, overwrite=True)
    print(f"\tPSF ready for PyPHER: {os.path.basename(output_file)} "
          f"({data.shape[0]}px @ {target_pixel_scale_arcsec:.4f} arcsec/px)")


def required_blur(width_source: float, width_target: float) -> float:
    """Kernel width needed to take `source` to `target`, added in quadrature."""
    return float(np.sqrt(max(width_target ** 2 - width_source ** 2, 0.0)))


def choose_bin_factor(native_scale_arcsec: float, fwhm_master_arcsec: float,
                      min_required_blur_arcsec: float,
                      px_per_fwhm: float = 3.0, px_per_blur: float = 3.0,
                      max_factor: int = 20) -> int:
    """Largest binning factor that still samples the master AND resolves the
    smallest blur any pair needs.

    The binning factor is a property of the TARGET, not of the source survey:
    it depends on which filter ends up being the master. Hard-coding it per
    survey silently breaks whenever the survey combination changes - a
    PHANGS-only cube with bin=5 samples its own master at 0.4 px/FWHM.

    Two independent constraints, the tighter one wins:
      1. the grid must sample the master PSF, otherwise the target resolution
         does not exist on the grid;
      2. the grid must resolve the smallest required blur, otherwise no pair
         is matchable.

    The defaults of 3.0 leave headroom over the hard Nyquist limit of 2.0.
    """
    grid_limit = min(fwhm_master_arcsec / px_per_fwhm,
                     min_required_blur_arcsec / px_per_blur)
    factor = int(np.floor(grid_limit / native_scale_arcsec))
    return max(1, min(factor, max_factor))


def report_convolution_grid(psf_widths: dict, master: str,
                            grid_scale_arcsec: float,
                            fwhm_master_arcsec: float | None = None,
                            min_kernel_px: float = 2.0,
                            min_px_per_fwhm: float = 2.0) -> dict:
    """Print the consequences of the chosen convolution grid, pair by pair.

    Reports; it does not decide. Returns a summary dict so the caller
    (typically the preflight) can turn it into errors or warnings.
    """
    print(f"      grid: {grid_scale_arcsec:.4f} arcsec/px")

    master_sampling = None
    if fwhm_master_arcsec:
        master_sampling = fwhm_master_arcsec / grid_scale_arcsec
        flag = "ok" if master_sampling >= min_px_per_fwhm else "<<< UNDERSAMPLED"
        print(f"      master '{master}' sampled at {master_sampling:.2f} "
              f"px/FWHM  {flag}")

    pairs, n_ok = [], 0
    for filt, width in sorted(psf_widths.items(), key=lambda kv: kv[1]):
        if filt == master:
            continue
        ok, req = psf_matching_resolvable(width, psf_widths[master],
                                          grid_scale_arcsec, min_kernel_px)
        n_ok += ok
        pairs.append((filt, req, req / grid_scale_arcsec, ok))
        print(f"        {filt:10s} -> {master:10s}  blur {req:7.4f} arcsec "
              f"= {req / grid_scale_arcsec:6.2f} px   "
              f"{'match' if ok else 'skip'}")

    print(f"      => {n_ok}/{len(pairs)} pairs matchable")
    return {
        'grid': grid_scale_arcsec,
        'master_sampling': master_sampling,
        'n_matchable': n_ok,
        'n_pairs': len(pairs),
        'pairs': pairs,
        'max_residual': max((p[1] for p in pairs if not p[3]), default=0.0),
    }


def psf_matching_resolvable(width_source: float, width_target: float,
                            grid_scale_arcsec: float,
                            min_px: float = 2.0) -> tuple[bool, float]:
    """Does the blur needed to match two PSFs fit on the grid?

    Kernel width adds in quadrature: `sqrt(target^2 - source^2)`. Below ~2
    pixels of the convolution grid the kernel is a near-delta, and what comes
    out of the Fourier division is ringing (negative sinc side lobes), not PSF
    matching. In that regime the "correction" introduces artefacts far larger
    than the effect it would correct.

    Args:
        width_source: width of the source PSF (r80, FWHM - any measure, as
            long as it is the same one for both).
        width_target: width of the master PSF, in the same unit.
        grid_scale_arcsec: scale of the convolution grid.
        min_px: minimum kernel width, in pixels, for matching to be valid.

    Returns:
        (resolvable, required_width_in_arcsec)
    """
    required = required_blur(width_source, width_target)
    return (required / grid_scale_arcsec) >= min_px, required


def pypher_kernel_creation(cleaned_psf_by_filter: dict, psf_master_name: str,
                           output_dir: Path, grid_scale_arcsec: float,
                           clear_dir: bool = True,
                           psf_widths: dict | None = None,
                           min_kernel_px: float = 2.0) -> tuple[list, list]:
    """Build the PyPHER commands for one specific GRID.

    Every PSF in `cleaned_psf_by_filter` must sit on `grid_scale_arcsec`, the
    convolution grid of the source survey. The resulting kernel comes out on
    that same grid, which is the whole point of the fix.

    Use `clear_dir=False` when calling this in a per-survey loop, and clear
    the kernel directory once before the loop; otherwise the second survey
    deletes the first survey's kernels.

    If `psf_widths` is given (dict of {filter: width in arcsec}, typically the
    r80 from `calculate_half_light_radii`), pairs whose required blur does not
    fit on the grid are SKIPPED - see `psf_matching_resolvable`. Those bands
    must be written out with `copy_as_convolved`.

    Returns:
        (pypher_commands, skipped), where `skipped` is a list of
        (filter, residual_mismatch_arcsec).
    """
    if psf_master_name not in cleaned_psf_by_filter:
        raise KeyError(f"Master PSF '{psf_master_name}' missing on this grid.")

    # local copy: never mutate the caller's dict
    cleaned_psf_by_filter = dict(cleaned_psf_by_filter)
    psf_master_path = cleaned_psf_by_filter[psf_master_name]

    # --- drop pairs whose required blur does not fit on the grid ----------
    skipped = []
    if psf_widths is not None and psf_master_name in psf_widths:
        w_master = psf_widths[psf_master_name]
        for filt in list(cleaned_psf_by_filter):
            if filt == psf_master_name or filt not in psf_widths:
                continue
            ok, required = psf_matching_resolvable(
                psf_widths[filt], w_master, grid_scale_arcsec, min_kernel_px)
            if not ok:
                print(f"    [!] {filt} -> {psf_master_name}: required blur "
                      f"{required:.3f} arcsec = "
                      f"{required/grid_scale_arcsec:.2f} px "
                      f"(minimum {min_kernel_px:.1f} px).")
                print(f"        The kernel would be a near-delta and PyPHER "
                      f"would return ringing. Skipping the match;")
                print(f"        residual mismatch of {required:.3f} arcsec "
                      f"goes to the header as PSFRESID.")
                cleaned_psf_by_filter.pop(filt)
                skipped.append((filt, required))

    if clear_dir and output_dir.exists():
        print(f">>> Removing previous directory: {output_dir.name}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- sanity: same grid AND same array shape? --------------------------
    master_shape = fits.getdata(psf_master_path).shape
    for filt, path in cleaned_psf_by_filter.items():
        hdr = fits.getheader(path)
        px = hdr.get('PIXSCALE')
        if px is None or abs(px - grid_scale_arcsec) > 1e-6:
            raise ValueError(
                f"PSF '{filt}' is on {px} arcsec/px but the requested grid is "
                f"{grid_scale_arcsec}. The kernel would come out on the wrong "
                f"grid.")
        shape = fits.getdata(path).shape
        if shape != master_shape:
            print(f"==> Warning: PSF '{filt}' has shape {shape} while the "
                  f"master has {master_shape}. PyPHER will pad on its own; "
                  f"prefer passing output_size to clean_psf.")

    pypher_commands = []
    for filt, psf_path in cleaned_psf_by_filter.items():
        if filt == psf_master_name:
            continue
        kernel_name = output_dir / f"kernel_{filt}_to_{psf_master_name}.fits"
        pypher_commands.append(f"pypher {psf_path} {psf_master_path} {kernel_name}")

    print(f">>> {len(pypher_commands)} kernels on grid "
          f"{grid_scale_arcsec:.4f} arcsec/px ({master_shape[0]}px arrays)")
    return pypher_commands, skipped


def convolved_dict(img_files: list, kernel_files: list, drivers: dict) -> dict:
    """Pair each science image with its kernel, indexed by filter.

    Returns
    -------
    dict of {str : dict}
        ``{filter: {'img': Path, 'kernel': Path, 'survey': str}}``. The master
        filter naturally falls out (it has no kernel) - the caller must write
        that image out unconvolved.
    """
    conv_dict = {}
    for img_path in img_files:
        survey = drivers["BASE"].get_survey(file_path=img_path)
        filt = drivers[survey].get_sci_filter_name(filename=str(img_path))
        conv_dict.setdefault(filt, {})['img'] = img_path
        conv_dict.setdefault(filt, {})['survey'] = survey

    for kernel_path in kernel_files:
        filt = kernel_path.stem.split('_')[1]   # kernel_{filt}_to_{master}.fits
        conv_dict.setdefault(filt, {})['kernel'] = kernel_path

    complete, incomplete = {}, {}
    for filt, paths in conv_dict.items():
        if 'img' in paths and 'kernel' in paths:
            complete[filt] = paths
        else:
            incomplete[filt] = paths

    if incomplete:
        print(f"==> Note: {list(incomplete.keys())} have no kernel pair "
              f"(expected for the master filter and for any band skipped as "
              f"unresolvable).")

    return complete


# ======================================================================
# Binning before convolution
# ======================================================================
def bin_for_convolution(img_data: np.ndarray, header: fits.Header,
                        factor: int, native_pixel_area_arcsec2: float
                        ) -> tuple[np.ndarray, fits.Header]:
    """Bin the image by `factor` before convolving, with a NaN-aware mean.

    Rationale: the final product is resampled onto the master grid and the
    image will be blurred to the master resolution anyway. Convolving on HST's
    native grid carries ~25x more pixels than the physics requires and
    exhausts memory. Binning 5x leaves 8.6 px per FWHM, well above the Nyquist
    minimum of 2.

    Uses the MEAN (not the sum) deliberately: this keeps each pixel value
    meaning "electrons/s per NATIVE pixel", which is exactly what step 1 of
    `convert2Jansky` expects. With a sum, the native area would change and the
    unit conversion would break.

    Binning by mean is the EXACT area average, unlike interpolation. There is
    no aliasing and no loss of information.
    """
    if factor is None or factor <= 1:
        return img_data, header

    ny, nx = img_data.shape
    ny_c, nx_c = (ny // factor) * factor, (nx // factor) * factor
    data = img_data[:ny_c, :nx_c]

    # NaN-aware mean: reshape into blocks and nanmean over the inner axes.
    # For a C-ordered (ny, nx) array, reshaping to (ny/f, f, nx/f, f) puts the
    # row within the block on axis 1 and the column on axis 3.
    blocks = data.reshape(ny_c // factor, factor, nx_c // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN blocks
        binned = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)

    new_header = header.copy()

    # --- WCS: the scale multiplies, CRPIX is repositioned ------------------
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2"):
        if key in new_header:
            new_header[key] = float(new_header[key]) * factor

    for key in ("CRPIX1", "CRPIX2"):
        if key in new_header:
            new_header[key] = (float(new_header[key]) - 0.5) / factor + 0.5

    new_header["NAXIS1"], new_header["NAXIS2"] = binned.shape[1], binned.shape[0]

    # --- SIP coefficients are in pixel units; they do not survive binning --
    for key in list(new_header):
        if key.startswith(("A_", "B_", "AP_", "BP_")):
            del new_header[key]
    for key in ("CTYPE1", "CTYPE2"):
        if key in new_header:
            new_header[key] = str(new_header[key]).replace("-SIP", "")

    # --- the NATIVE area does not change: values stay "per native pixel" ---
    # Without this, convert2Jansky would read the (binned) WCS area as if it
    # were the native one, recreating the unit bug with a factor**2 error.
    new_header["NATPXAR"] = (native_pixel_area_arcsec2,
                             "native pixel area (arcsec2), pre-binning")
    new_header["BINFACT"] = (factor, "binning factor applied before convolution")

    print(f"\t\tbinned {factor}x for convolution: "
          f"{img_data.shape} -> {binned.shape}")
    return binned, new_header


# ======================================================================
# Convolution
# ======================================================================
def diagnose_negatives(convolved_img, invalid_mask, filt, survey):
    """Diagnose the origin of negative pixels after convolution.

    `invalid_mask` is the already-binned input invalid mask (NaN), passed in
    from outside: some surveys' `== 0` convention no longer holds at this
    point in the flow.
    """
    neg_mask = convolved_img < 0
    n_neg = int(np.sum(neg_mask))
    pct_neg = n_neg / convolved_img.size * 100

    print(f"\n--- Negative pixel diagnosis: {filt} ({survey}) ---")
    print(f"Total negative pixels: {n_neg} ({pct_neg:.2f}%)")
    if n_neg == 0:
        print(50 * '-')
        return

    vmin, vmax = np.nanmin(convolved_img), np.nanmax(convolved_img)
    print(f"Min value:  {vmin:.6e}")
    print(f"Max value:  {vmax:.6e}")
    if vmax > 0:
        print(f"Ratio min/max: {abs(vmin)/vmax:.4%}")

    # --- Check 1: are the negatives at the border? ---
    ny, nx = convolved_img.shape
    border_width = min(50, ny // 8, nx // 8)
    border_region = np.zeros_like(neg_mask, dtype=bool)
    border_region[:border_width, :] = True
    border_region[-border_width:, :] = True
    border_region[:, :border_width] = True
    border_region[:, -border_width:] = True

    neg_in_border = int(np.sum(neg_mask & border_region))
    neg_in_interior = int(np.sum(neg_mask & ~border_region))
    print(f"\nNegatives in border region:   {neg_in_border} "
          f"({neg_in_border/max(n_neg,1)*100:.1f}%)")
    print(f"Negatives in interior region: {neg_in_interior} "
          f"({neg_in_interior/max(n_neg,1)*100:.1f}%)")

    # --- Check 2: do they coincide with invalid input? ---
    neg_at_invalid = int(np.sum(neg_mask & invalid_mask))
    neg_at_valid = int(np.sum(neg_mask & ~invalid_mask))
    print(f"\nNegatives at invalid input pixels: {neg_at_invalid} "
          f"({neg_at_invalid/max(n_neg,1)*100:.1f}%)")
    print(f"Negatives at valid input pixels:   {neg_at_valid} "
          f"({neg_at_valid/max(n_neg,1)*100:.1f}%)")

    # --- Check 3: magnitude relative to the noise ---
    valid_data = convolved_img[~invalid_mask]
    valid_data = valid_data[np.isfinite(valid_data)]
    if valid_data.size > 100:
        noise = np.nanstd(valid_data[valid_data < np.nanpercentile(valid_data, 10)])
        print(f"\nEstimated noise level: {noise:.6e}")
        print(f"Negatives beyond 3-sigma of noise: "
              f"{int(np.sum(convolved_img < -3*noise))}")
        if noise > 0:
            worst = abs(vmin) / noise
            print(f"Worst negative: {worst:.0f} sigma below zero")
            if worst > 10:
                print("[!] This is not noise, it is kernel ringing. If it is")
                print("    concentrated in the interior, the kernel likely has")
                print("    strong negative lobes (deconvolution).")
    print(50 * '-')


def inspect_kernel(kernel_norm: np.ndarray, filt: str) -> None:
    """Print metrics of the kernel actually being applied.

    A legitimate smoothing kernel is almost entirely positive. High negative
    power indicates sharpening (deconvolution), which amplifies noise and
    produces ringing near strong gradients.

    The four metrics cover the four failure modes found in this pipeline:
    size (wrong grid), sum (normalisation), negative power (wrong master or
    unresolvable pair), centroid (parity-crop shift).
    """
    abs_sum = np.abs(kernel_norm).sum()
    neg_frac = 100.0 * abs(kernel_norm[kernel_norm < 0].sum()) / abs_sum if abs_sum > 0 else np.nan

    total = kernel_norm.sum()
    y, x = np.indices(kernel_norm.shape)
    k = np.abs(kernel_norm)
    t = k.sum()
    cy, cx = (k * y).sum() / t, (k * x).sum() / t
    c0y, c0x = (kernel_norm.shape[0] - 1) / 2, (kernel_norm.shape[1] - 1) / 2

    print(f"\t\tkernel {filt}: {kernel_norm.shape[0]}px | sum {total:.4f} | "
          f"negative power {neg_frac:.2f}% | "
          f"centroid off ({cy-c0y:+.2f}, {cx-c0x:+.2f}) px")


def _prepare_image(original_fits: Path, driver, bin_factor: int | None = None
                   ) -> tuple[np.ndarray, fits.Header, np.ndarray]:
    """Read the image, mark invalid pixels as NaN, bin to the convolution grid.

    The NaN marking comes BEFORE the binning deliberately: otherwise the
    border zeros enter the mean and contaminate valid neighbouring pixels.

    Args:
        bin_factor: binning factor to apply. Pass the value DERIVED for this
            run (see `choose_bin_factor`), not the config default. The factor
            depends on which filter is the master, so a per-survey constant is
            only correct for the survey combination it was tuned for. Falls
            back to the driver's value when None, which is right only for
            standalone calls.
    """
    with fits.open(original_fits) as hdu_i:
        img_data = hdu_i[0].data.astype(np.float32)
        img_header = hdu_i[0].header.copy()

    invalid = driver.get_invalid_mask(img_data) | ~np.isfinite(img_data)
    img_data = np.where(invalid, np.nan, img_data).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        native_area = proj_plane_pixel_area(WCS(img_header, naxis=2)) * 3600 ** 2

    if bin_factor is None:
        bin_factor = driver.get_convolution_bin_factor

    img_data, img_header = bin_for_convolution(
        img_data, img_header,
        factor=bin_factor,
        native_pixel_area_arcsec2=native_area,
    )
    return img_data, img_header, ~np.isfinite(img_data)


def _output_filename(original_fits, survey, psf_master_name, master_survey,
                     output_dir, driver, is_master=False) -> tuple[Path, str, str]:
    name = (original_fits.name if hasattr(original_fits, 'name')
            else os.path.basename(original_fits))
    gal_name = driver.get_galaxy_name(name)
    filt = driver.get_sci_filter_name(name)
    output_path = output_dir / gal_name

    if is_master:
        # `discover_convolved_files` identifies the master by this suffix
        out_file = output_path / f'{gal_name}_{survey.lower()}_{filt}_master.fits'
    else:
        out_file = (output_path /
                    f'{gal_name}_{survey.lower()}_{filt}_to_'
                    f'{master_survey.lower()}_{psf_master_name}_convolved.fits')
    return out_file, gal_name, filt


def copy_as_convolved(original_fits: Path, survey: str, psf_master_name: str,
                      master_survey: str, output_dir: Path, drivers: dict,
                      psf_residual_arcsec: float = 0.0,
                      is_master: bool = False,
                      bin_factor: int | None = None,
                      force: bool = False) -> Path:
    """Write a band that is NOT PSF-matched, using the convolved convention.

    Two callers:
      - bands whose required blur does not fit on the grid (see
        `psf_matching_resolvable`);
      - the master itself, which IS the target and needs no matching.

    Either way the image goes through the same preparation as the convolved
    bands (NaN marking + binning) but is not convolved. This matters for the
    master: if its survey has a binning factor > 1 and the master were merely
    copied, it would land on a different grid from its own survey's convolved
    bands, and the alignment step would resample between grids.

    The residual mismatch is recorded in the header, so the decision travels
    with the data instead of living only in whoever ran the pipeline.
    """
    driver = drivers[survey]
    out_file, _, filt = _output_filename(
        original_fits, survey, psf_master_name, master_survey, output_dir,
        driver, is_master=is_master)

    if out_file.exists() and not force:
        # An existing file may belong to a DIFFERENT master: the filename is
        # not enough to tell, since only the target part changes. Trust the
        # header, and redo the file when it disagrees.
        try:
            existing_target = fits.getheader(out_file).get('PSFTARGT')
        except Exception:
            existing_target = "<unreadable>"
        if existing_target not in (None, psf_master_name):
            print(f">>> Stale or unreadable file "
                  f"('{existing_target}' != '{psf_master_name}'), redoing: "
                  f"{out_file.name}")
        else:
            print(f">>> Already present, skipping: {out_file.name}")
            return out_file

    out_file.parent.mkdir(parents=True, exist_ok=True)

    role = "master" if is_master else "unmatched band"
    print(200 * '-' + f'\n>>> Writing {filt} from {survey} ({role}, no PSF matching):')
    img_data, img_header, _ = _prepare_image(original_fits, driver, bin_factor)

    # PSFTARGT is written on ALL three routes (matched, unmatched, master) so
    # the discovery functions can select on the header instead of the
    # filename. Writing it only for the master would leave the most unusual
    # band - the unmatched one - without provenance.
    img_header['PSFMATCH'] = (False, 'PSF matching applied?')
    img_header['PSFTARGT'] = (psf_master_name,
                              'this band IS the target' if is_master
                              else 'intended target; not matched')
    img_header['PSFRESID'] = (round(float(psf_residual_arcsec), 4),
                              'residual PSF mismatch, arcsec')

    fits.PrimaryHDU(data=img_data, header=img_header).writeto(out_file, overwrite=True)
    print(f'\tPSFRESID = {psf_residual_arcsec:.3f} arcsec')
    print(f'\tFITS saved to: {out_file}\n' + 100 * '-')
    return out_file


def create_convolvedFITS(original_fits: Path, kernel_fits: Path,
                         survey: str, psf_master_name: str, master_survey: str,
                         output_dir: Path, drivers: dict,
                         bin_factor: int | None = None,
                         force: bool = False) -> Path:
    """Convolve a science image with its PSF-matching kernel.

    Flow: mark invalid as NaN -> bin to the convolution grid -> verify the
    kernel grid against the image -> convolve -> write with provenance.
    """
    driver = drivers[survey]
    out_file, gal_name, filt = _output_filename(
        original_fits, survey, psf_master_name, master_survey, output_dir, driver)

    if out_file.exists() and not force:
        # An existing file may belong to a DIFFERENT master: the filename is
        # not enough to tell, since only the target part changes. Trust the
        # header, and redo the file when it disagrees.
        try:
            existing_target = fits.getheader(out_file).get('PSFTARGT')
        except Exception:
            existing_target = "<unreadable>"
        if existing_target not in (None, psf_master_name):
            print(f">>> Stale or unreadable file "
                  f"('{existing_target}' != '{psf_master_name}'), redoing: "
                  f"{out_file.name}")
        else:
            print(f">>> Already convolved, skipping: {out_file.name}")
            return out_file

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(kernel_fits) as hdu_k:
        kernel_data = np.nan_to_num(hdu_k[0].data).astype(np.float64)

    print(200 * '-' + f'\n>>> Convolving {filt} filter from {survey} survey:')

    img_data, img_header, invalid_binned = _prepare_image(
        original_fits, driver, bin_factor)

    # --- prepare the kernel -----------------------------------------------
    ksum = np.sum(kernel_data)
    if ksum == 0:
        raise ValueError(f"Kernel {kernel_fits} has zero sum - invalid kernel!")

    # Dividing by a negative sum flips the sign and normalises to +1, which is
    # the correct behaviour: PyPHER sometimes returns the kernel with the
    # global sign inverted, and that is cosmetic.
    kernel_norm = (kernel_data / ksum).astype(np.float32)
    kernel_size = kernel_norm.shape[0]
    inspect_kernel(kernel_norm, filt)

    # --- grid sanity: was the kernel built on the grid it is applied to? ---
    k_px = fits.getheader(kernel_fits).get('PIXSCALE')
    if k_px is None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                k_px = np.sqrt(proj_plane_pixel_area(
                    WCS(fits.getheader(kernel_fits), naxis=2)) * 3600 ** 2)
            except Exception:
                k_px = None
    if k_px is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            img_px = np.sqrt(proj_plane_pixel_area(WCS(img_header, naxis=2)) * 3600 ** 2)
        if abs(k_px - img_px) / img_px > 0.02:
            raise ValueError(
                f"Kernel for '{filt}' is on {k_px:.4f} arcsec/px but the "
                f"convolution image is on {img_px:.4f} arcsec/px. The blur "
                f"would be wrong by a factor of ~{img_px/k_px:.1f}. "
                f"Regenerate the kernels with --create_kernel.")

    # --- convolve -----------------------------------------------------------
    convolved_img = driver.convolve(img_data, kernel_norm, kernel_size)

    n_neg_before = int(np.sum(np.nan_to_num(img_data, nan=0.0) < 0))
    n_neg_after = int(np.sum(np.nan_to_num(convolved_img, nan=0.0) < 0))
    if n_neg_after > max(n_neg_before * 1.5, 10):
        print(f">>> WARNING: convolution increased negative pixels in {filt}!")
        diagnose_negatives(convolved_img, invalid_binned, filt, survey)

    img_header['PSFMATCH'] = (True, 'PSF matching applied?')
    img_header['PSFTARGT'] = (psf_master_name, 'PSF matched to this filter')
    img_header['PSFRESID'] = (0.0, 'residual PSF mismatch, arcsec')
    convolved_fits = fits.PrimaryHDU(data=convolved_img, header=img_header)
    convolved_fits.writeto(out_file, overwrite=True)
    print(f'\tConvolved FITS saved to: {out_file}\n' + 100 * '-')

    return out_file
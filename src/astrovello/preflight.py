"""
Preflight validation for the AsTrovello pipeline.

Validates the whole configuration BEFORE any processing happens, and fails
loudly on anything that would silently produce an incomplete or incoherent
cube. Warns (without failing) on things that are legitimate scientific
choices.

Why this exists
---------------
Most of the defects found in this pipeline share one shape: an implicit
assumption that stopped being true at some point in the flow. The grid a
kernel was built on stopped being true when it was applied to another image.
The `== 0` invalid sentinel stopped being true after interpolation. The
binning factor tuned for an IRAC master stops being true the moment the
survey combination changes.

The antidote is to check the assumption at the point of use. This module does
that for the configuration as a whole, once, cheaply, before an hour of
compute is spent.

Usage
-----
    from preflight import preflight

    ok, plan = preflight(img_files, psf_files, DRIVERS, input_survey_list,
                         r80, fwhm_dict, bin_factors, PIVOT_WAVELENGTHS)
    if not ok:
        raise SystemExit("Preflight failed. Fix the ERRORs above.")
"""

import warnings

import numpy as np
from astropy.io import fits

from convolution_2_0 import report_convolution_grid, required_blur
from drivers import BASE_Driver
from utils_2_0 import science_pixel_scale

SEP = "=" * 70

# Methods every survey driver must implement (not merely inherit as a stub)
REQUIRED_DRIVER_METHODS = (
    'get_sci_filter_name',
    'get_psf_filter_name',
    'get_galaxy_name',
    'get_invalid_mask',
    'convolve',
    'convert2Jansky',
)


def _band_inventory(img_files, psf_files, drivers):
    """Map filter -> survey for science images, and the set of PSF filters."""
    science = {}
    for path in img_files:
        survey = drivers["BASE"].get_survey(file_path=path)
        if survey is None:
            continue
        science[drivers[survey].get_sci_filter_name(str(path))] = survey

    psf = set()
    for path in psf_files:
        survey = drivers["BASE"].get_survey(file_path=path)
        if survey is None:
            continue
        psf.add(drivers[survey].get_psf_filter_name(str(path)))

    return science, psf


def _check_band_coverage(science_bands, psf_bands, r80, pivot_wavelengths,
                         errors, warns):
    """Every science band needs a PSF, a pivot wavelength and a measured width."""
    print(f"\n1. Band coverage ({len(science_bands)} science images)")
    for filt, survey in sorted(science_bands.items()):
        missing = []
        if filt not in psf_bands:
            missing.append("PSF")
        if filt not in pivot_wavelengths:
            missing.append("pivot wavelength")
        if filt not in r80:
            missing.append("r80")
        status = "ok" if not missing else f"MISSING {', '.join(missing)}"
        print(f"   {filt:10s} ({survey:8s})  {status}")
        if missing:
            errors.append(f"band '{filt}' is missing: {', '.join(missing)}")

    orphans = psf_bands - set(science_bands)
    if orphans:
        warns.append(f"PSFs with no science image: {sorted(orphans)}")

    if len(science_bands) < 2:
        errors.append("fewer than two bands: nothing to match, no cube to build")


def _check_drivers(drivers, survey_list, errors):
    """A new survey's driver must actually implement the abstract methods."""
    print("\n2. Drivers")
    for survey in survey_list:
        driver = drivers[survey]
        absent = [m for m in REQUIRED_DRIVER_METHODS if not hasattr(driver, m)]
        # inherited-but-unimplemented stubs raise NotImplementedError at runtime
        stubs = [m for m in REQUIRED_DRIVER_METHODS
                 if getattr(type(driver), m, None) is getattr(BASE_Driver, m, None)]
        ok = not absent and not stubs
        print(f"   {survey:10s} {'ok' if ok else 'INCOMPLETE'}")
        if absent:
            errors.append(f"driver {survey} lacks: {absent}")
        if stubs:
            errors.append(f"driver {survey} does not implement: {stubs}")


def _check_grids(img_files, drivers, survey_list, r80, fwhm_dict, bin_factors,
                 master, min_kernel_px, min_px_per_fwhm, errors, warns):
    """Per-survey convolution grid: does it sample the master and resolve the
    pairs it will try to match?"""
    print(f"\n3. Convolution grid (master = {master})")
    plan = {}

    for survey in survey_list:
        survey_imgs = [p for p in img_files
                       if drivers["BASE"].get_survey(file_path=p) == survey]
        if not survey_imgs:
            continue

        native = science_pixel_scale(
            survey_imgs[0], hdu_ext=drivers[survey].get_hdu_sci_position)
        factor = bin_factors.get(survey, 1)
        grid = native * factor

        print(f"   {survey}: native {native:.4f} x bin {factor}")
        widths = {master: r80[master]}
        for path in survey_imgs:
            filt = drivers[survey].get_sci_filter_name(str(path))
            if filt in r80:
                widths[filt] = r80[filt]

        summary = report_convolution_grid(
            widths, master, grid,
            fwhm_master_arcsec=float(fwhm_dict.get(master, 0.0)) or None,
            min_kernel_px=min_kernel_px, min_px_per_fwhm=min_px_per_fwhm)
        plan[survey] = summary

        sampling = summary['master_sampling']
        if sampling is not None and sampling < min_px_per_fwhm \
                and summary['n_matchable'] > 0:
            errors.append(
                f"{survey}: the grid undersamples the master "
                f"({sampling:.2f} px/FWHM) yet still tries to match "
                f"{summary['n_matchable']} pairs. Incoherent: lower the "
                f"binning factor or accept no matching at all.")

        if summary['n_matchable'] == 0 and summary['n_pairs'] > 0:
            warns.append(
                f"{survey}: no pair is matchable on this grid; the bands will "
                f"enter unmatched with PSFRESID up to "
                f"{summary['max_residual']:.3f} arcsec")

    return plan


def _check_master_grid(science_bands, bin_factors, master, errors):
    """The master must end up on the same grid as its own survey's bands."""
    master_survey = science_bands.get(master)
    if master_survey and bin_factors.get(master_survey, 1) > 1:
        print(f"\n4. Master grid: '{master}' belongs to {master_survey} "
              f"(bin {bin_factors[master_survey]})")
        print("   The master must go through copy_as_convolved(is_master=True) "
              "so it is binned like its own survey's bands.")
    return master_survey


def _estimate_cube_size(img_files, drivers, science_bands, master_survey,
                        bin_factors, warns):
    """Rough size of the final cube, in pixels and GB."""
    if not master_survey:
        return None

    imgs = [p for p in img_files
            if drivers["BASE"].get_survey(file_path=p) == master_survey]
    if not imgs:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        header = fits.getheader(
            imgs[0], ext=drivers[master_survey].get_hdu_sci_position)

    n_px = int(header.get('NAXIS1', 0)) * int(header.get('NAXIS2', 0))
    n_px //= max(bin_factors.get(master_survey, 1), 1) ** 2
    size_gb = n_px * len(science_bands) * 4 / 1e9

    print(f"\n5. Estimated cube: ~{n_px:,} px x {len(science_bands)} bands "
          f"= {size_gb:.2f} GB")
    if size_gb > 1.0:
        warns.append(
            f"cube of ~{size_gb:.1f} GB; the segmentation step clusters "
            f"per pixel and may not fit in memory")
    return size_gb


def preflight(img_files, psf_files, drivers, survey_list,
              r80, fwhm_dict, bin_factors, pivot_wavelengths,
              min_kernel_px: float = 2.0,
              min_px_per_fwhm: float = 2.0) -> tuple[bool, dict]:
    """Validate the whole configuration before processing anything.

    Args:
        img_files, psf_files: as collected by the CLI.
        drivers: the driver registry, including the "BASE" entry.
        survey_list: surveys selected for this run.
        r80: {filter: r80 in arcsec}, from `calculate_half_light_radii`.
        fwhm_dict: {filter: FWHM in arcsec}, from `calculateFWHM`.
        bin_factors: {survey: binning factor} for this run.
        pivot_wavelengths: PIVOT_WAVELENGTHS from config.
        min_kernel_px: minimum kernel width in pixels for a pair to be matched.
        min_px_per_fwhm: minimum sampling of the master PSF.

    Returns:
        (ok, plan). `ok` is False if any hard error was found. `plan` holds
        the per-survey grid summaries, useful for logging.
    """
    errors, warns = [], []

    print("\n" + SEP)
    print("PREFLIGHT")
    print(SEP)

    science_bands, psf_bands = _band_inventory(img_files, psf_files, drivers)

    _check_band_coverage(science_bands, psf_bands, r80, pivot_wavelengths,
                         errors, warns)
    _check_drivers(drivers, survey_list, errors)

    plan = {}
    if r80:
        master = max(r80, key=r80.get)
        plan = _check_grids(img_files, drivers, survey_list, r80, fwhm_dict,
                            bin_factors, master, min_kernel_px,
                            min_px_per_fwhm, errors, warns)
        master_survey = _check_master_grid(science_bands, bin_factors, master,
                                           errors)
        _estimate_cube_size(img_files, drivers, science_bands, master_survey,
                            bin_factors, warns)
    else:
        errors.append("no PSF widths measured; cannot pick a master")

    print("\n" + "-" * 70)
    for w in warns:
        print(f"   WARNING: {w}")
    for e in errors:
        print(f"   ERROR:   {e}")
    if not errors and not warns:
        print("   All checks passed.")
    print("-" * 70)

    return (len(errors) == 0), plan

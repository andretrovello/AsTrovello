"""
AsTrovello 2.0 - pipeline command-line interface.

Stages: PSF matching and convolution -> alignment onto the master grid ->
unit conversion to Jy/pixel -> data cube.

Design notes worth knowing before editing this file
---------------------------------------------------
1. The convolution grid is DERIVED, not configured. The binning factor is a
   property of the target (the master filter), not of the source survey, so
   it has to be recomputed whenever the survey combination changes. See
   `choose_bin_factor`. The value in `config.py` is a fallback only.

2. The master is chosen by r80 (the radius enclosing 80% of the flux), not by
   the Gaussian FWHM. PSF matching is dominated by the wings, and a Gaussian
   fit responds to the core: it underestimates the IRAC PRF width by ~25% and
   inverts the ordering of the two channels, which differ by only ~4%.

3. Pixel scales always come from the WCS of the file being processed, never
   from `config.py`. The constant there is the PSF reference scale, which for
   S4G (1.221") is not the mosaic scale (0.75").

4. Every band takes exactly one of three routes: convolved (has a kernel),
   unmatched (blur below the grid's resolution limit), or master. The summary
   at the end checks that the three add up to the number of bands found.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from astropy.io import fits

from config import SURVEY_CONFIG, PIVOT_WAVELENGTHS
from drivers import BASE_Driver, PHANGS_Driver, S4G_Driver
from convolution_2_0 import (
    calculateFWHM,
    calculate_half_light_radii,
    choose_bin_factor,
    clean_psf,
    convolved_dict,
    copy_as_convolved,
    create_convolvedFITS,
    psf_matching_resolvable,
    pypher_kernel_creation,
    required_blur,
)
from alignment_2_0 import (
    discover_convolved_files,
    reproject_to_reference,
)
from units_2_0 import convert2Jansky
from cube_2_0 import (
    create_data_cube,
    discover_jansky_files,
)
from utils_2_0 import (
    science_pixel_scale,
    sort_filters_by_wavelength,
)
from preflight import preflight


def find_science_image(img_files, drivers, survey, filt):
    """Locate the science image for a given (survey, filter) pair."""
    return next(
        (p for p in img_files
         if drivers["BASE"].get_survey(p) == survey
         and drivers[survey].get_sci_filter_name(str(p)) == filt),
        None)


def select_master(psf_files, drivers, fwhm_dict):
    """Pick the homogenisation target: the widest PSF, measured by r80.

    Returns (master_filter, r80_dict). Prints the ranking so the choice is
    visible in the log - with two similar PSFs this decision determines
    whether a kernel is a smoothing or a sharpening operation.
    """
    r80 = calculate_half_light_radii(psf_files, drivers, fraction=0.8)
    if not r80:
        raise RuntimeError("Could not measure any PSF width; cannot pick a master.")

    master = max(r80, key=r80.get)

    print("\nEnclosed-energy radius at 80% (master selection criterion):")
    for filt in sorted(r80, key=r80.get):
        mark = "  <== master" if filt == master else ""
        gauss = fwhm_dict.get(filt)
        gauss_s = f"  (Gaussian FWHM {float(gauss):.4f})" if gauss else ""
        print(f"   {filt:8s} r80 = {r80[filt]:.4f} arcsec{gauss_s}{mark}")
    print(f"\n==> Master PSF: {master}")
    return master, r80


def derive_bin_factors(img_files, drivers, survey_list, r80, fwhm_dict,
                       master, override=None):
    """Convolution binning factor per survey, derived from the target.

    Convolving on HST's native grid carries ~25x more pixels than the physics
    requires (the product ends up on the master grid anyway) and exhausts
    memory. But the factor cannot be a per-survey constant: with an HST master
    a factor of 5 would sample the target at 0.4 px/FWHM and destroy it.

    `override` (from --bin_factor) applies to every survey and skips the
    derivation, for when the run needs a specific grid.
    """
    blurs = [required_blur(r80[b], r80[master]) for b in r80 if b != master]
    min_blur = min(blurs) if blurs else float('inf')

    factors = {}
    for survey in survey_list:
        imgs = [p for p in img_files if drivers["BASE"].get_survey(p) == survey]
        if not imgs:
            continue
        if override:
            factors[survey] = int(override)
            continue
        native = science_pixel_scale(imgs[0])
        factors[survey] = choose_bin_factor(
            native, float(fwhm_dict.get(master, 0.0)) or native * 2, min_blur)
    return factors


def build_kernels(img_files, psf_files, drivers, survey_list, input_dir,
                  kernel_dir, master, r80, bin_factors, min_kernel_px):
    """Clean the PSFs and generate the kernels, one grid per source survey.

    Returns the list of unmatched bands as (filter, survey, residual_arcsec).
    """
    print("\n>>> Cleaning PSFs and generating kernels, one grid per survey...")

    master_psf_raw = next(
        (p for p in psf_files
         if drivers[drivers["BASE"].get_survey(p)].get_psf_filter_name(str(p))
         == master), None)
    if master_psf_raw is None:
        raise FileNotFoundError(f"Master PSF '{master}' not found in psf_files.")

    master_psf_scale = drivers[
        drivers["BASE"].get_survey(master_psf_raw)
    ].get_psf_pixel_scale(filter_name=master)

    # Clear the kernel directory ONCE, before the loop. Inside the loop we use
    # clear_dir=False, otherwise the second survey wipes the first one's kernels.
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)

    unmatched = []

    for survey in survey_list:
        survey_imgs = [p for p in img_files
                       if drivers["BASE"].get_survey(p) == survey]
        if not survey_imgs:
            continue

        bin_factor = bin_factors.get(survey, 1)
        grid = science_pixel_scale(survey_imgs[0]) * bin_factor
        print(f"\n>>> Survey {survey}: convolution grid = {grid:.4f} arcsec/px "
              f"(native x bin {bin_factor})")

        clean_dir = input_dir / survey / "PSF_CLEAN"
        if clean_dir.is_dir():
            shutil.rmtree(clean_dir)
        clean_dir.mkdir(parents=True)

        # The MASTER goes first: it fixes the array size used as `output_size`
        # for every other PSF on this grid. One copy per grid, since each
        # source survey convolves on its own.
        out_master = clean_dir / f"master_{master_psf_raw.name}"
        clean_psf(
            input_file=master_psf_raw,
            output_file=out_master,
            psf_pixel_scale_arcsec=master_psf_scale,
            target_pixel_scale_arcsec=grid,
            max_extent_arcsec=0,    # whole PRF; cheap on a binned grid
        )
        if not out_master.exists():
            raise RuntimeError(f"clean_psf failed for the master in {survey}.")

        target_size = fits.getdata(out_master).shape[0]
        cleaned = {master: out_master}

        for psf_path in [p for p in psf_files
                         if drivers["BASE"].get_survey(p) == survey]:
            driver = drivers[survey]
            filt = driver.get_psf_filter_name(filename=str(psf_path))
            if filt == master:
                continue    # already handled above

            clean_psf(
                input_file=psf_path,
                output_file=clean_dir / psf_path.name,
                psf_pixel_scale_arcsec=driver.get_psf_pixel_scale(filter_name=filt),
                target_pixel_scale_arcsec=grid,
                max_extent_arcsec=0,
                output_size=target_size,
            )
            cleaned[filt] = clean_dir / psf_path.name

        if len(cleaned) < 2:
            print(f"    (no source PSF in {survey}; no kernel to generate)")
            continue

        commands, skipped = pypher_kernel_creation(
            cleaned_psf_by_filter=cleaned,
            psf_master_name=master,
            output_dir=kernel_dir,
            grid_scale_arcsec=grid,
            clear_dir=False,
            psf_widths=r80,
            min_kernel_px=min_kernel_px,
        )
        for cmd in commands:
            print(f"----- Running: {cmd} -----")
            subprocess.run(cmd, shell=True, check=True)

        unmatched.extend((filt, survey, resid) for filt, resid in skipped)

    n_kernels = len(list(kernel_dir.glob("kernel_*_to_*.fits")))
    print(f"\n>>> Kernel processing completed! ({n_kernels} kernels)")
    return unmatched


def rediscover_unmatched(img_files, drivers, survey_list, kernel_dir, master,
                         r80, bin_factors, min_kernel_px):
    """Find the unmatched bands when kernels already exist on disk.

    Without --create_kernel we never call `pypher_kernel_creation`, so the
    skipped bands have to be recovered by comparing the kernels on disk with
    the filters available. Otherwise they would silently vanish from the cube.
    """
    print(">>> Matching kernels already exist. Proceeding to image convolution...")

    have_kernel = {k.stem.split('_')[1]
                   for k in kernel_dir.glob(f"kernel_*_to_{master}.fits")}
    unmatched = []

    for survey in survey_list:
        survey_imgs = [p for p in img_files
                       if drivers["BASE"].get_survey(p) == survey]
        if not survey_imgs:
            continue

        grid = science_pixel_scale(survey_imgs[0]) * bin_factors.get(survey, 1)

        for img in survey_imgs:
            filt = drivers[survey].get_sci_filter_name(str(img))
            if filt == master or filt in have_kernel:
                continue
            if filt not in r80 or master not in r80:
                print(f"==> Warning: '{filt}' has no kernel and no r80; it will "
                      f"be written unmatched with PSFRESID = 0.")
                unmatched.append((filt, survey, 0.0))
                continue
            _, resid = psf_matching_resolvable(r80[filt], r80[master], grid,
                                               min_kernel_px)
            print(f"    '{filt}' has no kernel: treating as unmatched "
                  f"(residual {resid:.3f} arcsec).")
            unmatched.append((filt, survey, resid))

    return unmatched


def print_matching_summary(master, master_survey, r80, conv_pairs, unmatched,
                           n_bands):
    """Confirm every band took one of the three routes."""
    print("\n" + "=" * 70)
    print("PSF MATCHING SUMMARY")
    print("=" * 70)
    print(f"   master: {master} ({master_survey}), r80 = {r80[master]:.4f} arcsec")
    print(f"   matched   ({len(conv_pairs)}): {', '.join(sorted(conv_pairs))}")
    if unmatched:
        detail = ', '.join(f"{f} ({r:.3f} arcsec)" for f, _, r in unmatched)
        print(f"   unmatched ({len(unmatched)}): {detail}")
        print("      -> required blur below the grid resolution limit; matching")
        print("         would produce ringing. Residual stored in PSFRESID.")
    total = len(conv_pairs) + len(unmatched) + 1
    print(f"   total: {total} of {n_bands} bands")
    if total != n_bands:
        print("   [!] WARNING: not every band was accounted for.")
    print("=" * 70 + "\n" + 100 * '#')


def main():
    parser = argparse.ArgumentParser('AsTrovello Pipeline Control')
    parser.add_argument('--mode', type=str,
                        choices=['full', 'alignment_only', 'conv_only', 'cube_only'],
                        default='full', help='Execution mode')
    parser.add_argument('--galaxy', type=str, required=True,
                        help='Galaxy name (e.g., ngc1566)')
    parser.add_argument('--create_kernel', action='store_true',
                        help='Trigger PSF cleaning and PyPHER kernel generation')
    parser.add_argument('--apply_mask', action='store_true',
                        help='Generate a signal-based sky mask for the final cube')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Sigma threshold for the sky mask cut')
    parser.add_argument('--error', action='store_true', help='Create error cube')
    parser.add_argument('--valid_pixels_cut', action='store_true',
                        help='Cut the image to a central radius of valid pixels')
    parser.add_argument('--force_convolution', action='store_true',
                        help='Force convolution even if convolved files exist')
    parser.add_argument('--bin_factor', type=int, default=None,
                        help='Override the derived convolution binning factor '
                             '(applies to every survey)')
    parser.add_argument('--min_kernel_px', type=float, default=2.0,
                        help='Minimum kernel width in pixels for a PSF pair to '
                             'be matched; below this the kernel is a near-delta '
                             'and PyPHER returns ringing')
    parser.add_argument('--preflight_only', action='store_true',
                        help='Validate the configuration and exit without '
                             'processing anything')
    parser.add_argument('--skip_preflight', action='store_true',
                        help='Skip validation (not recommended)')
    parser.add_argument('--allow_mixed', action='store_true',
                        help='Tolerate leftover files from other '
                             'configurations in the output directories. They '
                             'are ignored either way; without this flag their '
                             'presence is an error, because it usually means '
                             'the directory holds mixed results.')
    parser.add_argument('--quiet', action='store_true', help='Suppress alignment logs')

    args = parser.parse_args()
    galaxy = args.galaxy

    print(100 * '#')
    print(f'Executing AsTrovello for {args.galaxy}...\n')

    # ------------------------------ Directories ------------------------------
    CWD = Path.cwd()
    BASE_DIR = CWD.parents[1]
    input_dir = BASE_DIR / 'Input'
    output_dir = BASE_DIR / "Output"

    kernel_dir = output_dir / 'PSF_Kernels'
    convolved_fits_dir = output_dir / 'convolved_fits'
    reprojected_dir = output_dir / 'reprojected_files'

    print(f'Root Directory: {BASE_DIR}')

    if not input_dir.exists():
        print(f"==> Error: 'Input' folder not found in {BASE_DIR}")
        print("Make sure you are in the correct directory.")
        return

    # ------------------------------ Survey selection ------------------------
    SURVEYS = [x.name for x in input_dir.iterdir() if x.is_dir()]
    print(f">>> Found Surveys: {SURVEYS}")

    input_survey_list = None
    while input_survey_list is None:
        choice = str(input("\t1. Build datacube for all (Y/n)? ")).upper().strip()

        if choice == "Y":
            print(">>> Proceeding with all surveys...")
            input_survey_list = SURVEYS

        elif choice == "N":
            while input_survey_list is None:
                selection = str(input("\tSelect desired cubes (PHANGS, S4G, JPAS,...): "))
                candidate = [x.strip().upper() for x in selection.split(",")]
                available = set(SURVEYS).intersection(set(SURVEY_CONFIG.keys()))
                if set(candidate).issubset(available):
                    print(f">>> Proceeding with selected surveys: {candidate}")
                    input_survey_list = candidate
                else:
                    print(f">>> Error: choose only from available and configured "
                          f"surveys: {sorted(available)}")
        else:
            print("\tProvide Y/n answer.")

    DRIVERS = {
        "BASE": BASE_Driver(config_dict=SURVEY_CONFIG),
        "PHANGS": PHANGS_Driver(config_dict=SURVEY_CONFIG["PHANGS"]),
        "S4G": S4G_Driver(config_dict=SURVEY_CONFIG["S4G"]),
    }

    # ------------------------------ Gather files -----------------------------
    img_files, psf_files = [], []
    for survey in input_survey_list:
        img_dir = input_dir / survey / "galaxies" / galaxy
        psf_dir = input_dir / survey / "PSF"
        img_files.extend(DRIVERS[survey].get_files(dir_path=img_dir, mode="sci"))
        psf_files.extend(DRIVERS[survey].get_files(dir_path=psf_dir, mode="psf"))

    if not img_files:
        raise FileNotFoundError(f"No science images found for '{galaxy}'.")

    # ------------------ Resolutions, master, convolution grid ----------------
    print(">>> Calculating survey resolutions...")
    fwhm_dict, valid_files = calculateFWHM(psf_file_list=psf_files, drivers=DRIVERS)
    print(valid_files)
    df_fwhm = (pd.DataFrame(list(fwhm_dict.items()),
                            columns=["Filter", "FWHM_arcsec"])
               .sort_values(by="FWHM_arcsec")
               .reset_index(drop=True))
    print("\nResolutions Table:\n", df_fwhm)

    psf_master_name, r80 = select_master(psf_files, DRIVERS, fwhm_dict)

    bin_factors = derive_bin_factors(img_files, DRIVERS, input_survey_list,
                                     r80, fwhm_dict, psf_master_name,
                                     override=args.bin_factor)
    source = "overridden by --bin_factor" if args.bin_factor else "derived from the master"
    print(f"\n>>> Convolution binning factors ({source}): {bin_factors}")

    # Filter -> science image map, reused by the alignment stage
    img_by_filter = {}
    for img_path in img_files:
        survey_i = DRIVERS["BASE"].get_survey(file_path=img_path)
        filt_i = DRIVERS[survey_i].get_sci_filter_name(filename=str(img_path))
        img_by_filter[filt_i] = {'path': img_path, 'survey': survey_i}

    if psf_master_name not in img_by_filter:
        raise FileNotFoundError(
            f"No science image for the master filter '{psf_master_name}'.")

    master_survey = img_by_filter[psf_master_name]['survey']
    master_img_path = img_by_filter[psf_master_name]['path']

    # ------------------------------ Preflight --------------------------------
    if not args.skip_preflight:
        ok, _ = preflight(img_files, psf_files, DRIVERS, input_survey_list,
                          r80, fwhm_dict, bin_factors, PIVOT_WAVELENGTHS,
                          min_kernel_px=args.min_kernel_px)
        if not ok:
            raise SystemExit("\nPreflight failed. Fix the ERRORs above, or "
                             "re-run with --skip_preflight to override.")
    if args.preflight_only:
        print("\n--preflight_only: nothing was processed.")
        return

    # =====================================================================
    # ========================== CONVOLUTION ==============================
    # =====================================================================
    if args.mode in ('full', 'conv_only'):
        print(">>> Initiating convolution process...")

        if args.create_kernel:
            unmatched = build_kernels(
                img_files, psf_files, DRIVERS, input_survey_list, input_dir,
                kernel_dir, psf_master_name, r80, bin_factors,
                args.min_kernel_px)
        else:
            unmatched = rediscover_unmatched(
                img_files, DRIVERS, input_survey_list, kernel_dir,
                psf_master_name, r80, bin_factors, args.min_kernel_px)

        convolved_fits_dir_gal = convolved_fits_dir / galaxy
        if convolved_fits_dir_gal.exists():
            print('\n\tConvolution directory already exists!')
        else:
            print('\n\tCreating convolution directory...')
            convolved_fits_dir_gal.mkdir(parents=True, exist_ok=True)

        # New kernels invalidate every convolved image on disk.
        force_conv = args.force_convolution or args.create_kernel

        # --- bands WITH a kernel: convolve --------------------------------
        kernel_files = sorted(kernel_dir.glob("kernel_*_to_*.fits"))
        conv_pairs = convolved_dict(img_files, kernel_files, DRIVERS)

        for filt, paths in conv_pairs.items():
            create_convolvedFITS(
                original_fits=paths['img'],
                kernel_fits=paths['kernel'],
                survey=paths['survey'],
                psf_master_name=psf_master_name,
                master_survey=master_survey,
                output_dir=convolved_fits_dir,
                drivers=DRIVERS,
                bin_factor=bin_factors.get(paths['survey'], 1),
                force=force_conv,
            )

        # --- bands WITHOUT a kernel: write with PSFRESID -------------------
        for filt, survey, resid in unmatched:
            img = find_science_image(img_files, DRIVERS, survey, filt)
            if img is None:
                print(f"==> Warning: no science image for '{filt}' ({survey}); "
                      f"the band will be missing from the cube.")
                continue
            copy_as_convolved(
                original_fits=img,
                survey=survey,
                psf_master_name=psf_master_name,
                master_survey=master_survey,
                output_dir=convolved_fits_dir,
                drivers=DRIVERS,
                psf_residual_arcsec=resid,
                bin_factor=bin_factors.get(survey, 1),
                force=force_conv,
            )

        # --- master: same preparation, no convolution ----------------------
        # It goes through copy_as_convolved rather than a plain file copy so
        # that it is binned like its own survey's bands. With a master from a
        # survey whose bin factor is > 1, a plain copy would leave it on a
        # different grid and the alignment step would resample between grids.
        copy_as_convolved(
            original_fits=master_img_path,
            survey=master_survey,
            psf_master_name=psf_master_name,
            master_survey=master_survey,
            output_dir=convolved_fits_dir,
            drivers=DRIVERS,
            psf_residual_arcsec=0.0,
            is_master=True,
            bin_factor=bin_factors.get(master_survey, 1),
            force=force_conv,
        )

        print_matching_summary(psf_master_name, master_survey, r80, conv_pairs,
                               unmatched, len(img_by_filter))

    # =====================================================================
    # =========================== ALIGNMENT ===============================
    # =====================================================================
    if args.mode in ('full', 'alignment_only'):
        print(">>> Initiating image alignment process...")

        convolved_files_dict = discover_convolved_files(
            convolved_dir=convolved_fits_dir,
            galaxy=galaxy,
            target_master_filter=psf_master_name,
            selected_surveys=input_survey_list,
            strict=not args.allow_mixed,
        )

        ref_filter, reference_entry = next(
            (f, v) for f, v in convolved_files_dict.items() if v['is_master']
        )
        reference_fits = reference_entry['path']
        reference_survey = reference_entry["survey"]
        reference_apply_sip = DRIVERS[reference_survey.upper()].get_sip

        print(f">>> Aligning against chosen master: {reference_fits.name} "
              f"(Filter: {ref_filter})")

        user_input_filters = set(img_by_filter.keys())

        files_to_convert = []
        for filt, entry in convolved_files_dict.items():
            if entry['is_master'] or filt not in user_input_filters:
                continue

            output_filename = reproject_to_reference(
                img_to_reproject=entry["path"],
                img_survey=entry["survey"],
                img_filter=filt,
                reference_img=reference_fits,
                ref_survey=reference_survey,
                ref_filter=ref_filter,
                galaxy=galaxy,
                output_path=reprojected_dir,
                apply_sip_reference_img=reference_apply_sip,
                apply_sip_img_to_reproject=DRIVERS[entry['survey']].get_sip,
                verbose=not args.quiet,
            )
            files_to_convert.append({'path': output_filename,
                                     'survey': entry['survey']})

        # Copy (and normalise) the master into the reprojected directory
        reprojected_dir_gal = reprojected_dir / galaxy
        reprojected_dir_gal.mkdir(parents=True, exist_ok=True)
        master_reprojected_path = reprojected_dir_gal / reference_fits.name

        with fits.open(reference_fits) as hdu_ref:
            master_data = hdu_ref[0].data
            master_header = hdu_ref[0].header.copy()

        if reference_apply_sip:
            master_header['CTYPE1'] = 'RA---TAN-SIP'
            master_header['CTYPE2'] = 'DEC--TAN-SIP'

        fits.writeto(master_reprojected_path, master_data, master_header,
                     overwrite=True)
        print(f'\tCopied master FITS file: {master_reprojected_path}\n')
        files_to_convert.append({'path': master_reprojected_path,
                                 'survey': reference_survey})

        # ----------------------- UNIT CONVERSION -------------------------
        print(">>> Converting units to Jansky (Jy)...")
        unit_failures = []
        for item in files_to_convert:
            driver = DRIVERS[item['survey']]
            print(f"\tFile: {item['path']}")

            converted_data, converted_header = convert2Jansky(item['path'], driver)

            if converted_header.get('BUNIT') != 'Jy/pixel':
                unit_failures.append((item['path'].name,
                                      converted_header.get('BUNIT')))
                continue

            new_hdu = fits.PrimaryHDU(converted_data, converted_header)
            file_path = (item['path'].parent /
                         f"{item['path'].stem}_Jy_per_pixel{item['path'].suffix}")
            new_hdu.writeto(file_path, overwrite=True)
            print(f'\tSaved: {file_path}')
            print("\t" + 100 * "-")

        # A band that fails unit conversion would silently disappear from the
        # cube. Fail loudly instead.
        if unit_failures:
            print("\n==> Unit conversion failed for:")
            for name, bunit in unit_failures:
                print(f"      {name}: BUNIT = '{bunit}'")
            raise RuntimeError(
                f"{len(unit_failures)} band(s) were not converted to Jy/pixel. "
                f"The cube would be incomplete. Add the BUNIT handling to the "
                f"survey driver.")

    # =====================================================================
    # ============================ DATA CUBE ==============================
    # =====================================================================
    if args.mode in ('full', 'cube_only'):
        print(">>> Initiating data cube creation...")

        jansky_files_dict = discover_jansky_files(
            reprojected_dir=reprojected_dir,
            galaxy=galaxy,
            target_master_filter=psf_master_name,
            selected_surveys=input_survey_list,
            strict=not args.allow_mixed,
        )

        ordered_filters = sort_filters_by_wavelength(jansky_files_dict)
        print(f">>> Filters ordered by wavelength (UV -> IR): {ordered_filters}")

        if len(ordered_filters) != len(img_by_filter):
            print(f"==> Warning: the cube will have {len(ordered_filters)} "
                  f"planes but {len(img_by_filter)} bands were found. Check "
                  f"the alignment and unit-conversion logs.")

        cube_output_dir = output_dir / 'datacubes' / galaxy.lower()
        cube_output_dir.mkdir(parents=True, exist_ok=True)
        cube_base_name = cube_output_dir / f"{galaxy.lower()}_datacube"

        reference_entry = next(v for v in jansky_files_dict.values()
                               if v['is_master'])

        cubo, header = create_data_cube(
            jansky_files_dict=jansky_files_dict,
            ordered_filters=ordered_filters,
            reference_path=reference_entry['path'],
            drivers=DRIVERS,
            output_filename=cube_base_name,
            apply_mask=args.apply_mask,
            n_sigma=args.sigma,
            padding=50,
            sky_subtraction=True,
        )

        print(200 * '-' + "\n>>> DATACUBE COMPLETE\n" + 100 * '#')


if __name__ == "__main__":
    main()

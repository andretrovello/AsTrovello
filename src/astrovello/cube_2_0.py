from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

from mask_2_0 import (
    intersection_footprint_mask, 
    crop_to_mask_bbox,
    sky_level,
    sum_images,
    mask_after_sky_sub
)

def discover_jansky_files(
    reprojected_dir: Path,
    galaxy: str,
    target_master_filter: str | None = None,
    selected_surveys: list | None = None,
    strict: bool = True,
) -> dict:
    """Scan reprojected_dir/galaxy for the Jy/pixel files of ONE configuration.

    Same selection logic as `discover_convolved_files`: the header (PSFTARGT)
    decides, with the filename as a fallback for files written before that
    keyword existed. Filename-only matching is not enough - a leftover file
    from a run with a different master parses fine and is picked up silently.

    Args:
        strict: raise if files from another configuration are present. Set
            False only when several configurations are kept side by side on
            purpose.
    """
    gal_dir = Path(reprojected_dir) / galaxy
    if not gal_dir.is_dir():
        raise FileNotFoundError(
            f"No reprojected files found for galaxy '{galaxy}' at {gal_dir}")

    target_surveys = {s.upper() for s in selected_surveys} if selected_surveys else None
    result, foreign = {}, []

    def header_target(path):
        """Master this file was produced against, or None if not recorded."""
        try:
            return fits.getheader(path).get('PSFTARGT')
        except Exception:
            return None

    # ---------------- master ----------------
    matched_master = None
    for f in sorted(gal_dir.glob('*_master_Jy_per_pixel.fits')):
        # {gal}_{survey}_{filt}_master_Jy_per_pixel
        #          -6      -5     -4    -3  -2   -1
        # Parsed from the RIGHT so galaxy names containing '_' still work.
        parts = f.stem.split('_')
        try:
            survey, filt = parts[-6].upper(), parts[-5]
        except IndexError:
            print(f"==> Skipping unparseable master filename: {f.name}")
            continue

        if target_surveys is not None and survey not in target_surveys:
            foreign.append((f.name, f"survey {survey} not selected"))
            continue
        if target_master_filter is not None and filt != target_master_filter:
            foreign.append((f.name, f"master {filt}"))
            continue

        matched_master = (f, survey, filt)
        break

    if matched_master is None:
        existing = sorted(f.name for f in gal_dir.glob('*_master_Jy_per_pixel.fits'))
        raise ValueError(
            f"No Jy/pixel master for filter '{target_master_filter}' in "
            f"{gal_dir}.\nExisting: {existing or 'none'}\n"
            f"Run the alignment and unit-conversion stages first.")

    m_path, m_survey, m_filt = matched_master
    result[m_filt] = {'path': m_path, 'survey': m_survey, 'is_master': True}

    # ---------------- reprojected bands ----------------
    for f in sorted(gal_dir.glob('*_projection_Jy_per_pixel.fits')):
        # {gal}_{survey}_{filt}_on_{ref_survey}_{ref_filt}_projection_Jy_per_pixel
        #   -10     -9      -8   -7      -6         -5         -4      -3  -2   -1
        # Ten parts, not eight: the _Jy_per_pixel suffix adds three.
        parts = f.stem.split('_')
        try:
            survey, filt, name_target = parts[-9].upper(), parts[-8], parts[-5]
        except IndexError:
            print(f"==> Skipping unparseable filename: {f.name}")
            continue

        if target_surveys is not None and survey not in target_surveys:
            foreign.append((f.name, f"survey {survey} not selected"))
            continue

        # Header wins over filename; the filename is only a fallback for files
        # written before PSFTARGT existed.
        target = header_target(f) or name_target

        if target_master_filter is not None and target != target_master_filter:
            foreign.append((f.name, f"projected onto {target}"))
            continue

        if filt in result:
            raise ValueError(
                f"Two files claim band '{filt}': {result[filt]['path'].name} "
                f"and {f.name}. Clean {gal_dir} and reprocess.")

        result[filt] = {'path': f, 'survey': survey, 'is_master': False}

    if foreign:
        print(f"==> {len(foreign)} file(s) in {gal_dir.name} belong to another "
              f"configuration and were ignored:")
        for name, why in foreign[:10]:
            print(f"      {name}  ({why})")
        if strict:
            raise ValueError(
                f"Leftover files from a previous configuration in {gal_dir}. "
                f"Delete them, or pass strict=False (--allow_mixed).")

    print(f">>> Cube will use {len(result)} band(s): {sorted(result)}")
    return result


def create_data_cube(
    jansky_files_dict: dict, ordered_filters: list,
    reference_path: Path, drivers: dict, output_filename: Path,
    apply_mask: bool = True, n_sigma: float = 3, padding: int = 50,
    sky_subtraction: bool = True,
) -> tuple:
    print('\nInitiating hypercube creation...')

    # 1. Load every band
    raw_images, ref_header = [], None
    for filt in ordered_filters:
        entry = jansky_files_dict[filt]
        with fits.open(entry['path']) as hdu:
            raw_images.append(hdu[0].data)
            if entry['is_master']:
                ref_header = hdu[0].header.copy()
    if ref_header is None:
        with fits.open(reference_path) as hdu:
            ref_header = hdu[0].header.copy()

    # 2. Footprint: intersection of all selected bands
    print('Determining intersection footprint across surveys...')
    footprint_mask = intersection_footprint_mask(jansky_files_dict, drivers)

    # 3. Crop EVERY band to the smallest useful area BEFORE any processing
    print('Cropping to minimal useful area...')
    cropped_images, cropped_header, (y_off1, x_off1) = crop_to_mask_bbox(
        raw_images, ref_header, footprint_mask, padding=0
    )
    print(f"==> Footprint crop: {footprint_mask.shape} -> {cropped_images[0].shape}")

    ny, nx = cropped_images[0].shape
    cubo = np.empty((len(ordered_filters), ny, nx), dtype=np.float32)

    # 4. Sky subtraction, now on the already-cropped arrays
    if sky_subtraction:
        sub_images = []
        print('Performing sky subtraction...\n')
        print(154*'-')
        print(f"{'filter':6s} | {'valid_pixels_original':>22s} | {'sky_level_original':>23s} | {'%neg_original':>14s} | "
              f"{'valid_pixels_subtracted':>26s} | {'sky_level_subtracted':>27s} | {'%neg_subtracted':>18s} ")
        print(154*'-')

        for filt, img in zip(ordered_filters, cropped_images):
            # Valid pixels according to the survey driver's convention
            driver = drivers[jansky_files_dict[filt]['survey']]
            valid_pixels_mask = np.isfinite(img) & ~driver.get_invalid_mask(img)
            
            # Statistics before subtraction
            regular_dict = sky_level(img[valid_pixels_mask])
            
            # Subtract the sky level
            img_sub = np.where(valid_pixels_mask, img - regular_dict['sclip_median'], np.nan)
            
            # Statistics after subtraction
            subtracted_dict = sky_level(img_sub[valid_pixels_mask])
            sub_images.append(img_sub)
            
            # Print the formatted table row
            print(f"{filt:6s} | {regular_dict['valid_pixels']:>22d} | {regular_dict['sclip_median']:>+23.2e} | "
                  f"{regular_dict['pct_neg']:>14.2f} | {subtracted_dict['valid_pixels']:>26d} | {subtracted_dict['sclip_median']:>+27.2e} | "
                  f"{subtracted_dict['pct_neg']:>18.2f}")
                  
        print(154*'-')
        cropped_images = sub_images

    # 5. Signal (galaxy) mask, built on the summed cropped image
    mask_filename = output_filename.parent / 'master_signal_mask.fits'
    if apply_mask:
        summed = sum_images(cropped_images, footprint_mask=None)  # already cropped
        mask_final = mask_after_sky_sub(summed, N_SIGMA=n_sigma)
    else:
        mask_final = np.isfinite(cropped_images[0])

    fits.writeto(mask_filename, mask_final.astype(np.uint8), overwrite=True)
    print(f"==> Science mask saved to: {mask_filename.name}")

    for i, img in enumerate(cropped_images):
        cubo[i, :, :] = np.where(mask_final, img, np.nan)

    # 6. Second crop: now by the signal mask, with padding
    cube_planes, final_header, (y_off2, x_off2) = crop_to_mask_bbox(
        [cubo[i] for i in range(cubo.shape[0])], cropped_header, mask_final, padding=padding
    )
    cubo = np.stack(cube_planes, axis=0)
    print(f"==> Signal-mask cutout: {cropped_images[0].shape} -> {cubo.shape[1:]}")

    # 7. Final 3D header
    w_2d = WCS(final_header, naxis=2)
    pixel_area_arcsec2 = round(proj_plane_pixel_area(w_2d) * 3600**2, 4)
    w_3d = WCS(naxis=3)
    for i in [0, 1]:
        for p in ['crpix', 'crval', 'cdelt', 'ctype', 'cunit']:
            try:
                getattr(w_3d.wcs, p)[i] = getattr(w_2d.wcs, p)[i]
            except Exception:
                continue
    w_3d.wcs.crpix[2], w_3d.wcs.crval[2], w_3d.wcs.cdelt[2], w_3d.wcs.ctype[2] = 1, 0, 1, 'FILTER'
    
    cube_header = w_3d.to_header()
    cube_header['BUNIT'] = 'Jy/pixel'
    cube_header["PIXAREA"] = (pixel_area_arcsec2, 'area in square arcseconds')
    
    for i, filt in enumerate(ordered_filters):
        cube_header[f'FILT{i+1:03d}'] = filt

    # --- Header comments ---
    # Unique surveys contributing to this cube
    surveys_used = ", ".join(sorted(set(entry['survey'] for entry in jansky_files_dict.values())))
    # The filter flagged as master
    master_filter = next(filt for filt, entry in jansky_files_dict.items() if entry['is_master'])
    
    cube_header['COMMENT'] = f"Surveys combined in this cube: {surveys_used} (master filter : {master_filter})"
        
    # --- Dynamic output filename ---
    # Cube dimensions go into the filename, so several configurations coexist
    _, ny, nx = cubo.shape
    
    # stem (e.g. 'ngc1087_datacube') + _sci_{nx}x{ny}_Jy_per_pixel.fits
    final_output_path = output_filename.parent / f"{output_filename.stem}_sci_{nx}x{ny}_Jy_per_pixel.fits"
        
    fits.writeto(final_output_path, cubo, header=cube_header, overwrite=True)
    return cubo, cube_header
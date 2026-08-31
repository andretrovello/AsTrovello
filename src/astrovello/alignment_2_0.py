import os 
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Image alignment -------------------------------------------------------

def get_file_info_alignment(file_path: Path, drivers: dict) -> tuple:
    survey = drivers["BASE"].get_survey()

# Reproject S4G (Spitzer) images onto the PHANGS (HST) pixel grid
def survey_img_reproject(img_to_reproject_path: Path, reference_img: Path, drivers: dict,  output_path: Path) -> None:
    """
    Aligns an input survey image to the master file reference grid.
    Returns an array with the same spatial dimensions as the master reference.
    """
    hdu_img_base = fits.open(img_to_reproject_path)[0]
    hdu_ref = fits.open(reference_img)[0]

    survey_ref
    driver_
    galaxy_ref

     
    galaxy_img_base

    sci_file_s4g = s4g_file_path.name
    # Extract galaxy name and filter index (e.g., IRAC1) from filename
    galaxy_name, filter_mode = sci_file_s4g.split('.')[0].lower(), sci_file_s4g.split('.')[-2]

    # Initialize WCS (World Coordinate System) for both images
    w_ref = WCS(hdu_ref.header)
    w_img_base = WCS(hdu_img_base.header)
    
    # Force SIP (Simple Imaging Polynomial) correction type for Spitzer headers
    w_s4g.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    print('\n' + 100*'#' + '\nSIP correction added to the header!\n' + 100*'#')

    # Perform the interpolation/reprojection using reproject_interp
    # Surface brightness is preserved, but flux per pixel is not strictly conserved due to resampling
    array, _ = reproject_interp((hdu_img_base.data, w_img_base), w_ref, shape_out = hdu_ref.data.shape)

    img_base_new_header = hdu_img_base.header.copy()

    # Generate new WCS keywords based on the HST reference
    wcs_ref_header = w_ref.to_header(relax=True)

    # Clean old WCS keywords to prevent coordinate conflicts (especially CD vs PC matrices)
    wcs_keys_to_remove = [
        'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2',
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'
    ]
    for key in wcs_keys_to_remove:
        if key in img_base_new_header:
            del img_base_new_header[key]

    # Merge the new WCS information into the S4G header
    img_base_new_header.update(wcs_ref_header)
    img_base_new_header['CTYPE1'] = 'RA---TAN-SIP'
    img_base_new_header['CTYPE2'] = 'DEC--TAN-SIP'
    img_base_new_header['COMMENT'] = f'Reprojected to {ref_survey} grid. Flux not conserved per pixel, surface brightness preserved.'

    # Setup output directory
    output_path = os.path.expanduser(output_path)
    output_directory = os.path.join(output_path, galaxy_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        print(f"==> Directory created: {output_directory}")

    output_name = f'{galaxy_name}_s4g_irac{filter_mode}_on_phangs_projection.fits'
    fits.writeto(os.path.join(output_directory, output_name), array, s4g_new_header, overwrite=True)
    print('\n' + 100*'#' + f'\nReprojected FITS file: {output_name}\n' + 100*'#')
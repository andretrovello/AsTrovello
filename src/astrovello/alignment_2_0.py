import os 
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Image alignment -------------------------------------------------------

def discover_convolved_files(convolved_dir: Path, galaxy: str) -> dict:
    """
    Scans convolved_dir/galaxy for convolved + master FITS files produced
    by create_convolvedFITS, parsing survey/filter from the pipeline's own
    filename convention. Makes the alignment step runnable standalone
    (--mode alignment_only), independent of whether convolution ran in
    the same CLI invocation.

    Returns
    -------
    dict of {str : dict}
        ``{filter_name: {'path': Path, 'survey': str, 'is_master': bool}}``
    """
    gal_dir = Path(convolved_dir) / galaxy
    if not gal_dir.is_dir():
        raise FileNotFoundError(f"No convolved files found for galaxy '{galaxy}' at {gal_dir}")

    result = {}

    for f in gal_dir.glob('*_master.fits'):
        _, survey, filt, _ = f.stem.split('_')            # {gal}_{survey}_{filt}_master
        result[filt] = {'path': f, 'survey': survey.upper(), 'is_master': True}

    for f in gal_dir.glob('*_convolved.fits'):
        _, survey, filt, *_ = f.stem.split('_')            # {gal}_{survey}_{filt}_to_{master_survey}_{master}_convolved
        result[filt] = {'path': f, 'survey': survey.upper(), 'is_master': False}

    if not any(v['is_master'] for v in result.values()):
        raise ValueError(f"No master file found in {gal_dir} — run convolution first.")

    return result

# Reproject higher resolution images to master image reference frame
def reproject_to_reference(img_to_reproject: Path, reference_img: Path,
                           output_path: Path,
                           apply_sip_img_to_reproject = False,
                            apply_sip_reference_img = False) -> None:
    """
    Aligns an input survey image to the master file reference grid.
    Returns an array with the same spatial dimensions as the master reference.
    """
    hdu_img_base = fits.open(img_to_reproject)[0]
    hdu_ref = fits.open(reference_img)[0]

    img_base_name = img_to_reproject.name
    img_ref_name = reference_img.name
    galaxy, img_base_survey, img_base_filter, _, ref_survey, ref_filter, *_ = img_base_name.split("_")

    # Initialize WCS (World Coordinate System) for both images
    w_ref = WCS(hdu_ref.header)
    w_img_base = WCS(hdu_img_base.header)
    
    # Force SIP (Simple Imaging Polynomial) correction type for Spitzer headers
    if apply_sip_img_to_reproject:
        w_img_base.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        print('\n' + 100*'#' + f'\n\tSIP correction added to {img_base_name} header!\n' + 100*'#')
    if apply_sip_reference_img:
        w_ref.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        print('\n' + 100*'#' + f'\n\tSIP correction added to {img_ref_name} header!\n' + 100*'#')

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
    img_base_new_header['COMMENT'] = f'Reprojected to {ref_survey} grid. Flux not conserved per pixel, surface brightness preserved.'

    # Setup output directory
    output_directory =  output_path / galaxy
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok = True)
        print(f">>> Directory created: {output_directory}")

    output_name = f'{galaxy}_{img_base_survey}_{img_base_filter}_on_{ref_survey}_{ref_filter}_projection.fits'
    fits.writeto(os.path.join(output_directory, output_name), array, img_base_new_header, overwrite=True)
    print(f'\n\tReprojected FITS file: {output_name}\n')
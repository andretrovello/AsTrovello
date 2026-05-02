import os 
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Image alignment -------------------------------------------------------

# Reproject S4G (Spitzer) images onto the PHANGS (HST) pixel grid
def S4G2PHANGS_reproject(s4g_file_path, phangs_ref_file_path, output_path):
    """
    Aligns a Spitzer/IRAC image to the HST pixel grid.
    Returns an array with the same spatial dimensions as the PHANGS reference.
    """
    hdu_phangs = fits.open(phangs_ref_file_path)[0]
    hdu_s4g = fits.open(s4g_file_path)[0]

    sci_file_s4g = s4g_file_path.name
    # Extract galaxy name and filter index (e.g., IRAC1) from filename
    galaxy_name, filter_mode = sci_file_s4g.split('.')[0].lower(), sci_file_s4g.split('.')[-2]
    if 'mosaic' in galaxy_name: galaxy_name.replace('mosaic',  '') 

    # Initialize WCS (World Coordinate System) for both images
    w_phangs = WCS(hdu_phangs.header)
    w_s4g = WCS(hdu_s4g.header)
    
    # Force SIP (Simple Imaging Polynomial) correction type for Spitzer headers
    w_s4g.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    print('\n' + 100*'#' + '\nSIP correction added to the header!\n' + 100*'#')

    # Perform the interpolation/reprojection using reproject_interp
    # Surface brightness is preserved, but flux per pixel is not strictly conserved due to resampling
    array, footprint = reproject_interp((hdu_s4g.data, w_s4g), w_phangs, shape_out=hdu_phangs.data.shape)

    s4g_new_header = hdu_s4g.header.copy()

    # Generate new WCS keywords based on the HST reference
    wcs_phangs_header = w_phangs.to_header(relax=True)

    # Clean old WCS keywords to prevent coordinate conflicts (especially CD vs PC matrices)
    wcs_keys_to_remove = [
        'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2',
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'
    ]
    for key in wcs_keys_to_remove:
        if key in s4g_new_header:
            del s4g_new_header[key]

    # Merge the new WCS information into the S4G header
    s4g_new_header.update(wcs_phangs_header)
    s4g_new_header['CTYPE1'] = 'RA---TAN-SIP'
    s4g_new_header['CTYPE2'] = 'DEC--TAN-SIP'
    s4g_new_header['COMMENT'] = 'Reprojected to PHANGS grid. Flux not conserved per pixel, surface brightness preserved.'

    # Setup output directory
    output_path = os.path.expanduser(output_path)
    output_directory = os.path.join(output_path, galaxy_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        print(f"📁 Directory created: {output_directory}")

    output_name = f'{galaxy_name}_s4g_irac{filter_mode}_on_phangs_projection.fits'
    fits.writeto(os.path.join(output_directory, output_name), array, s4g_new_header, overwrite=True)
    print('\n' + 100*'#' + f'\nReprojected FITS file: {output_name}\n' + 100*'#')
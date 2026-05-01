import numpy as np
from astropy.io import fits

# -------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- FITS unit conversion-----------------------------------------------------

def convert2Jansky(fits_file):
    """
    Converts image units to Jansky per pixel.
    Handles HST (Electrons/s to Jy via PHOTFNU) and Spitzer (MJy/sr to Jy/pixel via pixel area).
    """
    with fits.open(fits_file) as hdu:
        data, header = hdu[0].data, hdu[0].header
    
    new_data, new_header = data.copy(), header.copy()

    # HST Case: Using Photometric Flux density (Jy*s/e)
    if header.get('BUNIT') == 'ELECTRONS/S':
        new_data *= header['PHOTFNU']
        new_header['BUNIT'] = 'Jy/pixel'
        print("HST: Converted using PHOTFNU")

    # Spitzer Case: Surface brightness to flux per pixel
    elif header.get('BUNIT') == 'MJy/sr': 
        pixel_area_arcsec2 = np.abs(header['PXSCAL1'] * header['PXSCAL2'])
        # 1 arcsec² ≈ 2.35e-11 steradians
        pixel_area_sr = pixel_area_arcsec2 * 2.3504430539e-11
        # Convert MJy -> Jy (1e6) and sr -> pixel area
        new_data = new_data * 1e6 * pixel_area_sr
        new_header['BUNIT'] = 'Jy/pixel'
        print(f"S4G: Converted using pixel area ({pixel_area_arcsec2} arcsec2)")

    return new_data, new_header
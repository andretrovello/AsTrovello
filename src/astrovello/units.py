import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

# ------------------------------------------- FITS unit conversion-----------------------------------------------------

def convert2Jansky(fits_file):
    """
    Converts image units to Jansky per pixel.
    Handles HST (Electrons/s to Jy via PHOTFNU) and Spitzer (MJy/sr to Jy/pixel 
    using the true pixel area via WCS).
    """
    with fits.open(fits_file) as hdu:
        data, header = hdu[0].data, hdu[0].header
    
    new_data, new_header = data.copy(), header.copy()

    # HST Case: Default PHANGS unit (ELECTRONS/S)
    if header.get('BUNIT') == 'ELECTRONS/S':
        # PHOTFNU is the photometric flux density (Jy*s/e-)
        new_data *= header['PHOTFNU']
        new_header['BUNIT'] = 'Jy/pixel'
        print(f"HST: Converted {fits_file.name} using PHOTFNU.")

    # Spitzer/S4G Case: Surface brightness (MJy/sr) to Flux per pixel
    elif header.get('BUNIT') == 'MJy/sr': 
        # Create WCS object to read the true image geometry
        w = WCS(header)
        
        # proj_plane_pixel_area returns the area in square degrees (deg^2)
        # This is immune to static header keywords that might be outdated after reprojection
        pixel_area_deg2 = proj_plane_pixel_area(w)
        
        # Convert from square degrees to steradians (sr)
        # 1 deg^2 = (pi/180)^2 steradians
        pixel_area_sr = pixel_area_deg2 * (np.pi / 180)**2
        
        # Final conversion: MJy -> Jy (1e6) and sr -> pixel area
        new_data = new_data * 1e6 * pixel_area_sr
        new_header['BUNIT'] = 'Jy/pixel'
        
        # Log for verification
        pixel_area_arcsec2 = pixel_area_deg2 * (3600**2)
        print(f"S4G: Converted {fits_file.name} using true WCS area ({pixel_area_arcsec2:.4f} arcsec2/px).")

    else:
        print(f"Warning: Unit {header.get('BUNIT')} not recognized for automatic conversion.")

    return new_data, new_header
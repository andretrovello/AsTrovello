import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

def convert2Jansky(fits_file):
    """
    Converts image units to Jansky per pixel.
    Handles HST flux/error maps and Spitzer flux/error maps.
    Recovers missing photometric keywords dynamically.
    """
    with fits.open(fits_file) as hdu:
        data, header = hdu[0].data, hdu[0].header
    
    new_data, new_header = data.copy(), header.copy()
    filename_str = fits_file.name

    # --- Subfunção para resgatar o PHOTFNU perdido do AstroDrizzle ---
    def get_photfnu(hdr):
        if 'PHOTFNU' in hdr:
            return hdr['PHOTFNU']
        elif 'PHOTFLAM' in hdr:
            filt = hdr.get('FILTER', '').strip().upper()
            # Comprimentos de onda pivô (Angstroms) para os filtros do PHANGS-HST
            pivots = {
                'F275W': 2707.19995, 
                'F336W': 3354.84995, 
                'F438W': 4325.55005, 
                'F555W': 5305.94995, 
                'F814W': 8048.1001
            }
            if filt in pivots:
                # Calcula PHOTFNU a partir do PHOTFLAM
                return 3.34e4 * hdr['PHOTFLAM'] * (pivots[filt]**2)
            else:
                raise KeyError(f"PHOTFNU missing and pivot wavelength unknown for filter '{filt}'.")
        else:
            raise KeyError("Header missing photometric keywords (PHOTFNU/PHOTFLAM).")

    # ---------------------------------------------------------
    # 1. HST Case: Flux maps OR Convolved Error maps (Sigma)
    # ---------------------------------------------------------
    if 'phangs' in filename_str:
        # Flux data
        if header.get('BUNIT') == 'ELECTRONS/S':
            # PHOTFNU is the photometric flux density (Jy*s/e-)
            new_data *= header['PHOTFNU']
            new_header['BUNIT'] = 'Jy/pixel'
            print(f"HST: Converted {filename_str} using PHOTFNU.")

        # Error data
        elif header.get('BUNIT') == 'UNITLESS':
            photfnu = get_photfnu(header)
            new_data *= photfnu
            new_header['BUNIT'] = 'Jy/pixel'
            print(f"HST Error Map: Converted {filename_str} using PHOTFNU.")

    # ---------------------------------------------------------
    # 2. Spitzer/S4G Case: Surface brightness (MJy/sr) to Jy/pixel
    # ---------------------------------------------------------
    elif 's4g' in filename_str:
        if header.get('BUNIT') == 'MJy/sr': 
            w = WCS(header)
            pixel_area_deg2 = proj_plane_pixel_area(w)
            pixel_area_sr = pixel_area_deg2 * (np.pi / 180)**2
            
            new_data = new_data * 1e6 * pixel_area_sr
            new_header['BUNIT'] = 'Jy/pixel'
            
            pixel_area_arcsec2 = pixel_area_deg2 * (3600**2)
            print(f"S4G: Converted {fits_file.name} using true WCS area ({pixel_area_arcsec2:.4f} arcsec2/px).")

    else:
        print(f"Warning: Unit {header.get('BUNIT')} not recognized for automatic conversion in {fits_file.name}.")

    return new_data, new_header


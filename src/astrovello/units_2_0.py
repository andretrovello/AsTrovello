import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from pathlib import Path
from drivers import BASE_Driver, PHANGS_Driver, S4G_Driver

def convert2Jansky(fits_file: Path, driver) -> tuple:
    """
    Converts image units to Jansky per pixel.
    Handles HST flux/error maps and Spitzer flux/error maps.
    Recovers missing photometric keywords dynamically.
    """
    with fits.open(fits_file) as hdu:
        data, header = hdu[0].data, hdu[0].header

    new_data = np.where(data == 0, np.nan, data)
    new_header = header.copy()
    filename_str = fits_file.name

    print(f"\tConverting: {filename_str}")
    converted_data, converted_header = driver.convert2Jansky(new_data, new_header)
    return converted_data, converted_header


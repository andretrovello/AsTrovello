from config import PIVOT_WAVELENGTHS
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astropy.io import fits
import numpy as np
import warnings

def science_pixel_scale(sci_path, hdu_ext = 0) -> float:
    """Escala de pixel da imagem de ciencia, em arcsec/px, lida do WCS."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = fits.getheader(sci_path, ext = hdu_ext)
        return float(np.sqrt(proj_plane_pixel_area(WCS(h, naxis=2)) * 3600**2))

def sort_filters_by_wavelength(jansky_files_dict: dict) -> list:
    """
    Returns the filter names from jansky_files_dict sorted by increasing
    pivot wavelength (UV -> IR), so the final data cube's spectral axis
    is physically ordered regardless of which survey each filter came from.

    Parameters
    ----------
    jansky_files_dict : dict
        As returned by `discover_jansky_files` — keyed by filter name.

    Returns
    -------
    list of str
        Filter names in ascending wavelength order.
    """
    missing = [f for f in jansky_files_dict if f.lower() not in PIVOT_WAVELENGTHS]
    if missing:
        raise KeyError(
            f"Pivot wavelength unknown for filter(s): {missing}. "
            f"Add them to PIVOT_WAVELENGTHS in config.py before building the cube."
        )
    return sorted(jansky_files_dict.keys(), key=lambda f: PIVOT_WAVELENGTHS[f.lower()])


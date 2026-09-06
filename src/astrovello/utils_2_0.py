from config import PIVOT_WAVELENGTHS

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
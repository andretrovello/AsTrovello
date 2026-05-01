from astropy.io import fits
import tqdm
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- Mask ----------------------------------------------------------

def phangs_intersection_mask(ref_file):
    """Detects the valid observation area by checking where data exists (!= 0)."""
    print('Finding intersection area between surveys...')
    if not ref_file: return None    
    ref_data = fits.getdata(ref_file[0], ext=0)
    return (ref_data != 0) 

def soma_img(aligned_images, ref_file):
    """Integrates all images into a single 2D plane to create a signal-based mask."""
    res, inter_mask = None, phangs_intersection_mask(ref_file)
    for data_orig in tqdm(aligned_images, desc="Integrating for mask"):
        data = data_orig.copy()
        if inter_mask is not None and data.shape == inter_mask.shape:
            data[~inter_mask] = 0 # Zero out regions outside the observation footprint
        res = data if res is None else res + data
    return res

def mask(data, N_SIGMA=3):
    """
    Performs sky subtraction and generates a signal mask using Median Absolute Deviation (MAD).
    Identifies 'objects' as pixels N_SIGMA above the background noise level.
    """
    data_filtrada = data[data != 0]
    if data_filtrada.size == 0: return np.zeros_like(data), np.zeros_like(data, dtype=bool)

    local_bg = np.nanmedian(data_filtrada)
    data_subtraida = data - local_bg
    residuo_filtrado = data_subtraida[data != 0] 
    
    noise_median = np.nanmedian(residuo_filtrado)
    mad = np.nanmedian(np.abs(residuo_filtrado - noise_median))
    sigma_bg = 1.4826 * mad # Conversion factor from MAD to Sigma

    mask_res = (data_subtraida > (N_SIGMA * sigma_bg)) if sigma_bg > 0 else np.zeros_like(data, dtype=bool)
    if sigma_bg > 0: data_subtraida[data_subtraida < 0] = 0
    
    return data_subtraida, mask_res

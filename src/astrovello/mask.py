from astropy.io import fits
from tqdm import tqdm
import numpy as np
from astropy.stats import sigma_clipped_stats

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- Mask ----------------------------------------------------------

def phangs_intersection_mask(ref_file):
    """Detects the valid observation area by checking where data exists (!= 0)."""
    print('Finding intersection area between surveys...')
    if not ref_file: return None    
    ref_data = fits.getdata(ref_file[0], ext=0)
    return (ref_data != 0) 

def sum_images(aligned_images, ref_file):
    """Integrates all images into a single 2D plane to create a signal-based mask."""
    res, inter_mask = None, phangs_intersection_mask(ref_file)
    for data_orig in tqdm(aligned_images, desc="Integrating for mask"):
        data = data_orig.copy()
        if inter_mask is not None and data.shape == inter_mask.shape:
            data[~inter_mask] = 0 # Zero out regions outside the observation footprint
        res = data if res is None else res + data
    return res

def create_mask(data, n_sigma=3):
    """
    Performs sky subtraction and generates a signal mask using Median Absolute Deviation (MAD).
    Identifies 'objects' as pixels n_sigma above the background noise level.
    """
    filtered_data = data[data != 0]
    if filtered_data.size == 0: return np.zeros_like(data), np.zeros_like(data, dtype=bool)

    local_bg = np.nanmedian(filtered_data)
    subtracted_data = data - local_bg
    filtered_residual = subtracted_data[data != 0] 
    
    noise_median = np.nanmedian(filtered_residual)
    mad = np.nanmedian(np.abs(filtered_residual - noise_median))
    sigma_bg = 1.4826 * mad # Conversion factor from MAD to Sigma

    mask_res = (subtracted_data > (n_sigma * sigma_bg)) if sigma_bg > 0 else np.zeros_like(data, dtype=bool)
    if sigma_bg > 0: subtracted_data[subtracted_data < 0] = 0
    
    return subtracted_data, mask_res

def mask_after_sky_sub(data, n_sigma=3):
    """
    One-sided object mask (e.g., galaxy) post-sky subtraction.
    Estimates background noise via sigma-clipping (using the same criteria 
    applied during sky level estimation) and masks pixels > n_sigma * noise.
    """
    valid = data[np.isfinite(data) & (data != 0)]
    if valid.size == 0:
        return np.zeros_like(data, dtype=bool)

    # Sky-level sigma clipping: 3-sigma, 5 iterations
    _, sky_median, sky_std = sigma_clipped_stats(valid, sigma=3.0, maxiters=5)
    threshold = sky_median + (n_sigma * sky_std)
    # One-sided thresholding: only select pixels above the noise level
    return data > threshold
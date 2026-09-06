import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

def intersection_footprint_mask(jansky_files_dict: dict, drivers: dict) -> np.ndarray:
    footprint_mask = None
    for filt, entry in jansky_files_dict.items():
        data = fits.getdata(entry['path'])
        valid = ~np.isnan(data)
        footprint_mask = valid if footprint_mask is None else (footprint_mask & valid)
    return footprint_mask

def crop_to_mask_bbox(images: list, header, mask: np.ndarray, padding: int = 0) -> tuple:
    ny, nx = mask.shape
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Mask is empty — no valid pixels to crop to.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    y_min, y_max = max(0, y_min - padding), min(ny, y_max + padding)
    x_min, x_max = max(0, x_min - padding), min(nx, x_max + padding)

    cropped_mask = mask[y_min:y_max, x_min:x_max]
    cropped_images = []
    for img in images:
        cropped = img[y_min:y_max, x_min:x_max].copy()
        cropped[~cropped_mask] = np.nan   
        cropped_images.append(cropped)

    new_header = header.copy()
    if 'CRPIX1' in new_header and 'CRPIX2' in new_header:
        new_header['CRPIX1'] -= x_min
        new_header['CRPIX2'] -= y_min
    new_header['NAXIS1'] = x_max - x_min
    new_header['NAXIS2'] = y_max - y_min

    return cropped_images, new_header, (y_min, x_min)

def sky_level(plane):
    v = plane[np.isfinite(plane)]      
    v = v[v != 0.0]                    
    _, sclip_median, _ = sigma_clipped_stats(v, sigma=3.0, maxiters=5)  
    return dict(valid_pixels = v.size,
                sclip_median = float(sclip_median),
                pct_neg = 100.0 * np.mean(v < 0))

def sum_images(aligned_images, footprint_mask=None):
    res = None
    for data_orig in aligned_images:
        data = data_orig.copy()
        if footprint_mask is not None and data.shape == footprint_mask.shape:
            data[~footprint_mask] = 0.0 
        data_clean = np.nan_to_num(data, nan=0.0)
        res = data_clean if res is None else res + data_clean
    return res

def mask_after_sky_sub(data, N_SIGMA=3):
    valid = data[np.isfinite(data) & (data != 0)]
    if valid.size == 0:
        return np.zeros_like(data, dtype=bool)

    _, sky_median, sky_std = sigma_clipped_stats(valid, sigma=3.0, maxiters=5)
    threshold = sky_median + (N_SIGMA * sky_std)
    return data > threshold
import numpy as np
import os
from astropy.io import fits
from astropy.convolution import convolve_fft
import shutil
from collections import defaultdict
from pathlib import Path
from scipy.ndimage import binary_dilation, label
from astropy.nddata import block_reduce 

# -------------------------------------------- Image convolution ------------------------------------------------------

def get_fwhm_simple(data):
    """
    Estimates the Full Width at Half Maximum (FWHM) using spatial moments.
    Suitable for centered point sources (PSFs).
    FWHM = 2.355 * sigma (for a Gaussian profile).
    """
    y, x = np.indices(data.shape)
    center_y, center_x = np.array(data.shape) // 2
    
    # Calculate brightness-weighted spatial variance
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    variance = np.sum(data * r**2) / np.sum(data)
    sigma = np.sqrt(variance)
    
    return 2.355 * sigma

def radial_profile(data, center):
    """Generates a 1D radial brightness profile from a 2D image."""
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr

def calculaFWHM_radial_profile(files_path):
    """
    Iterates through a folder of PSF files, filters them by survey,
    and returns dictionaries containing their FWHM and radial profiles.
    """
    FWHM_dict, dict_profiles, valid_files = {}, {}, []
    file_list = list(files_path.glob('*.fits'))
    
    for file in file_list:
        # Survey-specific filename parsing
        if 'S4G' in str(files_path):
            if file.name == 'IRAC1_col129_row129.fits': filter_name = 'irac1'
            elif file.name == 'IRAC2_col129_row129.fits': filter_name = 'irac2'
            else: continue
        elif 'PHANGS' in str(files_path):
            filter_name = file.name.replace('.fits', '').split('_')[-1].lower() 
        else: continue

        try:
            with fits.open(file, ignore_missing_end=True) as hdu:
                # Dynamically find the data HDU
                data = next((h.data for h in hdu if h.data is not None), None)
                if data is not None:
                    if data.ndim == 3: data = data[0] # Flatten 3D PSF cubes
                    
                    FWHM_dict[filter_name] = get_fwhm_simple(data)
                    dict_profiles[filter_name] = radial_profile(data, (data.shape[1]//2, data.shape[0]//2))
                    valid_files.append(file.name)
                    print(f"Succesfully read: {filter_name}")
        except Exception as e:
            print(f"Processing error {file.name}: {e}")

    return FWHM_dict, dict_profiles, valid_files

def final_clean_psf(input_file, output_file):
    """
    Standardizes PSF headers and performs true downsampling for PyPHER compatibility.
    Calculates pixel scales, bins down oversampled PSFs, and ensures correct 
    centering and coordinate keywords.
    """
    if 'WFC3UV' in input_file:
        # HST scale: native 0.0395"/pix. We will bin down the 4x oversampled data.
        pixel_scale_arcsec = 0.0395 
        pixel_scale_deg = pixel_scale_arcsec / 3600.0

        with fits.open(input_file, ignore_missing_end=True) as hdu:
            # Average the PSF cube to get a 2D representative PSF
            data_2d = np.mean(hdu[0].data, axis=0)
            
            # Downsample the array by a factor of 4 (summing blocks of 4x4 pixels)
            data_2d_binned = block_reduce(data_2d, block_size=4, func=np.sum)
            
            # Force odd parity: PyPHER prefers kernels/PSFs with an odd number of pixels
            if data_2d_binned.shape[0] % 2 == 0:
                data_2d_binned = data_2d_binned[:-1, :-1]
                
            # Normalize to ensure flux conservation
            data_2d_binned = data_2d_binned / np.sum(data_2d_binned)

            new_hdu = fits.PrimaryHDU(data_2d_binned)
            
            # Inject WCS keywords required by PyPHER/Astropy
            new_hdu.header.update({
                'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                'CRVAL1': 0.0, 'CRVAL2': 0.0,
                'CRPIX1': (data_2d_binned.shape[1] // 2) + 1, 'CRPIX2': (data_2d_binned.shape[0] // 2) + 1,
                'CDELT1': -pixel_scale_deg, 'CDELT2': pixel_scale_deg,
                'PIXSCALE': pixel_scale_arcsec
            })
            new_hdu.writeto(output_file, overwrite=True)
            print(f"==> File ready to be applied in PyPHER (Binned 4x): {os.path.basename(output_file)}")

    elif any(x in input_file for x in ['IRAC1', 'IRAC2']):
        # Spitzer scale: native ~1.22"/pix. We will bin down the 5x oversampled data.
        pixel_scale_arcsec = 1.221 if 'IRAC1' in input_file else 1.213
        pixel_scale_deg = pixel_scale_arcsec / 3600.0

        with fits.open(input_file) as hdu:
            data_raw = hdu[0].data
            
            # --- THE MAGIC HAPPENS HERE ---
            # Downsample the array by a factor of 5
            data_2d_binned = block_reduce(data_raw, block_size=5, func=np.sum)
            
            # Force odd parity
            if data_2d_binned.shape[0] % 2 == 0:
                data_2d_binned = data_2d_binned[:-1, :-1]
                
            # Normalize to ensure flux conservation
            data_2d_binned = data_2d_binned / np.sum(data_2d_binned)

            new_hdu = fits.PrimaryHDU(data_2d_binned)
            new_hdu.header.update({
                'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                'CRVAL1': 0.0, 'CRVAL2': 0.0,
                'CRPIX1': (data_2d_binned.shape[1] // 2) + 1, 'CRPIX2': (data_2d_binned.shape[0] // 2) + 1,
                'CDELT1': -pixel_scale_deg, 'CDELT2': pixel_scale_deg,
                'PIXSCALE': pixel_scale_arcsec
            })
            new_hdu.writeto(output_file, overwrite=True)
            print(f"==> File ready to be applied in PyPHER (Binned 5x): {os.path.basename(output_file)}")

def pypher_kernel_creation(todos_fwhm, psf_master_path, input_dir, output_dir):
    """
    Prepares a list of shell commands for the PyPHER library to generate homogenization kernels.
    It identifies which PSF belongs to which survey to locate files in the 'PSF_LIMPAS' folders.
    """
    psf_path_phangs_clean = input_dir / 'PHANGS' / 'PSF_LIMPAS'
    psf_path_s4g_clean = input_dir / 'S4G' / 'PSF_LIMPAS'
    psf_master_name = psf_master_path.stem.split('_')[0].lower()

    if os.path.exists(output_dir):
        print(f"==>  Removing previous directory: {os.path.basename(output_dir)}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    comandos_pypher = []
    for filtro in todos_fwhm.keys():
        if filtro == psf_master_name: continue
        
        # Determine High-Resolution PSF path
        if filtro.startswith('f'):
            psf_high_res = os.path.join(psf_path_phangs_clean, f"PSFSTD_WFC3UV_{filtro.upper()}.fits")
        elif filtro.startswith('i'):    
            psf_high_res = os.path.join(psf_path_s4g_clean, f"{filtro.upper()}_col129_row129.fits")
        else: continue

        kernel_name = os.path.join(output_dir, f"kernel_{filtro}_to_{psf_master_name}.fits")
        # Format command: pypher [HR_PSF] [Target_PSF] [Output_Kernel]
        comandos_pypher.append(f"pypher {psf_high_res} {psf_master_path} {kernel_name}")

    return comandos_pypher

def convolved_dict(path_phangs, path_s4g_reprojected, path_kernels):
    """
    Organizes all images and kernels into a nested dictionary indexed by filter.
    Used to pair the correct kernel with its corresponding image for convolution.
    """
    phangs_files = list(path_phangs.glob('*exp-drc-sci.fits'))
    s4g_files = list(path_s4g_reprojected.glob('*.fits'))
    kernel_files = list(path_kernels.glob('*.fits'))

    all_files = phangs_files + s4g_files + kernel_files
    filter_info = [f.name.split('_')[1] for f in sorted(kernel_files)]

    fftconvolve_dict = defaultdict(lambda: {'kernel': {}, 'img': {}}) 
    for f in filter_info:
        for file in all_files:
            file_string = str(file).lower()
            if f in file_string:
                key = 'kernel' if 'kernel' in file_string else 'img'
                fftconvolve_dict[f][key]['path'] = file
                fftconvolve_dict[f][key]['name'] = file.name
    return fftconvolve_dict

# PHANGS convolution function 

def convolve_phangs(img_data, kernel_norm, kernel_size):
    """
    Performs FFT-based convolution for HST/PHANGS images.
    
    PHANGS images contain two types of invalid pixels:
    - Border zeros: outside the detector field of view
    - Internal zeros: dead pixels or masked regions
    
    Strategy:
    - Border zeros → fill=0 + expand mask (no real signal to interpolate)
    - Internal zeros → interpolate (surrounded by real signal)
    """
    # Convert zeros to NaN for proper treatment
    img_nan = img_data.copy().astype(float)
    img_nan[img_data == 0] = np.nan
    nan_mask = np.isnan(img_nan)

    # --- Separate border NaNs from internal NaNs ---
    # Border mask: NaNs touching the image edges
    border_seed = np.zeros_like(nan_mask)
    border_seed[0, :]  = nan_mask[0, :]
    border_seed[-1, :] = nan_mask[-1, :]
    border_seed[:, 0]  = nan_mask[:, 0]
    border_seed[:, -1] = nan_mask[:, -1]

    # Expand border mask to include all connected NaN regions touching the edge
    labeled, _ = label(nan_mask)
    border_labels = set(labeled[border_seed & nan_mask])
    border_mask = np.isin(labeled, list(border_labels))

    # Internal mask: NaN pixels not connected to the border
    internal_mask = nan_mask & ~border_mask

    # --- Prepare image for convolution ---
    # Border NaNs → 0 (fill, no interpolation)
    # Internal NaNs → kept as NaN (will be interpolated by convolve_fft)
    img_to_conv = img_nan.copy()
    img_to_conv[border_mask] = 0.0

    # Convolve — interpolate handles internal NaNs correctly
    convolved_img = convolve_fft(
        img_to_conv,
        kernel_norm,
        normalize_kernel = False,
        nan_treatment    = 'interpolate',
        preserve_nan     = False, 
        allow_huge = True
    )

    # --- Restore border zeros ---
    # Expand border mask by kernel radius to mask pixels affected by the zero-fill
    structure = np.ones((kernel_size, kernel_size))
    expanded_border = binary_dilation(border_mask, structure=structure)
    convolved_img[expanded_border] = 0.0

    return convolved_img

# S4G convolution function 

def convolve_irac(img_data, kernel_norm, kernel_size):
    """
    Performs FFT-based convolution for Spitzer/IRAC images.
    
    IRAC images contain a NaN gap between the science array and the sky region.
    After reprojection onto the PHANGS grid, some of these NaNs are carried over.
    
    Strategy:
    - NaN gap → fill=0 (interpolating across the gap would invent signal)
    - Expand NaN mask by kernel radius to mask pixels affected by the gap boundary
    """
    # Save original NaN mask before any modification
    nan_mask_original = np.isnan(img_data)

    # Convolve with fill=0 — avoids interpolating across the NaN gap
    convolved_img = convolve_fft(
        img_data,
        kernel_norm,
        normalize_kernel = False,
        nan_treatment    = 'fill',
        fill_value       = 0.0,
        preserve_nan     = False,
        allow_huge = True
    )

    # Expand NaN mask to cover pixels affected by the gap boundary
    # Pixels within kernel_size/2 of the gap are potentially contaminated
    structure = np.ones((kernel_size, kernel_size))
    expanded_nan_mask = binary_dilation(nan_mask_original, structure=structure)

    # Restore NaNs in the expanded region
    convolved_img[expanded_nan_mask] = np.nan

    return convolved_img

# Diganose negatives
# In case negative number Warning appears

def diagnose_negatives(convolved_img, img_data, filt, survey):
    """
    Diagnoses the origin of negative pixels after convolution.
    Helps determine if negatives are border artifacts or internal signal issues.
    """
    neg_mask = convolved_img < 0
    n_neg = np.sum(neg_mask)
    pct_neg = n_neg / convolved_img.size * 100
    
    print(f"\n--- Negative pixel diagnosis: {filt} ({survey}) ---")
    print(f"Total negative pixels: {n_neg} ({pct_neg:.2f}%)")
    print(f"Min value:  {np.nanmin(convolved_img):.6e}")
    print(f"Max value:  {np.nanmax(convolved_img):.6e}")
    print(f"Ratio min/max: {abs(np.nanmin(convolved_img))/np.nanmax(convolved_img):.4%}")
    
    # --- Check 1: Are negatives concentrated at the border? ---
    ny, nx = convolved_img.shape
    border_width = 50  # pixels
    border_region = np.zeros_like(neg_mask, dtype=bool)
    border_region[:border_width, :]  = True
    border_region[-border_width:, :] = True
    border_region[:, :border_width]  = True
    border_region[:, -border_width:] = True
    
    neg_in_border   = np.sum(neg_mask & border_region)
    neg_in_interior = np.sum(neg_mask & ~border_region)
    
    print(f"\nNegatives in border region:   {neg_in_border} ({neg_in_border/n_neg*100:.1f}%)")
    print(f"Negatives in interior region: {neg_in_interior} ({neg_in_interior/n_neg*100:.1f}%)")
    
    # --- Check 2: Are negatives where input was zero/NaN? ---
    if survey == 'phangs':
        zero_mask = (img_data == 0)
    else:
        zero_mask = np.isnan(img_data)
        
    neg_at_invalid = np.sum(neg_mask & zero_mask)
    neg_at_valid   = np.sum(neg_mask & ~zero_mask)
    
    print(f"\nNegatives at invalid input pixels (zero/NaN): {neg_at_invalid} ({neg_at_invalid/n_neg*100:.1f}%)")
    print(f"Negatives at valid input pixels:              {neg_at_valid} ({neg_at_valid/n_neg*100:.1f}%)")
    
    # --- Check 3: Magnitude relative to noise ---
    # Estimate background noise from valid pixels
    if survey == 'phangs':
        valid_data = convolved_img[img_data != 0]
    else:
        valid_data = convolved_img[~np.isnan(img_data)]
        
    noise = np.nanstd(valid_data[valid_data < np.nanpercentile(valid_data, 10)])
    print(f"\nEstimated noise level: {noise:.6e}")
    print(f"Negatives within 3-sigma of noise: {np.sum(convolved_img < -3*noise)}")
    print(f"Negatives within 1-sigma of noise: {np.sum(convolved_img < -1*noise)}")
    print(50*'-')

# Convolution function

def create_convolvedFITS(original_fits, kernel_fits, output_dir, GAL_NAME=False):
    """
    Applies a homogenization kernel to an image using FFT-based convolution.
    Handles NaN and zero regions correctly for both PHANGS and IRAC images.
    Saves the convolved image as a new FITS file.
    """
    output_dir = Path(output_dir).expanduser()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with fits.open(kernel_fits) as hdu_k, fits.open(original_fits) as hdu_i:
        kernel_data = np.nan_to_num(hdu_k[0].data)
        img_data    = hdu_i[0].data
        img_header  = hdu_i[0].header

    # Validate kernel before convolution
    if np.sum(kernel_data) == 0:
        raise ValueError(f"Kernel {kernel_fits} has zero sum — invalid kernel!")
    kernel_norm = kernel_data / np.sum(kernel_data)

    # Kernel size needed for border mask expansion
    kernel_size = kernel_data.shape[0]

    original_file_name = original_fits.name
    info = original_file_name.split('_')

    if 'phangs-hst' in info:
        convolved_img = convolve_phangs(img_data, kernel_norm, kernel_size)
        gal_name, survey, filt = info[4].lower(), 'phangs', info[5].lower()
        if 'mosaic' in gal_name:
            gal_name = gal_name.replace('mosaic', '')

    elif 's4g' in original_file_name:
        convolved_img = convolve_irac(img_data, kernel_norm, kernel_size)
        gal_name, survey, filt = info[0].lower(), 's4g', info[2].lower()

    else:
        raise ValueError(f"Unrecognized survey for file: {original_file_name}")

    # --- Validate output ---
    # Only warn if convolution significantly increased negative pixels
    # Pre-existing negatives from sky subtraction are physically valid
    n_neg_before = np.sum(img_data < 0)
    n_neg_after  = np.sum(convolved_img < 0)

    if n_neg_after > n_neg_before * 1.5:
        print(f"WARNING: convolution increased negative pixels in {filt}!")
        print(f"  Before convolution: {n_neg_before}")
        print(f"  After convolution:  {n_neg_after}")
        print(f"  Ratio: {n_neg_after/max(n_neg_before,1):.2f}x")

    # Save convolved FITS
    convolved_fits = fits.PrimaryHDU(data=convolved_img, header=img_header)
    print(100*'#' + f'\nConvolving {filt} filter from {survey} survey:')

    output_path = os.path.join(output_dir, gal_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    out_file = os.path.join(output_path, f'{gal_name}_{survey}_{filt}_convolved.fits')
    convolved_fits.writeto(out_file, overwrite=True)
    print(f'Convolved FITS saved to: {out_file}\n' + 100*'#')

    if GAL_NAME:
        return gal_name





# -------------------------------------------------------------------------------------------------------
# def create_convolvedFITS(original_fits, kernel_fits, output_dir, GAL_NAME=False):
#     """
#     Applies the convolution kernel to an image using FFT (Fast Fourier Transform).
#     Standardizes data to handle NaNs and saves the degraded resolution image.
#     """
#     output_dir = Path(output_dir).expanduser()
#     if not os.path.exists(output_dir): os.makedirs(output_dir)

#     with fits.open(kernel_fits) as hdu_k, fits.open(original_fits) as hdu_i:
#         kernel_data = np.nan_to_num(hdu_k[0].data)
#         img_data = hdu_i[0].data
#         img_header = hdu_i[0].header

#     # Normalize kernel to preserve flux
#     if np.sum(kernel_data) != 0:
#         kernel_norm = kernel_data / np.sum(kernel_data)

#     original_file_name = original_fits.name
#     info = original_file_name.split('_')
    
#     # Identify survey to apply specific NaN handling/naming
#     if 'phangs-hst' in info:
#         convolved_img = fftconvolve(img_data, kernel_norm, mode='same')
#         gal_name, survey, filt = info[4].lower(), 'phangs', info[5].lower()
#         if 'mosaic' in gal_name:
#             gal_name = gal_name.replace('mosaic', '')
#     elif 's4g' in original_file_name:
#         # IRAC images often have NaNs that break FFT; convert to zero
#         img_data_limpa = np.nan_to_num(img_data, nan=0.0)
#         convolved_img = fftconvolve(img_data_limpa, kernel_norm, mode='same')
#         gal_name, survey, filt = info[0].lower(), 's4g', info[2].lower()
    
#     convolved_fits = fits.PrimaryHDU(data=convolved_img, header=img_header)
#     print(100*'#' + f'\nConvolving {filt} filter from {survey} survey:')

#     output_path = os.path.join(output_dir, gal_name)
#     if not os.path.exists(output_path): os.makedirs(output_path)

#     out_file = os.path.join(output_path, f'{gal_name}_{survey}_{filt}_convolved.fits')
#     convolved_fits.writeto(out_file, overwrite=True)
#     print(f'Convolved FITS saved to: {out_file}\n' + 100*'#')

#     if GAL_NAME: return gal_name
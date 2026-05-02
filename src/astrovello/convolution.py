import numpy as np
import os
from astropy.io import fits
import shutil
from scipy.signal import fftconvolve
from collections import defaultdict
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------
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
    Standardizes PSF headers for PyPHER compatibility.
    Calculates pixel scales and ensures correct centering and coordinate keywords.
    """
    if 'WFC3UV' in input_file:
        # HST scale: native 0.04"/pix, oversampled 4x
        pixel_scale_arcsec = 0.0395 / 4.0 
        pixel_scale_deg = pixel_scale_arcsec / 3600.0

        with fits.open(input_file, ignore_missing_end=True) as hdu:
            # Average the PSF cube to get a 2D representative PSF
            data_2d = np.mean(hdu[0].data, axis=0)
            new_hdu = fits.PrimaryHDU(data_2d)
            
            # Inject WCS keywords required by PyPHER/Astropy
            new_hdu.header.update({
                'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                'CRVAL1': 0.0, 'CRVAL2': 0.0,
                'CRPIX1': 51.0, 'CRPIX2': 51.0,
                'CDELT1': -pixel_scale_deg, 'CDELT2': pixel_scale_deg,
                'PIXSCALE': pixel_scale_arcsec
            })
            new_hdu.writeto(output_file, overwrite=True)
            print(f"==> File ready to be applied in PyPHER: {os.path.basename(output_file)}")

    elif any(x in input_file for x in ['IRAC1', 'IRAC2']):
        # Spitzer scale: native ~1.22"/pix, oversampled 5x
        pixel_scale_arcsec = 1.221/5.0 if 'IRAC1' in input_file else 1.213/5.0
        pixel_scale_deg = pixel_scale_arcsec / 3600.0

        with fits.open(input_file) as hdu:
            data_raw = hdu[0].data
            # Force odd parity: PyPHER prefers kernels/PSFs with an odd number of pixels
            data_2d = data_raw[:-1, :-1] if data_raw.shape[0] % 2 == 0 else data_raw

            new_hdu = fits.PrimaryHDU(data_2d)
            new_hdu.header.update({
                'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                'CRVAL1': 0.0, 'CRVAL2': 0.0,
                'CRPIX1': (data_2d.shape[1] // 2) + 1, 'CRPIX2': (data_2d.shape[0] // 2) + 1,
                'CDELT1': -pixel_scale_deg, 'CDELT2': pixel_scale_deg,
                'PIXSCALE': pixel_scale_arcsec
            })
            new_hdu.writeto(output_file, overwrite=True)
            print(f"==> File ready to be applied in PyPHER: {os.path.basename(output_file)}")

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

def create_convolvedFITS(original_fits, kernel_fits, output_dir, GAL_NAME=False):
    """
    Applies the convolution kernel to an image using FFT (Fast Fourier Transform).
    Standardizes data to handle NaNs and saves the degraded resolution image.
    """
    output_dir = Path(output_dir).expanduser()
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    with fits.open(kernel_fits) as hdu_k, fits.open(original_fits) as hdu_i:
        kernel_data = np.nan_to_num(hdu_k[0].data)
        img_data = hdu_i[0].data
        img_header = hdu_i[0].header

    # Normalize kernel to preserve flux
    if np.sum(kernel_data) != 0:
        kernel_norm = kernel_data / np.sum(kernel_data)

    original_file_name = original_fits.name
    info = original_file_name.split('_')
    
    # Identify survey to apply specific NaN handling/naming
    if 'phangs-hst' in info:
        convolved_img = fftconvolve(img_data, kernel_norm, mode='same')
        gal_name, survey, filt = info[4].lower(), 'phangs', info[5].lower()
        if 'mosaic' in gal_name:
            gal_name = gal_name.replace('mosaic', '')
    elif 's4g' in original_file_name:
        # IRAC images often have NaNs that break FFT; convert to zero
        img_data_limpa = np.nan_to_num(img_data, nan=0.0)
        convolved_img = fftconvolve(img_data_limpa, kernel_norm, mode='same')
        gal_name, survey, filt = info[0].lower(), 's4g', info[2].lower()
    
    convolved_fits = fits.PrimaryHDU(data=convolved_img, header=img_header)
    print(100*'#' + f'\nConvolving {filt} filter from {survey} survey:')

    output_path = os.path.join(output_dir, gal_name)
    if not os.path.exists(output_path): os.makedirs(output_path)

    out_file = os.path.join(output_path, f'{gal_name}_{survey}_{filt}_convolved.fits')
    convolved_fits.writeto(out_file, overwrite=True)
    print(f'Convolved FITS saved to: {out_file}\n' + 100*'#')

    if GAL_NAME: return gal_name


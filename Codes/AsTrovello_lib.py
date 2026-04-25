from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import numpy as np
import psutil
import shutil
import gc
from tqdm import tqdm  
from pathlib import Path
import matplotlib.pyplot as plt
import os 
from scipy.signal import fftconvolve
from collections import defaultdict

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Image alignment -------------------------------------------------------

# Reproject S4G (Spitzer) images onto the PHANGS (HST) pixel grid
def S4G2PHANGS_reproject(s4g_file_path, phangs_ref_file_path, output_path):
    """
    Aligns a Spitzer/IRAC image to the HST pixel grid.
    Returns an array with the same spatial dimensions as the PHANGS reference.
    """
    hdu_phangs = fits.open(phangs_ref_file_path)[0]
    hdu_s4g = fits.open(s4g_file_path)[0]

    sci_file_s4g = s4g_file_path.name
    # Extract galaxy name and filter index (e.g., IRAC1) from filename
    galaxy_name, filter_mode = sci_file_s4g.split('.')[0].lower(), sci_file_s4g.split('.')[-2]

    # Initialize WCS (World Coordinate System) for both images
    w_phangs = WCS(hdu_phangs.header)
    w_s4g = WCS(hdu_s4g.header)
    
    # Force SIP (Simple Imaging Polynomial) correction type for Spitzer headers
    w_s4g.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    print('\n' + 100*'#' + '\nSIP correction added to the header!\n' + 100*'#')

    # Perform the interpolation/reprojection using reproject_interp
    # Surface brightness is preserved, but flux per pixel is not strictly conserved due to resampling
    array, footprint = reproject_interp((hdu_s4g.data, w_s4g), w_phangs, shape_out=hdu_phangs.data.shape)

    s4g_new_header = hdu_s4g.header.copy()

    # Generate new WCS keywords based on the HST reference
    wcs_phangs_header = w_phangs.to_header(relax=True)

    # Clean old WCS keywords to prevent coordinate conflicts (especially CD vs PC matrices)
    wcs_keys_to_remove = [
        'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2',
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'
    ]
    for key in wcs_keys_to_remove:
        if key in s4g_new_header:
            del s4g_new_header[key]

    # Merge the new WCS information into the S4G header
    s4g_new_header.update(wcs_phangs_header)
    s4g_new_header['CTYPE1'] = 'RA---TAN-SIP'
    s4g_new_header['CTYPE2'] = 'DEC--TAN-SIP'
    s4g_new_header['COMMENT'] = 'Reprojected to PHANGS grid. Flux not conserved per pixel, surface brightness preserved.'

    # Setup output directory
    output_path = os.path.expanduser(output_path)
    output_directory = os.path.join(output_path, galaxy_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        print(f"📁 Directory created: {output_directory}")

    output_name = f'{galaxy_name}_s4g_irac{filter_mode}_on_phangs_projection.fits'
    fits.writeto(os.path.join(output_directory, output_name), array, s4g_new_header, overwrite=True)
    print('\n' + 100*'#' + f'\nReprojected FITS file: {output_name}\n' + 100*'#')


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
            print(f"✅ File ready to be applied in PyPHER: {os.path.basename(output_file)}")

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
            print(f"✅ File ready to be applied in PyPHER: {os.path.basename(output_file)}")

def pypher_kernel_creation(todos_fwhm, psf_master_path, input_dir, output_dir):
    """
    Prepares a list of shell commands for the PyPHER library to generate homogenization kernels.
    It identifies which PSF belongs to which survey to locate files in the 'PSF_LIMPAS' folders.
    """
    psf_path_phangs_clean = input_dir / 'PHANGS' / 'PSF_LIMPAS'
    psf_path_s4g_clean = input_dir / 'S4G' / 'PSF_LIMPAS'
    psf_master_name = psf_master_path.stem.split('_')[0].lower()

    if os.path.exists(output_dir):
        print(f"🗑️  Removing previous directory: {os.path.basename(output_dir)}")
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


# -------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- FITS unit conversion-----------------------------------------------------

def convert2Jansky(fits_file):
    """
    Converts image units to Jansky per pixel.
    Handles HST (Electrons/s to Jy via PHOTFNU) and Spitzer (MJy/sr to Jy/pixel via pixel area).
    """
    with fits.open(fits_file) as hdu:
        data, header = hdu[0].data, hdu[0].header
    
    new_data, new_header = data.copy(), header.copy()

    # HST Case: Using Photometric Flux density (Jy*s/e)
    if header.get('BUNIT') == 'ELECTRONS/S':
        new_data *= header['PHOTFNU']
        new_header['BUNIT'] = 'Jy/pixel'
        print("HST: Converted using PHOTFNU")

    # Spitzer Case: Surface brightness to flux per pixel
    elif header.get('BUNIT') == 'MJy/sr': 
        pixel_area_arcsec2 = np.abs(header['PXSCAL1'] * header['PXSCAL2'])
        # 1 arcsec² ≈ 2.35e-11 steradians
        pixel_area_sr = pixel_area_arcsec2 * 2.3504430539e-11
        # Convert MJy -> Jy (1e6) and sr -> pixel area
        new_data = new_data * 1e6 * pixel_area_sr
        new_header['BUNIT'] = 'Jy/pixel'
        print(f"S4G: Converted using pixel area ({pixel_area_arcsec2} arcsec2)")

    return new_data, new_header
    
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

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Data Cube creation-------------------------------------------------------

def create_data_cube(aligned_images, filter_names, ref_file, ref_header, output_filename, 
                     aplicar_mask=True, N_SIGMA=3, padding=50):
    """
    Constructs a 3D FITS Hypercube (RA, DEC, Filter).
    Includes automatic sky masking, background subtraction, and Bounding Box cutout.
    Updates WCS to 3D.
    """
    print(f'\nInitiating hypercube creation (N_SIGMA={N_SIGMA})...')
    ny, nx = aligned_images[0].shape
    cubo = np.empty((len(filter_names), ny, nx), dtype=np.float32)
    
    # 1. Determine final processing mask (Signal-based or Border-based)
    if aplicar_mask:
        summed = soma_img(aligned_images, ref_file)
        _, mask_final = mask(summed, N_SIGMA=N_SIGMA)
    else:
        mask_final = phangs_intersection_mask(ref_file)

    # 2. Fill the cube layers, setting non-mask regions to NaN
    for i, img_atual in enumerate(aligned_images):
        cubo[i, :, :] = np.where(mask_final, img_atual, np.nan) if mask_final is not None else img_atual

    # 3. Bounding Box Cutout: Shrink the cube to the relevant area plus padding
    y_off, x_off = 0, 0
    if mask_final is not None:
        coords = np.argwhere(mask_final)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
            y_min, y_max = max(0, y_min - padding), min(ny, y_max + padding)
            x_min, x_max = max(0, x_min - padding), min(nx, x_max + padding)
            cubo = cubo[:, y_min:y_max, x_min:x_max]
            y_off, x_off = y_min, x_min
            print(f"✂️ Bounding box cutout: {ny}x{nx} -> {cubo.shape[1]}x{cubo.shape[2]}")

    # 4. Construct 3D WCS Header
    # Adjust Reference Pixels (CRPIX) to reflect the BBox shift
    w_2d = WCS(ref_header, naxis=2)
    w_3d = WCS(naxis=3)
    for i in [0, 1]:
        for p in ['crpix', 'crval', 'cdelt', 'ctype', 'cunit']:
            try:
                val = getattr(w_2d.wcs, p)[i]
                if p == 'crpix': getattr(w_3d.wcs, p)[i] = val - (x_off if i == 0 else y_off)
                else: getattr(w_3d.wcs, p)[i] = val
            except: continue
    
    # Set the 3rd axis (Filters)
    w_3d.wcs.crpix[2], w_3d.wcs.crval[2], w_3d.wcs.cdelt[2], w_3d.wcs.ctype[2] = 1, 0, 1, 'FILTER'
    cube_header = w_3d.to_header()
    cube_header['BUNIT'] = 'Jy/pixel'
    for i, filt in enumerate(filter_names): cube_header[f'FILT{i+1:03d}'] = filt
        
    fits.writeto(output_filename, cubo, header=cube_header, overwrite=True)
    return cubo, cube_header

def create_cutouts(cube, cube_header, regions):
    """Extracts spatial sub-cubes (zooms) from the main hypercube."""
    for path, x_start, x_end, y_start, y_end in regions:
        if x_end > cube.shape[2] or y_end > cube.shape[1] or x_start < 0 or y_start < 0:
            print(f"Warning: Zoom '{path.name}' out of bounds. Continuing.")
            continue
        
        cube_cut = cube[:, y_start:y_end, x_start:x_end]
        cut_h = cube_header.copy()
        cut_h['CRPIX1'] -= x_start
        cut_h['CRPIX2'] -= y_start
        fits.writeto(path, cube_cut, header=cut_h, overwrite=True)
        print(f"🔎 Cutout saved to: {path.name}")

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Utility Functions -----------------------------------------------------

def get_filters(file_list, start, position):
    """Extracts unique filter names from a list of filenames based on naming conventions."""
    return list(set(f.split('_')[position] for f in file_list if f.startswith(start)))

def log_memory_usage():
    """Prints the current resident set size (RSS) memory consumption of the script."""
    process = psutil.Process(os.getpid())
    print(f"Uso de memória: {process.memory_info().rss / 1024 ** 2:.2f} MB")
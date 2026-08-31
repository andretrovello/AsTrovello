"""
Convolution functions for the AsTrovello pipeline.

"""

from pathlib import Path
import numpy as np
import pypher
import warnings
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.modeling import models, fitting
from astropy.nddata import block_reduce
import os
import shutil

def get_fwhm(data: np.ndarray) -> float:
    """
    Estimates the Full Width at Half Maximum (FWHM) using a 2D Gaussian fit.
    
    This method is robust against background noise, negative pixels, and large 
    image bounding boxes. It utilizes the Levenberg-Marquardt least squares 
    algorithm to fit a 2D Gaussian model to the data. To account for slightly 
    elliptical PSFs, the effective sigma is calculated as the geometric mean 
    of the X and Y standard deviations.

    Args:
        data (np.ndarray): A 2D array representing the Point Spread Function (PSF) image.

    Returns:
        float: The estimated FWHM measured in pixels.
    """
    # 1. Remove qualquer NaN que possa quebrar o algoritmo
    data_clean = np.nan_to_num(data, nan=0.0)
    
    # 2. Cria uma malha de coordenadas X e Y do mesmo tamanho da imagem
    y, x = np.mgrid[:data_clean.shape[0], :data_clean.shape[1]]
    
    # 3. Estima os parâmetros iniciais (chutes) para ajudar o algoritmo a convergir mais rápido
    max_val = np.max(data_clean)
    y_center, x_center = np.unravel_index(np.argmax(data_clean), data_clean.shape)
    
    # Cria o modelo inicial da Gaussiana
    g_init = models.Gaussian2D(amplitude=max_val, x_mean=x_center, y_mean=y_center, 
                               x_stddev=2.0, y_stddev=2.0)
    
    # 4. Inicializa o algoritmo de ajuste (Levenberg-Marquardt Mínimos Quadrados)
    fit_g = fitting.LevMarLSQFitter()
    
    # 5. Ajusta o modelo aos dados
    with warnings.catch_warnings():
        # Ignora avisos inofensivos do astropy caso a PSF seja muito ruidosa
        warnings.simplefilter('ignore')
        g_fit = fit_g(g_init, x, y, data_clean)
    
    # 6. Extrai o desvio padrão (sigma) do eixo X e Y e tira a média geométrica
    # A média geométrica lida melhor com PSFs ligeiramente elípticas
    sigma_eff = np.sqrt(abs(g_fit.x_stddev.value * g_fit.y_stddev.value))
    
    # 7. Converte Sigma para FWHM em pixels
    fwhm_pixels = 2.3548 * sigma_eff
    
    return float(fwhm_pixels)


def calculateFWHM(psf_file_list: list, drivers: dict) -> tuple[dict, list]:
    """    
    Iterates through a folder of PSF files, filters them by survey,
    and returns dictionaries containing their physical FWHM (in arcsec).

    Determines to which survey the file belongs based on its parent directory
    and extracts the filter name. Then, determines the PSF's binning factor 
    and its correct pixel scale from the SURVEY_CONFIG dictionary. Finally, 
    calculates the FWHM for each filter and returns a FWHM dictionary and a 
    list with valid file names. 

    Args:
        psf_file_list (list): List of PSF files paths.
        SURVEY_CONFIG (dict): Survey configurations dictionary.

    Returns:
        tuple: A tuple containing:
            - FWHM_dict (dict): Dictionary with filter names as keys and FWHM (in arcsec) as values.
            - valid_files (list): List of valid file names (where FWHM was successfully calculated).

    Note:
        For 3D PSF files, the FWHM is calculated over the mean PSF of the cube.
    """
    FWHM_dict, valid_files = {}, []
    
    # Silencia os avisos chatos de cabeçalho do Astropy
    warnings.simplefilter('ignore', category=AstropyWarning)
    
    for file in psf_file_list:
        if file.name.startswith('.'):
            continue
        
        str_file = str(file)
        survey = drivers["BASE"].get_survey(file_path = str_file)
        if not survey or survey not in drivers:
            continue

        driver = drivers[survey]

        filter_name = driver.get_psf_filter_name(filename = str_file)
        psf_pixscale = driver.get_psf_pixel_scale(filter_name = filter_name)

        try:
            with fits.open(file, ignore_missing_end=True, ignore_missing_simple=True) as hdu:
                data = next((h.data for h in hdu if h.data is not None), None)
                
                if data is not None:
                    if data.ndim == 3: 
                        data = np.mean(data, axis=0) 
                    
                    # Usa o ajuste Gaussiano (ou a função robusta) que retorna em pixels
                    fwhm_pixels = get_fwhm(data)
                    
                    # Converte para escala física (arcsec) usando a escala de pixel superamostrada
                    FWHM_dict[filter_name] = np.float32(fwhm_pixels * psf_pixscale)
                    
                    valid_files.append(file.name)
                    print(f"Successfully read: {filter_name} (FWHM: {FWHM_dict[filter_name]:.4f} arcsec)")
                    
        except Exception as e:
            print(f"Processing error {file.name}: {e}")
            
    # Restaura os avisos para o resto do seu código
    warnings.simplefilter('default', category=AstropyWarning)
    
    return FWHM_dict, valid_files


def clean_psf(input_file: str, output_file: str, pixel_scale_arcsec: float, binned_factor: int):
    """Standardizes PSF headers and performs true downsampling for PyPHER compatibility.
    
    Agnostic function that calculates pixel scales, bins down oversampled PSFs, 
    forces odd parity, normalizes flux, and ensures correct centering and WCS keywords.

    Args:
        input_file (str): Path to the input PSF FITS file.
        output_file (str): Path to save the cleaned PSF.
        pixel_scale_arcsec (float): Native pixel scale of the instrument.
        binned_factor (int): Factor by which the PSF is oversampled.
        is_3d (bool): If True, averages the data along axis 0 to create a 2D representation.
    """
    pixel_scale_deg = pixel_scale_arcsec / 3600.0

    with fits.open(input_file, ignore_missing_end=True, ignore_missing_simple=True) as hdu:
        data = next((h.data for h in hdu if h.data is not None), None)
        
        if data is None:
            print(f"==> Error: No valid data found in {input_file}")
            return

        # 1. Trata cubos 3D (ex: WFC3 do Hubble)
        if data.ndim == 3:
            data = np.mean(data, axis=0)

        # 2. Aplica o downsampling apenas se o fator for maior que 1
        if binned_factor > 1:
            data_processed = block_reduce(data, block_size=binned_factor, func=np.sum)
        else:
            data_processed = data.copy()

        # 3. Força paridade ímpar para o PyPHER
        if data_processed.shape[0] % 2 == 0:
            data_processed = data_processed[:-1, :-1]
            
        # 4. Normaliza para assegurar conservação de fluxo
        data_processed = data_processed / np.sum(data_processed)

        # 5. Criação do novo FITS e injeção do WCS
        new_hdu = fits.PrimaryHDU(data_processed)
        new_hdu.header.update({
            'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
            'CRVAL1': 0.0, 'CRVAL2': 0.0,
            'CRPIX1': (data_processed.shape[1] // 2) + 1, 'CRPIX2': (data_processed.shape[0] // 2) + 1,
            'CDELT1': -pixel_scale_deg, 'CDELT2': pixel_scale_deg,
            'PIXSCALE': pixel_scale_arcsec
        })
        
        new_hdu.writeto(output_file, overwrite=True)
        print(f"\tFile ready for PyPHER (Binned {binned_factor}x): {os.path.basename(output_file)}")

def pypher_kernel_creation(cleaned_psf_by_filter: dict, psf_master_name: str, output_dir: Path) -> list:
    """
    Builds the PyPHER shell commands to homogenize every non-master PSF
    onto the master's resolution.

    Parameters
    ----------
    cleaned_psf_by_filter : dict of {str : Path}
        Filter name -> path to its cleaned PSF file (as produced by clean_psf).
    psf_master_name : str
        Filter name of the PSF chosen as the homogenization target.
    output_dir : Path
        Directory where the generated kernels are written (recreated fresh).

    Returns
    -------
    list of str
        One 'pypher <source_psf> <master_psf> <kernel_path>' command per non-master filter.
    """
    psf_master_path = cleaned_psf_by_filter[psf_master_name]

    if output_dir.exists():
        print(f">>> Removing previous directory: {output_dir.name}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    pypher_commands = []
    for filt, psf_path in cleaned_psf_by_filter.items():
        if filt == psf_master_name:
            continue
        kernel_name = output_dir / f"kernel_{filt}_to_{psf_master_name}.fits"
        pypher_commands.append(f"pypher {psf_path} {psf_master_path} {kernel_name}")

    return pypher_commands

def convolved_dict(img_files: list, kernel_files: list, drivers: dict) -> dict:
    """
    Pairs each science image with its homogenization kernel, indexed by filter.

    Parameters
    ----------
    img_files : list of Path
        Science images to be convolved.
    kernel_files : list of Path
        Kernel FITS files generated by `pypher_kernel_creation`.
    drivers : dict
        Registry of driver instances, used to resolve each image's filter
        name via its own survey's naming convention.

    Returns
    -------
    dict of {str : dict}
        ``{filter_name: {'img': Path, 'kernel': Path, 'survey': str}}``. The
        master filter is naturally excluded (it has no kernel) — the caller
        must copy the master image over as-is, without convolution.
    """
    conv_dict = {}
    for img_path in img_files:
        survey = drivers["BASE"].get_survey(file_path=img_path)
        filt = drivers[survey].get_sci_filter_name(filename=str(img_path))
        conv_dict.setdefault(filt, {})['img'] = img_path
        conv_dict.setdefault(filt, {})['survey'] = survey   

    for kernel_path in kernel_files:
        filt = kernel_path.stem.split('_')[1]   # kernel_{filt}_to_{master}.fits
        conv_dict.setdefault(filt, {})['kernel'] = kernel_path

    complete, incomplete = {}, {}
    for filt, paths in conv_dict.items():
        if 'img' in paths and 'kernel' in paths:
            complete[filt] = paths
        else:
            incomplete[filt] = paths

    if incomplete:
        print(f"==> Note: {list(incomplete.keys())} have no kernel pair "
              f"(expected only for the master filter — double-check if others appear here).")

    return complete

def diagnose_negatives(convolved_img, img_data, filt, survey, driver):
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

    print(f"\nNegatives in border region:   {neg_in_border} ({neg_in_border/max(n_neg,1)*100:.1f}%)")
    print(f"Negatives in interior region: {neg_in_interior} ({neg_in_interior/max(n_neg,1)*100:.1f}%)")

    # --- Check 2: Are negatives where input was invalid (zero/NaN, per survey convention)? ---
    invalid_mask = driver.get_invalid_mask(img_data)

    neg_at_invalid = np.sum(neg_mask & invalid_mask)
    neg_at_valid   = np.sum(neg_mask & ~invalid_mask)

    print(f"\nNegatives at invalid input pixels: {neg_at_invalid} ({neg_at_invalid/max(n_neg,1)*100:.1f}%)")
    print(f"Negatives at valid input pixels:   {neg_at_valid} ({neg_at_valid/max(n_neg,1)*100:.1f}%)")

    # --- Check 3: Magnitude relative to noise ---
    valid_data = convolved_img[~invalid_mask]

    noise = np.nanstd(valid_data[valid_data < np.nanpercentile(valid_data, 10)])
    print(f"\nEstimated noise level: {noise:.6e}")
    print(f"Negatives within 3-sigma of noise: {np.sum(convolved_img < -3*noise)}")
    print(f"Negatives within 1-sigma of noise: {np.sum(convolved_img < -1*noise)}")
    print(50*'-')

def create_convolvedFITS(original_fits: Path, kernel_fits: Path, 
                         survey: str, psf_master_name: str,
                         output_dir: Path, drivers: dict, 
                         force:bool = False) -> Path:
    
    driver = drivers[survey]
    original_file_name = original_fits.name if hasattr(original_fits, 'name') else os.path.basename(original_fits)
    gal_name = driver.get_galaxy_name(original_file_name)
    filt = driver.get_sci_filter_name(original_file_name)

    output_path = output_dir / gal_name
    out_file = output_path / f'{gal_name}_{filt}_to_{psf_master_name}_convolved.fits'

    if out_file.exists() and not force:
        print(f">>> Already convolved, skipping: {out_file.name}")
        return out_file

    output_path.mkdir(parents=True, exist_ok=True)

    with fits.open(kernel_fits) as hdu_k, fits.open(original_fits) as hdu_i:
        kernel_data = np.nan_to_num(hdu_k[0].data)
        img_data    = hdu_i[0].data
        img_header  = hdu_i[0].header

    if np.sum(kernel_data) == 0:
        raise ValueError(f"Kernel {kernel_fits} has zero sum — invalid kernel!")

    kernel_norm = kernel_data / np.sum(kernel_data)
    kernel_size = kernel_data.shape[0]

    convolved_img = driver.convolve(img_data, kernel_norm, kernel_size)

    n_neg_before = np.sum(img_data < 0)
    n_neg_after  = np.sum(convolved_img < 0)
    if n_neg_after > n_neg_before * 1.5:
        print(f">>> WARNING: Convolution increased negative pixels in {filt}!")
        diagnose_negatives(convolved_img, img_data, filt, survey, driver)

    convolved_fits = fits.PrimaryHDU(data=convolved_img, header=img_header)
    print(200*'-' + f'\n>>> Convolving {filt} filter from {survey} survey:')
    convolved_fits.writeto(out_file, overwrite=True)
    print(f'\tConvolved FITS saved to: {out_file}\n' + 100*'-')

    return out_file
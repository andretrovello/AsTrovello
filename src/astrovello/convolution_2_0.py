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

        filter_name = driver.get_filter_name(filename = str_file)
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
        print(f"==> File ready for PyPHER (Binned {binned_factor}x): {os.path.basename(output_file)}")
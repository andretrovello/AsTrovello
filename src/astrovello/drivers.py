"""
Survey drivers for the AsTrovello pipeline.

`Base_Driver` implements generic behavior shared by every survey,
reading its parameters from ``SURVEY_CONFIG`` in :mod:`config`. Each
survey gets a thin subclass that points at its own config entry, and
can override any method individually if a survey's data needs
genuinely different logic rather than just different constants.

Examples
--------
>>> driver = get_driver('PHANGS')
>>> galaxy, filt = driver.parse_filename('ngc2903_mosaic_f814w.fits')
>>> driver.get_pixel_scale(filt)
0.0395
"""

from pathlib import Path
import numpy as np
from scipy.ndimage import label, binary_dilation
from astropy.convolution import convolve_fft

# ================================= Base Class =================================
class BASE_Driver:
    def __init__(self, config_dict: dict):
        self.config = config_dict

    def get_files(self, dir_path: Path, mode: str) -> list:
        suffix_key = f"{mode}_glob"
        
        if suffix_key in self.config:
            # O f-string no glob permite injetar variáveis caso o sufixo use o nome da galáxia!
            pattern = self.config[suffix_key]
            return list(dir_path.glob(pattern))
            
        raise ValueError(f"Mode '{mode}' not configured for {self.__class__.__name__}.")

    def get_survey(self, file_path: Path) -> str:
        AVAILABLE_SURVEYS = self.config.keys()
        for survey in AVAILABLE_SURVEYS:
            if survey in str(file_path):
                return survey

    def get_pixel_scale(self, filter_name: str) -> float:
        # Padrão genérico: se for um valor simples no dicionário, já resolve aqui no pai!
        return self.config["pixel_scale_arcsec"]

    @property
    def get_binned_factor(self) -> int:
        return self.config.get("binned_factor", 1)

    def get_psf_pixel_scale(self, filter_name: str) -> float:
    # Pega a escala nativa do survey/canal e divide pelo binned_factor
        raw_scale = self.get_pixel_scale(filter_name)
        return raw_scale / self.get_binned_factor

    def convolve(self, img_data: np.ndarray, kernel: np.ndarray, kernel_size: int) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} must implement convolve().")

    def get_galaxy_name(self, filename: str) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_galaxy_name().")

    def get_invalid_mask(self, img_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_invalid_mask().")



# ================================= PHANGS Class =================================
class PHANGS_Driver(BASE_Driver):
    """Herda get_files e get_pixel_scale de BaseDriver."""
    
    def get_filter_name(self, filename: str) -> str:
        return filename.replace('.fits', '').split('_')[-1].lower()

    def get_galaxy_name(self, filename: str) -> str:
        gal_name = Path(filename).name.split('_')[4].lower()
        return gal_name.replace('mosaic', '')

    def convolve(self, img_data: np.ndarray, kernel: np.ndarray, kernel_size: int) -> np.ndarray:
        img_nan = img_data.copy().astype(float)
        img_nan[img_data == 0] = np.nan
        nan_mask = np.isnan(img_nan)

        border_seed = np.zeros_like(nan_mask)
        border_seed[0, :]  = nan_mask[0, :]
        border_seed[-1, :] = nan_mask[-1, :]
        border_seed[:, 0]  = nan_mask[:, 0]
        border_seed[:, -1] = nan_mask[:, -1]

        labeled, _ = label(nan_mask)
        border_labels = set(labeled[border_seed & nan_mask])
        border_mask = np.isin(labeled, list(border_labels))

        img_to_conv = img_nan.copy()
        img_to_conv[border_mask] = 0.0

        convolved_img = convolve_fft(
            img_to_conv, kernel,
            normalize_kernel=False, nan_treatment='interpolate',
            preserve_nan=False, allow_huge=True,
        )

        structure = np.ones((kernel_size, kernel_size))
        expanded_border = binary_dilation(border_mask, structure=structure)
        convolved_img[expanded_border] = 0.0

        return convolved_img
    
    def get_invalid_mask(self, img_data: np.ndarray) -> np.ndarray:
        return img_data == 0



# ================================= PHANGS Class =================================
class S4G_Driver(BASE_Driver):
    def get_filter_name(self, filename: str) -> str:
        if 'IRAC1' in filename: return 'irac1'
        if 'IRAC2' in filename: return 'irac2'
        return 'unknown'

    def get_galaxy_name(self, filename: str) -> str:
        return Path(filename).name.split('_')[0].lower()

    def get_pixel_scale(self, filter_name: str) -> float:
        channel = 1 if filter_name == 'irac1' else 2
        return self.config["pixel_scale_arcsec"][channel]

    def convolve(self, img_data: np.ndarray, kernel: np.ndarray, kernel_size: int) -> np.ndarray:
        nan_mask_original = np.isnan(img_data)

        convolved_img = convolve_fft(
            img_data, kernel,
            normalize_kernel=False, nan_treatment='fill', fill_value=0.0,
            preserve_nan=False, allow_huge=True,
        )

        structure = np.ones((kernel_size, kernel_size))
        expanded_nan_mask = binary_dilation(nan_mask_original, structure=structure)
        convolved_img[expanded_nan_mask] = np.nan

        return convolved_img
    
    def get_invalid_mask(self, img_data: np.ndarray) -> np.ndarray:
        return np.isnan(img_data)
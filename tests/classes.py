# ======================================== Libraries ==========================================
from pathlib import Path

# ======================================== Base Driver ========================================
# 1. A CLASSE PAI (Guarda o que é GENÉRICO para todos os surveys)
class BASE_Driver:
    """
    General driver for all surveys. Is inherited by each survey specific driver.
    """
    def __init__(self, config_dict: dict):
        self.config = config_dict

    def get_files(self, dir_path: Path, mode: str) -> list:
        # A busca via glob agora mora apenas AQUI no pai!
        suffix_key = f"{mode}_suffix"
        
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

# ======================================== PHANGS Driver ========================================
# 2. OS DRIVERS FILHOS (Herdam do pai e só escrevem o que for ESPECÍFICO)
class PHANGS_Driver(BASE_Driver):
    """
    PHANGS survey driver, inherits functions from BASE_Driver.
    """
    def get_filter_name(self, filename: str) -> str:
        return filename.replace('.fits', '').split('_')[-1].lower()

# ======================================== S4G Driver ========================================
class S4G_Driver(BASE_Driver):
    """Herda get_files de BaseDriver, mas sobrescreve o que é peculiar do S4G."""
    
    def get_filter_name(self, filename: str) -> str:
        if 'IRAC1' in filename: return 'irac1'
        if 'IRAC2' in filename: return 'irac2'
        return 'unknown'

    def get_pixel_scale(self, filter_name: str) -> float:
        # Sobrescreve o método do pai apenas porque o S4G tem escalas diferentes por canal!
        channel = 1 if filter_name == 'irac1' else 2
        return self.config["pixel_scale_arcsec"][channel]
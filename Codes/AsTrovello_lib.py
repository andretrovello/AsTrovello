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

# Reproject S4G on PHANGS header (returns PHANGS size array)
def S4G2PHANGS_reproject(s4g_file_path, phangs_ref_file_path, output_path = '~/Desktop/AsTrovello/Input/reprojected_files'):
    hdu_phangs = fits.open(phangs_ref_file_path)[0]
    hdu_s4g = fits.open(s4g_file_path)[0]


    sci_file_s4g = s4g_file_path.name
    galaxy_name, filter_mode = sci_file_s4g.split('.')[0].lower(), sci_file_s4g.split('.')[-2]

    w_phangs = WCS(hdu_phangs.header)
    w_s4g = WCS(hdu_s4g.header)
    
    w_s4g.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    print('\n')
    print(100*'#')
    print('SIP correction added to the header!')
    print(100*'#')


    array, footprint = reproject_interp((hdu_s4g.data, w_s4g), w_phangs, shape_out=hdu_phangs.data.shape)

    s4g_new_header = hdu_s4g.header.copy()

    # Geramos as novas keywords de WCS baseadas no PHANGS
    wcs_phangs_header = w_phangs.to_header(relax=True)


    # Removemos keywords de WCS antigas do S4G para evitar conflitos de coordenadas
    wcs_keys_to_remove = [
        'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2',
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'
    ]
    for key in wcs_keys_to_remove:
        if key in s4g_new_header:
            del s4g_new_header[key]

    s4g_new_header.update(wcs_phangs_header)

    # Garantimos que o CTYPE final reflita a natureza do dado (TAN-SIP)
    s4g_new_header['CTYPE1'] = 'RA---TAN-SIP'
    s4g_new_header['CTYPE2'] = 'DEC--TAN-SIP'
    s4g_new_header['COMMENT'] = 'Reprojected to PHANGS grid. Flux not conserved per pixel, surface brightness preserved.'

    output_path = os.path.expanduser(output_path)
    output_directory = os.path.join(output_path, galaxy_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        print(f"📁 Diretório criado: {output_directory}")

    output_name = f'{galaxy_name}_s4g_irac{filter_mode}_on_phangs_projection.fits'

    fits.writeto(os.path.join(output_directory, output_name), array, s4g_new_header, overwrite=True)
    print('\n')
    print(100*'#')
    print(f'Arquivo fits reprojetado: {output_name}')
    print(100*'#')


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Image convolution ------------------------------------------------------

def get_fwhm_simple(data):
    """
    Calcula uma estimativa do FWHM baseada no desvio padrão (momento) 
    para PSFs centralizadas.
    """
    # Criar um grid de coordenadas
    y, x = np.indices(data.shape)
    center_y, center_x = np.array(data.shape) // 2
    
    # Calcular a variância espacial ponderada pelo brilho
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    variance = np.sum(data * r**2) / np.sum(data)
    sigma = np.sqrt(variance)
    
    # FWHM = 2.355 * sigma (para uma Gaussiana)
    return 2.355 * sigma

def radial_profile(data, center):
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr

def calculaFWHM_radial_profile(files_path):
    FWHM_dict = {}
    dict_profiles = {}
    path = Path(files_path).expanduser()
    file_list = list(path.glob('*.fits'))
    
    # Lista para retornar ao main quais arquivos foram realmente validados
    valid_files = []

    for file in file_list:
        # IMPORTANTE: file.name para pegar só o nome do arquivo
        if 'S4G' in str(path):
            if file.name == 'IRAC1_col129_row129.fits':
                filter_name = 'irac1'
            elif file.name == 'IRAC2_col129_row129.fits':
                filter_name = 'irac2'
            else:
                continue
        elif 'PHANGS' in str(path):
            parts = file.name.replace('.fits', '').split('_')
            filter_name = parts[-1].lower() 
        else:
            continue

        try:
            with fits.open(file, ignore_missing_end=True) as hdu:
                data = None
                for h in hdu:
                    if h.data is not None:
                        data = h.data
                        break
                
                if data is not None:
                    if data.ndim == 3:
                        data = data[0]
                    
                    fwhm = get_fwhm_simple(data)
                    prof = radial_profile(data, (data.shape[1]//2, data.shape[0]//2))

                    FWHM_dict[filter_name] = fwhm
                    dict_profiles[filter_name] = prof
                    valid_files.append(file.name) # Guardamos o nome do arquivo funcional
                    print(f"Sucesso ao ler: {filter_name}")
                    
        except Exception as e:
            print(f"Erro ao processar {file.name}: {e}")

    return FWHM_dict, dict_profiles, valid_files

# Clean PSF to be applied on Pypher
def final_clean_psf(input_file, output_file):
    if 'WFC3UV' in input_file:
        # Fator de 4x oversampling sobre a escala nativa de 0.04"/pix
        pixel_scale_arcsec = 0.0395 / 4.0 
        # Converter para graus (que é o padrão FITS CDELT)
        pixel_scale_deg = pixel_scale_arcsec / 3600.0

        with fits.open(input_file, ignore_missing_end=True) as hdu:
            # 1. Achata o cubo (56, 101, 101) para uma imagem 2D (101, 101)
            # Usamos a média para ter a PSF representativa do campo todo
            data_2d = np.mean(hdu[0].data, axis=0)

            # 2. Cria um novo HDU com os dados 2D
            new_hdu = fits.PrimaryHDU(data_2d)
            
            # 3. Injeta as keywords que o PyPHER (e o Astropy) usam para escala
            new_hdu.header['CTYPE1'] = 'RA---TAN'
            new_hdu.header['CTYPE2'] = 'DEC--TAN'
            new_hdu.header['CRVAL1'] = 0.0
            new_hdu.header['CRVAL2'] = 0.0
            new_hdu.header['CRPIX1'] = 51.0  # Centro do 101x101
            new_hdu.header['CRPIX2'] = 51.0
            new_hdu.header['CDELT1'] = -pixel_scale_deg # RA cresce para a esquerda
            new_hdu.header['CDELT2'] = pixel_scale_deg
            
            # Keyword extra que o PyPHER costuma buscar diretamente
            new_hdu.header['PIXSCALE'] = pixel_scale_arcsec

            # 4. Salva o arquivo pronto para o combate
            new_hdu.writeto(output_file, overwrite=True)
            print(f"✅ Arquivo pronto para o PyPHER: {os.path.basename(output_file)}")

    elif any(x in input_file for x in ['IRAC1', 'IRAC2']):
        if 'IRAC1' in input_file:
            # Fator de 5x oversampling sobre a escala nativa de 1.221"/pix
            pixel_scale_arcsec = 1.221 / 5.0    
        elif 'IRAC2' in input_file:
            # Fator de 5x oversampling sobre a escala nativa de 1.213"/pix
            pixel_scale_arcsec = 1.213 / 5.0  

        # Converter para graus (que é o padrão FITS CDELT)
        pixel_scale_deg = pixel_scale_arcsec / 3600.0

        with fits.open(input_file) as hdu:
            data_raw = hdu[0].data

            # --- CORREÇÃO DE PARIDADE ---
            # Se for 128x128, vamos cortar 1 pixel para virar 127x127 (Ímpar)
            if data_raw.shape[0] % 2 == 0:
                data_2d = data_raw[:-1, :-1] # Remove a última linha e coluna
            else:
                data_2d = data_raw
            # ----------------------------

            new_hdu = fits.PrimaryHDU(data_2d)
            
            # 3. Injeta as keywords que o PyPHER (e o Astropy) usam para escala
            new_hdu.header['CTYPE1'] = 'RA---TAN'
            new_hdu.header['CTYPE2'] = 'DEC--TAN'
            new_hdu.header['CRVAL1'] = 0.0
            new_hdu.header['CRVAL2'] = 0.0
            new_hdu.header['CRPIX1'] = (data_2d.shape[1] // 2) + 1 # Centro do CCD
            new_hdu.header['CRPIX2'] = (data_2d.shape[0] // 2) + 1
            new_hdu.header['CDELT1'] = -pixel_scale_deg # RA cresce para a esquerda
            new_hdu.header['CDELT2'] = pixel_scale_deg
            
            # Keyword extra que o PyPHER costuma buscar diretamente
            new_hdu.header['PIXSCALE'] = pixel_scale_arcsec

            # 4. Salva o arquivo pronto para o combate
            new_hdu.writeto(output_file, overwrite=True)
            print(f"✅ Arquivo pronto para o PyPHER: {os.path.basename(output_file)}")

    else:
        print('Forneça dados dos surveys PHANGS(HST/WFC3) ou S4G(IRAC1/IRAC2)')


def pypher_kernel_creation(todos_fwhm, psf_master_name):
    if psf_master_name.upper().startswith('F'):
        psf_master_path = os.path.expanduser(f'~/Desktop/AsTrovello/Input/PHANGS/PSF_LIMPAS/PSFSTD_WFC3UV_{psf_master_name.upper()}.fits')
    elif psf_master_name.upper().startswith('I'):
        psf_master_path = os.path.expanduser(f'~/Desktop/AsTrovello/Input/S4G/PSF_LIMPAS/{psf_master_name.upper()}_col129_row129.fits')
    output_dir = os.path.expanduser('~/Desktop/AsTrovello/Output/PSF_Kernels')

    psf_path_phangs_clean = os.path.expanduser('~/Desktop/AsTrovello/Input/PHANGS/PSF_LIMPAS')
    psf_path_s4g_clean = os.path.expanduser('~/Desktop/AsTrovello/Input/S4G/PSF_LIMPAS')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"🗑️  Removing previous directory: {os.path.basename(output_dir)}")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    # Lista de comandos para rodar no terminal
    comandos_pypher = []

    for filtro in todos_fwhm.keys():
        if (filtro != psf_master_name) and (filtro.startswith('f')):
            psf_high_res = os.path.join(psf_path_phangs_clean, f"PSFSTD_WFC3UV_{filtro.upper()}.fits")

            kernel_name = os.path.join(output_dir, f"kernel_{filtro}_to_{psf_master_name}.fits")
            cmd = f"pypher {psf_high_res} {psf_master_path} {kernel_name}"
            comandos_pypher.append(cmd)

        elif (filtro != psf_master_name) and (filtro.startswith('i')):    
            psf_high_res = os.path.join(psf_path_s4g_clean, f"{filtro.upper()}_col129_row129.fits")

            kernel_name = os.path.join(output_dir, f"kernel_{filtro}_to_{psf_master_name}.fits")
            cmd = f"pypher {psf_high_res} {psf_master_path} {kernel_name}"
            comandos_pypher.append(cmd)


    return comandos_pypher

def convolved_dict(path_phangs = Path('~/Desktop/AsTrovello/Input/PHANGS/phangs_hst/ngc1087/images').expanduser(), \
    path_s4g_reprojected = Path('~/Desktop/AsTrovello/Input/reprojected_files/ngc1087').expanduser(), \
        path_kernels = Path('~/Desktop/AsTrovello/Output/PSF_Kernels').expanduser()):

    # 2. Listando e filtrando TUDO de uma vez só com .glob()
    # O '*' é um coringa que significa "qualquer coisa"
    phangs_files = list(path_phangs.glob('*exp-drc-sci.fits'))
    s4g_files = list(path_s4g_reprojected.glob('*.fits'))
    kernel_files = list(path_kernels.glob('*.fits'))

    all_files = phangs_files + s4g_files + kernel_files

    kernel_files_names = sorted([f.name for f in kernel_files])
    filter_info = [f.split('_')[1] for f in kernel_files_names]


    fftconvolve_dict = defaultdict(lambda: {'kernel': {}, 'img': {}}) 
    for f in filter_info:
        for file in all_files:
            file_string = str(file)
            
            # 1. Verifica se o filtro 'f' está no nome do arquivo
            if f in file_string:
                
                # 2. Se for um kernel, guarda na gaveta de kernel
                if 'kernel' in file_string: 
                    fftconvolve_dict[f]['kernel']['path'] = file
                    fftconvolve_dict[f]['kernel']['name'] = file.name
                
                # 3. Se NÃO for kernel, é uma imagem (seja PHANGS ou S4G)
                else:
                    # Como você disse que os filtros não se repetem, 
                    # o arquivo que cair aqui será o único dono desse filtro.
                    fftconvolve_dict[f]['img']['path'] = file
                    fftconvolve_dict[f]['img']['name'] = file.name
    return fftconvolve_dict

def create_convolvedFITS(original_fits , kernel_fits, output_dir = '~/Desktop/AsTrovello/Input/convolved_fits', GAL_NAME = False):
    output_dir = Path(output_dir).expanduser()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with fits.open(kernel_fits) as hdu_kernel, \
     fits.open(original_fits) as hdu_img:
        kernel_data = hdu_kernel[0].data
        kernel_header = hdu_kernel[0].header
        img_data = hdu_img[0].data
        img_header = hdu_img[0].header


    kernel_data = np.nan_to_num(kernel_data)

    if np.sum(kernel_data) != 0:
        kernel_norm = kernel_data / np.sum(kernel_data)

    original_file_name = original_fits.name
    info = original_file_name.split('_')
    
    if 'phangs-hst' in info:
        convolved_img = fftconvolve(img_data, kernel_norm, mode='same')
        galaxy_name = info[4].lower()
        survey = 'phangs'
        filter_name = info[5].lower()
    elif 's4g' in original_file_name:
        img_data_limpa = np.nan_to_num(img_data, nan=0.0)
        convolved_img = fftconvolve(img_data_limpa, kernel_norm, mode='same')
        galaxy_name = info[0].lower()
        survey = 's4g'
        filter_name = info[2].lower()
    
    convolved_fits = fits.PrimaryHDU(data = convolved_img, header = img_header)

    print(100 * '#')
    print(f'Convoluindo para filtro {filter_name} do survey {survey}:')

    output_path = os.path.join(output_dir, galaxy_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_name = f'{galaxy_name}_{survey}_{filter_name}_convolved.fits'
    output_file = os.path.join(output_path, output_name)
    convolved_fits.writeto(output_file, overwrite=True)

    print(f'FITS convoluído salvo em: {output_file}')
    print(100 * '#')

    if GAL_NAME == True:
        return galaxy_name


# -------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- FITS unit conversion-----------------------------------------------------
def convert2Jansky(fits_file):
    with fits.open(fits_file) as hdu:
        data = hdu[0].data
        header = hdu[0].header
    
    new_data = data.copy()
    new_header = header.copy()

    # --- CASO HST (PHANGS) ---
    if header.get('BUNIT') == 'ELECTRONS/S':
        # Conversão direta: (e/s) * (Jy*s/e) = Jy
        new_data = new_data * header['PHOTFNU']
        new_header['BUNIT'] = 'Jy/pixel'
        print(f"HST: Convertido usando PHOTFNU")

    # --- CASO SPITZER (S4G) ---
    elif header.get('BUNIT') == 'MJy/sr': 
        # 1. Área do pixel em arcsec²
        pixel_area_arcsec2 = np.abs(header['PXSCAL1'] * header['PXSCAL2'])
        
        # 2. Conversão de arcsec² para steradian (sr)
        # 1 sr = (180/pi * 3600)² arcsec² ≈ 4.25e10 arcsec²
        # Portanto, 1 arcsec² ≈ 2.350443e-11 sr
        sr_per_arcsec2 = 2.3504430539e-11
        
        pixel_area_sr = pixel_area_arcsec2 * sr_per_arcsec2
        
        # 3. Conversão final:
        # Valor[MJy/sr] * 10^6 [Jy/MJy] * Área[sr/pixel] = Jy/pixel
        new_data = new_data * 1e6 * pixel_area_sr
        
        new_header['BUNIT'] = 'Jy/pixel'
        new_header['HISTORY'] = f"Converted from MJy/sr to Jy/pix using area {pixel_area_sr:.4e} sr"
        print(f"S4G: Convertido usando área do pixel ({pixel_area_arcsec2} arcsec2)")

    return new_data, new_header
    
#----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- Mask ----------------------------------------------------------
# Cuts survey intersection areas

def phangs_intersection_mask(ref_file):
    print('Encontrando zona de intersecção de surveys...')
    if ref_file is None or len(ref_file) == 0:
        print('CUIDADO: Arquivo de referência não encontrado. Prosseguindo sem máscara de borda...')
        mask_ref = None    
    else:
        ref_data = fits.getdata(ref_file[0], ext=0)
        mask_ref = (ref_data != 0) 
    return mask_ref

# Creates integrated 2D image to generate sky mask
def soma_img(aligned_images, ref_file):
    res = None
    intersection_mask = phangs_intersection_mask(ref_file)

    for data_orig in tqdm(aligned_images, desc="Somando para máscara"):
        data = data_orig.copy() # Cópia para não alterar o dado original da lista
        if intersection_mask is not None:
            if data.shape == intersection_mask.shape:
                data[~intersection_mask] = 0
        if res is None:
            res = data
        else:
            res += data
    return res

# Generates sky mask
def mask(data, N_SIGMA=3):
    data_subtraida = np.zeros_like(data)
    mask_result = np.zeros_like(data, dtype=bool)
    data_filtrada = data[data != 0]

    if data_filtrada.size == 0:
        return data_subtraida, mask_result

    local_bg = np.nanmedian(data_filtrada)
    data_subtraida = data - local_bg
    residuo_filtrado = data_subtraida[data != 0] 
    
    noise_median = np.nanmedian(residuo_filtrado)
    mad = np.nanmedian(np.abs(residuo_filtrado - noise_median))
    sigma_bg = 1.4826 * mad

    if sigma_bg > 0:
        mask_result = data_subtraida > (N_SIGMA * sigma_bg)
        data_subtraida[data_subtraida < 0] = 0
    
    return data_subtraida, mask_result

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Data Cube creation-------------------------------------------------------

# Creates complete data cube
def create_data_cube(aligned_images, filter_names, ref_file, ref_header, output_filename, \
     aplicar_mask=True, N_SIGMA=3, padding=50):
    
    print(f'\nIniciando criação do cubo (N_SIGMA={N_SIGMA})...')
    n_filtros = len(filter_names)
    ny, nx = aligned_images[0].shape
    cubo = np.empty((n_filtros, ny, nx), dtype=np.float32)
    
    # 1. Obter a máscara de intersecção (essencial para o recorte sempre)
    inter_mask = phangs_intersection_mask(ref_file)
    
    # 2. Decidir a máscara final de processamento
    if aplicar_mask:
        print('Gerando máscara de Sigma (N_SIGMA)...')
        summed = soma_img(aligned_images, ref_file)
        _, sigma_mask = mask(summed, N_SIGMA=N_SIGMA)
        mask_final = sigma_mask
    else:
        print('Usando apenas bordas de intersecção.')
        mask_final = inter_mask

    # 3. Preenchimento do cubo com NaNs nas regiões inválidas
    print('Limpando bordas e preenchendo camadas...')
    for i in range(n_filtros):
        img_atual = aligned_images[i]
        if mask_final is not None:
            cubo[i, :, :] = np.where(mask_final, img_atual, np.nan)
        else:
            cubo[i, :, :] = img_atual

    # --- RECORTE AUTOMÁTICO (BBOX) ---
    # Agora o recorte acontece SEMPRE que houver uma máscara válida
    y_off, x_off = 0, 0
    if mask_final is not None:
        coords = np.argwhere(mask_final)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            y_min, y_max = max(0, y_min - padding), min(ny, y_max + padding)
            x_min, x_max = max(0, x_min - padding), min(nx, x_max + padding)

            cubo = cubo[:, y_min:y_max, x_min:x_max]
            y_off, x_off = y_min, x_min
            print(f"✂️ Recorte: {ny}x{nx} -> {cubo.shape[1]}x{cubo.shape[2]}")

    # --- CONFIGURAÇÃO DO WCS 3D ---
    wcs_2d = WCS(ref_header, naxis=2)
    wcs_3d = WCS(naxis=3)
    
    for i in [0, 1]:
        for param in ['crpix', 'crval', 'cdelt', 'ctype', 'cunit']:
            try:
                val = getattr(wcs_2d.wcs, param)[i]
                if param == 'crpix':
                    getattr(wcs_3d.wcs, param)[i] = val - (x_off if i == 0 else y_off)
                else:
                    getattr(wcs_3d.wcs, param)[i] = val
            except: continue
    
    wcs_3d.wcs.crpix[2], wcs_3d.wcs.crval[2], wcs_3d.wcs.cdelt[2], wcs_3d.wcs.ctype[2] = 1, 0, 1, 'FILTER'
    
    cube_header = wcs_3d.to_header()
    cube_header['BUNIT'] = 'Jy/pixel'
    for i, filt in enumerate(filter_names):
        cube_header[f'FILT{i+1:03d}'] = filt
        
    fits.writeto(output_filename, cubo, header=cube_header, overwrite=True)
    return cubo, cube_header

# Creates smaller cubes
def create_cutouts(cube, cube_header, regions):
    for path, x_start, x_end, y_start, y_end in regions:
        if x_end > cube.shape[2] or y_end > cube.shape[1] or x_start < 0 or y_start < 0:
            print(f"Aviso: Zoom '{path.name}' fora dos limites. Pulando.")
            continue
        
        cube_cut = cube[:, y_start:y_end, x_start:x_end]
        cut_header = cube_header.copy()
        cut_header['CRPIX1'] -= x_start
        cut_header['CRPIX2'] -= y_start
        fits.writeto(path, cube_cut, header=cut_header, overwrite=True)
        print(f"🔎 Zoom salvo: {path.name}")


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Main function -----------------------------------------------------------

# Main script to generate data cubes
def main():
    # CONFIGURAÇÕES
    VALOR_N_SIGMA = 1 # Valor sugerido para NGC 1087
    APLICAR_MASCARA = False
    output_dir = Path("~/Desktop/Capivara_mestrado/Input/").expanduser()
    loc = Path("~/Desktop/Capivara_mestrado/Input/convolved_fits/ngc1087").expanduser()
    ref_dir = Path('~/Desktop/Capivara_mestrado/Input/PHANGS/phangs_hst/ngc1087/images/').expanduser()
    
    file_list = list(loc.glob('*_Jy_per_pixel.fits'))
    ref_file = list(ref_dir.glob('*f275w*sci.fits')) 

    if not file_list: return

    fits_files = sorted([(f, f.name.split('_')[2]) for f in file_list])
    ref_header = fits.getheader(fits_files[0][0], ext=0)

    aligned_images, filter_names = [], []
    for file, filt in tqdm(fits_files, desc="Lendo FITS"):
        aligned_images.append(fits.getdata(file, ext=0).astype(np.float32))
        filter_names.append(filt)

    if aligned_images:
        temp_name = output_dir / 'temp_cube.fits'
        cube, cube_header = create_data_cube(
            aligned_images, filter_names, ref_file, ref_header, 
            temp_name, aplicar_mask=APLICAR_MASCARA, N_SIGMA=VALOR_N_SIGMA
        )

        # RENOMEAÇÃO DINÂMICA (Baseada no objeto em memória)
        _, final_ny, final_nx = cube.shape
        final_name = output_dir / f'datacube_sci_{final_nx}x{final_ny}_Jy_per_pixel.fits'
        temp_name.rename(final_name)
        print(f"✅ Cubo finalizado: {final_name.name}")

        del aligned_images
        gc.collect()
        
        # ZOOM AUTOMATIZADO (Centralizado no CRPIX)
        cx, cy = int(cube_header['CRPIX1']), int(cube_header['CRPIX2'])
        regions = [(output_dir / f'datacube_sci_{final_nx}x{final_ny}_zoomed.fits', 
                    cx-150, cx+150, cy-150, cy+150)]
        create_cutouts(cube, cube_header, regions)




# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Additional functios -----------------------------------------------------

# Obtains filters in file names
def get_filters(file_list, start, position):
    return list(set(file.split('_')[position] for file in file_list if file.startswith(start)))

# Feedback on memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Uso de memória: {process.memory_info().rss / 1024 ** 2:.2f} MB")


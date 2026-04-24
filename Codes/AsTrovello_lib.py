from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import numpy as np
import psutil
import gc
from tqdm import tqdm  
from pathlib import Path
import matplotlib.pyplot as plt
import os 


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


import numpy as np
from .mask import soma_img, mask, phangs_intersection_mask
from astropy.wcs import WCS
from astropy.io import fits
from scipy.ndimage import center_of_mass
from astropy.stats import sigma_clipped_stats


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Data Cube creation-------------------------------------------------------

def sky_level(plane):
    v = plane[np.isfinite(plane)]      # 2. drop NaNs
    v = v[v != 0.0]                    # 3. drop exact-zero padding
    _, sclip_median, sclip_std = sigma_clipped_stats(v, sigma=3.0, maxiters=5)  # 4.
    return dict(valid_pixels = v.size,
                sclip_median = float(sclip_median),
                pct_neg = 100.0 * np.mean(v < 0))

def create_data_cube(aligned_images, filter_names, ref_file, ref_header, output_filename, 
                     aplicar_mask=True, N_SIGMA=3, padding=50, is_error = False, sky_subtraction = True):
    """
    Constructs a 3D FITS Hypercube (RA, DEC, Filter).
    Includes automatic sky masking, background subtraction, and Bounding Box cutout.
    Updates WCS to 3D.
    """
    print('\nInitiating hypercube creation...')
    ny, nx = aligned_images[0].shape
    cubo = np.empty((len(filter_names), ny, nx), dtype=np.float32)

    print('Determinig intersecting area between surveys...')
    inter_mask = phangs_intersection_mask(ref_file)

    if (sky_subtraction) and (not is_error):
        sub_aligned_images = []

        print('Performing sky subtraction...\n')
        print(154*'-')
        print(f"{'filter':6s} | {'valid_pixels_original':>22s} | {'sky_level_original':>23s} | {'%neg_original':>14s} | "
        f"{'valid_pixels_subtracted':>26s} | {'sky_level_subtracted':>27s} | {'%neg_subtracted':>18s} ")
        print(154*'-')

        for i, img in enumerate(aligned_images):
            band = filter_names[i]

            if inter_mask is not None:
                valid_pixels_mask = inter_mask & np.isfinite(img) & (img != 0)
            else:
                valid_pixels_mask = np.isfinite(img) & (img != 0)
            regular_dict = sky_level(img[valid_pixels_mask])

            img_sub = np.where(valid_pixels_mask, img - regular_dict['sclip_median'], np.nan)
            subtracted_dict = sky_level(img_sub)
            sub_aligned_images.append(img_sub)

            print(f"{band:6s} | {regular_dict['valid_pixels']:>22d} | {regular_dict['sclip_median']:>+23.2e} | "
            f"{regular_dict['pct_neg']:>14.2f} | {subtracted_dict['valid_pixels']:>26d} | {subtracted_dict['sclip_median']:>+27.2e} | "
            f"{subtracted_dict['pct_neg']:>18.2f}")
            del(img)
            
        print(154*'-')
        aligned_images = sub_aligned_images

        print('\nSubtraction executed. Building datacube...')
    
    # 1. Determine final processing mask (Signal-based or Border-based)
    if aplicar_mask:
        summed = soma_img(aligned_images, ref_file)
        _, mask_final = mask(summed, N_SIGMA=N_SIGMA)

        # 2. Fill the cube layers, setting non-mask regions to NaN
        for i, img_atual in enumerate(aligned_images):

            if is_error:
                # Error cube: apply mask only, no sky subtraction
                # Errors represent uncertainty — there is no physical background to remove
                cubo[i, :, :] = np.where(mask_final, img_atual, np.nan) if mask_final is not None else img_atual

            else:
                # Science cube: subtract sky background before masking
                cubo[i, :, :] = np.where(mask_final, img_atual, np.nan) if mask_final is not None else img_atual
    else:
        mask_final = inter_mask 

    for i, img_atual in enumerate(aligned_images):
        if mask_final is not None:
            cubo[i, :, :] = np.where(mask_final, img_atual, np.nan)
        else:
            cubo[i, :, :] = img_atual

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
            print(f"==> Bounding box cutout: {ny}x{nx} -> {cubo.shape[1]}x{cubo.shape[2]}")

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

def create_cutout(data, header, output_filename):

    # 1. Determina as dimensões e o centro de massa uma única vez
    ref_img = data[0, :, :] 
    ny, nx = ref_img.shape

    center_y, center_x = center_of_mass(np.nan_to_num(ref_img))
    center_y, center_x = int(center_y), int(center_x)
    print(f"Image center of mass: y={center_y}, x={center_x}")

    # 2. OTIMIZAÇÃO 1: Cria a grade 2D de distâncias uma ÚNICA vez para a imagem inteica
    y_grid, x_grid = np.indices((ny, nx))
    dist_map = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)

    # 3. OTIMIZAÇÃO 2: Procura pixels inválidos em todo o cubo de uma vez só (sem loops!)
    # data <= 0 ou np.isnan(data) gera uma matriz 3D booleana. 
    # O .any(axis=0) colapsa o cubo: se o pixel for inválido em QUALQUER filtro, vira True.
    invalid_mask = (data <= 0) | np.isnan(data)
    invalid_pixels_map = invalid_mask.any(axis=0)

    # 4. Determina o Raio Crítico instantaneamente
    if not np.any(invalid_pixels_map):
        print("Whole cube has valid pixels. No cuts needed.")
        radius_mask = np.ones((ny, nx), dtype=bool)
    else:
        # O truque de mestre: pegamos o mapa de distâncias e filtramos apenas as posições inválidas
        # O np.min() extrai o menor raio diretamente, sem passar por loops ou funções externas
        radius = np.min(dist_map[invalid_pixels_map])
        print(f"Maximum safety radius: {radius:.2f} pixels.")
        
        # Cria a máscara circular perfeita
        radius_mask = dist_map <= radius

    # 5. Aplica a máscara no cubo inteiro (em todas as dimensões de uma vez só!)
    # O np.where aplica a máscara 2D ao longo de todo o array 3D 'data' de forma vetorizada
    cube_clean = np.where(radius_mask[np.newaxis, :, :], data, np.nan)

    # ... (código anterior que gerou o cube_clean) ...

    # 6. Definindo os limites do Bounding Box tangente ao círculo válido
    # O int() garante que o índice seja inteiro para o recorte da matriz
    # O max() e min() garantem que o recorte nunca tente pedir um pixel fora da imagem original
    y_min = max(0, int(center_y - radius))
    y_max = min(ny, int(center_y + radius) + 1)  # +1 para o slice do Python incluir a borda

    x_min = max(0, int(center_x - radius))
    x_max = min(nx, int(center_x + radius) + 1)

    print(f"Cutting hypercube... Bounding Box: y[{y_min}:{y_max}], x[{x_min}:{x_max}]")

    # 7. O Recorte Final Volumétrico
    # Fatiamos os eixos Y e X, mantendo todos os filtros (eixo 0) intocados
    cube_cropped = cube_clean[:, y_min:y_max, x_min:x_max]
    dim, ny_new, nx_new = cube_cropped.shape

    print(f"Original cube size : {ny}x{nx} pixels")
    print(f"Cutout cube size: {ny_new}x{nx_new} pixels")

    # ---------------------------------------------------------
    # 8. O Conserto da Astrometria (WCS Header)
    # ---------------------------------------------------------
    print('Adjusting WCS to new cut...')
    
    # Cria uma cópia independente para não corromper o header original na memória
    cube_header_cropped = header.copy()

    # Aplica a translação no pixel de referência
    if 'CRPIX1' in cube_header_cropped and 'CRPIX2' in cube_header_cropped:
        cube_header_cropped['CRPIX1'] -= x_min
        cube_header_cropped['CRPIX2'] -= y_min
        
    # Atualiza as dimensões físicas da matriz no cabeçalho (NAXIS)
    cube_header_cropped['NAXIS1'] = nx_new  # Largura (Eixo X)
    cube_header_cropped['NAXIS2'] = ny_new # Altura (Eixo Y)
    cube_header_cropped['NAXIS3'] = dim # Profundidade (Filtros)
    
    # (Opcional) Adiciona um comentário no arquivo para rastreabilidade
    cube_header_cropped.add_history(f"BBox cutout applied: X_offset={x_min}, Y_offset={y_min}")

    output_filename_new = output_filename.parent /  output_filename.name.replace(f'{nx}x{ny}', f'{nx_new}x{ny_new}')
    fits.writeto(output_filename_new, cube_cropped, header=cube_header_cropped, overwrite=True)
    print(f"Cut cube saved successfully: {output_filename_new}")

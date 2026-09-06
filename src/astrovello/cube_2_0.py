from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from mask_2_0 import (
    intersection_footprint_mask, 
    crop_to_mask_bbox,
    sky_level,
    sum_images,
    mask_after_sky_sub
)

def discover_jansky_files(
    reprojected_dir: Path,
    galaxy: str,
    target_master_filter: str | None = None,
    selected_surveys: list | None = None,
) -> dict:
    """
    Scans reprojected_dir/galaxy for Jansky-converted FITS files (produced
    by convert2Jansky), parsing survey/filter from the pipeline's own
    filename convention. Makes the data-cube step runnable standalone
    (--mode cube_only), independent of whether alignment/conversion ran
    in the same CLI invocation.

    - If target_master_filter is given, only matches that master file.
    - If selected_surveys is given, only matches files from those surveys.

    Returns
    -------
    dict of {str : dict}
        ``{filter_name: {'path': Path, 'survey': str, 'is_master': bool}}``
    """
    gal_dir = Path(reprojected_dir) / galaxy
    if not gal_dir.is_dir():
        raise FileNotFoundError(f"No reprojected files found for galaxy '{galaxy}' at {gal_dir}")

    result = {}
    target_surveys = {s.upper() for s in selected_surveys} if selected_surveys else None

    # Busca o arquivo master já convertido
    master_files = list(gal_dir.glob('*_master_Jy_per_pixel.fits'))
    if not master_files:
        raise ValueError(f"No Jy/pixel master file found in {gal_dir} — run unit conversion first.")

    matched_master = None
    for f in master_files:
        parts = f.stem.split('_')
        survey, filt = parts[1].upper(), parts[2]

        if target_master_filter is not None:
            if filt == target_master_filter:
                matched_master = (f, survey, filt)
                break
        else:
            matched_master = (f, survey, filt)
            break

    if matched_master is None:
        raise ValueError(
            f"Requested master filter '{target_master_filter}' not found among existing Jy/pixel masters: "
            f"{[f.name for f in master_files]}"
        )

    m_path, m_survey, m_filt = matched_master
    result[m_filt] = {'path': m_path, 'survey': m_survey, 'is_master': True}

    # Busca os demais arquivos já convertidos, filtrando por survey e master atuais
    for f in gal_dir.glob('*_projection_Jy_per_pixel.fits'):
        parts = f.stem.split('_')
        survey = parts[1].upper()
        filt = parts[2]
        conv_ref_filt = parts[5] if len(parts) >= 6 else None   # {gal}_{survey}_{filt}_on_{ref_survey}_{ref_filt}_...

        if target_surveys is not None and survey not in target_surveys:
            continue

        if target_master_filter is not None and conv_ref_filt is not None:
            if conv_ref_filt != target_master_filter:
                continue

        result[filt] = {'path': f, 'survey': survey, 'is_master': False}

    return result

def create_data_cube(
    jansky_files_dict: dict, ordered_filters: list,
    reference_path: Path, drivers: dict, output_filename: Path,
    apply_mask: bool = True, n_sigma: float = 3, padding: int = 50,
    sky_subtraction: bool = True,
) -> tuple:
    print('\nInitiating hypercube creation...')

    # 1. Carrega todas as bandas
    raw_images, ref_header = [], None
    for filt in ordered_filters:
        entry = jansky_files_dict[filt]
        with fits.open(entry['path']) as hdu:
            raw_images.append(hdu[0].data)
            if entry['is_master']:
                ref_header = hdu[0].header.copy()
    if ref_header is None:
        with fits.open(reference_path) as hdu:
            ref_header = hdu[0].header.copy()

    # 2. Footprint: intersecção de todas as bandas selecionadas
    print('Determining intersection footprint across surveys...')
    footprint_mask = intersection_footprint_mask(jansky_files_dict, drivers)

    # 3. Corta TODAS as bandas para a menor área útil ANTES de qualquer processamento
    print('Cropping to minimal useful area...')
    cropped_images, cropped_header, (y_off1, x_off1) = crop_to_mask_bbox(
        raw_images, ref_header, footprint_mask, padding=0
    )
    print(f"==> Footprint crop: {footprint_mask.shape} -> {cropped_images[0].shape}")

    ny, nx = cropped_images[0].shape
    cubo = np.empty((len(ordered_filters), ny, nx), dtype=np.float32)

    # 4. Sky subtraction, agora sobre arrays já reduzidos
    if sky_subtraction:
        sub_images = []
        print('Performing sky subtraction...\n')
        print(154*'-')
        print(f"{'filter':6s} | {'valid_pixels_original':>22s} | {'sky_level_original':>23s} | {'%neg_original':>14s} | "
              f"{'valid_pixels_subtracted':>26s} | {'sky_level_subtracted':>27s} | {'%neg_subtracted':>18s} ")
        print(154*'-')

        for filt, img in zip(ordered_filters, cropped_images):
            # Identifica os pixels válidos usando as regras do driver do survey
            driver = drivers[jansky_files_dict[filt]['survey']]
            valid_pixels_mask = np.isfinite(img) & ~driver.get_invalid_mask(img)
            
            # Calcula as estatísticas originais
            regular_dict = sky_level(img[valid_pixels_mask])
            
            # Aplica a subtração do céu
            img_sub = np.where(valid_pixels_mask, img - regular_dict['sclip_median'], np.nan)
            
            # Calcula as estatísticas após a subtração
            subtracted_dict = sky_level(img_sub[valid_pixels_mask])
            sub_images.append(img_sub)
            
            # Imprime a linha formatada da tabela
            print(f"{filt:6s} | {regular_dict['valid_pixels']:>22d} | {regular_dict['sclip_median']:>+23.2e} | "
                  f"{regular_dict['pct_neg']:>14.2f} | {subtracted_dict['valid_pixels']:>26d} | {subtracted_dict['sclip_median']:>+27.2e} | "
                  f"{subtracted_dict['pct_neg']:>18.2f}")
                  
        print(154*'-')
        cropped_images = sub_images

    # 5. Máscara de sinal (galáxia), sobre a imagem somada já cortada
    mask_filename = output_filename.parent / 'master_signal_mask.fits'
    if apply_mask:
        summed = sum_images(cropped_images, footprint_mask=None)  # já cortado, sem precisar de ref_file
        mask_final = mask_after_sky_sub(summed, N_SIGMA=n_sigma)
    else:
        mask_final = np.isfinite(cropped_images[0])

    fits.writeto(mask_filename, mask_final.astype(np.uint8), overwrite=True)
    print(f"==> Science mask saved to: {mask_filename.name}")

    for i, img in enumerate(cropped_images):
        cubo[i, :, :] = np.where(mask_final, img, np.nan)

    # 6. Segundo corte: agora pela máscara de sinal, com padding
    cube_planes, final_header, (y_off2, x_off2) = crop_to_mask_bbox(
        [cubo[i] for i in range(cubo.shape[0])], cropped_header, mask_final, padding=padding
    )
    cubo = np.stack(cube_planes, axis=0)
    print(f"==> Signal-mask cutout: {cropped_images[0].shape} -> {cubo.shape[1:]}")

    # 7. Header 3D final
    w_2d = WCS(final_header, naxis=2)
    w_3d = WCS(naxis=3)
    for i in [0, 1]:
        for p in ['crpix', 'crval', 'cdelt', 'ctype', 'cunit']:
            try:
                getattr(w_3d.wcs, p)[i] = getattr(w_2d.wcs, p)[i]
            except Exception:
                continue
    w_3d.wcs.crpix[2], w_3d.wcs.crval[2], w_3d.wcs.cdelt[2], w_3d.wcs.ctype[2] = 1, 0, 1, 'FILTER'
    
    cube_header = w_3d.to_header()
    cube_header['BUNIT'] = 'Jy/pixel'
    
    for i, filt in enumerate(ordered_filters):
        cube_header[f'FILT{i+1:03d}'] = filt

    # --- Inserção dos Comentários no Cabeçalho ---
    # Extrai os surveys únicos usando um set comprehension
    surveys_used = ", ".join(sorted(set(entry['survey'] for entry in jansky_files_dict.values())))
    # Identifica a chave que contém a tag is_master
    master_filter = next(filt for filt, entry in jansky_files_dict.items() if entry['is_master'])
    
    cube_header['COMMENT'] = f"Surveys combined in this cube: {surveys_used} (master filter : {master_filter})"
        
    # --- Nomenclatura Dinâmica do Arquivo Final ---
    # Extrai o número de filtros (ignorando) e as dimensões Y e X do shape do cubo
    _, ny, nx = cubo.shape
    
    # Utiliza o stem (ex: 'ngc1087_datacube') e injeta o sufixo _sci_{nx}x{ny}_Jy_per_pixel.fits
    final_output_path = output_filename.parent / f"{output_filename.stem}_sci_{nx}x{ny}_Jy_per_pixel.fits"
        
    fits.writeto(final_output_path, cubo, header=cube_header, overwrite=True)
    return cubo, cube_header
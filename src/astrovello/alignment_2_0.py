from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Image alignment -------------------------------------------------------

def discover_convolved_files(
    convolved_dir: Path, 
    galaxy: str, 
    target_master_filter: str | None = None,
    selected_surveys: list | None = None
) -> dict:
    """
    Scans convolved_dir/galaxy for convolved + master FITS files.
    - If target_master_filter is given, only matches that master file.
    - If selected_surveys is given, only matches convolved files from those surveys.
    """
    gal_dir = Path(convolved_dir) / galaxy
    if not gal_dir.is_dir():
        raise FileNotFoundError(f"No convolved files found for galaxy '{galaxy}' at {gal_dir}")

    result = {}
    target_surveys = {s.upper() for s in selected_surveys} if selected_surveys else None

    # Busca os arquivos master
    master_files = list(gal_dir.glob('*_master.fits'))
    if not master_files:
        raise ValueError(f"No master file found in {gal_dir} — run convolution first.")

    matched_master = None
    for f in master_files:
        _, survey, filt, _ = f.stem.split('_')
        survey = survey.upper()  

        if target_master_filter is not None:
            if filt == target_master_filter:
                matched_master = (f, survey, filt)
                break
        else:
            matched_master = (f, survey, filt)
            break

    if matched_master is None:
        raise ValueError(
            f"Requested master filter '{target_master_filter}' not found among existing masters: "
            f"{[f.name for f in master_files]}"
        )

    m_path, m_survey, m_filt = matched_master
    result[m_filt] = {'path': m_path, 'survey': m_survey, 'is_master': True}

    # Busca os arquivos convoluídos que batem com o survey e com o master correto
    for f in gal_dir.glob('*_convolved.fits'):
        parts = f.stem.split('_')
        survey = parts[1].upper()   
        filt = parts[2]
        conv_to_master = parts[5] if len(parts) >= 6 else None

        if target_surveys is not None and survey not in target_surveys:
            continue

        if target_master_filter is not None and conv_to_master is not None:
            if conv_to_master != target_master_filter:
                continue

        result[filt] = {'path': f, 'survey': survey, 'is_master': False}

    return result

# Reproject higher resolution images to master image reference frame
def reproject_to_reference(
    img_to_reproject: Path, img_survey: str, img_filter: str,
    reference_img: Path, ref_survey: str, ref_filter: str,
    galaxy: str, output_path: Path,
    apply_sip_img_to_reproject: bool = False,
    apply_sip_reference_img: bool = False,
    verbose: bool = True,
) -> None:
    """Aligns a convolved image onto the reference (master) file's pixel grid."""
    with fits.open(img_to_reproject) as hdu_i, fits.open(reference_img) as hdu_r:
        hdu_img_base, hdu_ref = hdu_i[0], hdu_r[0]

        # --- Imagem a ser reprojetada ---
        if apply_sip_img_to_reproject:
            hdu_img_base.header['CTYPE1'] = 'RA---TAN-SIP'
            hdu_img_base.header['CTYPE2'] = 'DEC--TAN-SIP'
            if verbose:
                print(f"\tSIP correction added to {img_to_reproject.name} header!")
            w_img_base = WCS(hdu_img_base.header)
        else:
            w_img_base = WCS(hdu_img_base.header)
            w_img_base.sip = None

        # --- Imagem de referência (master) ---
        if apply_sip_reference_img:
            hdu_ref.header['CTYPE1'] = 'RA---TAN-SIP'
            hdu_ref.header['CTYPE2'] = 'DEC--TAN-SIP'
            if verbose:
                print(f"\tSIP correction added to {reference_img.name} header!")
            w_ref = WCS(hdu_ref.header)
        else:
            w_ref = WCS(hdu_ref.header)
            w_ref.sip = None

        # Executa a reprojeção
        array, _ = reproject_interp(
            (hdu_img_base.data, w_img_base),
            w_ref,
            shape_out=hdu_ref.data.shape
        )
        img_base_new_header = hdu_img_base.header.copy()

    # Atualiza o cabeçalho final com o grid WCS da referência
    wcs_ref_header = w_ref.to_header(relax=True)
    wcs_keys_to_remove = [
        'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2',
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
    ]
    for key in wcs_keys_to_remove:
        img_base_new_header.pop(key, None)

    img_base_new_header.update(wcs_ref_header)
    img_base_new_header['COMMENT'] = (
        f'Reprojected onto {ref_survey} ({ref_filter}) grid. '
        f'Surface brightness preserved; flux per pixel not strictly conserved.'
    )

    # Gravação do arquivo de saída
    output_directory = Path(output_path) / galaxy
    output_directory.mkdir(parents = True, exist_ok=True)
    output_name = f'{galaxy}_{img_survey}_{img_filter}_on_{ref_survey}_{ref_filter}_projection.fits'
    output_filename = output_directory / output_name
    fits.writeto(output_filename, array, img_base_new_header, overwrite=True)

    if verbose:
        print(f'\tReprojected FITS file: {output_name}\n')

    return output_filename
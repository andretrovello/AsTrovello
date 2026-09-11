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
    selected_surveys: list | None = None,
    strict: bool = True,
) -> dict:
    """Scan convolved_dir/galaxy for the files belonging to ONE configuration.

    Selection happens on the header (PSFTARGT), with the filename as a
    fallback for files written before that keyword existed. Filename-only
    matching is not enough: a leftover file from a previous run with a
    different master parses fine and gets picked up silently.

    Args:
        strict: raise if files from another configuration are present. Set
            False only when several configurations are kept side by side on
            purpose.
    """
    gal_dir = Path(convolved_dir) / galaxy
    if not gal_dir.is_dir():
        raise FileNotFoundError(
            f"No convolved files found for galaxy '{galaxy}' at {gal_dir}")

    target_surveys = {s.upper() for s in selected_surveys} if selected_surveys else None

    def header_target(path):
        """Master this file was produced against, or None if not recorded."""
        try:
            return fits.getheader(path).get('PSFTARGT')
        except Exception:
            return None

    def parse_stem(path, is_master):
        """(survey, filter) from the pipeline naming convention.

        master:    {gal}_{survey}_{filt}_master
        convolved: {gal}_{survey}_{filt}_to_{ref_survey}_{ref_filt}_convolved
        Parsed from the RIGHT so galaxy names containing '_' still work.
        """
        parts = path.stem.split('_')
        if is_master:
            return parts[-3].upper(), parts[-2]
        return parts[-6].upper(), parts[-5]

    result, foreign = {}, []

    # ---------------- master ----------------
    matched_master = None
    for f in sorted(gal_dir.glob('*_master.fits')):
        try:
            survey, filt = parse_stem(f, is_master=True)
        except IndexError:
            print(f"==> Skipping unparseable master filename: {f.name}")
            continue

        if target_surveys is not None and survey not in target_surveys:
            foreign.append((f.name, f"survey {survey} not selected"))
            continue
        if target_master_filter is not None and filt != target_master_filter:
            foreign.append((f.name, f"master {filt}"))
            continue

        matched_master = (f, survey, filt)
        break

    if matched_master is None:
        existing = sorted(f.name for f in gal_dir.glob('*_master.fits'))
        raise ValueError(
            f"No master file for filter '{target_master_filter}' in {gal_dir}.\n"
            f"Existing masters: {existing or 'none'}\n"
            f"Re-run the convolution stage with --create_kernel.")

    m_path, m_survey, m_filt = matched_master
    result[m_filt] = {'path': m_path, 'survey': m_survey, 'is_master': True}

    # ---------------- convolved / unmatched bands ----------------
    for f in sorted(gal_dir.glob('*_convolved.fits')):
        try:
            survey, filt = parse_stem(f, is_master=False)
        except IndexError:
            print(f"==> Skipping unparseable filename: {f.name}")
            continue

        if target_surveys is not None and survey not in target_surveys:
            foreign.append((f.name, f"survey {survey} not selected"))
            continue

        # A band matched to itself is always a leftover: the master's
        # canonical file carries the _master.fits suffix, never _convolved.
        if filt == target_master_filter:
            foreign.append((f.name, "band matched to itself (stale)"))
            continue

        # Header wins over filename: PSFTARGT is written by both
        # create_convolvedFITS and copy_as_convolved.
        target = header_target(f)
        if target is None:
            parts = f.stem.split('_')
            target = parts[-2] if len(parts) >= 2 else None

        if target_master_filter is not None and target != target_master_filter:
            foreign.append((f.name, f"matched to {target}"))
            continue

        if filt in result:
            raise ValueError(
                f"Two files claim band '{filt}': {result[filt]['path'].name} "
                f"and {f.name}. Clean {gal_dir} and reprocess.")

        result[filt] = {'path': f, 'survey': survey, 'is_master': False}

    if foreign:
        print(f"==> {len(foreign)} file(s) in {gal_dir.name} belong to another "
              f"configuration and were ignored:")
        for name, why in foreign[:10]:
            print(f"      {name}  ({why})")
        if strict:
            raise ValueError(
                f"Leftover files from a previous configuration are present in "
                f"{gal_dir}. They will not be used, but their presence means "
                f"the directory holds mixed results.\n"
                f"Delete them, or pass strict=False (--allow_mixed).")

    print(f">>> Selected {len(result)} band(s) for master "
          f"'{target_master_filter}': {sorted(result)}")
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

        # --- Image to be reprojected ---
        if apply_sip_img_to_reproject:
            hdu_img_base.header['CTYPE1'] = 'RA---TAN-SIP'
            hdu_img_base.header['CTYPE2'] = 'DEC--TAN-SIP'
            if verbose:
                print(f"\tSIP correction added to {img_to_reproject.name} header!")
            w_img_base = WCS(hdu_img_base.header)
        else:
            w_img_base = WCS(hdu_img_base.header)
            w_img_base.sip = None

        # --- Reference (master) image ---
        if apply_sip_reference_img:
            hdu_ref.header['CTYPE1'] = 'RA---TAN-SIP'
            hdu_ref.header['CTYPE2'] = 'DEC--TAN-SIP'
            if verbose:
                print(f"\tSIP correction added to {reference_img.name} header!")
            w_ref = WCS(hdu_ref.header)
        else:
            w_ref = WCS(hdu_ref.header)
            w_ref.sip = None

        # Run the reprojection
        array, _ = reproject_interp(
            (hdu_img_base.data, w_img_base),
            w_ref,
            shape_out=hdu_ref.data.shape
        )
        img_base_new_header = hdu_img_base.header.copy()

    # Replace the WCS of the output header with the reference grid
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

    # Write the output file
    output_directory = Path(output_path) / galaxy
    output_directory.mkdir(parents = True, exist_ok=True)
    output_name = f'{galaxy}_{img_survey.lower()}_{img_filter}_on_{ref_survey.lower()}_{ref_filter}_projection.fits'
    output_filename = output_directory / output_name
    fits.writeto(output_filename, array, img_base_new_header, overwrite=True)

    if verbose:
        print(f'\tReprojected FITS file: {output_name}\n')

    return output_filename
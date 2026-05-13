import numpy as np
from .mask import soma_img, mask, phangs_intersection_mask
from astropy.wcs import WCS
from astropy.io import fits



# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Data Cube creation-------------------------------------------------------

def create_data_cube(aligned_images, filter_names, ref_file, ref_header, output_filename, 
                     aplicar_mask=True, N_SIGMA=3, padding=50):
    """
    Constructs a 3D FITS Hypercube (RA, DEC, Filter).
    Includes automatic sky masking, background subtraction, and Bounding Box cutout.
    Updates WCS to 3D.
    """
    print(f'\nInitiating hypercube creation (N_SIGMA={N_SIGMA})...')
    ny, nx = aligned_images[0].shape
    cubo = np.empty((len(filter_names), ny, nx), dtype=np.float32)
    
    # 1. Determine final processing mask (Signal-based or Border-based)
    if aplicar_mask:
        summed = soma_img(aligned_images, ref_file)
        _, mask_final = mask(summed, N_SIGMA=N_SIGMA)
        # 2. Fill the cube layers, setting non-mask regions to NaN and 
        for i, img_atual in enumerate(aligned_images):
            sky = np.nanmedian(img_atual[~mask_final])
            cubo[i, :, :] = np.where(mask_final, img_atual - sky, np.nan) if mask_final is not None else img_atual

    else:
        mask_final = phangs_intersection_mask(ref_file)
        # 2. Fill the cube layers, setting non-mask regions to NaN
        for i, img_atual in enumerate(aligned_images):
            cubo[i, :, :] = np.where(mask_final, img_atual, np.nan) if mask_final is not None else img_atual

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
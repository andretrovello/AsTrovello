import numpy as np
from .mask import soma_img, mask_after_sky_sub, phangs_intersection_mask
from astropy.wcs import WCS
from astropy.io import fits
from scipy.ndimage import center_of_mass
from astropy.stats import sigma_clipped_stats


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Data Cube creation-------------------------------------------------------

def sky_level(plane):
    v = plane[np.isfinite(plane)]      # 2. drop NaNs
    v = v[v != 0.0]                    # 3. drop exact-zero padding
    _, sclip_median, _ = sigma_clipped_stats(v, sigma=3.0, maxiters=5)  # 4.
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

    print('Determining intersecting area between surveys...')
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
    mask_filename = output_filename.parent / 'master_signal_mask.fits'

    if not is_error:
        # SCIENCE RUN: Calculate the mask and save it to disk
        if aplicar_mask:
            summed = soma_img(aligned_images, ref_file)
            mask_final = mask_after_sky_sub(summed, N_SIGMA=N_SIGMA)
        else:
            mask_final = inter_mask
            
        # Save the boolean matrix (converted to integer) for the error cube to inherit
        fits.writeto(mask_filename, mask_final.astype(np.uint8), overwrite=True)
        print(f"==> Science mask saved to: {mask_filename.name}")

    else:
        # ERROR RUN: Load the previously calculated science mask
        if aplicar_mask and mask_filename.exists():
            print("==> Loading signal-based mask from the science run...")
            mask_final = fits.getdata(mask_filename).astype(bool)
        else:
            print("==> Warning: Mask not found or disabled. Using fallback inter_mask.")
            mask_final = inter_mask

    # Apply the final mask to the image planes
    for i, img_atual in enumerate(aligned_images):
        cubo[i, :, :] = np.where(mask_final, img_atual, np.nan)


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

    # 1. Determine dimensions and center of mass only once
    ref_img = data[0, :, :] 
    ny, nx = ref_img.shape

    center_y, center_x = center_of_mass(np.nan_to_num(ref_img))
    center_y, center_x = int(center_y), int(center_x)
    print(f"Image center of mass: y={center_y}, x={center_x}")

    # 2. OPTIMIZATION 1: Create the 2D distance grid ONCE for the entire image
    y_grid, x_grid = np.indices((ny, nx))
    dist_map = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)

    # 3. OPTIMIZATION 2: Search for invalid pixels in the entire cube at once (no loops!)
    # data <= 0 or np.isnan(data) generates a 3D boolean matrix. 
    # The .any(axis=0) collapses the cube: if the pixel is invalid in ANY filter, it becomes True.
    invalid_mask = (data <= 0) | np.isnan(data)
    invalid_pixels_map = invalid_mask.any(axis=0)

    # 4. Determine the Critical Radius instantly
    if not np.any(invalid_pixels_map):
        print("Whole cube has valid pixels. No cuts needed.")
        radius_mask = np.ones((ny, nx), dtype=bool)
    else:
        # The neat trick: take the distance map and filter only the invalid positions
        # np.min() directly extracts the smallest radius, without loops or external functions
        radius = np.min(dist_map[invalid_pixels_map])
        print(f"Maximum safety radius: {radius:.2f} pixels.")
        
        # Creates the perfect circular mask
        radius_mask = dist_map <= radius

    # 5. Apply the mask to the entire cube (across all dimensions at once!)
    # np.where applies the 2D mask along the entire 3D array 'data' in a vectorized manner
    cube_clean = np.where(radius_mask[np.newaxis, :, :], data, np.nan)

    # 6. Defining the limits of the Bounding Box tangent to the valid circle
    # int() ensures the index is an integer for slicing the matrix
    # max() and min() ensure the slice never attempts to ask for a pixel outside the original image
    y_min = max(0, int(center_y - radius))
    y_max = min(ny, int(center_y + radius) + 1)  # +1 for the Python slice to include the border

    x_min = max(0, int(center_x - radius))
    x_max = min(nx, int(center_x + radius) + 1)

    print(f"Cutting hypercube... Bounding Box: y[{y_min}:{y_max}], x[{x_min}:{x_max}]")

    # 7. The Final Volumetric Cutout
    # Slice the Y and X axes, keeping all filters (axis 0) untouched
    cube_cropped = cube_clean[:, y_min:y_max, x_min:x_max]
    dim, ny_new, nx_new = cube_cropped.shape

    print(f"Original cube size : {ny}x{nx} pixels")
    print(f"Cutout cube size: {ny_new}x{nx_new} pixels")

    # ---------------------------------------------------------
    # 8. Fixing Astrometry (WCS Header)
    # ---------------------------------------------------------
    print('Adjusting WCS to new cut...')
    
    # Create an independent copy to avoid corrupting the original header in memory
    cube_header_cropped = header.copy()

    # Apply translation to the reference pixel
    if 'CRPIX1' in cube_header_cropped and 'CRPIX2' in cube_header_cropped:
        cube_header_cropped['CRPIX1'] -= x_min
        cube_header_cropped['CRPIX2'] -= y_min
        
    # Update physical dimensions of the matrix in the header (NAXIS)
    cube_header_cropped['NAXIS1'] = nx_new  # Width (X-axis)
    cube_header_cropped['NAXIS2'] = ny_new # Height (Y-axis)
    cube_header_cropped['NAXIS3'] = dim # Depth (Filters)
    
    # (Optional) Add a history comment to the file for traceability
    cube_header_cropped.add_history(f"BBox cutout applied: X_offset={x_min}, Y_offset={y_min}")

    output_filename_new = output_filename.parent /  output_filename.name.replace(f'{nx}x{ny}', f'{nx_new}x{ny_new}')
    fits.writeto(output_filename_new, cube_cropped, header=cube_header_cropped, overwrite=True)
    print(f"Cut cube saved successfully: {output_filename_new}")
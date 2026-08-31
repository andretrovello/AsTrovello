import argparse
import gc
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from config import SURVEY_CONFIG
from drivers import BASE_Driver, PHANGS_Driver, S4G_Driver
from convolution_2_0 import (
    calculateFWHM,
    clean_psf,
    pypher_kernel_creation,
    convolved_dict,
    diagnose_negatives,
    create_convolvedFITS,
)
from alignment_2_0 import (
    discover_convolved_files,
    reproject_to_reference
)

def main():
    parser = argparse.ArgumentParser('AsTrovello Pipeline Control')
    parser.add_argument('--mode', type = str, choices = ['full', 'alignment_only', 'conv_only', 'cube_only'], \
        default = 'full', help = 'Execution mode')
    parser.add_argument('--galaxy', type = str, required = True, help = 'Galaxy name (e.g., ngc1566)')
    parser.add_argument('--create_kernel', action='store_true', help='If set, triggers PSF cleaning and PyPHER kernel generation')
    parser.add_argument('--apply_mask', action='store_true', help='If set, generates a signal-based sky mask for the final cube')
    parser.add_argument('--sigma', type = float, default = 1.0, help = 'Sigma threshold for sky mask cutting')
    parser.add_argument('--error', action='store_true', help='If set, creates error cube')
    parser.add_argument('--valid_pixels_cut', action='store_true', help='If set, cuts image only in a central radius where flux > 0 and not NaN')
    parser.add_argument('--force_convolution', action='store_true', help='If set, forces convolution, even if convolved files already exist')

    # --- Parse arguments ---
    args = parser.parse_args()
    galaxy = args.galaxy

    print(100*'#')
    print(f'Executing AsTrovello for {args.galaxy}...\n')

    # --- Essential directories ---
    CWD = Path.cwd()
    BASE_DIR = CWD.parents[1]
    input_dir = BASE_DIR / 'Input'
    output_dir = BASE_DIR / "Output"

    # Define Output directory structure
    kernel_dir = output_dir / 'PSF_Kernels'
    convolved_fits_dir = output_dir / 'convolved_fits' 
    reprojected_dir = output_dir / 'reprojected_files' 

    print(f'Root Directory: {BASE_DIR}')

    if not input_dir.exists():
        print(f"==> Error: 'Input' folder not found in {BASE_DIR}")
        print("Make sure you are in the correct directory.")
        return

    # --- Find surveys in input directory ---
    SURVEYS = [x.name for x in input_dir.iterdir() if x.is_dir()]
    print(f">>> Found Surveys: {SURVEYS}")

    answer_validation = False
    while answer_validation == False:
        cube_selection = str(input("\t1. Build datacube for all (Y/n)? "))

        if cube_selection.upper().strip() == "Y":
            print(">>> Proceeding with all surveys...")
            input_survey_list = SURVEYS
            answer_validation = True

        elif cube_selection.upper().strip() == "N":
            cube_validation = False
            while not cube_validation:
                survey_selection = str(input("\tSelect desired cubes (PHANGS, S4G, JPAS,...): "))
                input_survey_list = [x.strip().upper() for x in survey_selection.split(",")]

                available_and_configured = set(SURVEYS).intersection(set(SURVEY_CONFIG.keys()))
                cube_validation = set(input_survey_list).issubset(available_and_configured)

                if cube_validation:
                    print(f">>> Proceeding with selected surveys: {input_survey_list}")
                    answer_validation = True
                else:
                    print(f">>> Error: Please select only from available and configured surveys: {list(available_and_configured)}")

        else:
            print("\tProvide Y/n answer.")

    DRIVERS = {
        "BASE": BASE_Driver(config_dict = SURVEY_CONFIG),
        "PHANGS": PHANGS_Driver(config_dict = SURVEY_CONFIG["PHANGS"]),
        "S4G": S4G_Driver(config_dict = SURVEY_CONFIG["S4G"])
    }

    # ------ Get files ------
    img_files = []
    psf_files = []

    for survey in input_survey_list:
        img_dir = input_dir / survey / "galaxies" / galaxy 
        psf_dir = input_dir / survey / "PSF" 

        # print(img_dir)
        
        current_img_files = DRIVERS[survey].get_files(dir_path = img_dir, mode = "sci")
        current_psf_files = DRIVERS[survey].get_files(dir_path = psf_dir, mode = "psf")
        
        img_files = img_files + current_img_files
        psf_files = psf_files + current_psf_files
    # ----------------- Calculate Survey Resolutions -----------------
    # print(len(img_files))
    # print(len(psf_files))
    print(">>> Calculating survey resolutions...")
    fwhm_dict, valid_files = calculateFWHM(psf_file_list = psf_files, drivers = DRIVERS)
    print(valid_files)
    df_fwhm = pd.DataFrame(list(fwhm_dict.items()), columns=["Filter", "FWHM_arcsec"])
    df_fwhm = df_fwhm.sort_values(by = "FWHM_arcsec").reset_index(drop=True)
    print("\nResolutions Table:\n", df_fwhm)
    psf_master_name = df_fwhm.iloc[-1]['Filter']
    print(f"\n==> Recommended PSF (master): {psf_master_name}")

    # Handle the Master image (it doesn't need convolution, just a copy to the final folder)
    img_by_filter = {}
    # print(img_files)
    for img_path in img_files:
        survey_i = DRIVERS["BASE"].get_survey(file_path = img_path)
        filt_i = DRIVERS[survey_i].get_sci_filter_name(filename = str(img_path))
        img_by_filter[filt_i] = {'path': img_path, 'survey': survey_i}

    master_survey = img_by_filter[psf_master_name]['survey']
    master_img_path = img_by_filter[psf_master_name]['path']
    # =================================================================================================
    # ====================================== CONVOLUTION ALGORITHM ==================================== 
    print(">>> Initiating convolution process...")

    if args.mode == 'full' or args.mode == 'conv_only':
        if args.create_kernel:
            print(">>> Cleaning PSFs...")
            # ------ Create directory (and delete previously existing one) ------
            for survey in input_survey_list:
                clean_psf_dir = input_dir / survey / "PSF_CLEAN"
                if clean_psf_dir.is_dir():
                    print(f"\tRemoving old PSF_CLEAN directory and setting up a new one ({survey})")
                    shutil.rmtree(clean_psf_dir)
                os.mkdir(clean_psf_dir)
                print(f"\tCreated: {clean_psf_dir}\n")

            # ------ PSF cleaning ------
            cleaned_psf_by_filter = {}

            for psf_file_path in psf_files:
                survey = DRIVERS["BASE"].get_survey(file_path = psf_file_path)
                driver = DRIVERS[survey]
                filter_name = driver.get_psf_filter_name(filename = str(psf_file_path))

                output_clean_psf_path = input_dir / survey / "PSF_CLEAN" / psf_file_path.name
                clean_psf(
                    input_file = psf_file_path,
                    output_file = output_clean_psf_path,
                    pixel_scale_arcsec = driver.get_pixel_scale(filter_name = filter_name),
                    binned_factor = driver.get_binned_factor,
                )

                cleaned_psf_by_filter[filter_name] = output_clean_psf_path

            # ------ Pypher kernel creation ------
            print(">>> Initiating kernel creation...")
            # print(master_psf_path)
            # print(cleaned_psf_by_filter)

            pypher_commands = pypher_kernel_creation(cleaned_psf_by_filter = cleaned_psf_by_filter, 
                                psf_master_name = psf_master_name,
                                output_dir = kernel_dir)
            # print(pypher_commands)
            print(f"\n--- Creating {len(pypher_commands)} kernels via PyPHER ---")
            for c in pypher_commands:
                print(f"----- Running: {c} -----")
                try:
                    # Execute PyPHER in the shell; check=True raises an error if it fails
                    subprocess.run(c, shell=True, check=True)
                    print("==> Kernel generated successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"==> PyPHER error: {e}")
                    continue 

            print("\n>>> Kernel processing completed!")
        else:
            # If kernels already exist, skip generation and identify the current Master filter
            print('>>> Matching kernels already exist. Proceeding to image convolution...')
        kernel_files = list(kernel_dir.glob('*.fits'))

        # --- IMAGE CONVOLUTION ---
        convolved_fits_dir_gal = convolved_fits_dir / args.galaxy
        if os.path.exists(convolved_fits_dir_gal):
            print('\n\tConvolution directory already exists!')
        else:
            print('\n\tCreating convolution directory...')
            os.makedirs(convolved_fits_dir_gal, exist_ok=True)

        print(f"IMAGE FILES: {len(img_files)}")
        # Pair images with their specific kernels
        fftconvolve_dict = convolved_dict(
            img_files = img_files,
            kernel_files = kernel_files,
            drivers=DRIVERS,
        )

        # print(fftconvolve_dict)

        for key in fftconvolve_dict:
            original_fits = fftconvolve_dict[key]['img']
            kernel_fits = fftconvolve_dict[key]['kernel']
            survey = fftconvolve_dict[key]['survey']

            # Run the convolution (FFT based)
            convolved_file_path = create_convolvedFITS(
                original_fits = original_fits, kernel_fits = kernel_fits,
                survey = survey, psf_master_name = psf_master_name,
                master_survey = master_survey, output_dir = convolved_fits_dir,
                drivers = DRIVERS,
                force = args.force_convolution
            )

            fftconvolve_dict[key]["convolved_file_path"] = convolved_file_path

        master_dest_path = convolved_fits_dir / galaxy / f'{galaxy}_{master_survey.lower()}_{psf_master_name}_master.fits'
        shutil.copy2(master_img_path, master_dest_path)

        print(100 * '#')
        print(f'Master file {psf_master_name} from {master_survey} survey:\nFITS saved to: {master_dest_path}\n' + 100 * '#')
            
    # # =================================================================================================
    # # ====================================== ALINGMENT ALGORITHM ====================================== 
    if args.mode == 'full' or args.mode == 'alignment_only':
        print(">>> Initiating image alignment process...")
        convolved_files_dict = discover_convolved_files(convolved_fits_dir, galaxy)
        reference_entry = next(v for v in convolved_files_dict.values() if v['is_master'])
        reference_fits = reference_entry['path']

        ref_driver = DRIVERS[reference_entry["survey"]]
        reference_apply_sip = ref_driver.get_sip

        for filt, entry in convolved_files_dict.items():
            if entry['is_master']:
                continue

            img_driver = DRIVERS[entry['survey']]
            img_to_reproject_apply_sip = img_driver.get_sip
            reproject_to_reference(
                img_to_reproject = entry['path'],
                reference_img = reference_fits,
                output_path = reprojected_dir,
                apply_sip_reference_img = reference_apply_sip,
                apply_sip_img_to_reproject = img_to_reproject_apply_sip
            )
        reprojected_dir_gal = reprojected_dir / galaxy
        shutil.copy2(reference_fits, reprojected_dir_gal)
        print(f'\n\tCopied master FITS file: {reprojected_dir_gal / reference_fits}\n')
        
        # print(conv_img_files) # debugging
        # print(master_dest_path) # debugging

    # # =================================================================================================
    # # ====================================== DATA CUBE ALGORITHM ====================================== 
    # if args.mode == 'full' or args.mode == 'alignment_only':



if __name__ == "__main__":
    main()
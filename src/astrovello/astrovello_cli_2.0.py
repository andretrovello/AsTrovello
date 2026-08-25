from config import SURVEY_CONFIG
from drivers import BASE_Driver, PHANGS_Driver, S4G_Driver
from convolution_2_0 import get_fwhm, calculateFWHM
from pathlib import Path
import argparse
import gc
from pathlib import Path
import os
from tqdm import tqdm
from astropy.io import fits
import pandas as pd
import subprocess
import shutil
# import astrovello as aat
import numpy as np

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

    args = parser.parse_args()

    print(100*'#')
    print(f'Executing AsTrovello for {args.galaxy}...\n')

    CWD = Path.cwd()
    BASE_DIR = CWD.parents[1]
    print(f'Root Directory: {BASE_DIR}')

    galaxy = args.galaxy

    input_dir = BASE_DIR / 'Input'
    output_dir = BASE_DIR / "Output"

    if not input_dir.exists():
        print(f"==> Error: 'Input' folder not found in {BASE_DIR}")
        print("Make sure you are in the correct directory.")
        return

    SURVEYS = [x.name for x in input_dir.iterdir() if x.is_dir()]
    print(f">>> Found Surveys: {SURVEYS}")

    answer_validation = False
    while answer_validation == False:
        cube_selection = str(input("\t1. Build datacube for all (Y/n)? "))

        if cube_selection.upper().strip() == "Y":
            print(">>> Proceeding with all surveys...")
            clean_survey_list = SURVEYS
            answer_validation = True

        elif cube_selection.upper().strip() == "N":
            cube_validation = False
            while not cube_validation:
                survey_selection = str(input("\tSelect desired cubes (PHANGS, S4G, JPAS,...): "))
                clean_survey_list = [x.strip().upper() for x in survey_selection.split(",")]

                available_and_configured = set(SURVEYS).intersection(set(SURVEY_CONFIG.keys()))
                cube_validation = set(clean_survey_list).issubset(available_and_configured)

                if cube_validation:
                    print(f">>> Proceeding with selected surveys: {clean_survey_list}")
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
    image_files = []
    psf_files = []

    for survey in clean_survey_list:
        img_dir = input_dir / survey / "galaxies" / galaxy 
        psf_dir = input_dir / survey / "PSF" 

        # print(img_dir)
        
        current_img_files = DRIVERS[survey].get_files(dir_path = img_dir, mode = "sci")
        current_psf_files = DRIVERS[survey].get_files(dir_path = psf_dir, mode = "psf")
        
        image_files = image_files + current_img_files
        psf_files = psf_files + current_psf_files
    # ----------------- Calculate Survey Resolutions -----------------
    # print(len(image_files))
    # print(len(psf_files))
    print(">>> Calculating survey resolutions...")
    fwhm_dict, valid_files = calculateFWHM(psf_file_list = psf_files, drivers = DRIVERS)
    print(valid_files)
    df_fwhm = pd.DataFrame(list(fwhm_dict.items()), columns=["Filter", "FWHM_arcsec"])
    df_fwhm = df_fwhm.sort_values(by = "FWHM_arcsec").reset_index(drop=True)
    print("\nResolutions Table:\n", df_fwhm)
    psf_master_name = df_fwhm.iloc[-1]['Filter']
    print(f"\n==> Recommended PSF (master): {psf_master_name}")

    # =================================================================================================
    # ====================================== CONVOLUTION ALGORITHM ==================================== 
    print(">>> Initianting convolution process...")


    # if args.mode == 'full' or args.mode == 'conv_only':
    #     if args.create_kernel:
            
    # # =================================================================================================
    # # ====================================== ALINGMENT ALGORITHM ====================================== 
    # if args.mode == 'full' or args.mode == 'alignment_only':

    # # =================================================================================================
    # # ====================================== DATA CUBE ALGORITHM ====================================== 
    # if args.mode == 'full' or args.mode == 'alignment_only':



if __name__ == "__main__":
    main()
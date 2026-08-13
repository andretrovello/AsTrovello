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
    # --- COMMAND LINE ARGUMENT PARSING ---
    # Setup the CLI (Command Line Interface) for the Capivara/AsTrovello pipeline
    parser = argparse.ArgumentParser('Capivara Pipeline Control')

    parser.add_argument('--galaxy', type = str, required = True, help = 'Galaxy name (e.g., ngc1566)')

    args = parser.parse_args()

    print(100*'#')
    print(f'Executing AsTrovello for {args.galaxy}...\n')

    # --- DYNAMIC PATH CONFIGURATION ---
    # Identify the location of the running script to define the project root (BASE_DIR)
    # This ensures the code runs correctly regardless of where the AsTrovello folder is placed
    BASE_DIR = Path.cwd().parents[1]
    print(f'Root Directory: {BASE_DIR}')

    # Define Input hierarchy (PHANGS/HST and S4G/Spitzer)
    input_dir = BASE_DIR / 'Input'
    if not input_dir.exists():
        print(f"==> Error: 'Input' folder not found in {BASE_DIR}")
        print("Make sure you are in the correct directory.")
        return 
    

    SURVEY_CONFIG = {
                        "PHANGS": 
                        {
                            "TELESCOP": "HST",
                            "INSTRUME": "WFC3",
                            "pixel_scale_arcsec": 0.0395,
                            "binned_factor": 4,
                            "unit_type": "electrons/s", # usado em units.py
                            "force_tan_sip": False,
                            "psf_suffix": "*PSFSTD*.fits"
                        },
                        "S4G":
                        {
                            "TELESCOP": "Spitzer",
                            "INSTRUME": "IRAC",
                            "pixel_scale_arcsec": 
                            {
                                1: 1.221, # Channel 1
                                2: 1.223 # Channel 2
                            },
                            "binned_factor": 5,
                            "unit_type": "mjy/sr", # usado em units.py
                            "force_tan_sip": True,
                            "psf_suffix": "*_col129_row129.fits"
                        }
                    }


    INPUT_DIR = BASE_DIR / "Input"
    SURVEYS = [x.name for x in INPUT_DIR.iterdir() if x.is_dir()]
    print(f">>> Found Surveys: {SURVEYS}")
    cube_selection = str(input("\t1. Build datacube for all (Y/n)? "))
    answer_validation = False
    while answer_validation == False:
        if cube_selection.upper() == "Y":
            print(">>> Proceeding with all surveys...")
            answer_validation = True
        elif cube_selection.upper() == "N":
            survey_selection = str(input("Select desired cubes (PHANGS, S4G, JPAS,...): "))
            clean_survey_list = survey_selection.split(",")
            clean_survey_list = [x.strip().upper() for x in clean_survey_list]
            cube_validation = set(clean_survey_list).issubset(set(SURVEY_CONFIG.keys()))
            while cube_validation == False:
                if validation:
                    print(f">>> Proceeding with selected surveys: {clean_survey_list}")
                    cube_validation = True
                    answer_validation = True
                else:
                    print(">>> Not all surveys were found. Please select again")
                    cube_validation = False
        else:
            print("\tProvide Y/n answer.")

if __name__ == "__main__":
    main()
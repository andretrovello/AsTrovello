import subprocess
from pathlib import Path

# --- PATH CONFIGURATION ---
# Identify the root directory of the project to locate Input folders
BASE_DIR = Path(__file__).resolve().parent.parent
input_dir = BASE_DIR / 'Input'

# Define paths for the galaxy folders in both surveys
input_phangs = input_dir / 'PHANGS' / 'galaxies' / 'phangs_hst'
input_s4g = input_dir / 'S4G' / 'galaxies'  

# --- GALAXY LISTING ---
# Generate sorted lists of directory names (galaxy names like ngc1566)
# iterdir() scans the folder and x.is_dir() ensures we only get folders, not files
gal_names_phangs = sorted([x.name for x in input_phangs.iterdir() if x.is_dir()])
gal_names_s4g = sorted([x.name for x in input_s4g.iterdir() if x.is_dir()])

# --- EXECUTION PARAMETERS ---
mode = 'full'
apply_mask = '--apply_mask'
create_kernel = '--create_kernel'
sigma = '1.0'

# --- BATCH PROCESSING LOGIC ---
# Only proceed if the list of galaxies is identical across both surveys
if gal_names_phangs == gal_names_s4g:
    print(f'Galaxies found: {len(gal_names_phangs)} - {gal_names_phangs}')
    print(100*'*')
    
    # Iterate through the list of galaxies
    for i, galaxy in enumerate(gal_names_phangs):
        
        # LOGIC: Only generate kernels for the first galaxy (i == 0)
        # This prevents the pipeline from recalculating the same kernels (e.g., F275W to IRAC2) 
        # multiple times, since kernels are often instrument-dependent, not galaxy-dependent.
        extra_args = f"{apply_mask} {create_kernel}" if i == 0 else f"{apply_mask}"
        
        # Shell command
        cmd = (
            f"python AsTrovello_run.py --mode {mode} --galaxy {galaxy} "
            f"{extra_args} --sigma {sigma}"
        )
        
        # Execute the main pipeline script for the current galaxy
        # shell=True allows running the command as a string
        # check=True stops the entire batch if one galaxy fails
        subprocess.run(cmd, shell=True, check=True)
else: 
    # Safety check to avoid mismatched data between HST and Spitzer folders
    print('Lists do not match!. Guarantee that all surveys have the same galaxies')
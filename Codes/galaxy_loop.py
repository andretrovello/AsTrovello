import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
input_dir = BASE_DIR / 'Input'
input_phangs = input_dir / 'PHANGS' / 'galaxies' / 'phangs_hst'
input_s4g = input_dir / 'S4G' / 'galaxies'  

gal_names_phangs = sorted([x.name for x in input_phangs.iterdir() if x.is_dir()])
gal_names_s4g = sorted([x.name for x in input_s4g.iterdir() if x.is_dir()])

mode = 'full'
apply_mask = '--apply_mask'
create_kernel = '--create_kernel'
sigma = '1.0'

if gal_names_phangs == gal_names_s4g:
    print(f'Galaxies found: {len(gal_names_phangs)} - {gal_names_phangs}')
    print(100*'*')
    for i, galaxy in enumerate(gal_names_phangs):
        # Definimos os argumentos extras baseados na condição
        extra_args = f"{apply_mask} {create_kernel}" if i == 0 else f"{apply_mask}"
        
        # A mágica acontece aqui: as strings se juntam ignorando a indentação
        cmd = (
            f"python AsTrovello_run.py --mode {mode} --galaxy {galaxy} "
            f"{extra_args} --sigma {1.0}"
        )
        
        subprocess.run(cmd, shell=True, check=True)
else: 
    print('Lists do not match!. Guarantee that all surveys have the same galaxies')
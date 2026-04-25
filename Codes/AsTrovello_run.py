import argparse
import gc
from pathlib import Path
import os
from tqdm import tqdm
from astropy.io import fits
import pandas as pd
import subprocess
import shutil
from scipy.signal import fftconvolve
from collections import defaultdict
from AsTrovello_lib import * 


def main():
    parser = argparse.ArgumentParser('Capivara Pipeline Control')

    parser.add_argument('--mode', type = str, choices = ['full', 'alignment_only', 'conv_only', 'cube_only'], \
        default = 'full', help = 'Execution mode')
    parser.add_argument('--galaxy', type = str, required = True, help = 'Galaxy name')
    parser.add_argument('--create_kernel', action='store_true', help='Apply sky mask')
    parser.add_argument('--apply_mask', action='store_true', help='Create kerne? (True/False)')
    parser.add_argument('--sigma', type = float, default = 1.0, help = 'Sigma value for sky mask cut')

    args = parser.parse_args()

    print(100*'#')
    print(f'Executing AsTrovello for {args.galaxy}...\n')
    # Find script path (inside Codes dir)
    script_path = Path(__file__).resolve()

    # Obtain base dir
    BASE_DIR = script_path.parent.parent
    print(f'Root Directory: {BASE_DIR}')

    # Input paths
    input_dir = BASE_DIR / 'Input'
    # 1. PHANGS
    phangs_dir = input_dir / "PHANGS" 
    phangs_imgs = phangs_dir / "phangs_hst" / args.galaxy.lower()
    phangs_psf_path = phangs_dir / 'PSF'
    # 2. S4G
    s4g_dir = input_dir / "S4G"
    s4g_imgs = s4g_dir / args.galaxy.lower()
    s4g_psf_path = s4g_dir / 'PSF' 

    # Output paths
    output_dir = BASE_DIR / 'Output'
    reprojected_path = output_dir / 'reprojected_files' 
    kernel_dir = output_dir / 'PSF_Kernels'
    convolved_fits_path = output_dir / 'convolved_fits' 
    

# ------------------------------------------------ Image alignment --------------------------------------------------
    if args.mode == 'full' or args.mode == 'alignment_only':
        print('Starting image reprojection and alignment...\n')


        sci_files_phangs = list(phangs_imgs.glob('*exp-drc-sci.fits'))
        sci_files_s4g = list(s4g_imgs.glob('*.fits'))

        phangs_ref_file =  sci_files_phangs[0]

        for file in sci_files_s4g:
            S4G2PHANGS_reproject(file, phangs_ref_file, output_path = reprojected_path)

# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Image convolution --------------------------------------------------

    if args.mode == 'full' or args.mode == 'conv_only':
        if args.create_kernel:
            print('Initiating convolution process...\n')

            # Pegamos os dados e a lista de arquivos reais
            fwhm_s4g, _, files_s4g = calculaFWHM_radial_profile(s4g_psf_path)
            fwhm_phangs, _, files_phangs = calculaFWHM_radial_profile(phangs_psf_path)
                
            todos_fwhm = {**fwhm_s4g, **fwhm_phangs}
            df_fwhm = pd.DataFrame(list(todos_fwhm.items()), columns=['Filtro', 'FWHM_pixels'])
            df_fwhm = df_fwhm.sort_values(by='FWHM_pixels').reset_index(drop=True)

            print("\nResolutions Table:\n", df_fwhm)

            psf_master_name = df_fwhm.iloc[-1]['Filtro']
            print(f"\n⭐ Recommended PSF (master): {psf_master_name}")
            
            # Dicionário para facilitar o loop de limpeza
            survey_data = {
                'PHANGS': {'path': phangs_psf_path, 'files': files_phangs},
                'S4G': {'path': s4g_psf_path, 'files': files_s4g}
            }

            print(f'\nInitiating PSF cleaning...')
            for s_name, s_info in survey_data.items():
                print(f'\nCleaning {s_name} PSFs...')
                input_p = os.path.expanduser(s_info['path'])
                output_p = os.path.expanduser(str(s_info['path']) + '_LIMPAS')

                os.makedirs(output_p, exist_ok=True)

                for f_name in s_info['files']:
                    final_clean_psf(os.path.join(input_p, f_name), os.path.join(output_p, f_name))

            if psf_master_name.upper().startswith('F'):
                psf_master_path = phangs_dir / 'PSF_LIMPAS' / f'PSFSTD_WFC3UV_{psf_master_name.upper()}.fits'
            elif psf_master_name.upper().startswith('I'):
                psf_master_path = s4g_dir / 'PSF_LIMPAS' / f'{psf_master_name.upper()}_col129_row129.fits'
            comandos_pypher = pypher_kernel_creation(todos_fwhm, psf_master_path, input_dir, kernel_dir)

            print(f"\n--- Creating {len(comandos_pypher)} kernels via PyPHER ---")
        
            for c in comandos_pypher:
                print(f"🚀 Runnning: {c}")
                try:
                    # shell=True permite rodar o comando como string
                    # check=True faz o Python dar erro se o PyPHER falhar
                    subprocess.run(c, shell=True, check=True)
                    print("✅ Kernel generated succesfully!")
                except subprocess.CalledProcessError as e:
                    print(f"❌ PyPHER error: {e}")
                    # Aqui você decide se quer parar tudo ou continuar para o próximo
                    continue 

            print("\nKernel processing completed!")

        else:
            print('Matching kernels already exist. Proceeding to image convolution...')
            kernel_files = list(kernel_dir.glob('*.fits'))
            psf_master_name = str(kernel_files[0].stem).split('_')[-1]
        
        convolved_fits_path_gal = convolved_fits_path / args.galaxy

        if os.path.exists(convolved_fits_path_gal):
            print(f'Removing existing convolved directory: {convolved_fits_path_gal}')
            shutil.rmtree(convolved_fits_path_gal)
        

        fftconvolve_dict = convolved_dict(path_phangs = phangs_imgs, \
            path_s4g_reprojected = reprojected_path / args.galaxy, path_kernels = kernel_dir)

        for k in fftconvolve_dict.keys():
            original_fits = fftconvolve_dict[k]['img']['path']
            kernel_fits = fftconvolve_dict[k]['kernel']['path']

            galaxy_name = create_convolvedFITS(original_fits , kernel_fits, output_dir = convolved_fits_path, GAL_NAME = True)

        master_source_dir = reprojected_path / galaxy_name
        master_source_name = f'{galaxy_name}_s4g_{psf_master_name}_on_phangs_projection.fits'
        master_source_path = master_source_dir / master_source_name

        master_dest_path = convolved_fits_path / args.galaxy / f'{galaxy_name}_s4g_irac2_master.fits'

        # Ensure the destination directory exists
        os.makedirs(convolved_fits_path, exist_ok=True)

        # Copy with metadata preservation
        shutil.copy2(master_source_path, master_dest_path)

        print(100 * '#')
        print(f'Master file {psf_master_name} from s4g survey:')
        print(f'FITS saved to: {master_dest_path}')
        print(100 * '#')

# -------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- FITS unit conversion-----------------------------------------------------
        print(100 * '#')
        print('\n ----- Starting unit coversion -----')
        convolved_fits_path_gal = convolved_fits_path / args.galaxy
        conv_files = list(convolved_fits_path_gal.glob('*.fits'))

        for file in conv_files:
            filter_name = file.name.split('_')[-2]
            print(f'Converting {filter_name} image units to Jy/pixel:')
            new_data, new_header = convert2Jansky(file)

            new_hdu = fits.PrimaryHDU(new_data, new_header)
            file_path = file.parent / f'{file.stem}_Jy_per_pixel{file.suffix}'

            new_hdu.writeto(file_path, overwrite=True) 
            print(f'Saved file in: {file_path}')
            print(100 * '#')

# -------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Hypercube creation -----------------------------------------------------
    if args.mode == 'full' or args.mode == 'cube_only':
        # CONFIGURAÇÕES
        VALOR_N_SIGMA = args.sigma # Valor sugerido para NGC 1087
        APLICAR_MASCARA = args.apply_mask

        output_dir_cube = output_dir / 'datacubes' / args.galaxy
        loc = convolved_fits_path / args.galaxy

        if not os.path.exists(output_dir_cube):
            os.makedirs(output_dir_cube)

        file_list = list(loc.glob('*_Jy_per_pixel.fits'))
        ref_file = list(phangs_imgs.glob('*f275w*sci.fits')) 

        print(f"Arquivos encontrados: {len(file_list)}") # Adicione isso
        if not file_list: 
            print("Erro: A lista de arquivos está vazia! Verifique os caminhos.")
            return

        fits_files = sorted([(f, f.name.split('_')[2]) for f in file_list])
        ref_header = fits.getheader(fits_files[0][0], ext=0)

        aligned_images, filter_names = [], []
        for file, filt in tqdm(fits_files, desc="Lendo FITS"):
            aligned_images.append(fits.getdata(file, ext=0).astype(np.float32))
            filter_names.append(filt)

        if aligned_images:
            temp_name = output_dir_cube / 'temp_cube.fits'
            cube, cube_header = create_data_cube(
                aligned_images, filter_names, ref_file, ref_header, 
                temp_name, aplicar_mask=APLICAR_MASCARA, N_SIGMA=VALOR_N_SIGMA
            )

            # RENOMEAÇÃO DINÂMICA (Baseada no objeto em memória)
            _, final_ny, final_nx = cube.shape
            final_name = output_dir_cube / f'{args.galaxy}_datacube_sci_{final_nx}x{final_ny}_Jy_per_pixel.fits'
            temp_name.rename(final_name)
            print(f"✅ Cubo finalizado: {final_name.name}")

            del aligned_images
            gc.collect()
            
            # ZOOM AUTOMATIZADO (Centralizado no CRPIX)
            cx, cy = int(cube_header['CRPIX1']), int(cube_header['CRPIX2'])
            regions = [(output_dir_cube / f'{args.galaxy}_datacube_sci_{final_nx}x{final_ny}_zoomed.fits', 
                        cx-150, cx+150, cy-150, cy+150)]
            create_cutouts(cube, cube_header, regions)

if __name__ == "__main__":
    main()
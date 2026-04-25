import argparse
import gc
from pathlib import Path
import os
from tqdm import tqdm
from astropy.io import fits
import pandas as pd
import subprocess
from scipy.signal import fftconvolve
from collections import defaultdict
from AsTrovello_lib import * 


def main():
    parser = argparse.ArgumentParser('Capivara Pipeline Control')

    parser.add_argument('--mode', type = str, choices = ['full', 'alignment_only', 'conv_only', 'cube_only'], \
        default = 'full', help = 'Execution mode')
    parser.add_argument('--galaxy', type = str, required = True, help = 'Galaxy name')
    parser.add_argument('--apply_mask', action='store_true', help='Apply sky mask')
    parser.add_argument('--create_kernel', action='store_true', help='Create PSF matching kernels')
    parser.add_argument('--sigma', type = float, default = 1.0, help = 'Sigma value for sky mask cut')

    args = parser.parse_args()

# ------------------------------------------------ Image alignment --------------------------------------------------
    if args.mode == 'full' or args.mode == 'alignment_only':
        print('Starting image reprojection and alignment...')
        path_phangs = Path(f'~/Desktop/AsTrovello/Input/PHANGS/phangs_hst/{args.galaxy}/images').expanduser()
        path_s4g = Path(f'~/Desktop/AsTrovello/Input/S4G/{args.galaxy}').expanduser()

        sci_files_phangs = list(path_phangs.glob('*exp-drc-sci.fits'))
        sci_files_s4g = list(path_s4g.glob('*.fits'))

        print(sci_files_s4g)
        phangs_ref_file =  sci_files_phangs[0]

        for file in sci_files_s4g:
            S4G2PHANGS_reproject(file, phangs_ref_file)

# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Image convolution --------------------------------------------------

    if args.mode == 'full' or args.mode == 'conv_only':
        if args.create_kernel:
            print('Initiating convolution process...\n')
            
            # Caminhos base
            path_s4g = '~/Desktop/AsTrovello/Input/S4G/PSF'
            path_phangs = '~/Desktop/AsTrovello/Input/PHANGS/PSF'

            # Pegamos os dados e a lista de arquivos reais
            fwhm_s4g, _, files_s4g = calculaFWHM_radial_profile(path_s4g)
            fwhm_phangs, _, files_phangs = calculaFWHM_radial_profile(path_phangs)
                
            todos_fwhm = {**fwhm_s4g, **fwhm_phangs}
            df_fwhm = pd.DataFrame(list(todos_fwhm.items()), columns=['Filtro', 'FWHM_pixels'])
            df_fwhm = df_fwhm.sort_values(by='FWHM_pixels').reset_index(drop=True)

            print("\nTabela de Resoluções:\n", df_fwhm)

            psf_master_name = df_fwhm.iloc[-1]['Filtro']
            print(f"\n⭐ PSF Master recomendada: {psf_master_name}")
            
            # Dicionário para facilitar o loop de limpeza
            survey_data = {
                'PHANGS': {'path': path_phangs, 'files': files_phangs},
                'S4G': {'path': path_s4g, 'files': files_s4g}
            }

            print(f'\nInitiating PSF cleaning...')
            for s_name, s_info in survey_data.items():
                print(f'\nCleaning {s_name} PSFs...')
                input_p = os.path.expanduser(s_info['path'])
                output_p = os.path.expanduser(s_info['path'] + '_LIMPAS')

                os.makedirs(output_p, exist_ok=True)

                for f_name in s_info['files']:
                    final_clean_psf(os.path.join(input_p, f_name), os.path.join(output_p, f_name))

            comandos_pypher = pypher_kernel_creation(todos_fwhm, psf_master_name)

            print(f"\n--- Iniciando geração de {len(comandos_pypher)} kernels via PyPHER ---")
        
            for c in comandos_pypher:
                print(f"🚀 Rodando: {c}")
                try:
                    # shell=True permite rodar o comando como string
                    # check=True faz o Python dar erro se o PyPHER falhar
                    subprocess.run(c, shell=True, check=True)
                    print("✅ Kernel gerado com sucesso!")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Erro ao rodar o PyPHER: {e}")
                    # Aqui você decide se quer parar tudo ou continuar para o próximo
                    continue 

            print("\nProcessamento de Kernels finalizado!")

        else:
            print('Matching kernels already exist. Proceeding to image convolution...')
        
        fftconvolve_dict = convolved_dict()

        for k in fftconvolve_dict.keys():
            original_fits = fftconvolve_dict[k]['img']['path']
            kernel_fits = fftconvolve_dict[k]['kernel']['path']

            galaxy_name = create_convolvedFITS(original_fits , kernel_fits, output_dir = '~/Desktop/AsTrovello/Input/convolved_fits', GAL_NAME = True)

        master_source_dir = Path(f'~/Desktop/AsTrovello/Input/reprojected_files/{galaxy_name}').expanduser()
        master_source_name = f'{galaxy_name}_s4g_irac2_on_phangs_projection.fits'
        master_source_path = os.path.join(master_source_dir, master_source_name)

        master_dest_dir = Path(f'~/Desktop/AsTrovello/Input/convolved_fits/{galaxy_name}').expanduser()
        master_dest_name = f'{galaxy_name}_s4g_irac2_master.fits'
        master_dest_path = os.path.join(master_dest_dir, master_dest_name)

        # Ensure the destination directory exists
        os.makedirs(master_dest_dir, exist_ok=True)

        # Copy with metadata preservation
        shutil.copy2(master_source_path, master_dest_path)

        print(100 * '#')
        print(f'Arquivo master do filtro irac2 do survey s4g:')
        print(f'FITS salvo em: {master_dest_path}')
        print(100 * '#')

# -------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- FITS unit conversion-----------------------------------------------------
        print(100 * '#')
        print('Starting unit coversion...')
        conv_dir = Path('~/Desktop/AsTrovello/Input/convolved_fits/ngc1087').expanduser()
        conv_files = list(conv_dir.glob('*.fits'))

        for file in conv_files:
            filter_name = file.name.split('_')[-2]
            print(f'Convertendo unidades do cubo {filter_name} para Jy/pixel:')
            new_data, new_header = convert2Jansky(file)

            new_hdu = fits.PrimaryHDU(new_data, new_header)
            file_path = file.parent / f'{file.stem}_Jy_per_pixel{file.suffix}'
            print(file_path)

            new_hdu.writeto(file_path, overwrite=True) 
            print(f'Arquivo salvo em: {file_path}')
            print(100 * '#')

# -------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Hypercube creation -----------------------------------------------------
    if args.mode == 'full' or args.mode == 'cube_only':
        # CONFIGURAÇÕES
        VALOR_N_SIGMA = args.sigma # Valor sugerido para NGC 1087
        APLICAR_MASCARA = args.apply_mask
        output_dir = Path("~/Desktop/AsTrovello/Input/").expanduser()
        loc = Path(f"~/Desktop/AsTrovello/Input/convolved_fits/{args.galaxy}").expanduser()
        ref_dir = Path(f'~/Desktop/AsTrovello/Input/PHANGS/phangs_hst/{args.galaxy}/images/').expanduser()
        
        file_list = list(loc.glob('*_Jy_per_pixel.fits'))
        ref_file = list(ref_dir.glob('*f275w*sci.fits')) 

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
            temp_name = output_dir / 'temp_cube.fits'
            cube, cube_header = create_data_cube(
                aligned_images, filter_names, ref_file, ref_header, 
                temp_name, aplicar_mask=APLICAR_MASCARA, N_SIGMA=VALOR_N_SIGMA
            )

            # RENOMEAÇÃO DINÂMICA (Baseada no objeto em memória)
            _, final_ny, final_nx = cube.shape
            final_name = output_dir / f'datacube_sci_{final_nx}x{final_ny}_Jy_per_pixel.fits'
            temp_name.rename(final_name)
            print(f"✅ Cubo finalizado: {final_name.name}")

            del aligned_images
            gc.collect()
            
            # ZOOM AUTOMATIZADO (Centralizado no CRPIX)
            cx, cy = int(cube_header['CRPIX1']), int(cube_header['CRPIX2'])
            regions = [(output_dir / f'datacube_sci_{final_nx}x{final_ny}_zoomed.fits', 
                        cx-150, cx+150, cy-150, cy+150)]
            create_cutouts(cube, cube_header, regions)

if __name__ == "__main__":
    main()
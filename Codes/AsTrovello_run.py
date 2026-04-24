import argparse
import gc
from pathlib import Path
import os
from tqdm import tqdm
from astropy.io import fits
from AsTrovello_lib import * 
import pandas as pd


def main():
    parser = argparse.ArgumentParser('Capivara Pipeline Control')

    parser.add_argument('--mode', type = str, choices = ['full', 'alignment_only', 'conv_only', 'cube_only'], \
        default = 'full', help = 'Execution mode')
    parser.add_argument('--galaxy', type = str, required = True, help = 'Galaxy name')
    parser.add_argument('--apply_mask', action='store_true', help='Apply sky mask')
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

# --------------------------------------------------------------------------------------------------------------------
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
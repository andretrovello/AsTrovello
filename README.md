<p align="center">
  <img src="AsTrovello_logo.png" alt="AsTrovello Logo" width="300">
</p>

## AsTrovello

**AsTrovello** is a Python library for multi-wavelength galaxy data analysis. It integrates high-resolution photometry from **PHANGS (HST)** with mid-infrared data from **S4G (Spitzer/IRAC)**, providing tools for image alignment, PSF homogenization, unit conversion, and 3D hypercube creation.

---

## Main Features

* **Image Alignment:** Reprojects S4G images onto the HST pixel grid (conserving surface brightness).
* **PSF Homogenization:** Calculates FWHM and generates convolution kernels via **PyPHER** to match the resolution of different filters to a common Master PSF.
* **Unit Standardization:** Converts `ELECTRONS/S` (HST) and `MJy/sr` (Spitzer) to `Jy/pixel`.
* **Hypercube Creation:** Builds a 3D FITS hypercube (RA, Dec, Filter) with automated sky masking and spatial bounding box cropping.

---

## Data Sources

The pipeline is designed to work with the following public datasets:

**S4G (Spitzer):** https://irsa.ipac.caltech.edu/data/SPITZER/S4G/
```
└── <target>.phot.1.fits and <target>.phot.2.fits
```

**PHANGS (HST):** https://archive.stsci.edu/hlsp/phangs
```
└── hlsp_phangs-hst_hst_wfc3-uvis_<target>_<filter>_v1_exp-drc-sci.fits
```

---

## Installation

AsTrovello requires Python >= 3.10 and is recommended to run inside a Conda environment on Linux.

**From PyPI:**
```bash
pip install astrovello
```

**From GitHub (latest development version):**
```bash
pip install git+https://github.com/your-username/AsTrovello.git
```

**Setting up a dedicated environment:**
```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install astrovello
```

---

## How to Run the Command Line Interface (CLI)

AsTrovello can be executed as a CLI using

```bash
astrovello --galaxy [GALAXY_NAME] --mode [MODE] [FLAGS]
```
### Execution Modes

The pipeline can be executed in different stages depending on your needs. Use the `--mode` argument to select one of the following:

| Mode | Description |
| :--- | :--- |
| **`full`** | **Complete Pipeline:** Executes alignment, PSF homogenization, convolution, and hypercube creation in a single run. |
| **`alignment_only`** | **Spatial Alignment:** Only reprojects S4G images to the PHANGS pixel grid. Use this to check coordinate consistency. |
| **`conv_only`** | **Resolution Matching:** Performs PSF cleaning, kernel generation via PyPHER, and image convolution. |
| **`cube_only`** | **Data Integration:** Skips processing and builds the final 3D hypercube using existing convolved files. |

> **Note:** When using `conv_only` or `full`, remember to include the `--create_kernel` flag if you need to generate new homogenization kernels for the current galaxy.

The command must be executed inside the same directory where galaxy data and survey PSFs are located. These inputs need to be organized as follows:

```text
BASE_DIRECTORY/
├── Input/
│   ├── PHANGS/             # HST images and PSF models
│   |    ├── galaxies/
|   |    |    ├── galaxies/
|   |    |        ├── phangs_hst/
|   |    |            ├── ngc.../
|   |    ├── PSF/
│   └── S4G/                # Spitzer/IRAC images and PSFs
│        ├── galaxies/
|        |    ├── galaxies/
|        |        ├── ngc.../
|        ├── PSF/
```


### Usage Examples

Here are the most common ways to run the **AsTrovello** pipeline:

#### 1. Full Processing (Standard)
Runs everything from alignment to the final hypercube. Ideal for a first-time run on a new galaxy.
```bash
python Codes/AsTrovello_run.py --galaxy ngc1566 --mode full --create_kernel --apply_mask --sigma 1.5
```

## Quick Start

```python
from astrovello import S4G2PHANGS_reproject, create_data_cube, convert2Jansky

# Reproject a Spitzer image onto the PHANGS pixel grid
S4G2PHANGS_reproject(s4g_file, phangs_ref_file, output_path)

# Convert image units to Jy/pixel
data, header = convert2Jansky(fits_file)

# Build a 3D hypercube from aligned images
cube, cube_header = create_data_cube(aligned_images, filter_names, ref_file, header, "output.fits")
```

---

## Project Structure

```
AsTrovello/
├── src/
│   └── astrovello/
│       ├── __init__.py
│       ├── alignment.py     # Image reprojection (S4G → PHANGS grid)
│       ├── convolution.py   # PSF/FWHM estimation and kernel convolution
│       ├── units.py         # Flux unit conversion to Jy/pixel
│       ├── masking.py       # Sky background subtraction and signal masking
│       ├── cube.py          # 3D FITS hypercube creation and cutouts
│       └── utils.py         # Utility functions
├── examples/
│   ├── AsTrovello_run.py    # Main execution script
│   └── galaxy_loop.py       # Batch processing for multiple galaxies
├── pyproject.toml
└── README.md
```

---

## API Reference

### alignment
`S4G2PHANGS_reproject(s4g_file_path, phangs_ref_file_path, output_path)` — Aligns a Spitzer/IRAC image to the HST pixel grid using WCS reprojection.

### convolution
`calculaFWHM_radial_profile(files_path)` — Estimates FWHM from PSF files using radial profiles.  
`final_clean_psf(input_file, output_file)` — Standardizes PSF headers for PyPHER compatibility.  
`pypher_kernel_creation(todos_fwhm, psf_master_path, input_dir, output_dir)` — Generates PyPHER shell commands to create homogenization kernels.  
`create_convolvedFITS(original_fits, kernel_fits, output_dir)` — Applies a convolution kernel to an image via FFT.

### units
`convert2Jansky(fits_file)` — Converts HST (`ELECTRONS/S`) and Spitzer (`MJy/sr`) images to `Jy/pixel`.

### masking
`mask(data, N_SIGMA=3)` — Sky subtraction and signal mask generation using MAD-based sigma clipping.

### cube
`create_data_cube(aligned_images, filter_names, ref_file, ref_header, output_filename)` — Builds a 3D FITS hypercube with masking and bounding box cutout.  
`create_cutouts(cube, cube_header, regions)` — Extracts spatial sub-cubes from the main hypercube.

---

## Requirements

All dependencies are installed automatically via pip. Main dependencies:

`numpy` `astropy` `scipy` `matplotlib` `reproject` `photutils` `pypher` `tqdm` `psutil`

Full list available in `pyproject.toml`.

---

## License

MIT License. See `LICENSE` for details.

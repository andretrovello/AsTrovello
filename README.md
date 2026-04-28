
<p align="center">
  <img src="AsTrovello_logo.png" alt="AsTrovello Logo" width="300">
</p>

## AsTrovello

**AsTrovello** is a Python-based astronomical data analysis framework designed to process and align multi-wavelength galaxy data. It focuses on integrating high-resolution photometry from **PHANGS (HST)** with mid-infrared data from **S4G (Spitzer/IRAC)**.

The pipeline performs image reprojection, PSF (Point Spread Function) homogenization through convolution, unit conversion to Jansky, and final 3D hypercube creation.
 
The respective data cubes can be found downloaded in:
**S4G:** https://irsa.ipac.caltech.edu/data/SPITZER/S4G/
```text
    ├── File: <target>.phot.1.fits and <target>.phot.2.fits
```
**PHANGS (HST images):** https://archive.stsci.edu/hlsp/phangs 
```text
    ├── File: File: HST science images (drz) -> hlsp_phangs-hst_hst_wfc3-uvis_<target>_<filter>_v1_exp-drc-sci.fits
```

---

## Main Features

* **Image Alignment:** Reprojects S4G images onto the HST pixel grid (conserving surface brightness).
* **PSF Homogenization:** Calculates FWHM and generates convolution kernels via **PyPHER** to match the resolution of different filters to a common "Master" PSF.
* **Unit Standardization:** Automatically converts `ELECTRONS/S` (HST) and `MJy/sr` (Spitzer) to `Jy/pixel`.
* **Hypercube Creation:** Builds a 3D FITS hypercube (RA, Dec, Filter) with automated sky masking and spatial cropping (Bounding Box).

---

## Installation & Requirements

The pipeline runs on **Ubuntu Linux** and is optimized for use within a Conda environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AsTrovello.git
    cd AsTrovello
    ```

2.  **Setup the environment:**
    We recommend using the provided `environment.yml` 
    ```bash
      conda env create -f environment.yml -n new_env_name
    ```
    or creating a dedicated environment with:
    ```bash
    conda create -n capivara python=3.10
    conda activate capivara
    pip install astropy reproject scipy tqdm photutils pypher pandas
    ```

---

## How to Run

The main execution script is `AsTrovello_run.py`, located in the `Codes/` directory.

### Basic Syntax
```bash
python Codes/AsTrovello_run.py --galaxy [GALAXY_NAME] --mode [MODE] [FLAGS]
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

### Usage Examples

Here are the most common ways to run the **AsTrovello** pipeline:

#### 1. Full Processing (Standard)
Runs everything from alignment to the final hypercube. Ideal for a first-time run on a new galaxy.
```bash
python Codes/AsTrovello_run.py --galaxy ngc1566 --mode full --create_kernel --apply_mask --sigma 1.5
```
## Project Structure

The repository is organized to separate source code, documentation, and data surveys. Below is the standard directory tree:

```text
AsTrovello/
├── Codes/
│   ├── AsTrovello_run.py   # Main execution script (Master)
│   ├── AsTrovello_lib.py   # Core functions library
│   └── galaxy_loop.py     # Automation for multiple galaxies
├── Input/
│   ├── PHANGS/             # HST images and PSF models
│       ├── galaxies/
|       |    ├── galaxies/
|       |        ├── phangs_hst/
|       |            ├── ngc.../
|       ├── PSF/
│   └── S4G/                # Spitzer/IRAC images and PSFs
│       ├── galaxies/
|       |    ├── galaxies/
|       |        ├── ngc.../
|       ├── PSF/
└── Output/                 # Processed FITS and Hypercubes
```

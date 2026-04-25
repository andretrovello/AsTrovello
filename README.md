
<p align="center">
  <img src="AsTrovello_logo.png" alt="AsTrovello Logo" width="300">
</p>

## AsTrovello

**AsTrovello** (aka Capivara Pipeline) is a Python-based astronomical data analysis framework designed to process and align multi-wavelength galaxy data. It focuses on integrating high-resolution photometry from **PHANGS (HST)** with mid-infrared data from **S4G (Spitzer/IRAC)**.

The pipeline performs image reprojection, PSF (Point Spread Function) homogenization through convolution, unit conversion to Jansky, and final 3D hypercube creation.

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
    git clone [https://github.com/your-username/AsTrovello.git](https://github.com/your-username/AsTrovello.git)
    cd AsTrovello
    ```

2.  **Setup the environment:**
    We recommend using the provided `environment.yml` or creating a dedicated environment with:
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

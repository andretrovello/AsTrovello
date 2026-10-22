<p align="center">
  <img src="https://raw.githubusercontent.com/andretrovello/AsTrovello/main/AsTrovello_logo.png" alt="AsTrovello Logo" width="300">
</p>

## AsTrovello
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23569095.svg)](https://doi.org/10.5281/zenodo.23569095)

**AsTrovello** builds multi-survey datacubes of a galaxy for spatially-resolved
SED fitting. It takes **PHANGS-HST**, **PHANGS-JWST** and **S4G (Spitzer/IRAC)**
imaging and produces a single FITS cube in **Jy/pixel**, at a common angular
resolution, on a common pixel grid.

---

## What it does

1. **PSF matching and convolution.** Every band is degraded to the resolution of
   the worst-resolved band (the *master*, chosen by the r80 of its PSF).
   Kernels are built with **PyPHER** on the grid where they are applied.
2. **Reprojection.** Every band is resampled onto the master's pixel grid.
3. **Unit conversion.** `ELECTRONS/S` (HST), `MJy/sr` (JWST, Spitzer) to `Jy/pixel`.
4. **Datacube.** Bands are stacked by wavelength, sky-subtracted, masked and cropped.

Two validation gates abort the run on failure: PSF matching (r50 ratio within
5%) after the kernels are built, and flux conservation (< 1%) after reprojection.

The pipeline trusts the input WCS: it **reprojects**, it does not measure or
correct astrometric offsets between surveys.

---

## Installation

Python >= 3.10. A dedicated Conda environment is recommended.

```bash
conda create -n astrovello python=3.11
conda activate astrovello
pip install astrovello
```

Latest development version, from GitHub:

```bash
pip install git+https://github.com/andretrovello/AsTrovello.git
```

From a local clone (editable, for development):

```bash
git clone https://github.com/andretrovello/AsTrovello.git
cd AsTrovello
pip install -e .
```

This installs three commands: `astrovello` (the pipeline),
`astrovello-check-psf` and `astrovello-check-flux` (standalone validation
reports). PyPHER is installed as a dependency and is called as the `pypher`
command, so run AsTrovello with its environment activated.

---

## Running

AsTrovello works on a **project directory** of your choice. Put your data in an
`Input/` folder there and run the command from that directory; the pipeline
builds `Output/` next to it.

```text
my_project/                      <- run `astrovello` from here
├── Input/
│   ├── PHANGS-HST/
│   │   ├── galaxies/<galaxy>/   hlsp_phangs-hst_..._<filter>_v1_exp-drc-sci.fits
│   │   └── PSF/                 PSFSTD_WFC3UV_<FILTER>.fits
│   ├── PHANGS-JWST/
│   │   ├── galaxies/<galaxy>/   hlsp_phangs-jwst_jwst_<instr>_..._v1p1_img.fits
│   │   └── PSF/                 WebbPSF PSFs (*PSF*.fits)
│   └── S4G/
│       ├── galaxies/<galaxy>/   <GALAXY>.phot.1.fits, <GALAXY>.phot.2.fits
│       └── PSF/                 IRAC PRFs (*_col129_row129.fits)
└── Output/                      created by the pipeline
```

The survey directory names must be exactly `PHANGS-HST`, `PHANGS-JWST` and
`S4G`: the survey of each file is identified from its path. You only need the
surveys you want to combine.

```bash
cd my_project
astrovello --galaxy ngc1087 --surveys PHANGS-HST,PHANGS-JWST --create_kernel
```

`--surveys` chooses which surveys to combine: a comma-separated list, or `all`
for every configured survey found in `Input/`. The choice sets the master
filter and so the whole cube, so a name that is not available stops the run
instead of being skipped. Without `--surveys` the pipeline asks
interactively. That needs a terminal, so pass `--surveys` for batch or
background runs. Use `--root /path/to/my_project` to run it from elsewhere,
and `--preflight_only` to validate the configuration in seconds without
processing anything. `astrovello --help` lists every option.

Several galaxies in a row:

```bash
for g in ngc1087 ngc1433 ngc2903; do
  astrovello --galaxy $g --surveys PHANGS-HST,PHANGS-JWST --create_kernel || break
done
```

Keep `--create_kernel` for every galaxy: `Output/PSF_Kernels/` is shared
between galaxies, and the convolution grids are derived from each galaxy's
images.

### Modes

| Mode | Description |
| :--- | :--- |
| `full` (default) | PSF matching and convolution, reprojection, unit conversion and datacube. |
| `conv_only` | PSF cleaning, kernel generation and convolution only. |
| `reprojection_only` | Reprojection and unit conversion, from existing convolved files. |
| `cube_only` | Datacube only, from existing Jy/pixel files. |

Use `--create_kernel` the first time, and whenever the combination of surveys
(and therefore the master) changes. It regenerates PSFs and kernels and forces
reconvolution.

### Outputs

```text
Input/<SURVEY>/PSF_CLEAN/          PSFs resampled onto each convolution grid
Output/PSF_Kernels/                kernel_<filter>_to_<master>.fits
Output/convolved_fits/<galaxy>/    convolved bands and the master
Output/reprojected_files/<galaxy>/ reprojected bands and *_Jy_per_pixel.fits
Output/datacubes/<galaxy>/         <galaxy>_datacube_sci_<ny>x<nx>_Jy_per_pixel.fits
Output/checks/<check>/             standalone check outputs
```

### Standalone checks

Run from the same project directory, after the pipeline:

```bash
astrovello-check-psf                     # PSF matching, every kernel
astrovello-check-flux --galaxy ngc1087   # flux conservation through reprojection
```

---

## Citing

If you use AsTrovello, please cite it through Zenodo:
[doi:10.5281/zenodo.23569095](https://doi.org/10.5281/zenodo.23569095)
(all versions). Each release also has its own DOI, listed on that page; cite
the one for the version you used. GitHub's **Cite this repository** button
gives the reference in APA and BibTeX.

---

## Data sources

- **PHANGS-HST:** https://archive.stsci.edu/hlsp/phangs
- **PHANGS-JWST:** https://archive.stsci.edu/hlsp/phangs
- **S4G (Spitzer):** https://irsa.ipac.caltech.edu/data/SPITZER/S4G/

---

## Project structure

```text
src/astrovello/
├── astrovello_cli.py              pipeline orchestrator (`astrovello`)
├── config.py                      per-survey constants
├── drivers.py                     per-survey behaviour (HDU, masks, units)
├── convolution.py                 PSF widths, PSF cleaning, kernels, convolution
├── reprojection.py                reprojection onto the master grid
├── units.py                       conversion to Jy/pixel
├── cube.py, mask.py               datacube assembly, masks, sky subtraction
├── validation.py                  in-pipeline validation gates
├── preflight.py                   configuration checks before any compute
├── utils.py                       pixel scales, wavelength ordering
├── check2_psf_matching.py         `astrovello-check-psf`
└── check3_flux_conservation.py    `astrovello-check-flux`
```

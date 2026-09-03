"""
Pure per-survey constants for the AsTrovello pipeline.

This module holds ONLY data — measured or calibration values that may
need tweaking without touching any pipeline logic. All *behavior* (how
to parse a filename, how to convert units, how to pick the error-map
convention) lives in :mod:`drivers`, which reads its constants from
here via ``SURVEY_CONFIG``.

To add a new survey: add one entry to `SURVEY_CONFIG` and one
`Driver` subclass in `drivers.py` that points at it. See the
``'PHANGS'`` and ``'S4G'`` entries below for the full set of expected
keys, and the commented-out ``'JPAS'`` block as a fill-in template.

Attributes
----------
PIVOT_WAVELENGTHS : dict of {str : float}
    Pivot wavelength in Angstroms, keyed by lowercase filter name.
    Shared across surveys — used both for HST PHOTFNU recovery (when
    missing from the header) and for sorting the final multi-survey
    hypercube by wavelength (UV -> IR), regardless of which survey a
    filter came from.
SURVEY_CONFIG : dict of {str : dict}
    One entry per survey, keyed by uppercase survey name (e.g.
    ``'PHANGS'``, ``'S4G'``). All pixel scales are native to each
    survey's own detector — the convolution-before-alignment workflow
    means nobody needs to be pre-matched to a common grid here; that
    happens only at the alignment step.

Notesf
-----
Keys expected in each `SURVEY_CONFIG` entry:

is_reference : bool
    Whether this survey provides the astrometric grid that other
    surveys get reprojected onto. Exactly one survey should have this
    set to True.
pixel_scale_arcsec : float or dict of {str : float}
    Native pixel scale in arcsec. A dict when it varies by filter
    (e.g. S4G's IRAC1 vs IRAC2).
instrument_check, instrument_value : str
    Header keyword and expected value used to identify this survey
    from a FITS header (instead of matching on filename substrings).
psf_binned_factor : int
    Oversampling factor of the raw PSF file relative to the native
    detector scale; used to bin the PSF back down before kernel
    generation.
force_tan_sip : bool
    Whether SIP correction must be forced onto this survey's WCS
    headers during reprojection.
sci_glob, err_glob, psf_glob : str
    Glob patterns identifying science, error, and PSF files.
bunit_map : dict of {str : str}
    Maps a header BUNIT value to the name of the conversion routine
    that turns it into Jy/pixel.
error_type : {'weight', 'sigma', 'variance'}
    Convention used by this survey's error maps.
filename_parse : dict
    Generic rule consumed by ``Base_Driver.parse_filename()``:
    ``delimiter``, ``galaxy_index``, ``filter_index``, and
    ``strip_substrings`` (removed from the parsed galaxy name).
psf_filename_map : dict of {str : str}
    Exact filename -> filter lookup for PSF files that don't follow
    the generic science-filename convention.
psf_filename_parse : dict or None
    Fallback generic rule for PSF filenames not covered by
    `psf_filename_map`.
"""

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Pivot wavelengths --------------------------------------------------
PIVOT_WAVELENGTHS = {
    'f275w': 2707.19995,
    'f336w': 3354.84995,
    'f438w': 4325.55005,
    'f555w': 5305.94995,
    'f814w': 8048.1001,
    'irac1': 35075.0,
    'irac2': 44366.0,
    # 'jpas_j0378': ...,
}


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- Survey registry --------------------------------------------------

SURVEY_CONFIG = {
                    "PHANGS": 
                    {
                        "TELESCOP": "HST",
                        "INSTRUME": "WFC3",
                        "pixel_scale_arcsec": 0.0395,
                        "binned_factor": 4,
                        "unit_type": "electrons/s", # usado em units.py
                        "force_tan_sip": False,
                        "sci_glob": "*_exp-drc-sci.fits",
                        "psf_glob": "*PSFSTD*.fits", 
                        "apply_sip_correction": False,
                        "sci_unit": "Jy/px",
                        "err_unit": "UNITLESS"              
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
                        "foscian_sip": True,
                        "sci_glob": "*.phot.*.fits",
                        "psf_glob": "*_col129_row129.fits",
                        "apply_sip_correction": True,
                        "sci_unit": "MJy/sr",
                        "err_unit": "UNITLESS"   
                    }
                }
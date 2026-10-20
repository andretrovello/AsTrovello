"""
AsTrovello - multi-survey datacube construction for spatially-resolved SED
fitting.

The pipeline is run from the command line, from a project directory that
holds an `Input/` folder; `Output/` is created next to it:

    cd /path/to/project      # contains Input/<SURVEY>/...
    astrovello --galaxy ngc1087 --create_kernel

The stages are importable from their modules (`astrovello.convolution`,
`astrovello.reprojection`, `astrovello.units`, `astrovello.cube`, ...).
Nothing is imported here, so `import astrovello` stays cheap.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astrovello")
except PackageNotFoundError:        # running from a source tree, not installed
    __version__ = "unknown"

"""
BaseLoader: the contract every dataset-specific FWI loader must satisfy.

A loader is responsible for everything that differs between datasets:
  - where the raw files live and how they're discovered
  - unit conversions
  - grid quirks (staggered wind grid regridding, broken time coordinates, etc.)
  - dimension naming -> normalised to 'latitude'/'longitude' before returning
  - spin-up policy (which time steps to keep after cffwis_indices is computed)
  - output file naming / splitting

FWICalculator (fwi_core.py) owns everything that IS shared: cluster setup,
align/ffill/clip/rechunk/persist, the cffwis_indices call itself, and the common
parts of writing NetCDF output.
"""

from abc import ABC, abstractmethod


class BaseLoader(ABC):
    #: short identifier used in logging, e.g. 'era5', 'hg3_attribution', 'hg3_historical'
    name: str = "base"

    #: directory sub-indices are written to
    out_dir: str = "."

    #: lat/lon chunk size used throughout loading and computation
    spatial_chunk: int = 30

    #: sub-indices to compute/write: any of dc, dmc, ffmc, isi, bui, fwi, dsr
    output_indices = ["fwi", "dsr"]

    #: kwargs forwarded to xclim.indices.cffwis_indices (tas/pr/sfcWind/hurs/lat excluded)
    cffwis_kwargs = {"initial_start_up": True}

    #: Fixed time units reference for all yearly output files (rather than
    #: xarray's per-file default, which would pick each file's own start
    #: date) so Iris can concatenate cubes loaded from different yearly
    #: files -- otherwise concatenate_cube() sees differing time-coordinate
    #: metadata and errors.
    TIME_UNITS: str = "days since 1900-01-01"

    @abstractmethod
    def load(self, chunks: dict) -> dict:
        """Load and unit-convert the four FWI input variables.

        Must return a dict with keys 'tas', 'pr', 'sfcWind', 'hurs', each an
        xr.DataArray on 'time', 'latitude', 'longitude' dims (renamed from any
        dataset-native dim names), already chunked per `chunks`.
        """
        raise NotImplementedError

    def trim_output(self, index_map: dict) -> dict:
        """Apply the dataset's spin-up/output-window policy to the computed
        sub-indices. Default: no trimming.

        index_map: {index_name: (DataArray, long_name, units)}
        Override in subclasses that need to drop spin-up years or clip to a
        specific output window.
        """
        return index_map

    @abstractmethod
    def write(self, index_map: dict) -> None:
        """Write the (already-trimmed) sub-indices to NetCDF, using whatever
        file naming/splitting convention this dataset requires."""
        raise NotImplementedError

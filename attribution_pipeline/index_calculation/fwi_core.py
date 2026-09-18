"""
FWICalculator: the shared pipeline that every dataset's FWI creation run goes
through. Dataset-specific behaviour is delegated to a BaseLoader implementation.

Usage:
    calc = FWICalculator(loader)
    calc.run()
"""

import logging
import os
import time
import warnings

import numpy as np
import xarray as xr
import xclim as xc
from dask.distributed import Client, LocalCluster, wait

logging.getLogger("distributed").setLevel(logging.WARNING)
# prevents xclim's dask layer from fighting numpy multi-threading.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=UserWarning, message=".*chunking.*")
warnings.filterwarnings("ignore", category=FutureWarning)

INDEX_METADATA = {
    "dc": ("Drought Code", "1"),
    "dmc": ("Duff Moisture Code", "1"),
    "ffmc": ("Fine Fuel Moisture Code", "1"),
    "isi": ("Initial Spread Index", "1"),
    "bui": ("Build-Up Index", "1"),
    "fwi": ("Fire Weather Index", "1"),
}


class FWICalculator:
    def __init__(self, loader):
        self.loader = loader
        self.start_time = time.time()

    def _elapsed(self):
        return np.round(time.time() - self.start_time, 2)

    def run(self):
        loader = self.loader
        print(
            f"[{loader.name}] Starting FWI calculation. out_dir={loader.out_dir}"
        )

        cluster = LocalCluster(
            n_workers=loader.cluster.n_workers,
            threads_per_worker=loader.cluster.threads_per_worker,
            memory_limit=f"{loader.cluster.memory_per_worker_gb}GB",
        )
        client = Client(cluster)
        print(f"[{loader.name}] Dask dashboard: {client.dashboard_link}")

        try:
            chunks = {
                "latitude": loader.spatial_chunk,
                "longitude": loader.spatial_chunk,
            }

            print(f"[{loader.name}] Loading variables...")
            data = loader.load(chunks)
            tas, pr, ws, hurs = (
                data["tas"],
                data["pr"],
                data["sfcWind"],
                data["hurs"],
            )

            # --- Align all variables on their common time axis ---
            tas, pr, ws, hurs = xr.align(tas, pr, ws, hurs, join="inner")
            print(
                f"[{loader.name}] Aligned time dimension: {tas.time.size} steps"
            )
            if tas.time.size == 0:
                raise ValueError("No overlapping dates after alignment.")

            # --- Forward-fill NaNs, clip humidity to a physical range ---
            for varname, da in [
                ("tas", tas),
                ("pr", pr),
                ("ws", ws),
                ("hurs", hurs),
            ]:
                n_nan = da.isnull().sum().values
                if n_nan > 0:
                    print(
                        f"[{loader.name}]  {varname}: {int(n_nan)} NaN values detected, applying forward-fill"
                    )
            tas = tas.ffill(dim="time")
            pr = pr.ffill(dim="time")
            ws = ws.ffill(dim="time")
            hurs = hurs.ffill(dim="time")
            # prevents moisture-code NaNs from super-saturation / rounding above 100%.
            hurs = hurs.clip(min=0, max=100)

            # xclim treats time as a core dim, so it must be a single chunk.
            compute_chunks = {
                "time": -1,
                "latitude": loader.spatial_chunk,
                "longitude": loader.spatial_chunk,
            }
            tas = tas.chunk(compute_chunks)
            pr = pr.chunk(compute_chunks)
            ws = ws.chunk(compute_chunks)
            hurs = hurs.chunk(compute_chunks)
            print(
                f"[{loader.name}] shapes: tas={tas.shape}, pr={pr.shape}, ws={ws.shape}, hurs={hurs.shape}"
            )

            tas, pr, ws, hurs = client.persist([tas, pr, ws, hurs])
            wait([tas, pr, ws, hurs])

            print(
                f"[{loader.name}] Computing FWI (cffwis_kwargs={loader.cffwis_kwargs})..."
            )
            dc, dmc, ffmc, isi, bui, fwi = xc.indices.cffwis_indices(
                tas=tas,
                pr=pr,
                sfcWind=ws,
                hurs=hurs,
                lat=tas.latitude,
                **loader.cffwis_kwargs,
            )

            index_map = {
                "dc": (dc, *INDEX_METADATA["dc"]),
                "dmc": (dmc, *INDEX_METADATA["dmc"]),
                "ffmc": (ffmc, *INDEX_METADATA["ffmc"]),
                "isi": (isi, *INDEX_METADATA["isi"]),
                "bui": (bui, *INDEX_METADATA["bui"]),
                "fwi": (fwi, *INDEX_METADATA["fwi"]),
            }
            if "dsr" in loader.output_indices:
                print(f"[{loader.name}] Computing DSR from FWI...")
                dsr = 0.0272 * fwi**1.77  # Van Wagner (1970, 1987)
                index_map["dsr"] = (dsr, "Daily Severity Rating", "1")

            # keep only the requested sub-indices
            index_map = {
                k: v
                for k, v in index_map.items()
                if k in loader.output_indices
            }

            index_map = loader.trim_output(index_map)

            os.makedirs(loader.out_dir, exist_ok=True)
            loader.write(index_map)

            print(f"[{loader.name}] --- {self._elapsed()} seconds ---")
        finally:
            try:
                client.close(timeout=30)
                cluster.close(timeout=30)
            except Exception as e:
                print(
                    f"[{loader.name}] Warning: cluster shutdown raised {e!r} (ignoring)"
                )
        print(f"[{loader.name}] Finished")

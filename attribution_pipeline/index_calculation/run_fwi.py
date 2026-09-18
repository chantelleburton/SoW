"""
Entrypoint for the unified FWI creation framework.

Selects a loader by dataset name (CLI arg or CYLC_TASK_PARAM_dataset env var)
and runs it through the shared FWICalculator pipeline. Supports running a
single dataset (for the case where only one of the three needs regenerating)
or being called once per dataset from an orchestrating cylc workflow.

Usage:
    python run_fwi.py era5
    python run_fwi.py hg3_attribution
    python run_fwi.py hg3_historical
"""

import os
import sys

from attribution_pipeline.index_calculation.fwi_core import FWICalculator
from attribution_pipeline.index_calculation.loaders.era5 import ERA5Loader
from attribution_pipeline.index_calculation.loaders.hadgem3_attribution import (
    HadGEM3AttributionLoader,
)
from attribution_pipeline.index_calculation.loaders.hadgem3_historical import (
    HadGEM3HistoricalLoader,
)

LOADERS = {
    "era5": ERA5Loader,
    "hg3_attribution": HadGEM3AttributionLoader,
    "hg3_historical": HadGEM3HistoricalLoader,
}


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CYLC_TASK_PARAM_dataset")
    if dataset not in LOADERS:
        raise SystemExit(f"Unknown/missing dataset {dataset!r}. Valid options: {sorted(LOADERS)}")

    loader = LOADERS[dataset]()
    FWICalculator(loader).run()


if __name__ == "__main__":
    main()

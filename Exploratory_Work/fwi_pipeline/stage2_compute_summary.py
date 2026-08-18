"""Stage 2 CLI: interim .nc -> summary CSV. Dataset-agnostic; driven purely by
$CONFIG_PATH and cylc CYLC_TASK_PARAM_* env vars."""
from __future__ import annotations

import os

import iris

from config import load_config, parse_cylc_task_params
from regions import get_region
from summaries.registry import SUMMARY_REGISTRY


def main() -> None:
    cfg = load_config(os.environ["CONFIG_PATH"], os.environ["DATASET"])
    task_params = parse_cylc_task_params()
    print(f"dataset={cfg.dataset}, task_params={task_params}")

    summary_cls = SUMMARY_REGISTRY[cfg.summary.method]
    summary = summary_cls(percentile=cfg.summary.percentile, shp_file=cfg.summary.shapefile)

    start_year = int(cfg.extra["start_year"])
    end_year = int(cfg.extra["end_year"])

    region_names = [task_params["region"]] if "region" in task_params else list(cfg.summary.regions.keys())
    file_params = {k: v for k, v in task_params.items() if k != "region"}
    for region in region_names:
        region_cfg = get_region(region, cfg.summary.regions[region], start_year, end_year)
        interim_path = cfg.interim_filename("fwi", **file_params)
        cube = iris.load_cube(interim_path)
        df = summary.compute(cube, region_cfg)
        out_path = cfg.summary_filename(region, **file_params)
        df.to_csv(out_path, index=False)
        print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()

# New xclim-based FWI Pipeline — Design Plan

## Goal
Replace the ImpactsToolBox dependency entirely with a single, config-driven cylc
pipeline that computes FWI (and other fire-weather metrics in future) via `xclim`
for three data sources, then derives summary metrics (FWI95 etc.) from the
resulting interim `.nc` files. Code must be modular (functions/classes) so new
metrics beyond FWI95 can be added without duplicating pipeline plumbing.

Confirmed working today: `Exploratory_Work/xclim_work/attribution_ensemble/`
(HadGEM3-A 525-member attribution ensemble, 2019-11..2024-12). This becomes the
reference implementation for three new variants:

1. **ERA5 present-day** — 2024-2026 (event period)
2. **ERA5 historical** — 1980-2013 (baseline)
3. **HadGEM3-A 15-member historical** — 1980-2013 (baseline, per member)

Each variant runs the same two-stage process already proven for the attribution
ensemble:
- **Stage 1 — Index calculation**: load raw met variables → xclim → interim `.nc`
  (this already exists for ERA5 in `explore_xclim_FWI.py` and for HadGEM3
  attribution in `explore_hadgem_attribution_xclim_FWI.py`).
- **Stage 2 — Derived metric**: load interim `.nc` → shapefile mask → temporal
  percentile → spatial percentile → CSV per region (already exists as the
  `Historical_FWI/*.py` scripts and the xclim-adapted
  `attribution_ensemble/bias_correction/HadGEM3_xclim_bias_correction_multi_year.py`).

The plan below turns these three proven-but-copy/pasted scripts into a shared
package so the "second bit of processing" (percentile/metric calc) and the
region/shapefile logic is written once, not three (soon more) times.

## Proposed package layout

```
fwi_pipeline/                          # new top-level package (sits next to utils/)
    __init__.py
    config.py                          # dataclasses + YAML loader, one config per run
    sources/
        __init__.py
        base.py                        # DataSource ABC
        era5.py                        # ERA5Source (present-day & historical share this,
                                        #   only date range / directories differ via config)
        hadgem3.py                     # HadGEM3Source (per-member loader, staggered
                                        #   sfcWind regrid, historical file-naming)
    metrics/
        __init__.py
        base.py                        # Metric ABC: compute(variables) -> xr.Dataset
        fwi.py                         # FWIMetric wraps xclim.indices.cffwis_indices
                                        #   (future: DroughtCodeMetric, custom indices, etc.)
    summaries/
        __init__.py
        base.py                        # SummaryMetric ABC: compute(cube, region_cfg) -> value/series
        percentile.py                  # PercentileSummary (generalises the existing
                                        #   "temporal 95th then spatial 95th" pattern)
    regions.py                         # region/shapefile/month definitions (thin wrapper
                                        #   around utils.cubefuncs.apply_shapefile_inclusive)
    stage1_compute_index.py            # CLI entrypoint: source+metric -> interim .nc
    stage2_compute_summary.py          # CLI entrypoint: interim .nc -> summary CSV
    flow.cylc                          # single cylc workflow driving both stages
    configs/
        era5_present.yaml
        era5_historical.yaml
        hadgem3_historical.yaml
```

Reused as-is: `utils/cubefuncs.py`, `utils/constrain_cubes_standard.py` (shapefile
masking, `ConstrainToYear`, `constrain_cube_to_months`, `CountryPercentile`,
`TimePercentile`, `ensemble_member_id`). No need to rewrite these — Stage 2 calls
them directly, exactly as the bias-correction script already does for xclim
output.

## Config-driven design

One YAML file per dataset variant, loaded into a `RunConfig` dataclass:

```yaml
# configs/era5_historical.yaml
dataset: era5_historical
source: era5                 # dispatch key -> sources.era5.ERA5Source
start_year: 1980
end_year: 2013
wind_stat: mean
rh_stat: mean
basepath: /data/users/appldata/Data/OBS-ERA5/daily
interim_dir: /data/scratch/bob.potts/sowf/xclim_pipeline/era5_historical/raw_fwi
metrics: [fwi]
summary:
  method: percentile
  percentile: 95
  regions: [Iberia, Scotland, Chile, Canada, Korea]   # -> regions.py lookup
  output_dir: /data/scratch/bob.potts/sowf/xclim_pipeline/era5_historical/summary
```

```yaml
# configs/hadgem3_historical.yaml
dataset: hadgem3_historical
source: hadgem3
run_type: historical          # not historicalExt/historicalNatExt
members: [1..15]              # 15-member historical ensemble, single realisation
tld: /data/users/opatt/HadGEM3-A-N216
start_year: 1980
end_year: 2013
interim_dir: /data/scratch/bob.potts/sowf/xclim_pipeline/hadgem3_historical/raw_fwi
metrics: [fwi]
summary:
  method: percentile
  percentile: 95
  regions: [Iberia, Scotland, Chile, Canada, Korea]
  output_dir: /data/scratch/bob.potts/sowf/xclim_pipeline/hadgem3_historical/summary
```

`config.py` loads YAML → dataclass, validates required fields per source type,
and exposes `.interim_filename(**task_params)` / `.summary_filename(**task_params)`
helper methods so naming stays centralised (avoids the current copy-pasted
f-string naming scattered across scripts).

## Class design

### `sources/base.py`
```python
class DataSource(ABC):
    def __init__(self, cfg: RunConfig): ...

    @abstractmethod
    def load_variables(self, **task_params) -> dict[str, xr.DataArray]:
        """Return aligned, unit-converted, ffilled tas/pr/sfcWind/hurs, ready for xclim."""

    def dask_cluster_kwargs(self) -> dict:
        """Per-source dask sizing (ERA5 whole-globe vs single HadGEM3 member differ)."""
```

`ERA5Source.load_variables()` = the existing `explore_xclim_FWI.py` loading logic
(temperature/precip/wind/humidity glob+open_mfdataset+align+ffill), parametrised
by `start_year`/`end_year` instead of hardcoded. Present-day and historical ERA5
runs are the *same class*, just different `start_year`/`end_year` in config —
no code duplication needed.

`HadGEM3Source.load_variables(member=...)` = the existing
`explore_hadgem_attribution_xclim_FWI.py` logic (`load_variable`,
`regrid_to_tracer`, window clipping), parametrised by `run_type` (`historical` for
the 15-member historical set vs `historicalExt`/`historicalNatExt` for
attribution) and `member` id format (single-digit `r00{n}i1p1` for the 15-member
set vs `r{NNN}i1p{R}` for the 525-member set — handle via a `member_id()` method
overridden/configured per variant).

### `metrics/base.py`
```python
class Metric(ABC):
    name: str
    @abstractmethod
    def compute(self, variables: dict[str, xr.DataArray]) -> xr.Dataset:
        """Return one or more named DataArrays (e.g. {'fwi': ..., 'dc': ...})."""
```

`FWIMetric.compute()` wraps `xc.indices.cffwis_indices(...)`, returns whichever
sub-indices are requested in config (`OUTPUT_INDICES` today → `cfg.metrics`).
Adding a new metric later (e.g. a heat-based index) means adding one new
`Metric` subclass — Stage 1 script doesn't change.

### `summaries/base.py`
```python
class SummaryMetric(ABC):
    @abstractmethod
    def compute(self, cube, region_cfg) -> pd.DataFrame:
        """e.g. temporal-then-spatial percentile per year, per region."""
```

`PercentileSummary` = the existing pattern used identically in
`Historical_FWI/*.py` and the bias-correction scripts:
`apply_shapefile_inclusive` → `ConstrainToYear` → `constrain_cube_to_months` →
`CountryPercentile` → `TimePercentile`. Generalising this into one class removes
the current triplication across ERA5/HadGEM3/attribution scripts. Future metrics
(e.g. mean instead of 95th, or a fixed-threshold exceedance count) become new
`SummaryMetric` subclasses reusing the same region-masking helpers.

## Stage scripts (CLI entrypoints, called by cylc)

`stage1_compute_index.py`:
```python
cfg = load_config(os.environ["CONFIG_PATH"])
task_params = parse_cylc_task_params()   # member=..., start_year=... depending on dataset
source = SOURCE_REGISTRY[cfg.source](cfg)
variables = source.load_variables(**task_params)
for metric_name in cfg.metrics:
    metric = METRIC_REGISTRY[metric_name]()
    ds = metric.compute(variables)
    out_path = cfg.interim_filename(metric_name, **task_params)
    write_netcdf(ds, out_path)   # shared chunking/encoding helper
```

`stage2_compute_summary.py`:
```python
cfg = load_config(os.environ["CONFIG_PATH"])
task_params = parse_cylc_task_params()   # region=..., member=...
summary = SUMMARY_REGISTRY[cfg.summary.method](cfg.summary)
for region in ([task_params["region"]] if "region" in task_params else cfg.summary.regions):
    cube = iris.load_cube(cfg.interim_filename("fwi", **task_params))
    df = summary.compute(cube, REGIONS[region])
    df.to_csv(cfg.summary_filename(region, **task_params), index=False)
```

Both scripts are dataset-agnostic — the same two files run for all three
variants (and the existing attribution ensemble, if migrated later), driven
purely by `$CONFIG_PATH` and cylc task parameters. This directly answers "split
into functions and classes... FWI95 is not the only metric" — new metrics slot
into `METRIC_REGISTRY`/`SUMMARY_REGISTRY` without touching the stage scripts.

## Single cylc flow

One `flow.cylc` with a dataset task-parameter selecting the config, plus
per-dataset sub-parameters (member/region) via Jinja2, mirroring the existing
`attribution_ensemble/flow.cylc` pattern:

```
[task parameters]
    dataset = era5_present, era5_historical, hadgem3_historical
    region = Iberia, Scotland, Chile, Canada, Korea
    # hadgem3 member list generated via Jinja2 loop (1..15), used only when
    # dataset == hadgem3_historical (cylc "family"/conditional graph or simply
    # run member=1 for era5 datasets and ignore it in stage1 for those sources)

[scheduling]
    [[graph]]
        R1 = """
            stage1_compute_index<dataset, member> => stage2_compute_summary<dataset, member, region>
        """

[runtime]
    [[stage1_compute_index<dataset, member>]]
        script = """
            set -eux
            conda activate sowf
            cd /data/users/bob.potts/StateOfFires_2025-26/code/fwi_pipeline
            export CONFIG_PATH=configs/${CYLC_TASK_PARAM_dataset}.yaml
            python stage1_compute_index.py
        """
        platform = spice
        [[[directives]]]
            --mem = 150G
            --partition = cpu-long
            --time = 720
            --cpus-per-task = 4

    [[stage2_compute_summary<dataset, member, region>]]
        script = """
            set -eux
            conda activate sowf
            cd /data/users/bob.potts/StateOfFires_2025-26/code/fwi_pipeline
            export CONFIG_PATH=configs/${CYLC_TASK_PARAM_dataset}.yaml
            python stage2_compute_summary.py
        """
        platform = spice
        [[[directives]]]
            --mem = 20G
            --time = 120
            --cpus-per-task = 1
```

(Exact parameter cross-product/exclusion between `dataset`/`member` needs a
Jinja2 conditional or splitting into two graphs — flagged as an implementation
detail to resolve once we start writing `flow.cylc`, not a blocker for the plan.)

## Migration steps (proposed order)

1. Create `fwi_pipeline/` skeleton + `config.py` (dataclass + YAML loader).
2. Extract `sources/era5.py` from `explore_xclim_FWI.py` (parametrise start/end
   year, wind/rh stat already config-driven).
3. Extract `sources/hadgem3.py` from `explore_hadgem_attribution_xclim_FWI.py`
   (parametrise `run_type`/member id format for the 15-member historical case).
4. Write `metrics/fwi.py` (thin wrapper, ~unchanged from current inline call).
5. Write `summaries/percentile.py` generalising the repeated
   mask→year→month→percentile→percentile block from `Historical_FWI/*.py` /
   `HadGEM3_xclim_bias_correction_multi_year.py`.
6. Write `stage1_compute_index.py` / `stage2_compute_summary.py` CLI wrappers.
7. Write three configs (era5_present, era5_historical, hadgem3_historical).
8. Write single `flow.cylc`, test each dataset's stage1 task individually,
   then stage2, before wiring the full graph.
9. Once outputs validated against existing `Historical_FWI` CSVs (sanity check:
   xclim FWI95 vs ImpactsToolBox FWI95 for the same years), retire dependency on
   ImpactsToolBox-derived baseline files in the bias-correction step.

## Open questions to confirm before implementation
- 15-member historical HadGEM3-A member/run_type file-naming — confirm directory
  layout under `/data/users/opatt/HadGEM3-A-N216/historical/` matches the
  attribution tree's `{var}/day/{var}_day_HadGEM3-A-N216_{experiment}_{member}_*.nc`
  pattern, or if it differs (single realisation vs 5 physics variants).
- Whether summary step should move from iris to xarray/xclim-native percentile
  (`xclim` has `xclim.core.calendar` percentile helpers) to drop the iris
  dependency long-term, or keep iris for continuity with `utils/cubefuncs.py`.
  Recommend keeping iris for now since region-masking utilities are iris-based
  and already validated.

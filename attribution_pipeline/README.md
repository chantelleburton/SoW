# Attribution pipeline

Four-stage Cylc pipeline that turns raw ERA5 / HadGEM3-A Historical /
HadGEM3-A Attribution climate variables into Fire Weather Index (FWI) and
Daily Severity Rating (DSR) risk-ratio / amplification estimates. See
`State Of Wildfire - Attribution Pipeline.png` for the full data-flow diagram.

## Environment setup

The pipeline runs in the `sowf` conda environment, pinned via
`conda-lock.yaml` at the repo root. Create it with:

```bash
conda-lock install --name sowf ../conda-lock.yaml
conda activate sowf
```

(run from this directory, or drop the `../` if run from the repo root).

## Configuration

Paths and region definitions are read from `config.json` (sibling to
`pipeline_config.py`) at import time, so they can be edited without touching
code.

`config.json` schema:

| Key | Purpose |
| --- | --- |
| `pipeline_root_default` | Default output root (see `ATTRIBUTION_PIPELINE_ROOT` below) |
| `raw_fwi_output.{era5,hg3_historical,hg3_attribution}` | Per-source raw FWI/DSR output location. Relative path -> resolved under the pipeline root; absolute path -> used as-is  |
| `external_data_sources.{era5_obs_basepath,impacttb_historical_fwi_dir,shapefile}` | Read-only external input locations |
| `hadgem3_attribution_window` | Dataset-availability window for the HadGEM3-A Attribution ensemble |
| `regions` | Per-country region definitions (months, baseline years, bias-correction years, etc.) |

Two environment variables let you override without editing `config.json`
either:

| Env var | Overrides | Default |
| --- | --- | --- |
| `ATTRIBUTION_PIPELINE_CONFIG` | Location of the `config.json` file | `attribution_pipeline/config.json` |
| `ATTRIBUTION_PIPELINE_ROOT` | Output root for all pipeline-generated files (raw FWI/DSR, interim metric CSVs, bias-corrected ensembles, risk-ratio exports) | `config.json`'s `pipeline_root_default` |

### Output directory structure

Everything the pipeline generates lives under a single root
(`ATTRIBUTION_PIPELINE_ROOT`, or `pipeline_root_default` in `config.json`):

```
<PIPELINE_ROOT>/
  raw_fwi/
    era5/               # index_calculation/ output (or wherever raw_fwi_output.era5 points)
    hg3_historical/
    hg3_attribution/
  metrics/               # metrics/ interim per-year metric CSVs
  bias_corrected_metrics/  # bias_correction/ output
  uncorrected_metrics/
  exports/
    {historical_source}/  # probability_ratio/ summary CSVs + plots
```

**Example -- raw FWI output moved to another user's directory:** if you've
already run `index_calculation/` and then moved (or someone else owns) the
raw FWI output for one source, point `config.json`'s
`raw_fwi_output.<source>` at the new absolute path (or set
`ATTRIBUTION_PIPELINE_CONFIG` to a copy of `config.json` with that override),
rather than editing `pipeline_config.py`:

```json
"raw_fwi_output": {
    "era5": "/data/users/<other_user>/sowf/raw_fwi/era5",
    "hg3_historical": "raw_fwi/hg3_historical",
    "hg3_attribution": "raw_fwi/hg3_attribution"
}
```

Each stage's Cylc workflow also reads its own `rose-suite.conf` (see
`<stage>/rose-suite.conf`), which sets an `ATTRIBUTION_PIPELINE_CODE_DIR`
environment variable (the repo checkout the task scripts `cd` into) plus the
Cylc template variables for that stage (year ranges, ensemble members,
run-time options). Edit `rose-suite.conf` to change how a workflow runs --
you should not need to edit the `flow.cylc` files themselves.


## Running the pipeline

The 4 stages must be run in this order, since each one reads the previous
stage's output:

1. **`index_calculation/`** -- computes raw FWI/DSR from ERA5, HadGEM3-A
   Historical and HadGEM3-A Attribution input variables.
2. **`metrics/`** -- turns daily FWI/DSR NetCDF into per-year metric CSVs
   (p95, 7-day max, cumulative).
3. **`bias_correction/`** -- bias-corrects the HadGEM3-A Historical/Attribution
   ensembles against ERA5, using the metrics from stage 2.
4. **`probability_ratio/`** -- computes risk-ratio / amplification estimates
   and supplement figures from the bias-corrected ensembles.

For each stage:

```bash
cd <stage>/   # e.g. index_calculation/
cylc install
cylc play <workflow-id>
```

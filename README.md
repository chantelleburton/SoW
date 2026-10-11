# Attribution pipeline

Four-stage Cylc pipeline that turns raw ERA5 / HadGEM3-A Historical /
HadGEM3-A Attribution climate variables into Fire Weather Index (FWI) and
Daily Severity Rating (DSR) risk-ratio / amplification estimates. See
`State Of Wildfire - Attribution Pipeline.png` for the full data-flow diagram.
This is the primary, actively-maintained pipeline for computing State of
Wildfires attribution results.

**The only files you should need to edit to configure or run the pipeline
are `config.json` and each stage's `rose-suite.conf` (edited via the Rose
editor, `rose edit` -- see below). You should not need to edit any `.py` or
`flow.cylc` file for normal use.**

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
run-time options). Edit `rose-suite.conf` -- via `rose edit`, not by hand
(see "Via Cylc" below) -- to change how a workflow runs; you should not
need to edit the `flow.cylc` files themselves.

### Output schema

| Output | Directory | CSV format |
| --- | --- | --- |
| Interim per-year metrics | `metrics/` | One CSV per `{dataset}_{metric}_{country}_{member}[_{run_type}].csv`: two columns, `Year,<metric_stem>` (e.g. `Year,FWI_P95`), one row per year. |
| Bias-corrected ensembles | `bias_corrected_metrics/{historical_source}/` | One CSV per `{country}_{metric}_baseline{N}_{runtype}{percentile}percent_LogTransform_Target_{year}_DataYear_{year}_BaselinePeriod_{start}_{end}.csv`: `Year` column + one column per ensemble member (bias-corrected value). |
| Raw (uncorrected) ensembles | `uncorrected_metrics/` | Same per-member CSV layout as above, but values are not bias-corrected -- used only by supplement-figure panels. |

`exports/{historical_source}/` holds one summary CSV plus a matching `.png`
plot per metric x statistic:

- **Risk ratio** -- `{metric}_Risk_Ratio_{historical_source}_{regions}_Summary.csv`,
  columns: `Country, Metric, ERA5_Threshold, Hist_95th, HistNat_95th,
  N_Hist_Members, N_HistNat_Members, RR_Median, RR_5th, RR_25th, RR_75th,
  RR_95th, Likelihood`.
- **Amplification** -- `{metric}_Intensity_Amplification_{historical_source}_{regions}_Summary.csv`,
  columns: `Country, Metric, N_Members, Amp_Mean, Amp_Median, Amp_5th,
  Amp_25th, Amp_75th, Amp_95th`.

## Architecture

Each stage is a self-contained Python package with one Cylc-facing entrypoint
(`run_*.py`) plus the logic it calls into:

1. **`index_calculation/`**
   - `run_fwi.py` -- entrypoint; picks a loader by dataset name and runs it
     through the shared calculator.
   - `fwi_core.py` -- `FWICalculator`, the FWI/DSR computation shared by all
     three datasets.
   - `loaders/` -- one loader per dataset (`era5.py`, `hadgem3_historical.py`,
     `hadgem3_attribution.py`), each reading that dataset's raw variables
     into the shape `FWICalculator` expects.

2. **`metrics/`**
   - `run_metrics.py` -- entrypoint; resolves the dataset's raw FWI/DSR
     file(s), masks to the region shapefile, runs the requested metric,
     writes the CSV.
   - `base.py` -- `BaseMetric` contract (subclass + implement `compute()` to
     add a new metric).
   - `percentile.py` / `extreme_window.py` / `cumulative.py` -- the `p95`,
     `7x` (7-day max) and `cum` (cumulative) metric implementations.

3. **`bias_correction/`**
   - `run_bias_correction.py` -- entrypoint (bias-corrected path).
   - `run_uncorrected.py` -- entrypoint (raw/uncorrected extraction path,
     used only for supplement-figure panels).
   - `core.py` -- `run_bias_correction()`: orchestrates per-member loading,
     log-linear regression and CSV output.
   - `baseline.py` / `regression.py` -- the log-linear quantile regression
     (fit against ERA5, applied to each HadGEM3-A member).
   - `member_loader.py` -- loads a single ensemble member's metric CSV,
     raising `MissingMemberError` for members absent on disk.
   - `uncorrected.py` -- raw ensemble extraction (no regression applied).

4. **`probability_ratio/`**
   - `run_probability_ratio.py` -- entrypoint; runs `risk_ratio` and/or
     `amplification` across all configured regions for a given metric.
   - `core.py` -- `compute_region_risk_ratio()` / `compute_region_amplification()`:
     per-region orchestration (threshold + ensemble + statistic).
   - `ensemble.py` -- `EnsembleLoader`, including `derive_paired_members()`
     for the paired-only (hist/histnat present in both) mode.
   - `statistics.py` -- `RiskRatioStatistic` (bootstrap CI) and
     `AmplificationStatistic`.
   - `threshold.py` -- `get_era5_threshold()`, the ERA5-derived event
     threshold.
   - `plotting.py` -- density-histogram (risk ratio) and box-whisker
     (amplification) figures.


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

### Via Cylc (normal usage)

Each stage's task families are gated behind boolean flags in `rose-suite.conf`
that default to `false` (opt-in). Edit them with the Rose editor rather than
a text editor -- it reads the sibling `meta/rose-meta.conf` to show a titled,
described, validated form for every setting:

```bash
cd <stage>/   # e.g. index_calculation/
rose edit
```

Set the ones you want `true`, save, and close the editor. For example:

- `index_calculation/rose-suite.conf`: `RUN_ERA5` (and/or
  `RUN_HG3_HISTORICAL`, `RUN_HG3_ATTRIBUTION`) to pick which dataset(s) get
  raw FWI/DSR computed.
- `metrics/rose-suite.conf`: `RUN_ERA5_METRICS` / `RUN_HISTXCLIM_METRICS`
  / `RUN_HISTIMPACTTB_METRICS` to pick which dataset(s)' metrics get computed.
- `bias_correction/rose-suite.conf`: `RUN_BIAS_CORRECTED` and/or
  `RUN_UNCORRECTED_EXTRACTION`.
- `probability_ratio/rose-suite.conf`: `STATISTIC=['risk_ratio']` and/or
  `['amplification']`, and `RUN_PLOT_SUPPLEMENT` if you also want the
  5-panel supplement figure.

Then, still from the stage directory, install and run the workflow:

```bash
cylc install
cylc play <workflow-id>
```

### Standalone (debugging a single run)

Each stage's `run_*.py` can also be called directly for one dataset/region/
member at a time, bypassing Cylc entirely -- useful when iterating on one
loader/metric/region without waiting on a full workflow:

```bash
# index_calculation -- one dataset
python -m attribution_pipeline.index_calculation.run_fwi era5

# metrics -- one dataset/country/member
CYLC_TASK_PARAM_dataset=hg3_historical_xclim \
CYLC_TASK_PARAM_country=Iberia \
CYLC_TASK_PARAM_index=fwi \
CYLC_TASK_PARAM_metric=p95 \
CYLC_TASK_PARAM_member=1 \
python -m attribution_pipeline.metrics.run_metrics

# bias_correction -- one country/member/run-type
CYLC_TASK_PARAM_country=Iberia \
CYLC_TASK_PARAM_member=1 \
CYLC_TASK_PARAM_runtype=hist \
CYLC_TASK_PARAM_index=fwi \
CYLC_TASK_PARAM_metric=p95 \
CYLC_TASK_PARAM_historical_source=xclim \
python -m attribution_pipeline.bias_correction.run_bias_correction

# probability_ratio -- one metric, all configured regions
CYLC_TASK_PARAM_metric=FWI_P95 \
CYLC_TASK_PARAM_statistic=risk_ratio \
CYLC_TASK_PARAM_historical_source=xclim \
python -m attribution_pipeline.probability_ratio.run_probability_ratio
```

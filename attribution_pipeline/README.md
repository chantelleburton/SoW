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

Paths and region definitions are read from `config.json`/
 #TODO Expand on this later with default formatting

| Env var | Overrides | Default |
| --- | --- | --- |
| `ATTRIBUTION_PIPELINE_CONFIG` | Location of the `config.json` file | `attribution_pipeline/config.json` |
| `ATTRIBUTION_PIPELINE_ROOT` | Output root for all pipeline-generated files (raw FWI/DSR, interim metric CSVs, bias-corrected ensembles, risk-ratio exports) | `config.json`'s `pipeline_root_default` |

Each stage's Cylc workflow also reads its own `rose-suite.conf` (see
`<stage>/rose-suite.conf`), which sets an `ATTRIBUTION_PIPELINE_CODE_DIR`
environment variable (the repo checkout the task scripts `cd` into) plus the
Cylc template variables for that stage (year ranges, ensemble members,
run-time options). Edit `rose-suite.conf` to change how a workflow runs --


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

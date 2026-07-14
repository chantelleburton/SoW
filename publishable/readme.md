This readme gives a high level overview of the inputs and outputs required to generate the Probability Ratio (PR) values for 3 wildifre events examined in the State of Wildfires 2025-2026 report.

**/Historical_FWI**

This folder contains 2 primary scripts (.py) and two job scheduler scripts (cylc.flow) to distribute the running of the primaries, this structure is mirror in many other places.

*/ERA5/ERA5_historical_FWI.py*
This takes in daily global gridded FWI data as output by the ImpactsToolBox, performs spatial slicing based on a shapefile and calculates the monthly 95th percentile values across a given spatial extent. Based on the event month/s from 25/2026 these are supselected and exported as the ERA5_baseline for later incorporation in both the bias correction and general context. A baseline period of 1980-2013 is default but configurable. 

*/ERA5/flow.cylc*
Distributes the running of multiple regions to different slurm jobs and allows for the extended distribution of components of the FWI.

*/HadGEM3_Historical_FWI.py*
Takes in daily HadGEM3-A-N216 FWI data across all historical HADGEM3-A 15 ensemble members, performs spatial slicing based on the SoW shapefile, and computes the monthly 95th percentile across the selected region. Each member is processed independently via the cylc wrapper, producing per-member CSV outputs used as the baseline distributions in the bias correction step. Region, month, and percentile are configured automatically from the `CYLC_TASK_PARAM_country` environment variable.

*/flow.cylc*
Distributes HadGEM3 historical FWI processing across all 15 ensemble members and 3 regions (Chile, Canada, Iberia) as parallel Slurm jobs. 

---

**/Bias_Correction**

This folder contains the log-linear bias correction step applied to the HadGEM3 attribution ensemble prior to Risk Ratio calculation.

*/HadGEM3_LogBiasCorrection_MultiYear.py*
Applies a log-linear quantile regression bias correction to each HadGEM3 ensemble member (hist and histnat) relative to the ERA5 baseline distribution. The regression is fitted over the 1980–2013 baseline period and applied forward to a configurable list of target years (`DATA_YEARS`). Outputs bias-corrected per-member monthly 95th percentile CSV files. One country/member/runtype combination is processed per job, dispatched by cylc.

*/flow.cylc*
Distributes bias correction across all 15 members, 2 run types (hist, histnat) and 3 regions, yielding 90 parallel Slurm jobs. Requests 60 GB memory and a 6-hour time limit per task, reflecting the statsmodels regression overhead relative to the Historical_FWI step.

---

**/Plotting**

This folder contains scripts for computing and visualising Probability Ratios (PR) and generating supplementary figures.

*/Explore_Risk_Ratio.py*
Calculates and visualises PRs comparing ALL (anthropogenic) and NAT (natural-only) HadGEM3 scenarios across all three study regions. Produces a single 3×2 multi-panel summary figure with bootstrapped confidence intervals (default 10,000 iterations) and exports numerical PR values. ERA5 observations define the event threshold as the monthly 95th percentile over the baseline period.

*/Supplement2_5Panel.py*
Loads pre-computed bias-corrected CSV outputs and generates a 5-panel supplementary figure combining PDF and timeseries subplots for each region alongside a combined PR summary panel. Generalised across all 15 baseline members. Designed as a plotting-only script with no heavy data processing; typical runtime 1–2 minutes.

*/Supplement2_5Panel_ReducedSet.py*
Variant of `Supplement2_5Panel.py` where subplot (e) uses the multiyear reduced-set bias-corrected ensemble rather than the fixed single-year Condensed_Log_Transforms data. Globs all matching CSVs under `Reduced_Set_Log_Transforms/` across all baselines and target years. A `PAIRED_ONLY` flag (default `True`) restricts both hist and histnat to members present in both run types, consistent with the reduced-set methodology.

---

**/utils**

Shared utility library used across all scripts in this repository. Imported via relative paths (`sys.path.insert`) in each script.

*cubefuncs.py*
Iris cube manipulation helpers covering area-weighted spatial aggregation (`CountryMean`, `CountryMax`, `CountryPercentile`), time-axis reductions (`TimeMean`, `TimeMax`, `TimePercentile`), year subsetting (`ConstrainToYear`), and the core `RiskRatio()` function used throughout the attribution pipeline.

*constrain_cubes_standard.py*
Spatial subsetting utilities built on Iris, Shapely, and Cartopy. 

*branded_colours.py*
Defines the State of Wildfires 2025-26 colour palette including gradient scales (teal, red, orange, hot pink), diverging schemes (Teal–Orange, Green–Pink, Blue–Red), categorical highlight colours, and seasonal colours used in timeseries figures.

*__init__.py*
Package initialisation that re-exports the most commonly used functions from `cubefuncs` and `constrain_cubes_standard` for convenient top-level imports across the codebase.
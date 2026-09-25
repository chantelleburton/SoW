#!/usr/bin/env bash
# One-off cleanup: strip the stale "_modified" suffix from existing
# HadGEM3-A Historical FWI output files, to match the current loader
# (attribution_pipeline/index_calculation/loaders/hadgem3_historical.py)
# and resolver (attribution_pipeline/metrics/run_metrics.py), neither of
# which use that suffix any more.
set -euo pipefail

DIR="/data/scratch/bob.potts/sowf/attribution_pipeline/raw_fwi/hg3_historical"

for f in "$DIR"/*_modified.nc; do
    [ -e "$f" ] || continue  # no matches
    newname="${f/_modified.nc/.nc}"
    echo "Renaming: $f -> $newname"
    mv -n "$f" "$newname"
done

"""
check_july_aug_duplicate.py

For every sfcWind file flagged as having August data but a July filename,
compare its data array against the genuine August file for that member.
Reports whether they are identical (or near-identical within float tolerance).
Also writes a detailed diagnostic log with data samples for manual inspection.
"""

import csv
import netCDF4
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

CSV_PATH   = '/data/scratch/bob.potts/sowf/test_output/Exports/time_metadata_issuesExt.csv'
WIND_DIR   = Path('/data/scratch/chantelle.burton/SoW2526/HadGEM_Data/historicalExt/sfcWind/day')
OUTPUT_CSV = '/data/users/bob.potts/StateOfFires_2025-26/code/Exploratory_Work/july_aug_comparison.csv'
OUTPUT_LOG = '/data/users/bob.potts/StateOfFires_2025-26/code/Exploratory_Work/july_aug_diagnostic.log'

# Spatial patch to sample: centre of grid, 4x4 block
PATCH_SIZE = 4


def load_data(filepath):
    """Return the sfcWind data array (masked filled with nan) and time points."""
    with netCDF4.Dataset(filepath, 'r') as ds:
        var_name = [v for v in ds.variables if v not in ('time', 'lat', 'lon',
                                                           'latitude', 'longitude',
                                                           'bnds', 'time_bnds')][0]
        data = ds.variables[var_name][:]
        time_var = ds.variables['time']
        units    = time_var.units
        calendar = getattr(time_var, 'calendar', 'standard')
        points   = netCDF4.num2date(time_var[:], units=units, calendar=calendar)
    return np.ma.filled(data, np.nan), points


def _fmt_patch(arr2d):
    """Format a 2-D array as aligned rows of fixed-width floats."""
    rows = []
    for row in arr2d:
        rows.append('  ' + '  '.join(f'{v:8.4f}' for v in row))
    return '\n'.join(rows)


def _timestep_block(label, july_data, aug_data, t_idx):
    """Return a diagnostic string for one timestep."""
    jslice = july_data[t_idx]
    aslice = aug_data[t_idx]
    cy = jslice.shape[0] // 2
    cx = jslice.shape[1] // 2
    p  = PATCH_SIZE // 2

    jpatch = jslice[cy - p: cy + p, cx - p: cx + p]
    apatch = aslice[cy - p: cy + p, cx - p: cx + p]
    dpatch = jpatch - apatch

    lines = [
        f'  --- Timestep {t_idx} ({label}) ---',
        f'  July  centre {PATCH_SIZE}x{PATCH_SIZE} patch  (mean={np.nanmean(jslice):.4f}  min={np.nanmin(jslice):.4f}  max={np.nanmax(jslice):.4f}):',
        _fmt_patch(jpatch),
        f'  Aug   centre {PATCH_SIZE}x{PATCH_SIZE} patch  (mean={np.nanmean(aslice):.4f}  min={np.nanmin(aslice):.4f}  max={np.nanmax(aslice):.4f}):',
        _fmt_patch(apatch),
        f'  Diff  (July - Aug)  max|diff|={np.nanmax(np.abs(dpatch)):.6g}:',
        _fmt_patch(dpatch),
    ]
    return '\n'.join(lines)


def compare_pair(args):
    july_fname, aug_fname = args
    july_path = WIND_DIR / july_fname
    aug_path  = WIND_DIR / aug_fname

    result = {
        'july_file': july_fname,
        'aug_file':  aug_fname,
        'aug_exists':       aug_path.exists(),
        'identical':        None,
        'max_abs_diff':     None,
        'n_timesteps_july': None,
        'n_timesteps_aug':  None,
        'note':             '',
        'diag':             '',   # detailed diagnostic text for the log
    }

    if not aug_path.exists():
        result['note'] = 'August file not found'
        result['diag'] = '  August file does not exist.\n'
        return result

    try:
        july_data, july_times = load_data(july_path)
        aug_data,  aug_times  = load_data(aug_path)
    except Exception as e:
        result['note'] = f'Load error: {e}'
        result['diag'] = f'  Load error: {e}\n'
        return result

    nt_j = len(july_times)
    nt_a = len(aug_times)
    result['n_timesteps_july'] = nt_j
    result['n_timesteps_aug']  = nt_a

    # Time-point listings
    j_dates = '  ' + ', '.join(str(t) for t in july_times)
    a_dates = '  ' + ', '.join(str(t) for t in aug_times)

    diag_lines = [
        f'  Shapes      : july={july_data.shape}  aug={aug_data.shape}',
        f'  July times  : {j_dates}',
        f'  Aug  times  : {a_dates}',
    ]

    if july_data.shape != aug_data.shape:
        result['note'] = f'Shape mismatch: {july_data.shape} vs {aug_data.shape}'
        result['identical'] = False
        diag_lines.append('  ** SHAPE MISMATCH — cannot compare values **')
        result['diag'] = '\n'.join(diag_lines) + '\n'
        return result

    # Per-timestep diff summary
    diag_lines.append('')
    diag_lines.append('  Per-timestep  max|diff|:')
    per_ts_max = []
    for t in range(nt_j):
        d = float(np.nanmax(np.abs(july_data[t] - aug_data[t])))
        per_ts_max.append(d)
        diag_lines.append(f'    t={t:3d}  ({july_times[t]})  max|diff|={d:.6g}')

    # Spatial sample blocks for first, middle, last timestep
    diag_lines.append('')
    sample_indices = sorted({0, nt_j // 2, nt_j - 1})
    labels = {0: 'first', nt_j // 2: 'middle', nt_j - 1: 'last'}
    for t_idx in sample_indices:
        diag_lines.append(_timestep_block(labels.get(t_idx, str(t_idx)),
                                          july_data, aug_data, t_idx))
        diag_lines.append('')

    max_diff = float(np.nanmax(np.abs(july_data - aug_data)))
    result['max_abs_diff'] = max_diff
    result['identical']    = (max_diff == 0.0)

    if max_diff == 0.0:
        result['note'] = 'Data arrays are byte-for-byte identical'
    else:
        result['note'] = f'Data differs (max |diff| = {max_diff:.6g})'

    diag_lines.append(f'  OVERALL max|diff| = {max_diff:.6g}  →  {result["note"]}')
    result['diag'] = '\n'.join(diag_lines) + '\n'
    return result


def main():
    # Read CSV – keep only sfcWind 202207 mismatch rows
    pairs = []
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['var'] == 'sfcWind'
                    and row['issue'] == 'Time metadata does not match filename'
                    and '_202207.nc' in row['file']):
                july_fname = row['file']
                aug_fname  = july_fname.replace('_202207.nc', '_202208.nc')
                pairs.append((july_fname, aug_fname))

    print(f"Comparing {len(pairs)} July/August file pairs...")

    results = []
    with Pool(processes=cpu_count()) as pool:
        for i, res in enumerate(pool.imap_unordered(compare_pair, pairs, chunksize=10), 1):
            results.append(res)
            if i % 20 == 0:
                print(f"  {i}/{len(pairs)} done...")

    results.sort(key=lambda x: x['july_file'])

    # --- Summary ---
    n_identical  = sum(1 for r in results if r['identical'] is True)
    n_different  = sum(1 for r in results if r['identical'] is False)
    n_errors     = sum(1 for r in results if r['identical'] is None)

    print(f"\n{'='*65}")
    print(f"RESULTS  ({len(results)} pairs checked)")
    print(f"  Identical     : {n_identical}")
    print(f"  Different     : {n_different}")
    print(f"  Errors/missing: {n_errors}")
    print(f"{'='*65}")

    if n_different:
        print("\nPairs where data differs:")
        for r in results:
            if r['identical'] is False:
                print(f"  {r['july_file']}  →  {r['note']}")

    if n_errors:
        print("\nPairs with errors:")
        for r in results:
            if r['identical'] is None:
                print(f"  {r['july_file']}  →  {r['note']}")

    # --- Write diagnostic log ---
    SEP = '=' * 72
    with open(OUTPUT_LOG, 'w') as log:
        log.write(f'July vs August sfcWind diagnostic log\n')
        log.write(f'{len(results)} pairs checked\n')
        log.write(f'  Identical     : {n_identical}\n')
        log.write(f'  Different     : {n_different}\n')
        log.write(f'  Errors/missing: {n_errors}\n')
        log.write(f'{SEP}\n\n')
        for r in results:
            log.write(f'{SEP}\n')
            log.write(f'JULY : {r["july_file"]}\n')
            log.write(f'AUG  : {r["aug_file"]}\n')
            log.write(f'STATUS: {r["note"]}\n')
            log.write(f'{"-" * 60}\n')
            log.write(r['diag'])
            log.write('\n')
    print(f"\nDiagnostic log written to : {OUTPUT_LOG}")

    # --- Write CSV ---
    fieldnames = ['july_file', 'aug_file', 'aug_exists', 'identical',
                  'max_abs_diff', 'n_timesteps_july', 'n_timesteps_aug', 'note']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k in fieldnames}
                         for r in results)

    print(f"\nFull results written to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()

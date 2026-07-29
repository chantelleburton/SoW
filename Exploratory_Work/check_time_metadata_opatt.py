"""
check_time_metadata_opatt.py

Check sfcWind files in the opatt directory for the same time metadata bug
found in the chantelle.burton copy (files mislabelled with wrong month).
"""

import re
import csv
import netCDF4
from pathlib import Path
from multiprocessing import Pool, cpu_count

WIND_DIR   = Path('/data/users/opatt/HadGEM3-A-N216/historicalExt/sfcWind/day')
OUTPUT_CSV = '/data/scratch/bob.potts/sowf/test_output/Exports/time_metadata_issues_opatt_EXT.csv'


def check_file(filepath):
    fname = Path(filepath).name

    range_match  = re.search(r'_(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$', fname)
    single_match = re.search(r'_(\d{4})(\d{2})\.nc$', fname)

    if range_match:
        exp_start_year,  exp_start_month = int(range_match.group(1)), int(range_match.group(2))
        exp_end_year,    exp_end_month   = int(range_match.group(3)), int(range_match.group(4))
        expected = f'{exp_start_year}-{exp_start_month:02d} → {exp_end_year}-{exp_end_month:02d}'
    elif single_match:
        exp_start_year  = exp_end_year  = int(single_match.group(1))
        exp_start_month = exp_end_month = int(single_match.group(2))
        expected = f'{exp_start_year}-{exp_start_month:02d}'
    else:
        return {'file': fname, 'issue': 'Could not parse date from filename',
                'expected': '', 'actual_start': '', 'actual_end': ''}

    try:
        with netCDF4.Dataset(filepath, 'r') as ds:
            time_var   = ds.variables['time']
            units      = time_var.units
            calendar   = getattr(time_var, 'calendar', 'standard')
            raw_values = time_var[:]
            points     = netCDF4.num2date(raw_values, units=units, calendar=calendar)
            t_start    = points[0]
            t_end      = points[-1]

        start_ok = (t_start.year == exp_start_year and t_start.month == exp_start_month)
        end_ok   = (t_end.year   == exp_end_year   and t_end.month   == exp_end_month)

        if not (start_ok and end_ok):
            return {
                'file':         fname,
                'issue':        'Time metadata does not match filename',
                'expected':     expected,
                'actual_start': str(t_start),
                'actual_end':   str(t_end),
            }

        # 360_day calendar: every month is exactly 30 days
        expected_steps = (
            (exp_end_year - exp_start_year) * 12 * 30
            + (exp_end_month - exp_start_month) * 30
            + 30
        )
        if len(points) != expected_steps:
            return {
                'file':         fname,
                'issue':        f'Wrong step count: got {len(points)}, expected {expected_steps}',
                'expected':     expected,
                'actual_start': str(t_start),
                'actual_end':   str(t_end),
            }

        # Start on day 1, end on day 30
        if t_start.day != 1:
            return {
                'file':         fname,
                'issue':        f'Does not start on day 1 (starts day {t_start.day})',
                'expected':     expected,
                'actual_start': str(t_start),
                'actual_end':   str(t_end),
            }
        if t_end.day != 30:
            return {
                'file':         fname,
                'issue':        f'Does not end on day 30 (ends day {t_end.day})',
                'expected':     expected,
                'actual_start': str(t_start),
                'actual_end':   str(t_end),
            }

        # Check all points are in the expected month(s) / monotonically increasing / no gaps
        for idx in range(len(points)):
            p = points[idx]
            # All points must have correct year/month sequence
            expected_step_month = exp_start_month + idx % 30 // 30  # simplified for multi-month
            # Monotonically increasing (no duplicates, no reversals)
            if idx > 0 and raw_values[idx] <= raw_values[idx - 1]:
                return {
                    'file':         fname,
                    'issue':        f'Non-monotonic time at step {idx}: {points[idx-1]} → {p}',
                    'expected':     expected,
                    'actual_start': str(t_start),
                    'actual_end':   str(t_end),
                }
            # Gap check: consecutive steps should be exactly 24 hours apart
            if idx > 0:
                delta_hours = (raw_values[idx] - raw_values[idx - 1])
                # units are "hours since ...", so delta is already in hours
                if abs(delta_hours - 24.0) > 1e-6:
                    return {
                        'file':         fname,
                        'issue':        f'Unexpected time gap at step {idx}: {delta_hours:.1f} h (expected 24 h)',
                        'expected':     expected,
                        'actual_start': str(t_start),
                        'actual_end':   str(t_end),
                    }

        # Any point outside the expected year/month range
        for idx, p in enumerate(points):
            in_range = (
                (p.year, p.month) >= (exp_start_year, exp_start_month)
                and (p.year, p.month) <= (exp_end_year, exp_end_month)
            )
            if not in_range:
                return {
                    'file':         fname,
                    'issue':        f'Point {idx} outside expected range: {p}',
                    'expected':     expected,
                    'actual_start': str(t_start),
                    'actual_end':   str(t_end),
                }

    except Exception as e:
        return {'file': fname, 'issue': str(e),
                'expected': expected, 'actual_start': '', 'actual_end': ''}

    return None


def main():
    if not WIND_DIR.exists():
        print(f"ERROR: directory not found: {WIND_DIR}")
        return

    files = sorted(WIND_DIR.glob('*.nc'))
    print(f"Found {len(files)} .nc files in {WIND_DIR}")
    print(f"Using {cpu_count()} workers...\n")

    issues = []
    completed = 0

    with Pool(processes=cpu_count()) as pool:
        for result in pool.imap_unordered(check_file, [str(f) for f in files], chunksize=50):
            completed += 1
            if completed % 1000 == 0:
                print(f"  {completed}/{len(files)} checked ({len(issues)} issues so far)...")
            if result is not None:
                issues.append(result)

    print(f"\n{'='*65}")
    print(f"SUMMARY  —  {len(issues)} issue(s) found across {len(files)} files")
    print(f"{'='*65}")

    # Group by issue type for a cleaner summary
    mismatch = [i for i in issues if i['issue'] == 'Time metadata does not match filename']
    corrupt  = [i for i in issues if 'Errno' in i['issue'] or 'NetCDF' in i['issue']]
    other    = [i for i in issues if i not in mismatch and i not in corrupt]

    print(f"  Time mismatch : {len(mismatch)}")
    print(f"  Corrupt/unreadable: {len(corrupt)}")
    print(f"  Other         : {len(other)}")

    if mismatch:
        print("\nTime mismatches (expected → actual):")
        for i in sorted(mismatch, key=lambda x: x['file']):
            print(f"  {i['file']}")
            print(f"    Expected : {i['expected']}")
            print(f"    Actual   : {i['actual_start']} → {i['actual_end']}")

    if corrupt:
        print("\nCorrupt / unreadable files:")
        for i in sorted(corrupt, key=lambda x: x['file']):
            print(f"  {i['file']}  —  {i['issue']}")

    if issues:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'issue', 'expected',
                                                    'actual_start', 'actual_end'])
            writer.writeheader()
            writer.writerows(sorted(issues, key=lambda x: x['file']))
        print(f"\nIssues written to: {OUTPUT_CSV}")
    else:
        print("All files match their filenames.")


if __name__ == '__main__':
    main()

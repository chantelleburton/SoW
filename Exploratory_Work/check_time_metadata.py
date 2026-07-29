import re
import csv
import netCDF4
import cftime
from pathlib import Path
from multiprocessing import Pool, cpu_count

BASE = '/data/scratch/chantelle.burton/SoW2526/HadGEM_Data/historicalNatExt'

VARIABLES = {
    # 'hurs':    f'{BASE}/hurs/day',
    # 'pr':      f'{BASE}/pr/day',
    'sfcWind': f'{BASE}/sfcWind/day',
#     'tasmax':  f'{BASE}/tasmax/day',
}

OUTPUT_CSV = '/data/users/bob.potts/StateOfFires_2025-26/code/Exploratory_Work/time_metadata_issuesNatExt.csv'


def check_file(args):
    """Check a single file's time metadata against its filename. Returns a result dict or None."""
    filepath, var = args
    fname = Path(filepath).name

    # Match either a date range (_YYYYMM-YYYYMM.nc) or a single month (_YYYYMM.nc)
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
        return {'var': var, 'file': fname, 'issue': 'Could not parse date from filename',
                'expected': '', 'actual_start': '', 'actual_end': ''}

    try:
        with netCDF4.Dataset(filepath, 'r') as ds:
            time_var = ds.variables['time']
            units    = time_var.units
            calendar = getattr(time_var, 'calendar', 'standard')
            points  = netCDF4.num2date(time_var[:], units=units, calendar=calendar)
            t_start = points[0]
            t_end   = points[-1]

        start_ok = (t_start.year == exp_start_year and t_start.month == exp_start_month)
        end_ok   = (t_end.year   == exp_end_year   and t_end.month   == exp_end_month)

        if not (start_ok and end_ok):
            return {
                'var':          var,
                'file':         fname,
                'issue':        'Time metadata does not match filename',
                'expected':     expected,
                'actual_start': str(t_start),
                'actual_end':   str(t_end),
            }

    except Exception as e:
        return {'var': var, 'file': fname, 'issue': str(e),
                'expected': expected,
                'actual_start': '', 'actual_end': ''}

    return None  # file is fine


def main():
    # Build full task list
    tasks = []
    for var, dirpath in VARIABLES.items():
        files = sorted(Path(dirpath).glob('*.nc'))
        tasks.extend((str(f), var) for f in files)

    print(f"Total files to check: {len(tasks)}")
    print(f"Using {cpu_count()} workers...\n")

    issues = []
    completed = 0

    with Pool(processes=cpu_count()) as pool:
        for result in pool.imap_unordered(check_file, tasks, chunksize=50):
            completed += 1
            if completed % 5000 == 0:
                print(f"  {completed}/{len(tasks)} checked ({len(issues)} issues so far)...")
            if result is not None:
                issues.append(result)

    # --- Summary to terminal ---
    print(f"\n{'='*65}")
    print(f"SUMMARY  —  {len(issues)} issue(s) found across {len(tasks)} files")
    print(f"{'='*65}")

    if issues:
        for i in issues:
            print(f"  [{i['var']}] {i['file']}")
            print(f"    Issue    : {i['issue']}")
            if i['expected']:
                print(f"    Expected : {i['expected']}")
                print(f"    Actual   : {i['actual_start']} → {i['actual_end']}")
    else:
        print("All files match their filenames.")

    # --- Write CSV ---
    if issues:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['var', 'file', 'issue', 'expected', 'actual_start', 'actual_end'])
            writer.writeheader()
            writer.writerows(issues)
        print(f"\nIssues written to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
